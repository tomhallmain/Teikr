"""The log key's identity after the rename, and reading logs of either identity.

The suite pins the log passphrase through the environment; these tests unset
it and work against the fake keyring ``tests/conftest.py`` installs.
"""

import importlib.util
import logging
import os
import sys

import pytest

import lib.encryptor as encryptor
import lib.logging_setup as logging_setup
from lib.encryptor import StreamingLogCipher, namespaced_key

SERVICE = logging_setup.LOG_ENCRYPTION_SERVICE
CURRENT = logging_setup.LOG_ENCRYPTION_APP_ID
LEGACY = logging_setup.LEGACY_LOG_ENCRYPTION_APP_IDS[0]
ENV_VAR = f"{SERVICE.upper()}_PASSPHRASE"


def item(identity):
    return namespaced_key(identity, "passphrase")


def stored(identity):
    return encryptor.keyring.get_password(SERVICE, item(identity))


@pytest.fixture
def notices(monkeypatch):
    collected = []
    monkeypatch.setattr(logging_setup, "_app_data_notices", collected)
    return collected


@pytest.fixture(autouse=True)
def keyring_only(monkeypatch, notices):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(encryptor, "_passphrase_cache", {})
    monkeypatch.setattr(logging_setup, "_log_key_id", None)
    for identity in (CURRENT, LEGACY):
        try:
            encryptor.keyring.delete_password(SERVICE, item(identity))
        except Exception:
            pass


class TestTheIdentities:
    def test_the_current_identity_is_not_a_legacy_one(self):
        assert CURRENT == "teikr"
        assert "sd_runner" in logging_setup.LEGACY_LOG_ENCRYPTION_APP_IDS
        assert CURRENT not in logging_setup.LEGACY_LOG_ENCRYPTION_APP_IDS


class TestResolution:
    def test_a_new_installation_uses_the_current_identity_and_writes_nothing(self, notices):
        assert logging_setup.log_key_identifier() == CURRENT
        assert stored(CURRENT) is None
        assert notices == []

    def test_a_legacy_passphrase_is_copied(self, notices):
        encryptor.keyring.set_password(SERVICE, item(LEGACY), "old-passphrase")
        assert logging_setup.log_key_identifier() == CURRENT
        assert stored(CURRENT) == "old-passphrase"
        assert stored(LEGACY) == "old-passphrase"
        assert [level for level, _message in notices] == [logging.INFO]

    def test_logs_written_before_the_copy_read_under_the_current_identity(self):
        encryptor.keyring.set_password(SERVICE, item(LEGACY), "old-passphrase")
        old_key = encryptor.get_log_cipher_key(SERVICE, LEGACY)
        record = StreamingLogCipher.encrypt_record(old_key, b"before the rename")
        logging_setup.log_key_identifier()
        new_key = encryptor.get_log_cipher_key(SERVICE, CURRENT)
        assert StreamingLogCipher.decrypt_record(new_key, record[4:]) == b"before the rename"

    def test_a_current_passphrase_is_kept(self, notices):
        encryptor.keyring.set_password(SERVICE, item(CURRENT), "new-passphrase")
        encryptor.keyring.set_password(SERVICE, item(LEGACY), "old-passphrase")
        assert logging_setup.log_key_identifier() == CURRENT
        assert stored(CURRENT) == "new-passphrase"
        assert notices == []

    def test_the_result_is_kept_for_the_process(self, monkeypatch):
        assert logging_setup.log_key_identifier() == CURRENT
        monkeypatch.setattr(logging_setup, "_resolve_log_key_identifier",
                            lambda: pytest.fail("resolved twice"))
        assert logging_setup.log_key_identifier() == CURRENT

    def test_a_passphrase_from_the_environment_needs_no_copy(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "from-env")
        encryptor.keyring.set_password(SERVICE, item(LEGACY), "old-passphrase")
        assert logging_setup.log_key_identifier() == CURRENT
        assert stored(CURRENT) is None


class TestAFailedCopy:
    def test_a_failing_write_keeps_the_legacy_identity(self, notices, monkeypatch):
        encryptor.keyring.set_password(SERVICE, item(LEGACY), "old-passphrase")

        def refuse(*_args):
            raise OSError("locked")

        with monkeypatch.context() as m:
            m.setattr(encryptor.keyring, "set_password", refuse)
            assert logging_setup.log_key_identifier() == LEGACY
        assert stored(CURRENT) is None
        assert [level for level, _message in notices] == [logging.WARNING]

    def test_a_copy_that_does_not_read_back_is_removed(self, notices, monkeypatch):
        encryptor.keyring.set_password(SERVICE, item(LEGACY), "old-passphrase")
        real_get = encryptor.keyring.get_password

        def read_wrong(service, key):
            value = real_get(service, key)
            return "changed" if key == item(CURRENT) and value else value

        with monkeypatch.context() as m:
            m.setattr(encryptor.keyring, "get_password", read_wrong)
            assert logging_setup.log_key_identifier() == LEGACY
        assert stored(CURRENT) is None


def load_decrypt_logs():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "decrypt_logs.py")
    spec = importlib.util.spec_from_file_location("decrypt_logs_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class TestTheReader:
    def write(self, path, *records):
        with open(path, "wb") as f:
            for key, text in records:
                f.write(StreamingLogCipher.encrypt_record(key, text.encode("utf-8")))

    def test_each_record_is_tried_with_every_key(self, tmp_path, capsys):
        decrypt_logs = load_decrypt_logs()
        old_key, new_key = os.urandom(32), os.urandom(32)
        path = tmp_path / "mixed.log.enc"
        self.write(path, (old_key, "old line"), (new_key, "new line"))
        lines = list(decrypt_logs.decrypt(path, False, [new_key, old_key]))
        assert lines == ["old line", "new line"]
        assert capsys.readouterr().err == ""

    def test_it_stops_at_a_record_no_key_reads(self, tmp_path, capsys):
        decrypt_logs = load_decrypt_logs()
        key, other = os.urandom(32), os.urandom(32)
        path = tmp_path / "foreign.log.enc"
        self.write(path, (key, "readable"), (other, "foreign"), (key, "after"))
        assert list(decrypt_logs.decrypt(path, False, [key])) == ["readable"]
        assert "record 2" in capsys.readouterr().err

    def test_the_keys_include_a_legacy_identity_with_its_own_passphrase(self):
        decrypt_logs = load_decrypt_logs()
        encryptor.keyring.set_password(SERVICE, item(CURRENT), "new-passphrase")
        encryptor.keyring.set_password(SERVICE, item(LEGACY), "old-passphrase")
        keys = decrypt_logs.log_keys()
        assert keys == [encryptor.get_log_cipher_key(SERVICE, CURRENT),
                        encryptor.get_log_cipher_key(SERVICE, LEGACY)]
        assert len(set(keys)) == 2

    def test_no_passphrase_is_generated_for_a_legacy_identity(self):
        decrypt_logs = load_decrypt_logs()
        decrypt_logs.log_keys()
        assert stored(LEGACY) is None
