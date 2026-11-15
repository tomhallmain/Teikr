"""Moving an installation's encryption keys from a legacy app identifier.

Keys are generated under the legacy identifier the way an earlier version
generated them, against the fake keyring and the per-test cache directory that
``tests/conftest.py`` installs, so nothing here reaches the real key store.
"""

import pytest

import lib.encryptor as encryptor
from teikr.globals import Globals
from teikr.persistence import key_identity

SERVICE = Globals.SERVICE_NAME
LEGACY = Globals.LEGACY_APP_IDENTIFIERS[0]
CURRENT = Globals.APP_IDENTIFIER


def make_legacy_install(tmp_path, password=None):
    """Encrypt a file, and optionally store a password, under the legacy id."""
    path = str(tmp_path / "legacy_cache.enc")
    encryptor.encrypt_data_to_file(b"cached state", SERVICE, LEGACY, path)
    if password is not None:
        assert encryptor.store_encrypted_password(SERVICE, LEGACY, "base", password)
    encryptor.clear_key_store_cache()
    return path


def has_material(identifier):
    return encryptor.has_prior_key_material(SERVICE, identifier)


class TestTheIdentifiers:
    def test_the_current_identifier_is_not_a_legacy_one(self):
        assert CURRENT == "teikr"
        assert CURRENT not in Globals.LEGACY_APP_IDENTIFIERS
        assert "sd_runner" in Globals.LEGACY_APP_IDENTIFIERS


class TestResolution:
    def test_a_new_installation_uses_the_current_identifier_and_creates_nothing(self):
        assert key_identity.key_identifier() == CURRENT
        assert not has_material(CURRENT)
        assert not has_material(LEGACY)

    def test_keys_already_under_the_current_identifier_are_used_as_they_are(self, tmp_path, monkeypatch):
        encryptor.encrypt_data_to_file(b"x", SERVICE, CURRENT, str(tmp_path / "c.enc"))
        make_legacy_install(tmp_path)
        monkeypatch.setattr(key_identity, "_copy_key_material",
                            lambda *a: pytest.fail("nothing should be copied"))
        assert key_identity.key_identifier() == CURRENT

    def test_the_result_is_kept_for_the_process(self, tmp_path, monkeypatch):
        make_legacy_install(tmp_path)
        assert key_identity.key_identifier() == CURRENT
        monkeypatch.setattr(key_identity, "_resolve", lambda: pytest.fail("resolved twice"))
        assert key_identity.key_identifier() == CURRENT


class TestTheCopy:
    def test_files_encrypted_under_the_legacy_identifier_open_under_the_current_one(self, tmp_path):
        path = make_legacy_install(tmp_path)
        assert key_identity.key_identifier() == CURRENT
        assert encryptor.decrypt_data_from_file(path, SERVICE, CURRENT) == b"cached state"

    def test_the_stored_password_comes_along(self, tmp_path):
        make_legacy_install(tmp_path, password="hunter2")
        key_identity.key_identifier()
        assert encryptor.retrieve_encrypted_password(SERVICE, CURRENT, "base") == "hunter2"

    def test_the_legacy_material_is_left_in_place(self, tmp_path):
        path = make_legacy_install(tmp_path)
        legacy_store = encryptor.read_key_store(SERVICE, LEGACY)
        key_identity.key_identifier()
        encryptor.clear_key_store_cache()
        assert encryptor.read_key_store(SERVICE, LEGACY) == legacy_store
        assert encryptor.decrypt_data_from_file(path, SERVICE, LEGACY) == b"cached state"

    def test_the_key_store_is_copied_unchanged(self, tmp_path):
        make_legacy_install(tmp_path)
        legacy_store = encryptor.read_key_store(SERVICE, LEGACY)
        key_identity.key_identifier()
        encryptor.clear_key_store_cache()
        assert encryptor.read_key_store(SERVICE, CURRENT) == legacy_store
        assert encryptor.peek_passphrase(SERVICE, CURRENT) == encryptor.peek_passphrase(SERVICE, LEGACY)

    def test_quantum_keys_are_copied_without_being_used(self, tmp_path, monkeypatch):
        # No decryption is needed for the copy, so a process without OQS can
        # still move keys it cannot use.
        store = {"version": 1, encryptor.ENCRYPTOR_TYPE_KEY: "quantum", "salt": "00",
                 "nonce": "00", "tag": "00", "encrypted_priv": "00", "public_key": "00"}
        encryptor.write_key_store(SERVICE, LEGACY, store)
        encryptor.keyring.set_password(SERVICE, encryptor.namespaced_key(LEGACY, "passphrase"), "p")
        encryptor.clear_key_store_cache()
        assert key_identity.key_identifier() == CURRENT
        encryptor.clear_key_store_cache()
        assert encryptor.read_key_store(SERVICE, CURRENT) == store


class TestAFailedCopy:
    # The patches below are undone through monkeypatch.context(), never
    # monkeypatch.undo(): that would also undo the per-test cache directory
    # the isolation fixture set through the same monkeypatch.

    def test_a_failing_write_keeps_the_legacy_identifier_and_leaves_nothing_behind(self, tmp_path, monkeypatch):
        path = make_legacy_install(tmp_path)

        def fail(*_args):
            raise OSError("disk full")

        with monkeypatch.context() as m:
            m.setattr(encryptor, "write_key_store", fail)
            assert key_identity.key_identifier() == LEGACY
        encryptor.clear_key_store_cache()
        assert not has_material(CURRENT)
        assert encryptor.decrypt_data_from_file(path, SERVICE, LEGACY) == b"cached state"

    def test_a_copy_that_does_not_read_back_is_undone(self, tmp_path, monkeypatch):
        make_legacy_install(tmp_path)
        real_read = encryptor.read_key_store

        def read_wrong(service, identifier):
            store = real_read(service, identifier)
            return dict(store, public_key="changed") if identifier == CURRENT and store else store

        with monkeypatch.context() as m:
            m.setattr(encryptor, "read_key_store", read_wrong)
            assert key_identity.key_identifier() == LEGACY
        encryptor.clear_key_store_cache()
        assert not has_material(CURRENT)

    def test_no_readable_passphrase_keeps_the_legacy_identifier(self, tmp_path, monkeypatch):
        make_legacy_install(tmp_path)
        monkeypatch.setattr(encryptor, "peek_passphrase", lambda *_args: None)
        monkeypatch.setattr(encryptor, "has_prior_key_material",
                            lambda service, identifier: identifier == LEGACY)
        assert key_identity.key_identifier() == LEGACY

    def test_a_retry_after_a_failure_succeeds(self, tmp_path, monkeypatch):
        path = make_legacy_install(tmp_path)
        with monkeypatch.context() as m:
            m.setattr(encryptor, "write_key_store", lambda *_a: (_ for _ in ()).throw(OSError("busy")))
            assert key_identity.key_identifier() == LEGACY
        key_identity.reset()
        assert key_identity.key_identifier() == CURRENT
        assert encryptor.decrypt_data_from_file(path, SERVICE, CURRENT) == b"cached state"


class TestAPassphraseFromTheEnvironment:
    """The variable answers for every identifier, so it is no sign of keys."""

    @staticmethod
    def legacy_install_for_the_environment(tmp_path, monkeypatch):
        # The encryptor refuses to generate keys while the variable is set (it
        # counts the passphrase as prior material), and so it does while any
        # passphrase is stored. So the keys are generated as usual, and the
        # passphrase generated with them moves from the keyring to the variable.
        path = make_legacy_install(tmp_path)
        passphrase_key = encryptor.namespaced_key(LEGACY, "passphrase")
        generated = encryptor.keyring.get_password(SERVICE, passphrase_key)
        assert generated
        encryptor.keyring.delete_password(SERVICE, passphrase_key)
        monkeypatch.setenv(f"{SERVICE.upper()}_PASSPHRASE", generated)
        encryptor.clear_key_store_cache()
        return path

    def test_the_keys_are_still_copied(self, tmp_path, monkeypatch):
        path = self.legacy_install_for_the_environment(tmp_path, monkeypatch)
        assert key_identity.key_identifier() == CURRENT
        assert encryptor.decrypt_data_from_file(path, SERVICE, CURRENT) == b"cached state"

    def test_a_new_installation_is_not_mistaken_for_one_with_keys(self, monkeypatch):
        monkeypatch.setenv(f"{SERVICE.upper()}_PASSPHRASE", "from-env")
        monkeypatch.setattr(key_identity, "_copy_key_material",
                            lambda *a: pytest.fail("nothing to copy"))
        assert key_identity.key_identifier() == CURRENT

    def test_nothing_is_written_to_the_keyring(self, tmp_path, monkeypatch):
        path = self.legacy_install_for_the_environment(tmp_path, monkeypatch)
        assert key_identity.key_identifier() == CURRENT
        assert encryptor.keyring.get_password(SERVICE, encryptor.namespaced_key(CURRENT, "passphrase")) is None
        assert encryptor.decrypt_data_from_file(path, SERVICE, CURRENT) == b"cached state"
