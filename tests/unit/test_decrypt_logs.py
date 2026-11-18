"""Tests for scripts/decrypt_logs.py."""

import importlib.util
import os
import sys

from lib.encryptor import StreamingLogCipher

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "decrypt_logs.py")


def load_decrypt_logs():
    spec = importlib.util.spec_from_file_location("decrypt_logs_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def write(path, key, *lines):
    with open(path, "wb") as f:
        for line in lines:
            f.write(StreamingLogCipher.encrypt_record(key, line.encode("utf-8")))


def test_every_record_is_read(tmp_path, capsys):
    decrypt_logs = load_decrypt_logs()
    key = os.urandom(32)
    path = tmp_path / "log.log.enc"
    write(path, key, "first line", "second line")
    assert list(decrypt_logs.decrypt(path, False, key)) == ["first line", "second line"]
    assert capsys.readouterr().err == ""


def test_it_stops_at_a_record_the_key_does_not_read(tmp_path, capsys):
    decrypt_logs = load_decrypt_logs()
    key, other = os.urandom(32), os.urandom(32)
    path = tmp_path / "foreign.log.enc"
    with open(path, "wb") as f:
        for record_key, line in ((key, "readable"), (other, "foreign"), (key, "after")):
            f.write(StreamingLogCipher.encrypt_record(record_key, line.encode("utf-8")))
    assert list(decrypt_logs.decrypt(path, False, key)) == ["readable"]
    assert "record 2" in capsys.readouterr().err
