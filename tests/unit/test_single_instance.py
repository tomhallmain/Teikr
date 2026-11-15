"""The single-instance lock, held under the current name and the legacy ones.

The Windows mutex is system-wide, so these tests never create one: they block
``win32event`` and use the lock file in a temp folder of their own, under
names no real installation uses. A held lock makes the code wait for Enter and
exit the process, so ``input`` and ``os._exit`` are replaced.
"""

import sys
import uuid

import pytest

import lib.utils as utils_module
from lib.utils import Utils
from teikr.globals import Globals


class Exited(Exception):
    pass


@pytest.fixture
def file_locks(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "win32event", None)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda *_args: "")

    def exit_instead(code):
        raise Exited(code)

    monkeypatch.setattr(utils_module.os, "_exit", exit_instead)
    return tmp_path


def name():
    return f"TeikrTest{uuid.uuid4().hex[:8]}"


class TestTheNames:
    def test_the_lock_is_named_after_the_app_and_keeps_the_old_name(self):
        assert Globals.SINGLE_INSTANCE_LOCK == "Teikr"
        assert "SDRunner" in Globals.LEGACY_SINGLE_INSTANCE_LOCKS


class TestHoldingEveryName:
    def test_each_name_is_taken_and_released_in_reverse(self, monkeypatch):
        calls = []

        def acquire(lock_name, app_name, mutex_name=None, lock_filename=None):
            calls.append(("take", lock_name, app_name))
            return f"/tmp/{lock_name}.lock", lambda: calls.append(("release", lock_name))

        monkeypatch.setattr(Utils, "_acquire_single_instance", staticmethod(acquire))
        lock_path, cleanup = Utils.check_single_instance("New", legacy_app_names=("Old", "Older"))
        assert lock_path == "/tmp/New.lock"
        assert calls == [("take", "New", "New"), ("take", "Old", "New"), ("take", "Older", "New")]
        cleanup()
        assert calls[3:] == [("release", "Older"), ("release", "Old"), ("release", "New")]


class TestWithLockFiles:
    def test_a_second_instance_stops(self, file_locks):
        current = name()
        _path, cleanup = Utils.check_single_instance(current)
        try:
            with pytest.raises(Exited):
                Utils.check_single_instance(current)
        finally:
            cleanup()

    def test_an_earlier_version_already_running_stops_this_one(self, file_locks):
        current, legacy = name(), name()
        _path, old_version = Utils.check_single_instance(legacy)
        try:
            with pytest.raises(Exited):
                Utils.check_single_instance(current, legacy_app_names=(legacy,))
        finally:
            old_version()

    def test_an_earlier_version_started_later_stops(self, file_locks):
        current, legacy = name(), name()
        _path, cleanup = Utils.check_single_instance(current, legacy_app_names=(legacy,))
        try:
            # An earlier version checks only its own name.
            with pytest.raises(Exited):
                Utils.check_single_instance(legacy)
        finally:
            cleanup()

    def test_releasing_frees_every_name(self, file_locks):
        current, legacy = name(), name()
        _path, cleanup = Utils.check_single_instance(current, legacy_app_names=(legacy,))
        cleanup()
        _path, again = Utils.check_single_instance(current, legacy_app_names=(legacy,))
        again()
