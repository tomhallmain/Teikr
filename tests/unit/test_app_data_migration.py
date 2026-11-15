"""Moving the app data folder from a legacy name to APP_DATA_FOLDER.

``migrate_app_data_folder`` is called with a base folder in ``tmp_path``; the
suite's ``TEIKR_APP_DATA_DIR`` keeps ``app_data_dir`` itself from moving
anything real.
"""

import logging
import os
import sys
import uuid

import pytest

import lib.logging_setup as logging_setup
import lib.utils as utils_module
from lib.logging_setup import APP_DATA_FOLDER, LEGACY_APP_DATA_FOLDERS, migrate_app_data_folder
from lib.utils import Utils

LEGACY = LEGACY_APP_DATA_FOLDERS[0]
#: Before the autouse fixture below replaces it.
_real_running_instance = logging_setup._running_instance


@pytest.fixture(autouse=True)
def notices():
    logging_setup._app_data_notices.clear()
    yield logging_setup._app_data_notices
    logging_setup._app_data_notices.clear()


@pytest.fixture(autouse=True)
def no_running_instance(monkeypatch):
    """The real locks belong to the machine; an instance running there must not
    decide these tests."""
    monkeypatch.setattr(logging_setup, "_running_instance", lambda: None)


def write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def legacy(tmp_path):
    """A legacy app data folder with a config, a cache and an old build's unpack folder."""
    old = tmp_path / LEGACY
    write(old / "configs" / "config.json", '{"user": true}')
    write(old / "app_info_cache.enc", "cache")
    write(old / "unpacked" / "sdrunner-20250101-000000" / "file.bin")
    return tmp_path


def levels(notices):
    return [level for level, _message in notices]


class TestNames:
    def test_the_folder_is_named_after_the_app(self):
        assert APP_DATA_FOLDER == "teikr"
        assert "sd_runner" in LEGACY_APP_DATA_FOLDERS

    def test_the_override_suppresses_the_move(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TEIKR_APP_DATA_DIR", str(tmp_path / "override"))
        monkeypatch.setattr(logging_setup, "migrate_app_data_folder", lambda base: pytest.fail("moved"))
        assert logging_setup.app_data_dir("logs") == tmp_path / "override" / "logs"


class TestNothingToMove:
    def test_a_new_installation_gets_the_new_folder(self, tmp_path, notices):
        assert migrate_app_data_folder(tmp_path) == tmp_path / APP_DATA_FOLDER
        assert notices == []

    def test_an_old_folder_holding_only_unpack_folders_is_left_to_the_cleanup(self, tmp_path, notices):
        write(tmp_path / LEGACY / "unpacked" / "sdrunner-20250101-000000" / "file.bin")
        assert migrate_app_data_folder(tmp_path) == tmp_path / APP_DATA_FOLDER
        assert (tmp_path / LEGACY / "unpacked").is_dir()
        assert notices == []


class TestFromACheckout:
    def test_the_whole_folder_is_renamed(self, legacy, notices):
        new = migrate_app_data_folder(legacy)
        assert new == legacy / APP_DATA_FOLDER
        assert (new / "configs" / "config.json").read_text(encoding="utf-8") == '{"user": true}'
        assert (new / "unpacked" / "sdrunner-20250101-000000" / "file.bin").is_file()
        assert not (legacy / LEGACY).exists()
        assert levels(notices) == [logging.INFO]

    def test_a_failed_rename_keeps_the_old_folder_in_use(self, legacy, notices, monkeypatch):
        def refuse(src, dst):
            raise PermissionError("in use")

        with monkeypatch.context() as m:
            m.setattr(logging_setup.os, "rename", refuse)
            assert migrate_app_data_folder(legacy) == legacy / LEGACY
        assert (legacy / LEGACY / "configs" / "config.json").is_file()
        assert not (legacy / APP_DATA_FOLDER).exists()
        assert levels(notices) == [logging.WARNING]


class TestFromABuild:
    """The build has unpacked into the new folder before any Python ran."""

    @pytest.fixture
    def unpacked_new(self, legacy):
        write(legacy / APP_DATA_FOLDER / "unpacked" / "teikr-20260101-000000" / "file.bin")
        return legacy

    def test_the_entries_move_in_beside_the_unpack_folder(self, unpacked_new, notices):
        new = migrate_app_data_folder(unpacked_new)
        assert new == unpacked_new / APP_DATA_FOLDER
        assert (new / "configs" / "config.json").is_file()
        assert (new / "app_info_cache.enc").is_file()
        assert (new / "unpacked" / "teikr-20260101-000000").is_dir()
        # The old build's unpack folder stays behind, for the build's cleanup.
        assert sorted(os.listdir(unpacked_new / LEGACY)) == ["unpacked"]
        assert levels(notices) == [logging.INFO]

    def test_a_failed_move_is_undone(self, unpacked_new, notices, monkeypatch):
        real_rename = os.rename

        def refuse_the_cache(src, dst):
            if os.path.basename(src) == "configs":
                raise PermissionError("in use")
            real_rename(src, dst)

        # Entries move in sorted order: app_info_cache.enc first, then configs.
        with monkeypatch.context() as m:
            m.setattr(logging_setup.os, "rename", refuse_the_cache)
            assert migrate_app_data_folder(unpacked_new) == unpacked_new / LEGACY
        old = unpacked_new / LEGACY
        assert (old / "app_info_cache.enc").is_file()
        assert (old / "configs" / "config.json").is_file()
        assert sorted(os.listdir(unpacked_new / APP_DATA_FOLDER)) == ["unpacked"]
        assert levels(notices) == [logging.WARNING]

    def test_an_entry_that_cannot_be_moved_back_is_reported(self, unpacked_new, notices, monkeypatch):
        real_rename = os.rename

        def one_way(src, dst):
            if os.path.basename(src) == "configs" or os.path.dirname(src).endswith(APP_DATA_FOLDER):
                raise PermissionError("in use")
            real_rename(src, dst)

        with monkeypatch.context() as m:
            m.setattr(logging_setup.os, "rename", one_way)
            assert migrate_app_data_folder(unpacked_new) == unpacked_new / LEGACY
        assert levels(notices) == [logging.ERROR]
        assert "app_info_cache.enc" in notices[0][1]


class TestBothHoldData:
    def test_the_new_folder_wins_and_nothing_is_merged(self, legacy, notices):
        write(legacy / APP_DATA_FOLDER / "configs" / "config.json", '{"new": true}')
        new = migrate_app_data_folder(legacy)
        assert new == legacy / APP_DATA_FOLDER
        assert (new / "configs" / "config.json").read_text(encoding="utf-8") == '{"new": true}'
        assert not (new / "app_info_cache.enc").exists()
        assert (legacy / LEGACY / "app_info_cache.enc").is_file()
        assert levels(notices) == [logging.WARNING]


class TestNoticesReachTheLog:
    def test_they_are_logged_by_the_next_logger_with_a_file(self, notices, caplog):
        notices.append((logging.INFO, "Moved the app data folder a to b"))
        logger = logging_setup.get_logger("app_data_notice_test")
        assert notices == []
        logger.handlers.clear()


class TestARunningInstance:
    """The move runs before this process checks the single-instance lock."""

    def test_the_folder_is_not_moved_away_from_it(self, legacy, notices, monkeypatch):
        monkeypatch.setattr(logging_setup, "_running_instance", lambda: "SDRunner")
        assert migrate_app_data_folder(legacy) == legacy / LEGACY
        assert (legacy / LEGACY / "configs" / "config.json").is_file()
        assert not (legacy / APP_DATA_FOLDER).exists()
        assert levels(notices) == [logging.INFO]

    def test_every_lock_name_is_checked(self, monkeypatch):
        checked = []
        monkeypatch.setattr(logging_setup, "_single_instance_lock_held",
                            lambda name: checked.append(name) or False)
        assert _real_running_instance() is None
        assert checked == [logging_setup.SINGLE_INSTANCE_LOCK, *logging_setup.LEGACY_SINGLE_INSTANCE_LOCKS]
        assert "SDRunner" in checked


class TestTheLockProbe:
    """Against the lock file Utils takes, in a temp folder of the test's own
    and under a name no installation uses; the mutex is system-wide."""

    @pytest.fixture
    def lock_files(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "win32event", None)
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        monkeypatch.setattr(logging_setup, "_mutex_exists", lambda name: False)
        return tmp_path

    def test_a_held_lock_is_seen(self, lock_files):
        name = f"TeikrTest{uuid.uuid4().hex[:8]}"
        _path, release = Utils._acquire_single_instance(name, name)
        try:
            assert logging_setup._single_instance_lock_held(name)
        finally:
            release()

    def test_a_released_or_missing_lock_is_not(self, lock_files):
        name = f"TeikrTest{uuid.uuid4().hex[:8]}"
        assert not logging_setup._single_instance_lock_held(name)
        (lock_files / f"{name.lower()}.lock").write_text("123", encoding="utf-8")
        assert not logging_setup._single_instance_lock_held(name)

    def test_looking_takes_nothing(self, lock_files, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *_args: "")
        monkeypatch.setattr(utils_module.os, "_exit", lambda code: pytest.fail("the lock was held"))
        name = f"TeikrTest{uuid.uuid4().hex[:8]}"
        (lock_files / f"{name.lower()}.lock").write_text("123", encoding="utf-8")
        logging_setup._single_instance_lock_held(name)
        _path, release = Utils._acquire_single_instance(name, name)
        release()
