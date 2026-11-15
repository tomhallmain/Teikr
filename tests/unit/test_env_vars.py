"""The app's environment variables are named TEIKR_*; the former names are only reported."""

from lib.encryptor import KEY_BACKUP_DIR_ENV_VAR
from teikr import app_paths


class TestTheNames:
    def test_the_key_backup_variable(self):
        assert KEY_BACKUP_DIR_ENV_VAR == "TEIKR_KEY_BACKUP_DIR"

    def test_the_cache_dir_is_read_under_the_current_name(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TEIKR_CACHE_DIR", str(tmp_path / "override"))
        assert app_paths.cache_dir() == str(tmp_path / "override")


class TestAFormerName:
    def test_it_is_not_read(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TEIKR_CACHE_DIR", raising=False)
        monkeypatch.setenv("SD_RUNNER_CACHE_DIR", str(tmp_path / "former"))
        assert app_paths.legacy_migration_allowed()

    def test_it_is_reported_with_the_name_to_use(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(app_paths.logger, "warning", warnings.append)
        for name in app_paths._RENAMED_VARIABLES:
            monkeypatch.delenv("SD_RUNNER_" + name, raising=False)
        monkeypatch.setenv("SD_RUNNER_KEY_BACKUP_DIR", "/somewhere")
        app_paths._warn_about_former_variable_names()
        assert len(warnings) == 1
        assert "SD_RUNNER_KEY_BACKUP_DIR" in warnings[0] and "TEIKR_KEY_BACKUP_DIR" in warnings[0]

    def test_nothing_is_reported_without_one(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(app_paths.logger, "warning", warnings.append)
        for name in app_paths._RENAMED_VARIABLES:
            monkeypatch.delenv("SD_RUNNER_" + name, raising=False)
        app_paths._warn_about_former_variable_names()
        assert warnings == []
