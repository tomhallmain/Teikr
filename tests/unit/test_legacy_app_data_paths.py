"""Configured paths into a legacy app data folder, after the folder has moved.

The Config dialog stores the resolved save paths, so a config.json can hold
absolute paths into ``<app data>/sd_runner``. ``relocate_legacy_app_data_path``
is called with an explicit root in ``tmp_path``; the suite's
``TEIKR_APP_DATA_DIR`` keeps the default root from relocating anything.
"""

import json
import os
from functools import partial

import pytest

import lib.logging_setup as logging_setup
import teikr.config as config_module
from lib.logging_setup import APP_DATA_FOLDER, LEGACY_APP_DATA_FOLDERS, relocate_legacy_app_data_path

LEGACY = LEGACY_APP_DATA_FOLDERS[0]


@pytest.fixture
def base(tmp_path):
    """An app data base with the new folder in use and the legacy one beside it."""
    folder = tmp_path / "base"
    (folder / APP_DATA_FOLDER / "outputs").mkdir(parents=True)
    (folder / LEGACY / "unpacked").mkdir(parents=True)
    return folder


class TestRelocate:
    def test_the_legacy_folder_itself_becomes_the_new_one(self, base):
        assert relocate_legacy_app_data_path(str(base / LEGACY), base / APP_DATA_FOLDER) == str(base / APP_DATA_FOLDER)

    def test_a_folder_inside_it_keeps_its_place(self, base):
        relocated = relocate_legacy_app_data_path(str(base / LEGACY / "outputs"), base / APP_DATA_FOLDER)
        assert relocated == str(base / APP_DATA_FOLDER / "outputs")

    def test_a_place_only_the_legacy_folder_has_is_left_alone(self, base):
        path = str(base / LEGACY / "only_here")
        (base / LEGACY / "only_here").mkdir()
        assert relocate_legacy_app_data_path(path, base / APP_DATA_FOLDER) == path

    def test_a_path_elsewhere_is_left_alone(self, base, tmp_path):
        path = str(tmp_path / "images")
        assert relocate_legacy_app_data_path(path, base / APP_DATA_FOLDER) == path

    def test_a_sibling_sharing_the_prefix_is_left_alone(self, base):
        (base / (LEGACY + "_other")).mkdir()
        path = str(base / (LEGACY + "_other"))
        assert relocate_legacy_app_data_path(path, base / APP_DATA_FOLDER) == path

    def test_nothing_is_relocated_while_the_legacy_folder_is_in_use(self, base):
        path = str(base / LEGACY / "outputs")
        assert relocate_legacy_app_data_path(path, base / LEGACY) == path

    def test_the_override_suppresses_it(self, monkeypatch, base):
        monkeypatch.setenv("TEIKR_APP_DATA_DIR", str(base / "override"))
        monkeypatch.setattr(logging_setup, "_resolved_app_data_root", lambda: pytest.fail("resolved"))
        path = str(base / LEGACY / "outputs")
        assert relocate_legacy_app_data_path(path) == path


class TestConfigOnLoad:
    @pytest.fixture
    def load(self, base, tmp_path, monkeypatch):
        monkeypatch.setattr(config_module, "relocate_legacy_app_data_path",
                            partial(relocate_legacy_app_data_path, root=base / APP_DATA_FOLDER))
        config_path = tmp_path / "configs" / "config.json"

        def load(**values):
            data = json.loads(config_path.read_text(encoding="utf-8"))
            data.update(values)
            config_path.write_text(json.dumps(data), encoding="utf-8")
            return config_module.Config(), config_path

        return load

    def test_a_save_path_into_the_legacy_folder_is_rewritten_and_saved(self, base, load):
        cfg, config_path = load(sd_webui_save_path=str(base / LEGACY))
        new = os.path.normpath(str(base / APP_DATA_FOLDER))
        assert cfg.sd_webui_save_path == new
        assert json.loads(config_path.read_text(encoding="utf-8"))["sd_webui_save_path"] == new

    def test_the_cloud_save_path_is_rewritten(self, base, load):
        cfg, _unused = load(cloud_backends={"save_path": str(base / LEGACY / "outputs")})
        assert cfg.get_cloud_save_path() == os.path.normpath(str(base / APP_DATA_FOLDER / "outputs"))

    def test_a_config_without_such_paths_is_not_written(self, load):
        cfg, config_path = load()
        before = config_path.stat().st_mtime_ns
        config_module.Config()
        assert config_path.stat().st_mtime_ns == before
