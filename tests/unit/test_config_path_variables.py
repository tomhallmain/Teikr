"""Placeholders in configured paths: {HOME} and the names in path_variables."""

import json
import os
from pathlib import Path

import pytest

import teikr.config as _cfg_module
from teikr.config import Config


@pytest.fixture
def config_file(tmp_path):
    return tmp_path / "configs" / "config.json"


@pytest.fixture
def drives(tmp_path):
    """Two drives holding the same layout, as a drive and its backup copy."""
    root = tmp_path / "drives"
    for drive in ("a", "b"):
        (root / drive / "ComfyUI" / "models").mkdir(parents=True)
        (root / drive / "img").mkdir()
    return root


@pytest.fixture
def warnings(monkeypatch):
    messages = []

    class Recorder:
        def warning(self, message, *args, **kwargs):
            messages.append(str(message))

        def info(self, *args, **kwargs):
            pass

        error = debug = info

    monkeypatch.setattr(_cfg_module, "logger", Recorder())
    return messages


def load(config_file, **values):
    data = json.loads(config_file.read_text(encoding="utf-8"))
    data.update(values)
    config_file.write_text(json.dumps(data), encoding="utf-8")
    return Config()


def same(a, b):
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))


class TestLoading:
    def test_paths_follow_the_variable(self, config_file, drives):
        paths = {"models_dir": "{DRIVE}/ComfyUI/models", "img_dir": "{DRIVE}/img"}

        on_a = load(config_file, path_variables={"DRIVE": str(drives / "a")}, **paths)
        assert same(on_a.models_dir, drives / "a" / "ComfyUI" / "models")
        assert same(on_a.img_dir, drives / "a" / "img")

        on_b = load(config_file, path_variables={"DRIVE": str(drives / "b")}, **paths)
        assert same(on_b.models_dir, drives / "b" / "ComfyUI" / "models")
        assert same(on_b.img_dir, drives / "b" / "img")

    def test_windows_separators_in_the_path(self, config_file, drives):
        c = load(config_file, path_variables={"DRIVE": str(drives / "a")},
                 models_dir="{DRIVE}\\ComfyUI\\models")
        assert same(c.models_dir, drives / "a" / "ComfyUI" / "models")

    def test_a_value_may_start_with_home(self, config_file, monkeypatch, drives):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(drives) if p == "~" else p)
        c = load(config_file, path_variables={"DRIVE": "{HOME}/b"}, models_dir="{DRIVE}/ComfyUI/models")
        assert same(c.models_dir, drives / "b" / "ComfyUI" / "models")

    def test_home_still_works_without_variables(self, config_file, monkeypatch, drives):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(drives / "a") if p == "~" else p)
        c = load(config_file, models_dir="{HOME}/ComfyUI/models")
        assert same(c.models_dir, drives / "a" / "ComfyUI" / "models")

    def test_an_undefined_placeholder_is_reported_and_not_used(self, config_file, warnings):
        c = load(config_file, path_variables={}, models_dir="{MISSING}/ComfyUI/models")
        assert "{MISSING}" not in str(c.models_dir or "")
        assert any("{MISSING}" in message and "path_variables" in message for message in warnings)

    @pytest.mark.parametrize("name", ["HOME", "1DRIVE", "MY-DRIVE"])
    def test_an_invalid_name_is_ignored(self, config_file, warnings, name):
        c = load(config_file, path_variables={name: "/somewhere", "GOOD": "/elsewhere"})
        assert c.path_variables == {"GOOD": "/elsewhere"}
        assert any(name in message for message in warnings)

    def test_a_blank_value_is_ignored(self, config_file, warnings):
        c = load(config_file, path_variables={"DRIVE": "  "})
        assert c.path_variables == {}
        assert warnings

    def test_path_or_name_values_only_get_placeholders_filled(self, config_file, drives):
        c = load(config_file, path_variables={"DRIVE": str(drives / "a")},
                 clip_model_path="{DRIVE}/clip.onnx",
                 reference_similarity_model="openai/clip-vit-base-patch32")
        assert c.clip_model_path == str(drives / "a") + "/clip.onnx"
        assert c.reference_similarity_model == "openai/clip-vit-base-patch32"

    def test_format_fields_are_not_placeholders(self, config_file):
        c = load(config_file, path_variables={"DRIVE": "/x"})
        assert c.expand_path_variables("a photo of {0}") == "a photo of {0}"


class TestSaving:
    def test_the_dialog_text_keeps_the_placeholder(self, config_file, drives):
        c = load(config_file, path_variables={"DRIVE": str(drives / "a")}, models_dir="{DRIVE}/ComfyUI/models")
        assert c.configured_text("models_dir") == "{DRIVE}/ComfyUI/models"

    def test_switching_the_variable_moves_the_saved_paths(self, config_file, drives):
        c = load(config_file, path_variables={"DRIVE": str(drives / "a")}, models_dir="{DRIVE}/ComfyUI/models")

        errors = c.apply_and_persist({
            "path_variables": {"DRIVE": str(drives / "b")},
            "models_dir": c.configured_text("models_dir"),
        })

        assert errors == []
        assert same(c.models_dir, drives / "b" / "ComfyUI" / "models")
        on_disk = json.loads(Path(c.config_path).read_text(encoding="utf-8"))
        assert on_disk["models_dir"] == "{DRIVE}/ComfyUI/models"
        assert on_disk["path_variables"] == {"DRIVE": str(drives / "b")}

    def test_a_relative_save_path_is_stored_as_written(self, config_file):
        c = load(config_file)
        assert c.apply_and_persist({"sd_webui_save_path": "."}) == []
        assert c.dict["sd_webui_save_path"] == "."
        assert os.path.isabs(c.sd_webui_save_path)

    def test_an_invalid_name_is_a_save_error(self, config_file, drives):
        c = load(config_file, path_variables={"DRIVE": str(drives / "a")})

        errors = c.apply_and_persist({"path_variables": {"HOME": "/x", "1BAD": "/y"}})

        assert len(errors) == 2
        assert c.path_variables == {"DRIVE": str(drives / "a")}
        assert c.dict["path_variables"] == {"DRIVE": str(drives / "a")}
