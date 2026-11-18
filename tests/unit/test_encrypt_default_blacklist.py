"""Tests for scripts/encrypt_default_blacklist.py.

The target file is redirected into tmp_path and the app cache is the isolated
per-test instance, so the shipped default blacklist is never touched.
"""

import importlib.util
import io
import os
import subprocess
import sys

import pytest

from teikr.prompts import blacklist_state
from teikr.prompts.blacklist import Blacklist, BlacklistItem, ModelBlacklistItem

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "encrypt_default_blacklist.py")


@pytest.fixture(scope="module")
def tool():
    name = "encrypt_default_blacklist"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def target(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    path = out / "blacklist_default.enc"
    monkeypatch.setattr(Blacklist, "DEFAULT_BLACKLIST_FILE_LOC", str(path))
    return path


def backup_of(path):
    return path.with_name(path.name + ".bak")


def tag_dicts(*strings, **fields):
    return [BlacklistItem(s, **fields).to_dict() for s in strings]


def model_dicts(*strings):
    return [ModelBlacklistItem(s).to_dict() for s in strings]


def write_default(tags=(), models=()):
    """Write the target file from the given entries, as a previous run would."""
    Blacklist.set_blacklist(list(tags))
    Blacklist.set_model_blacklist(list(models))
    Blacklist.encrypt_blacklist()
    Blacklist.TAG_BLACKLIST = []
    Blacklist.MODEL_BLACKLIST = []


def set_working(app_cache, tags=(), models=(), confirmed=True):
    app_cache.set(blacklist_state.BLACKLIST_CACHE_KEY, list(tags))
    app_cache.set(blacklist_state.MODEL_BLACKLIST_CACHE_KEY, list(models))
    app_cache.set(blacklist_state.DEFAULT_BLACKLIST_KEY, confirmed)


@pytest.fixture
def existing(target):
    """A previous default file; returns its bytes."""
    write_default(tag_dicts("old tag", "kept tag"), model_dicts("old model"))
    return target.read_bytes()


class TestWrite:
    def test_normal_run_writes_backs_up_and_verifies(self, tool, target, existing, app_cache):
        tags = tag_dicts("kept tag", "new tag")
        models = model_dicts("new model")
        set_working(app_cache, tags, models)

        assert tool.main([]) == tool.EXIT_OK

        assert backup_of(target).read_bytes() == existing
        assert tool.read_default_file() == {"tag": tags, "model": models}

    def test_no_previous_file_writes_without_backup(self, tool, target, app_cache):
        tags = tag_dicts("only tag")
        set_working(app_cache, tags)

        assert tool.main([]) == tool.EXIT_OK

        assert not backup_of(target).exists()
        assert tool.read_default_file()["tag"] == tags

    def test_mismatch_on_read_back_restores_the_previous_file(
            self, tool, target, existing, app_cache, monkeypatch):
        set_working(app_cache, tag_dicts("new tag"))
        real_read = tool.read_default_file
        calls = []

        def read_with_extra_entry_on_read_back():
            lists = real_read()
            if calls:
                lists["tag"].append(BlacklistItem("smuggled").to_dict())
            calls.append(1)
            return lists

        monkeypatch.setattr(tool, "read_default_file", read_with_extra_entry_on_read_back)

        assert tool.main([]) == tool.EXIT_RESTORED
        assert target.read_bytes() == existing

    def test_changed_field_on_read_back_counts_as_a_mismatch(
            self, tool, target, existing, app_cache, monkeypatch):
        set_working(app_cache, tag_dicts("new tag"))
        real_read = tool.read_default_file
        calls = []

        def read_with_flipped_field():
            lists = real_read()
            if calls:
                lists["tag"][0]["enabled"] = not lists["tag"][0]["enabled"]
            calls.append(1)
            return lists

        monkeypatch.setattr(tool, "read_default_file", read_with_flipped_field)

        assert tool.main([]) == tool.EXIT_RESTORED
        assert target.read_bytes() == existing

    def test_failed_read_back_without_previous_file_removes_the_new_one(
            self, tool, target, app_cache, monkeypatch):
        set_working(app_cache, tag_dicts("new tag"))

        def unreadable():
            raise RuntimeError("cannot decrypt")

        monkeypatch.setattr(tool, "read_default_file", unreadable)

        assert tool.main([]) == tool.EXIT_RESTORED
        assert not target.exists()


class TestRefusals:
    def test_empty_tag_list_writes_nothing(self, tool, target, existing, app_cache):
        set_working(app_cache, [], model_dicts("a model"))

        assert tool.main([]) == tool.EXIT_REFUSED

        assert target.read_bytes() == existing
        assert not backup_of(target).exists()

    def test_unconfirmed_list_writes_nothing_and_names_the_source(
            self, tool, target, existing, app_cache, capsys):
        set_working(app_cache, tag_dicts("new tag"), confirmed=False)

        assert tool.main([]) == tool.EXIT_REFUSED

        assert "Source: the shipped default" in capsys.readouterr().out
        assert target.read_bytes() == existing
        assert not backup_of(target).exists()

    def test_unreadable_existing_file_writes_nothing(self, tool, target, app_cache):
        target.write_bytes(b"not an encrypted blacklist")
        set_working(app_cache, tag_dicts("new tag"))

        assert tool.main([]) == tool.EXIT_UNREADABLE

        assert target.read_bytes() == b"not an encrypted blacklist"
        assert not backup_of(target).exists()

    def test_empty_model_list_is_allowed_but_reported(
            self, tool, target, existing, app_cache, capsys):
        set_working(app_cache, tag_dicts("new tag"), [])

        assert tool.main(["--dry-run"]) == tool.EXIT_OK

        assert "1 model entries in the existing file would be dropped" in capsys.readouterr().out


class TestDryRun:
    @pytest.fixture
    def working(self, app_cache):
        tags = tag_dicts("kept tag") + tag_dicts("old tag", enabled=False) + tag_dicts("added tag")
        set_working(app_cache, tags, model_dicts("old model"))

    def test_writes_nothing(self, tool, target, existing, working):
        assert tool.main(["--dry-run"]) == tool.EXIT_OK

        assert target.read_bytes() == existing
        assert not backup_of(target).exists()

    def test_summary_without_entries(self, tool, existing, working, capsys):
        tool.main(["--dry-run"])
        out = capsys.readouterr().out

        assert "--- tag blacklist: 3 entries to write, file holds 2 ---" in out
        assert "unchanged : 1" in out
        assert "changed   : 1" in out
        assert "added     : 1" in out
        assert "removed   : 0" in out
        assert "+ 'added tag'" not in out

    def test_show_diff_lists_entries_and_changed_fields(self, tool, existing, working, capsys):
        tool.main(["--dry-run", "--show-diff"])
        out = capsys.readouterr().out

        assert "+ 'added tag'" in out
        assert "~ 'old tag'" in out
        assert "enabled: True -> False" in out


class TestDuplicates:
    def test_reported_and_counted_as_written(self, tool, target, existing, app_cache, capsys):
        set_working(app_cache, tag_dicts("dup", "dup", "other"))

        assert tool.main(["--dry-run"]) == tool.EXIT_OK
        out = capsys.readouterr().out
        assert "'dup' x2" in out
        assert "3 entries to write" in out

        assert tool.main([]) == tool.EXIT_OK
        assert len(tool.read_default_file()["tag"]) == 3


def test_output_survives_a_cp1252_console(tool, target, existing, app_cache, monkeypatch):
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    set_working(app_cache, tag_dicts("smile \U0001f642"))

    assert tool.main(["--dry-run", "--show-diff"]) == tool.EXIT_OK

    stream.flush()
    assert b"\\U0001f642" in stream.buffer.getvalue()


@pytest.mark.parametrize("command", [
    [SCRIPT, "--dry-run"],
    ["-m", "scripts.encrypt_default_blacklist", "--dry-run"],
], ids=["direct", "module"])
def test_starts_directly_and_as_module(command):
    # The subprocess inherits this test's isolated config and cache folders, in
    # which nothing marks a working list, so a complete run ends refusing.
    result = subprocess.run(
        [sys.executable, *command], cwd=REPO_ROOT, env=dict(os.environ),
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 1, result.stderr
    assert "Source: the shipped default" in result.stdout


def test_written_file_loads_as_the_apps_default(tool, target, app_cache):
    tags = tag_dicts("tag one", "tag two")
    models = model_dicts("model one")
    set_working(app_cache, tags, models)
    assert tool.main([]) == tool.EXIT_OK

    app_cache.set(blacklist_state.DEFAULT_BLACKLIST_KEY, False)
    Blacklist.TAG_BLACKLIST = []
    Blacklist.MODEL_BLACKLIST = []
    blacklist_state.set_blacklist()

    assert [item.to_dict() for item in Blacklist.get_items()] == tags
    assert [item.to_dict() for item in Blacklist.get_model_items()] == models
