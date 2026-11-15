"""Removing the unpack folders of earlier builds."""

import os

import pytest

import teikr.runtime.unpack_cleanup as cleanup


@pytest.fixture
def unpacked(tmp_path, monkeypatch):
    """An unpacked/ folder with this build (A) and an older one (B), each with
    both executables' folders, plus an unrelated folder."""
    root = tmp_path / "unpacked"
    for name in (
        "teikr-20260101-000000", "teikr-headless-20260101-000000",
        "teikr-20250101-000000", "teikr-headless-20250101-000000",
        "something-else",
    ):
        (root / name).mkdir(parents=True)
        (root / name / "file.bin").write_bytes(b"x")
    monkeypatch.setattr(cleanup, "_running_executables", lambda: set())
    return root


def remove(root, current="teikr-20260101-000000"):
    return cleanup.remove_stale_unpack_dirs(resource_root=str(root / current), is_build=True)


def test_nothing_happens_in_a_checkout(unpacked):
    assert cleanup.remove_stale_unpack_dirs(resource_root=str(unpacked / "teikr-20260101-000000"),
                                            is_build=False) == []
    assert len(os.listdir(unpacked)) == 5


def test_older_builds_are_removed_and_this_build_kept(unpacked):
    removed = remove(unpacked)
    assert len(removed) == 2
    assert sorted(os.listdir(unpacked)) == [
        "something-else", "teikr-20260101-000000", "teikr-headless-20260101-000000",
    ]


def test_the_headless_build_keeps_the_gui_folder_of_its_own_build(unpacked):
    remove(unpacked, current="teikr-headless-20260101-000000")
    assert (unpacked / "teikr-20260101-000000").exists()


def test_a_folder_holding_a_running_executable_is_kept(unpacked, monkeypatch):
    running = cleanup._normalize(str(unpacked / "teikr-20250101-000000" / "Teikr.exe"))
    monkeypatch.setattr(cleanup, "_running_executables", lambda: {running})
    remove(unpacked)
    assert (unpacked / "teikr-20250101-000000").exists()
    assert not (unpacked / "teikr-headless-20250101-000000").exists()


def test_a_folder_left_mid_removal_is_finished(unpacked):
    (unpacked / "teikr-20240101-000000.removing").mkdir()
    remove(unpacked)
    assert not (unpacked / "teikr-20240101-000000.removing").exists()


def test_outside_an_unpacked_folder_nothing_is_removed(tmp_path):
    other = tmp_path / "elsewhere" / "teikr-20260101-000000"
    other.mkdir(parents=True)
    (tmp_path / "elsewhere" / "teikr-20250101-000000").mkdir()
    assert cleanup.remove_stale_unpack_dirs(resource_root=str(other), is_build=True) == []


def test_build_ids_are_read_from_either_prefix():
    assert cleanup._build_id("teikr-headless-20260101-000000") == "20260101-000000"
    assert cleanup._build_id("teikr-20260101-000000.removing") == "20260101-000000"
    assert cleanup._build_id("something-else") is None


class TestBuildsUnderTheFormerName:
    """Folders left by builds made before the rename to Teikr."""

    @pytest.fixture
    def with_legacy(self, unpacked):
        for name in ("sdrunner-20250101-000000", "sdrunner-headless-20250101-000000"):
            (unpacked / name).mkdir()
            (unpacked / name / "file.bin").write_bytes(b"x")
        return unpacked

    def test_their_build_ids_are_read(self):
        assert cleanup._build_id("sdrunner-headless-20250101-000000") == "20250101-000000"
        assert cleanup._build_id("sdrunner-20250101-000000.removing") == "20250101-000000"

    def test_their_folders_are_removed(self, with_legacy):
        remove(with_legacy)
        assert not (with_legacy / "sdrunner-20250101-000000").exists()
        assert not (with_legacy / "sdrunner-headless-20250101-000000").exists()
        assert (with_legacy / "teikr-20260101-000000").exists()

    def test_one_still_running_is_kept(self, with_legacy, monkeypatch):
        running = cleanup._normalize(str(with_legacy / "sdrunner-20250101-000000" / "SDRunner.exe"))
        monkeypatch.setattr(cleanup, "_running_executables", lambda: {running})
        remove(with_legacy)
        assert (with_legacy / "sdrunner-20250101-000000").exists()

    def test_the_current_prefixes_are_not_among_them(self):
        assert not set(cleanup.UNPACK_DIR_PREFIXES) & set(cleanup.LEGACY_UNPACK_DIR_PREFIXES)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_the_running_folder_is_recognised_under_another_name(unpacked, tmp_path):
    """Stands in for the 8.3 short name a Windows build can see itself under."""
    alias = tmp_path / "alias"
    try:
        os.symlink(unpacked / "teikr-20250101-000000", alias, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted here")
    assert cleanup._is_same_folder(str(alias), str(unpacked / "teikr-20250101-000000"))
