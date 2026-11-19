"""A run whose image does not exist on this machine.

Such a path used to be dropped without a word: the workflow then found no
image, left its LoadImage node blank, and ComfyUI resolved the blank name to
its whole input folder -- failing with ``[Errno 13] Permission denied`` on that
folder, which says nothing about which image was meant.
"""

import pytest

from lib.translations import _
from teikr.globals import WorkflowType
from teikr.models.model_adapters import ControlNet, IPAdapter
from teikr.runs import run as run_module
from teikr.runs.run import MissingImageError, Run
from teikr.runs.run_config import RunConfig


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "images" / "source.png"
    path.parent.mkdir()
    path.write_bytes(b"\x89PNG")
    return str(path)


@pytest.fixture
def missing(tmp_path):
    return str(tmp_path / "images" / "moved_away.png")


def require(workflow, items, is_dir=False):
    Run.require_existing_images(workflow, items, RunConfig.IP_ADAPTER_REQUIRED_WORKFLOWS, is_dir)


class TestRequireExistingImages:
    def test_an_image_edit_whose_file_is_missing_is_refused(self, missing):
        with pytest.raises(MissingImageError) as raised:
            require(WorkflowType.IMAGE_EDIT, [IPAdapter(missing)])
        assert missing in str(raised.value)

    def test_one_existing_file_among_several_is_enough(self, image, missing):
        require(WorkflowType.IMAGE_EDIT, [IPAdapter(missing), IPAdapter(image)])

    def test_nothing_named_is_left_to_run_config_validation(self):
        require(WorkflowType.IMAGE_EDIT, [])
        require(WorkflowType.IMAGE_EDIT, [None])

    def test_a_workflow_that_does_not_need_the_image_is_not_refused(self, missing):
        require(WorkflowType.SIMPLE_IMAGE_GEN, [IPAdapter(missing)])

    def test_directory_mode_is_not_checked(self, missing):
        require(WorkflowType.IMAGE_EDIT, [IPAdapter(missing)], is_dir=True)

    def test_control_nets_use_their_own_list(self, missing):
        with pytest.raises(MissingImageError):
            Run.require_existing_images(WorkflowType.CONTROLNET, [ControlNet(missing)],
                                        RunConfig.CONTROL_NET_REQUIRED_WORKFLOWS, False)


class TestDroppedPathsAreLogged:
    def test_the_dropped_path_is_named(self, image, missing, monkeypatch):
        # Teikr's loggers do not propagate, so caplog would see nothing.
        warnings = []
        monkeypatch.setattr(run_module.logger, "warning", warnings.append)
        groups = Run._group_for_iteration([IPAdapter(missing), IPAdapter(image)], False,
                                          lambda i: i.is_valid())
        assert [a.id for a in groups[0]] == [image]
        assert len(warnings) == 1 and missing in warnings[0]


class TestDescribeImagePath:
    def test_a_long_path_on_windows_names_its_length(self, monkeypatch):
        monkeypatch.setattr(run_module.sys, "platform", "win32")
        path = "C:\\" + "a" * 300 + ".png"
        assert Run.describe_image_path(path) == _(
            "{0} ({1} characters, over the Windows limit of {2})").format(path, len(path), 259)

    def test_a_short_path_or_another_platform_is_unchanged(self, monkeypatch):
        monkeypatch.setattr(run_module.sys, "platform", "win32")
        assert Run.describe_image_path("C:\\a.png") == "C:\\a.png"
        monkeypatch.setattr(run_module.sys, "platform", "linux")
        long_path = "/" + "a" * 300
        assert Run.describe_image_path(long_path) == long_path
