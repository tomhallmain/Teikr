"""Selecting a preset by a server request's edit suffix, and active presets
that share one."""

import pytest

from teikr.globals import PromptMode
from teikr.presets import presets_state
from teikr.presets.preset import Preset
from teikr.presets.presets_state import PresetsState


def preset(name, edit_suffix, active=True):
    # Distinct tags per name, since presets compare by content.
    return Preset(name, PromptMode.SFW, f"tags of {name}", "", edit_suffix=edit_suffix, active=active)


@pytest.fixture
def logged(monkeypatch):
    messages = []

    class Recorder:
        def info(self, message, *args, **kwargs):
            messages.append(message)

        warning = info

    monkeypatch.setattr(presets_state, "logger", Recorder())
    return messages


@pytest.fixture
def choices(monkeypatch):
    """Records the candidates of each random pick and returns the last one."""
    offered = []

    def last(candidates):
        offered.append(list(candidates))
        return candidates[-1]

    monkeypatch.setattr(presets_state.random, "choice", last)
    return offered


def use(*presets):
    PresetsState.recent_presets[:] = list(presets)


class TestLookup:
    def test_an_active_preset_is_found(self, logged):
        cherry = preset("Cherry", "_cherry")
        use(cherry)
        assert PresetsState.get_preset_by_suffix("_cherry") is cherry
        assert logged == []

    def test_an_inactive_preset_is_skipped(self):
        use(preset("Cherry", "_cherry", active=False))
        assert PresetsState.get_preset_by_suffix("_cherry") is None

    def test_the_next_active_match_is_used_past_an_inactive_one(self, logged):
        later = preset("Later", "_cherry")
        use(preset("First", "_cherry", active=False), later)
        assert PresetsState.get_preset_by_suffix("_cherry") is later
        assert logged == []

    def test_a_prefix_still_matches(self):
        cher = preset("Cher", "_cher")
        use(cher)
        assert PresetsState.get_preset_by_suffix("_cherry") is cher

    def test_a_shared_suffix_picks_at_random_among_the_active_ones(self, logged, choices):
        first, second = preset("First", "_x"), preset("Second", "_x")
        use(first, preset("Off", "_x", active=False), preset("Other", "_y"), second)

        assert PresetsState.get_preset_by_suffix("_x") is second
        assert choices == [[first, second]]
        assert len(logged) == 1
        assert '"First"' in logged[0] and '"Second"' in logged[0]

    def test_a_shared_suffix_reaches_every_candidate(self):
        first, second = preset("First", "_x"), preset("Second", "_x")
        use(first, second)
        picked = {id(PresetsState.get_preset_by_suffix("_x")) for _unused in range(200)}
        assert picked == {id(first), id(second)}

    def test_the_group_is_the_suffix_of_the_first_match(self, choices):
        # "_cher" comes first and matches "_cherry", so the "_cherry" pair is
        # not consulted.
        cher = preset("Cher", "_cher")
        use(cher, preset("Cherry 1", "_cherry"), preset("Cherry 2", "_cherry"))
        assert PresetsState.get_preset_by_suffix("_cherry") is cher
        assert choices == []

    def test_prefix_matches_with_different_suffixes_do_not_warn(self, logged):
        use(preset("Cher", "_cher"), preset("Cherry", "_cherry"))
        PresetsState.get_preset_by_suffix("_cherry")
        assert logged == []


class TestConflicts:
    def test_active_presets_sharing_a_suffix(self):
        first, second = preset("First", "_x"), preset("Second", "_x")
        use(first, preset("Other", "_y"), second)
        assert PresetsState.active_suffix_conflicts() == {"_x": [first, second]}

    def test_an_inactive_preset_does_not_conflict(self):
        use(preset("First", "_x"), preset("Second", "_x", active=False))
        assert PresetsState.active_suffix_conflicts() == {}

    def test_presets_without_a_suffix_do_not_conflict(self):
        use(preset("First", ""), preset("Second", ""))
        assert PresetsState.active_suffix_conflicts() == {}
