
from teikr.globals import PromptMode

class Preset:
    def __init__(self, name, prompt_mode, positive_tags, negative_tags, edit_suffix="",
                 active=True) -> None:
        self.name = name
        self.prompt_mode = prompt_mode.name if isinstance(prompt_mode, PromptMode) else prompt_mode
        self.positive_tags = positive_tags
        self.negative_tags = negative_tags
        self.edit_suffix = edit_suffix
        #: Whether a server request's edit suffix can select this preset. Not
        #: part of equality: switching it does not make a different preset.
        self.active = bool(active)

    def is_valid(self):
        return True

    def readable_str(self):
        return f"{self.prompt_mode}: {self.name}"

    def __str__(self):
        return self.readable_str()

    def __eq__(self, other):
        if not isinstance(other, Preset):
            return False
        return self.positive_tags == other.positive_tags \
            and self.negative_tags == other.negative_tags \
            and self.edit_suffix == other.edit_suffix

    def __hash__(self):
        return hash((self.positive_tags, self.negative_tags, self.edit_suffix))

    def to_dict(self):
        return {
            'name': self.name,
            'prompt_mode': self.prompt_mode,
            'positive_tags': self.positive_tags,
            'negative_tags': self.negative_tags,
            'edit_suffix': self.edit_suffix,
            'active': self.active,
            }

    @classmethod
    def from_dict(cls, dict_data: dict) -> 'Preset':
        return cls(**dict_data)

    @staticmethod
    def from_runner_app_config(name, runner_app_config) -> 'Preset':
        return Preset(
            name,
            runner_app_config.prompter_config.prompt_mode,
            runner_app_config.positive_tags,
            runner_app_config.negative_tags,
            edit_suffix=getattr(runner_app_config, 'edit_suffix', ''),
        )
