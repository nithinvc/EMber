"""Checkpoint format and state-dict key primitives."""

from typing import Any

TEACHER_CHECKPOINT_FORMAT = "teacher_checkpoint"
TEACHER_CHECKPOINT_VERSION = 2
WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def strip_wrapper_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip any leading combination of DDP and torch.compile prefixes."""

    def strip(key: str) -> str:
        changed = True
        while changed:
            changed = False
            for prefix in WRAPPER_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        return key

    return {strip(key): value for key, value in state_dict.items()}


def teacher_backbone_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Select the EMA teacher backbone, omitting optional teacher heads."""
    prefix = "model_ema.backbone."
    return {
        key.removeprefix(prefix): value
        for key, value in strip_wrapper_prefixes(state_dict).items()
        if key.startswith(prefix)
    }
