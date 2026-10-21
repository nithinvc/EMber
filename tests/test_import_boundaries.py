"""The installed model must work without the training or evaluation stack."""

import subprocess
import sys


def test_model_imports_without_training_or_evaluation_dependencies():
    script = """
import importlib.abc
import sys

class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = (
            "jax", "cryojax", "cryospax", "wandb", "tyro", "evals_pkg",
            "torchvision", "timm", "EMber.training", "EMber.ibot",
            "EMber.common.config.conf_simulator",
            "EMber.models.layers.dino_head",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError(f"Unexpected dependency imported: {fullname}")

sys.meta_path.insert(0, BlockTraining())
from EMber.models.load_model import load_model_for_evaluation
from EMber.models.build import build_models
from EMber.common.config.conf_model import DINOConfig
assert callable(load_model_for_evaluation)
assert callable(build_models)
assert DINOConfig().in_chans == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
