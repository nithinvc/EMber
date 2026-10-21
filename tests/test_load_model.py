import json
from dataclasses import asdict

import huggingface_hub.hub_mixin
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from EMber import EMberModel, normalize_images
from EMber.common.checkpoint_keys import (
    TEACHER_CHECKPOINT_FORMAT,
    teacher_backbone_state_dict,
)
from EMber.common.config.conf_model import DINOConfig
from EMber.models import load_model
from EMber.models.build import build_models
from EMber.models.load_model import TeacherEvaluationModel
from EMber.models.vision_transformer import VisionTransformer


class FakeBackbone(nn.Module):
    def forward_features(self, images):
        return {"raw": images}

    def forward(self, images, is_training=False):
        assert is_training
        batch_size = len(images)
        return {
            "x_norm_clstoken": torch.ones(batch_size, 4),
            "x_norm_patchtokens": torch.full((batch_size, 3, 4), 2.0),
        }


def test_teacher_evaluation_model_adapts_backbone_features():
    model = TeacherEvaluationModel(FakeBackbone())

    cls_token, image_tokens = model.forward_features({"image": torch.zeros(2, 16, 16)})
    pooled = model.pool_features(cls_token, image_tokens)

    assert cls_token.shape == (2, 4)
    assert image_tokens.shape == (2, 3, 4)
    assert pooled.shape == (2, 8)
    torch.testing.assert_close(pooled[:, :4], torch.ones(2, 4))
    torch.testing.assert_close(pooled[:, 4:], torch.full((2, 4), 2.0))


def test_teacher_evaluation_model_preserves_tensor_backbone_interface():
    model = TeacherEvaluationModel(FakeBackbone())
    images = torch.zeros(2, 1, 16, 16)

    output = model.forward_features(images)

    assert isinstance(output, dict)
    assert output["raw"] is images


@pytest.mark.parametrize("version", [1, 2])
def test_exported_teacher_loads_without_run_manifest(tmp_path, monkeypatch, version):
    config = DINOConfig(img_size=16, patch_size=8, in_chans=1)

    def build(config, teacher_only):
        assert teacher_only
        return VisionTransformer(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=24,
            depth=1,
            num_heads=3,
            pos_embed_rope_dtype="fp32",
        )

    monkeypatch.setattr(load_model, "build_models", build)
    teacher = build(config, True)
    teacher.init_weights()
    teacher.eval()
    state = {
        f"module._orig_mod.model_ema.backbone.{key}": value
        for key, value in teacher.state_dict().items()
    }
    state["model_ema.dino_head.weight"] = torch.ones(2, 3)
    path = tmp_path / "teacher.pth"
    torch.save(
        {
            "format": TEACHER_CHECKPOINT_FORMAT,
            "format_version": version,
            "step": 42,
            "model_config": asdict(config),
            "model_state_dict": state,
        },
        path,
    )
    restored = load_model.load_model_for_evaluation(path)
    assert not restored.training
    assert not restored.backbone.training
    images = torch.randn(2, 1, 16, 16)
    expected = teacher.forward_features(images)
    actual = restored.forward_features(images)
    for key in ("x_norm_clstoken", "x_norm_patchtokens"):
        torch.testing.assert_close(actual[key], expected[key])


@pytest.mark.parametrize(
    "checkpoint", [{"model_state_dict": {}}, {"format": TEACHER_CHECKPOINT_FORMAT}]
)
def test_released_loader_rejects_non_self_contained_checkpoints(tmp_path, checkpoint):
    path = tmp_path / "teacher.pth"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        load_model.load_model_for_evaluation(path, load_weights=False)


def test_teacher_checkpoint_key_selection_omits_heads_and_student():
    value = torch.ones(1)
    state = {
        "_orig_mod.module.model_ema.backbone.weight": value,
        "model_ema.ibot_head.weight": torch.zeros(1),
        "model.backbone.weight": torch.zeros(1),
    }
    assert teacher_backbone_state_dict(state) == {"weight": value}


@pytest.fixture(scope="module")
def exported_vit_small(tmp_path_factory):
    config = DINOConfig(backbone="vit_small", img_size=16, patch_size=8)
    teacher = build_models(config, teacher_only=True)
    teacher.init_weights()
    teacher.eval()
    # Real exports also contain training and nested simulator settings.
    model_config = asdict(config) | {
        "dino_head_n_prototypes": 65536,
        "simulator_config": {"image_shape": [16, 16]},
    }
    path = tmp_path_factory.mktemp("checkpoints") / "teacher.pth"
    torch.save(
        {
            "format": TEACHER_CHECKPOINT_FORMAT,
            "format_version": 2,
            "model_config": model_config,
            "model_state_dict": {
                f"model_ema.backbone.{key}": value
                for key, value in teacher.state_dict().items()
            },
        },
        path,
    )
    return path, teacher


def test_real_vit_small_round_trip_and_feature_interfaces(exported_vit_small):
    path, teacher = exported_vit_small
    restored = load_model.load_model_for_evaluation(str(path))
    images = torch.randn(2, 1, 16, 24)
    with torch.inference_mode():
        expected = teacher.forward_features(images)
        actual = restored.forward_features(images)
        cls_token, patch_tokens = restored.forward_features({"image": images})
        cls_3d, patches_3d = restored.forward_features({"image": images[:, 0]})
        pooled = restored.pool_features(cls_token, patch_tokens)
    for key in ("x_norm_clstoken", "x_norm_patchtokens", "x_storage_tokens"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(cls_token, expected["x_norm_clstoken"])
    torch.testing.assert_close(patch_tokens, expected["x_norm_patchtokens"])
    torch.testing.assert_close(cls_3d, cls_token)
    torch.testing.assert_close(patches_3d, patch_tokens)
    torch.testing.assert_close(pooled[:, 384:], patch_tokens.mean(dim=1))
    assert cls_token.shape == (2, 384)
    assert patch_tokens.shape == (2, 6, 384)
    assert pooled.shape == (2, 768)
    assert torch.isfinite(pooled).all()


def test_random_backbone_is_initialized(exported_vit_small):
    path, teacher = exported_vit_small
    model = load_model.load_model_for_evaluation(path, load_weights=False)
    assert not model.training
    assert torch.count_nonzero(model.backbone.mask_token) == 0
    assert not torch.equal(model.backbone.cls_token, teacher.cls_token)
    with torch.inference_mode():
        output = model.forward_features(torch.randn(1, 1, 16, 16))
    assert torch.isfinite(output["x_norm_clstoken"]).all()


def test_mismatched_weights_are_rejected(exported_vit_small, tmp_path):
    path, _ = exported_vit_small
    checkpoint = torch.load(path, weights_only=True)
    del checkpoint["model_state_dict"]["model_ema.backbone.cls_token"]
    incomplete = tmp_path / "incomplete.pth"
    torch.save(checkpoint, incomplete)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model.load_model_for_evaluation(incomplete)


@pytest.fixture(scope="module")
def hub_directory(exported_vit_small, tmp_path_factory):
    path, _ = exported_vit_small
    model = EMberModel.from_teacher_checkpoint(path)
    directory = tmp_path_factory.mktemp("hub-model")
    model.save_pretrained(directory)
    return directory


def test_hub_export_preserves_config_and_features(exported_vit_small, hub_directory):
    _, teacher = exported_vit_small
    config = json.loads((hub_directory / "config.json").read_text())
    assert config == asdict(DINOConfig(backbone="vit_small", img_size=16, patch_size=8))
    assert (hub_directory / "model.safetensors").is_file()
    model = EMberModel.from_pretrained(hub_directory, local_files_only=True)
    assert not model.training
    assert model.config.backbone == "vit_small"
    images = normalize_images(torch.randn(2, 1, 16, 24))
    with torch.inference_mode():
        expected = teacher.forward_features(images)
        actual = model(images)
    torch.testing.assert_close(
        actual.cls_token, expected["x_norm_clstoken"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.patch_tokens, expected["x_norm_patchtokens"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.pooler_output,
        torch.cat(
            (expected["x_norm_clstoken"], expected["x_norm_patchtokens"].mean(dim=1)),
            dim=-1,
        ),
        rtol=0,
        atol=0,
    )


def test_hub_repo_loading_passes_download_options(hub_directory, monkeypatch, tmp_path):
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(hub_directory / kwargs["filename"])

    monkeypatch.setattr(huggingface_hub.hub_mixin, "hf_hub_download", download)
    model = EMberModel.from_pretrained(
        "example/EMber-small",
        revision="release-v1",
        local_files_only=True,
        cache_dir=tmp_path,
        token=False,
    )
    assert not model.training
    assert {call["filename"] for call in downloads} == {
        "config.json",
        "model.safetensors",
    }
    for call in downloads:
        assert call["repo_id"] == "example/EMber-small"
        assert call["revision"] == "release-v1"
        assert call["local_files_only"] is True
        assert call["cache_dir"] == tmp_path
        assert call["token"] is False


def test_hub_loading_is_strict_by_default(hub_directory, tmp_path):
    (tmp_path / "config.json").write_bytes((hub_directory / "config.json").read_bytes())
    state = load_file(hub_directory / "model.safetensors")
    del state["backbone.cls_token"]
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        EMberModel.from_pretrained(tmp_path, local_files_only=True)
