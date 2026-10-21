from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from EMber.common.checkpoint_keys import (
    TEACHER_CHECKPOINT_FORMAT,
    teacher_backbone_state_dict,
)
from EMber.common.config.conf_model import DINOConfig
from EMber.models.build import build_models


class TeacherEvaluationModel(nn.Module):
    """Teacher backbone with the downstream feature-extraction contract."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward_features(
        self, batch: dict[str, Any] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        if isinstance(batch, torch.Tensor):
            return self.backbone.forward_features(batch)
        images = batch["image"]
        if images.ndim == 3:
            batch_size, height, width = images.shape
        elif images.ndim == 4:
            batch_size, channels, height, width = images.shape
            assert channels == 1, f"expected one image channel, got {channels}"
        else:
            raise ValueError(
                "expected image shape (B, H, W) or (B, 1, H, W), "
                f"got {tuple(images.shape)}"
            )
        images = images.reshape(batch_size, 1, height, width)
        output = self.backbone(images, is_training=True)
        cls_token = output["x_norm_clstoken"].reshape(batch_size, -1)
        image_tokens = output["x_norm_patchtokens"].reshape(
            batch_size, output["x_norm_patchtokens"].shape[1], -1
        )
        return cls_token, image_tokens

    def pool_features(
        self, cls_token: torch.Tensor, image_tokens: torch.Tensor
    ) -> torch.Tensor:
        assert cls_token.ndim == 2
        assert image_tokens.ndim == 3
        return torch.cat((cls_token, image_tokens.mean(dim=1)), dim=-1)


@dataclass
class EMberOutput:
    """CLS (B, D), patch (B, N, D), and concatenated pooled (B, 2D) features."""

    cls_token: torch.Tensor
    patch_tokens: torch.Tensor
    pooler_output: torch.Tensor


class EMberModel(
    TeacherEvaluationModel,
    PyTorchModelHubMixin,
    library_name="EMber",
    tags=["cryo-em", "image-feature-extraction"],
):
    """Cryo-EM backbone with Hugging Face Hub saving and loading.

    from_pretrained accepts a Hub repository ID or a save_pretrained directory.
    Inputs to forward are already normalized grayscale tensors, shaped (B, H, W)
    or (B, 1, H, W). Use EMber.normalize_images for per-image standardization.
    """

    def __init__(self, config: DINOConfig):
        backbone = build_models(config, teacher_only=True)
        assert isinstance(backbone, nn.Module)
        super().__init__(backbone)
        self.config = config
        self.backbone.init_weights()

    def forward(self, pixel_values: torch.Tensor) -> EMberOutput:
        cls_token, patch_tokens = self.forward_features({"image": pixel_values})
        return EMberOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            pooler_output=self.pool_features(cls_token, patch_tokens),
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, *, strict: bool = True, **kwargs
    ) -> "EMberModel":
        """Load Hub or local weights strictly, on CPU and in eval mode by default.

        Hugging Face options such as revision, cache_dir, local_files_only,
        token, and map_location are passed through to the Hub mixin.
        """
        return super().from_pretrained(
            pretrained_model_name_or_path, strict=strict, **kwargs
        )

    @classmethod
    def from_teacher_checkpoint(
        cls, ckpt_path: str | Path, *, load_weights: bool = True
    ) -> "EMberModel":
        """Import an existing exported teacher .pth; no training repo is required.

        Call save_pretrained on the result to create config.json and
        model.safetensors for upload to the Hub.
        """
        ckpt_path = Path(ckpt_path)
        if ckpt_path.suffix != ".pth":
            raise ValueError(f"Expected an exported teacher .pth file: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        config = model_config_dict(ckpt_path, checkpoint)
        model_conf = DINOConfig(
            **{
                field.name: config[field.name]
                for field in fields(DINOConfig)
                if field.name in config
            }
        )
        model = cls(model_conf)
        if load_weights:
            model.backbone.load_state_dict(
                teacher_backbone_state_dict(checkpoint["model_state_dict"])
            )
        return model.eval()


def model_config_dict(ckpt_path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Read an exported teacher's self-contained model configuration."""
    if checkpoint.get("format") != TEACHER_CHECKPOINT_FORMAT:
        raise ValueError(f"Expected an exported teacher checkpoint: {ckpt_path}")
    if checkpoint.get("model_config") is None:
        raise ValueError(
            f"Teacher checkpoint has no embedded model_config: {ckpt_path}"
        )
    return checkpoint["model_config"]


def load_model_for_evaluation(
    ckpt_path: str | Path, *, load_weights: bool = True
) -> EMberModel:
    """Load a self-contained teacher export on CPU in evaluation mode.

    Only backbone configuration is needed; training settings in the exported
    model_config are ignored. With load_weights=False, initialize a random
    backbone with the checkpoint's architecture instead.
    """
    return EMberModel.from_teacher_checkpoint(ckpt_path, load_weights=load_weights)
