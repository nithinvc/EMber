"""Construct transformer backbones from their saved model configuration."""

from typing import Any

from torch import nn

from EMber.common.config.conf_model import DINOConfig
from EMber.models.vision_transformer import (
    vit_base,
    vit_giant2,
    vit_huge2,
    vit_large,
    vit_small,
)


def backbone_kwargs(conf: DINOConfig) -> dict[str, Any]:
    return dict(
        img_size=conf.img_size,
        patch_size=conf.patch_size,
        pos_embed_rope_base=conf.pos_embed_rope_base,
        pos_embed_rope_min_period=conf.pos_embed_rope_min_period,
        pos_embed_rope_max_period=conf.pos_embed_rope_max_period,
        pos_embed_rope_normalize_coords=conf.pos_embed_rope_normalize_coords,
        pos_embed_rope_shift_coords=conf.pos_embed_rope_shift_coords,
        pos_embed_rope_jitter_coords=conf.pos_embed_rope_jitter_coords,
        pos_embed_rope_rescale_coords=conf.pos_embed_rope_rescale_coords,
        pos_embed_rope_dtype=conf.pos_embed_rope_dtype,
        qkv_bias=conf.qkv_bias,
        layerscale_init=conf.layerscale_init,
        norm_layer=conf.norm_layer,
        ffn_layer=conf.ffn_layer,
        ffn_bias=conf.ffn_bias,
        proj_bias=conf.proj_bias,
        n_storage_tokens=conf.n_storage_tokens,
        mask_k_bias=conf.mask_k_bias,
        untie_cls_and_patch_norms=conf.untie_cls_and_patch_norms,
        untie_global_and_local_cls_norm=conf.untie_global_and_local_cls_norm,
        in_chans=conf.in_chans,
    )


def build_models(
    conf: DINOConfig, teacher_only: bool = False
) -> nn.Module | tuple[nn.Module, nn.Module, int]:
    assert conf.backbone in (
        "vit_small",
        "vit_base",
        "vit_large",
        "vit_huge2",
        "vit_giant2",
    )
    kwargs = backbone_kwargs(conf)
    if conf.backbone == "vit_small":
        cls_ = vit_small
    elif conf.backbone == "vit_base":
        cls_ = vit_base
    elif conf.backbone == "vit_large":
        cls_ = vit_large
    elif conf.backbone == "vit_huge2":
        cls_ = vit_huge2
    elif conf.backbone == "vit_giant2":
        cls_ = vit_giant2
    teacher = cls_(**kwargs)
    if teacher_only:
        return teacher
    student = cls_(**kwargs, drop_path_rate=conf.drop_path_rate)
    return teacher, student, teacher.embed_dim
