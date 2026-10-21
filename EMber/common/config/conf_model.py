from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class DINOConfig:
    """Backbone settings retained from the training checkpoint configuration."""

    # defaults pulled from https://github.com/facebookresearch/dinov3/blob/31703e4cbf1ccb7c4a72daa1350405f86754b6d1/dinov3/configs/train/dinov3_vit7b16_pretrain.yaml
    backbone: str = "vit_base"
    img_size: int = 128
    patch_size: int = 16
    pos_embed_rope_base: float = 100.0
    pos_embed_rope_min_period: float | None = None
    pos_embed_rope_max_period: float | None = None
    pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate"
    pos_embed_rope_shift_coords: float | None = None
    pos_embed_rope_jitter_coords: float | None = None
    pos_embed_rope_rescale_coords: int | None = 2
    pos_embed_rope_dtype: str = "bf16"
    qkv_bias: bool = False
    layerscale_init: float | None = 1.0e-05
    norm_layer: str = "layernormbf16"
    ffn_layer: str = "swiglu64"
    ffn_bias: bool = True
    proj_bias: bool = True
    n_storage_tokens: int = 4
    mask_k_bias: bool = True
    untie_cls_and_patch_norms: bool = False
    untie_global_and_local_cls_norm: bool = True
    in_chans: int = 1
    # Only used when constructing a student backbone.
    drop_path_rate: float = 0.2


ModelConfig = DINOConfig
