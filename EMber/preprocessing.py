"""Preprocessing shared by downstream EMber users."""

import torch


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Standardize each image to mean 0 and sample standard deviation 1.

    Accept a tensor with height and width as its last two dimensions, e.g.
    (H, W), (B, H, W), or (B, 1, H, W). Returns float32 on the input device,
    preserving shape without changing the input. Matches the training/evaluation
    normalization, including the 1e-6 standard-deviation floor for constant images.
    """
    images = images.float()
    mean = images.mean(dim=(-2, -1), keepdim=True)
    std = images.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (images - mean) / std
