import pytest
import torch

from EMber import normalize_images


@pytest.mark.parametrize("shape", [(4, 8), (3, 4, 8), (3, 1, 4, 8)])
def test_normalization_is_per_image_and_preserves_input(shape):
    images = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
    original = images.clone()
    result = normalize_images(images)
    assert result.shape == images.shape
    assert result.dtype == torch.float32
    assert result.device == images.device
    torch.testing.assert_close(images, original)
    torch.testing.assert_close(
        result.mean(dim=(-2, -1)), torch.zeros(shape[:-2]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(result.std(dim=(-2, -1)), torch.ones(shape[:-2]))


def test_constant_images_normalize_to_zero():
    result = normalize_images(torch.full((2, 1, 16, 16), 7.0))
    torch.testing.assert_close(result, torch.zeros_like(result))
