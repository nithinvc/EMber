<h1 align="center"><em>EMber</em>: Learning the Shared Latent Structure of Cryo-EM Inverse Problems</h1>

<!-- Replace each href="#" with the author's website URL. -->
<p align="center">
  <a href="https://nithinvc.github.io/">Nithin Chalapathi</a> ·
  <a href="https://www.theosternlieb.com/">Theo Sternlieb</a> ·
  <a href="#">Ethan Han</a> ·
  <a href="https://sonyahanson.github.io/">Sonya M. Hanson</a> ·
  <a href="https://pilarcossio.org/">Pilar Cossio</a> ·
  <a href="https://a1k12.github.io/">Aditi S. Krishnapriyan</a> ·
  <a href="https://ritwikgupta.me/">Ritwik Gupta</a>
</p>

EMber is a family of pretrained vision transformers for extracting representations from single-particle cryo-EM particle images. Given a particle image, EMber produces an image-level embedding and spatial patch features for use in downstream analysis. The models were trained using self-distillation on over 50 million experimental particle images from 214 EMPIAR datasets, together with simulated projections from 268,000 molecular assemblies. This repository contains the model definition and instructions for inference. Evaluation, training, and data processing code coming soon! 

Check out our preprint [here](https://arxiv.org/abs/2610.22523)!

This is joint work between the [ASK Lab at BAIR](https://a1k12.github.io/), the [Flatiron Institute CCM](https://www.simonsfoundation.org/flatiron/computational-mathematics/), and the [RAGS Lab at UMD](https://ragslab.ai/). If you find this work useful, please consider citing:

```bibtex
@misc{chalapathi2026learningsharedlatentstructure,
      title={Learning the Shared Latent Structure of Cryo-EM Inverse Problems}, 
      author={Nithin Chalapathi and Theo Sternlieb and Ethan Han and Sonya M. Hanson and Pilar Cossio and Aditi S. Krishnapriyan and Ritwik Gupta},
      year={2026},
      eprint={2610.22523},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.22523}, 
}
```

## Updates

- [Oct 21, 2026] Initial Model Release!

## Quick Start Guide

### Running Inference
The trained model weights are stored on HuggingFace.

| Model | Parameters | Hugging Face Link |
| --- | --- | --- |
| EMber-S | 21.4M | [EMber-small](https://huggingface.co/nithinvc/EMber-small) |
| EMber-B | 85.3M | [EMber-base](https://huggingface.co/nithinvc/EMber-base) |
| EMber-L | 304.2M | [EMber-large](https://huggingface.co/nithinvc/EMber-large) |
| EMber-g/2 | 1.1B | [EMber-giant](https://huggingface.co/nithinvc/EMber-giant) |


First, setup the repository using [uv](https://docs.astral.sh/uv/):

```bash
# If directly working out of the EMber repository root
uv sync
# Install in an existing uv venv
uv pip install .
# Alternatively: pip install .
```

The weights can be loaded and run from Huggingface using the repository model definition:
```python
import torch
from EMber import EMberModel, normalize_images

device =  torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

model = EMberModel.from_pretrained("nithinvc/EMber-base")
# Optional, but highly recommended for performance
model.compile() 

# Replace with your particle images: (B, 1, H, W) or (B, H, W).
images = torch.randn(2, 1, 384, 384)
# Normalize images to 0 mean, unit standard deviation
images = normalize_images(images.to(device))

amp_dtype = (
    torch.bfloat16
    if device.type == "cuda"
    else torch.float32
)

with torch.inference_mode(), torch.autocast(
    device_type=device.type,
    dtype=amp_dtype,
    enabled=device.type == "cuda",
):
    output = model(images)

# Dataclass output with keys: 
# - cls_token: shape (B, Embed Dim)
# - patch_tokens: Shape (B, Num patches, Embed Dim)
# - pooler_output: Shape (B, 2 * Embed Dim), concatenation of cls_token + avg pooled patch tokens
embeddings = output.pooler_output.float().cpu()
```


## Acknowledgements
This repository contains a mix of code from [DinoV2](https://github.com/facebookresearch/dinov2)/[DinoV3](https://github.com/facebookresearch/dinov3), [CryoSBI](https://github.com/flatironinstitute/cryoSBI), [CryoJax](https://github.com/michael-0brien/cryojax), and [CryoSpax](https://github.com/michael-0brien/cryospax). 
All third-party code remains subject to the licenses of its respective original repositories.
