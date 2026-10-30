# Third-party notices

This repository is released under the MIT license (see `LICENSE`). Parts of it are
adapted from, or depend on, the following projects, which keep their own licenses.

## Code adapted in this repository

| Path | Origin | License |
|---|---|---|
| `src/diffusion/` | [openai/guided-diffusion](https://github.com/openai/guided-diffusion) (Gaussian diffusion, timestep respacing), itself a port of [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) | MIT |
| `src/losses/lpips.py` | [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers), stripped from [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) | MIT / BSD-2-Clause |
| `src/losses/discriminators/discriminator.py` (PatchGAN) | [taming-transformers](https://github.com/CompVis/taming-transformers), after [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | MIT / BSD |
| `src/losses/discriminators/util.py` (`DiffAugment`) | [mit-han-lab/data-efficient-gans](https://github.com/mit-han-lab/data-efficient-gans) | BSD-2-Clause |
| `src/losses/discriminators/util.py` (LeCAM regularizer) | [google/lecam-gan](https://github.com/google/lecam-gan) | Apache-2.0 |

## Dependencies and external assets

- **[MedLat](https://github.com/niklasbubeck/MedLat)** provides the tokenizer and DiT
  architectures. It is installed as a pinned dependency, not vendored here; see its
  repository for license terms.
- **MedMNIST+** ([medmnist.com](https://medmnist.com/)): dataset released under
  CC BY 4.0; the individual source datasets carry their own terms.
- **MedMNIST baseline ResNets**, used as the per-dataset feature extractor for FMD:
  published by the MedMNIST authors.
- **[SSCD](https://github.com/facebookresearch/sscd-copy-detection)** copy-detection
  descriptor (MIT) and **[DINOv2](https://github.com/facebookresearch/dinov2)**
  (Apache-2.0), used for the memorization analysis.
- **LPIPS VGG weights**, downloaded at runtime from the taming-transformers release.
- `results/natural.csv` reproduces the natural-image numbers reported by Xu et al.,
  *Making Reconstruction FID Predictive of Diffusion Generation FID* (2026),
  used for the natural-image half of Figure 2.
