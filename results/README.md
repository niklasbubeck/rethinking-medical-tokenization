# Released results

Every number in the paper's tables and figures is computed from these files by the
scripts in `analysis/`. They cover the 30 configurations of the study (10 tokenizer
families x compression factors f = 4, 8, 16) on the 12 MedMNIST+ 2D datasets.

Models use their `medlat` registry names, e.g. `discrete.rqvae.f8_d4_e16384`:
family `rqvae`, spatial compression `f8`, latent channels `d4`, codebook size
`e16384` (lookup-free families use `b<bits>` instead).

| File | Rows | Produced by | Used for |
|---|---|---|---|
| `tokenizer_results.csv` | 360 + pooled | `evaluate_tokenizer.py` + `calculate_fids.py`, merged by `analysis/aggregate_results.py` | Table 1, Figs. 2 and 4, supplementary tables |
| `fids_reconstruction.csv` | 360 | `calculate_fids.py --mode reconstruction` | Table 1 (rFID, rFMD) |
| `fids_generation.csv` | 360 | `calculate_fids.py --mode generation` | gFID, gFMD |
| `memorization.csv` | 720 | `memorization.py --extractor both` | Fig. 5, memorization tables |
| `linear_probing.csv` | 648 | `linear_probing.py` | Fig. 4 (probe gap), probing table |
| `dperbit_entropy_f8.csv` | 8 | `codebook_misalignment.py`, collected by `analysis/collect_rate_distortion.py` | rate-distortion figure, Fig. 3 annotations |
| `dataset_variance.csv` | 12 | `analysis/table_dataset_variance.py` | dataset-variance table, diversity figure |
| `natural.csv` | 40 | Xu et al. (2026), reported values | natural-image half of Fig. 2 |

## Columns

**`tokenizer_results.csv`**, one row per (model, dataset); `dataset == "_all_"` rows
pool all test images of a model.
- `psnr`, `ssim`, `lpips`, `mse`, `mae` (+ `_std`): reconstruction quality on the test split.
- `rfid`, `rfmd`: reconstruction Frechet distances in Inception-V3 / MedMNIST
  ResNet-18 feature space; `gfid`, `gfmd`: the same for DiT-B/2 samples.
- `codebook_usage`, `codebook_entropy`, `codebook_utility`: fraction of codes used,
  normalized usage entropy, and 1/(1 + CV) of the code frequencies.
- `latent_stable_rank`, `latent_effective_rank`, `latent_anisotropy`: geometry of
  the pre-quantization latents (the paper reports stable rank / d).

**`fids_*.csv`**: `fid`, `fmd`, the FMD backbone (`fmd_arch`, `seed`), and the
number of sample / reference images.

**`memorization.csv`**, one row per (model, dataset, extractor in {sscd, dinov2}):
`copy_rate` (fraction of samples whose top-1 train similarity exceeds the 95th
percentile of test-to-train similarities), `mem_ratio` (median nearest / mean
next-k neighbour distance, k = 50), `base_p95` (the calibrated threshold), and
sample counts.

**`linear_probing.csv`**, one row per (model, latent variant, dataset): `variant`
is `pre` (z_e) or `post` (z_q) for discrete models and `default` for continuous
ones; test metrics `test_top1`, `test_auroc_macro`, `test_qwk`. The primary metric
is top-1 (multi-class), macro-AUROC (multi-label ChestMNIST) or quadratic-weighted
kappa (ordinal RetinaMNIST). `run` is an internal run identifier.

**`dperbit_entropy_f8.csv`**, one row per discrete family at f = 8: allocated bits
`B_bits`, empirical code entropy `H_C_bits`, quantization error `D`, and
`D_per_H_new` = D / H(C).

**`dataset_variance.csv`**: per-dataset pixel std, effective rank, mean
nearest-neighbour cosine similarity and mean pairwise cosine distance of up to 1,200
grayscale test images.
