#!/usr/bin/env python3
"""
Per-dataset variance / diversity of the reference images (table dataset-variance),
supporting the memorization analysis.

Operates on the on-disk reference (test) PNG dumps, grayscale, fixed sample per
dataset. Reports four complementary metrics and writes a LaTeX table plus a
CSV (results/dataset_variance.csv is the released copy).

Needs the reference dumps written by evaluate_generation.py --save_references,
under <RMT_BASE_DIR>/outputs/<RMT_PROJECT>/reference/.

  python analysis/table_dataset_variance.py
"""
import csv, os, numpy as np
from matplotlib.image import imread

import _paths

REF = os.path.join(_paths.RUNS, "reference")
DS = [("pathmnist","Path"),("chestmnist","Chest"),("dermamnist","Derma"),
      ("octmnist","OCT"),("pneumoniamnist","Pneu"),("retinamnist","Reti"),
      ("breastmnist","Breast"),("bloodmnist","Blood"),("tissuemnist","Tiss"),
      ("organamnist","OrgA"),("organcmnist","OrgC"),("organsmnist","OrgS")]
N = 1200          # sample size per dataset
SEED = 0


def load_sample(ds, n):
    d = os.path.join(REF, ds, "reference")
    fs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    rng = np.random.RandomState(SEED)
    if len(fs) > n:
        fs = [fs[i] for i in rng.choice(len(fs), n, replace=False)]
    X = []
    for f in fs:
        im = imread(os.path.join(d, f))
        if im.ndim == 3:
            im = im[..., :3].mean(2)        # luminance-ish grayscale
        X.append(im.astype(np.float64).ravel())
    return np.stack(X)                       # (m, 224*224)


def metrics(X):
    m = X.shape[0]
    pixel_std = float(X.std())               # global intensity spread
    # effective (stable) rank via Gram matrix of centred data
    Xc = X - X.mean(0, keepdims=True)
    G = Xc @ Xc.T                            # (m, m)
    ev = np.linalg.eigvalsh(G)
    ev = np.clip(ev, 0, None)
    eff_rank = float(ev.sum() / ev.max())    # trace / lambda_max
    # cosine sim on unit-normalised raw images
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = U @ U.T                              # cosine sim matrix
    iu = np.triu_indices(m, 1)
    mean_pair_dist = float(1.0 - S[iu].mean())   # mean pairwise cosine distance
    np.fill_diagonal(S, -np.inf)
    nn_sim = float(np.mean(S.max(1)))        # mean nearest-neighbour cos sim
    return pixel_std, eff_rank, nn_sim, mean_pair_dist


def main():
    rows = []
    for ds, lbl in DS:
        X = load_sample(ds, N)
        ps, er, nn, pd = metrics(X)
        rows.append((lbl, ds, X.shape[0], ps, er, nn, pd))
        print(f"{lbl:7s} m={X.shape[0]:4d}  std={ps:.3f}  eff_rank={er:6.1f}  "
              f"nn_sim={nn:.3f}  pair_dist={pd:.3f}")
    # rank by effective rank (ascending = lowest variance first)
    print("\nLowest-variance (by effective rank):")
    for r in sorted(rows, key=lambda r: r[4])[:4]:
        print(f"  {r[0]:7s} eff_rank={r[4]:.1f}  nn_sim={r[5]:.3f}")

    # LaTeX table
    with open(_paths.out("dataset_variance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "label", "n", "pixel_std", "eff_rank", "nn_sim", "pairwise_dist"])
        for lbl, ds, m, ps, er, nn, pd in sorted(rows, key=lambda r: r[4]):
            w.writerow([ds, lbl, m, f"{ps:.6f}", f"{er:.6f}", f"{nn:.6f}", f"{pd:.6f}"])
    out = _paths.out("table_dataset_variance.tex")
    with open(out, "w") as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Dataset & Pixel std $\\downarrow$ & Eff.\\ rank $\\downarrow$ & "
                "NN sim $\\uparrow$ & Pairwise dist $\\downarrow$ \\\\\n\\midrule\n")
        for lbl, ds, m, ps, er, nn, pd in sorted(rows, key=lambda r: r[4]):
            f.write(f"{lbl} & {ps:.3f} & {er:.1f} & {nn:.3f} & {pd:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
