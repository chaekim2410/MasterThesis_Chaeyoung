#!/usr/bin/env python3
"""
02c_validate_domain_labels.py  (REVISED v2, minimal)

Purpose
-------
Validate the bin-level domain labels produced in 02b:

  1. Spatial coherence (mean Moran's I) BEFORE (raw mmc_class_top1, from 02a's
     adata_annotated.h5ad) vs AFTER (domain_from_bin_class, from 02b's
     adata_labeled.h5ad) the confidence QC cut. Numeric equivalent of the
     MapMyCells paper's Fig.4B improvement (ARI 0.47 -> 0.743 after a 0.95 cut).
  2. A 1-figure, multi-panel MASK GRID: one subplot per domain, all domains
     shown together in a SINGLE saved PNG (not one file per domain).
  3. OPTIONAL marker-gene plausibility check, grouped by DOMAIN (not cluster),
     2-5 top genes per domain, all combined into ONE dotplot. This step is
     SWITCHED OFF by default (RUN_MARKER_VALIDATION = False) because the DE
     step is slow on the full bin-level dataset -- flip it on when needed.

Changes vs. previous version:
  - bin-vs-cluster "agreement/discordant" bookkeeping: removed (no separate
    cluster-level path exists anymore).
  - per-domain individual mask PNGs: replaced by a single combined grid figure.
  - marker validation: now grouped by domain (was by class), capped at
    2-5 genes/domain, and gated behind a runtime toggle.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


# =========================================================
# USER SETTINGS
# =========================================================
ANNOTATED_H5AD = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/02_annotation/cell_type_annotation/adata_annotated.h5ad"
)
LABELED_H5AD = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/02_annotation/domain_labels/adata_labeled.h5ad"
)

OUTDIR = Path("results/02_annotation/domain_validation")
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_CLASS_COL = "mmc_class_top1"          # from ANNOTATED_H5AD ("before")
DOMAIN_COL = "domain_from_bin_class"      # from LABELED_H5AD ("after")
LIBRARY_ID = None
KNN_K = 6

# --- Mask grid settings ---
MASK_GRID_NCOLS = 4

# --- Marker validation toggle (OFF by default -- slow on full bin-level data) ---
RUN_MARKER_VALIDATION = False
N_MARKER_GENES_PER_DOMAIN = 3   # between 2 and 5
MARKER_METHOD = "wilcoxon"


# =========================================================
# HELPERS
# =========================================================
def build_spatial_knn_graph(coords, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    idx = idx[:, 1:]
    n = coords.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    data = np.ones(len(rows), dtype=np.float32)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    return sparse.diags(1.0 / row_sum) @ W


def morans_I(x, W):
    x = np.asarray(x, dtype=np.float64)
    z = x - x.mean()
    num = z @ (W @ z)
    den = (z ** 2).sum()
    if den == 0:
        return np.nan
    n = len(x)
    s0 = W.sum()
    return (n / s0) * (num / den)


def spatial_coherence_score(labels, W, min_count=20, exclude_labels=("Unresolved",)):
    """
    One-vs-rest Moran's I per label, weighted-averaged by label frequency.
    Higher = more spatially contiguous (less salt-and-pepper).

    IMPORTANT: the full-length label array is kept aligned with W (the
    spatial graph) -- we do NOT drop bins. 'Unresolved' is skipped as a
    label to SCORE (not removed from the universe): a growing Unresolved
    mass would otherwise dominate the weighted average and mask whether
    the CONFIDENT/assigned labels actually got more spatially coherent.
    """
    labels = pd.Series(labels).astype(str)
    vc = labels.value_counts()
    vc = vc[~vc.index.isin(exclude_labels)]
    vc = vc[vc >= min_count]
    scores, weights = [], []
    for lab, n in vc.items():
        indicator = (labels == lab).astype(float).values
        scores.append(morans_I(indicator, W))
        weights.append(n)
    scores = np.array(scores)
    weights = np.array(weights, dtype=float)
    valid = ~np.isnan(scores)
    if valid.sum() == 0:
        return np.nan
    return float(np.average(scores[valid], weights=weights[valid]))


def plot_domain_mask_grid(coords, labels, out_png, ncols=4):
    """
    One combined figure: each domain gets its own subplot, highlighted
    against all other bins in light gray. All domains in a single PNG.
    """
    labels = pd.Series(labels).astype(str).values
    cats = sorted(pd.unique(labels))
    n = len(cats)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, cat in enumerate(cats):
        ax = axes[i]
        mask = labels == cat
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=1, c="lightgray", linewidths=0)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=1, c="crimson", linewidths=0)
        ax.set_title(cat, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.invert_yaxis()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================
def main():
    print("Reading annotated h5ad (for raw/before class)...")
    adata_raw = sc.read_h5ad(ANNOTATED_H5AD)
    adata_raw.obs_names_make_unique()

    print("Reading labeled h5ad (for domain/after)...")
    adata = sc.read_h5ad(LABELED_H5AD)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if DOMAIN_COL not in adata.obs.columns:
        raise ValueError(f"{DOMAIN_COL} not found in {LABELED_H5AD}. Run 02b first.")
    if RAW_CLASS_COL not in adata_raw.obs.columns:
        raise ValueError(f"{RAW_CLASS_COL} not found in {ANNOTATED_H5AD}.")

    # Align the two objects by obs_names (same bins, same order expected).
    common = adata.obs_names.intersection(adata_raw.obs_names)
    if len(common) != adata.n_obs:
        print(f"[WARN] {adata.n_obs - len(common)} bins not found in annotated h5ad; "
              f"restricting comparison to {len(common)} common bins.")
    adata = adata[common].copy()
    raw_class = adata_raw.obs.loc[common, RAW_CLASS_COL]

    coords = np.asarray(adata.obsm["spatial"])

    # -----------------------------------------------------
    # 1) Spatial coherence (Moran's I) before vs after QC
    # -----------------------------------------------------
    print("Computing spatial coherence (Moran's I) before vs after QC...")
    W = build_spatial_knn_graph(coords, KNN_K)
    score_before = spatial_coherence_score(raw_class, W)
    score_after = spatial_coherence_score(adata.obs[DOMAIN_COL], W)

    coherence_df = pd.DataFrame({
        "stage": ["before_QC (raw class)", "after_QC (domain_from_bin_class)"],
        "mean_morans_I": [score_before, score_after],
    })
    coherence_df.to_csv(OUTDIR / "spatial_coherence_before_after.csv", index=False)
    print(coherence_df.to_string(index=False))

    # -----------------------------------------------------
    # 2) Domain mask grid (all domains in ONE combined figure)
    # -----------------------------------------------------
    print("Saving combined domain mask grid...")
    plot_domain_mask_grid(
        coords, adata.obs[DOMAIN_COL], OUTDIR / "domain_mask_grid.png",
        ncols=MASK_GRID_NCOLS,
    )

    # -----------------------------------------------------
    # 3) OPTIONAL marker validation, grouped by domain
    #    (OFF by default -- set RUN_MARKER_VALIDATION = True to enable)
    # -----------------------------------------------------
    if RUN_MARKER_VALIDATION:
        print("Running marker DE grouped by domain (this can be slow)...")
        adata_plot = adata.copy()
        sc.pp.normalize_total(adata_plot, target_sum=1e4)
        sc.pp.log1p(adata_plot)

        vc = adata_plot.obs[DOMAIN_COL].value_counts()
        valid_domains = [d for d in vc[vc >= 10].index.tolist() if d != "Unresolved"]

        if len(valid_domains) >= 2:
            sub = adata_plot[adata_plot.obs[DOMAIN_COL].isin(valid_domains)].copy()
            sub.obs[DOMAIN_COL] = sub.obs[DOMAIN_COL].cat.remove_unused_categories()
            sc.tl.rank_genes_groups(
                sub, groupby=DOMAIN_COL, method=MARKER_METHOD,
                use_raw=False, key_added="rank_genes_domain",
            )
            top_markers = sc.get.rank_genes_groups_df(sub, group=None, key="rank_genes_domain")
            top_markers = top_markers.groupby("group", sort=False).head(N_MARKER_GENES_PER_DOMAIN)
            top_markers.to_csv(OUTDIR / "domain_top_markers.csv", index=False)

            genes = top_markers["names"].astype(str).unique().tolist()
            if len(genes) > 0:
                dp = sc.pl.dotplot(
                    sub, var_names=genes, groupby=DOMAIN_COL,
                    standard_scale="var", show=False, return_fig=True,
                )
                dp.savefig(OUTDIR / "domain_marker_dotplot.png", dpi=200, bbox_inches="tight")
                plt.close("all")
        else:
            print("[skip] fewer than 2 valid domains -- no marker DE run")
    else:
        print("[skip] RUN_MARKER_VALIDATION is False -- skipping marker DE/dotplot")

    # -----------------------------------------------------
    # 4) Summary
    # -----------------------------------------------------
    summary = [
        "=== 02c Validation: confidence-QC effect + domain plausibility ===",
        f"Annotated h5ad (before): {ANNOTATED_H5AD}",
        f"Labeled h5ad (after):    {LABELED_H5AD}",
        "",
        "Spatial coherence (mean Moran's I, higher = less salt-and-pepper):",
        f"  before QC (raw class):           {score_before:.4f}",
        f"  after QC (domain_from_bin_class): {score_after:.4f}",
        "",
        f"Marker validation run: {RUN_MARKER_VALIDATION} "
        f"(set RUN_MARKER_VALIDATION=True at top of script to enable)",
        "",
        "Output files:",
        "  - spatial_coherence_before_after.csv",
        "  - domain_mask_grid.png  (single figure, all domains)",
        "  - domain_top_markers.csv / domain_marker_dotplot.png (only if enabled)",
    ]
    (OUTDIR / "summary_02c_validation.txt").write_text("\n".join(summary))

    print("[DONE]")


if __name__ == "__main__":
    main()