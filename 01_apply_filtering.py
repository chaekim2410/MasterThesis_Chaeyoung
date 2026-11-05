#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy import sparse


# =========================
# USER SETTINGS
# =========================
H5AD = Path("./results/00_prefilter_qc/adata_prefilter_cluster.h5ad")
OUTDIR = Path("./results/01_postfilter")
OUTDIR.mkdir(parents=True, exist_ok=True)

LIBRARY_ID = None
CLUSTER_COL = "spaceranger_cluster"
PASS_COL = "pass_postfilter_qc"

# Filtering thresholds
MIN_COUNTS = 300
MIN_GENES = 250
MAX_PCT_MT = 10.0
MIN_BINS_PER_GENE = 10

# UMAP settings (visualization only; no new clustering performed)
USE_HVG_FOR_UMAP = True
N_TOP_HVGS = 3000
N_PCS = 30
N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.5
UMAP_SPREAD = 1.0
RANDOM_SEED = 42


# =========================
# HELPERS
# =========================
def get_library_id(adata, user_library_id=None):
    if user_library_id is not None:
        return user_library_id

    libs = list(adata.uns.get("spatial", {}).keys())
    if not libs:
        raise ValueError("No spatial library found in adata.uns['spatial'].")
    return libs[0]


def choose_img_key(adata, library_id):
    available_images = adata.uns["spatial"][library_id]["images"]
    return "hires" if "hires" in available_images else list(available_images.keys())[0]


def calculate_gene_detection(adata):
    if sparse.issparse(adata.X):
        return np.asarray((adata.X > 0).sum(axis=0)).ravel()
    return np.asarray((adata.X > 0).sum(axis=0)).ravel()


def get_mt_vmax_for_plot(adata):
    mt_mask = (adata.obs["total_counts"] > 0) & np.isfinite(adata.obs["pct_counts_mt"])
    mt_vals = adata.obs.loc[mt_mask, "pct_counts_mt"].to_numpy()
    if len(mt_vals) == 0:
        return None
    return float(np.percentile(mt_vals, 99))


def short_list(values, max_n=30):
    values = [str(v) for v in values]
    if len(values) <= max_n:
        return ", ".join(values) if values else "None"
    return ", ".join(values[:max_n]) + f", ... (+{len(values) - max_n} more)"


def infer_x_status(adata):
    x = adata.X

    if sparse.issparse(x):
        x_data = x.data
        is_sparse = True
        x_dtype = x.dtype
        x_format = x.getformat()
    else:
        x_data = np.asarray(x).ravel()
        is_sparse = False
        x_dtype = np.asarray(x).dtype
        x_format = "dense"

    if x_data.size == 0:
        return {
            "is_sparse": is_sparse,
            "x_dtype": str(x_dtype),
            "x_format": x_format,
            "x_min": np.nan,
            "x_max": np.nan,
            "x_has_negative": False,
            "x_has_fractional": False,
            "looks_integer_like": False,
            "guess": "unknown (empty matrix)",
        }

    finite = np.isfinite(x_data)
    x_data = x_data[finite]

    if x_data.size == 0:
        return {
            "is_sparse": is_sparse,
            "x_dtype": str(x_dtype),
            "x_format": x_format,
            "x_min": np.nan,
            "x_max": np.nan,
            "x_has_negative": False,
            "x_has_fractional": False,
            "looks_integer_like": False,
            "guess": "unknown (non-finite values)",
        }

    x_min = float(np.min(x_data))
    x_max = float(np.max(x_data))
    x_has_negative = bool(np.any(x_data < 0))
    x_has_fractional = bool(np.any(np.abs(x_data - np.round(x_data)) > 1e-8))
    looks_integer_like = not x_has_fractional

    if x_has_negative:
        guess = "possibly scaled / centered data (contains negative values)"
    elif looks_integer_like:
        guess = "likely raw counts / count-like matrix"
    else:
        row_sums = (
            np.asarray(adata.X.sum(axis=1)).ravel()
            if sparse.issparse(adata.X)
            else np.asarray(adata.X.sum(axis=1)).ravel()
        )
        row_sums = row_sums[np.isfinite(row_sums)]
        if len(row_sums) > 0:
            median_row_sum = float(np.median(row_sums))
            if 5_000 <= median_row_sum <= 20_000:
                guess = "possibly normalized data (row sums around target_sum)"
            elif x_max < 20:
                guess = "possibly log-transformed normalized data"
            else:
                guess = "non-integer non-negative matrix; normalization/log may have been applied"
        else:
            guess = "unknown"

    return {
        "is_sparse": is_sparse,
        "x_dtype": str(x_dtype),
        "x_format": x_format,
        "x_min": x_min,
        "x_max": x_max,
        "x_has_negative": x_has_negative,
        "x_has_fractional": x_has_fractional,
        "looks_integer_like": looks_integer_like,
        "guess": guess,
    }


# =========================
# FILTERING
# =========================
def apply_filtering(adata):
    """
    Apply bin-level QC filtering and gene-level detection filtering.
    Keeps vendor cluster column intact.
    Adds PASS_COL to the original object.
    """
    ad = adata.copy()

    bin_mask = (
        (ad.obs["total_counts"] >= MIN_COUNTS)
        & (ad.obs["n_genes_by_counts"] >= MIN_GENES)
        & (ad.obs["pct_counts_mt"] <= MAX_PCT_MT)
    )

    # mark pass/fail on original object
    adata.obs[PASS_COL] = np.where(bin_mask.values, "PASS", "FAIL")
    adata.obs[PASS_COL] = pd.Categorical(
        adata.obs[PASS_COL],
        categories=["PASS", "FAIL"],
        ordered=True,
    )

    ad = ad[bin_mask].copy()

    gene_det = calculate_gene_detection(ad)
    gene_mask = gene_det >= MIN_BINS_PER_GENE
    ad = ad[:, gene_mask].copy()

    return ad


def build_filtering_impact_table(adata_before, adata_after):
    gene_det_before = calculate_gene_detection(adata_before)

    fail_counts = int((adata_before.obs["total_counts"] < MIN_COUNTS).sum())
    fail_genes = int((adata_before.obs["n_genes_by_counts"] < MIN_GENES).sum())
    fail_mt = int((adata_before.obs["pct_counts_mt"] > MAX_PCT_MT).sum())
    fail_gene_det = int((gene_det_before < MIN_BINS_PER_GENE).sum())

    combined_fail = int(adata_before.n_obs - adata_after.n_obs)

    return pd.DataFrame(
        [
            {
                "level": "bin",
                "rule": f"total_counts < {MIN_COUNTS}",
                "n_fail": fail_counts,
                "n_total": adata_before.n_obs,
                "pct_fail": fail_counts / adata_before.n_obs * 100,
            },
            {
                "level": "bin",
                "rule": f"n_genes_by_counts < {MIN_GENES}",
                "n_fail": fail_genes,
                "n_total": adata_before.n_obs,
                "pct_fail": fail_genes / adata_before.n_obs * 100,
            },
            {
                "level": "bin",
                "rule": f"pct_counts_mt > {MAX_PCT_MT}",
                "n_fail": fail_mt,
                "n_total": adata_before.n_obs,
                "pct_fail": fail_mt / adata_before.n_obs * 100,
            },
            {
                "level": "gene",
                "rule": f"detected_in_bins < {MIN_BINS_PER_GENE}",
                "n_fail": fail_gene_det,
                "n_total": adata_before.n_vars,
                "pct_fail": fail_gene_det / adata_before.n_vars * 100,
            },
            {
                "level": "final",
                "rule": "after all filters",
                "n_fail": combined_fail,
                "n_total": adata_before.n_obs,
                "pct_fail": combined_fail / adata_before.n_obs * 100,
            },
        ]
    )


def build_cluster_retention_table(adata_before, adata_after, cluster_col):
    before_counts = adata_before.obs[cluster_col].value_counts(dropna=False).sort_index()
    after_counts = adata_after.obs[cluster_col].value_counts(dropna=False).sort_index()

    all_clusters = sorted(set(before_counts.index.astype(str)).union(set(after_counts.index.astype(str))))

    rows = []
    for cl in all_clusters:
        n_before = int(before_counts.get(cl, 0))
        n_after = int(after_counts.get(cl, 0))
        retained_pct = (n_after / n_before * 100) if n_before > 0 else np.nan
        removed_n = n_before - n_after
        rows.append(
            {
                "cluster": cl,
                "n_before": n_before,
                "n_after": n_after,
                "n_removed": removed_n,
                "retained_pct": retained_pct,
            }
        )

    return pd.DataFrame(rows)


# =========================
# PLOTS: QC before/after
# =========================
def save_hist_compare(adata_before, adata_after, out_png):
    metrics = [
        ("total_counts", "Total counts"),
        ("n_genes_by_counts", "Detected genes"),
        ("pct_counts_mt", "Mito %"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()

    for i, (col, title) in enumerate(metrics):
        before = np.asarray(adata_before.obs[col], dtype=float)
        after = np.asarray(adata_after.obs[col], dtype=float)

        before = before[np.isfinite(before)]
        after = after[np.isfinite(after)]

        axes[i].hist(before, bins=100, alpha=0.7, label="before")
        axes[i].hist(after, bins=100, alpha=0.7, label="after")
        axes[i].set_title(title)
        axes[i].legend()

        axes[i + 3].hist(np.log1p(before), bins=100, alpha=0.7, label="before")
        axes[i + 3].hist(np.log1p(after), bins=100, alpha=0.7, label="after")
        axes[i + 3].set_title(f"{title} (log1p)")
        axes[i + 3].legend()

        if col == "pct_counts_mt":
            axes[i].set_xlim(0, 20)
            axes[i + 3].set_xlim(0, np.log1p(20))

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_spatial_compare(adata_before, adata_after, library_id, out_png):
    img_key_before = choose_img_key(adata_before, library_id)
    img_key_after = choose_img_key(adata_after, library_id)

    mt_vmax_before = get_mt_vmax_for_plot(adata_before)
    mt_vmax_after = get_mt_vmax_for_plot(adata_after)

    mt_candidates = [x for x in [mt_vmax_before, mt_vmax_after] if x is not None]
    mt_vmax_shared = max(mt_candidates) if len(mt_candidates) > 0 else None

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    sc.pl.spatial(
        adata_before, color="total_counts", library_id=library_id, img_key=img_key_before,
        size=1.1, ax=axes[0, 0], show=False
    )
    axes[0, 0].set_title("Before: total_counts")

    sc.pl.spatial(
        adata_before, color="n_genes_by_counts", library_id=library_id, img_key=img_key_before,
        size=1.1, ax=axes[0, 1], show=False
    )
    axes[0, 1].set_title("Before: n_genes_by_counts")

    sc.pl.spatial(
        adata_before, color="pct_counts_mt", library_id=library_id, img_key=img_key_before,
        size=1.1, vmin=0, vmax=mt_vmax_shared, ax=axes[0, 2], show=False
    )
    axes[0, 2].set_title("Before: pct_counts_mt")

    sc.pl.spatial(
        adata_after, color="total_counts", library_id=library_id, img_key=img_key_after,
        size=1.1, ax=axes[1, 0], show=False
    )
    axes[1, 0].set_title("After: total_counts")

    sc.pl.spatial(
        adata_after, color="n_genes_by_counts", library_id=library_id, img_key=img_key_after,
        size=1.1, ax=axes[1, 1], show=False
    )
    axes[1, 1].set_title("After: n_genes_by_counts")

    sc.pl.spatial(
        adata_after, color="pct_counts_mt", library_id=library_id, img_key=img_key_after,
        size=1.1, vmin=0, vmax=mt_vmax_shared, ax=axes[1, 2], show=False
    )
    axes[1, 2].set_title("After: pct_counts_mt")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_vendor_cluster_compare(adata_before, adata_after, library_id, out_png):
    img_key_before = choose_img_key(adata_before, library_id)
    img_key_after = choose_img_key(adata_after, library_id)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sc.pl.spatial(
        adata_before,
        color=CLUSTER_COL,
        library_id=library_id,
        img_key=img_key_before,
        size=1.1,
        ax=axes[0],
        show=False,
    )
    axes[0].set_title("Before filtering")

    sc.pl.spatial(
        adata_after,
        color=CLUSTER_COL,
        library_id=library_id,
        img_key=img_key_after,
        size=1.1,
        ax=axes[1],
        show=False,
    )
    axes[1].set_title("After filtering")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_filtering_mask_spatial(adata_with_mask, library_id, out_png):
    img_key = choose_img_key(adata_with_mask, library_id)

    fig = sc.pl.spatial(
        adata_with_mask,
        color=PASS_COL,
        library_id=library_id,
        img_key=img_key,
        size=1.1,
        show=False,
        return_fig=True,
    )
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================
# UMAP (visualization only)
# =========================
def prepare_umap_object(adata_filtered):
    """
    Build a copy for UMAP visualization only.
    No new clustering is performed.
    Vendor cluster labels are only used for coloring.
    """
    ad = adata_filtered.copy()

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    if USE_HVG_FOR_UMAP:
        sc.pp.highly_variable_genes(
            ad,
            flavor="seurat",
            n_top_genes=min(N_TOP_HVGS, ad.n_vars),
            inplace=True,
        )

    n_pcs_use = min(N_PCS, ad.n_obs - 1, ad.n_vars - 1)
    if n_pcs_use < 2:
        raise ValueError(
            f"Not enough dimensions for PCA/UMAP. "
            f"n_obs={ad.n_obs}, n_vars={ad.n_vars}, computed n_pcs={n_pcs_use}"
        )

    sc.tl.pca(
        ad,
        n_comps=n_pcs_use,
        svd_solver="arpack",
        use_highly_variable=USE_HVG_FOR_UMAP,
    )
    sc.pp.neighbors(ad, n_neighbors=N_NEIGHBORS, n_pcs=n_pcs_use)
    sc.tl.umap(ad, min_dist=UMAP_MIN_DIST, spread=UMAP_SPREAD, random_state=RANDOM_SEED)

    return ad


def save_umap_vendor_cluster(adata_umap, out_png):
    fig = sc.pl.umap(
        adata_umap,
        color=CLUSTER_COL,
        return_fig=True,
        show=False,
    )
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================
# SUMMARY
# =========================
def write_summary(
    adata_before,
    adata_with_mask,
    adata_after,
    filtering_impact_df,
    cluster_retention_df,
    out_txt,
):
    before_x = infer_x_status(adata_before)
    after_x = infer_x_status(adata_after)

    lines = [
        "=== 01 Postfilter (filtering only; no custom clustering) ===",
        f"Input h5ad: {H5AD}",
        f"Output directory: {OUTDIR}",
        "",
        "=== Thresholds ===",
        f"MIN_COUNTS: {MIN_COUNTS}",
        f"MIN_GENES: {MIN_GENES}",
        f"MAX_PCT_MT: {MAX_PCT_MT}",
        f"MIN_BINS_PER_GENE: {MIN_BINS_PER_GENE}",
        "",
        "=== Shape before filtering ===",
        f"Bins x Genes: {adata_before.n_obs} x {adata_before.n_vars}",
        "",
        "=== Shape after filtering ===",
        f"Bins x Genes: {adata_after.n_obs} x {adata_after.n_vars}",
        "",
        "=== Removed ===",
        f"Bins removed: {adata_before.n_obs - adata_after.n_obs}",
        f"Genes removed: {adata_before.n_vars - adata_after.n_vars}",
        "",
        "=== Vendor cluster status after filtering ===",
        f"Cluster column used: {CLUSTER_COL}",
        "Custom Leiden clustering: NOT performed",
        "",
        f"=== Filtering mask column stored in object ===",
        f"{PASS_COL}: added to obs",
        "",
        "=== UMAP settings (visualization only) ===",
        f"USE_HVG_FOR_UMAP: {USE_HVG_FOR_UMAP}",
        f"N_TOP_HVGS: {N_TOP_HVGS if USE_HVG_FOR_UMAP else 'NA'}",
        f"N_PCS: {N_PCS}",
        f"N_NEIGHBORS: {N_NEIGHBORS}",
        f"UMAP_MIN_DIST: {UMAP_MIN_DIST}",
        f"UMAP_SPREAD: {UMAP_SPREAD}",
        f"RANDOM_SEED: {RANDOM_SEED}",
        "",
        "=== Matrix status: before filtering ===",
        f"X sparse: {before_x['is_sparse']}",
        f"X format: {before_x['x_format']}",
        f"X dtype: {before_x['x_dtype']}",
        f"X min: {before_x['x_min']}",
        f"X max: {before_x['x_max']}",
        f"X has negative values: {before_x['x_has_negative']}",
        f"X has fractional values: {before_x['x_has_fractional']}",
        f"X looks integer-like: {before_x['looks_integer_like']}",
        f"Interpretation: {before_x['guess']}",
        "",
        "=== Matrix status: after filtering ===",
        f"X sparse: {after_x['is_sparse']}",
        f"X format: {after_x['x_format']}",
        f"X dtype: {after_x['x_dtype']}",
        f"X min: {after_x['x_min']}",
        f"X max: {after_x['x_max']}",
        f"X has negative values: {after_x['x_has_negative']}",
        f"X has fractional values: {after_x['x_has_fractional']}",
        f"X looks integer-like: {after_x['looks_integer_like']}",
        f"Interpretation: {after_x['guess']}",
        "Normalization status of saved filtered h5ad: not normalized/log1p by this script",
        "Scaling status of saved filtered h5ad: not scaled by this script",
        "",
        "=== AnnData structure: after filtering ===",
        f"obs columns ({adata_after.obs.shape[1]}): {short_list(adata_after.obs.columns.astype(str).tolist(), max_n=100)}",
        f"var columns ({adata_after.var.shape[1]}): {short_list(adata_after.var.columns.astype(str).tolist(), max_n=100)}",
        f"obsm keys ({len(adata_after.obsm.keys())}): {short_list(list(adata_after.obsm.keys()), max_n=100)}",
        f"varm keys ({len(adata_after.varm.keys())}): {short_list(list(adata_after.varm.keys()), max_n=100)}",
        f"layers keys ({len(adata_after.layers.keys())}): {short_list(list(adata_after.layers.keys()), max_n=100)}",
        f"obsp keys ({len(adata_after.obsp.keys())}): {short_list(list(adata_after.obsp.keys()), max_n=100)}",
        f"uns keys ({len(adata_after.uns.keys())}): {short_list(list(adata_after.uns.keys()), max_n=100)}",
        "",
        "=== Filtering impact ===",
    ]

    for _, row in filtering_impact_df.iterrows():
        lines.append(
            f"{row['level']} | {row['rule']} | "
            f"n_fail={int(row['n_fail'])} / {int(row['n_total'])} "
            f"({row['pct_fail']:.2f}%)"
        )

    lines.extend([
        "",
        "=== PASS/FAIL counts ===",
    ])

    pass_counts = adata_with_mask.obs[PASS_COL].value_counts(dropna=False)
    for label, n in pass_counts.items():
        lines.append(f"{label}: {n}")

    lines.extend([
        "",
        "=== Vendor cluster counts after filtering ===",
    ])

    counts_after = adata_after.obs[CLUSTER_COL].value_counts().sort_index()
    for cluster, n in counts_after.items():
        lines.append(f"cluster {cluster}: {n}")

    lines.extend([
        "",
        "=== Cluster retention after filtering ===",
    ])

    for _, row in cluster_retention_df.iterrows():
        retained_pct = row["retained_pct"]
        retained_str = f"{retained_pct:.2f}%" if pd.notna(retained_pct) else "NA"
        lines.append(
            f"cluster {row['cluster']}: "
            f"before={int(row['n_before'])}, "
            f"after={int(row['n_after'])}, "
            f"removed={int(row['n_removed'])}, "
            f"retained={retained_str}"
        )

    lines.extend([
        "",
        "=== Notes ===",
        "This step performs filtering only.",
        "Vendor clustering from Space Ranger is retained.",
        "UMAP is computed only for visualization of vendor clusters on a temporary copy.",
        "No custom Leiden clustering is performed in this step.",
        f"{PASS_COL} is stored in obs to track bin-level QC pass/fail status.",
        "The saved filtered h5ad is the raw/canonical filtered object.",
    ])

    out_txt.write_text("\n".join(lines))


# =========================
# MAIN
# =========================
def main():
    print("Reading h5ad...")
    adata = sc.read_h5ad(H5AD)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names_make_unique()

    if CLUSTER_COL not in adata.obs.columns:
        raise KeyError(f"{CLUSTER_COL} not found in adata.obs")

    library_id = get_library_id(adata, LIBRARY_ID)

    print("Preparing pre-filter reference object...")
    adata_before = adata.copy()

    print("Applying filtering...")
    adata_filtered = apply_filtering(adata)

    print("Building filtering impact table...")
    filtering_impact_df = build_filtering_impact_table(adata_before, adata_filtered)

    print("Building cluster retention table...")
    cluster_retention_df = build_cluster_retention_table(adata_before, adata_filtered, CLUSTER_COL)
    cluster_retention_df.to_csv(OUTDIR / "cluster_retention_after_filter.csv", index=False)

    print("Saving before/after histograms...")
    save_hist_compare(
        adata_before,
        adata_filtered,
        OUTDIR / "qc_before_after_hist.png",
    )

    print("Saving before/after spatial QC...")
    save_spatial_compare(
        adata_before,
        adata_filtered,
        library_id,
        OUTDIR / "qc_before_after_spatial.png",
    )

    print("Saving filtering mask spatial plot...")
    save_filtering_mask_spatial(
        adata,
        library_id,
        OUTDIR / "filtering_mask_spatial.png",
    )

    print("Saving vendor cluster comparison plot...")
    save_vendor_cluster_compare(
        adata_before,
        adata_filtered,
        library_id,
        OUTDIR / "vendor_cluster_spatial_before_after.png",
    )

    print("Saving filtered raw h5ad...")
    adata_filtered.write_h5ad(OUTDIR / "adata_filtered.h5ad", compression="gzip")

    print("Preparing UMAP object (visualization only)...")
    adata_umap = prepare_umap_object(adata_filtered)

    print("Saving UMAP colored by vendor cluster...")
    save_umap_vendor_cluster(
        adata_umap,
        OUTDIR / "umap_spaceranger_cluster.png",
    )

    print("Writing summary...")
    write_summary(
        adata_before=adata_before,
        adata_with_mask=adata,
        adata_after=adata_filtered,
        filtering_impact_df=filtering_impact_df,
        cluster_retention_df=cluster_retention_df,
        out_txt=OUTDIR / "filtering_summary.txt",
    )

    print(f"Done: {OUTDIR}")


if __name__ == "__main__":
    main()