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
INPUT_DIR = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/"
    "Visium_HD_mouse_brain/Visium_SpaceRanger_output_data/"
    "binned_outputs/square_008um"
)

COUNT_FILE = "filtered_feature_bc_matrix.h5"

CLUSTER_FILE = INPUT_DIR / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv"

TISSUE_POSITIONS_FILE = INPUT_DIR / "spatial" / "tissue_positions_list.csv"

OUTDIR = Path("./results/00_prefilter_qc")
OUTDIR.mkdir(parents=True, exist_ok=True)

LIBRARY_ID = None
CLUSTER_KEY = "spaceranger_cluster"

# UMAP settings (visualization only; no new clustering performed)
USE_HVG_FOR_UMAP = False
N_TOP_HVGS = 3000
N_PCS = 30
N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.5
UMAP_SPREAD = 1.0
RANDOM_SEED = 42


"""
00_prefilter_qc.py

Purpose
-------
Create a NEW backbone h5ad from Space Ranger output and perform pre-filter QC.

Backbone contents
-----------------
- filtered_feature_bc_matrix.h5
- spatial metadata loaded by sc.read_visium()
- vendor clustering attached from clusters.csv
- QC metrics added
- no user-defined filtering
- no custom clustering

Main outputs
------------
- adata_prefilter_cluster.h5ad
- qc_histograms_panel.png
- qc_spatial_overlay_panel.png
- vendor_cluster_spatial.png
- umap_prefilter_qc.png
- threshold_candidates_counts.csv
- summary_preQC.txt
"""


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


def add_mt_flag(adata):
    """
    Annotate mitochondrial genes in adata.var.
    """
    var_upper = pd.Index(adata.var_names.astype(str)).str.upper()
    adata.var["mt"] = (
        var_upper.str.startswith("MT-")
        | var_upper.str.startswith("MT.")
        | var_upper.str.startswith("MT_")
    )


def calculate_gene_detection(adata):
    """
    Count in how many bins each gene is detected (>0).
    """
    if sparse.issparse(adata.X):
        return np.asarray((adata.X > 0).sum(axis=0)).ravel()
    return np.asarray((adata.X > 0).sum(axis=0)).ravel()


def describe_with_quantiles(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return (
            "min=nan, max=nan, mean=nan, std=nan, "
            "Q01=nan, Q05=nan, Q10=nan, Q25=nan, "
            "Q50=nan, Q75=nan, Q90=nan, Q95=nan, Q99=nan"
        )

    q = np.percentile(vals, [1, 5, 10, 25, 50, 75, 90, 95, 99])

    return (
        f"min={np.min(vals):.4g}, max={np.max(vals):.4g}, "
        f"mean={np.mean(vals):.4g}, std={np.std(vals):.4g}, "
        f"Q01={q[0]:.4g}, Q05={q[1]:.4g}, Q10={q[2]:.4g}, "
        f"Q25={q[3]:.4g}, Q50={q[4]:.4g}, Q75={q[5]:.4g}, "
        f"Q90={q[6]:.4g}, Q95={q[7]:.4g}, Q99={q[8]:.4g}"
    )


def build_threshold_table(adata):
    """
    Build a candidate threshold table for later filtering.
    No filtering is applied here.
    """
    rules = {
        "total_counts": [200, 300, 500, 1000, 2000, 5000],
        "n_genes_by_counts": [100, 150, 200, 300, 500, 800],
        "pct_counts_mt": [5, 10, 15, 20, 25, 30],
    }

    records = []
    n_total = adata.n_obs

    for metric, cutoffs in rules.items():
        arr = np.asarray(adata.obs[metric].values, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)

        for cutoff in cutoffs:
            if metric == "pct_counts_mt":
                n_flagged = int(np.nansum(arr > cutoff))
                rule = f">{cutoff}"
            else:
                n_flagged = int(np.nansum(arr < cutoff))
                rule = f"<{cutoff}"

            records.append(
                {
                    "metric": metric,
                    "rule": rule,
                    "n_flagged": n_flagged,
                    "n_total": n_total,
                    "pct_flagged": n_flagged / n_total * 100,
                }
            )

    return pd.DataFrame(records)


def infer_x_status(adata):
    """
    Heuristic summary of whether X looks raw / normalized / log-transformed.
    """
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


def short_list(values, max_n=30):
    values = [str(v) for v in values]
    if len(values) <= max_n:
        return ", ".join(values) if values else "None"
    return ", ".join(values[:max_n]) + f", ... (+{len(values) - max_n} more)"


# =========================
# VENDOR CLUSTER ATTACHMENT
# =========================
def load_vendor_clusters(cluster_file: Path):
    df = pd.read_csv(cluster_file)

    expected = {"Barcode", "Cluster"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"{cluster_file} must contain columns {expected}. "
            f"Found: {list(df.columns)}"
        )

    df = df[["Barcode", "Cluster"]].copy()
    df["Barcode"] = df["Barcode"].astype(str)
    df["Cluster"] = df["Cluster"].astype(str)
    df = df.set_index("Barcode")
    return df


def attach_vendor_clusters(adata, cluster_df, cluster_key=CLUSTER_KEY):
    obs_names = pd.Index(adata.obs_names.astype(str))
    cluster_index = pd.Index(cluster_df.index.astype(str))

    missing_in_csv = obs_names.difference(cluster_index)
    extra_in_csv = cluster_index.difference(obs_names)
    overlap = obs_names.intersection(cluster_index)

    adata.obs[cluster_key] = pd.Series(index=obs_names, dtype="object")
    adata.obs.loc[overlap, cluster_key] = cluster_df.loc[overlap, "Cluster"].astype(str).values
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

    return missing_in_csv, extra_in_csv, overlap


# =========================
# QC PLOTS
# =========================
def qc_hist_panel(adata, out_png: Path):
    """
    Save a multi-panel histogram figure for:
    - total_counts
    - n_genes_by_counts
    - pct_counts_mt
    both raw scale and log1p scale.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    metrics = [
        ("total_counts", "Total counts"),
        ("n_genes_by_counts", "Detected genes"),
        ("pct_counts_mt", "Mito %"),
    ]

    for i, (col, title) in enumerate(metrics):
        vals = np.asarray(adata.obs[col].values, dtype=float)
        vals = vals[np.isfinite(vals)]

        axes[i].hist(vals, bins=100)
        axes[i].set_title(title)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Number of bins")

        axes[i + 3].hist(np.log1p(vals), bins=100)
        axes[i + 3].set_title(f"{title} (log1p)")
        axes[i + 3].set_xlabel(f"log1p({col})")
        axes[i + 3].set_ylabel("Number of bins")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_qc_spatial_panel(adata, library_id: str, out_png: Path):
    """
    Save spatial overlays for QC metrics.
    This does NOT apply filtering.
    """
    img_key = choose_img_key(adata, library_id)

    adata.obs["pct_counts_mt"] = pd.to_numeric(adata.obs["pct_counts_mt"], errors="coerce")
    adata.obs["pct_counts_mt"] = adata.obs["pct_counts_mt"].replace([np.inf, -np.inf], np.nan)

    mt_mask = (adata.obs["total_counts"] > 0) & np.isfinite(adata.obs["pct_counts_mt"])
    mt_vals = adata.obs.loc[mt_mask, "pct_counts_mt"].to_numpy()
    mt_vmax = np.percentile(mt_vals, 99) if len(mt_vals) > 0 else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sc.pl.spatial(
        adata,
        color="total_counts",
        library_id=library_id,
        img_key=img_key,
        size=1.1,
        ax=axes[0],
        show=False,
    )

    sc.pl.spatial(
        adata,
        color="n_genes_by_counts",
        library_id=library_id,
        img_key=img_key,
        size=1.1,
        ax=axes[1],
        show=False,
    )

    sc.pl.spatial(
        adata,
        color="pct_counts_mt",
        library_id=library_id,
        img_key=img_key,
        size=1.1,
        vmin=0,
        vmax=mt_vmax,
        ax=axes[2],
        show=False,
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_vendor_cluster_spatial(adata, library_id: str, out_png: Path):
    """
    Save spatial plot of vendor cluster labels.
    """
    img_key = choose_img_key(adata, library_id)

    fig = sc.pl.spatial(
        adata,
        color=CLUSTER_KEY,
        library_id=library_id,
        img_key=img_key,
        size=1.1,
        show=False,
        return_fig=True,
    )
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================
# UMAP ONLY
# =========================
def save_umap_prefilter(adata, out_png: Path):
    """
    Build an exploratory UMAP only.
    No clustering is performed.
    This is just for visual exploration of the prefilter object.
    """
    ad = adata.copy()

    # normalize/log1p only for embedding
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    if USE_HVG_FOR_UMAP:
        sc.pp.highly_variable_genes(
            ad,
            flavor="seurat",
            n_top_genes=min(N_TOP_HVGS, ad.n_vars),
            subset=True,
        )

    n_pcs_use = min(N_PCS, ad.n_obs - 1, ad.n_vars - 1)
    if n_pcs_use < 2:
        raise ValueError(
            f"Not enough dimensions for PCA/UMAP. "
            f"n_obs={ad.n_obs}, n_vars={ad.n_vars}, computed n_pcs={n_pcs_use}"
        )

    sc.pp.pca(ad, n_comps=n_pcs_use)
    sc.pp.neighbors(ad, n_neighbors=N_NEIGHBORS, n_pcs=n_pcs_use)
    sc.tl.umap(
        ad,
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        random_state=RANDOM_SEED,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sc.pl.umap(
        ad,
        color="total_counts",
        ax=axes[0],
        show=False,
        title="UMAP: total_counts",
    )
    sc.pl.umap(
        ad,
        color="n_genes_by_counts",
        ax=axes[1],
        show=False,
        title="UMAP: n_genes_by_counts",
    )
    sc.pl.umap(
        ad,
        color=CLUSTER_KEY,
        ax=axes[2],
        show=False,
        title=f"UMAP: {CLUSTER_KEY}",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================
# SUMMARY
# =========================
def save_summary_txt(adata, missing_in_csv, extra_in_csv, overlap, out_txt: Path):
    gene_det = calculate_gene_detection(adata)
    x_info = infer_x_status(adata)

    lines = []
    lines.append("=== Visium HD prefilter QC backbone (with vendor clustering attached) ===")
    lines.append(f"Input directory: {INPUT_DIR}")
    lines.append(f"Count file: {COUNT_FILE}")
    lines.append(f"Cluster file: {CLUSTER_FILE}")
    lines.append(f"Tissue positions file exists: {TISSUE_POSITIONS_FILE.exists()}")
    lines.append(f"Output directory: {OUTDIR}")
    lines.append("")

    lines.append("=== Backbone definition ===")
    lines.append("Backbone source: Space Ranger filtered matrix + spatial metadata")
    lines.append("Vendor clustering: attached in this step")
    lines.append("Custom clustering: NOT applied")
    lines.append("Filtering: NOT applied")
    lines.append("UMAP: exploratory only, generated on a temporary normalized/log1p copy")
    lines.append("Stored adata.X in saved backbone: original prefilter matrix after NA-cluster removal")
    lines.append("")

    lines.append("=== Data shape ===")
    lines.append(f"Bins x Genes: {adata.n_obs} x {adata.n_vars}")
    lines.append("")

    lines.append("=== Matrix / normalization status ===")
    lines.append(f"X sparse: {x_info['is_sparse']}")
    lines.append(f"X format: {x_info['x_format']}")
    lines.append(f"X dtype: {x_info['x_dtype']}")
    lines.append(f"X min: {x_info['x_min']}")
    lines.append(f"X max: {x_info['x_max']}")
    lines.append(f"X has negative values: {x_info['x_has_negative']}")
    lines.append(f"X has fractional values: {x_info['x_has_fractional']}")
    lines.append(f"X looks integer-like: {x_info['looks_integer_like']}")
    lines.append(f"Interpretation of stored X: {x_info['guess']}")
    lines.append("Normalization status of saved adata.X: not normalized/log1p by this script")
    lines.append("Scaling status of saved adata.X: not scaled by this script")
    lines.append("")

    lines.append("=== UMAP settings ===")
    lines.append(f"USE_HVG_FOR_UMAP: {USE_HVG_FOR_UMAP}")
    lines.append(f"N_TOP_HVGS: {N_TOP_HVGS}")
    lines.append(f"N_PCS: {N_PCS}")
    lines.append(f"N_NEIGHBORS: {N_NEIGHBORS}")
    lines.append(f"UMAP_MIN_DIST: {UMAP_MIN_DIST}")
    lines.append(f"UMAP_SPREAD: {UMAP_SPREAD}")
    lines.append(f"RANDOM_SEED: {RANDOM_SEED}")
    lines.append("")

    lines.append("=== AnnData structure ===")
    lines.append(f"obs columns ({adata.obs.shape[1]}): {short_list(adata.obs.columns.astype(str).tolist(), max_n=100)}")
    lines.append(f"var columns ({adata.var.shape[1]}): {short_list(adata.var.columns.astype(str).tolist(), max_n=100)}")
    lines.append(f"obsm keys ({len(adata.obsm.keys())}): {short_list(list(adata.obsm.keys()), max_n=100)}")
    lines.append(f"varm keys ({len(adata.varm.keys())}): {short_list(list(adata.varm.keys()), max_n=100)}")
    lines.append(f"layers keys ({len(adata.layers.keys())}): {short_list(list(adata.layers.keys()), max_n=100)}")
    lines.append(f"obsp keys ({len(adata.obsp.keys())}): {short_list(list(adata.obsp.keys()), max_n=100)}")
    lines.append(f"uns keys ({len(adata.uns.keys())}): {short_list(list(adata.uns.keys()), max_n=100)}")
    lines.append("")

    lines.append("=== Example obs/var names ===")
    lines.append(f"First 10 obs_names: {short_list(adata.obs_names[:10].tolist(), max_n=10)}")
    lines.append(f"First 10 var_names: {short_list(adata.var_names[:10].tolist(), max_n=10)}")
    lines.append("")

    lines.append("=== Tissue metadata ===")
    if "in_tissue" in adata.obs.columns:
        vals = adata.obs["in_tissue"].astype(int)
        lines.append(f"in_tissue==1: {(vals == 1).sum()} ({(vals == 1).mean() * 100:.2f}%)")
        lines.append(f"in_tissue==0: {(vals == 0).sum()} ({(vals == 0).mean() * 100:.2f}%)")
    else:
        lines.append("in_tissue column not found in adata.obs")

    lines.append("")
    lines.append("=== QC summaries (kept bins after removing NA vendor clusters, no user-defined filtering) ===")
    for col in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]:
        lines.append(f"{col}: {describe_with_quantiles(adata.obs[col].values)}")

    lines.append("")
    lines.append("=== Gene detection ===")
    lines.append(
        f"min={int(gene_det.min())}, median={int(np.median(gene_det))}, max={int(gene_det.max())}"
    )
    q = np.percentile(gene_det, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    lines.append(
        "gene_detection_quantiles: "
        f"Q01={q[0]:.4g}, Q05={q[1]:.4g}, Q10={q[2]:.4g}, "
        f"Q25={q[3]:.4g}, Q50={q[4]:.4g}, Q75={q[5]:.4g}, "
        f"Q90={q[6]:.4g}, Q95={q[7]:.4g}, Q99={q[8]:.4g}"
    )

    for k in [3, 10, 20, 50]:
        n_low = int((gene_det < k).sum())
        lines.append(f"genes expressed in < {k} bins: {n_low} ({n_low / adata.n_vars * 100:.2f}%)")

    lines.append("")
    lines.append("=== Vendor clustering mapping ===")
    lines.append(f"Cluster key stored in adata.obs: {CLUSTER_KEY}")
    lines.append(f"n overlap: {len(overlap)}")
    lines.append(f"n obs_names missing in cluster CSV: {len(missing_in_csv)}")
    lines.append(f"n extra cluster CSV entries not used by adata: {len(extra_in_csv)}")
    lines.append("")

    lines.append("Cluster counts:")
    vc = adata.obs[CLUSTER_KEY].value_counts(dropna=False)
    for cl, n in vc.items():
        lines.append(f"{cl}: {n}")

    if len(missing_in_csv) > 0:
        lines.append("")
        lines.append("First few obs_names missing in cluster CSV:")
        lines.extend([str(x) for x in missing_in_csv[:10]])

    if len(extra_in_csv) > 0:
        lines.append("")
        lines.append("First few cluster CSV barcodes not used by adata:")
        lines.extend([str(x) for x in extra_in_csv[:10]])

    lines.append("")
    lines.append("=== Notes ===")
    lines.append("This step creates the new backbone object.")
    lines.append("No user-defined filtering has been applied yet.")
    lines.append("Bins with NA vendor cluster labels were removed as a minimal intervention.")
    lines.append("Vendor clustering is already attached and will be subset together with later filtering.")
    lines.append("Spatial QC mt color scaling is stabilized only for visualization and does not alter the dataset.")
    lines.append("UMAP is included only as an exploratory embedding and does not imply clustering.")
    lines.append("UMAP was computed on a temporary copy after normalize_total + log1p + PCA + neighbors.")
    if USE_HVG_FOR_UMAP:
        lines.append("HVG selection was applied before PCA/UMAP on the temporary UMAP copy.")
    else:
        lines.append("HVG selection was NOT applied before PCA/UMAP on the temporary UMAP copy.")
    lines.append("The saved backbone object itself remains a prefilter object and does not store that normalized UMAP copy.")

    out_txt.write_text("\n".join(lines))


# =========================
# MAIN
# =========================
def main():
    print("Reading Visium data from Space Ranger output...")
    adata = sc.read_visium(path=INPUT_DIR, count_file=COUNT_FILE, load_images=True)

    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names_make_unique()

    print("Resolving library id...")
    library_id = get_library_id(adata, LIBRARY_ID)
    print(f"Using library_id: {library_id}")

    print("Annotating mitochondrial genes...")
    add_mt_flag(adata)

    print("Calculating QC metrics...")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    print("Loading vendor clusters...")
    cluster_df = load_vendor_clusters(CLUSTER_FILE)

    print("Attaching vendor clusters by barcode/bin ID...")
    missing_in_csv, extra_in_csv, overlap = attach_vendor_clusters(adata, cluster_df, CLUSTER_KEY)

    print("Removing bins with NA cluster labels...")
    before_n = adata.n_obs
    mask = adata.obs[CLUSTER_KEY].notna()

    adata = adata[mask].copy()
    adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].astype("category")

    after_n = adata.n_obs

    print(f"Removed {before_n - after_n} bins with NA clusters")
    print(f"Remaining bins: {after_n}")
    print(f"Overlap: {len(overlap)}")
    print(f"Missing in cluster CSV: {len(missing_in_csv)}")
    print(f"Extra in cluster CSV: {len(extra_in_csv)}")

    print("Saving backbone h5ad...")
    adata.write_h5ad(
        OUTDIR / "adata_prefilter_cluster.h5ad",
        compression="gzip"
    )

    print("Saving QC histogram panel...")
    qc_hist_panel(adata, OUTDIR / "qc_histograms_panel.png")

    print("Saving spatial QC panel...")
    save_qc_spatial_panel(adata, library_id, OUTDIR / "qc_spatial_overlay_panel.png")

    print("Saving vendor cluster spatial plot...")
    save_vendor_cluster_spatial(adata, library_id, OUTDIR / "vendor_cluster_spatial.png")

    print("Saving exploratory UMAP...")
    save_umap_prefilter(adata, OUTDIR / "umap_prefilter_qc.png")

    print("Saving threshold table...")
    build_threshold_table(adata).to_csv(OUTDIR / "threshold_candidates_counts.csv", index=False)

    print("Saving summary text...")
    save_summary_txt(
        adata=adata,
        missing_in_csv=missing_in_csv,
        extra_in_csv=extra_in_csv,
        overlap=overlap,
        out_txt=OUTDIR / "summary_preQC.txt",
    )

    print(f"Done: {OUTDIR}")


if __name__ == "__main__":
    main()