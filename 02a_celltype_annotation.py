#!/usr/bin/env python3
"""
02a_celltype_annotation.py  (REVISED v2, minimal)

Purpose
-------
Attach MapMyCells (MMC) bin-level class predictions directly to each bin,
WITHOUT aggregating to spaceranger clusters first.

Protocol (MapMyCells paper, Daniel et al. 2026, bioRxiv, Sec 3.3 / Fig.3A / Fig.4B):
  - Map bin-by-gene data DIRECTLY (no pre-clustering).
  - Suppress noise with a CONFIDENCE (bootstrap probability) THRESHOLD.

This script:
  1. Loads MMC bin-level class_name + class_bootstrapping_probability.
  2. Sweeps confidence thresholds, reporting retained fraction (Fig.3A-style).
  3. Applies one chosen threshold -> mmc_class_qc ("Unresolved" below cut).
  4. Saves adata_annotated.h5ad with the new columns.

Changes vs. previous version:
  - Marker-gene DE/dotplot section REMOVED from this script. Marker validation
    is now done once, at the domain level, in 02c (after 02b assigns domains),
    instead of twice (once per class here, once per domain in 02c).

spaceranger_cluster is kept ONLY as an optional diagnostic crosstab.
It is NOT used to assign or aggregate class labels.

Note: this h5ad is an INTERMEDIATE file (still carries mmc_class_top1/
mmc_class_conf/mmc_class_qc, needed by 02b). The FINAL, downstream-facing
h5ad (produced by 02b) is trimmed down to only domain/neurotransmitter/
region_proxy columns.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


# =========================================================
# USER SETTINGS
# =========================================================
H5AD = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/01_postfilter/adata_filtered.h5ad"
)
MMC_CSV = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/external_data/"
    "adata_filtered_10xWholeMouseBrain(CCN20230722)_HierarchicalMapping_UTC_1774964556798.csv"
)

OUTDIR = Path("results/02_annotation/cell_type_annotation")
OUTDIR.mkdir(parents=True, exist_ok=True)

CLUSTER_KEY = "spaceranger_cluster"   # diagnostic only, not used for assignment
LIBRARY_ID = None

# Confidence threshold sweep range (Fig. 3A style)
SWEEP_THRESHOLDS = np.round(np.arange(0.0, 1.01, 0.05), 2)

# Chosen QC cut applied to mmc_class_qc.
# MapMyCells paper used 0.95 on single-cell-resolution spatial data (Fig.4B).
# Visium HD 8um bins carry less signal per bin than single cells, so check
# confidence_threshold_sweep.csv first -- lower this if 0.95 discards too much.
CLASS_CONF_CUT = 0.5

POINT_SIZE = 1.1
SPATIAL_DPI = 220

OUT_H5AD = OUTDIR / "adata_annotated.h5ad"


# =========================================================
# HELPERS
# =========================================================
def get_library_id(adata, user_library_id=None):
    if user_library_id is not None:
        return user_library_id
    libs = list(adata.uns.get("spatial", {}).keys())
    if not libs:
        raise ValueError("No spatial library found in adata.uns['spatial'].")
    return libs[0]


def choose_img_key(adata, library_id):
    imgs = adata.uns["spatial"][library_id]["images"]
    return "hires" if "hires" in imgs else list(imgs.keys())[0]


def save_spatial_plot(adata, library_id, color_key, out_png):
    if color_key not in adata.obs.columns:
        print(f"[skip] {color_key} not found")
        return
    if pd.api.types.is_categorical_dtype(adata.obs[color_key]):
        adata.obs[color_key] = adata.obs[color_key].cat.remove_unused_categories()
    img_key = choose_img_key(adata, library_id)
    fig = sc.pl.spatial(
        adata, color=color_key, library_id=library_id, img_key=img_key,
        size=POINT_SIZE, show=False, return_fig=True,
    )
    fig.savefig(out_png, dpi=SPATIAL_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_unresolved_mask(coords, qc_labels, out_png):
    """
    Single highlight plot: which bins failed the confidence QC cut
    ('Unresolved') vs which passed, shown together on one figure.
    """
    qc_labels = pd.Series(qc_labels).astype(str).values
    is_unresolved = qc_labels == "Unresolved"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(coords[~is_unresolved, 0], coords[~is_unresolved, 1],
               s=1, c="lightgray", linewidths=0, label="resolved")
    ax.scatter(coords[is_unresolved, 0], coords[is_unresolved, 1],
               s=1, c="crimson", linewidths=0, label="Unresolved (QC failed)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"QC-rejected bins (Unresolved): {is_unresolved.mean() * 100:.1f}%")
    ax.legend(markerscale=5, loc="upper right", fontsize=8)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def confidence_threshold_sweep(conf, thresholds):
    """Fig.3A-style sweep: fraction of bins retained at each threshold."""
    conf = np.asarray(conf, dtype=float)
    rows = []
    for t in thresholds:
        retained = conf >= t
        rows.append({
            "threshold": t,
            "n_retained": int(retained.sum()),
            "frac_retained": float(retained.mean()),
        })
    return pd.DataFrame(rows)


def plot_threshold_sweep(sweep_df, out_png):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(sweep_df["threshold"], sweep_df["frac_retained"], marker="o")
    ax.axvline(CLASS_CONF_CUT, color="red", linestyle="--", linewidth=1,
               label=f"chosen cut = {CLASS_CONF_CUT}")
    ax.set_xlabel("Bootstrap probability threshold")
    ax.set_ylabel("Fraction of bins retained")
    ax.set_title("Confidence threshold sweep (class level)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================
def main():
    print("Reading h5ad...")
    adata = sc.read_h5ad(H5AD)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names_make_unique()

    library_id = get_library_id(adata, LIBRARY_ID)

    print("Reading MapMyCells CSV...")
    mmc = pd.read_csv(MMC_CSV, comment="#")
    mmc.columns = [c.strip() for c in mmc.columns]

    required_cols = ["cell_id", "class_name", "class_bootstrapping_probability"]
    missing = [c for c in required_cols if c not in mmc.columns]
    if missing:
        raise ValueError(f"Missing MMC columns: {missing}")

    mmc = mmc.rename(columns={
        "cell_id": "obs_name",
        "class_name": "mmc_class_top1",
        "class_bootstrapping_probability": "mmc_class_conf",
    })
    mmc["obs_name"] = mmc["obs_name"].astype(str)
    mmc = mmc.set_index("obs_name").reindex(adata.obs_names)

    adata.obs["mmc_class_top1"] = mmc["mmc_class_top1"].values
    adata.obs["mmc_class_conf"] = mmc["mmc_class_conf"].values

    # -----------------------------------------------------
    # 1) Confidence threshold sweep (Fig.3A style)
    # -----------------------------------------------------
    print("Running confidence threshold sweep...")
    sweep_df = confidence_threshold_sweep(adata.obs["mmc_class_conf"], SWEEP_THRESHOLDS)
    sweep_df.to_csv(OUTDIR / "confidence_threshold_sweep.csv", index=False)
    plot_threshold_sweep(sweep_df, OUTDIR / "confidence_threshold_sweep.png")
    print(sweep_df.to_string(index=False))

    # -----------------------------------------------------
    # 2) Apply QC cut (Fig.4B style)
    # -----------------------------------------------------
    print(f"Applying QC cut at threshold={CLASS_CONF_CUT}...")
    passed = adata.obs["mmc_class_conf"] >= CLASS_CONF_CUT
    adata.obs["mmc_class_qc"] = np.where(
        passed, adata.obs["mmc_class_top1"].astype(str), "Unresolved"
    )
    adata.obs["mmc_class_qc"] = adata.obs["mmc_class_qc"].astype("category")
    print(f"Retained: {passed.mean() * 100:.1f}% of bins at threshold {CLASS_CONF_CUT}")

    # -----------------------------------------------------
    # 2b) Count + spatial summary of QC-rejected ("Unresolved") bins
    # -----------------------------------------------------
    n_unresolved = int((~passed).sum())
    pd.DataFrame({
        "status": ["resolved", "Unresolved"],
        "n_bins": [int(passed.sum()), n_unresolved],
        "fraction": [float(passed.mean()), float(1 - passed.mean())],
    }).to_csv(OUTDIR / "unresolved_bin_counts.csv", index=False)

    coords = np.asarray(adata.obsm["spatial"])
    plot_unresolved_mask(coords, adata.obs["mmc_class_qc"], OUTDIR / "unresolved_bins_spatial.png")
    print(f"Unresolved (QC-rejected): {n_unresolved} bins ({(1 - passed.mean()) * 100:.1f}%)")

    # -----------------------------------------------------
    # 3) Optional diagnostic crosstab (NOT used for assignment)
    # -----------------------------------------------------
    if CLUSTER_KEY in adata.obs.columns:
        pd.crosstab(adata.obs[CLUSTER_KEY], adata.obs["mmc_class_qc"]).to_csv(
            OUTDIR / "cluster_vs_qc_class_crosstab_sanitycheck.csv"
        )

    # -----------------------------------------------------
    # 4) Spatial plots: before vs after QC cut
    # -----------------------------------------------------
    print("Saving spatial plots...")
    save_spatial_plot(adata, library_id, "mmc_class_top1", OUTDIR / "mmc_class_top1_spatial.png")
    save_spatial_plot(adata, library_id, "mmc_class_qc", OUTDIR / "mmc_class_qc_spatial.png")

    # -----------------------------------------------------
    # 5) Summary + save
    # -----------------------------------------------------
    summary = [
        "=== 02a Cell-type annotation (bin-level, confidence-QC'd) ===",
        f"Input h5ad: {H5AD}",
        f"Input MMC CSV: {MMC_CSV}",
        "",
        "Protocol: bin-level direct mapping + bootstrap-probability QC cut,",
        "following MapMyCells paper Sec 3.3 / Fig.3A / Fig.4B.",
        f"{CLUSTER_KEY} is NOT used for assignment, only as a diagnostic crosstab.",
        "Marker-gene validation moved to 02c (domain level, after 02b).",
        "",
        f"QC threshold used: {CLASS_CONF_CUT}",
        f"Fraction retained: {passed.mean():.3f}",
        "",
        "Added obs columns (intermediate -- trimmed down later in 02b):",
        "  - mmc_class_top1   (raw MapMyCells class, no QC)",
        "  - mmc_class_conf   (bootstrap probability)",
        "  - mmc_class_qc     (QC'd class; 'Unresolved' below threshold)",
        "",
        "Output files:",
        "  - confidence_threshold_sweep.csv / .png",
        "  - cluster_vs_qc_class_crosstab_sanitycheck.csv",
        "  - mmc_class_top1_spatial.png / mmc_class_qc_spatial.png",
        "  - unresolved_bin_counts.csv / unresolved_bins_spatial.png",
        "  - adata_annotated.h5ad",
    ]
    (OUTDIR / "annotation_summary.txt").write_text("\n".join(summary))

    print("Saving final h5ad...")
    adata.write_h5ad(OUT_H5AD, compression="gzip")
    print(f"[DONE] {OUT_H5AD}")


if __name__ == "__main__":
    main()