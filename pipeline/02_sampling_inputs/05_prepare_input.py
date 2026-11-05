#!/usr/bin/env python3

import gc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from pathlib import Path


# =========================
# USER SETTINGS
# =========================

RUN_NAME = "d0"

ADATA_PATH = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/04_sampling/d0/sampled_adatas/cluster_x_neighbor/sampled_adata.h5ad"
)

TF_LIST_PATH = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/external_data/allTFs_mm.txt"
)

BASE_OUTDIR = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/05_pyscenic_input"
)

# If True, export full spot x gene matrix as TSV.
# For 30k x 16k this can be very large, so keep False unless needed.
SAVE_FULL_TSV = True

# Minimum number of sampled spots/bins where a gene must be detected.

MIN_DETECTED_BINS = 10

RANDOM_SEED = 42
GENE_SELECTION = "ALL_DETECTED_GENES"


# =========================
# HELPER FUNCTIONS
# =========================

def read_tf_list(tf_path: Path) -> pd.Series:
    """
    Robust TF list reader.
    Handles whitespace / comments / messy formatting.
    """
    tf_list = []

    with tf_path.open("r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            tf = line.split()[0].strip()

            if tf:
                tf_list.append(tf)

    return pd.Series(tf_list, dtype=str).drop_duplicates()


def make_obs_names_unique(adata):
    if not adata.obs_names.is_unique:
        print("[INFO] obs_names not unique. Fixing...")
        adata.obs_names_make_unique()


def convert_sparse_to_dataframe(adata_sub):
    X = adata_sub.X

    if sparse.issparse(X):
        X = X.toarray()
    elif not isinstance(X, np.ndarray):
        X = np.asarray(X)

    expr_df = pd.DataFrame(
        X,
        index=adata_sub.obs_names,
        columns=adata_sub.var_names,
    )

    return expr_df


def infer_x_status(adata):
    """
    Simple check of whether X looks raw count-like or already normalized/log.
    """
    X = adata.X

    if sparse.issparse(X):
        data = X.data
        is_sparse = True
    else:
        data = np.asarray(X).ravel()
        is_sparse = False

    data = data[np.isfinite(data)]

    if data.size == 0:
        return {
            "is_sparse": is_sparse,
            "x_min": np.nan,
            "x_max": np.nan,
            "has_fractional": np.nan,
            "looks_count_like": False,
            "guess": "unknown",
        }

    has_fractional = bool(np.any(np.abs(data - np.round(data)) > 1e-8))
    has_negative = bool(np.any(data < 0))

    if has_negative:
        guess = "possibly scaled / centered"
        looks_count_like = False
    elif not has_fractional:
        guess = "likely raw count-like"
        looks_count_like = True
    elif np.max(data) < 30:
        guess = "possibly log-normalized"
        looks_count_like = False
    else:
        guess = "possibly normalized but not log-transformed"
        looks_count_like = False

    return {
        "is_sparse": is_sparse,
        "x_min": float(np.min(data)),
        "x_max": float(np.max(data)),
        "has_fractional": has_fractional,
        "looks_count_like": looks_count_like,
        "guess": guess,
    }


def detected_bins_per_gene(adata):
    """
    Number of sampled spots/bins where each gene has expression > 0.
    """
    X = adata.X
    if sparse.issparse(X):
        return np.asarray((X > 0).sum(axis=0)).ravel()
    return np.asarray((X > 0).sum(axis=0)).ravel()


# =========================
# MAIN
# =========================

def main():

    print("\n==============================")
    print("pySCENIC STEP1 INPUT PREP")
    print("GENE SELECTION: ALL DETECTED GENES")
    print("NO HVG / NO SVG FILTERING")
    print("==============================\n")

    print("[1/7] Checking input files...")

    if not ADATA_PATH.exists():
        raise FileNotFoundError(f"AnnData file not found:\n{ADATA_PATH}")

    if not TF_LIST_PATH.exists():
        raise FileNotFoundError(f"TF list file not found:\n{TF_LIST_PATH}")

    OUTDIR = BASE_OUTDIR / RUN_NAME
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] RUN_NAME: {RUN_NAME}")
    print(f"[INFO] Output directory: {OUTDIR}")

    # ------------------------
    # Load AnnData
    # ------------------------

    print("\n[2/7] Loading AnnData...")

    adata = sc.read_h5ad(str(ADATA_PATH))

    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)

    adata.var_names_make_unique()
    make_obs_names_unique(adata)

    print(f"[INFO] AnnData shape: {adata.shape}")

    x_status = infer_x_status(adata)
    print(f"[INFO] X status guess: {x_status['guess']}")

    # ------------------------
    # Read TF list
    # ------------------------

    print("\n[3/7] Reading TF list...")

    tf_series = read_tf_list(TF_LIST_PATH)
    tf_input = set(tf_series.tolist())

    print(f"[INFO] Input TF count: {len(tf_input)}")

    # ------------------------
    # All detected gene universe
    # ------------------------

    print("\n[4/7] Building all-detected gene universe...")
    print("[INFO] No normalization is performed.")
    print("[INFO] No HVG selection is performed.")
    print("[INFO] No SVG selection is performed.")

    detected = detected_bins_per_gene(adata)
    keep_mask = detected >= MIN_DETECTED_BINS

    gene_universe = adata.var_names[keep_mask].astype(str).tolist()
    gene_universe = sorted(set(gene_universe))

    genes_not_detected = adata.var_names[~keep_mask].astype(str).tolist()

    tf_present_in_adata = sorted(tf_input.intersection(set(adata.var_names)))
    tf_final = sorted(tf_input.intersection(set(gene_universe)))
    tf_present_but_not_detected = sorted(set(tf_present_in_adata) - set(tf_final))

    print(f"[INFO] Original gene count: {adata.n_vars:,}")
    print(f"[INFO] MIN_DETECTED_BINS: {MIN_DETECTED_BINS}")
    print(f"[INFO] Final gene universe size: {len(gene_universe):,}")
    print(f"[INFO] Genes removed by detection filter: {len(genes_not_detected):,}")
    print(f"[INFO] TFs present in adata: {len(tf_present_in_adata):,}")
    print(f"[INFO] Final retained TF count: {len(tf_final):,}")
    print(f"[INFO] TFs present but removed by detection filter: {len(tf_present_but_not_detected):,}")

    if len(gene_universe) == 0:
        raise ValueError("Final gene universe is empty.")

    if len(tf_final) == 0:
        raise ValueError(
            "No TFs retained after intersecting TF list with all-detected gene universe.\n"
            "Check gene symbol format or TF list file."
        )

    gene_universe_path = OUTDIR / "grn_gene_universe.txt"
    with gene_universe_path.open("w") as f:
        for gene in gene_universe:
            f.write(f"{gene}\n")

    detected_gene_path = OUTDIR / "selected_all_detected_genes.txt"
    with detected_gene_path.open("w") as f:
        for gene in gene_universe:
            f.write(f"{gene}\n")

    removed_gene_path = OUTDIR / "removed_low_detection_genes.txt"
    with removed_gene_path.open("w") as f:
        for gene in genes_not_detected:
            f.write(f"{gene}\n")

    # ------------------------
    # TF list after all-gene selection
    # ------------------------

    print("\n[5/7] Saving final TF list...")

    tf_final_path = OUTDIR / "grn_tf_list.txt"
    with tf_final_path.open("w") as f:
        for tf in tf_final:
            f.write(f"{tf}\n")

    # ------------------------
    # Save monitoring tables
    # ------------------------

    print("\n[6/7] Saving metadata and monitoring tables...")

    gene_selection_df = pd.DataFrame({
        "gene": adata.var_names.astype(str),
        "n_detected_bins": detected,
    })

    gene_selection_df["is_detected_gene"] = gene_selection_df["n_detected_bins"] >= MIN_DETECTED_BINS
    gene_selection_df["is_tf_input"] = gene_selection_df["gene"].isin(tf_input)
    gene_selection_df["is_in_final_gene_universe"] = gene_selection_df["gene"].isin(set(gene_universe))
    gene_selection_df["is_in_final_tf_list"] = gene_selection_df["gene"].isin(set(tf_final))

    gene_selection_df.to_csv(
        OUTDIR / "gene_selection_status.tsv",
        sep="\t",
        index=False,
    )

    # Keep sampled spot metadata for later AUCell/domain mapping.
    adata.obs.to_csv(
        OUTDIR / "spot_metadata.tsv",
        sep="\t",
    )

    pd.Series(adata.obs_names.astype(str)).to_csv(
        OUTDIR / "spots_used.txt",
        index=False,
        header=False,
    )

    metadata = {
        "run_name": RUN_NAME,
        "adata_path": str(ADATA_PATH),
        "tf_list_path": str(TF_LIST_PATH),

        "gene_selection": GENE_SELECTION,
        "min_detected_bins": int(MIN_DETECTED_BINS),
        "normalization_performed": False,
        "hvg_selection_performed": False,
        "svg_selection_performed": False,

        "n_bins": int(adata.n_obs),
        "n_genes_original": int(adata.n_vars),
        "gene_universe_size": int(len(gene_universe)),
        "n_genes_removed_by_detection_filter": int(len(genes_not_detected)),

        "tf_input_count": int(len(tf_input)),
        "tf_present_in_adata_count": int(len(tf_present_in_adata)),
        "tf_retained_count": int(len(tf_final)),
        "tf_present_but_removed_by_detection_filter": int(len(tf_present_but_not_detected)),

        "save_full_tsv": bool(SAVE_FULL_TSV),
        "random_seed": RANDOM_SEED,

        "x_status_guess": x_status["guess"],
        "x_is_sparse": x_status["is_sparse"],
        "x_min": x_status["x_min"],
        "x_max": x_status["x_max"],
        "x_has_fractional": x_status["has_fractional"],
    }

    metadata_path = OUTDIR / "step1_metadata.tsv"
    pd.DataFrame([metadata]).to_csv(metadata_path, sep="\t", index=False)

    # ------------------------
    # Optional expression export
    # ------------------------

    expr_tsv_path = OUTDIR / "grn_expression_matrix.tsv"

    if SAVE_FULL_TSV:
        print("\n[INFO] Exporting all-detected-gene expression matrix...")

        adata_sub = adata[:, gene_universe].copy()
        expr_df = convert_sparse_to_dataframe(adata_sub)
        expr_df.to_csv(expr_tsv_path, sep="\t")

        del expr_df
        del adata_sub
        gc.collect()

        print(f"[INFO] Expression matrix saved:\n{expr_tsv_path}")

    else:
        print("\n[INFO] SAVE_FULL_TSV = False → skipping matrix export")

    # ------------------------
    # Summary
    # ------------------------

    print("\n[7/7] Writing summary...")

    summary_path = OUTDIR / "step1_summary.txt"

    with summary_path.open("w") as f:
        f.write("pySCENIC STEP1 INPUT PREPARATION\n")
        f.write("GENE SELECTION: ALL DETECTED GENES\n")
        f.write("NO HVG / NO SVG FILTERING\n")
        f.write("================================\n\n")

        f.write(f"RUN_NAME: {RUN_NAME}\n")
        f.write(f"AnnData input: {ADATA_PATH}\n")
        f.write(f"TF list input: {TF_LIST_PATH}\n\n")

        f.write("Input AnnData:\n")
        f.write(f"  - n_bins: {adata.n_obs}\n")
        f.write(f"  - n_genes_original: {adata.n_vars}\n")
        f.write(f"  - X status guess: {x_status['guess']}\n\n")

        f.write("Gene selection:\n")
        f.write("  - Strategy: all detected genes in sampled dataset\n")
        f.write(f"  - MIN_DETECTED_BINS: {MIN_DETECTED_BINS}\n")
        f.write("  - HVG selection: not performed\n")
        f.write("  - SVG selection: not performed\n")
        f.write("  - Normalization/log1p: not performed by this script\n")
        f.write(f"  - final gene universe size: {len(gene_universe)}\n")
        f.write(f"  - genes removed by detection filter: {len(genes_not_detected)}\n\n")

        f.write("TF retention:\n")
        f.write(f"  - input TF count: {len(tf_input)}\n")
        f.write(f"  - TFs present in adata: {len(tf_present_in_adata)}\n")
        f.write(f"  - final retained TF count: {len(tf_final)}\n")
        f.write(f"  - TFs present but removed by detection filter: {len(tf_present_but_not_detected)}\n\n")

        f.write("Output files:\n")
        f.write(f"  - gene universe file: {gene_universe_path}\n")
        f.write(f"  - selected all-detected genes file: {detected_gene_path}\n")
        f.write(f"  - removed low-detection genes file: {removed_gene_path}\n")
        f.write(f"  - TF list file: {tf_final_path}\n")
        f.write(f"  - metadata file: {metadata_path}\n")
        f.write(f"  - gene selection status table: {OUTDIR / 'gene_selection_status.tsv'}\n")
        f.write(f"  - spot metadata table: {OUTDIR / 'spot_metadata.tsv'}\n")
        f.write(f"  - spots used file: {OUTDIR / 'spots_used.txt'}\n")

        f.write(f"\nExpression matrix exported: {SAVE_FULL_TSV}\n")
        if SAVE_FULL_TSV:
            f.write(f"Expression matrix path: {expr_tsv_path}\n")

        f.write("\nImportant note:\n")
        f.write(
            "This script prepares the sampled dataset for GRNBoost2/pySCENIC using all detected genes.\n"
            "It does not perform HVG selection, SVG selection, normalization, or log transformation.\n"
            "The exported gene universe and optional expression matrix use the original values in adata.X.\n"
        )

    print("\n[DONE] All-detected-gene pySCENIC input preparation complete")
    print(f"[DONE] Output directory: {OUTDIR}\n")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    main()
