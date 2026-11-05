#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import scanpy as sc


# =========================
# USER SETTINGS
# =========================

# Choose ONE input h5ad
ADATA_PATH = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/02_annotation/domain_labels/adata_labeled.h5ad"
)

# Sampling bin ID files
BIN_DIR = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/04_sampling/bin_ids"
)

BASE_OUTDIR = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/04_sampling/d0/sampled_adatas"
)
BASE_OUTDIR.mkdir(parents=True, exist_ok=True)

SAMPLING_FILES = {
    "cluster_x_neighbor": BIN_DIR / "cluster_x_neighbor_bin_ids.txt"
}


# =========================
# HELPERS
# =========================
def read_bin_ids(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Bin ID file not found: {path}")

    with path.open() as f:
        ids = [x.strip() for x in f if x.strip()]

    return ids


def standardize_adata(adata):
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def write_list(values, outpath: Path):
    with outpath.open("w") as f:
        for x in values:
            f.write(f"{x}\n")


def apply_sampling(adata, bin_file: Path, method_name: str):
    print(f"\nApplying sampling: {method_name}")

    requested_ids = read_bin_ids(bin_file)
    adata_ids = set(adata.obs_names.astype(str))

    matched_ids = [x for x in requested_ids if x in adata_ids]
    dropped_ids = [x for x in requested_ids if x not in adata_ids]

    print(f"requested bins: {len(requested_ids)}")
    print(f"matched bins: {len(matched_ids)}")
    print(f"dropped bins: {len(dropped_ids)}")

    if len(matched_ids) == 0:
        raise ValueError(f"No requested IDs matched AnnData for {method_name}")

    adata_sub = adata[matched_ids, :].copy()

    outdir = BASE_OUTDIR / method_name
    outdir.mkdir(parents=True, exist_ok=True)

    out_file = outdir / "sampled_adata.h5ad"
    adata_sub.write_h5ad(out_file, compression="gzip")

    write_list(matched_ids, outdir / "matched_bin_ids.txt")
    write_list(dropped_ids, outdir / "dropped_requested_ids.txt")

    with (outdir / "summary.txt").open("w") as f:
        f.write("SAMPLING SUMMARY\n")
        f.write("================\n\n")
        f.write(f"METHOD: {method_name}\n")
        f.write(f"INPUT_H5AD: {ADATA_PATH}\n")
        f.write(f"BIN_FILE: {bin_file}\n\n")
        f.write(f"INPUT_SHAPE: {adata.shape}\n")
        f.write(f"OUTPUT_SHAPE: {adata_sub.shape}\n")
        f.write(f"REQUESTED_BINS: {len(requested_ids)}\n")
        f.write(f"MATCHED_BINS: {len(matched_ids)}\n")
        f.write(f"DROPPED_BINS: {len(dropped_ids)}\n\n")
        f.write(f"OUTPUT: {out_file}\n")

    print(f"saved: {out_file}")

    return {
        "method": method_name,
        "bin_file": str(bin_file),
        "requested_bins": len(requested_ids),
        "matched_bins": len(matched_ids),
        "dropped_bins": len(dropped_ids),
        "input_n_obs": adata.n_obs,
        "input_n_vars": adata.n_vars,
        "output_n_obs": adata_sub.n_obs,
        "output_n_vars": adata_sub.n_vars,
        "output": str(out_file),
    }


# =========================
# MAIN
# =========================
def main():
    print("==============================")
    print("APPLY SAMPLING BIN IDS")
    print("==============================")

    print("\nLoading AnnData...")
    adata = standardize_adata(sc.read_h5ad(ADATA_PATH))

    print("Input shape:", adata.shape)

    rows = []

    print("\n===== SAMPLING CANDIDATES =====")

    for name, bin_file in SAMPLING_FILES.items():
        if not bin_file.exists():
            print(f"[WARN] Skipping missing sampling file: {bin_file}")
            continue

        row = apply_sampling(
            adata=adata,
            bin_file=bin_file,
            method_name=name,
        )
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        BASE_OUTDIR / "applied_sampling_summary.csv",
        index=False,
    )

    print("\n[DONE]")
    print(f"Saved sampled candidates under:\n{BASE_OUTDIR}")


if __name__ == "__main__":
    main()