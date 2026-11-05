#!/usr/bin/env python3
"""Run SpaGTL core inference using the high-resolution reconstruction workflow.

The target expression matrix is spatially reconstructed before SpaGTL fitting,
following the SpaGTL Slide-seqV2 tutorial: log1p transformation without
library-size normalization, followed by coordinate-based neighborhood
aggregation. The input is then restricted to detected genes in the pretrained
vocabulary, preserving the pretrained gene order.
"""

from pathlib import Path
import logging
import os
import random
import sys
import time

# Must be set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis")
ADATA_PATH = ROOT / "results/04_sampling/d0/sampled_adatas/cluster_x_neighbor/sampled_adata.h5ad"
PARAMS_PATH = ROOT / "external_data/params_dict.npy"
TF_LIST_PATH = ROOT / "external_data/allTFs_mm.txt"
# Keep this high-resolution workflow separate from the previously fitted
# non-reconstructed SpaGTL runs.
BASE_OUTDIR = ROOT / "results/08_spagtl_high_resolution"

RUN_IDS = range(1, 11)
BASE_SEED = 777
N_EPOCHS = 1000
N_HIDDEN = 128
N_LATENT = 10

# SpaGTL high-resolution tutorial settings.
SPATIAL_RECONSTRUCTION_ALPHA = 8
SPATIAL_RECONSTRUCTION_NEIGHBORS = 10
SPATIAL_RECONSTRUCTION_N_PCS = 15

SKIP_COMPLETED = True

import numpy as np
np.object = object  # Compatibility with older SpaGTL dependencies.

import pandas as pd
import scanpy as sc
from scipy import sparse
import torch
import SpaGTL


def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def read_tf_list(path: Path) -> list[str]:
    with path.open() as handle:
        return sorted(
            {
                line.split()[0]
                for line in handle
                if line.strip() and not line.startswith("#")
            }
        )


def detected_bins_per_gene(adata) -> np.ndarray:
    return np.asarray((adata.X > 0).sum(axis=0)).ravel()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_logger(run_id: int, outdir: Path) -> logging.Logger:
    logger = logging.getLogger(f"spagtl_high_resolution_run{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(outdir / f"08_spagtl_core_run{run_id}.log", mode="w"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def make_qk_edges(adata, tf_present: list[str], out_path: Path) -> pd.DataFrame:
    qk = adata.varp["QK"]
    qk = qk.toarray() if sparse.issparse(qk) else np.asarray(qk)
    tf_idx = np.flatnonzero(np.isin(adata.var_names.astype(str), tf_present))
    if tf_idx.size == 0:
        raise ValueError("No TF indices found for QK edge export.")

    edge_df = pd.DataFrame(
        qk[tf_idx, :],
        index=adata.var_names[tf_idx].astype(str),
        columns=adata.var_names.astype(str),
    ).melt(ignore_index=False)
    edge_df.columns = ["target", "importance"]
    edge_df = edge_df.reset_index().rename(columns={"index": "TF"})
    edge_df["importance"] = edge_df["importance"].astype(float)
    edge_df = edge_df.loc[
        (edge_df["importance"] > 0) & (edge_df["TF"] != edge_df["target"])
    ].sort_values("importance", ascending=False).reset_index(drop=True)
    edge_df.to_csv(out_path, sep="\t", index=False)
    return edge_df


def prepare_base_input():
    check_file(ADATA_PATH)
    check_file(PARAMS_PATH)
    check_file(TF_LIST_PATH)

    adata = sc.read_h5ad(ADATA_PATH)
    adata.obs_names_make_unique()
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required for spatial reconstruction.")

    original_shape = adata.shape
    params_dict = np.load(PARAMS_PATH, allow_pickle=True).item()
    pretrained_genes = list(map(str, params_dict["genes"]))

    # Exclude genes that are absent in every sampled bin before PCA/reconstruction.
    adata = adata[:, detected_bins_per_gene(adata) > 0].copy()

    # This operation stores raw counts, applies log1p without normalize_total,
    # and aggregates expression from coordinate-defined spatial neighbors.
    SpaGTL.spatial_reconstruction(
        adata,
        alpha=SPATIAL_RECONSTRUCTION_ALPHA,
        n_neighbors=SPATIAL_RECONSTRUCTION_NEIGHBORS,
        n_pcs=SPATIAL_RECONSTRUCTION_N_PCS,
        use_highly_variable=False,
        normalize_total=False,
    )

    available_genes = set(adata.var_names.astype(str))
    genes_used = [gene for gene in pretrained_genes if gene in available_genes]
    if not genes_used:
        raise ValueError("No detected genes overlap the SpaGTL pretrained vocabulary.")
    adata = adata[:, genes_used].copy()

    tf_present = [tf for tf in read_tf_list(TF_LIST_PATH) if tf in set(genes_used)]
    if not tf_present:
        raise ValueError("No TFs are present after pretrained-vocabulary matching.")

    adata.uns["spagtl_input_preprocessing"] = {
        "source": str(ADATA_PATH),
        "min_cells_filter": False,
        "hvg_selection": False,
        "all_zero_gene_removal": True,
        "spatial_reconstruction": True,
        "spatial_reconstruction_alpha": SPATIAL_RECONSTRUCTION_ALPHA,
        "spatial_reconstruction_neighbors": SPATIAL_RECONSTRUCTION_NEIGHBORS,
        "spatial_reconstruction_n_pcs": SPATIAL_RECONSTRUCTION_N_PCS,
        "normalize_total": False,
        "log1p": True,
        "pretrained_gene_matching": True,
    }
    return adata, params_dict, tf_present, original_shape, len(pretrained_genes)


def run_one(run_id: int, shared) -> None:
    adata_base, params_dict, tf_present, original_shape, n_pretrained = shared
    seed = BASE_SEED + run_id - 1
    outdir = BASE_OUTDIR / f"run{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    core_path = outdir / "adata_spagtl_core.h5ad"
    summary_path = outdir / "summary_spagtl_core.txt"
    if SKIP_COMPLETED and core_path.exists() and summary_path.exists():
        print(f"[SKIP] run{run_id}: existing core output found.", flush=True)
        return

    logger = make_logger(run_id, outdir)
    start = time.time()
    try:
        set_seed(seed)
        adata = adata_base.copy()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("SpaGTL high-resolution core run %d; seed=%d; device=%s", run_id, seed, device)
        logger.info("Input shape: %s -> %s; pretrained genes=%d; TFs=%d", original_shape, adata.shape, n_pretrained, len(tf_present))
        logger.info("Spatial reconstruction: alpha=%s, n_neighbors=%s, n_pcs=%s, normalize_total=False, use_highly_variable=False", SPATIAL_RECONSTRUCTION_ALPHA, SPATIAL_RECONSTRUCTION_NEIGHBORS, SPATIAL_RECONSTRUCTION_N_PCS)

        pd.Series(adata.var_names.astype(str)).to_csv(outdir / "spagtl_genes_used.txt", index=False, header=False)
        pd.Series(tf_present).to_csv(outdir / "spagtl_tf_present.txt", index=False, header=False)

        SpaGTL.run_SpaGTL(
            adata,
            n_epochs=N_EPOCHS,
            n_hidden=N_HIDDEN,
            n_latent=N_LATENT,
            params_dict=params_dict,
            batch_key=None,
            covar_key=None,
            device=device,
            copy=False,
        )
        for key, collection in (("qz", adata.obsm), ("x4", adata.layers), ("QK", adata.varp)):
            if key not in collection:
                raise RuntimeError(f"SpaGTL did not create {key}.")

        pd.DataFrame(adata.obsm["qz"], index=adata.obs_names).to_csv(outdir / "qz_latent.tsv", sep="\t")
        edge_df = make_qk_edges(adata, tf_present, outdir / "spagtl_QK_edges.tsv")
        adata.write_h5ad(core_path, compression="gzip")

        with summary_path.open("w") as handle:
            handle.write("SpaGTL high-resolution core summary\n\n")
            handle.write(f"Run ID: {run_id}\nSeed: {seed}\n")
            handle.write(f"Original shape: {original_shape}\nSpaGTL input shape: {adata.shape}\n")
            handle.write("HVG selection: False\nMin-cells filter: False\n")
            handle.write("Spatial reconstruction: True\n")
            handle.write(f"Spatial reconstruction alpha: {SPATIAL_RECONSTRUCTION_ALPHA}\n")
            handle.write(f"Spatial reconstruction neighbors: {SPATIAL_RECONSTRUCTION_NEIGHBORS}\n")
            handle.write("Library-size normalization: False\nLog1p: True\n")
            handle.write(f"QK edges: {len(edge_df)}\n")
            handle.write(f"Runtime hours: {(time.time() - start) / 3600:.3f}\n")
        logger.info("Completed run %d with %d QK edges.", run_id, len(edge_df))
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def main() -> None:
    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)
    shared = prepare_base_input()
    failures = []
    for run_id in RUN_IDS:
        try:
            run_one(run_id, shared)
        except Exception as exc:
            failures.append((run_id, repr(exc)))
            print(f"[ERROR] run{run_id} failed: {exc}", file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(f"SpaGTL core failed for runs: {failures}")


if __name__ == "__main__":
    main()
