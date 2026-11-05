#!/usr/bin/env python3

import logging
import time
from pathlib import Path

import numpy as np

# Compatibility for older pySCENIC/SpaGRN dependencies
np.object = object

from scipy import sparse
from spagrn.regulatory_network import InferNetwork as irn


# =========================================================
# USER SETTINGS
# =========================================================

PROJECT_ROOT = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis"
)

ADATA_PATH = (
    PROJECT_ROOT
    / "results/04_sampling/d0/sampled_adatas/"
      "cluster_x_neighbor/sampled_adata.h5ad"
)

TF_LIST_PATH = (
    PROJECT_ROOT
    / "external_data/allTFs_mm.txt"
)

DATABASES = str(
    PROJECT_ROOT
    / "external_data/"
      "mm10_*_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
)

MOTIF_ANNO_FN = (
    PROJECT_ROOT
    / "external_data/"
      "motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl"
)

OUTDIR = (
    PROJECT_ROOT
    / "results/07_spagrn"
)

# Previously generated SpaGRN-selected spatial genes
SELECTED_GENES_PATH = (
    OUTDIR
    / "tmp_files/selected_genes.txt"
)

N_WORKERS = 16
CLUSTER_LABEL = "spaceranger_cluster"


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def check_file(path: Path):
    """Check that a required file exists."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Path exists but is not a regular file: {path}"
        )


def check_database_pattern(pattern: str):
    """Check motif ranking database wildcard pattern."""
    import glob

    matches = sorted(glob.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No feather databases matched pattern: {pattern}"
        )

    logger.info("Ranking databases found:")

    for match in matches:
        logger.info(f"  - {match}")

    return matches


def remove_all_zero_genes(adata):
    """
    Remove only genes with zero expression across all sampled bins.

    This is not SVG/HVG filtering. It only prevents downstream
    problems caused by completely unexpressed genes.
    """
    X = adata.X

    if sparse.issparse(X):
        detected = np.asarray(
            (X > 0).sum(axis=0)
        ).ravel()
    else:
        detected = np.asarray(
            (X > 0).sum(axis=0)
        ).ravel()

    n_zero = int((detected == 0).sum())

    logger.info(
        f"All-zero genes in input: {n_zero}"
    )

    if n_zero > 0:
        adata = adata[:, detected > 0].copy()

        logger.info(
            f"Removed all-zero genes. New shape: {adata.shape}"
        )
    else:
        logger.info(
            "No all-zero genes detected."
        )

    return adata


def load_selected_genes(
    selected_genes_path: Path,
    available_genes
):
    """
    Load SpaGRN-selected genes from selected_genes.txt.

    The file is expected to contain one gene name per line.
    Empty lines and duplicate genes are removed.

    Only genes still present in the AnnData object are retained.
    """
    selected_genes_path = Path(selected_genes_path)

    check_file(selected_genes_path)

    logger.info(
        f"Loading selected genes from: {selected_genes_path}"
    )

    with selected_genes_path.open("r") as handle:
        genes_from_file = [
            line.strip()
            for line in handle
            if line.strip()
        ]

    if len(genes_from_file) == 0:
        raise ValueError(
            f"Selected-gene file is empty: {selected_genes_path}"
        )

    # Remove duplicates while preserving the original order
    genes_from_file = list(
        dict.fromkeys(genes_from_file)
    )

    available_gene_set = set(
        map(str, available_genes)
    )

    selected_genes = [
        gene
        for gene in genes_from_file
        if gene in available_gene_set
    ]

    missing_genes = [
        gene
        for gene in genes_from_file
        if gene not in available_gene_set
    ]

    logger.info(
        f"Genes read from selected_genes.txt: "
        f"{len(genes_from_file)}"
    )

    logger.info(
        f"Selected genes present in AnnData: "
        f"{len(selected_genes)}"
    )

    logger.info(
        f"Selected genes absent from AnnData: "
        f"{len(missing_genes)}"
    )

    if missing_genes:
        logger.warning(
            "Example selected genes absent from AnnData: "
            + ", ".join(missing_genes[:20])
        )

    if len(selected_genes) == 0:
        raise ValueError(
            "None of the genes in selected_genes.txt "
            "were found in adata.var_names."
        )

    return selected_genes


def inspect_existing_cache(outdir: Path):
    """
    Report existing SpaGRN intermediate files.

    Because cache=True is used, valid completed intermediates can
    be reused. This function only reports them; it does not delete
    or modify anything.
    """
    tmp_dir = Path(outdir) / "tmp_files"

    files_to_check = [
        tmp_dir / "more_stats.csv",
        tmp_dir / "selected_genes.txt",
        tmp_dir / "geary_adj.csv",
        tmp_dir / "modules.pkl",
        tmp_dir / "motifs.csv",
        tmp_dir / "regulons.json",
        tmp_dir / "auc_mtx.csv",
        tmp_dir / "regulon_specificity_scores.txt",
    ]

    logger.info("Existing SpaGRN temporary files:")

    for path in files_to_check:
        if path.exists():
            logger.info(
                f"  FOUND: {path} "
                f"({path.stat().st_size / 1024**2:.2f} MB)"
            )
        else:
            logger.info(
                f"  MISSING: {path}"
            )


# =========================================================
# MAIN
# =========================================================

def main():
    start = time.time()

    logger.info(
        "========== SpaGRN resume from selected genes: START =========="
    )

    # -----------------------------------------------------
    # 1. Check inputs
    # -----------------------------------------------------

    check_file(ADATA_PATH)
    check_file(TF_LIST_PATH)
    check_file(MOTIF_ANNO_FN)
    check_file(SELECTED_GENES_PATH)
    check_database_pattern(DATABASES)

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True
    )

    inspect_existing_cache(OUTDIR)

    # -----------------------------------------------------
    # 2. Load AnnData
    # -----------------------------------------------------

    logger.info("[1] Loading AnnData")

    adata = irn.read_file(
        str(ADATA_PATH)
    )

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    logger.info(
        f"Original AnnData shape: {adata.shape}"
    )

    if "spatial" not in adata.obsm:
        raise KeyError(
            "adata.obsm['spatial'] is missing."
        )

    if CLUSTER_LABEL not in adata.obs.columns:
        raise KeyError(
            f"adata.obs['{CLUSTER_LABEL}'] is missing."
        )

    # -----------------------------------------------------
    # 3. Remove all-zero genes
    # -----------------------------------------------------

    logger.info(
        "[2] Removing only all-zero genes"
    )

    adata = remove_all_zero_genes(adata)

    # -----------------------------------------------------
    # 4. Prepare raw_counts
    # -----------------------------------------------------

    logger.info(
        "[3] Preparing raw_counts layer"
    )

    if "raw_counts" not in adata.layers:
        logger.info(
            "raw_counts layer not found. "
            "Copying adata.X to adata.layers['raw_counts']"
        )

        adata.layers["raw_counts"] = adata.X.copy()
    else:
        logger.info(
            "Using existing adata.layers['raw_counts']"
        )

    # -----------------------------------------------------
    # 5. Load previously selected genes
    # -----------------------------------------------------

    logger.info(
        "[4] Loading previously selected spatial genes"
    )

    selected_genes = load_selected_genes(
        selected_genes_path=SELECTED_GENES_PATH,
        available_genes=adata.var_names
    )

    logger.info(
        "SpaGRN Module 1 spatial gene selection will be skipped."
    )

    logger.info(
        "The run will start from spatial GRN inference "
        "using the saved selected-gene list."
    )

    # -----------------------------------------------------
    # 6. Initialize SpaGRN
    # -----------------------------------------------------

    logger.info(
        "[5] Initializing InferNetwork"
    )

    grn = irn(
        adata,
        project_name="spagrn"
    )

    # -----------------------------------------------------
    # 7. Resume SpaGRN
    # -----------------------------------------------------

    logger.info(
        "[6] Running SpaGRN from saved selected genes"
    )

    grn.infer(
        databases=DATABASES,
        motif_anno_fn=str(MOTIF_ANNO_FN),
        tfs_fn=str(TF_LIST_PATH),

        # IMPORTANT:
        # Pass the loaded Python list, not the file path.
        # This skips spatial autocorrelation and gene selection.
        gene_list=selected_genes,

        # Disable optional ligand-receptor / niche module
        niche_df=None,

        # Input layer and metadata
        layers="raw_counts",
        latent_obsm_key="spatial",
        cluster_label=CLUSTER_LABEL,

        # These settings are still needed for the GRN step.
        # n_neighbors is used to reconstruct the spatial graph.
        model="bernoulli",
        n_neighbors=10,
        weighted_graph=False,

        # These gene-selection options are retained for consistency,
        # but they are not used when gene_list is supplied.
        local=True,
        methods=["FDR_I", "FDR_C", "FDR_G"],
        operation="intersection",
        combine=False,

        # Spatial TF-target inference method
        mode="geary",

        # Runtime/output
        num_workers=N_WORKERS,
        cache=True,
        save_tmp=True,
        output_dir=str(OUTDIR),
    )

    # -----------------------------------------------------
    # 8. Finish
    # -----------------------------------------------------

    runtime_hours = (
        time.time() - start
    ) / 3600

    logger.info(
        "========== SpaGRN DONE =========="
    )

    logger.info(
        f"Runtime hours: {runtime_hours:.2f}"
    )

    logger.info(
        f"Output directory: {OUTDIR}"
    )


if __name__ == "__main__":
    main()