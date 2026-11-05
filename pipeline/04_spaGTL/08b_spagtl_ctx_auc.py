#!/usr/bin/env python3
"""
Run pySCENIC CTX and/or AUCell for SpaGTL run1-run10.

Default execution mode
----------------------
RUN_CTX = True
RUN_AUCELL = True

Therefore, by default, this script:

    adata_spagtl_core.h5ad
    spagtl_regulons.pkl
        -> AUCell using adata.layers["x4"]
        -> spagtl_aucell.tsv
        -> adata_spagtl_core_regulons_aucell.h5ad

It does NOT rerun:
    modules_from_adjacencies
    prune2df
    df2regulons

Expression matrices
-------------------
CTX:
    adata.X

AUCell:
    adata.layers["x4"]

Expected existing inputs in each runN directory
-----------------------------------------------
- adata_spagtl_core.h5ad
- spagtl_QK_edges.tsv          required only when RUN_CTX=True
- spagtl_regulons.pkl          required when RUN_AUCELL=True and CTX is not run

CTX outputs
-----------
- spagtl_modules.pkl
- spagtl_motifs_prune2df.csv
- spagtl_regulons.csv
- spagtl_regulons.pkl
- summary_spagtl_ctx.txt

AUCell outputs
--------------
- spagtl_aucell.tsv
- adata_spagtl_core_regulons_aucell.h5ad
- summary_spagtl_aucell.txt
"""

from pathlib import Path
import glob
import logging
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from arboreto.utils import load_tf_names
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.aucell import aucell
from pyscenic.prune import df2regulons, prune2df
from pyscenic.utils import modules_from_adjacencies


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis"
)

BASE_OUTDIR = ROOT / "results/08_spagtl"

TF_LIST_PATH = ROOT / "external_data/allTFs_mm.txt"

MOTIF_ANNOTATION_PATH = (
    ROOT
    / "external_data/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl"
)

DATABASE_GLOB = str(
    ROOT
    / "external_data/mm10_*_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
)

RUN_IDS = range(1, 11)


# =============================================================================
# Execution mode
# =============================================================================

# False:
#   Do not rerun modules_from_adjacencies, prune2df, or df2regulons.
#
# True:

RUN_CTX = True

# False:
#   Do not calculate regulon activity.
#
# True:

RUN_AUCELL = True


# Skip CTX if all final CTX outputs already exist.
SKIP_COMPLETED_CTX = True

# Skip AUCell if all final AUCell outputs already exist.
SKIP_COMPLETED_AUCELL = True


# =============================================================================
# pySCENIC options
# =============================================================================

# Used only during modules_from_adjacencies.
RHO_MASK_DROP_OUTS = True

# Used only during prune2df.
NUM_WORKERS = 8

# Used only during AUCell.
AUCELL_NUM_WORKERS = 8


# =============================================================================
# Helpers
# =============================================================================

def check_file(path: Path) -> None:
    """Raise an informative error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def make_logger(run_id: int, outdir: Path) -> logging.Logger:
    """Create separate console and file logging for one run."""
    logger = logging.getLogger(f"spagtl_ctx_aucell_run{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    log_path = outdir / f"08b_spagtl_ctx_aucell_run{run_id}.log"

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush and close all handlers attached to a logger."""
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


def load_databases():
    """
    Load motif-ranking databases.

    This function is called only when RUN_CTX=True.
    """
    database_paths = sorted(glob.glob(DATABASE_GLOB))

    if not database_paths:
        raise FileNotFoundError(
            "No motif ranking databases matched:\n"
            f"  {DATABASE_GLOB}"
        )

    databases = [
        RankingDatabase(
            fname=database_path,
            name=Path(database_path).stem,
        )
        for database_path in database_paths
    ]

    return database_paths, databases


def matrix_to_dataframe(
    matrix,
    adata,
    source_name: str,
) -> pd.DataFrame:
    """
    Convert an AnnData matrix into a bins-by-genes DataFrame.

    Rows:
        observations / bins

    Columns:
        genes
    """
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError(
            f"{source_name} must be two-dimensional, "
            f"but its shape is {matrix.shape}."
        )

    if matrix.shape != adata.shape:
        raise ValueError(
            f"{source_name} shape {matrix.shape} does not match "
            f"AnnData shape {adata.shape}."
        )

    expression_df = pd.DataFrame(
        matrix,
        index=adata.obs_names.astype(str),
        columns=adata.var_names.astype(str),
    )

    return expression_df


def collapse_duplicate_gene_columns(
    expression_df: pd.DataFrame,
    source_name: str,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Collapse duplicated gene-symbol columns by summing their values.

    pySCENIC expects one expression column per gene symbol. SpaGTL core
    AnnData files can contain repeated gene symbols, so duplicate columns
    are combined before CTX or AUCell. Gene names are not renamed with
    suffixes because that would break matching to TF lists, ranking
    databases, and regulon targets.
    """
    expression_df.columns = expression_df.columns.astype(str)

    if expression_df.columns.is_unique:
        return expression_df

    duplicated_mask = expression_df.columns.duplicated(keep=False)
    duplicated_symbols = pd.Index(
        expression_df.columns[duplicated_mask]
    ).unique()

    message = (
        f"{source_name} contains {len(duplicated_symbols)} duplicated "
        "gene symbols. Collapsing duplicate columns by sum."
    )

    if logger is None:
        print(f"[WARNING] {message}", flush=True)
    else:
        logger.warning(message)

    expression_df = (
        expression_df.T
        .groupby(level=0, sort=False)
        .sum()
        .T
    )

    if not expression_df.columns.is_unique:
        raise RuntimeError(
            f"{source_name} still contains duplicated gene symbols "
            "after collapsing."
        )

    if logger is not None:
        logger.info(
            "%s shape after duplicate collapse: %s",
            source_name,
            expression_df.shape,
        )

    return expression_df


def load_ctx_expression_dataframe(adata, logger=None):
    """
    Load the SpaGTL reconstructed-expression matrix for CTX.

    """

    expression_df = matrix_to_dataframe(
        matrix=adata.X,
        adata=adata,
        source_name='adata.X',
    )

    expression_df = collapse_duplicate_gene_columns(
        expression_df=expression_df,
        source_name='adata.X',
        logger=logger,
    )

    return expression_df, 'adata.X'


def load_aucell_expression_dataframe(adata, logger=None):
    """
    Load the observed/preprocessed expression matrix for AUCell.

    AUCell uses:
        adata.layers["x4"]
    """
    expression_df = matrix_to_dataframe(
        matrix=adata.layers["x4"],
        adata=adata,
        source_name='adata.layers["x4"]',
    )

    expression_df = collapse_duplicate_gene_columns(
        expression_df=expression_df,
        source_name="adata.X",
        logger=logger,
    )

    return expression_df, 'adata.layers["x4"]'


def load_adjacencies(
    path: Path,
    gene_names,
) -> pd.DataFrame:
    """
    Read the QK edge table created by the SpaGTL core script.

    Required columns:
        TF
        target
        importance

    This function validates and cleans the existing edge file.
    It does not recreate QK edges.
    """
    adjacencies = pd.read_csv(path, sep="\t")

    required = {"TF", "target", "importance"}
    missing = required.difference(adjacencies.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}. "
            f"Observed columns: {list(adjacencies.columns)}"
        )

    adjacencies = adjacencies.loc[
        :,
        ["TF", "target", "importance"],
    ].copy()

    adjacencies["TF"] = adjacencies["TF"].astype(str)
    adjacencies["target"] = adjacencies["target"].astype(str)

    adjacencies["importance"] = pd.to_numeric(
        adjacencies["importance"],
        errors="coerce",
    )

    adjacencies = adjacencies.dropna(
        subset=["TF", "target", "importance"]
    )

    gene_set = set(map(str, gene_names))

    adjacencies = adjacencies.loc[
        adjacencies["TF"].isin(gene_set)
        & adjacencies["target"].isin(gene_set)
        & (adjacencies["TF"] != adjacencies["target"])
        & (adjacencies["importance"] > 0)
    ].copy()

    adjacencies = (
        adjacencies
        .sort_values("importance", ascending=False)
        .drop_duplicates(
            subset=["TF", "target"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    if adjacencies.empty:
        raise ValueError(
            f"No valid QK edges remain after checking {path}."
        )

    return adjacencies


def regulons_to_csv(
    regulons,
    out_path: Path,
) -> pd.DataFrame:
    """
    Save regulons as a long-format CSV.

    Columns:
        regulon
        TF
        target
        weight
    """
    rows = []

    for regulon in regulons:
        regulon_name = str(regulon.name)
        tf_name = regulon_name

        if tf_name.endswith("(+)") or tf_name.endswith("(-)"):
            tf_name = tf_name[:-3]

        for target, weight in regulon.gene2weight.items():
            rows.append(
                {
                    "regulon": regulon_name,
                    "TF": tf_name,
                    "target": str(target),
                    "weight": float(weight),
                }
            )

    regulon_df = pd.DataFrame(
        rows,
        columns=[
            "regulon",
            "TF",
            "target",
            "weight",
        ],
    )

    regulon_df.to_csv(
        out_path,
        index=False,
    )

    return regulon_df


def load_existing_regulons(
    regulons_pkl_path: Path,
):
    """Load previously generated pySCENIC regulon objects."""
    check_file(regulons_pkl_path)

    with regulons_pkl_path.open("rb") as handle:
        regulons = pickle.load(handle)

    regulons = list(regulons)

    if not regulons:
        raise RuntimeError(
            f"No regulons were found in {regulons_pkl_path}."
        )

    return regulons


# =============================================================================
# One run
# =============================================================================

def run_one(
    run_id: int,
    databases,
    database_paths,
    tf_names,
) -> None:
    outdir = BASE_OUTDIR / f"run{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Core inputs
    # -------------------------------------------------------------------------
    core_path = outdir / "adata_spagtl_core.h5ad"
    edge_path = outdir / "spagtl_QK_edges.tsv"

    # -------------------------------------------------------------------------
    # CTX outputs
    # -------------------------------------------------------------------------
    modules_path = outdir / "spagtl_modules.pkl"
    motifs_path = outdir / "spagtl_motifs_prune2df.csv"
    regulons_csv_path = outdir / "spagtl_regulons.csv"
    regulons_pkl_path = outdir / "spagtl_regulons.pkl"
    ctx_summary_path = outdir / "summary_spagtl_ctx.txt"

    # -------------------------------------------------------------------------
    # AUCell outputs
    # -------------------------------------------------------------------------
    auc_path = outdir / "spagtl_aucell.tsv"

    auc_h5ad_path = (
        outdir
        / "adata_spagtl_core_regulons_aucell.h5ad"
    )

    auc_summary_path = (
        outdir
        / "summary_spagtl_aucell.txt"
    )

    # -------------------------------------------------------------------------
    # Completion checks
    # -------------------------------------------------------------------------
    ctx_complete = (
        modules_path.exists()
        and motifs_path.exists()
        and regulons_csv_path.exists()
        and regulons_pkl_path.exists()
        and ctx_summary_path.exists()
    )

    aucell_complete = (
        auc_path.exists()
        and auc_h5ad_path.exists()
        and auc_summary_path.exists()
    )

    need_ctx = (
        RUN_CTX
        and not (
            SKIP_COMPLETED_CTX
            and ctx_complete
        )
    )

    need_aucell = (
        RUN_AUCELL
        and not (
            SKIP_COMPLETED_AUCELL
            and aucell_complete
        )
    )

    if not need_ctx and not need_aucell:
        print(
            f"[SKIP] run{run_id}: requested outputs already exist "
            "or both execution modes are disabled.",
            flush=True,
        )
        return

    logger = make_logger(
        run_id=run_id,
        outdir=outdir,
    )

    total_start = time.time()

    try:
        logger.info(
            "========== SpaGTL run%d START ==========",
            run_id,
        )

        logger.info("Output directory: %s", outdir)
        logger.info("RUN_CTX: %s", RUN_CTX)
        logger.info("RUN_AUCELL: %s", RUN_AUCELL)
        logger.info("Need CTX: %s", need_ctx)
        logger.info("Need AUCell: %s", need_aucell)

        # Required for both CTX and AUCell.
        check_file(core_path)

        logger.info("Reading AnnData: %s", core_path)

        adata = sc.read_h5ad(core_path)

        adata.obs_names = adata.obs_names.astype(str)
        adata.var_names = adata.var_names.astype(str)

        if not adata.obs_names.is_unique:
            raise ValueError(
                "adata.obs_names contains duplicated observation names."
            )

        if not adata.var_names.is_unique:
            n_duplicated_entries = int(
                adata.var_names.duplicated(keep=False).sum()
            )
            n_duplicated_symbols = int(
                pd.Index(
                    adata.var_names[
                        adata.var_names.duplicated(keep=False)
                    ]
                ).nunique()
            )
            logger.warning(
                "adata.var_names contains %d entries belonging to %d "
                "duplicated gene symbols. Expression columns will be "
                "collapsed by sum separately for CTX and AUCell.",
                n_duplicated_entries,
                n_duplicated_symbols,
            )

        logger.info("AnnData shape: %s", adata.shape)

        # Regulons can be created during CTX or loaded for AUCell.
        regulons = None

        # =====================================================================
        # CTX
        # =====================================================================
        if need_ctx:
            ctx_start = time.time()

            logger.info(
                "CTX enabled: using adata.layers['x4']."
            )

            check_file(edge_path)
            check_file(TF_LIST_PATH)
            check_file(MOTIF_ANNOTATION_PATH)

            ctx_expression_df, ctx_expression_source = (
                load_ctx_expression_dataframe(adata, logger=logger)
            )

            logger.info(
                "CTX expression source: %s",
                ctx_expression_source,
            )

            logger.info(
                "CTX expression shape: %s",
                ctx_expression_df.shape,
            )

            adjacencies = load_adjacencies(
                path=edge_path,
                gene_names=ctx_expression_df.columns,
            )

            logger.info(
                "Input QK edges: %d",
                len(adjacencies),
            )

            logger.info(
                "Input QK unique TFs: %d",
                adjacencies["TF"].nunique(),
            )

            logger.info(
                "Input QK unique targets: %d",
                adjacencies["target"].nunique(),
            )

            tf_present = sorted(
                set(map(str, tf_names)).intersection(
                    ctx_expression_df.columns
                )
            )

            if not tf_present:
                raise ValueError(
                    "No TFs from the TF list are present "
                    "in the CTX expression matrix."
                )

            logger.info(
                "TFs present in expression matrix: %d",
                len(tf_present),
            )

            logger.info(
                "Motif ranking databases: %d",
                len(database_paths),
            )

            # -----------------------------------------------------------------
            # 1. QK adjacencies -> co-expression modules
            # -----------------------------------------------------------------
            logger.info(
                "Building modules from QK adjacencies..."
            )

            modules = list(
                modules_from_adjacencies(
                    adjacencies=adjacencies,
                    ex_mtx=ctx_expression_df,
                    rho_mask_dropouts=RHO_MASK_DROP_OUTS,
                )
            )

            if not modules:
                raise RuntimeError(
                    "modules_from_adjacencies returned no modules."
                )

            with modules_path.open("wb") as handle:
                pickle.dump(
                    modules,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            logger.info(
                "Modules generated: %d",
                len(modules),
            )

            # -----------------------------------------------------------------
            # 2. Motif pruning
            # -----------------------------------------------------------------
            logger.info(
                "Running prune2df / CTX motif pruning..."
            )

            motifs_df = prune2df(
                rnkdbs=databases,
                modules=modules,
                motif_annotations_fname=str(
                    MOTIF_ANNOTATION_PATH
                ),
                num_workers=NUM_WORKERS,
            )

            motifs_df.to_csv(
                motifs_path,
                index=True,
            )

            logger.info(
                "Motif-pruning rows: %d",
                len(motifs_df),
            )

            # -----------------------------------------------------------------
            # 3. Motif pruning -> regulons
            # -----------------------------------------------------------------
            regulons = list(
                df2regulons(motifs_df)
            )

            if not regulons:
                raise RuntimeError(
                    "df2regulons returned no regulons."
                )

            with regulons_pkl_path.open("wb") as handle:
                pickle.dump(
                    regulons,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            regulon_df = regulons_to_csv(
                regulons=regulons,
                out_path=regulons_csv_path,
            )

            n_regulons = (
                regulon_df["regulon"].nunique()
            )

            n_regulon_edges = len(regulon_df)

            n_regulon_tfs = (
                regulon_df["TF"].nunique()
            )

            n_regulon_targets = (
                regulon_df["target"].nunique()
            )

            ctx_runtime_hours = (
                time.time() - ctx_start
            ) / 3600.0

            with ctx_summary_path.open("w") as handle:
                handle.write("SpaGTL CTX summary\n")
                handle.write("==================\n\n")

                handle.write(
                    f"Run ID: {run_id}\n"
                )
                handle.write(
                    f"Core input: {core_path}\n"
                )
                handle.write(
                    f"Edge input: {edge_path}\n"
                )
                handle.write(
                    f"Expression source: "
                    f"{ctx_expression_source}\n"
                )
                handle.write(
                    f"AnnData shape: {adata.shape}\n"
                )
                handle.write(
                    f"Input QK edges: "
                    f"{len(adjacencies)}\n"
                )
                handle.write(
                    f"Input QK unique TFs: "
                    f"{adjacencies['TF'].nunique()}\n"
                )
                handle.write(
                    f"Input QK unique targets: "
                    f"{adjacencies['target'].nunique()}\n"
                )
                handle.write(
                    f"Modules: {len(modules)}\n"
                )
                handle.write(
                    f"Motif-pruning rows: "
                    f"{len(motifs_df)}\n"
                )
                handle.write(
                    f"Regulons: {n_regulons}\n"
                )
                handle.write(
                    f"Regulon TF-target edges: "
                    f"{n_regulon_edges}\n"
                )
                handle.write(
                    f"Unique regulon TFs: "
                    f"{n_regulon_tfs}\n"
                )
                handle.write(
                    f"Unique regulon targets: "
                    f"{n_regulon_targets}\n"
                )
                handle.write(
                    f"NUM_WORKERS: {NUM_WORKERS}\n"
                )
                handle.write(
                    f"RHO_MASK_DROP_OUTS: "
                    f"{RHO_MASK_DROP_OUTS}\n"
                )
                handle.write(
                    f"Runtime hours: "
                    f"{ctx_runtime_hours:.3f}\n"
                )

            logger.info(
                "CTX regulons: %d",
                n_regulons,
            )

            logger.info(
                "CTX regulon TF-target edges: %d",
                n_regulon_edges,
            )

            logger.info(
                "CTX runtime hours: %.3f",
                ctx_runtime_hours,
            )

        elif RUN_CTX:
            logger.info(
                "CTX outputs already exist. "
                "CTX was skipped."
            )

        # =====================================================================
        # AUCell
        # =====================================================================
        if need_aucell:
            aucell_start = time.time()

            logger.info(
                'AUCell enabled: using adata.layers["x4"].'
            )

            auc_expression_df, auc_expression_source = (
                load_aucell_expression_dataframe(adata, logger=logger)
            )

            logger.info(
                "AUCell expression source: %s",
                auc_expression_source,
            )

            logger.info(
                "AUCell expression shape: %s",
                auc_expression_df.shape,
            )

            # If CTX was not run during this execution, load existing regulons.
            if regulons is None:
                logger.info(
                    "Loading existing regulons: %s",
                    regulons_pkl_path,
                )

                regulons = load_existing_regulons(
                    regulons_pkl_path
                )

            logger.info(
                "Regulon objects for AUCell: %d",
                len(regulons),
            )

            # Optional diagnostic: regulon sizes.
            regulon_sizes = np.array(
                [
                    len(regulon.gene2weight)
                    for regulon in regulons
                ],
                dtype=int,
            )

            logger.info(
                "Regulon size min/median/max: "
                "%d / %.1f / %d",
                int(regulon_sizes.min()),
                float(np.median(regulon_sizes)),
                int(regulon_sizes.max()),
            )

            logger.info("Running AUCell...")

            auc_mtx = aucell(
                auc_expression_df,
                regulons,
                num_workers=AUCELL_NUM_WORKERS,
            )

            if not isinstance(auc_mtx, pd.DataFrame):
                auc_mtx = pd.DataFrame(auc_mtx)

            auc_mtx.index = (
                auc_mtx.index.astype(str)
            )

            auc_mtx.columns = (
                auc_mtx.columns.astype(str)
            )

            # Validate observation correspondence.
            missing_obs = adata.obs_names.difference(
                auc_mtx.index
            )

            extra_obs = auc_mtx.index.difference(
                adata.obs_names
            )

            if len(missing_obs) > 0:
                raise ValueError(
                    "AUCell output is missing "
                    f"{len(missing_obs)} AnnData observations. "
                    f"Examples: {list(missing_obs[:5])}"
                )

            if len(extra_obs) > 0:
                logger.warning(
                    "AUCell output contains %d observations "
                    "that are absent from AnnData.",
                    len(extra_obs),
                )

            # Ensure exact row order correspondence with AnnData.
            auc_mtx = auc_mtx.reindex(
                adata.obs_names
            )

            if auc_mtx.isna().any().any():
                n_missing = int(
                    auc_mtx.isna().sum().sum()
                )

                raise ValueError(
                    "AUCell matrix contains "
                    f"{n_missing} missing values after "
                    "reindexing to AnnData observations."
                )

            logger.info(
                "AUCell matrix shape: %s",
                auc_mtx.shape,
            )

            # -----------------------------------------------------------------
            # Save standalone AUCell table
            # -----------------------------------------------------------------
            auc_mtx.to_csv(
                auc_path,
                sep="\t",
                index=True,
            )

            logger.info(
                "AUCell TSV saved: %s",
                auc_path,
            )

            # -----------------------------------------------------------------
            # Save regulons and AUCell matrix in a new AnnData object
            # -----------------------------------------------------------------
            adata.uns["regulons"] = [
                {
                    "name": str(regulon.name),
                    "gene2weight": {
                        str(gene): float(weight)
                        for gene, weight
                        in regulon.gene2weight.items()
                    },
                }
                for regulon in regulons
            ]

            adata.obsm["X_aucell"] = (
                auc_mtx.to_numpy(
                    dtype=np.float32
                )
            )

            adata.uns["aucell_regulon_names"] = [
                str(name)
                for name in auc_mtx.columns
            ]

            adata.uns["aucell_expression_source"] = (
                auc_expression_source
            )

            adata.uns["aucell_num_workers"] = (
                AUCELL_NUM_WORKERS
            )

            adata.write_h5ad(
                auc_h5ad_path,
                compression="gzip",
            )

            logger.info(
                "AUCell AnnData saved: %s",
                auc_h5ad_path,
            )

            aucell_runtime_hours = (
                time.time() - aucell_start
            ) / 3600.0

            total_runtime_hours = (
                time.time() - total_start
            ) / 3600.0

            # -----------------------------------------------------------------
            # Save AUCell summary
            # -----------------------------------------------------------------
            with auc_summary_path.open("w") as handle:
                handle.write("SpaGTL AUCell summary\n")
                handle.write("====================\n\n")

                handle.write(
                    f"Run ID: {run_id}\n"
                )
                handle.write(
                    f"Core input: {core_path}\n"
                )
                handle.write(
                    f"Regulon input: "
                    f"{regulons_pkl_path}\n"
                )
                handle.write(
                    f"Expression source: "
                    f"{auc_expression_source}\n"
                )
                handle.write(
                    f"AnnData shape: {adata.shape}\n"
                )
                handle.write(
                    f"Expression shape: "
                    f"{auc_expression_df.shape}\n"
                )
                handle.write(
                    f"Number of regulons: "
                    f"{len(regulons)}\n"
                )
                handle.write(
                    f"Regulon size min: "
                    f"{int(regulon_sizes.min())}\n"
                )
                handle.write(
                    f"Regulon size median: "
                    f"{float(np.median(regulon_sizes)):.1f}\n"
                )
                handle.write(
                    f"Regulon size max: "
                    f"{int(regulon_sizes.max())}\n"
                )
                handle.write(
                    f"AUCell matrix shape: "
                    f"{auc_mtx.shape}\n"
                )
                handle.write(
                    f"AUCELL_NUM_WORKERS: "
                    f"{AUCELL_NUM_WORKERS}\n"
                )
                handle.write(
                    f"AUCell runtime hours: "
                    f"{aucell_runtime_hours:.3f}\n"
                )
                handle.write(
                    f"Total runtime hours: "
                    f"{total_runtime_hours:.3f}\n"
                )
                handle.write(
                    f"AUCell TSV: {auc_path}\n"
                )
                handle.write(
                    f"AUCell AnnData: "
                    f"{auc_h5ad_path}\n"
                )

            logger.info(
                "AUCell runtime hours: %.3f",
                aucell_runtime_hours,
            )

        elif RUN_AUCELL:
            logger.info(
                "AUCell outputs already exist. "
                "AUCell was skipped."
            )

        total_runtime_hours = (
            time.time() - total_start
        ) / 3600.0

        logger.info(
            "Total run runtime hours: %.3f",
            total_runtime_hours,
        )

        logger.info(
            "========== SpaGTL run%d DONE ==========",
            run_id,
        )

    except Exception:
        logger.error(
            "run%d failed:\n%s",
            run_id,
            traceback.format_exc(),
        )
        raise

    finally:
        close_logger(logger)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    BASE_OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not RUN_CTX and not RUN_AUCELL:
        raise ValueError(
            "Both RUN_CTX and RUN_AUCELL are False. "
            "At least one execution mode must be enabled."
        )

    # CTX resources are loaded only when CTX is requested.
    if RUN_CTX:
        check_file(TF_LIST_PATH)
        check_file(MOTIF_ANNOTATION_PATH)

        database_paths, databases = (
            load_databases()
        )

        tf_names = load_tf_names(
            str(TF_LIST_PATH)
        )

    else:
        database_paths = []
        databases = []
        tf_names = []

    print(
        "SpaGTL CTX/AUCell run1-run10",
        flush=True,
    )

    print(
        f"RUN_CTX: {RUN_CTX}",
        flush=True,
    )

    print(
        f"RUN_AUCELL: {RUN_AUCELL}",
        flush=True,
    )

    print(
        f"Base output directory: {BASE_OUTDIR}",
        flush=True,
    )

    if RUN_CTX:
        print(
            f"CTX expression source: "
            'adata.layers["x4"]',
            flush=True,
        )

        print(
            f"TF list: {TF_LIST_PATH}",
            flush=True,
        )

        print(
            f"Motif annotation: "
            f"{MOTIF_ANNOTATION_PATH}",
            flush=True,
        )

        print(
            "Ranking databases:",
            flush=True,
        )

        for path in database_paths:
            print(
                f"  - {path}",
                flush=True,
            )

    else:
        print(
            "CTX disabled: TF list, motif annotation, "
            "and ranking databases will not be loaded.",
            flush=True,
        )

    if RUN_AUCELL:
        print(
            'AUCell expression source: adata.layers["x4"]',
            flush=True,
        )

    failures = []

    for run_id in RUN_IDS:
        try:
            run_one(
                run_id=run_id,
                databases=databases,
                database_paths=database_paths,
                tf_names=tf_names,
            )

        except Exception as exc:
            failures.append(
                (run_id, repr(exc))
            )

            print(
                f"[ERROR] run{run_id} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    if failures:
        print(
            "\nSome SpaGTL runs failed:",
            file=sys.stderr,
            flush=True,
        )

        for run_id, error in failures:
            print(
                f"  run{run_id}: {error}",
                file=sys.stderr,
                flush=True,
            )

        raise RuntimeError(
            f"{len(failures)} of "
            f"{len(list(RUN_IDS))} SpaGTL runs failed."
        )

    print(
        "\n[OK] Requested SpaGTL calculations completed "
        "for run1-run10.",
        flush=True,
    )


if __name__ == "__main__":
    main()
