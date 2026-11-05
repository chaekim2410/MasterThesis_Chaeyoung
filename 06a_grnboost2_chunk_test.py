#!/usr/bin/env python3

import gc
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from dask.distributed import Client, LocalCluster
from arboreto.core import create_graph, SGBM_KWARGS


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

OUTDIR = (
    PROJECT_ROOT
    / "results/06_grnboost2/validation"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

SEEDS = [777, 778, 779]

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
DASK_TEMP = os.environ.get("DASK_TEMPORARY_DIRECTORY", None)

PILOT_N_TARGETS = 300
CHUNK_SIZE = 100

EARLY_STOP_WINDOW_LENGTH = 25
REPARTITION_MULTIPLIER = 1

RTOL = 1e-10
ATOL = 1e-12

TOP_N_VALUES = (1000, 5000, 10000)
TOP_K_TFS_PER_TARGET = 10


# =========================================================
# HELPERS
# =========================================================

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def elapsed_hms(seconds):
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def check_file(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def load_tf_list(path):
    tfs = []

    with Path(path).open("r") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            tfs.append(line.split()[0])

    return sorted(set(tfs))


def write_list(values, path):
    with Path(path).open("w") as handle:
        for value in values:
            handle.write(f"{value}\n")


def to_dense_float32_numpy(adata):
    x = adata.X

    if sparse.issparse(x):
        return x.toarray().astype(np.float32)

    return np.asarray(x, dtype=np.float32)


def start_dask():
    cluster_kwargs = {
        "n_workers": NUM_WORKERS,
        "threads_per_worker": 1,
        "processes": True,
        "dashboard_address": None,
        "memory_limit": "auto",
    }

    if DASK_TEMP:
        cluster_kwargs["local_directory"] = DASK_TEMP

    cluster = LocalCluster(**cluster_kwargs)
    client = Client(cluster, timeout="120s")

    return cluster, client


def clean_network(network, allowed_targets):
    network = network.rename(
        columns={
            network.columns[0]: "TF",
            network.columns[1]: "target",
            network.columns[2]: "importance",
        }
    )

    network = network[
        ["TF", "target", "importance"]
    ].copy()

    network["TF"] = network["TF"].astype(str)
    network["target"] = network["target"].astype(str)
    network["importance"] = network[
        "importance"
    ].astype(np.float64)

    allowed_targets = set(allowed_targets)

    network = network[
        network["target"].isin(allowed_targets)
    ].copy()

    network = network[
        network["TF"] != network["target"]
    ].copy()

    duplicate_count = int(
        network.duplicated(
            ["TF", "target"]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate edges."
        )

    return network.reset_index(drop=True)


def run_graph(
    expression_matrix,
    gene_names,
    tf_present,
    target_genes,
    client,
    seed,
):
    graph = create_graph(
        expression_matrix=expression_matrix,
        gene_names=gene_names,
        tf_names=tf_present,
        regressor_type="GBM",
        regressor_kwargs=SGBM_KWARGS,
        client=client,
        target_genes=target_genes,
        limit=None,
        include_meta=False,
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        repartition_multiplier=REPARTITION_MULTIPLIER,
        seed=seed,
    )

    network = graph.compute()

    return clean_network(
        network=network,
        allowed_targets=target_genes,
    )


def standardize_for_comparison(network):
    return (
        network[
            ["TF", "target", "importance"]
        ]
        .sort_values(
            ["TF", "target"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def edge_key_set(network):
    return set(
        zip(
            network["TF"].astype(str),
            network["target"].astype(str),
        )
    )


def top_n_jaccard(full, chunked, n):
    n = min(n, len(full), len(chunked))

    full_top = edge_key_set(
        full.nlargest(n, "importance")
    )

    chunked_top = edge_key_set(
        chunked.nlargest(n, "importance")
    )

    union = full_top | chunked_top

    if not union:
        return np.nan

    return len(full_top & chunked_top) / len(union)


def mean_target_top_k_overlap(full, chunked, k):
    targets = sorted(
        set(full["target"])
        & set(chunked["target"])
    )

    overlap_fractions = []
    jaccards = []

    for target in targets:
        full_tfs = set(
            full.loc[
                full["target"] == target
            ]
            .nlargest(k, "importance")["TF"]
            .astype(str)
        )

        chunked_tfs = set(
            chunked.loc[
                chunked["target"] == target
            ]
            .nlargest(k, "importance")["TF"]
            .astype(str)
        )

        denominator = max(
            1,
            min(
                k,
                len(full_tfs),
                len(chunked_tfs),
            ),
        )

        overlap_fractions.append(
            len(full_tfs & chunked_tfs)
            / denominator
        )

        union = full_tfs | chunked_tfs

        jaccards.append(
            len(full_tfs & chunked_tfs)
            / len(union)
            if union
            else np.nan
        )

    return (
        float(np.nanmean(overlap_fractions)),
        float(np.nanmean(jaccards)),
    )


def compare_networks(
    full_network,
    chunked_network,
    seed_dir,
):
    full_sorted = standardize_for_comparison(
        full_network
    )

    chunked_sorted = standardize_for_comparison(
        chunked_network
    )

    same_edge_count = (
        len(full_sorted)
        == len(chunked_sorted)
    )

    same_edge_keys = full_sorted[
        ["TF", "target"]
    ].equals(
        chunked_sorted[
            ["TF", "target"]
        ]
    )

    if not same_edge_keys:
        key_comparison = (
            full_sorted[
                ["TF", "target"]
            ]
            .merge(
                chunked_sorted[
                    ["TF", "target"]
                ],
                on=["TF", "target"],
                how="outer",
                indicator=True,
            )
        )

        key_comparison.to_csv(
            seed_dir
            / "edge_key_mismatches.csv",
            index=False,
        )

        raise AssertionError(
            "FULL and CHUNKED edge keys differ."
        )

    full_values = full_sorted[
        "importance"
    ].to_numpy(dtype=np.float64)

    chunked_values = chunked_sorted[
        "importance"
    ].to_numpy(dtype=np.float64)

    exact_equal = bool(
        np.array_equal(
            full_values,
            chunked_values,
            equal_nan=True,
        )
    )

    same_nonfinite_pattern = bool(
        np.array_equal(
            np.isnan(full_values),
            np.isnan(chunked_values),
        )
        and np.array_equal(
            np.isposinf(full_values),
            np.isposinf(chunked_values),
        )
        and np.array_equal(
            np.isneginf(full_values),
            np.isneginf(chunked_values),
        )
    )

    finite_mask = (
        np.isfinite(full_values)
        & np.isfinite(chunked_values)
    )

    differences = np.abs(
        full_values[finite_mask]
        - chunked_values[finite_mask]
    )

    max_abs_difference = (
        float(differences.max())
        if differences.size
        else np.nan
    )

    mean_abs_difference = (
        float(differences.mean())
        if differences.size
        else np.nan
    )

    median_abs_difference = (
        float(np.median(differences))
        if differences.size
        else np.nan
    )

    allclose = bool(
        np.allclose(
            full_values[finite_mask],
            chunked_values[finite_mask],
            rtol=RTOL,
            atol=ATOL,
        )
    ) if finite_mask.any() else False

    pearson = float(
        np.corrcoef(
            full_values[finite_mask],
            chunked_values[finite_mask],
        )[0, 1]
    )

    spearman = float(
        pd.Series(
            full_values[finite_mask]
        ).corr(
            pd.Series(
                chunked_values[finite_mask]
            ),
            method="spearman",
        )
    )

    comparison = full_sorted.rename(
        columns={
            "importance":
            "importance_full"
        }
    ).copy()

    comparison[
        "importance_chunked"
    ] = chunked_sorted["importance"]

    comparison[
        "absolute_difference"
    ] = np.abs(
        comparison["importance_full"]
        - comparison["importance_chunked"]
    )

    comparison.to_csv(
        seed_dir
        / "full_vs_chunked_edge_comparison.tsv",
        sep="\t",
        index=False,
    )

    top_n_results = {
        f"top_{n}_jaccard":
        top_n_jaccard(
            full_network,
            chunked_network,
            n,
        )
        for n in TOP_N_VALUES
    }

    (
        mean_target_top_k_overlap_fraction,
        mean_target_top_k_jaccard,
    ) = mean_target_top_k_overlap(
        full_network,
        chunked_network,
        TOP_K_TFS_PER_TARGET,
    )

    validation_passed = bool(
        same_edge_count
        and same_edge_keys
        and same_nonfinite_pattern
        and pearson >= 0.999
        and spearman >= 0.999
        and
        mean_target_top_k_overlap_fraction
        >= 0.99
    )

    return {
        "validation_passed":
            validation_passed,
        "same_edge_count":
            same_edge_count,
        "full_edge_count":
            len(full_sorted),
        "chunked_edge_count":
            len(chunked_sorted),
        "same_edge_keys":
            same_edge_keys,
        "exact_equal_with_nan":
            exact_equal,
        "same_nonfinite_pattern":
            same_nonfinite_pattern,
        "allclose":
            allclose,
        "rtol":
            RTOL,
        "atol":
            ATOL,
        "max_abs_difference":
            max_abs_difference,
        "mean_abs_difference":
            mean_abs_difference,
        "median_abs_difference":
            median_abs_difference,
        "pearson":
            pearson,
        "spearman":
            spearman,
        "target_top_k":
            TOP_K_TFS_PER_TARGET,
        "mean_target_top_k_overlap_fraction":
            mean_target_top_k_overlap_fraction,
        "mean_target_top_k_jaccard":
            mean_target_top_k_jaccard,
        **top_n_results,
    }


def save_seed_summary(seed_dir, summary):
    summary_path = (
        seed_dir
        / "grnboost2_chunk_validation_summary.txt"
    )

    with summary_path.open("w") as handle:
        handle.write(
            "GRNBOOST2 TARGET-CHUNK VALIDATION\n"
        )
        handle.write(
            "================================\n\n"
        )

        for key, value in summary.items():
            handle.write(
                f"{key}: {value}\n"
            )


# =========================================================
# MAIN
# =========================================================

def main():
    total_start = time.time()
    script_start = now()

    print("\n====================================")
    print("GRNBOOST2 MULTI-SEED CHUNK VALIDATION")
    print("SEEDS: 777, 778, 779")
    print("300 TARGETS VS 100 + 100 + 100")
    print("====================================\n")

    check_file(ADATA_PATH)
    check_file(TF_LIST_PATH)

    # -----------------------------------------------------
    # 1. Load data once
    # -----------------------------------------------------

    print("[1] Loading AnnData...")

    adata = sc.read_h5ad(
        ADATA_PATH
    )

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    gene_names = (
        adata.var_names.astype(str).tolist()
    )

    gene_set = set(gene_names)

    print(
        f"[INFO] AnnData shape: "
        f"{adata.shape}"
    )

    # -----------------------------------------------------
    # 2. Prepare TF list once
    # -----------------------------------------------------

    print("[2] Loading TF list...")

    tf_list = load_tf_list(
        TF_LIST_PATH
    )

    tf_present = [
        tf
        for tf in tf_list
        if tf in gene_set
    ]

    if not tf_present:
        raise ValueError(
            "No TFs from the TF list "
            "are present in AnnData."
        )

    print(
        f"[INFO] TFs present: "
        f"{len(tf_present)}"
    )

    # -----------------------------------------------------
    # 3. Select one common pilot target panel
    # -----------------------------------------------------

    print(
        "[3] Selecting the common "
        "300-target pilot panel..."
    )

    if len(gene_names) < PILOT_N_TARGETS:
        raise ValueError(
            "Not enough genes for the pilot."
        )

    pilot_targets = (
        gene_names[:PILOT_N_TARGETS]
    )

    target_chunks = [
        pilot_targets[
            start:start + CHUNK_SIZE
        ]
        for start in range(
            0,
            PILOT_N_TARGETS,
            CHUNK_SIZE,
        )
    ]

    if [
        len(chunk)
        for chunk in target_chunks
    ] != [100, 100, 100]:
        raise AssertionError(
            "Expected three chunks of 100."
        )

    flattened = [
        gene
        for chunk in target_chunks
        for gene in chunk
    ]

    if flattened != pilot_targets:
        raise AssertionError(
            "Chunks do not reconstruct "
            "the pilot target list."
        )

    write_list(
        pilot_targets,
        OUTDIR / "pilot_300_targets.txt",
    )

    write_list(
        tf_present,
        OUTDIR / "tf_used_validation.txt",
    )

    for chunk_id, chunk in enumerate(
        target_chunks
    ):
        write_list(
            chunk,
            OUTDIR
            / (
                f"pilot_chunk_"
                f"{chunk_id}_100_targets.txt"
            ),
        )

    # -----------------------------------------------------
    # 4. Convert expression matrix once
    # -----------------------------------------------------

    print(
        "[4] Converting to dense "
        "float32 matrix..."
    )

    expression_matrix = (
        to_dense_float32_numpy(adata)
    )

    print(
        f"[INFO] Matrix shape: "
        f"{expression_matrix.shape}"
    )

    del adata
    gc.collect()

    all_seed_summaries = []

    # -----------------------------------------------------
    # 5. Validate each seed
    # -----------------------------------------------------

    for seed in SEEDS:
        seed_start = time.time()

        seed_dir = (
            OUTDIR / f"seed{seed}"
        )

        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("\n------------------------------------")
        print(f"SEED {seed}")
        print("------------------------------------")

        cluster, client = start_dask()

        try:
            # FULL 300
            print(
                f"[SEED {seed}] "
                "Running FULL 300 targets..."
            )

            t0 = time.time()

            full_network = run_graph(
                expression_matrix=
                    expression_matrix,
                gene_names=gene_names,
                tf_present=tf_present,
                target_genes=
                    pilot_targets,
                client=client,
                seed=seed,
            )

            full_runtime = (
                time.time() - t0
            )

            full_network.to_csv(
                seed_dir
                / "grnboost2_full_300_targets.tsv",
                sep="\t",
                index=False,
            )

            # CHUNKED 100+100+100
            chunk_results = []
            chunk_runtimes = []

            for chunk_id, chunk_targets in enumerate(
                target_chunks
            ):
                print(
                    f"[SEED {seed}] "
                    f"Running chunk {chunk_id}..."
                )

                t0 = time.time()

                chunk_network = run_graph(
                    expression_matrix=
                        expression_matrix,
                    gene_names=gene_names,
                    tf_present=tf_present,
                    target_genes=
                        chunk_targets,
                    client=client,
                    seed=seed,
                )

                chunk_runtimes.append(
                    time.time() - t0
                )

                chunk_results.append(
                    chunk_network
                )

                chunk_network.to_csv(
                    seed_dir
                    / (
                        f"grnboost2_chunk_"
                        f"{chunk_id}_100_targets.tsv"
                    ),
                    sep="\t",
                    index=False,
                )

            chunked_network = pd.concat(
                chunk_results,
                ignore_index=True,
            )

            chunked_network.to_csv(
                seed_dir
                / (
                    "grnboost2_chunked_"
                    "300_targets_merged.tsv"
                ),
                sep="\t",
                index=False,
            )

        finally:
            client.close()
            cluster.close()

        # Compare
        result = compare_networks(
            full_network=
                full_network,
            chunked_network=
                chunked_network,
            seed_dir=
                seed_dir,
        )

        result.update({
            "seed":
                seed,
            "n_tfs":
                len(tf_present),
            "n_targets":
                PILOT_N_TARGETS,
            "chunk_sizes":
                [100, 100, 100],
            "full_runtime":
                elapsed_hms(
                    full_runtime
                ),
            "chunk_0_runtime":
                elapsed_hms(
                    chunk_runtimes[0]
                ),
            "chunk_1_runtime":
                elapsed_hms(
                    chunk_runtimes[1]
                ),
            "chunk_2_runtime":
                elapsed_hms(
                    chunk_runtimes[2]
                ),
            "chunked_total_runtime":
                elapsed_hms(
                    sum(chunk_runtimes)
                ),
            "seed_total_runtime":
                elapsed_hms(
                    time.time() - seed_start
                ),
        })

        save_seed_summary(
            seed_dir,
            result,
        )

        all_seed_summaries.append(
            result
        )

        print(
            f"[SEED {seed}] "
            f"Pearson: "
            f"{result['pearson']:.12f}"
        )

        print(
            f"[SEED {seed}] "
            f"Spearman: "
            f"{result['spearman']:.12f}"
        )

        print(
            f"[SEED {seed}] "
            f"Top-10 TF overlap: "
            f"{result['mean_target_top_k_overlap_fraction']:.12f}"
        )

        print(
            f"[SEED {seed}] "
            f"RESULT: "
            f"{'PASS' if result['validation_passed'] else 'FAIL'}"
        )

    # -----------------------------------------------------
    # 6. Save all-seed summary
    # -----------------------------------------------------

    summary_df = pd.DataFrame(
        all_seed_summaries
    )

    preferred_columns = [
        "seed",
        "validation_passed",
        "same_edge_count",
        "same_edge_keys",
        "exact_equal_with_nan",
        "allclose",
        "pearson",
        "spearman",
        "mean_target_top_k_overlap_fraction",
        "mean_target_top_k_jaccard",
        "top_1000_jaccard",
        "top_5000_jaccard",
        "top_10000_jaccard",
        "max_abs_difference",
        "mean_abs_difference",
        "median_abs_difference",
        "full_edge_count",
        "chunked_edge_count",
        "n_tfs",
        "n_targets",
        "full_runtime",
        "chunked_total_runtime",
        "seed_total_runtime",
    ]

    summary_df = summary_df[
        preferred_columns
    ]

    all_seed_path = (
        OUTDIR
        / "summary_all_seeds.tsv"
    )

    summary_df.to_csv(
        all_seed_path,
        sep="\t",
        index=False,
    )

    all_seeds_passed = bool(
        summary_df[
            "validation_passed"
        ].all()
    )

    overall_summary_path = (
        OUTDIR
        / "overall_validation_summary.txt"
    )

    with overall_summary_path.open(
        "w"
    ) as handle:
        handle.write(
            "GRNBOOST2 MULTI-SEED "
            "TARGET-CHUNK VALIDATION\n"
        )

        handle.write(
            "================================"
            "================\n\n"
        )

        handle.write(
            f"Seeds: {SEEDS}\n"
        )

        handle.write(
            f"Pilot targets: "
            f"{PILOT_N_TARGETS}\n"
        )

        handle.write(
            "Chunk sizes: "
            "100 + 100 + 100\n"
        )

        handle.write(
            f"All seeds passed: "
            f"{all_seeds_passed}\n"
        )

        handle.write(
            f"Script start: "
            f"{script_start}\n"
        )

        handle.write(
            f"Script end: "
            f"{now()}\n"
        )

        handle.write(
            f"Total runtime: "
            f"{elapsed_hms(time.time() - total_start)}\n\n"
        )

        for _, row in summary_df.iterrows():
            handle.write(
                f"Seed {int(row['seed'])}: "
                f"{'PASS' if row['validation_passed'] else 'FAIL'}, "
                f"Pearson={row['pearson']:.12f}, "
                f"Spearman={row['spearman']:.12f}, "
                f"Top10TFOverlap="
                f"{row['mean_target_top_k_overlap_fraction']:.12f}\n"
            )

    print("\n====================================")
    print("FINAL MULTI-SEED SUMMARY")
    print("====================================")

    print(
        summary_df[
            [
                "seed",
                "validation_passed",
                "pearson",
                "spearman",
                "mean_target_top_k_overlap_fraction",
                "top_1000_jaccard",
            ]
        ].to_string(index=False)
    )

    print(
        "\nOVERALL RESULT: "
        f"{'PASS' if all_seeds_passed else 'FAIL'}"
    )

    print(
        f"All-seed table: {all_seed_path}"
    )

    print(
        f"Overall summary: "
        f"{overall_summary_path}"
    )

    if not all_seeds_passed:
        raise RuntimeError(
            "At least one seed failed "
            "the chunk validation."
        )


if __name__ == "__main__":
    main()