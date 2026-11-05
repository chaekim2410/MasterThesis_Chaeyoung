#!/usr/bin/env python3
"""
Aggregate pySCENIC regulon memberships across multiple runs and analyse
TF-target edge recurrence without applying a consensus cutoff.

This version supports the standard pySCENIC CSV format with a 3-row header:

,,Enrichment,...
,,AUC,NES,...,TargetGenes,...
TF,MotifID,...

Input pattern:
  results/06_grnboost2/run*/step2/pyscenic_regulons.csv

Output directory:
  results/06_grnboost2/regulon_occurrence
"""

from __future__ import annotations

import argparse
import ast
import glob
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BASE = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis"
)
DEFAULT_INPUT_GLOB = str(
    DEFAULT_BASE / "results/06_grnboost2/run*/step2/pyscenic_regulons.csv"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_BASE / "results/06_grnboost2/regulon_occurrence"
)


def natural_run_key(path: str) -> Tuple[int, str]:
    match = re.search(r"/run(\d+)/", path)
    return (int(match.group(1)) if match else 10**9, path)


def flatten_multilevel_columns(columns: pd.Index) -> List[str]:
    """
    For pySCENIC's 3-level header, keep the deepest meaningful label.

    Examples:
      ('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'TF') -> 'TF'
      ('Enrichment', 'TargetGenes', 'Unnamed: 8_level_2') -> 'TargetGenes'
      ('Enrichment', 'AUC', 'Unnamed: 2_level_2') -> 'AUC'
    """
    flattened = []
    for col in columns:
        if not isinstance(col, tuple):
            flattened.append(str(col).strip())
            continue

        meaningful = []
        for part in col:
            text = str(part).strip()
            if not text:
                continue
            if text.lower().startswith("unnamed:"):
                continue
            meaningful.append(text)

        if not meaningful:
            flattened.append("")
        else:
            # In this file structure, the most specific field is the last
            # non-Unnamed header component.
            flattened.append(meaningful[-1])

    return flattened


def read_pyscenic_csv(csv_path: str) -> pd.DataFrame:
    """
    Read standard pySCENIC regulon CSV.

    First attempt the known 3-row header. A single-row fallback is retained
    for compatibility with already-flattened exports.
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    df.columns = flatten_multilevel_columns(df.columns)

    if "TargetGenes" not in df.columns or "TF" not in df.columns:
        fallback = pd.read_csv(csv_path)
        if "TargetGenes" in fallback.columns and "TF" in fallback.columns:
            return fallback

        raise ValueError(
            f"{csv_path}: could not locate TF and TargetGenes after reading "
            f"the 3-row pySCENIC header. Flattened columns: {list(df.columns)}"
        )

    return df


def parse_target_genes(value: object) -> List[str]:
    """
    Parse values such as:
      [('Fosl2', 0.67), ('Zfp422', 0.42)]
    """
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, (list, tuple, set)):
        parsed = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "[]"}:
            return []

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # Safe fallback for tuple-like malformed strings.
            genes = re.findall(r"\(\s*['\"]([^'\"]+)['\"]\s*,", text)
            return [gene.strip() for gene in genes if gene.strip()]

    genes: List[str] = []
    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                gene = str(item[0]).strip()
            else:
                gene = str(item).strip()

            if gene and gene.lower() not in {"nan", "none"}:
                genes.append(gene)

    return genes


def read_one_run(csv_path: str) -> tuple[pd.DataFrame, dict]:
    run_match = re.search(r"/run(\d+)/", csv_path)
    run_id = int(run_match.group(1)) if run_match else -1

    df = read_pyscenic_csv(csv_path)

    records = []
    unparsable_nonempty_rows = 0

    for _, row in df.iterrows():
        tf = str(row["TF"]).strip()
        if not tf or tf.lower() == "nan":
            continue

        targets = parse_target_genes(row["TargetGenes"])

        raw_target_value = str(row["TargetGenes"]).strip()
        if (
            not targets
            and raw_target_value.lower() not in {"", "nan", "none", "[]"}
        ):
            unparsable_nonempty_rows += 1

        # The supplied CSV contains positive regulons generated from
        # activating modules, so retain the conventional pySCENIC name.
        regulon = f"{tf}(+)"

        for target in targets:
            records.append(
                {
                    "run": run_id,
                    "regulon": regulon,
                    "TF": tf,
                    "target": target,
                }
            )

    edge_df = pd.DataFrame(records)
    if edge_df.empty:
        raise ValueError(
            f"{csv_path}: no TF-target memberships could be parsed."
        )

    n_before = len(edge_df)

    # Critical: one target can appear under several motif rows for the same TF.
    # It must count only once per run.
    edge_df = edge_df.drop_duplicates(
        subset=["run", "TF", "target"]
    ).reset_index(drop=True)

    metadata = {
        "run": run_id,
        "file": csv_path,
        "csv_rows": len(df),
        "parsed_memberships_before_within_run_dedup": n_before,
        "unique_tf_target_edges_within_run": len(edge_df),
        "n_tfs": edge_df["TF"].nunique(),
        "unparsable_nonempty_target_rows": unparsable_nonempty_rows,
    }

    return edge_df, metadata


def save_occurrence_histogram(hist: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(hist["occurrence"], hist["n_edges"])
    plt.xlabel("Occurrence across runs")
    plt.ylabel("Number of unique TF-target edges")
    plt.title("Regulon edge occurrence distribution")
    plt.xticks(hist["occurrence"])
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_threshold_curve(curve: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        curve["threshold"],
        curve["n_edges_remaining"],
        marker="o",
    )
    plt.xlabel("Minimum occurrence threshold")
    plt.ylabel("Number of retained TF-target edges")
    plt.title("Edges retained by occurrence threshold")
    plt.xticks(curve["threshold"])
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_regulon_size_curve(curve: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        curve["threshold"],
        curve["mean_regulon_size"],
        marker="o",
    )
    plt.xlabel("Minimum occurrence threshold")
    plt.ylabel("Mean regulon size")
    plt.title("Mean regulon size by occurrence threshold")
    plt.xticks(curve["threshold"])
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(
        glob.glob(args.input_glob),
        key=natural_run_key,
    )

    if len(input_files) != args.expected_runs:
        raise RuntimeError(
            f"Expected {args.expected_runs} files, found {len(input_files)} "
            f"with pattern:\n{args.input_glob}\n"
            + "\n".join(input_files)
        )

    all_edges = []
    run_metadata = []

    print(f"[INFO] Found {len(input_files)} regulon files.", flush=True)

    for path in input_files:
        print(f"[INFO] Reading {path}", flush=True)
        edges, metadata = read_one_run(path)
        all_edges.append(edges)
        run_metadata.append(metadata)

    run_edges = pd.concat(all_edges, ignore_index=True)

    run_summary = (
        pd.DataFrame(run_metadata)
        .sort_values("run")
        .reset_index(drop=True)
    )
    run_summary.to_csv(
        output_dir / "run_summary.tsv",
        sep="\t",
        index=False,
    )

    # One row per unique TF-target edge in the union across all runs.
    occurrence = (
        run_edges.groupby(["regulon", "TF", "target"], as_index=False)
        .agg(
            occurrence=("run", "nunique"),
            runs_present=(
                "run",
                lambda x: ",".join(map(str, sorted(set(x)))),
            ),
        )
    )
    occurrence["frequency"] = (
        occurrence["occurrence"] / len(input_files)
    )

    occurrence = occurrence.sort_values(
        ["occurrence", "TF", "target"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    occurrence.to_csv(
        output_dir / "occurrence_table.tsv",
        sep="\t",
        index=False,
    )

    histogram = (
        occurrence.groupby("occurrence")
        .size()
        .reindex(
            range(1, len(input_files) + 1),
            fill_value=0,
        )
        .rename("n_edges")
        .reset_index()
    )
    histogram["fraction_of_all_edges"] = (
        histogram["n_edges"] / len(occurrence)
    )
    histogram.to_csv(
        output_dir / "occurrence_histogram.tsv",
        sep="\t",
        index=False,
    )

    threshold_rows = []
    regulon_size_rows = []

    for threshold in range(1, len(input_files) + 1):
        kept = occurrence[
            occurrence["occurrence"] >= threshold
        ].copy()

        sizes = (
            kept.groupby(["regulon", "TF"])
            .size()
            .rename("regulon_size")
            .reset_index()
        )

        threshold_rows.append(
            {
                "threshold": threshold,
                "n_edges_remaining": len(kept),
                "fraction_edges_remaining": (
                    len(kept) / len(occurrence)
                ),
                "n_regulons_remaining": kept["regulon"].nunique(),
                "n_tfs_remaining": kept["TF"].nunique(),
            }
        )

        if sizes.empty:
            mean_size = median_size = min_size = max_size = 0
        else:
            mean_size = sizes["regulon_size"].mean()
            median_size = sizes["regulon_size"].median()
            min_size = sizes["regulon_size"].min()
            max_size = sizes["regulon_size"].max()

        regulon_size_rows.append(
            {
                "threshold": threshold,
                "n_regulons_remaining": len(sizes),
                "mean_regulon_size": mean_size,
                "median_regulon_size": median_size,
                "min_regulon_size": min_size,
                "max_regulon_size": max_size,
            }
        )

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(
        output_dir / "edge_count_by_threshold.tsv",
        sep="\t",
        index=False,
    )

    regulon_size_df = pd.DataFrame(regulon_size_rows)
    regulon_size_df.to_csv(
        output_dir / "regulon_size_by_threshold.tsv",
        sep="\t",
        index=False,
    )

    tf_summary = (
        occurrence.groupby("TF", as_index=False)
        .agg(
            n_unique_targets=("target", "nunique"),
            mean_occurrence=("occurrence", "mean"),
            median_occurrence=("occurrence", "median"),
            min_occurrence=("occurrence", "min"),
            max_occurrence=("occurrence", "max"),
            n_edges_occurrence_ge_5=(
                "occurrence",
                lambda x: int((x >= 5).sum()),
            ),
            n_edges_occurrence_ge_8=(
                "occurrence",
                lambda x: int((x >= 8).sum()),
            ),
            n_edges_occurrence_eq_10=(
                "occurrence",
                lambda x: int((x == len(input_files)).sum()),
            ),
        )
        .sort_values(
            ["mean_occurrence", "n_unique_targets"],
            ascending=[False, False],
        )
    )
    tf_summary.to_csv(
        output_dir / "tf_summary.tsv",
        sep="\t",
        index=False,
    )

    regulon_summary = (
        occurrence.groupby(["regulon", "TF"], as_index=False)
        .agg(
            n_targets=("target", "nunique"),
            mean_occurrence=("occurrence", "mean"),
            median_occurrence=("occurrence", "median"),
            min_occurrence=("occurrence", "min"),
            max_occurrence=("occurrence", "max"),
            n_targets_occurrence_ge_5=(
                "occurrence",
                lambda x: int((x >= 5).sum()),
            ),
            n_targets_occurrence_ge_8=(
                "occurrence",
                lambda x: int((x >= 8).sum()),
            ),
            n_targets_occurrence_eq_10=(
                "occurrence",
                lambda x: int((x == len(input_files)).sum()),
            ),
        )
        .sort_values(
            ["mean_occurrence", "n_targets"],
            ascending=[False, False],
        )
    )
    regulon_summary.to_csv(
        output_dir / "regulon_summary.tsv",
        sep="\t",
        index=False,
    )

    save_occurrence_histogram(
        histogram,
        output_dir / "occurrence_histogram.png",
    )
    save_threshold_curve(
        threshold_df,
        output_dir / "edge_threshold_curve.png",
    )
    save_regulon_size_curve(
        regulon_size_df,
        output_dir / "regulon_size_curve.png",
    )

    n_edges = len(occurrence)
    exact_counts = dict(
        zip(histogram["occurrence"], histogram["n_edges"])
    )

    lines = [
        "pySCENIC multirun regulon occurrence analysis",
        "=" * 48,
        f"Input files: {len(input_files)}",
        f"Unique TFs: {occurrence['TF'].nunique():,}",
        f"Unique TF-target edges (union): {n_edges:,}",
        "",
        "Exact occurrence counts:",
    ]

    for occurrence_value in range(1, len(input_files) + 1):
        count = int(exact_counts.get(occurrence_value, 0))
        percentage = (
            100.0 * count / n_edges if n_edges else 0.0
        )
        lines.append(
            f"  occurrence={occurrence_value:2d}: "
            f"{count:10,d} ({percentage:6.2f}%)"
        )

    lines.extend(
        [
            "",
            "Edges retained at selected thresholds:",
        ]
    )

    for threshold in [1, 2, 3, 5, 8, 10]:
        row = threshold_df[
            threshold_df["threshold"] == threshold
        ].iloc[0]
        lines.append(
            f"  occurrence >= {threshold:2d}: "
            f"{int(row['n_edges_remaining']):10,d} "
            f"({100.0 * row['fraction_edges_remaining']:6.2f}%)"
        )

    lines.extend(
        [
            "",
            "No cutoff was applied.",
            "occurrence_table.tsv contains the complete edge union.",
        ]
    )

    (output_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines), flush=True)
    print(
        f"\n[INFO] Results saved to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )