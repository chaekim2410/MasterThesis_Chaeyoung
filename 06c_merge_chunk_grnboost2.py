#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


RUN_NAME = "run10"
N_CHUNKS = 10

CHUNK_DIR = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/06_grnboost2"
) / RUN_NAME / "step1" / "step1_chunked"

OUTDIR = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/06_grnboost2"
) / RUN_NAME / "step1"

OUTDIR.mkdir(parents=True, exist_ok=True)


def read_list(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def main():
    rows = []
    dfs = []

    all_planned_targets = []

    for i in range(N_CHUNKS):
        target_file = CHUNK_DIR / f"target_genes_chunk{i:03d}.txt"
        adj_file = CHUNK_DIR / f"adjacencies_chunk{i:03d}.tsv"

        targets = read_list(target_file)
        all_planned_targets.extend(targets)

        if not target_file.exists():
            raise FileNotFoundError(f"Missing target file: {target_file}")

        if not adj_file.exists():
            raise FileNotFoundError(f"Missing adjacency file: {adj_file}")

        df = pd.read_csv(adj_file, sep="\t")

        required = {"TF", "target", "importance"}
        if not required.issubset(df.columns):
            raise ValueError(f"{adj_file} missing columns. Found: {list(df.columns)}")

        # safety: this chunk should only contain its assigned targets
        unexpected_targets = sorted(set(df["target"]) - set(targets))
        if unexpected_targets:
            raise ValueError(
                f"Chunk {i} has targets not assigned to this chunk. "
                f"Example: {unexpected_targets[:10]}"
            )

        rows.append({
            "chunk_id": i,
            "n_planned_targets": len(targets),
            "n_edges": len(df),
            "n_unique_targets_with_edges": df["target"].nunique(),
            "n_unique_tfs": df["TF"].nunique(),
            "adj_file": str(adj_file),
        })

        dfs.append(df)

    # check target chunk overlap
    planned_series = pd.Series(all_planned_targets)
    duplicated_targets = planned_series[planned_series.duplicated()].unique().tolist()

    if duplicated_targets:
        raise ValueError(
            f"Target genes are duplicated across chunks. "
            f"Examples: {duplicated_targets[:20]}"
        )

    chunk_summary = pd.DataFrame(rows)
    chunk_summary.to_csv(CHUNK_DIR / "chunk_completion_summary.csv", index=False)

    merged = pd.concat(dfs, ignore_index=True)

    raw_edges = len(merged)

    # should usually remove 0 if chunks were disjoint
    merged = merged.drop_duplicates(subset=["TF", "target"], keep="first")
    dedup_edges = len(merged)

    merged = merged.sort_values("importance", ascending=False).reset_index(drop=True)

    out_path = OUTDIR / "adjacencies.tsv"
    merged.to_csv(out_path, sep="\t", index=False)

    summary_path = OUTDIR / "summary.txt"
    with summary_path.open("w") as f:
        f.write("MERGED GRNBOOST2 SUMMARY\n")
        f.write("========================\n\n")
        f.write(f"RUN_NAME: {RUN_NAME}\n")
        f.write(f"N_CHUNKS: {N_CHUNKS}\n")
        f.write(f"CHUNK_DIR: {CHUNK_DIR}\n")
        f.write(f"OUTDIR: {OUTDIR}\n\n")

        f.write("CHUNK TARGET CHECK\n")
        f.write("------------------\n")
        f.write(f"Total planned target genes: {len(all_planned_targets)}\n")
        f.write(f"Duplicated planned targets across chunks: {len(duplicated_targets)}\n\n")

        f.write("MERGE COUNTS\n")
        f.write("------------\n")
        f.write(f"Raw merged edges: {raw_edges}\n")
        f.write(f"Duplicate TF-target edges removed: {raw_edges - dedup_edges}\n")
        f.write(f"Final edges: {dedup_edges}\n")
        f.write(f"Unique TFs: {merged['TF'].nunique()}\n")
        f.write(f"Unique targets with at least one edge: {merged['target'].nunique()}\n")
        f.write(f"Output: {out_path}\n")
        f.write(f"Chunk summary: {CHUNK_DIR / 'chunk_completion_summary.csv'}\n")

    print(chunk_summary)
    print(f"\n[DONE] Merged adjacency saved: {out_path}")
    print(f"[DONE] Final shape: {merged.shape}")
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()