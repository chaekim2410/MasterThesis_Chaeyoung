#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


RUN_NAME = "run10"
N_CHUNKS = 10

CHUNK_DIR = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/06_grnboost2"
) / RUN_NAME / "step1" / "step1_chunked"

OUT_CSV = CHUNK_DIR / "chunk_completion_summary.csv"


def count_lines(path):
    if not path.exists():
        return None
    with path.open() as f:
        return sum(1 for _ in f)


def main():
    rows = []

    for i in range(N_CHUNKS):
        target_file = CHUNK_DIR / f"target_genes_chunk{i:03d}.txt"
        adj_file = CHUNK_DIR / f"adjacencies_chunk{i:03d}.tsv"
        summary_file = CHUNK_DIR / f"summary_chunk{i:03d}.txt"

        n_targets = count_lines(target_file)

        if adj_file.exists():
            try:
                df = pd.read_csv(adj_file, sep="\t")
                n_edges = len(df)
                n_unique_targets = df["target"].nunique() if "target" in df.columns else None
                n_unique_tfs = df["TF"].nunique() if "TF" in df.columns else None
                status = "DONE"
            except Exception as e:
                n_edges = None
                n_unique_targets = None
                n_unique_tfs = None
                status = f"BROKEN: {e}"
        else:
            n_edges = None
            n_unique_targets = None
            n_unique_tfs = None
            status = "MISSING"

        rows.append({
            "chunk_id": i,
            "target_file_exists": target_file.exists(),
            "adj_file_exists": adj_file.exists(),
            "summary_file_exists": summary_file.exists(),
            "n_target_genes_planned": n_targets,
            "n_edges": n_edges,
            "n_unique_targets_in_edges": n_unique_targets,
            "n_unique_tfs_in_edges": n_unique_tfs,
            "status": status,
            "target_file": str(target_file),
            "adj_file": str(adj_file),
            "summary_file": str(summary_file),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print(out[[
        "chunk_id",
        "n_target_genes_planned",
        "n_edges",
        "n_unique_targets_in_edges",
        "n_unique_tfs_in_edges",
        "status",
    ]])

    print(f"\nSaved: {OUT_CSV}")

    if (out["status"] != "DONE").any():
        print("\n[WARNING] Some chunks are not complete.")
    else:
        print("\n[OK] All chunks completed.")


if __name__ == "__main__":
    main()