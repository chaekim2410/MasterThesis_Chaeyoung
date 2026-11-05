#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


# =========================================================
# USER SETTINGS
# =========================================================
ADATA_PATH = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/02_annotation/domain_labels/adata_labeled.h5ad"
)

OUTDIR = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/04_sampling/bin_ids"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_BINS_SAMPLE = 30000

CLUSTER_COL = "spaceranger_cluster"
# Main spatial-context label for sampling. Use the raw bin-level label here.
NEIGHBOR_COL = "domain_from_bin_class"

# Very small cluster x neighbor strata are merged into cluster-specific small groups
MIN_STRATUM_SIZE = 50
SMALL_STRATUM_LABEL = "small_stratum"

# Grid-based spatial sampling
GRID_N_X = 80
GRID_N_Y = 80

# Sketching settings
RUN_SKETCH = True
N_TOP_HVGS = 3000
N_PCS = 30


# =========================================================
# HELPERS
# =========================================================
def save_bin_ids(obs_names, idx, outpath: Path):
    selected_ids = obs_names[idx]
    with outpath.open("w") as f:
        for x in selected_ids:
            f.write(f"{x}\n")


def choose_random_indices(n_obs, n_select, seed):
    rng = np.random.default_rng(seed)
    n_select = min(n_select, n_obs)
    return np.sort(rng.choice(n_obs, size=n_select, replace=False))


def proportional_stratified_indices(labels, n_select, seed, min_per_nonempty=1):
    """
    Generic proportional stratified sampling.
    Each stratum receives samples approximately proportional to its size.
    """
    rng = np.random.default_rng(seed)

    labels = pd.Series(labels).astype(str)
    counts = labels.value_counts()
    proportions = counts / counts.sum()

    target_counts = (proportions * n_select).round().astype(int)

    # If possible, keep at least one sample for each non-empty stratum.
    if n_select >= len(target_counts):
        for lab in target_counts.index:
            if counts[lab] > 0 and target_counts[lab] == 0:
                target_counts[lab] = min_per_nonempty

    # Adjust to exact sample size.
    diff = n_select - int(target_counts.sum())
    order = counts.sort_values(ascending=False).index.tolist()

    i = 0
    while diff != 0 and len(order) > 0:
        lab = order[i % len(order)]
        if diff > 0:
            target_counts[lab] += 1
            diff -= 1
        else:
            if target_counts[lab] > 0:
                target_counts[lab] -= 1
                diff += 1
        i += 1

    selected = []
    for lab, target in target_counts.items():
        idx = np.where(labels.values == lab)[0]
        target = min(int(target), len(idx))
        if target <= 0:
            continue
        picked = rng.choice(idx, size=target, replace=False)
        selected.extend(picked.tolist())

    selected = np.array(sorted(set(selected)))

    if len(selected) < n_select:
        remaining = np.setdiff1d(np.arange(len(labels)), selected)
        need = min(n_select - len(selected), len(remaining))
        if need > 0:
            extra = rng.choice(remaining, size=need, replace=False)
            selected = np.sort(np.concatenate([selected, extra]))

    if len(selected) > n_select:
        selected = np.sort(rng.choice(selected, size=n_select, replace=False))

    return selected


def cluster_stratified_indices(cluster_labels, n_select, seed):
    return proportional_stratified_indices(cluster_labels, n_select, seed)


def make_cluster_x_neighbor_labels(obs, cluster_col, neighbor_col, min_stratum_size):
    """
    Build cluster x neighbor strata.

    Small raw strata are merged into cluster-specific fallback groups:
      clusterID__x__small_stratum
    """
    cluster = obs[cluster_col].astype(str).fillna("NA_cluster")
    neighbor = obs[neighbor_col].astype(str).fillna("NA_neighbor")

    raw_stratum = cluster + "__x__" + neighbor
    raw_counts = raw_stratum.value_counts()
    small_raw = set(raw_counts[raw_counts < min_stratum_size].index)

    final_stratum = raw_stratum.copy()
    small_mask = raw_stratum.isin(small_raw)
    final_stratum.loc[small_mask] = cluster.loc[small_mask] + "__x__" + SMALL_STRATUM_LABEL

    return raw_stratum, final_stratum


def cluster_x_neighbor_stratified_indices(
    obs,
    cluster_col,
    neighbor_col,
    n_select,
    seed,
    min_stratum_size,
    outdir,
):
    raw_stratum, final_stratum = make_cluster_x_neighbor_labels(
        obs=obs,
        cluster_col=cluster_col,
        neighbor_col=neighbor_col,
        min_stratum_size=min_stratum_size,
    )

    diag = pd.DataFrame({
        "obs_name": obs.index.astype(str),
        "cluster": obs[cluster_col].astype(str).values,
        "neighbor": obs[neighbor_col].astype(str).values,
        "raw_cluster_x_neighbor": raw_stratum.values,
        "final_cluster_x_neighbor": final_stratum.values,
    })
    diag.to_csv(outdir / "cluster_x_neighbor_stratum_per_bin.csv", index=False)

    raw_counts = raw_stratum.value_counts().rename_axis("raw_stratum").reset_index(name="n_bins")
    raw_counts.to_csv(outdir / "cluster_x_neighbor_raw_stratum_counts.csv", index=False)

    final_counts = final_stratum.value_counts().rename_axis("final_stratum").reset_index(name="n_bins")
    final_counts.to_csv(outdir / "cluster_x_neighbor_final_stratum_counts.csv", index=False)

    idx = proportional_stratified_indices(final_stratum, n_select, seed)
    return idx


def grid_based_spatial_indices(coords, n_select, grid_n_x, grid_n_y, seed):
    """
    Uniform grid-based spatial sampling.
    Each non-empty spatial grid cell contributes approximately equal numbers of bins.
    """
    rng = np.random.default_rng(seed)

    x = coords[:, 0]
    y = coords[:, 1]

    x_bins = np.linspace(x.min(), x.max(), grid_n_x + 1)
    y_bins = np.linspace(y.min(), y.max(), grid_n_y + 1)

    x_idx = np.clip(np.digitize(x, x_bins) - 1, 0, grid_n_x - 1)
    y_idx = np.clip(np.digitize(y, y_bins) - 1, 0, grid_n_y - 1)

    grid_cell = pd.Series([f"{i}_{j}" for i, j in zip(x_idx, y_idx)])
    groups = grid_cell.groupby(grid_cell).groups

    nonempty_cells = list(groups.keys())
    if len(nonempty_cells) == 0:
        raise ValueError("No non-empty grid cells found.")

    base_per_cell = max(1, n_select // len(nonempty_cells))

    selected = []
    leftover = []
    for cell in nonempty_cells:
        idx = np.array(list(groups[cell]))
        if len(idx) <= base_per_cell:
            selected.extend(idx.tolist())
        else:
            picked = rng.choice(idx, size=base_per_cell, replace=False)
            selected.extend(picked.tolist())
            remaining = np.setdiff1d(idx, picked)
            leftover.extend(remaining.tolist())

    selected = list(dict.fromkeys(selected))

    if len(selected) < n_select:
        remaining_pool = np.array(list(set(leftover).difference(set(selected))))
        if len(remaining_pool) > 0:
            need = min(n_select - len(selected), len(remaining_pool))
            extra = rng.choice(remaining_pool, size=need, replace=False)
            selected.extend(extra.tolist())

    selected = np.array(sorted(set(selected)))

    if len(selected) > n_select:
        selected = np.sort(rng.choice(selected, size=n_select, replace=False))

    return selected


def prepare_pca_for_sketch(adata):
    """
    Build PCA representation for geometric sketching.
    This copy is used only for representative bin selection.
    The saved sampled h5ad remains unchanged.
    """
    ad = adata.copy()

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    sc.pp.highly_variable_genes(
        ad,
        flavor="seurat",
        n_top_genes=min(N_TOP_HVGS, ad.n_vars),
        inplace=True,
    )

    n_hvg = int(ad.var["highly_variable"].sum())
    n_pcs_use = min(N_PCS, ad.n_obs - 1, n_hvg - 1)
    if n_pcs_use < 2:
        raise ValueError("Not enough dimensions for PCA sketching.")

    sc.tl.pca(
        ad,
        n_comps=n_pcs_use,
        use_highly_variable=True,
        svd_solver="arpack",
    )

    return ad.obsm["X_pca"]


def geometric_sketch_indices(adata, n_select, seed):
    """
    Geometric sketching based on PCA space.

    Requires:
      pip install geosketch
    """
    try:
        import geosketch as sketch
        use_module = True
    except ImportError:
        try:
            from geosketch import gs
            use_module = False
        except ImportError:
            raise ImportError(
                "geosketch is not installed. Install it first or set RUN_SKETCH=False."
            )

    X_pca = prepare_pca_for_sketch(adata)
    n_select = min(n_select, adata.n_obs)

    if use_module:
        idx = sketch.gs(X=X_pca, N=int(n_select), seed=seed)
    else:
        idx = gs(X=X_pca, N=int(n_select), seed=seed)

    return np.sort(np.asarray(idx, dtype=int))


def write_overlap_matrix(obs_names, index_dict, outpath):
    rows = []
    names = list(index_dict.keys())

    sets = {name: set(obs_names[idx].astype(str).tolist()) for name, idx in index_dict.items()}

    for a in names:
        for b in names:
            inter = len(sets[a].intersection(sets[b]))
            union = len(sets[a].union(sets[b]))
            rows.append({
                "method_a": a,
                "method_b": b,
                "n_intersection": inter,
                "n_union": union,
                "jaccard": inter / union if union > 0 else np.nan,
            })

    pd.DataFrame(rows).to_csv(outpath, index=False)


# =========================================================
# MAIN
# =========================================================
def main():
    print("[1/7] Loading AnnData...")
    adata = sc.read_h5ad(ADATA_PATH)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    print(f"[INFO] Shape: {adata.shape}")

    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] not found.")
    if CLUSTER_COL not in adata.obs.columns:
        raise KeyError(f"{CLUSTER_COL} not found in adata.obs")
    if NEIGHBOR_COL not in adata.obs.columns:
        raise KeyError(
            f"{NEIGHBOR_COL} not found in adata.obs. "
            "Use your raw neighborhood/domain column here."
        )

    coords = np.asarray(adata.obsm["spatial"])
    index_dict = {}

    print("[2/7] Random sampling...")
    index_dict["random"] = choose_random_indices(adata.n_obs, N_BINS_SAMPLE, RANDOM_SEED)
    save_bin_ids(adata.obs_names, index_dict["random"], OUTDIR / "random_bin_ids.txt")

    print("[3/7] Cluster-stratified sampling...")
    index_dict["cluster"] = cluster_stratified_indices(
        adata.obs[CLUSTER_COL],
        N_BINS_SAMPLE,
        RANDOM_SEED,
    )
    save_bin_ids(adata.obs_names, index_dict["cluster"], OUTDIR / "cluster_stratified_bin_ids.txt")

    print("[4/7] Cluster x neighbor stratified sampling...")
    index_dict["cluster_x_neighbor"] = cluster_x_neighbor_stratified_indices(
        obs=adata.obs,
        cluster_col=CLUSTER_COL,
        neighbor_col=NEIGHBOR_COL,
        n_select=N_BINS_SAMPLE,
        seed=RANDOM_SEED,
        min_stratum_size=MIN_STRATUM_SIZE,
        outdir=OUTDIR,
    )
    save_bin_ids(
        adata.obs_names,
        index_dict["cluster_x_neighbor"],
        OUTDIR / "cluster_x_neighbor_bin_ids.txt",
    )

    print("[5/7] Grid-based spatial sampling...")
    index_dict["grid"] = grid_based_spatial_indices(
        coords=coords,
        n_select=N_BINS_SAMPLE,
        grid_n_x=GRID_N_X,
        grid_n_y=GRID_N_Y,
        seed=RANDOM_SEED,
    )
    save_bin_ids(adata.obs_names, index_dict["grid"], OUTDIR / "grid_bin_ids.txt")

    print("[6/7] Geometric sketching...")
    if RUN_SKETCH:
        try:
            index_dict["sketch"] = geometric_sketch_indices(adata, N_BINS_SAMPLE, RANDOM_SEED)
            save_bin_ids(adata.obs_names, index_dict["sketch"], OUTDIR / "sketch_bin_ids.txt")
        except Exception as e:
            print("[WARN] Sketching failed.")
            print(f"[WARN] Reason: {e}")
            print("[WARN] Continuing without sketch.")
    else:
        print("[INFO] RUN_SKETCH=False, skipping sketch.")

    print("[7/7] Writing summaries...")
    write_overlap_matrix(adata.obs_names, index_dict, OUTDIR / "sampling_method_overlap_matrix.csv")

    file_map = {
        "random": "random_bin_ids.txt",
        "cluster": "cluster_stratified_bin_ids.txt",
        "cluster_x_neighbor": "cluster_x_neighbor_bin_ids.txt",
        "grid": "grid_bin_ids.txt",
        "sketch": "sketch_bin_ids.txt",
    }

    summary_rows = []
    for name, idx in index_dict.items():
        summary_rows.append({
            "method": name,
            "n_selected": len(idx),
            "bin_id_file": str(OUTDIR / file_map[name]),
        })

    pd.DataFrame(summary_rows).to_csv(OUTDIR / "sampling_bin_ids_summary.csv", index=False)

    with (OUTDIR / "sampling_bin_ids_summary.txt").open("w") as f:
        f.write("SAMPLING BIN IDS SUMMARY\n")
        f.write("========================\n\n")
        f.write(f"Input AnnData: {ADATA_PATH}\n")
        f.write(f"Shape: {adata.shape}\n")
        f.write(f"Sample size: {N_BINS_SAMPLE}\n")
        f.write(f"Cluster column: {CLUSTER_COL}\n")
        f.write(f"Neighbor/domain column: {NEIGHBOR_COL}\n")
        f.write(f"Minimum raw cluster x neighbor stratum size: {MIN_STRATUM_SIZE}\n")
        f.write(f"Small stratum label: {SMALL_STRATUM_LABEL}\n")
        f.write(f"Grid: {GRID_N_X} x {GRID_N_Y}\n")
        f.write(f"Run sketch: {RUN_SKETCH}\n")
        f.write(f"HVGs for sketch: {N_TOP_HVGS}\n")
        f.write(f"PCA components for sketch: {N_PCS}\n\n")
        f.write("Outputs:\n")
        for row in summary_rows:
            f.write(f"  - {row['method']}: {row['bin_id_file']}\n")
        f.write("\nExtra diagnostics for cluster x neighbor:\n")
        f.write("  - cluster_x_neighbor_stratum_per_bin.csv\n")
        f.write("  - cluster_x_neighbor_raw_stratum_counts.csv\n")
        f.write("  - cluster_x_neighbor_final_stratum_counts.csv\n")
        f.write("\nOverlap:\n")
        f.write("  - sampling_method_overlap_matrix.csv\n")

    print(f"[DONE] Saved bin ID files to: {OUTDIR}")


if __name__ == "__main__":
    main()
