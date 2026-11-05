#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# =========================
# USER SETTINGS
# =========================
REFERENCE_ADATA_PATH = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/02_annotation/domain_labels/adata_labeled.h5ad"
)

SAMPLED_BASEDIR = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/04_sampling/d0/sampled_adatas"
)

CANDIDATES = {
    "cluster_x_neighbor": SAMPLED_BASEDIR / "cluster_x_neighbor" / "sampled_adata.h5ad"
}

OUTDIR = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/04_sampling/d0/validation"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

CLUSTER_COL = "spaceranger_cluster"
NEIGHBOR_COL = "domain_from_bin_class"  # raw neighborhood / domain proxy
QC_COLS = ["total_counts", "n_genes_by_counts", "pct_counts_mt"]

GRID_N_X = 80
GRID_N_Y = 80
KNN_K = 6
RARE_CLUSTER_THRESHOLD = 0.01
RARE_NEIGHBOR_THRESHOLD = 0.01


# =========================
# METRICS
# =========================
def proportions(obs, col):
    return obs[col].astype(str).value_counts(normalize=True).sort_index()


def l1_distance(p, q):
    idx = sorted(set(p.index).union(set(q.index)))
    p2 = p.reindex(idx, fill_value=0)
    q2 = q.reindex(idx, fill_value=0)
    return float(np.abs(p2 - q2).sum())


def combined_label(obs, col_a, col_b):
    return obs[col_a].astype(str) + "__x__" + obs[col_b].astype(str)


def rare_label_retention(full_obs, sampled_obs, col, threshold):
    full_props = proportions(full_obs, col)
    rare_labels = full_props[full_props < threshold].index.tolist()
    sampled_labels = set(sampled_obs[col].astype(str).unique())
    retained = sum(label in sampled_labels for label in rare_labels)
    return retained, len(rare_labels), rare_labels


def median_relative_difference(full_vals, sampled_vals):
    full_med = np.nanmedian(full_vals)
    sampled_med = np.nanmedian(sampled_vals)
    if full_med == 0:
        return np.nan
    return float(abs(sampled_med - full_med) / full_med)


def grid_coverage_score(coords_full, coords_sampled, grid_n_x, grid_n_y):
    x_full = coords_full[:, 0]
    y_full = coords_full[:, 1]

    x_bins = np.linspace(x_full.min(), x_full.max(), grid_n_x + 1)
    y_bins = np.linspace(y_full.min(), y_full.max(), grid_n_y + 1)

    def cells(coords):
        x = coords[:, 0]
        y = coords[:, 1]
        x_idx = np.clip(np.digitize(x, x_bins) - 1, 0, grid_n_x - 1)
        y_idx = np.clip(np.digitize(y, y_bins) - 1, 0, grid_n_y - 1)
        return set(zip(x_idx, y_idx))

    full_cells = cells(coords_full)
    sampled_cells = cells(coords_sampled)
    if len(full_cells) == 0:
        return np.nan
    return len(sampled_cells.intersection(full_cells)) / len(full_cells)


def nearest_neighbor_distance_summary(coords, k):
    if coords.shape[0] <= k:
        return {
            "median_nn_distance": np.nan,
            "mean_nn_distance": np.nan,
            "q90_nn_distance": np.nan,
        }

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords)

    mean_neighbor_dist = distances[:, 1:].mean(axis=1)
    return {
        "median_nn_distance": float(np.median(mean_neighbor_dist)),
        "mean_nn_distance": float(np.mean(mean_neighbor_dist)),
        "q90_nn_distance": float(np.percentile(mean_neighbor_dist, 90)),
    }


# =========================
# PLOTS
# =========================
def make_proportion_plot(prop_table, title, xlabel, outpath, log=False):
    fig, ax = plt.subplots(figsize=(14, 6))
    prop_table.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Proportion (log scale)" if log else "Proportion")
    if log:
        ax.set_yscale("log")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def make_spatial_coverage_plot(ref_coords, sampled_dict, outpath):
    n = len(sampled_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, ad) in zip(axes, sampled_dict.items()):
        coords = np.asarray(ad.obsm["spatial"])
        ax.scatter(ref_coords[:, 0], ref_coords[:, 1], s=0.2, c="lightgray", alpha=0.3)
        ax.scatter(coords[:, 0], coords[:, 1], s=0.5, c="red", alpha=0.7)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def make_qc_boxplot(adata_ref, sampled_dict, outpath):
    frames = []
    for col in QC_COLS:
        if col not in adata_ref.obs.columns:
            continue

        frames.append(pd.DataFrame({
            "candidate": "full",
            "metric": col,
            "value": pd.to_numeric(adata_ref.obs[col], errors="coerce").values,
        }))

        for name, ad in sampled_dict.items():
            if col not in ad.obs.columns:
                continue
            frames.append(pd.DataFrame({
                "candidate": name,
                "metric": col,
                "value": pd.to_numeric(ad.obs[col], errors="coerce").values,
            }))

    if len(frames) == 0:
        return

    df = pd.concat(frames, axis=0)
    metrics = df["metric"].unique().tolist()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric]
        sub.boxplot(column="value", by="candidate", ax=ax, grid=False)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("QC distribution comparison")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def make_metrics_barplot(metrics_df, outpath):
    metrics = [
        ("cluster_l1_distance", "Cluster L1\nlower better"),
        ("neighbor_l1_distance", "Neighborhood L1\nlower better"),
        ("cluster_x_neighbor_l1_distance", "Cluster x neighbor L1\nlower better"),
        ("spatial_grid_coverage", "Spatial coverage\nhigher better"),
        ("median_nn_distance", "Median NN distance\nlower better"),
        ("rare_clusters_retained", "Rare clusters retained\nhigher better"),
        ("rare_neighbors_retained", "Rare neighborhoods retained\nhigher better"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, (col, title) in zip(axes, metrics):
        if col not in metrics_df.columns:
            continue
        ax.bar(metrics_df["candidate"], metrics_df[col])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


# =========================
# VALIDATION
# =========================
def validate_adata(adata, name):
    if CLUSTER_COL not in adata.obs.columns:
        raise KeyError(f"{name}: {CLUSTER_COL} not found.")
    if NEIGHBOR_COL not in adata.obs.columns:
        raise KeyError(f"{name}: {NEIGHBOR_COL} not found.")
    if "spatial" not in adata.obsm:
        raise KeyError(f"{name}: adata.obsm['spatial'] not found.")


def main():
    print("[1/7] Loading reference AnnData...")
    adata_ref = sc.read_h5ad(REFERENCE_ADATA_PATH)
    adata_ref.obs_names = adata_ref.obs_names.astype(str)
    adata_ref.var_names = adata_ref.var_names.astype(str)
    adata_ref.obs_names_make_unique()
    adata_ref.var_names_make_unique()
    validate_adata(adata_ref, "reference")

    print("[2/7] Loading sampled candidates...")
    sampled_dict = {}
    for name, path in CANDIDATES.items():
        if not path.exists():
            print(f"[WARN] Missing candidate file, skipping: {path}")
            continue
        ad = sc.read_h5ad(path)
        ad.obs_names = ad.obs_names.astype(str)
        ad.var_names = ad.var_names.astype(str)
        ad.obs_names_make_unique()
        ad.var_names_make_unique()
        validate_adata(ad, name)
        sampled_dict[name] = ad

    if len(sampled_dict) == 0:
        raise ValueError("No sampled candidate files were found.")

    print("[3/7] Computing metrics...")
    ref_coords = np.asarray(adata_ref.obsm["spatial"])
    ref_cluster_props = proportions(adata_ref.obs, CLUSTER_COL)
    ref_neighbor_props = proportions(adata_ref.obs, NEIGHBOR_COL)
    ref_joint_props = proportions(
        pd.DataFrame({"joint": combined_label(adata_ref.obs, CLUSTER_COL, NEIGHBOR_COL)}),
        "joint",
    )

    rows = []
    for name, ad in sampled_dict.items():
        coords = np.asarray(ad.obsm["spatial"])

        samp_cluster_props = proportions(ad.obs, CLUSTER_COL)
        samp_neighbor_props = proportions(ad.obs, NEIGHBOR_COL)
        samp_joint_props = proportions(
            pd.DataFrame({"joint": combined_label(ad.obs, CLUSTER_COL, NEIGHBOR_COL)}),
            "joint",
        )

        rare_clusters_retained, n_rare_clusters, _ = rare_label_retention(
            adata_ref.obs, ad.obs, CLUSTER_COL, RARE_CLUSTER_THRESHOLD
        )
        rare_neighbors_retained, n_rare_neighbors, _ = rare_label_retention(
            adata_ref.obs, ad.obs, NEIGHBOR_COL, RARE_NEIGHBOR_THRESHOLD
        )

        row = {
            "candidate": name,
            "n_bins": ad.n_obs,
            "cluster_l1_distance": l1_distance(ref_cluster_props, samp_cluster_props),
            "neighbor_l1_distance": l1_distance(ref_neighbor_props, samp_neighbor_props),
            "cluster_x_neighbor_l1_distance": l1_distance(ref_joint_props, samp_joint_props),
            "spatial_grid_coverage": grid_coverage_score(ref_coords, coords, GRID_N_X, GRID_N_Y),
            "rare_clusters_retained": rare_clusters_retained,
            "n_rare_clusters": n_rare_clusters,
            "rare_neighbors_retained": rare_neighbors_retained,
            "n_rare_neighbors": n_rare_neighbors,
            **nearest_neighbor_distance_summary(coords, KNN_K),
        }

        for qc in QC_COLS:
            if qc in adata_ref.obs.columns and qc in ad.obs.columns:
                row[f"median_rel_diff_{qc}"] = median_relative_difference(
                    pd.to_numeric(adata_ref.obs[qc], errors="coerce").to_numpy(),
                    pd.to_numeric(ad.obs[qc], errors="coerce").to_numpy(),
                )

        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTDIR / "sampling_validation_metrics.tsv", sep="\t", index=False)

    print("[4/7] Saving proportion tables and plots...")
    cluster_prop_table = pd.DataFrame({"full": ref_cluster_props})
    neighbor_prop_table = pd.DataFrame({"full": ref_neighbor_props})

    for name, ad in sampled_dict.items():
        cluster_prop_table[name] = proportions(ad.obs, CLUSTER_COL)
        neighbor_prop_table[name] = proportions(ad.obs, NEIGHBOR_COL)

    cluster_prop_table = cluster_prop_table.fillna(0)
    neighbor_prop_table = neighbor_prop_table.fillna(0)

    cluster_prop_table.to_csv(OUTDIR / "cluster_proportion_comparison.tsv", sep="\t")
    neighbor_prop_table.to_csv(OUTDIR / "neighbor_proportion_comparison.tsv", sep="\t")

    make_proportion_plot(
        cluster_prop_table,
        "Cluster proportion comparison",
        CLUSTER_COL,
        OUTDIR / "cluster_proportion_comparison.png",
        log=False,
    )
    make_proportion_plot(
        cluster_prop_table,
        "Cluster proportion comparison (log scale)",
        CLUSTER_COL,
        OUTDIR / "cluster_proportion_logscale.png",
        log=True,
    )
    make_proportion_plot(
        neighbor_prop_table,
        f"{NEIGHBOR_COL} proportion comparison",
        NEIGHBOR_COL,
        OUTDIR / "neighbor_proportion_comparison.png",
        log=False,
    )

    print("[5/7] Saving spatial and QC plots...")
    make_spatial_coverage_plot(ref_coords, sampled_dict, OUTDIR / "spatial_sampling_coverage.png")
    make_qc_boxplot(adata_ref, sampled_dict, OUTDIR / "qc_distribution_comparison.png")
    make_metrics_barplot(metrics_df, OUTDIR / "metrics_barplot.png")

    print("[6/7] Candidate ranking...")
    rec_df = metrics_df.copy()

    qc_rank_cols = []
    for qc in QC_COLS:
        col = f"median_rel_diff_{qc}"
        if col in rec_df.columns:
            rank_col = f"rank_{col}"
            rec_df[rank_col] = rec_df[col].rank(method="min")
            qc_rank_cols.append(rank_col)

    rec_df["rank_cluster"] = rec_df["cluster_l1_distance"].rank(method="min")
    rec_df["rank_neighbor"] = rec_df["neighbor_l1_distance"].rank(method="min")
    rec_df["rank_joint"] = rec_df["cluster_x_neighbor_l1_distance"].rank(method="min")
    rec_df["rank_spatial"] = (-rec_df["spatial_grid_coverage"]).rank(method="min")
    rec_df["rank_nn"] = rec_df["median_nn_distance"].rank(method="min")

    if len(qc_rank_cols) > 0:
        rec_df["rank_qc_mean"] = rec_df[qc_rank_cols].mean(axis=1)
    else:
        rec_df["rank_qc_mean"] = 0.0

    # Main ranking prioritizes preservation of cluster x neighborhood composition,
    # then neighborhood and cluster separately, with spatial coverage as a secondary criterion.
    rec_df["main_combined_rank"] = (
        0.40 * rec_df["rank_joint"]
        + 0.25 * rec_df["rank_neighbor"]
        + 0.20 * rec_df["rank_cluster"]
        + 0.10 * rec_df["rank_spatial"]
        + 0.05 * rec_df["rank_qc_mean"]
    )

    best_main = rec_df.sort_values("main_combined_rank").iloc[0]
    best_spatial = rec_df.sort_values("rank_spatial").iloc[0]

    rec_df.to_csv(OUTDIR / "sampling_candidate_ranking.tsv", sep="\t", index=False)

    print("[7/7] Writing summary...")
    with (OUTDIR / "sampling_validation_summary.txt").open("w") as f:
        f.write("SAMPLING VALIDATION SUMMARY\n")
        f.write("===========================\n\n")
        f.write(f"Reference AnnData: {REFERENCE_ADATA_PATH}\n")
        f.write(f"Reference shape: {adata_ref.shape}\n")
        f.write(f"Cluster column: {CLUSTER_COL}\n")
        f.write(f"Neighbor column: {NEIGHBOR_COL}\n")
        f.write(f"Rare cluster threshold: {RARE_CLUSTER_THRESHOLD}\n")
        f.write(f"Rare neighbor threshold: {RARE_NEIGHBOR_THRESHOLD}\n")
        f.write(f"Spatial grid: {GRID_N_X} x {GRID_N_Y}\n")
        f.write(f"K for nearest-neighbor distance: {KNN_K}\n\n")

        f.write("Candidates evaluated:\n")
        for name, path in CANDIDATES.items():
            f.write(f"  - {name}: {path}\n")

        f.write("\nMetric interpretation:\n")
        f.write("  - cluster_l1_distance: lower is better\n")
        f.write("  - neighbor_l1_distance: lower is better\n")
        f.write("  - cluster_x_neighbor_l1_distance: lower is better\n")
        f.write("  - spatial_grid_coverage: higher is better\n")
        f.write("  - median_nn_distance: lower means sampled bins remain spatially closer\n")
        f.write("  - rare_clusters_retained: higher is better\n")
        f.write("  - rare_neighbors_retained: higher is better\n")
        f.write("  - QC median relative differences: lower is better\n\n")

        f.write("Validation metrics:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        f.write("Main recommendation for GRN comparison:\n")
        f.write(f"  Best candidate: {best_main['candidate']}\n")
        f.write(
            "  This ranking prioritizes preservation of the joint cluster x neighborhood composition,\n"
        )
        f.write(
            "  which is the most relevant sampling target for downstream cluster x neighborhood comparisons.\n\n"
        )

        f.write("Best pure spatial-coverage candidate:\n")
        f.write(f"  {best_spatial['candidate']}\n\n")

        f.write("Important interpretation:\n")
        f.write(
            "  For D0/D1 GRN comparison, the same sampled bin IDs must be applied across depths.\n"
        )
        f.write(
            "  If cluster_x_neighbor is selected as the main method, cluster-only sampling can still be retained\n"
        )
        f.write(
            "  as a simpler sensitivity analysis, while grid sampling can be retained as a spatial-coverage control.\n"
        )

    print(f"[DONE] Results saved in: {OUTDIR}")


if __name__ == "__main__":
    main()
