#!/usr/bin/env python3
"""
02b_add_domain_labels.py  (REVISED v2, minimal)

Purpose
-------
Map the QC'd MapMyCells bin-level class (mmc_class_qc, from 02a) to:
  - domain_from_bin_class           (Class -> Neighbor lookup)
  - neurotransmitter_from_bin_class (Class -> Neurotransmitter lookup)
  - region_proxy_from_bin_class     (Class -> coarse anatomical region,
                                      figure-derived, NOT atlas-registered)

Column names are kept EXACTLY as in the original pipeline on purpose --
downstream analyses already reference these names.

Naming note (do not confuse with the column name): "domain_from_bin_class"
is really an Allen Brain Cell Atlas transcriptomic "Neighborhood" grouping,
not an algorithmically-detected spatial domain. Describe it that way in
methods/figures even though the column name stays "domain_from_bin_class".

What's different vs. the previous version:
  - Bin-level only (no separate cluster-level mapping path); input class is
    already confidence-QC'd in 02a.
  - The FINAL saved h5ad (adata_labeled.h5ad) is trimmed down to keep ONLY
    the three label columns above (plus whatever was already in the object
    before 02a/02b touched it). The intermediate mmc_class_top1/
    mmc_class_conf/mmc_class_qc columns are dropped at the very end, since
    they were only needed to derive these three labels.
"""

from pathlib import Path
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


# =========================================================
# USER SETTINGS
# =========================================================
INPUT_H5AD = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/results/02_annotation/cell_type_annotation/adata_annotated.h5ad"
)
CLASS_NEIGHBOR_CSV = Path(
    "/data/gent/vo/000/gvo00027/CBIGR/25KCY_master_thesis/external_data/class_neighbor_neurotransmitter.csv"
)

OUTDIR = Path("results/02_annotation/domain_labels")
OUTDIR.mkdir(parents=True, exist_ok=True)

CLASS_COL = "mmc_class_qc"   # confidence-QC'd in 02a
LIBRARY_ID = None
POINT_SIZE = 1.1
SPATIAL_DPI = 220

UNRESOLVED = "Unresolved"
NA_NT = "NA"
NA_REGION = "NA_region"

OUT_H5AD = OUTDIR / "adata_labeled.h5ad"

# Final obs columns kept in the downstream-facing h5ad.
# Names are unchanged from the original pipeline on purpose.
FINAL_LABEL_COLS = [
    "domain_from_bin_class",
    "neurotransmitter_from_bin_class",
    "region_proxy_from_bin_class",
]

# Figure-derived coarse region proxy (NOT atlas registration).
CLASS_TO_REGION_PROXY = {
    "01 IT-ET Glut": "Isocortex",
    "02 NP-CT-L6P Glut": "CTXsp",
    "02 NP-CT-L6b Glut": "CTXsp",
    "03 OB-CR Glut": "OLF",
    "04 DG-IMN Glut": "HPF",

    "05 OB-IMN GABA": "OLF",
    "06 CTX-CGE GABA": "Isocortex",
    "07 CTX-MGE GABA": "Isocortex",
    "08 CNU-MGE GABA": "PAL",
    "09 CNU-LGE GABA": "STR",
    "10 LSX GABA": "PAL",

    "11 CNU-HYa GABA": "HY",
    "12 HY GABA": "HY",
    "13 CNU-HYa Glut": "HY",
    "14 HY Glut": "HY",
    "15 HY Gnrh1 Glut": "HY",
    "16 HY MM Glut": "HY",

    "17 MH-LH Glut": "TH",
    "18 TH Glut": "TH",

    "19 MB Glut": "MB",
    "20 MB GABA": "MB",
    "21 MB Dopa": "MB",
    "22 MB-HB Sero": "Pons",
    "23 P Glut": "Pons",
    "24 MY Glut": "MY",
    "25 Pineal Glut": "MB",
    "26 P GABA": "Pons",
    "27 MY GABA": "MY",
    "28 CB GABA": "CB",
    "29 CB Glut": "CB",

    "30 Astro-Epen": NA_REGION,
    "31 OPC-Oligo": NA_REGION,
    "32 OEC": "OLF",
    "33 Vascular": NA_REGION,
    "34 Immune": NA_REGION,
}


# =========================================================
# HELPERS
# =========================================================
def get_library_id(adata, user_library_id=None):
    if user_library_id is not None:
        return user_library_id
    libs = list(adata.uns.get("spatial", {}).keys())
    if not libs:
        raise ValueError("No spatial library found in adata.uns['spatial'].")
    return libs[0]


def choose_img_key(adata, library_id):
    imgs = adata.uns["spatial"][library_id]["images"]
    return "hires" if "hires" in imgs else list(imgs.keys())[0]


def save_spatial_plot(adata, library_id, color_key, out_png):
    if color_key not in adata.obs.columns:
        print(f"[skip] {color_key} not found")
        return
    if pd.api.types.is_categorical_dtype(adata.obs[color_key]):
        adata.obs[color_key] = adata.obs[color_key].cat.remove_unused_categories()
    img_key = choose_img_key(adata, library_id)
    fig = sc.pl.spatial(
        adata, color=color_key, library_id=library_id, img_key=img_key,
        size=POINT_SIZE, show=False, return_fig=True,
    )
    fig.savefig(out_png, dpi=SPATIAL_DPI, bbox_inches="tight")
    plt.close(fig)


def normalize_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    return " ".join(s.replace("\u2013", "-").replace("\u2014", "-").split())


def load_class_neighbor_table(path):
    if not path.exists():
        raise FileNotFoundError(f"class_neighbor.csv not found: {path}")
    df = pd.read_csv(path)
    required = {"Class", "Neighbor", "Neurotransmitter"}
    if not required.issubset(df.columns):
        raise ValueError(f"class_neighbor.csv must contain {required}. Found: {list(df.columns)}")
    df["Class"] = df["Class"].map(normalize_label)
    df["Neighbor"] = df["Neighbor"].map(normalize_label)
    df["Neurotransmitter"] = df["Neurotransmitter"].map(normalize_label)
    df = df.dropna(subset=["Class", "Neighbor"]).drop_duplicates(subset=["Class"], keep="first")
    return df


def map_with_unresolved(class_col, lookup_dict, fill_value):
    """Map a class column through lookup_dict; Unresolved class stays Unresolved."""
    class_norm = class_col.astype(str).map(normalize_label)
    is_unresolved = class_col.astype(str) == UNRESOLVED
    mapped = class_norm.map(lookup_dict)
    mapped = mapped.where(~is_unresolved, UNRESOLVED)
    mapped = mapped.fillna(fill_value)
    return mapped.astype("category"), class_norm


# =========================================================
# MAIN
# =========================================================
def main():
    print("Reading annotated h5ad...")
    adata = sc.read_h5ad(INPUT_H5AD)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if CLASS_COL not in adata.obs.columns:
        raise ValueError(f"{CLASS_COL} not found in adata.obs. Run 02a first.")

    library_id = get_library_id(adata, LIBRARY_ID)

    print("Reading class_neighbor_neurotransmitter.csv...")
    mapping_df = load_class_neighbor_table(CLASS_NEIGHBOR_CSV)
    class_to_domain = dict(zip(mapping_df["Class"], mapping_df["Neighbor"]))
    class_to_nt = dict(zip(mapping_df["Class"], mapping_df["Neurotransmitter"]))

    print("Mapping bin-level class -> domain / neurotransmitter / region_proxy...")
    adata.obs["domain_from_bin_class"], class_norm = map_with_unresolved(
        adata.obs[CLASS_COL], class_to_domain, UNRESOLVED
    )
    adata.obs["neurotransmitter_from_bin_class"], _ = map_with_unresolved(
        adata.obs[CLASS_COL], class_to_nt, NA_NT
    )
    adata.obs["region_proxy_from_bin_class"], _ = map_with_unresolved(
        adata.obs[CLASS_COL], CLASS_TO_REGION_PROXY, NA_REGION
    )

    unmatched = sorted(
        set(class_norm.dropna()) - set(class_to_domain.keys()) - {UNRESOLVED}
    )
    if unmatched:
        pd.DataFrame({"unmatched_class": unmatched}).to_csv(
            OUTDIR / "unmatched_classes.csv", index=False
        )
        print(f"[WARN] {len(unmatched)} unmatched classes -> see unmatched_classes.csv")

    # -----------------------------------------------------
    # Quick sanity-check outputs (cheap: counts + one spatial plot each)
    # -----------------------------------------------------
    print("Saving composition tables...")
    for col in FINAL_LABEL_COLS:
        adata.obs[col].value_counts().to_csv(OUTDIR / f"{col}_counts.csv")

    print("Saving quick spatial sanity plots...")
    for col in FINAL_LABEL_COLS:
        save_spatial_plot(adata, library_id, col, OUTDIR / f"{col}_spatial.png")

    # -----------------------------------------------------
    # Trim down to the final, downstream-facing set of obs columns.
    # Drops mmc_class_top1 / mmc_class_conf / mmc_class_qc (intermediate,
    # only needed to derive the three labels above) and any other column
    # added during 02a (e.g. diagnostic crosstab inputs).
    # =====================================================
    drop_cols = ["mmc_class_top1", "mmc_class_conf", CLASS_COL]
    drop_cols = [c for c in drop_cols if c in adata.obs.columns]
    print(f"Dropping intermediate columns before final save: {drop_cols}")
    adata.obs = adata.obs.drop(columns=drop_cols)

    summary = [
        "=== 02b Domain / neurotransmitter / region_proxy labeling (bin-level) ===",
        f"Input h5ad: {INPUT_H5AD}",
        f"Class-neighbor CSV: {CLASS_NEIGHBOR_CSV}",
        f"Class column used as input: {CLASS_COL} (confidence-QC'd in 02a, dropped from final h5ad)",
        "",
        "Naming note: 'domain_from_bin_class' = Allen Brain Cell Atlas transcriptomic",
        "Neighborhood grouping, NOT an algorithmically-detected spatial domain.",
        "Column names kept unchanged from the original pipeline for downstream compatibility.",
        "",
        f"Final obs columns added: {FINAL_LABEL_COLS}",
        "All other intermediate annotation columns were dropped before saving.",
        "",
        "Output files:",
        *[f"  - {c}_counts.csv / {c}_spatial.png" for c in FINAL_LABEL_COLS],
        "  - unmatched_classes.csv (if any)",
        "  - adata_labeled.h5ad",
    ]
    (OUTDIR / "summary_02b_domain_labels.txt").write_text("\n".join(summary))

    print("Saving final (trimmed) h5ad...")
    adata.write_h5ad(OUT_H5AD, compression="gzip")
    print(f"[DONE] {OUT_H5AD}")
    print(f"Final obs columns: {list(adata.obs.columns)}")


if __name__ == "__main__":
    main()