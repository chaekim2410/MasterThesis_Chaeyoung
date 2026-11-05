# 00 · Data preparation and annotation

This stage turns the 10x Genomics Visium HD mouse brain Space Ranger output into a quality-controlled AnnData object with cell-type and broad-region labels. The Python files contain the analysis; the `.pbs` files record how jobs were submitted on the Ghent University HPC.

| File | Main job |
| --- | --- |
| [`00_prefilter_qc.py`](00_prefilter_qc.py) | Read the Space Ranger feature matrix, tissue coordinates, and vendor clusters; calculate and visualize quality-control measures before filtering. |
| [`01_apply_filtering.py`](01_apply_filtering.py) | Apply bin and gene quality thresholds, compare retained bins and clusters, and save the filtered AnnData object. |
| [`02a_celltype_annotation.py`](02a_celltype_annotation.py) | Join bin-level MapMyCells predictions to the filtered data, examine prediction confidence, and mark low-confidence classes unresolved. |
| [`02b_add_domain_labels.py`](02b_add_domain_labels.py) | Convert class labels to Allen transcriptomic Neighborhood and neurotransmitter labels, plus a coarse region proxy. The variable `domain_from_bin_class` is a Neighborhood label, not a spatial domain inferred from coordinates. |
| [`02c_validate_domain_labels.py`](02c_validate_domain_labels.py) | Check the coverage and spatial coherence of the derived labels and produce diagnostic plots. |

**Data and requests.** The input is the public [10x Visium HD mouse brain dataset](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-mouse-brain-fresh-frozen), specifically its Space Ranger outputs. `02a` reads a local CSV of [Allen MapMyCells](https://brain-map.org/our-research/cell-type-taxonomies/whole-mouse-brain-taxonomy) predictions using taxonomy CCN20230722; it does not submit a MapMyCells request itself. `02b` reads a local class-to-Neighborhood/neurotransmitter lookup table. These input files are external to this repository.

**Software and running.** Scanpy/AnnData, pandas, NumPy, SciPy, Matplotlib, and scikit-learn support this stage. [`00_prefilter_qc.pbs`](00_prefilter_qc.pbs), [`01_apply_filtering.pbs`](01_apply_filtering.pbs), and [`02_annotation.pbs`](02_annotation.pbs) are historical job scripts. Their project paths and script locations refer to the original HPC layout; update them for this folder structure before submitting jobs.
