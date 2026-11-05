# 01 · Sampling and shared inputs

Full-resolution inference on every retained bin was computationally expensive. This stage compares ways to select 30,000 bins, checks whether the chosen sample retains the tissue's main properties, and prepares the same expression and TF inputs for the inference methods.

| File | Main job |
| --- | --- |
| [`04a_make_sampling_bin_ids.py`](04a_make_sampling_bin_ids.py) | Generate candidate bin ID lists using random, cluster-stratified, cluster-by-Neighborhood, spatial-grid, and geometric-sketch strategies. |
| [`04b_apply_sampling_bin_ids.py`](04b_apply_sampling_bin_ids.py) | Subset the annotated AnnData object by the candidate ID lists and write sampled AnnData files. |
| [`04c_validate_sampling_candidates.py`](04c_validate_sampling_candidates.py) | Compare candidates against the full filtered dataset for cluster/label composition, rare-group retention, spatial coverage, and QC distributions. |
| [`05_prepare_input.py`](05_prepare_input.py) | Read the selected 30,000-bin sample, match the mouse TF list to measured genes, and export expression inputs for downstream GRN inference. |

**Data and requests.** Inputs come from stage 00 and a local mouse transcription-factor list from the Aerts Lab [cisTarget resources](https://resources.aertslab.org/cistarget/tf_lists/). These scripts read local files; they contain no remote download request. Large `.h5ad` and expression outputs are not included in the repository.

**Software and running.** Scanpy/AnnData, pandas, NumPy, SciPy, scikit-learn, and Matplotlib are used here. [`04a_make_sampling_bin_ids.pbs`](04a_make_sampling_bin_ids.pbs), [`04bc_apply_sampling.pbs`](04bc_apply_sampling.pbs), and [`05_prepare_input.pbs`](05_prepare_input.pbs) preserve the original HPC commands; paths in them need updating for the current folder layout.
