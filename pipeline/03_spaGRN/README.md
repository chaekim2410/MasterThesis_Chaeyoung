# 03 · SpaGRN

SpaGRN uses expression together with bin coordinates to infer candidate regulatory relationships. Its output enters the same motif and AUCell framework used for comparison with the other methods.

| File | Main job |
| --- | --- |
| [`07a_spagrn_core.py`](07a_spagrn_core.py) | Load the sampled AnnData object and TF list, run SpaGRN inference, inspect reusable intermediate files, and save network outputs. |
| [`07_spaGRN_chunk.py`](07_spaGRN_chunk.py) | An alternate SpaGRN inference script retained for provenance. It differs from `07a_spagrn_core.py` in the `local` setting passed to inference (`False` versus `True`); it is not an additional comparison method. |

[`07a_spagrn.pbs`](07a_spagrn.pbs) records the inference job and checks its environment. [`07b_spagrn_ctx_aucell.pbs`](07b_spagrn_ctx_aucell.pbs) runs pySCENIC motif pruning and AUCell on the SpaGRN outputs.

**Data and software.** Inputs are the shared 30,000-bin AnnData sample and mouse TF list from earlier stages. SpaGRN and pySCENIC use local cisTarget `.feather` ranking databases and motif-to-TF annotations. The scripts do not download those files; they expect them under the original project's `external_data/`. Key packages include `spagrn`, `hotspot`, pySCENIC, `pyarrow`, Scanpy/AnnData, NumPy, and SciPy. The PBS command calls `07a_spaGRN_core.py`, whereas the tracked file is named `07a_spagrn_core.py`; fix the capitalization and original HPC paths before running it from this repository.
