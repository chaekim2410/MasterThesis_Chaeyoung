# 04 · SpaGTL

SpaGTL combines expression, spatial information, and pretrained model weights. This stage runs the model with ten seeds, extracts TF–target edge scores, and applies motif pruning and AUCell to the resulting networks.

| File | Main job |
| --- | --- |
| [`08a_spagtl_core.py`](08a_spagtl_core.py) | Prepare the sampled data, align genes with pretrained weights, train/fine-tune SpaGTL for each run, and export scored relationships and latent representations. |
| [`08b_spagtl_ctx_auc.py`](08b_spagtl_ctx_auc.py) | Convert ranked SpaGTL edges into pySCENIC modules, prune with cisTarget motifs, score regulon activity with AUCell, and save per-run outputs. |

[`08a_spagtl.pbs`](08a_spagtl.pbs) requests an NVIDIA GPU and CUDA 12.1.1 for the model stage. [`08b_spagtl_ctx_auc.pbs`](08b_spagtl_ctx_auc.pbs) submits the downstream scoring stage.

**Data and software.** In addition to the shared sample and mouse TF list, `08a` requires a local pretrained `params_dict.npy` file. `08b` reads local cisTarget `.feather` rankings and motif annotations. These scripts do not fetch the weights or databases over the network. Key packages include SpaGTL, PyTorch, Scanpy/AnnData, Arboreto, `ctxcore`, pySCENIC, pandas, NumPy, and SciPy. The HPC-specific absolute paths and Conda environment in the PBS files must be updated before reuse.
