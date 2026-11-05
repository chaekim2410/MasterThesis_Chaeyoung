# 02 · GRNBoost2: expression-only reference

GRNBoost2 predicts transcription factor–target relationships from expression without spatial coordinates. The large target set was split into chunks for HPC runs; the resulting edges were checked and merged before motif-based regulon scoring.

| File | Main job |
| --- | --- |
| [`06a_grnboost2_chunk_test.py`](06a_grnboost2_chunk_test.py) | Pilot the chunk strategy across seeds and compare chunked with unchunked results using overlap and ranking diagnostics. |
| [`06a_grnboost2_chunked.py`](06a_grnboost2_chunked.py) | Run one target-gene chunk with Arboreto/GRNBoost2 and Dask; write ranked TF–target edges for that chunk. |
| [`06b_check_chunked_grnboost2.py`](06b_check_chunked_grnboost2.py) | Check that all expected chunks and outputs exist before merging. |
| [`06c_merge_chunk_grnboost2.py`](06c_merge_chunk_grnboost2.py) | Combine chunk outputs into a single ranked adjacency table. |
| [`06e_regulon_occurrence_analysis.py`](06e_regulon_occurrence_analysis.py) | Parse pySCENIC regulons across repeated runs and summarize how often TFs and their targets recur. |

**Other steps.** [`06a_grnboost2.pbs`](06a_grnboost2.pbs) submits chunk inference, [`06a_grnboost2_chunk_test.pbs`](06a_grnboost2_chunk_test.pbs) submits the pilot, and [`06bc_check_merge.pbs`](06bc_check_merge.pbs) checks and merges chunks. [`06d_grnboost2_ctx_auc.pbs`](06d_grnboost2_ctx_auc.pbs) uses pySCENIC command-line tools for motif pruning (`ctx`) and regulon activity (`aucell`); its saved configuration runs AUCell and assumes regulon files already exist. [`06e_regulon_occurrence_analysis.pbs`](06e_regulon_occurrence_analysis.pbs) submits the recurrence summary.

**Data and software.** The input is the stage 01 sampled expression data plus the local mouse TF list. `ctx` uses local Aerts Lab [cisTarget motif-ranking databases and motif annotations](https://resources.aertslab.org/cistarget/databases/mus_musculus/mm10/refseq_r80/). Arboreto, Dask, pySCENIC, Scanpy, pandas, NumPy, and SciPy are the main packages. The scripts make no database download requests. The PBS files contain original HPC paths; in particular, the `06e` PBS script names an older `07_regulon_occurrence_analysis.py` path and must be pointed to the file in this folder before reuse.
