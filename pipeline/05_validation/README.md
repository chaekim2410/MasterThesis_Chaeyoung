# 05 · Reproducibility and validation

These are the three final analysis notebooks. They ask whether repeated runs agree, whether inferred relationships have independent biological support, and whether regulon activity forms interpretable tissue patterns. The notebook outputs were cleared for GitHub; the source cells were retained.

| Notebook | Main job |
| --- | --- |
| [`reproducibility_final.ipynb`](reproducibility_final.ipynb) | Compare GRNBoost2 and SpaGTL across ten runs, including network size, ranked-edge overlap, and recurrent TF targets. |
| [`ground_validation_final_final2.ipynb`](ground_validation_final_final2.ipynb) | Test ranked edges against curated relationships, TF perturbation responses, and ChIP-seq binding; examine GO biological-process enrichment. |
| [`spatial_broad_region.ipynb`](spatial_broad_region.ipynb) | Examine spatial coherence and similarity of regulon maps and compare broad-region activity with an Allen ISH reference matrix. |

## Where the validation data came from

The notebooks read mostly local files under `/Users/kimchaeyoung/Desktop/thesis_data`, which are not part of this repository. The table distinguishes the original resource from the acquisition step visible in the final notebook. File metadata or source code cannot establish who personally initiated a download.

| Resource | Acquisition visible in the checked files | Use in the final notebook |
| --- | --- | --- |
| [CollecTRI via OmniPath](https://omnipathdb.org/) | Reads `external_data/collectri_mouse_genesymbol_clean.csv`. The checked notebooks and scripts do not show the download or mouse orthology-conversion command that produced this CSV. | Restrict curated TF–target pairs to measured genes and eligible TFs; remove duplicates and self-loops. |
| [KnockTF 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10767813/) | Reads the local `perturb/differential expression of genes in all datasets(Mouse).txt` file. Its macOS download metadata records a KnockTF v2 URL and Chrome as the downloading application, but does not identify the person or agent who initiated it. A separate local exploratory notebook analyzes `perturb/knocktf2/DataSet_11_*.txt` experiment files; those are not the final notebook's DEG input. | Read the large DEG table in chunks and retain the specified response direction with corrected p-value < 0.05, then restrict to the analysis gene/TF universe and remove self-loops. |
| [ChIP-Atlas Target Genes](https://chip-atlas.org/target_genes) | The final notebook contains `requests.get()` calls to `https://chip-atlas.dbcls.jp/data/mm10/target/{tf}.5.tsv`, one TF at a time; it skips existing cache CSVs and records unavailable TFs. The saved outputs were cleared, so this copy does not prove which requests executed in the final run. | Save TF/target/average-score tables in `external_data/chipatlas_mm10_5kb_cache/`; apply a TF-specific score cutoff and restrict to the analysis universe. |
| [g:Profiler](https://biit.cs.ut.ee/gprofiler/gost) | The final notebook uses the g:Profiler Python client for GO biological-process queries. | Interpret high-ranking consensus regulons using the measured gene panel as background. |
| [Allen Mouse Brain Atlas ISH](https://brain-map.org/support/documentation/documentation-allen-mouse-brain-atlas) | `spatial_broad_region.ipynb` reads a previously saved Allen matrix. Other local `thesis_data` notebooks contain `requests.get()` calls to the Allen API that generated such a matrix; those acquisition notebooks are not among the three published here. | Compare broad anatomical expression with regulon activity, without treating this as TF–target edge validation. |

The distinction matters: a script that contains a request shows how data *can* be retrieved, while a local input file shows what the analysis *used*. The reviewed files do not identify whether the researcher, Claude Code, or someone else initiated each earlier download.

## Re-running

Change the absolute `BASE`/input paths near the start of each notebook, provide the external files listed above and the upstream network outputs, and install the notebook imports. Main packages include pandas, NumPy, SciPy, scikit-learn, Scanpy/AnnData, Matplotlib, Seaborn, `requests`, `kneed`, and `gprofiler-official` where used. Large source and derived data files are intentionally outside the Git repository.
