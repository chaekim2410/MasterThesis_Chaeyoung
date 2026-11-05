# 05 · Reproducibility and validation

These are the three final analysis notebooks. They ask whether repeated runs agree, whether inferred relationships have independent biological support, and whether regulon activity forms interpretable tissue patterns. The notebook outputs were cleared for GitHub; the source cells were retained.

| Notebook | Main job |
| --- | --- |
| [`reproducibility_final.ipynb`](reproducibility_final.ipynb) | Compare GRNBoost2 and SpaGTL across ten runs, including network size, ranked-edge overlap, and recurrent TF targets. |
| [`ground_validation_final_final2.ipynb`](ground_validation_final_final2.ipynb) | Test ranked edges against curated relationships, TF perturbation responses, and ChIP-seq binding; examine GO biological-process enrichment. |
| [`spatial_broad_region.ipynb`](spatial_broad_region.ipynb) | Examine spatial coherence and similarity of regulon maps and compare broad-region activity with an Allen ISH reference matrix. |

## Where the validation data came from

The validation inputs came from the public resources below. The table states the source and the acquisition route supported by the notebooks and file metadata. The notebooks use the resulting files from the thesis data directory; those large files are not copied into this repository.

| Resource | Source and acquisition route | Use in the final notebook |
| --- | --- | --- |
| [CollecTRI via OmniPath](https://omnipathdb.org/) | Mouse-symbol TF–target relationships were obtained from the CollecTRI resource exposed through OmniPath and saved as a CSV. The final notebooks preserve the processed file and source link, but not the original download command. | Restrict curated TF–target pairs to measured genes and eligible TFs; remove duplicates and self-loops. |
| [KnockTF 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10767813/) | The mouse DEG table was downloaded from the KnockTF v2 download endpoint recorded in the file metadata: [`knocktf_v2_main_mouse.txt`](http://www.licpathway.net:8081/KnockTFv2/public/download_anno/knocktf_v2_main_mouse.txt). The final notebook reads that downloaded table. A separate notebook works from individual `DataSet_11_*.txt` files. | Read the large DEG table in chunks and retain the specified response direction with corrected p-value < 0.05, then restrict to the analysis gene/TF universe and remove self-loops. |
| [ChIP-Atlas Target Genes](https://chip-atlas.org/target_genes) | The notebook sends one HTTP `GET` request per TF to `https://chip-atlas.dbcls.jp/data/mm10/target/{tf}.5.tsv`, then saves the returned target-gene table as a TF-specific cache CSV. Existing cache files are skipped and unavailable TFs are recorded. | Apply a TF-specific score cutoff to the cached TF/target/average-score tables and restrict them to the analysis universe. |
| [g:Profiler](https://biit.cs.ut.ee/gprofiler/gost) | The notebook calls the g:Profiler web service through the `gprofiler` Python client (`GProfiler(return_dataframe=True)`); this is a live enrichment request rather than a downloaded database file. | Interpret high-ranking consensus regulons using the measured gene panel as background. |
| [Allen Mouse Brain Atlas ISH](https://brain-map.org/support/documentation/documentation-allen-mouse-brain-atlas) | Allen regional expression was retrieved through the Allen API endpoint `http://api.brain-map.org/api/v2/data/query.json` in the acquisition notebooks, then saved as a matrix for the final spatial comparison notebook. | Compare broad anatomical expression with regulon activity, without treating this as TF–target edge validation. |

The distinction matters: the source and request route above describe how each resource enters the analysis; the repository contains the analysis code and documentation, while the large downloaded data files remain external.

## Re-running

Change the absolute `BASE`/input paths near the start of each notebook, provide the external files listed above and the upstream network outputs, and install the notebook imports. Main packages include pandas, NumPy, SciPy, scikit-learn, Scanpy/AnnData, Matplotlib, Seaborn, `requests`, `kneed`, and `gprofiler-official` where used. Large source and derived data files are intentionally outside the Git repository.
