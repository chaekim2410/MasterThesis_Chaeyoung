## Repository contents

The six folders below follow the analysis from raw spatial data to validation. Each folder explains its Python files, PBS jobs, required inputs, and software. Folder numbers label stages; the original script numbers are retained inside them.

| Analysis group | Files | Purpose |
| --- | --- | --- |
| 00 · Preparation | [`pipeline/00_preprocessing_annotation/`](pipeline/00_preprocessing_annotation/) | Quality control, filtering, and brain cell-type labels |
| 01 · Sampling | [`pipeline/01_sampling_inputs/`](pipeline/01_sampling_inputs/) | Choose and check 30,000 bins; prepare common inputs |
| 02 · GRNBoost2 | [`pipeline/02_grnboost2/`](pipeline/02_grnboost2/) | Infer expression-only networks and score regulons |
| 03 · SpaGRN | [`pipeline/03_spaGRN/`](pipeline/03_spaGRN/) | Infer spatial networks and score regulons |
| 04 · SpaGTL | [`pipeline/04_spaGTL/`](pipeline/04_spaGTL/) | Fine-tune the spatial model and score regulons |
| 05 · Validation | [`pipeline/05_validation/`](pipeline/05_validation/) | Compare repeatability, independent evidence, and tissue patterns |

Notebook outputs were cleared to keep the files small enough for GitHub to render. The analysis code and explanatory markdown are unchanged; the main result figures are shown above. Each notebook currently defines an absolute `BASE` path near the top. Change that path to the local analysis directory before running it.
