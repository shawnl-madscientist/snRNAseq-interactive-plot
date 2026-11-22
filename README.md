# snRNAseq-interactive-plot
python script to process SoupX-corrected raw data from 148 722-cell by 21 765-gene matrix and visualize data interactively.
Repo includes python3 script, environment file with required libraries and dependencies, and exported plots
# zf_viz: interactive explorer for zebra finch snRNA-seq

Filter by sex, condition, age, gene(s) and cell type, and view per-animal bars, dot plots or a UMAP.

## Requirements
- Python >=3.10 (tested on 3.14)
- Packages: `h5py`, `numpy`, `pandas`, `plotly`, `dash` (see `requirements.txt`)
- The `.h5ad` data file (not included; ~1.1 GB). It must be an AnnData file with
  `obs` columns `sex`, `condition`, `age`, `anno_v4`, `lineage`, `orig.ident`,
  `obsm/X_umap`, and raw counts in `X` (cells x genes, CSC format).

`anndata`/`scanpy` are NOT needed.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ZF_H5AD="/path/to/ZF_RNA_final_browser_raw_counts.h5ad"
```

## Run
```bash
python zf_viz.py                                         # interactive app at http://127.0.0.1:8050
python zf_viz.py --export MEF2C MEF2A --sex MALE         # static HTML files, no server
```
The first launch takes a few minutes to compute per-cell totals, then caches them in `zf_cell_totals.npy`.

## Method
- Normalization: log(1 + counts / cell total x 10,000).
- Bar plot: counts summed per animal x cell type (pseudobulk), then normalized; dot = animal, bar = mean.
  Animal x cell-type combinations with fewer than 20 cells are dropped.
- Dot plot: dot size = % of cells with expression > 0, colour = mean normalized expression.