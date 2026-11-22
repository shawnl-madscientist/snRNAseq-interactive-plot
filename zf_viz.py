#!/usr/bin/env python3
## ============================================================================
## zf_viz.py  -  Interactive explorer for the zebra finch snRNA-seq dataset
## Run:      python zf_viz.py                       (opens a web page in your browser)
## Export:   python zf_viz.py --export MEF2C --sex MALE   (saves HTML files, no server)
## Needs:    pip install h5py numpy pandas plotly dash
## ============================================================================

## ---------------------------------------------------------------- 1. IMPORTS
import argparse                                   ## lets us read options typed after "python zf_viz.py" (like --export)
import os                                         ## lets us build file paths that work on Mac/Windows
from functools import lru_cache                   ## remembers results of a function so we never compute the same gene twice

import h5py                                       ## reads .h5ad files directly (the file format is HDF5 underneath)
import numpy as np                                ## fast math on big arrays of numbers
import pandas as pd                               ## spreadsheet-like tables (DataFrames)
import plotly.express as px                       ## quick interactive charts
import plotly.graph_objects as go                 ## lower-level plotly pieces (bars + dots in one chart)
from plotly.subplots import make_subplots         ## lets us tile many small charts in a grid

## ---------------------------------------------------------------- 2. SETTINGS (edit these)
H5AD_PATH = os.environ.get("ZF_H5AD", "/Users/shawnlam/Everything GRAD/Literature/Research/ZF_RNA_final_browser_raw_counts.h5ad")  ## where your data file lives; set the ZF_H5AD environment variable to override without editing the script
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zf_cell_totals.npy")  ## small file where we save per-cell total counts so the 2nd launch is fast
TARGET_SUM = 1e4                                  ## normalization target: every cell is rescaled to 10,000 counts (the Seurat/Scanpy default)
MIN_CELLS_PER_SAMPLE = 20                         ## ignore a sample x cell-type bar if it has fewer cells than this (too noisy to trust)
MAX_UMAP_CELLS = 60000                            ## draw at most this many cells on the UMAP so the browser stays fast
GROUP_COLORS = {                                  ## colour-blind-safe palette (Okabe-Ito): colour = age + condition
    "P23_SIL": "#9ecae1",                         ## P23 silent (no song) = light blue
    "P23_SONG": "#0072B2",                        ## P23 song-exposed   = dark blue
    "P30_SIL": "#f4b183",                         ## P30 silent         = light orange
    "P30_SONG": "#D55E00",                        ## P30 song-exposed   = dark orange
}

## ---------------------------------------------------------------- 3. LOAD THE DATA
f = h5py.File(H5AD_PATH, "r")                     ## open the file (does NOT load the 1.1 GB into memory, just opens it)
N_CELLS, N_GENES = [int(x) for x in f["X"].attrs["shape"]]  ## the matrix is cells x genes; read both sizes


def read_cat(col):                                ## helper: turn one stored category column into plain text labels
    cats = [c.decode() if isinstance(c, bytes) else str(c) for c in f["obs"][col]["categories"][:]]  ## the list of possible labels
    return pd.Categorical.from_codes(f["obs"][col]["codes"][:], cats)  ## attach the right label to every one of the 148,722 cells


obs = pd.DataFrame({                              ## build the per-cell table ("obs" = observations = cells)
    "sex": read_cat("sex"),                       ## FEMALE / MALE
    "condition": read_cat("condition"),           ## SIL (silent) / SONG (song-exposed)
    "age": read_cat("age"),                       ## P23 / P30
    "celltype": read_cat("anno_v4"),              ## the 36 cell types
    "lineage": read_cat("lineage"),               ## Glutamatergic / GABAergic / Other
    "sample": read_cat("orig.ident"),             ## which of the 24 animals/libraries the cell came from
})                                                ## end of table
obs["group"] = obs["age"].astype(str) + "_" + obs["condition"].astype(str)  ## combined label, e.g. "P23_SONG"
umap = f["obsm/X_umap"][:]                        ## the 2-D UMAP coordinates already computed for every cell
gene_names = [g.decode() if isinstance(g, bytes) else str(g) for g in f["var/name"][:]]  ## all 21,765 gene names
gene_index = {g: i for i, g in enumerate(gene_names)}  ## dictionary: gene name -> column number (fast lookup)
indptr = f["X/indptr"][:]                         ## "table of contents" telling us where each gene's data starts/ends in the file

## order cell types by lineage (so excitatory/inhibitory/other cluster together), then alphabetically
lineage_of = obs.drop_duplicates("celltype").set_index("celltype")["lineage"].astype(str)  ## cell type -> lineage
CELLTYPE_ORDER = sorted(lineage_of.index.astype(str), key=lambda c: (lineage_of[c], c))  ## the sorted list of 36 names
GROUP_ORDER = ["P23_SIL", "P23_SONG", "P30_SIL", "P30_SONG"]  ## fixed left-to-right order of groups on every plot


## ---------------------------------------------------------------- 4. NORMALIZATION HELPERS
def compute_cell_totals():                        ## total counts per cell (needed to normalize); cached on disk
    if os.path.exists(CACHE_PATH):                ## if we already did this once...
        return np.load(CACHE_PATH)                ## ...just reload it (instant)
    totals = np.zeros(N_CELLS)                    ## start every cell at 0
    n = f["X/data"].shape[0]                      ## number of non-zero entries in the whole matrix
    step = 20_000_000                             ## read the file in chunks of 20 million entries to save memory
    for start in range(0, n, step):               ## loop over chunks
        d = f["X/data"][start:start + step]       ## the counts in this chunk
        i = f["X/indices"][start:start + step]    ## which cell each count belongs to
        totals += np.bincount(i, weights=d, minlength=N_CELLS)  ## add each count onto its cell's running total
    np.save(CACHE_PATH, totals)                   ## save so we never repeat this slow step
    return totals                                 ## give the answer back


CELL_TOTALS = compute_cell_totals()               ## run it once at startup
CELL_TOTALS[CELL_TOTALS == 0] = 1                 ## avoid dividing by zero for any empty cell


@lru_cache(maxsize=256)                           ## keep the last 256 genes in memory
def raw_counts(gene):                             ## raw (SoupX-corrected) counts of ONE gene for ALL cells
    j = gene_index[gene]                          ## which column is this gene
    s, e = indptr[j], indptr[j + 1]               ## where its non-zero entries sit in the file
    v = np.zeros(N_CELLS, dtype=np.float32)       ## start with zero in every cell (most cells have 0)
    v[f["X/indices"][s:e]] = f["X/data"][s:e]     ## fill in only the cells that actually express it
    return v                                      ## return one number per cell


@lru_cache(maxsize=256)                           ## cache this too
def lognorm(gene):                                ## normalized expression: log(1 + counts / cell_total * 10,000)
    return np.log1p(raw_counts(gene) / CELL_TOTALS * TARGET_SUM).astype(np.float32)  ## fixes depth differences between cells, then tames big values


def make_mask(sexes, conds, ages, celltypes):     ## True/False per cell: does it pass all the filters?
    m = np.ones(N_CELLS, dtype=bool)              ## start by keeping everybody
    if sexes:                                     ## if the user picked any sexes...
        m &= obs["sex"].isin(sexes).values        ## ...keep only those
    if conds:                                     ## same for condition
        m &= obs["condition"].isin(conds).values  ## keep selected conditions
    if ages:                                      ## same for age
        m &= obs["age"].isin(ages).values         ## keep selected ages
    if celltypes:                                 ## same for cell type
        m &= obs["celltype"].isin(celltypes).values  ## keep selected cell types
    return m                                      ## the final filter


def group_label(df, multi_sex):                   ## x-axis label: add sex only when more than one sex is shown
    base = df["age"].astype(str) + "_" + df["condition"].astype(str)  ## e.g. P23_SONG
    return (df["sex"].astype(str).str[0] + " " + base) if multi_sex else base  ## "M P23_SONG" or just "P23_SONG"


## ---------------------------------------------------------------- 5. TABLE BUILDERS
def pseudobulk_table(gene, mask):                 ## one value per (animal sample x cell type) - the statistically honest unit
    df = obs.loc[mask, ["sex", "condition", "age", "celltype", "sample"]].copy()  ## the cells that pass the filters
    df["raw"] = raw_counts(gene)[mask]            ## this gene's counts in those cells
    df["total"] = CELL_TOTALS[mask]               ## each cell's total counts
    agg = (df.groupby(["sample", "celltype", "sex", "condition", "age"], observed=True)  ## group cells by animal + cell type
             .agg(raw=("raw", "sum"), total=("total", "sum"), n=("raw", "size"))  ## add up counts and count cells
             .reset_index())                      ## back to a flat table
    agg = agg[agg["n"] >= MIN_CELLS_PER_SAMPLE].copy()  ## drop tiny groups
    agg["value"] = np.log1p(agg["raw"] / agg["total"] * TARGET_SUM)  ## same normalization, applied to the summed counts
    return agg                                    ## one row per animal x cell type


def dot_table_by_group(genes, mask):              ## for the dot plot: mean + % expressing per (gene, cell type, group)
    out = []                                      ## collect one small table per gene
    multi_sex = obs.loc[mask, "sex"].nunique() > 1  ## are both sexes present?
    for g in genes:                               ## loop over chosen genes
        d = obs.loc[mask, ["sex", "condition", "age", "celltype"]].copy()  ## filtered cells
        d["expr"] = lognorm(g)[mask]              ## normalized expression of this gene
        d["pos"] = d["expr"] > 0                  ## is the gene detected in this cell?
        d["grp"] = group_label(d, multi_sex)      ## x-axis group label
        a = (d.groupby(["celltype", "grp"], observed=True)  ## summarize per cell type x group
               .agg(mean=("expr", "mean"), pct=("pos", "mean"), n=("expr", "size"))  ## average level, fraction detected, cell count
               .reset_index())                    ## flat table
        a["pct"] *= 100                           ## fraction -> percent
        a["gene"] = g                             ## remember which gene this row is
        out.append(a)                             ## add to the list
    return pd.concat(out, ignore_index=True)      ## stack all genes together


def dot_table_by_gene(genes, mask):               ## Seurat-style: genes (x) vs cell types (y), cells pooled
    rows = []                                     ## collect rows
    ct = obs.loc[mask, "celltype"]                ## cell type of each filtered cell
    for g in genes:                               ## loop over genes
        d = pd.DataFrame({"celltype": ct.values, "expr": lognorm(g)[mask]})  ## expression per cell
        d["pos"] = d["expr"] > 0                  ## detected or not
        a = d.groupby("celltype", observed=True).agg(mean=("expr", "mean"), pct=("pos", "mean"), n=("expr", "size")).reset_index()  ## summarize
        a["pct"] *= 100                           ## to percent
        a["gene"] = g                             ## label with gene name
        rows.append(a)                            ## store
    return pd.concat(rows, ignore_index=True)     ## one big table


def zscore_per_gene(t, col="mean"):               ## optional scaling: puts every gene on the same colour scale
    t = t.copy()                                  ## don't modify the original
    t[col] = t.groupby("gene")[col].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1))  ## (value - gene average) / gene spread
    return t                                      ## scaled table


## ---------------------------------------------------------------- 6. FIGURES
def fig_pseudobulk(gene, sexes, conds, ages, celltypes):  ## bars (mean of animals) + dots (each animal) per cell type
    mask = make_mask(sexes, conds, ages, celltypes)  ## apply the filters
    t = pseudobulk_table(gene, mask)              ## animal-level values
    if t.empty:                                   ## nothing left after filtering?
        return go.Figure().update_layout(title="No data after filtering")  ## show an empty chart with a message
    multi_sex = t["sex"].nunique() > 1            ## both sexes shown?
    t["grp"] = group_label(t, multi_sex)          ## x label
    t["col"] = t["age"].astype(str) + "_" + t["condition"].astype(str)  ## colour key
    cts = [c for c in CELLTYPE_ORDER if c in set(t["celltype"].astype(str))]  ## cell types that have data, in lineage order
    ncol = min(4, len(cts)) or 1                  ## up to 4 panels per row (keeps long cell-type names readable)
    nrow = int(np.ceil(len(cts) / ncol))          ## rows needed
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=cts, shared_yaxes=True,  ## same y-scale in every panel so panels are comparable
                        horizontal_spacing=0.02, vertical_spacing=0.12 / max(nrow, 1) * 2)  ## tidy spacing
    order = [g for g in (["M " + x for x in GROUP_ORDER] + ["F " + x for x in GROUP_ORDER] if multi_sex else GROUP_ORDER) if g in set(t["grp"])]  ## left-to-right order
    for k, ct in enumerate(cts):                  ## one panel per cell type
        r, c = k // ncol + 1, k % ncol + 1        ## which grid slot
        sub = t[t["celltype"].astype(str) == ct]  ## this cell type's rows
        m = sub.groupby("grp").agg(mean=("value", "mean"), col=("col", "first")).reindex(order).dropna()  ## mean across animals per group
        fig.add_trace(go.Bar(x=m.index, y=m["mean"], marker_color=[GROUP_COLORS[x] for x in m["col"]],  ## bar height = average of animals
                             showlegend=False, hoverinfo="skip"), row=r, col=c)  ## no legend entry, no hover on bars
        fig.add_trace(go.Scatter(x=sub["grp"], y=sub["value"], mode="markers",  ## one dot per animal
                                 marker=dict(color="black", size=6, opacity=0.8),  ## black dots stand out on coloured bars
                                 text=sub["sample"].astype(str) + " (" + sub["n"].astype(str) + " cells)",  ## hover shows animal + cell count
                                 hovertemplate="%{text}<br>%{y:.2f}<extra></extra>", showlegend=False), row=r, col=c)  ## tooltip format
    fig.update_xaxes(categoryorder="array", categoryarray=order, tickangle=45, tickfont=dict(size=9))  ## fixed order, slanted labels
    fig.update_annotations(font_size=11)          ## shrink the cell-type panel titles so they don't overlap
    fig.update_yaxes(rangemode="tozero")          ## bars always start at zero (honest bar charts)
    fig.update_layout(title=f"{gene}: per-animal pseudobulk expression, log(1 + CP10K)  (bar = mean of animals, dot = one animal)",  ## chart title
                      height=260 * nrow + 120, template="simple_white", margin=dict(t=90))  ## size scales with number of rows
    fig.update_yaxes(title_text="log(1+CP10K)", col=1)  ## y label only on the first column
    return fig                                    ## hand back the finished figure


def fig_dot_group(genes, sexes, conds, ages, celltypes, scaled):  ## cell type (y) x group (x), one panel per gene
    mask = make_mask(sexes, conds, ages, celltypes)  ## apply filters
    t = dot_table_by_group(genes, mask)           ## summarize
    if t.empty:                                   ## nothing to plot?
        return go.Figure().update_layout(title="No data after filtering")  ## message
    cbar = "z-score per gene" if scaled else "mean log(1+CP10K)"  ## colour-bar label
    if scaled:                                    ## if user wants scaling...
        t = zscore_per_gene(t)                    ## ...convert means to z-scores
    cts = [c for c in CELLTYPE_ORDER if c in set(t["celltype"].astype(str))]  ## y order
    fig = px.scatter(t, x="grp", y="celltype", size="pct", color="mean", facet_col="gene",  ## dot = size % cells, colour = level
                     category_orders={"celltype": cts[::-1]}, size_max=14,  ## flip so first cell type is on top
                     color_continuous_scale="Viridis", hover_data={"n": True, "pct": ":.1f", "mean": ":.2f"})  ## tooltip info
    fig.update_xaxes(tickangle=45, categoryorder="array", categoryarray=(["M " + x for x in GROUP_ORDER] + ["F " + x for x in GROUP_ORDER]) if obs.loc[mask, "sex"].nunique() > 1 else GROUP_ORDER)  ## fixed group order
    fig.update_layout(title="Expression by cell type and group (dot size = % of cells expressing, colour = " + cbar + ")",  ## title explains encoding
                      height=max(500, 22 * len(cts) + 160), template="simple_white", coloraxis_colorbar_title=cbar)  ## height grows with # cell types
    return fig                                    ## done


def fig_dot_gene(genes, sexes, conds, ages, celltypes, scaled):  ## genes (x) x cell types (y), Seurat DotPlot style
    mask = make_mask(sexes, conds, ages, celltypes)  ## filters
    t = dot_table_by_gene(genes, mask)            ## summarize
    cbar = "z-score per gene" if scaled else "mean log(1+CP10K)"  ## label
    if scaled:                                    ## scale per gene if asked
        t = zscore_per_gene(t)                    ## z-score
    cts = [c for c in CELLTYPE_ORDER if c in set(t["celltype"].astype(str))]  ## order
    fig = px.scatter(t, x="gene", y="celltype", size="pct", color="mean", size_max=16,  ## dot plot
                     category_orders={"celltype": cts[::-1], "gene": list(genes)},  ## keep user's gene order
                     color_continuous_scale="Viridis", hover_data={"n": True, "pct": ":.1f", "mean": ":.2f"})  ## tooltip
    fig.update_layout(title="Genes x cell types for the selected cells (dot size = % expressing, colour = " + cbar + ")",  ## title
                      height=max(500, 22 * len(cts) + 160), template="simple_white", coloraxis_colorbar_title=cbar)  ## sizing
    return fig                                    ## done


def fig_umap(gene, sexes, conds, ages, celltypes):  ## UMAP coloured by one gene's expression
    mask = make_mask(sexes, conds, ages, celltypes)  ## filters
    idx = np.where(mask)[0]                       ## numbers of the cells that pass
    if len(idx) > MAX_UMAP_CELLS:                 ## too many to draw smoothly?
        idx = np.random.default_rng(0).choice(idx, MAX_UMAP_CELLS, replace=False)  ## random subset (same each time)
    expr = lognorm(gene)[idx]                     ## expression of the gene in those cells
    order = np.argsort(expr)                      ## sort low->high so high-expressing cells are drawn on top
    idx, expr = idx[order], expr[order]           ## apply that order
    fig = go.Figure(go.Scattergl(x=umap[idx, 0], y=umap[idx, 1], mode="markers",  ## fast WebGL scatter
                                 marker=dict(size=2.5, color=expr, colorscale="Viridis", colorbar=dict(title="log(1+CP10K)")),  ## colour by expression
                                 text=obs["celltype"].astype(str).values[idx], hovertemplate="%{text}<extra></extra>"))  ## hover shows cell type
    fig.update_layout(title=f"UMAP coloured by {gene} (showing {len(idx):,} cells)", template="simple_white", height=650,  ## title + size
                      xaxis_title="UMAP 1", yaxis_title="UMAP 2")  ## axis labels
    return fig                                    ## done


## ---------------------------------------------------------------- 7. THE INTERACTIVE APP
def build_app():                                  ## creates the web page with dropdowns
    from dash import Dash, dcc, html, Input, Output  ## imported here so --export works even without dash installed
    app = Dash(__name__)                          ## create the app
    opts = lambda cats: [{"label": str(c), "value": str(c)} for c in cats]  ## helper: turn a list into dropdown options
    app.layout = html.Div(style={"fontFamily": "sans-serif", "margin": "16px"}, children=[  ## the page layout
        html.H3("Zebra finch snRNA-seq explorer"),  ## page title
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "10px"}, children=[  ## 4 filter boxes in a row
            html.Div([html.Label("Sex"), dcc.Dropdown(opts(obs["sex"].cat.categories), ["MALE"], multi=True, id="sex")]),  ## sex filter, default MALE
            html.Div([html.Label("Condition"), dcc.Dropdown(opts(obs["condition"].cat.categories), list(obs["condition"].cat.categories), multi=True, id="cond")]),  ## condition filter, default both
            html.Div([html.Label("Age"), dcc.Dropdown(opts(obs["age"].cat.categories), list(obs["age"].cat.categories), multi=True, id="age")]),  ## age filter, default both
            html.Div([html.Label("Plot type"), dcc.Dropdown(id="kind", clearable=False, value="bars", options=[  ## which chart to draw
                {"label": "Per-animal bars: gene across cell types x groups", "value": "bars"},  ## best for MEF2C-in-males
                {"label": "Dot plot: cell type x group (one panel per gene)", "value": "dot_group"},  ## all groups side by side
                {"label": "Dot plot: genes x cell type (Seurat style)", "value": "dot_gene"},  ## compare many genes
                {"label": "UMAP coloured by gene", "value": "umap"}])]),  ## spatial view of expression
        ]),                                       ## end of top row
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginTop": "10px"}, children=[  ## 2 wide boxes
            html.Div([html.Label("Gene(s) - type to search"), dcc.Dropdown(opts(sorted(gene_names)), ["MEF2C"], multi=True, id="genes")]),  ## gene picker (searchable)
            html.Div([html.Label("Cell types (empty = all 36)"), dcc.Dropdown(opts(CELLTYPE_ORDER), [], multi=True, id="cts")]),  ## cell-type picker
        ]),                                       ## end of second row
        dcc.Checklist(id="scaled", options=[{"label": " Scale colour per gene (z-score)", "value": "z"}], value=[]),  ## tick to z-score dot-plot colours
        dcc.Loading(dcc.Graph(id="plot", config={"toImageButtonOptions": {"format": "svg"}})),  ## the chart; camera icon saves SVG
    ])                                            ## end of layout

    @app.callback(Output("plot", "figure"), Input("sex", "value"), Input("cond", "value"), Input("age", "value"),  ## re-draw whenever any control changes
                  Input("kind", "value"), Input("genes", "value"), Input("cts", "value"), Input("scaled", "value"))  ## all inputs
    def update(sexes, conds, ages, kind, genes, cts, scaled):  ## the function Dash calls on each change
        genes = [g for g in (genes or []) if g in gene_index]  ## keep only valid gene names
        if not genes:                             ## nothing selected?
            return go.Figure().update_layout(title="Pick at least one gene")  ## prompt the user
        z = bool(scaled)                          ## True if the checkbox is ticked
        if kind == "bars":                        ## per-animal bars use the first gene only
            return fig_pseudobulk(genes[0], sexes, conds, ages, cts)  ## draw
        if kind == "dot_group":                   ## dot plot by group
            return fig_dot_group(genes, sexes, conds, ages, cts, z)  ## draw
        if kind == "dot_gene":                    ## dot plot by gene
            return fig_dot_gene(genes, sexes, conds, ages, cts, z)  ## draw
        return fig_umap(genes[0], sexes, conds, ages, cts)  ## UMAP uses the first gene
    return app                                    ## give the finished app back


## ---------------------------------------------------------------- 8. ENTRY POINT
if __name__ == "__main__":                        ## only runs when you execute the file (not when imported)
    ap = argparse.ArgumentParser()                ## set up command-line options
    ap.add_argument("--export", nargs="+", help="gene(s) to save as static HTML instead of launching the app")  ## e.g. --export MEF2C
    ap.add_argument("--sex", nargs="+", default=None, help="MALE and/or FEMALE")  ## optional sex filter for export
    ap.add_argument("--cond", nargs="+", default=None, help="SIL and/or SONG")  ## optional condition filter
    ap.add_argument("--age", nargs="+", default=None, help="P23 and/or P30")  ## optional age filter
    args = ap.parse_args()                        ## read what the user typed
    if args.export:                               ## export mode
        tag = "_".join(args.export[:3]) + ("_" + "_".join(args.sex) if args.sex else "")  ## file-name stem
        for g in args.export:                     ## one per-animal bar chart for EACH gene you listed
            fig_pseudobulk(g, args.sex, args.cond, args.age, None).write_html(f"{g}_{'_'.join(args.sex or ['ALL'])}_bars.html")  ## e.g. MEF2C_MALE_bars.html
        fig_dot_group(args.export, args.sex, args.cond, args.age, None, False).write_html(f"{tag}_dot_group.html")  ## one dot plot with a panel per gene
        print("saved bars for", args.export, "and", f"{tag}_dot_group.html")  ## tell the user
    else:                                         ## interactive mode
        build_app().run(debug=False, port=8050)   ## open http://127.0.0.1:8050 in your browser
