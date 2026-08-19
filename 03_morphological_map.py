"""
CRISPRi Morphological Analysis Pipeline  —  PCA + UMAP
=======================================================
Each analysis step saves both a static JPG and an interactive HTML (Bokeh).

Speed optimisations applied
  • Parquet loaded with column pruning + float32 cast on ingestion
  • Well→label and label→gene/subgroup lookups vectorised via dict mapping
  • FOV aggregation (median + std + dominant-gene) computed ONCE as a
    module-level function, then shared between PCAAnalyzer and UMAPAnalyzer
  • Global PCA fit is cached on the PCAAnalyzer instance: the all-genes plot
    and every per-gene highlight reuse the same embedding
  • Global UMAP fit is cached on the UMAPAnalyzer instance: the best-UMAP
    plot and every per-gene highlight reuse the same embedding
  • QuantileTransformer n_quantiles clipped to sample size so it never
    over-allocates on small per-gene subsets

Requires: pandas, numpy, matplotlib, seaborn, scikit-learn, pyarrow,
          umap-learn, bokeh
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import QuantileTransformer, RobustScaler

warnings.filterwarnings("ignore")

try:
    from bokeh.embed import file_html
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure
    from bokeh.resources import CDN
    BOKEH = True
except ImportError:
    BOKEH = False
    print("⚠  bokeh not found — HTML outputs skipped.  pip install bokeh")


# ================================================================================
# CONFIGURATION
# ================================================================================

class Config:
    DATA_FOLDER     = r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P1\CellposeSAM Segmentation results"
    AGGREGATED_FILE = "cell_measurements.parquet"   # written by 01_cellpose_segmentation.py
    PLATE_MAP_FILE  = "plate_map.xlsx"

    FEATURES = ["area_um2", "length_um", "width_um", "roundness", "solidity"]

    RANDOM_STATE            = 0
    DPI                     = 300
    OUTLIER_PERCENTILE      = 99.5
    FIGURE_SIZE_WIDE        = (24, 14)
    FIGURE_SIZE_STANDARD    = (16, 10)

    PCA_N_COMPONENTS        = 2
    PCA_WHITEN              = True

    PERCELL_MAX_CELLS_LABEL  = 2_000
    PERCELL_UMAP_N_NEIGHBORS = 15
    PERCELL_UMAP_MIN_DIST    = 0.1

    # Set at runtime by UMAP grid search
    BEST_METRIC:           str   = "euclidean"
    BEST_SPREAD:           float = 0.3
    BEST_UMAP_MIN_DIST:    float = 0.0
    BEST_UMAP_N_NEIGHBORS: int   = 50

    SUBGROUP_MARKERS = {"1": "o", "2": "s", "3": "^"}
    BOKEH_MARKERS    = {"o": "circle", "s": "square", "^": "triangle"}

    GENE_COLORS = {
        # Cell wall synthesis
        "mrcA": "#E57373", "mrcB": "#EF5350", "mrdA": "#F06292",
        "ftsI":  "#EC407A", "mreB":  "#FF8A65", "murA": "#FFB74D", "murC": "#FFA726",
        # LPS synthesis
        "lpxA": "#4DB6AC", "lpxC": "#26A69A", "lptA": "#4DD0E1",
        "lptC": "#26C6DA", "msbA": "#80DEEA",
        # DNA metabolism
        "gyrA": "#5C6BC0", "gyrB": "#3F51B5", "parC": "#7986CB",
        "parE": "#9FA8DA", "dnaE": "#9575CD", "dnaB": "#B39DDB",
        # Transcription & translation
        "rpoA": "#81C784", "rpoB": "#66BB6A", "rpsA": "#FFF176",
        "rpsL": "#FFEE58", "rplA": "#FFD54F", "rplC": "#FFCA28",
        # Metabolism & export
        "folA": "#AED581", "folP": "#9CCC65", "secY": "#80CBC4", "secA": "#4DB6AC",
        # Cell division
        "ftsZ": "#F06292", "minC": "#F48FB1",
        # Control
        "WT": "#000000",
    }


# ================================================================================
# UTILITIES
# ================================================================================

def setup_analysis_folder(base: str) -> str:
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(base, f"Morphological_Analysis_{ts}")
    os.makedirs(folder, exist_ok=True)
    return folder


def label_sort_key(label) -> tuple:
    try:
        return (0, int(label))
    except (ValueError, TypeError):
        return (1, str(label))


def parse_gene_subgroup(label: str) -> Tuple[str, Optional[str]]:
    s = str(label)
    if "_" in s and s != "WT":
        gene, _, sg = s.rpartition("_")
        if sg in ("1", "2", "3"):
            return gene, sg
    return s, None


def subsample_by_label(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    return (
        df.groupby(col, group_keys=False)
          .apply(lambda g: g if len(g) <= n
                           else g.sample(n, random_state=Config.RANDOM_STATE))
          .reset_index(drop=True)
    )


def scale(X: np.ndarray) -> np.ndarray:
    """RobustScaler then QuantileTransformer to normal distribution."""
    X = RobustScaler().fit_transform(X)
    return QuantileTransformer(
        n_quantiles=min(1_000, len(X)),
        output_distribution="normal",
        random_state=Config.RANDOM_STATE,
    ).fit_transform(X)


# ================================================================================
# DATA PROCESSING
# ================================================================================

class DataProcessor:
    def __init__(self, folder: str):
        self.parquet_path   = Path(folder) / Config.AGGREGATED_FILE
        self.plate_map_path = Path(folder) / Config.PLATE_MAP_FILE

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load only needed columns; cast features to float32 immediately."""
        cols = ["Well", "Point", "filename"] + Config.FEATURES
        data = pd.read_parquet(self.parquet_path, engine="pyarrow", columns=cols)
        for f in Config.FEATURES:
            if f in data.columns:
                data[f] = data[f].astype("float32")
        plate_map = pd.read_excel(self.plate_map_path, header=None)
        print(f"✓ Loaded {len(data):,} cells  ({len(cols)} columns)")
        return data, plate_map

    def assign_labels(self, data: pd.DataFrame, plate_map: pd.DataFrame) -> pd.DataFrame:
        # Build lookup once per unique well — O(unique_wells), not O(rows)
        def _lookup(well: str) -> str:
            if pd.isna(well) or len(well) < 3:
                return "nan"
            r = ord(well[0].upper()) - ord("A")
            c = int(well[1:]) - 1
            if 0 <= r < plate_map.shape[0] and 0 <= c < plate_map.shape[1]:
                v = plate_map.iat[r, c]
                return str(v) if pd.notna(v) else "nan"
            return "nan"

        well_map          = {w: _lookup(w) for w in data["Well"].unique()}
        data["Label"]     = data["Well"].map(well_map)

        # Parse gene / subgroup via dict — O(unique_labels), not O(rows)
        parsed            = {lbl: parse_gene_subgroup(lbl)
                             for lbl in data["Label"].unique()}
        data["Gene"]      = data["Label"].map(lambda x: parsed[x][0])
        data["Subgroup"]  = data["Label"].map(lambda x: parsed[x][1])
        return data

    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        for feat, (lo, hi) in {"roundness": (0.0, 1.0), "solidity": (0.0, 1.0)}.items():
            if feat in data.columns:
                data[feat] = data[feat].clip(lo, hi)
        for feat in Config.FEATURES:
            if feat in data.columns:
                data[feat] = pd.to_numeric(data[feat], errors="coerce")
                data       = data[np.isfinite(data[feat])]
        return data


# ================================================================================
# FOV AGGREGATION  —  computed ONCE in main(), shared by both analysers
# ================================================================================

def aggregate_fov(data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Per-FOV: median + std for each feature (outlier-clipped),
    plus the dominant gene label per FOV.
    """
    avail    = [f for f in features if f in data.columns]
    key_cols = ["Well", "Point", "filename"]

    clipped = data.copy()
    for f in avail:
        clipped[f] = clipped[f].clip(
            upper=float(clipped[f].quantile(Config.OUTLIER_PERCENTILE / 100))
        )

    medians = clipped.groupby(key_cols)[avail].median().reset_index()
    stds    = (clipped.groupby(key_cols)[avail].std()
                      .rename(columns={f: f"{f}_std" for f in avail})
                      .reset_index())
    fov = medians.merge(stds, on=key_cols)

    # Dominant gene = label with highest cell count per FOV
    counts   = (data.groupby(key_cols + ["Label", "Gene", "Subgroup"], dropna=False)
                    .size().reset_index(name="n"))
    dominant = counts.loc[
        counts.groupby(key_cols)["n"].idxmax(),
        key_cols + ["Label", "Gene", "Subgroup"],
    ]
    return fov.merge(dominant, on=key_cols)


# ================================================================================
# BASE ANALYSER  —  shared plot helpers and Bokeh export
# ================================================================================

class BaseAnalyzer:
    """Holds pre-aggregated FOV data; provides shared plotting utilities."""

    def __init__(self, analysis_folder: str, fov_data: pd.DataFrame):
        self.analysis_folder = analysis_folder
        self.fov_data        = fov_data   # shared reference, never recomputed

    # ── Matplotlib helpers ──────────────────────────────────────────────

    def _mpl_scatter(self, ax, emb: np.ndarray, df: pd.DataFrame,
                     gene: str, mask: np.ndarray,
                     s=80, alpha=0.9, lw=1.5, zorder=10):
        """Scatter one gene (subgroup-aware) onto a matplotlib axis."""
        color = Config.GENE_COLORS.get(gene, "#999999")
        gdf   = df[mask].reset_index(drop=True)
        gemb  = emb[mask]
        for sg in sorted(gdf["Subgroup"].fillna("1").unique()):
            sm  = (gdf["Subgroup"].fillna("1") == sg).values
            mk  = Config.SUBGROUP_MARKERS.get(str(sg), "o")
            lbl = gene if sg == "1" else f"{gene}_{sg}"
            ax.scatter(gemb[sm, 0], gemb[sm, 1],
                       c=[color], s=s, alpha=alpha, marker=mk,
                       edgecolors="white", linewidth=lw,
                       label=f"{lbl} (n={sm.sum()})", zorder=zorder)

    @staticmethod
    def _fmt_ax(ax, title: str, xlabel: str, ylabel: str, wide=False):
        fs_t = 16 if wide else 15
        fs_l = 14 if wide else 13
        ax.set_title(title, fontsize=fs_t, fontweight="bold", pad=15)
        ax.set_xlabel(xlabel, fontsize=fs_l)
        ax.set_ylabel(ylabel, fontsize=fs_l)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        if wide:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                      fontsize=11, framealpha=0.9, ncol=3)
        else:
            ax.legend(loc="best", fontsize=11, framealpha=0.9, ncol=2)

    def _save(self, fig, base: str, bokeh_fn=None):
        """Save matplotlib fig as JPG; optionally write interactive HTML."""
        plt.savefig(base + ".jpg", dpi=Config.DPI, bbox_inches="tight")
        plt.close(fig)
        if bokeh_fn:
            bokeh_fn(base + ".html")

    # ── Bokeh interactive export ─────────────────────────────────────────

    def _save_bokeh(self, emb: np.ndarray, df: pd.DataFrame,
                    title: str, xlabel: str, ylabel: str,
                    path: str, highlight_gene: Optional[str] = None):
        """
        Write a self-contained interactive HTML scatter.
        If highlight_gene is given, all non-WT, non-target genes render gray.
        df rows must be reset_index-aligned with emb rows.
        Legend entries are clickable (show/hide per gene group).
        """
        if not BOKEH:
            return

        from bokeh.plotting import figure as bokeh_figure
        p = bokeh_figure(title=title, width=1_000, height=700,
                         tools="pan,wheel_zoom,box_zoom,reset,save",
                         toolbar_location="above")
        p.add_tools(HoverTool(tooltips=[
            ("Gene",     "@gene"),
            ("Subgroup", "@sg"),
            ("Well",     "@well"),
            (xlabel,     "@x{0.3f}"),
            (ylabel,     "@y{0.3f}"),
        ]))
        p.title.text_font_size = "13px"
        p.xaxis.axis_label     = xlabel
        p.yaxis.axis_label     = ylabel

        gene_order = sorted(df["Gene"].unique(),
                            key=lambda g: (g != "WT", label_sort_key(g)))

        for gene in gene_order:
            gmask = (df["Gene"] == gene).values
            if not gmask.any():
                continue

            is_bg = highlight_gene is not None and gene not in (highlight_gene, "WT")
            color = "#cccccc" if is_bg else Config.GENE_COLORS.get(gene, "#999999")
            size  = 6   if (is_bg or gene == "WT") else 10
            alpha = 0.2 if is_bg else (0.55 if gene == "WT" else 0.85)

            gdf  = df[gmask].reset_index(drop=True)
            gemb = emb[gmask]

            for sg in sorted(gdf["Subgroup"].fillna("1").unique()):
                sm = (gdf["Subgroup"].fillna("1") == sg).values
                n  = int(sm.sum())
                if n == 0:
                    continue
                lbl   = gene if sg == "1" else f"{gene}_{sg}"
                bk_mk = Config.BOKEH_MARKERS.get(
                    Config.SUBGROUP_MARKERS.get(sg, "o"), "circle")

                src = ColumnDataSource(dict(
                    x    = gemb[sm, 0].tolist(),
                    y    = gemb[sm, 1].tolist(),
                    gene = [gene] * n,
                    sg   = [sg]   * n,
                    well = (gdf["Well"].values[sm].tolist()
                            if "Well" in gdf.columns else [""] * n),
                ))
                kw = dict(x="x", y="y", source=src,
                          color=color, size=size, alpha=alpha,
                          legend_label=f"{lbl} (n={n})")
                if not is_bg:
                    kw["line_color"] = "white"
                    kw["line_width"] = 0.5

                # Explicit dispatch — avoids fragile getattr()
                if bk_mk == "circle":
                    p.circle(**kw)
                elif bk_mk == "square":
                    p.square(**kw)
                else:
                    p.triangle(**kw)

        p.legend.click_policy         = "hide"
        p.legend.label_text_font_size = "10px"
        p.legend.location             = "top_left"

        Path(path).write_text(file_html(p, CDN, title), encoding="utf-8")


# ================================================================================
# PCA ANALYSER
# ================================================================================

class PCAAnalyzer(BaseAnalyzer):

    def __init__(self, analysis_folder: str, fov_data: pd.DataFrame):
        super().__init__(analysis_folder, fov_data)
        # Global PCA cached here — all_genes + highlight share one fit
        self._g_emb: Optional[np.ndarray]   = None
        self._g_var: Optional[np.ndarray]   = None
        self._g_fov: Optional[pd.DataFrame] = None

    def _fit_pca(self, X: np.ndarray) -> Tuple[np.ndarray, PCA]:
        pca = PCA(n_components=Config.PCA_N_COMPONENTS,
                  whiten=Config.PCA_WHITEN,
                  random_state=Config.RANDOM_STATE)
        return pca.fit_transform(X), pca

    def _global_pca(self, features: List[str]):
        """Fit all-genes PCA once; return cached result on subsequent calls."""
        if self._g_emb is None:
            all_f      = features + [f"{f}_std" for f in features]
            X          = self.fov_data[all_f].dropna().astype(np.float32)
            emb, pca   = self._fit_pca(scale(X.values))
            self._g_emb = emb
            self._g_var = pca.explained_variance_ratio_
            self._g_fov = self.fov_data.loc[X.index].reset_index(drop=True)
            print(f"   [Global PCA]  PC1={self._g_var[0]*100:.1f}%  "
                  f"PC2={self._g_var[1]*100:.1f}%")
        return self._g_emb, self._g_var, self._g_fov

    # ------------------------------------------------------------------

    def generate_per_gene_fov_pca(self, features: List[str] = None):
        """Individual PCA per gene vs WT, FOV-level."""
        features = features or Config.FEATURES
        print("\nPer-Gene PCA (FOV-based)")
        folder = Path(self.analysis_folder) / "PCA_PerGene_FOV"
        folder.mkdir(exist_ok=True)

        all_f   = features + [f"{f}_std" for f in features]
        targets = sorted([g for g in self.fov_data["Gene"].unique() if g != "WT"],
                         key=label_sort_key)
        print(f"  {len(self.fov_data)} FOVs | {len(targets)} genes")

        for gene in targets:
            print(f"  {gene}...", end=" ", flush=True)
            sub = self.fov_data[
                (self.fov_data["Gene"] == "WT") | (self.fov_data["Gene"] == gene)
            ].reset_index(drop=True)
            if len(sub) < 10:
                print(f"skip (n={len(sub)})"); continue

            X   = sub[all_f].dropna().astype(np.float32)
            sub = sub.loc[X.index].reset_index(drop=True)
            X   = X.reset_index(drop=True)
            emb, pca = self._fit_pca(scale(X.values))
            var = pca.explained_variance_ratio_

            fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_STANDARD)
            wt = (sub["Gene"] == "WT").values
            if wt.any():
                ax.scatter(emb[wt, 0], emb[wt, 1],
                           c="#000000", s=75, alpha=0.7, marker="o",
                           edgecolors="white", linewidth=0.5,
                           label=f"WT (n={wt.sum()})", zorder=5)
            self._mpl_scatter(ax, emb, sub, gene, (sub["Gene"] == gene).values,
                              s=120, alpha=0.9, lw=1.5, zorder=10)
            self._fmt_ax(ax,
                         f"PCA: {gene} vs WT (FOV) | {len(sub)} FOVs | "
                         f"PC1 {var[0]*100:.1f}% | PC2 {var[1]*100:.1f}%",
                         f"PC1 ({var[0]*100:.1f}%)", f"PC2 ({var[1]*100:.1f}%)")

            base = str(folder / f"PCA_{gene}_FOV")
            self._save(fig, base, lambda p, e=emb, s=sub, g=gene, v=var:
                self._save_bokeh(e, s, f"PCA: {g} vs WT (FOV)",
                                 f"PC1 ({v[0]*100:.1f}%)", f"PC2 ({v[1]*100:.1f}%)", p))
            print("✓")
        print(f"  → {folder}")

    def generate_all_genes_pca(self, features: List[str] = None):
        """Single PCA with all genes shown in color."""
        features = features or Config.FEATURES
        print("\nAll-Genes PCA")
        folder = Path(self.analysis_folder) / "PCA_AllGenes"
        folder.mkdir(exist_ok=True)

        emb, var, fov = self._global_pca(features)
        print(f"  {len(fov)} FOVs")

        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_WIDE)
        for gene in sorted(fov["Gene"].unique(),
                           key=lambda x: (x != "WT", label_sort_key(x))):
            mask    = (fov["Gene"] == gene).values
            s, a, z = (75, 0.6, 5) if gene == "WT" else (120, 0.8, 10)
            self._mpl_scatter(ax, emb, fov, gene, mask, s=s, alpha=a, lw=1.2, zorder=z)
        self._fmt_ax(ax,
                     f"PCA: All Genes | {len(fov)} FOVs | "
                     f"PC1 {var[0]*100:.1f}% | PC2 {var[1]*100:.1f}%",
                     f"PC1 ({var[0]*100:.1f}%)", f"PC2 ({var[1]*100:.1f}%)", wide=True)

        base = str(folder / "PCA_AllGenes")
        self._save(fig, base, lambda p:
            self._save_bokeh(emb, fov, "PCA: All Genes",
                             f"PC1 ({var[0]*100:.1f}%)", f"PC2 ({var[1]*100:.1f}%)", p))
        print(f"  → {folder}")

    def generate_per_gene_cell_pca(self, data: pd.DataFrame,
                                    features: List[str] = None):
        """Per-gene PCA on individual cell measurements (subsampled)."""
        features = features or Config.FEATURES
        print("\nPer-Gene PCA (Per-Cell)")
        folder = Path(self.analysis_folder) / "PCA_PerGene_CellBased"
        folder.mkdir(exist_ok=True)

        targets = sorted([g for g in data["Gene"].unique() if g != "WT"],
                         key=label_sort_key)
        print(f"  {len(targets)} genes | max {Config.PERCELL_MAX_CELLS_LABEL:,} cells/label")

        for gene in targets:
            print(f"  {gene}...", end=" ", flush=True)
            sub = data[(data["Gene"] == "WT") | (data["Gene"] == gene)].copy()
            sub["_lbl"] = sub["Gene"] + "_" + sub["Subgroup"].fillna("")
            sub = subsample_by_label(sub, "_lbl", Config.PERCELL_MAX_CELLS_LABEL)
            if len(sub) < 50:
                print(f"skip (n={len(sub)})"); continue

            X = sub[features].dropna().astype(np.float32)
            for f in features:
                X[f] = X[f].clip(upper=float(X[f].quantile(Config.OUTLIER_PERCENTILE / 100)))
            sub = sub.loc[X.index].reset_index(drop=True)
            X   = X.reset_index(drop=True)
            emb, pca = self._fit_pca(scale(X.values))
            var = pca.explained_variance_ratio_

            fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_STANDARD)
            wt = (sub["Gene"] == "WT").values
            if wt.any():
                ax.scatter(emb[wt, 0], emb[wt, 1],
                           c="#000000", s=15, alpha=0.4, marker=".",
                           label=f"WT (n={wt.sum():,})", zorder=1, rasterized=True)
            color = Config.GENE_COLORS.get(gene, "#999999")
            tmask = (sub["Gene"] == gene).values
            gdf, gemb = sub[tmask].reset_index(drop=True), emb[tmask]
            for sg in sorted(gdf["Subgroup"].fillna("1").unique()):
                sm  = (gdf["Subgroup"].fillna("1") == sg).values
                mk  = Config.SUBGROUP_MARKERS.get(str(sg), "o")
                lbl = gene if sg == "1" else f"{gene}_{sg}"
                ax.scatter(gemb[sm, 0], gemb[sm, 1],
                           c=[color], s=35, alpha=0.7, marker=mk,
                           edgecolors="white", linewidths=0.3,
                           label=f"{lbl} (n={sm.sum():,})", zorder=10, rasterized=True)
            self._fmt_ax(ax,
                         f"PCA: {gene} vs WT (Per-Cell) | {len(sub):,} cells | "
                         f"PC1 {var[0]*100:.1f}% | PC2 {var[1]*100:.1f}%",
                         f"PC1 ({var[0]*100:.1f}%)", f"PC2 ({var[1]*100:.1f}%)")
            base = str(folder / f"PCA_{gene}_CellBased")
            self._save(fig, base, lambda p, e=emb, s=sub, g=gene, v=var:
                self._save_bokeh(e, s, f"PCA: {g} vs WT (Per-Cell)",
                                 f"PC1 ({v[0]*100:.1f}%)", f"PC2 ({v[1]*100:.1f}%)", p))
            print("✓")
        print(f"  → {folder}")

    def generate_gene_highlight_pca_global(self, features: List[str] = None):
        """One plot per gene: target + WT highlighted; all others gray.
        Reuses the cached global PCA — no extra fitting."""
        features = features or Config.FEATURES
        print("\nGene-Highlight PCAs (Global)")
        folder = Path(self.analysis_folder) / "PCA_GeneHighlight_Global"
        folder.mkdir(exist_ok=True)

        emb, var, fov = self._global_pca(features)
        targets = sorted([g for g in fov["Gene"].unique() if g != "WT"],
                         key=label_sort_key)
        print(f"  {len(fov)} FOVs | {len(targets)} plots (shared embedding — no re-fitting)")

        for gene in targets:
            print(f"  {gene}...", end=" ", flush=True)
            fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_STANDARD)

            for bg in fov["Gene"].unique():
                if bg in (gene, "WT"):
                    continue
                bgm = (fov["Gene"] == bg).values
                ax.scatter(emb[bgm, 0], emb[bgm, 1],
                           c="lightgray", s=30, alpha=0.3,
                           edgecolors="none", zorder=1, rasterized=True)

            wt = (fov["Gene"] == "WT").values
            if wt.any():
                ax.scatter(emb[wt, 0], emb[wt, 1],
                           c="#000000", s=75, alpha=0.7, marker="o",
                           edgecolors="white", linewidth=0.5,
                           label=f"WT (n={wt.sum()})", zorder=15)

            self._mpl_scatter(ax, emb, fov, gene, (fov["Gene"] == gene).values,
                              s=150, alpha=0.9, lw=2.0, zorder=20)
            self._fmt_ax(ax,
                         f"Global PCA: {gene} + WT | {len(fov)} FOVs | "
                         f"PC1 {var[0]*100:.1f}% | PC2 {var[1]*100:.1f}%",
                         f"PC1 ({var[0]*100:.1f}%)", f"PC2 ({var[1]*100:.1f}%)")

            base = str(folder / f"GlobalPCA_Highlight_{gene}")
            self._save(fig, base, lambda p, g=gene:
                self._save_bokeh(emb, fov, f"Global PCA: {g} Highlighted",
                                 f"PC1 ({var[0]*100:.1f}%)", f"PC2 ({var[1]*100:.1f}%)",
                                 p, highlight_gene=g))
            print("✓")
        print(f"  → {folder}")


# ================================================================================
# UMAP ANALYSER
# ================================================================================

class UMAPAnalyzer(BaseAnalyzer):

    def __init__(self, analysis_folder: str, fov_data: pd.DataFrame):
        super().__init__(analysis_folder, fov_data)
        # Best UMAP embedding cached here — highlight plots reuse it
        self._g_emb: Optional[np.ndarray]   = None
        self._g_fov: Optional[pd.DataFrame] = None

    @staticmethod
    def _umap():
        try:
            import umap
            return umap
        except ImportError:
            raise ImportError("umap-learn is required:  pip install umap-learn")

    def _fit_umap(self, X: np.ndarray,
                  n_neighbors: int, min_dist: float,
                  spread: float, metric: str) -> np.ndarray:
        return self._umap().UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist,
            spread=spread, metric=metric,
            random_state=Config.RANDOM_STATE, n_jobs=-1,
        ).fit_transform(X)

    # ------------------------------------------------------------------

    def run_umap_grid_search(self, features: List[str] = None
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sweep n_neighbors 10–200 (step 10); cache the best embedding
        so highlight plots need no extra fitting.
        """
        features = features or Config.FEATURES
        print("=" * 70)
        print("UMAP GRID SEARCH  (Euclidean, min_dist=0, spread=0.3)")
        print("=" * 70)

        folder = Path(self.analysis_folder) / "UMAP_Grid_Search"
        folder.mkdir(exist_ok=True)

        all_f  = features + [f"{f}_std" for f in features]
        X      = self.fov_data[all_f].dropna().astype(np.float32)
        fov    = self.fov_data.loc[X.index].reset_index(drop=True)
        X_s    = scale(X.values)
        labels = fov["Gene"].values

        nn_grid = list(range(10, 210, 10))
        print(f"  {len(fov)} FOVs | {len(nn_grid)} n_neighbors values")

        results = []
        for i, nn in enumerate(nn_grid, 1):
            print(f"  {i:3d}/{len(nn_grid)}  nn={nn:3d}...", end=" ", flush=True)
            emb = self._fit_umap(X_s, nn, 0.0, 0.3, "euclidean")
            ch  = calinski_harabasz_score(emb, labels)
            results.append(dict(n_neighbors=nn, min_dist=0.0, spread=0.3,
                                metric="euclidean", calinski_harabasz_score=ch))
            print(f"CH={ch:.1f}")
            self._save_grid_plot(emb, fov, folder, nn, ch)

        df_res = pd.DataFrame(results).sort_values("calinski_harabasz_score", ascending=False)
        best   = df_res.iloc[0]

        Config.BEST_METRIC           = "euclidean"
        Config.BEST_SPREAD           = 0.3
        Config.BEST_UMAP_MIN_DIST    = 0.0
        Config.BEST_UMAP_N_NEIGHBORS = int(best["n_neighbors"])

        print(f"\n  Best → n_neighbors={Config.BEST_UMAP_N_NEIGHBORS}  "
              f"CH={best['calinski_harabasz_score']:.1f}")

        df_res.to_csv(folder / "grid_search_results.csv", index=False)
        self._save_grid_diagnostics(df_res, folder, best)

        # Cache the best embedding for highlight plots — no second fit
        print("  Computing best UMAP embedding...", end=" ", flush=True)
        best_emb    = self._fit_umap(X_s, Config.BEST_UMAP_N_NEIGHBORS,
                                     Config.BEST_UMAP_MIN_DIST,
                                     Config.BEST_SPREAD, Config.BEST_METRIC)
        self._g_emb = best_emb
        self._g_fov = fov
        print("✓")

        self._save_best_umap_plot(best_emb, fov, best, folder)
        return fov.assign(UMAP1=best_emb[:, 0], UMAP2=best_emb[:, 1]), df_res

    def _save_grid_plot(self, emb, fov, folder, nn, ch):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        for gene in sorted(fov["Gene"].unique(), key=label_sort_key):
            m = (fov["Gene"] == gene).values
            ax.scatter(emb[m, 0], emb[m, 1],
                       color=Config.GENE_COLORS.get(gene, "#999999"),
                       s=50, alpha=0.7, edgecolors="white", linewidths=0.3,
                       label=gene)
        ax.set_title(f"euclidean | md=0.0 | sp=0.3 | nn={nn} | CH={ch:.1f}", fontsize=10)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.2, linestyle="--")
        plt.tight_layout()
        plt.savefig(folder / f"grid_nn{nn:03d}.jpg", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _save_grid_diagnostics(self, df_res: pd.DataFrame,
                                folder: Path, best: pd.Series):
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        ordered = df_res.sort_values("n_neighbors")
        ax.plot(ordered["n_neighbors"], ordered["calinski_harabasz_score"],
                marker="o", linewidth=2, markersize=7)
        ax.axvline(best["n_neighbors"], color="red", linestyle="--",
                   linewidth=2, label=f"Best nn={int(best['n_neighbors'])}")
        ax.set_xlabel("n_neighbors", fontsize=12)
        ax.set_ylabel("Calinski-Harabasz Score", fontsize=12)
        ax.set_title("UMAP Grid Search — n_neighbors vs CH Score",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(folder / "grid_nn_vs_ch.jpg", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_best_umap_plot(self, emb, fov, best, folder):
        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_WIDE)
        for gene in sorted(fov["Gene"].unique(),
                           key=lambda x: (x != "WT", label_sort_key(x))):
            self._mpl_scatter(ax, emb, fov, gene, (fov["Gene"] == gene).values,
                              s=100, alpha=0.8, lw=1.0, zorder=10)
        ax.set_title(f"Best UMAP — nn={Config.BEST_UMAP_N_NEIGHBORS}  "
                     f"md={Config.BEST_UMAP_MIN_DIST}  sp={Config.BEST_SPREAD}  "
                     f"CH={best['calinski_harabasz_score']:.1f}",
                     fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("UMAP 1", fontsize=14); ax.set_ylabel("UMAP 2", fontsize=14)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, ncol=3)
        ax.grid(True, alpha=0.2, linestyle="--")
        plt.tight_layout()
        base = str(folder / "BEST_UMAP")
        self._save(fig, base, lambda p:
            self._save_bokeh(emb, fov, "Best UMAP (All Genes)",
                             "UMAP 1", "UMAP 2", p))
        print(f"  → {folder}/BEST_UMAP.jpg + .html")

    # ------------------------------------------------------------------

    def generate_per_gene_fov_umaps(self, features: List[str] = None):
        """Per-gene UMAP (FOV-level) using best grid-search params."""
        features = features or Config.FEATURES
        print("\nPer-Gene UMAP (FOV-based)")
        folder = Path(self.analysis_folder) / "UMAP_PerGene_FOV"
        folder.mkdir(exist_ok=True)

        all_f   = features + [f"{f}_std" for f in features]
        targets = sorted([g for g in self.fov_data["Gene"].unique() if g != "WT"],
                         key=label_sort_key)
        print(f"  {len(self.fov_data)} FOVs | {len(targets)} genes")

        for gene in targets:
            print(f"  {gene}...", end=" ", flush=True)
            sub = self.fov_data[
                (self.fov_data["Gene"] == "WT") | (self.fov_data["Gene"] == gene)
            ].reset_index(drop=True)
            if len(sub) < 10:
                print(f"skip (n={len(sub)})"); continue

            X   = sub[all_f].dropna().astype(np.float32)
            sub = sub.loc[X.index].reset_index(drop=True)
            emb = self._fit_umap(scale(X.values),
                                 Config.BEST_UMAP_N_NEIGHBORS,
                                 Config.BEST_UMAP_MIN_DIST,
                                 Config.BEST_SPREAD, Config.BEST_METRIC)

            fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_STANDARD)
            wt = (sub["Gene"] == "WT").values
            if wt.any():
                ax.scatter(emb[wt, 0], emb[wt, 1],
                           c="#424242", s=50, alpha=0.7, marker="o",
                           edgecolors="white", linewidth=0.5,
                           label=f"WT (n={wt.sum()})", zorder=5)
            self._mpl_scatter(ax, emb, sub, gene, (sub["Gene"] == gene).values,
                              s=80, alpha=0.9, lw=1.5, zorder=10)
            ax.set_title(f"UMAP: {gene} vs WT (FOV) | {len(sub)} FOVs | "
                         f"nn={Config.BEST_UMAP_N_NEIGHBORS} | "
                         f"md={Config.BEST_UMAP_MIN_DIST}",
                         fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel("UMAP 1", fontsize=11); ax.set_ylabel("UMAP 2", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9, framealpha=0.9, ncol=2)
            plt.tight_layout()

            base = str(folder / f"UMAP_{gene}_FOV")
            self._save(fig, base, lambda p, e=emb, s=sub, g=gene:
                self._save_bokeh(e, s, f"UMAP: {g} vs WT (FOV)", "UMAP 1", "UMAP 2", p))
            print("✓")
        print(f"  → {folder}")

    def generate_per_gene_cell_umaps(self, data: pd.DataFrame,
                                      features: List[str] = None):
        """Per-gene UMAP on individual cell measurements (subsampled)."""
        features = features or Config.FEATURES
        print("\nPer-Gene UMAP (Per-Cell)")
        folder = Path(self.analysis_folder) / "UMAP_PerGene_CellBased"
        folder.mkdir(exist_ok=True)

        targets = sorted([g for g in data["Gene"].unique() if g != "WT"],
                         key=label_sort_key)
        print(f"  {len(targets)} genes | max {Config.PERCELL_MAX_CELLS_LABEL:,} cells/label")

        for gene in targets:
            print(f"  {gene}...", end=" ", flush=True)
            sub = data[(data["Gene"] == "WT") | (data["Gene"] == gene)].copy()
            sub["_lbl"] = sub["Gene"] + "_" + sub["Subgroup"].fillna("")
            sub = subsample_by_label(sub, "_lbl", Config.PERCELL_MAX_CELLS_LABEL)
            if len(sub) < 50:
                print(f"skip (n={len(sub)})"); continue

            X = sub[features].dropna().astype(np.float32)
            for f in features:
                X[f] = X[f].clip(upper=float(X[f].quantile(Config.OUTLIER_PERCENTILE / 100)))
            sub = sub.loc[X.index].reset_index(drop=True)
            X   = X.reset_index(drop=True)
            emb = self._fit_umap(scale(X.values),
                                 Config.PERCELL_UMAP_N_NEIGHBORS,
                                 Config.PERCELL_UMAP_MIN_DIST,
                                 Config.BEST_SPREAD, Config.BEST_METRIC)

            fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_STANDARD)
            wt = (sub["Gene"] == "WT").values
            if wt.any():
                ax.scatter(emb[wt, 0], emb[wt, 1],
                           c="#424242", s=5, alpha=0.3, marker=".",
                           label=f"WT (n={wt.sum():,})", zorder=1, rasterized=True)
            color = Config.GENE_COLORS.get(gene, "#999999")
            tmask = (sub["Gene"] == gene).values
            gdf, gemb = sub[tmask].reset_index(drop=True), emb[tmask]
            for sg in sorted(gdf["Subgroup"].fillna("1").unique()):
                sm  = (gdf["Subgroup"].fillna("1") == sg).values
                mk  = Config.SUBGROUP_MARKERS.get(str(sg), "o")
                lbl = gene if sg == "1" else f"{gene}_{sg}"
                ax.scatter(gemb[sm, 0], gemb[sm, 1],
                           c=[color], s=15, alpha=0.7, marker=mk,
                           edgecolors="white", linewidths=0.3,
                           label=f"{lbl} (n={sm.sum():,})", zorder=10, rasterized=True)
            ax.set_title(f"UMAP: {gene} vs WT (Per-Cell) | {len(sub):,} cells | "
                         f"nn={Config.PERCELL_UMAP_N_NEIGHBORS} | "
                         f"md={Config.PERCELL_UMAP_MIN_DIST}",
                         fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel("UMAP 1", fontsize=11); ax.set_ylabel("UMAP 2", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9, framealpha=0.9, markerscale=2, ncol=2)
            plt.tight_layout()

            base = str(folder / f"UMAP_{gene}_CellBased")
            self._save(fig, base, lambda p, e=emb, s=sub, g=gene:
                self._save_bokeh(e, s, f"UMAP: {g} vs WT (Per-Cell)",
                                 "UMAP 1", "UMAP 2", p))
            print("✓")
        print(f"  → {folder}")

    def generate_gene_highlight_umaps_global(self, features: List[str] = None):
        """One UMAP per gene: target in color, WT in black, others gray.
        Reuses the cached best embedding — no re-fitting."""
        features = features or Config.FEATURES
        print("\nGene-Highlight UMAPs (Global)")
        folder = Path(self.analysis_folder) / "UMAP_GeneHighlight_Global"
        folder.mkdir(exist_ok=True)

        if self._g_emb is None:
            print("  (no cached embedding — running grid search first)")
            self.run_umap_grid_search(features)

        emb     = self._g_emb
        fov     = self._g_fov
        targets = sorted([g for g in fov["Gene"].unique() if g != "WT"],
                         key=label_sort_key)
        print(f"  {len(fov)} FOVs | {len(targets)} plots (shared embedding — no re-fitting)")

        for gene in targets:
            print(f"  {gene}...", end=" ", flush=True)
            fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_STANDARD)

            for bg in fov["Gene"].unique():
                if bg in (gene, "WT"):
                    continue
                bgm = (fov["Gene"] == bg).values
                ax.scatter(emb[bgm, 0], emb[bgm, 1],
                           c="lightgray", s=20, alpha=0.25,
                           edgecolors="none", zorder=1, rasterized=True)

            wt = (fov["Gene"] == "WT").values
            if wt.any():
                ax.scatter(emb[wt, 0], emb[wt, 1],
                           c="#000000", s=50, alpha=0.6, marker="o",
                           edgecolors="white", linewidth=0.5,
                           label=f"WT (n={wt.sum()})", zorder=10)

            self._mpl_scatter(ax, emb, fov, gene, (fov["Gene"] == gene).values,
                              s=100, alpha=0.9, lw=2.0, zorder=20)
            ax.set_title(f"Global UMAP: {gene} Highlighted | {len(fov)} FOVs | "
                         f"nn={Config.BEST_UMAP_N_NEIGHBORS} | "
                         f"md={Config.BEST_UMAP_MIN_DIST}",
                         fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel("UMAP 1", fontsize=11); ax.set_ylabel("UMAP 2", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=10, framealpha=0.95, ncol=2)
            plt.tight_layout()

            base = str(folder / f"GlobalUMAP_Highlight_{gene}")
            self._save(fig, base, lambda p, g=gene:
                self._save_bokeh(emb, fov, f"Global UMAP: {g} Highlighted",
                                 "UMAP 1", "UMAP 2", p, highlight_gene=g))
            print("✓")
        print(f"  → {folder}")


# ================================================================================
# MAIN PIPELINE
# ================================================================================

def main():
    print("=" * 70)
    print("CRISPRi Morphological Analysis Pipeline  —  PCA + UMAP")
    print("=" * 70)

    out_folder = setup_analysis_folder(Config.DATA_FOLDER)
    print(f"✓ Output: {out_folder}\n")

    # ── Step 1: Load & preprocess ────────────────────────────────────────
    print("STEP 1 / 9 — Data Loading & Preprocessing")
    print("-" * 70)
    proc        = DataProcessor(Config.DATA_FOLDER)
    data, pmap  = proc.load_data()
    data        = proc.assign_labels(data, pmap)
    data        = proc.preprocess_features(data)
    print(f"✓ Genes:     {sorted(data['Gene'].unique(), key=label_sort_key)}")
    print(f"✓ Subgroups: {sorted(s for s in data['Subgroup'].unique() if pd.notna(s))}\n")

    # ── Aggregate FOVs ONCE — shared by both analysers ───────────────────
    print("  Aggregating FOV statistics...", end=" ", flush=True)
    fov_data = aggregate_fov(data, Config.FEATURES)
    print(f"✓  {len(fov_data)} FOVs\n")

    # ── PCA ──────────────────────────────────────────────────────────────
    pca_a = PCAAnalyzer(out_folder, fov_data)

    print("STEP 2 / 9 — Per-Gene PCA (FOV-based)");       print("-" * 70)
    pca_a.generate_per_gene_fov_pca();                     print()

    print("STEP 3 / 9 — All-Genes PCA (FOV-based)");      print("-" * 70)
    pca_a.generate_all_genes_pca();                        print()

    print("STEP 4 / 9 — Per-Gene PCA (Per-Cell)");        print("-" * 70)
    pca_a.generate_per_gene_cell_pca(data);                print()

    print("STEP 5 / 9 — Gene-Highlight PCAs (Global)");   print("-" * 70)
    pca_a.generate_gene_highlight_pca_global();            print()

    # ── UMAP ─────────────────────────────────────────────────────────────
    umap_a = UMAPAnalyzer(out_folder, fov_data)

    print("STEP 6 / 9 — UMAP Grid Search");               print("-" * 70)
    _, grid_res = umap_a.run_umap_grid_search();           print()

    print("STEP 7 / 9 — Per-Gene UMAP (FOV-based)");      print("-" * 70)
    umap_a.generate_per_gene_fov_umaps();                  print()

    print("STEP 8 / 9 — Per-Gene UMAP (Per-Cell)");       print("-" * 70)
    umap_a.generate_per_gene_cell_umaps(data);             print()

    print("STEP 9 / 9 — Gene-Highlight UMAPs (Global)");  print("-" * 70)
    umap_a.generate_gene_highlight_umaps_global();         print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results: {out_folder}")
    print(f"\nFeatures ({len(Config.FEATURES)} + {len(Config.FEATURES)} STD = "
          f"{len(Config.FEATURES)*2} dims): {', '.join(Config.FEATURES)}")
    print(f"\nPCA   n_components={Config.PCA_N_COMPONENTS}  whiten={Config.PCA_WHITEN}")
    print(f"UMAP  metric='{Config.BEST_METRIC}'  spread={Config.BEST_SPREAD}  "
          f"min_dist={Config.BEST_UMAP_MIN_DIST}  n_neighbors={Config.BEST_UMAP_N_NEIGHBORS}")
    print(f"Per-cell UMAP  n_neighbors={Config.PERCELL_UMAP_N_NEIGHBORS}  "
          f"min_dist={Config.PERCELL_UMAP_MIN_DIST}  (fixed)")
    print(f"Max cells/label: {Config.PERCELL_MAX_CELLS_LABEL:,}")
    print(f"\n{'✓ Interactive HTML saved alongside every JPG.' if BOKEH else '⚠  pip install bokeh to enable HTML output.'}")


if __name__ == "__main__":
    main()
