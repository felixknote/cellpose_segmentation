# ============================================================================
# MULTI-PLATE MORPHOLOGICAL ANALYSIS PIPELINE
#
# Extends 02_morphological_analysis.py with multi-plate support.
# All shared code (utilities, analyzers, visualizers) is imported from the
# single-plate module via importlib (digit-prefix filename prevents standard
# import syntax).
#
# Experiment structure:
#   - Multiple plates (P1, P2, P3, P4, P5, P6, ...), each a full CRISPRi screen.
#   - Each plate is an independent biological replicate (no tech_rep pairing).
#   - Each plate has its own plate map CSV.
#   - Hierarchy: cell → well → plate (= bio_rep)
#
# Required folder layout:
#   ROOT_DATA_DIR/
#     P1/
#       <SEGMENTATION_SUBPATH>/
#         cell_measurements.parquet
#     P2/ ... P3/ ... (as many plates as needed)
#     P_1_plate_map.csv          ← one per plate, named P_<N>_plate_map.csv
#     P_2_plate_map.csv
#     ...
#
# Usage:
#   python 02_morphological_analysis_multi_plate.py [--root-data-dir DIR]
# ============================================================================

import argparse
import gc
import importlib.util
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

warnings.filterwarnings('ignore')


# ── Load single-plate module (digit prefix prevents standard import) ─────────
_sp_path = Path(__file__).parent / "02_morphological_analysis.py"
_spec    = importlib.util.spec_from_file_location("_sp", _sp_path)
_sp      = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sp)

# ── Re-export shared utilities and analyzers from single-plate module ─────────
EffectSizeCalculator      = _sp.EffectSizeCalculator
StatisticsCache           = _sp.StatisticsCache
parse_gene_subgroup       = _sp.parse_gene_subgroup
get_grouped_gene_name     = _sp.get_grouped_gene_name
label_sort_key            = _sp.label_sort_key
get_gene_color            = _sp.get_gene_color
extract_fov_from_filename = _sp.extract_fov_from_filename
fov_index_to_position     = _sp.fov_index_to_position
_clip_q                   = _sp._clip_q
_bins_for                 = _sp._bins_for
_save_fig                 = _sp._save_fig
_banner                   = _sp._banner
_draw_violin              = _sp._draw_violin
_draw_count_bars          = _sp._draw_count_bars
_stats                    = _sp._stats
_draw_abs_bars            = _sp._draw_abs_bars

# Single-plate pipeline runners — run on pooled multi-plate data
run_histogram_pipeline     = _sp.run_histogram_pipeline
run_visualization_pipeline = _sp.run_visualization_pipeline


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config(_sp.Config):
    # Root folder containing plate subdirectories (P1/, P2/, ...) and plate maps.
    # Set via --root-data-dir or override here before running.
    ROOT_DATA_DIR        = r"D:\2025_12_19 CRISPRi Reference Plate Imaging"
    SEGMENTATION_SUBPATH = r"CellposeSAM Segmentation results"

    # Extend feature units to all 8 features
    FEATURE_UNITS = {
        'roundness': '', 'area_um2': 'µm²', 'length_um': 'µm',
        'width_um': 'µm', 'perimeter_um': 'µm',
        'aspect_ratio': '', 'solidity': '', 'eccentricity': '',
    }

    # Plate-level QC thresholds
    WELL_ZSCORE_THRESHOLD  = 3.0
    PLATE_ZSCORE_THRESHOLD = 2.0
    MIN_CELLS_PER_WELL     = 1_000


# ============================================================================
# COLOUR SYSTEM
# ============================================================================
# One colour per plate (= bio_rep); extended to support P1–P8.
_PLATE_COL: Dict[str, str] = {
    'P1': '#4878D0',  # blue
    'P2': '#EF6548',  # coral
    'P3': '#6ACC65',  # green
    'P4': '#956CB4',  # purple
    'P5': '#8C613C',  # brown
    'P6': '#DC7EC0',  # pink
    'P7': '#797979',  # grey
    'P8': '#D5BB67',  # gold
}
_FEAT_COL: Dict[str, str] = {
    'roundness':    '#9467BD', 'area_um2':     '#1F77B4',
    'length_um':    '#FF7F0E', 'width_um':     '#2CA02C',
    'perimeter_um': '#D62728', 'aspect_ratio': '#8C564B',
    'solidity':     '#E377C2', 'eccentricity': '#7F7F7F',
}
CMAP_SEQ  = 'YlOrBr'
CMAP_DIV  = 'RdBu_r'
_COL_REF   = '#E74C3C'
_COL_WARN1 = '#E67E22'
_COL_WARN2 = '#C0392B'
_COL_OK    = '#27AE60'
_COL_GRID  = '#CCCCCC'
_COL_FOV   = '#5B9BD5'


def get_plate_palette(items) -> Dict[str, str]:
    return {str(x): _PLATE_COL.get(str(x), '#999999') for x in items}


def get_feature_palette(items) -> Dict[str, str]:
    return {str(x): _FEAT_COL.get(str(x), '#999999') for x in items}


# ============================================================================
# WELL POSITION UTILITIES
# ============================================================================
def parse_well_position(well_name: str) -> Optional[Tuple[int, int]]:
    """'A01' → (0, 0), 'H12' → (7, 11).  Returns None on failure."""
    s = str(well_name).strip()
    if len(s) >= 2 and s[0].isalpha() and s[1:].isdigit():
        r = ord(s[0].upper()) - ord('A')
        c = int(s[1:]) - 1
        if 0 <= r < 8 and 0 <= c < 12:
            return (r, c)
    return None


def is_edge_well(well_name: str) -> bool:
    """True if the well is on the outer border of the 96-well plate."""
    pos = parse_well_position(well_name)
    if pos is None:
        return False
    r, c = pos
    return r == 0 or r == 7 or c == 0 or c == 11


# ============================================================================
# MULTI-PLATE DATA LOADING
# Each plate subdirectory is one biological replicate of the full CRISPRi screen.
# Gene labels are derived from the plate-specific plate map CSV.
# ============================================================================
def find_plate_files(cfg: Config) -> List[Tuple[str, Path]]:
    root = Path(cfg.ROOT_DATA_DIR)
    if not root.exists():
        raise FileNotFoundError(f"Root data directory not found: {root}")
    found = []
    for plate_dir in sorted(root.iterdir()):
        if not plate_dir.is_dir():
            continue
        p = plate_dir / cfg.SEGMENTATION_SUBPATH / cfg.AGGREGATED_FILE
        if p.exists():
            found.append((plate_dir.name, p))
    if not found:
        raise FileNotFoundError(
            f"No parquet files found under {root}.\n"
            f"Expected: ROOT_DATA_DIR/P<N>/{cfg.SEGMENTATION_SUBPATH}/{cfg.AGGREGATED_FILE}")
    return found


def _load_single_plate(plate_name: str, parquet_path: Path,
                        cfg: Config) -> Optional[pd.DataFrame]:
    """Load one plate's parquet, apply plate-map label mapping, tag as bio_replicate."""
    all_features = list(set(cfg.MORPHOLOGY_FEATURES + cfg.FEATURES))
    requested    = ['Well', 'filename'] + all_features
    try:
        df = pd.read_parquet(parquet_path, columns=requested)
    except Exception:
        df = pd.read_parquet(parquet_path)
        df = df[[c for c in requested if c in df.columns]]

    # ── Plate-map lookup ─────────────────────────────────────────────────────
    # Plate dir name is e.g. 'P1', 'P2', 'P12' → plate map is P_1_plate_map.csv
    m = re.search(r'(?i)^P(\d+)', plate_name)
    if m is None:
        print(f"  WARN: Cannot parse plate number from '{plate_name}' — skipping")
        return None
    plate_num     = m.group(1)
    platemap_path = Path(cfg.ROOT_DATA_DIR) / f"P_{plate_num}_plate_map.csv"
    if not platemap_path.exists():
        print(f"  WARN: Plate map not found: {platemap_path} — skipping {plate_name}")
        return None

    plate_map = pd.read_csv(platemap_path, header=None)

    def _well_to_label(well) -> Optional[str]:
        if pd.isna(well) or len(str(well)) < 2:
            return None
        try:
            row_idx = ord(well[0].upper()) - ord('A')
            col_idx = int(well[1:]) - 1
            if 0 <= row_idx < plate_map.shape[0] and 0 <= col_idx < plate_map.shape[1]:
                val = plate_map.iloc[row_idx, col_idx]
                return str(val) if pd.notna(val) else None
        except Exception:
            return None

    well_map     = {w: _well_to_label(w) for w in df['Well'].unique()}
    label_series = df['Well'].map(well_map)
    unique_labels = [lb for lb in label_series.unique() if pd.notna(lb)]
    parsed        = {lb: parse_gene_subgroup(str(lb)) for lb in unique_labels}
    gene_map      = {lb: v[0] for lb, v in parsed.items()}
    subgroup_map  = {lb: v[1] for lb, v in parsed.items()}

    df['gene']     = pd.Categorical(label_series.map(gene_map))
    df['Label']    = label_series
    df['Gene']     = label_series.map(gene_map)
    df['Subgroup'] = label_series.map(subgroup_map)
    df['well']     = df['Well']
    df             = df[df['gene'].notna()].copy()

    # ── Plate tag (= bio_replicate) ───────────────────────────────────────────
    df['plate']         = plate_name
    df['bio_replicate'] = plate_name    # plate IS the bio_rep; no tech_rep level

    if 'filename' in df.columns:
        df['fov'] = df['filename'].apply(extract_fov_from_filename)
        df.drop(columns='filename', inplace=True)
    else:
        df['fov'] = 'unknown'

    return df


def load_all_plates(cfg: Config) -> pd.DataFrame:
    """Load all plates, apply gene label mapping, concatenate, filter, report."""
    print("=" * 80)
    print("LOADING MULTI-PLATE DATA")
    print("=" * 80)

    plate_files = find_plate_files(cfg)
    print(f"  Found {len(plate_files)} plates:")
    for name, path in plate_files:
        print(f"    {name}: {path}")

    frames = []
    for plate_name, parquet_path in plate_files:
        df_plate = _load_single_plate(plate_name, parquet_path, cfg)
        if df_plate is not None:
            frames.append(df_plate)
            print(f"  Loaded {plate_name}: {len(df_plate):,} cells")

    if not frames:
        raise RuntimeError(
            "No plates loaded successfully.\n"
            "Check ROOT_DATA_DIR, SEGMENTATION_SUBPATH, and plate map files.")

    df = pd.concat(frames, ignore_index=True)

    all_features = list(set(cfg.MORPHOLOGY_FEATURES + cfg.FEATURES))
    feat_cols    = [f for f in all_features if f in df.columns]
    df[feat_cols] = df[feat_cols].astype(np.float64)

    for feat, (lo, hi) in [('roundness', (0, 1)), ('solidity', (0, 1)),
                            ('eccentricity', (0, 1)), ('aspect_ratio', (1, 20))]:
        if feat in df.columns:
            df[feat] = df[feat].clip(lo, hi)

    for feat in feat_cols:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        df = df[np.isfinite(df[feat])]

    initial_count = len(df)
    wt_mask = df['gene'] == cfg.WT_LABEL
    if wt_mask.sum() > 100:
        wt_lo  = df.loc[wt_mask, feat_cols].quantile(0.01)
        keep   = np.ones(len(df), dtype=bool)
        for feat in feat_cols:
            keep &= df[feat].values >= float(wt_lo[feat])
        df = df.loc[keep]

    df = df[df['well'].notna() & (df['well'] != 'nan')]
    df.reset_index(drop=True, inplace=True)
    gc.collect()

    print(f"\n  After filtering: {len(df):,} cells  ({initial_count - len(df):,} removed)")
    print(f"  Plates (bio_reps): {sorted(df['plate'].unique())}")
    print(f"  Genes:             {df['gene'].nunique()}")
    print(f"  Memory:            {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df


# ============================================================================
# FOV AGGREGATION  (includes plate + gene for downstream analyzers)
# ============================================================================
def aggregate_fov(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Aggregate to FOV level; includes plate and gene columns."""
    feat_cols  = [f for f in features if f in df.columns]
    group_cols = [c for c in ['plate', 'well', 'fov', 'gene', 'bio_replicate']
                  if c in df.columns]
    return (
        df.groupby(group_cols, observed=True, sort=False)
          .agg(n_cells=('well', 'count'),
               **{f'{f}_mean': (f, 'mean') for f in feat_cols},
               **{f'{f}_std':  (f, 'std')  for f in feat_cols})
          .reset_index()
    )


# ============================================================================
# VARIANCE DECOMPOSITION  (3-level: bio_rep → well → within-well)
# Each plate is an independent bio_rep; no tech_rep level.
# ============================================================================
def _ss_between(groups: List[np.ndarray]) -> float:
    """Σ n_g (ȳ_g − ȳ)²"""
    all_vals   = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    return float(sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups if len(g) > 0))


def _ss_within(groups: List[np.ndarray]) -> float:
    """Σ_g Σ_i (x_i − ȳ_g)²"""
    return float(sum(np.sum((g - np.mean(g))**2) for g in groups if len(g) > 0))


def compute_variance_decomposition(df: pd.DataFrame, feature: str) -> Dict[str, float]:
    """Decompose SS_total = SS_bio_rep + SS_well + SS_within_well."""
    nan_result = {k: np.nan for k in ['frac_bio_rep', 'frac_well', 'frac_within_well']}
    vals     = df[feature].values
    ss_total = float(np.sum((vals - np.mean(vals))**2))
    if ss_total == 0:
        return nan_result

    # L1: between plates (bio_reps)
    ss_bio = _ss_between([g[feature].values for _, g in df.groupby('bio_replicate')])
    # L2: between wells within each plate
    ss_well = sum(_ss_between([g[feature].values for _, g in pl.groupby('well')])
                  for _, pl in df.groupby('bio_replicate'))
    # L3: within wells (residual)
    ss_within = _ss_within([g[feature].values
                             for _, g in df.groupby(['bio_replicate', 'well'])])
    ss_sum = ss_bio + ss_well + ss_within
    if ss_sum == 0:
        return nan_result
    return {
        'frac_bio_rep':     ss_bio    / ss_sum,
        'frac_well':        ss_well   / ss_sum,
        'frac_within_well': ss_within / ss_sum,
        'ss_total':         ss_total,
        'ss_bio_rep':       ss_bio,
        'ss_well':          ss_well,
        'ss_within_well':   ss_within,
    }


# ============================================================================
# 96-WELL PLATE HEATMAP HELPERS
# ============================================================================
def _plot_96well_heatmap(grid: np.ndarray, title: str, cbar_label: str,
                          out_path: Path, cfg: Config,
                          cmap: str = CMAP_SEQ,
                          vmin: Optional[float] = None,
                          vmax: Optional[float] = None,
                          fmt: str = '.2g') -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=cfg.DPI)
    cmap_obj = mpl_cm.get_cmap(cmap).copy()
    cmap_obj.set_bad('#D5D8DC')
    finite = grid[~np.isnan(grid)]
    if vmin is None: vmin = float(np.percentile(finite, 5))  if len(finite) else 0.0
    if vmax is None: vmax = float(np.percentile(finite, 95)) if len(finite) else 1.0
    im = ax.imshow(grid, cmap=cmap_obj, aspect='auto', vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03).set_label(cbar_label, fontsize=9)
    for r in range(8):
        for c in range(12):
            if not np.isnan(grid[r, c]):
                ax.text(c, r, format(grid[r, c], fmt), ha='center', va='center',
                        fontsize=6, color='black')
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([f'{i+1:02d}' for i in range(12)], fontsize=8)
    ax.set_yticks(np.arange(8))
    ax.set_yticklabels(list('ABCDEFGH'), fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='minor', length=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
    plt.close()


def _build_well_grid(rows_data: pd.DataFrame, value_col: str) -> np.ndarray:
    """Fill an 8×12 NaN grid from a DataFrame with 'well' and value_col columns."""
    grid = np.full((8, 12), np.nan)
    for _, row in rows_data.iterrows():
        pos = parse_well_position(row['well'])
        if pos is not None:
            grid[pos] = row[value_col]
    return grid


# ============================================================================
# 1.  WITHIN-WELL ANALYZER
# ============================================================================
class WithinWellAnalyzer:
    def __init__(self, df, fov_data, output_base, cfg):
        self.df, self.fov_data = df, fov_data
        self.output_base, self.cfg = output_base, cfg

    def run(self):
        print("\n" + "=" * 80)
        print("WITHIN-WELL ANALYSIS")
        print("=" * 80)
        out = self.output_base / 'within_well'
        out.mkdir(parents=True, exist_ok=True)

        records = []
        for feature in self.cfg.MORPHOLOGY_FEATURES:
            if feature not in self.df.columns:
                continue
            for (plate, well), grp in self.df.groupby(['plate', 'well']):
                vals = grp[feature].values
                if len(vals) < self.cfg.MIN_CELLS_PER_WELL:
                    continue
                m  = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1))
                records.append({
                    'plate': plate, 'well': well,
                    'feature': feature, 'n_cells': len(vals),
                    'mean': m, 'sd': sd,
                    'cv_pct':  (sd / m * 100) if m != 0 else np.nan,
                    'iqr':    float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    'median': float(np.median(vals)),
                })

        summary = pd.DataFrame(records)
        summary.to_csv(out / 'within_well_summary.csv', index=False)

        print(f"  [OK] Within-well summary → {out}")
        return summary


# ============================================================================
# 2.  WELL VARIABILITY ANALYZER
# ============================================================================
class WellVariabilityAnalyzer:
    def __init__(self, df, output_base, cfg):
        self.df, self.output_base, self.cfg = df, output_base, cfg

    def run(self):
        print("\n" + "=" * 80)
        print("WELL-LEVEL VARIABILITY ANALYSIS")
        print("=" * 80)
        out = self.output_base / 'well_variability'
        out.mkdir(parents=True, exist_ok=True)

        feat_cols = [f for f in self.cfg.MORPHOLOGY_FEATURES if f in self.df.columns]
        records = []
        for feature in feat_cols:
            for plate, pgrp in self.df.groupby('plate'):
                well_means = [float(np.mean(wg[feature].values))
                              for _, wg in pgrp.groupby('well')
                              if len(wg) >= self.cfg.MIN_CELLS_PER_WELL]
                if len(well_means) < 2:
                    continue
                m, s = float(np.mean(well_means)), float(np.std(well_means, ddof=1))
                records.append({
                    'plate': plate, 'feature': feature,
                    'n_wells': len(well_means), 'grand_mean': m,
                    'between_well_sd': s,
                    'between_well_cv': (s / m * 100) if m != 0 else np.nan,
                })
        summary = pd.DataFrame(records)
        summary.to_csv(out / 'well_variability_summary.csv', index=False)
        self._plot_between_well_cv(summary, out)

        well_means_df = self.df.groupby(['plate', 'well'])[feat_cols].mean().reset_index()
        self._plot_global_mean_heatmaps(well_means_df, feat_cols, out)
        self._edge_vs_interior(well_means_df, feat_cols, out)

        print(f"  [OK] Well variability summary → {out}")
        return summary

    def _plot_between_well_cv(self, summary, out):
        features = self.cfg.MORPHOLOGY_FEATURES
        plates   = sorted(summary['plate'].unique())
        x, width = np.arange(len(plates)), 0.8 / max(len(features), 1)
        palette  = get_feature_palette(features)
        fig, ax  = plt.subplots(figsize=(max(8, len(plates) * 1.5), 5), dpi=self.cfg.DPI)
        for i, feat in enumerate(features):
            sub  = summary[summary['feature'] == feat]
            vals = [sub.loc[sub['plate'] == p, 'between_well_cv'].mean() for p in plates]
            ax.bar(x + i * width - (len(features) - 1) * width / 2,
                   vals, width * 0.9, label=feat, color=palette.get(feat, '#999'),
                   alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(plates, rotation=30, ha='right')
        ax.set_ylabel('Between-well CV (%)', fontsize=11, fontweight='bold')
        ax.set_title('Between-Well Variability per Plate', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, title='Feature')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--', color=_COL_GRID)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(out / 'between_well_cv_per_plate.png', dpi=self.cfg.DPI, bbox_inches='tight')
        plt.close()

    def _plot_global_mean_heatmaps(self, well_means_df, feat_cols, out):
        global_out   = out / 'mean_all_plates'
        global_out.mkdir(exist_ok=True)
        global_means = well_means_df.groupby('well')[feat_cols].mean().reset_index()
        for feature in feat_cols:
            global_means['_v'] = global_means[feature]
            grid = _build_well_grid(global_means, '_v')
            unit = self.cfg.FEATURE_UNITS.get(feature, '')
            _plot_96well_heatmap(
                grid,
                title=f'Well-Level Mean (all plates) — {feature}',
                cbar_label=f'{feature} {unit}'.strip(),
                out_path=global_out / f'global_mean_{feature}.png',
                cfg=self.cfg,
            )

    def _edge_vs_interior(self, well_means_df, feat_cols, out):
        wm = well_means_df.copy()
        wm['is_edge'] = wm['well'].apply(is_edge_well)
        records = []
        for feature in feat_cols:
            ev = wm.loc[wm['is_edge'],  feature].dropna().values
            iv = wm.loc[~wm['is_edge'], feature].dropna().values
            if len(ev) < 2 or len(iv) < 2:
                continue
            d = EffectSizeCalculator.cohens_d(ev, iv)
            records.append({
                'feature':         feature,
                'edge_mean':       float(np.mean(ev)),
                'interior_mean':   float(np.mean(iv)),
                'edge_cv_pct':     float(np.std(ev, ddof=1) / np.mean(ev) * 100),
                'interior_cv_pct': float(np.std(iv, ddof=1) / np.mean(iv) * 100),
                'cohens_d':        d,
                'interpretation':  EffectSizeCalculator.interpret_cohens_d(d),
                'n_edge': len(ev), 'n_interior': len(iv),
            })
        if records:
            df_e = pd.DataFrame(records)
            df_e.to_csv(out / 'edge_vs_interior.csv', index=False)
            fig, ax = plt.subplots(figsize=(8, max(4, len(df_e) * 0.6)), dpi=self.cfg.DPI)
            colors = [_COL_WARN2 if abs(d) >= 0.5 else _COL_WARN1 if abs(d) >= 0.2 else _COL_OK
                      for d in df_e['cohens_d']]
            ax.barh(df_e['feature'], df_e['cohens_d'], color=colors, alpha=0.85,
                    edgecolor='white', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=1)
            for v in [0.2, -0.2]:
                ax.axvline(v, color=_COL_WARN1, linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xlabel("Cohen's d  (edge − interior)", fontsize=11, fontweight='bold')
            ax.set_title("Edge vs Interior Well Effect", fontsize=11, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(out / 'edge_vs_interior.png', dpi=self.cfg.DPI, bbox_inches='tight')
            plt.close()
        print(f"  [OK] Edge vs interior → {out}")


# ============================================================================
# 3.  BIOLOGICAL REPLICATE ANALYZER
# ============================================================================
class BioReplicateAnalyzer:
    def __init__(self, df, output_base, cfg):
        self.df, self.output_base, self.cfg = df, output_base, cfg

    def run(self):
        print("\n" + "=" * 80)
        print("BIOLOGICAL REPLICATE ANALYSIS")
        print("=" * 80)
        out = self.output_base / 'bio_replicate'
        out.mkdir(parents=True, exist_ok=True)

        feat_cols  = [f for f in self.cfg.MORPHOLOGY_FEATURES if f in self.df.columns]
        plates     = sorted(self.df['plate'].unique())
        bio_global = self.df.groupby('plate')[feat_cols].mean().reset_index()

        records = []
        for feature in feat_cols:
            vals   = bio_global[feature].values
            m, s   = float(np.mean(vals)), float(np.std(vals, ddof=1))
            per_pl = {row['plate']: row[feature] for _, row in bio_global.iterrows()}
            records.append({
                'feature': feature, 'n_plates': len(vals),
                'grand_mean': m, 'sd': s,
                'cv_pct': (s / m * 100) if m != 0 else np.nan,
                **{f'mean_{p}': per_pl.get(p, np.nan) for p in plates},
            })
        summary = pd.DataFrame(records)
        summary.to_csv(out / 'bio_rep_summary.csv', index=False)

        well_means = self.df.groupby(['plate', 'well'])[feat_cols].mean().reset_index()
        for feature in feat_cols:
            data    = well_means[well_means[feature].notna()]
            order   = sorted(data['plate'].unique())
            palette = get_plate_palette(order)
            fig, ax = plt.subplots(figsize=(max(6, len(order) * 1.5), 5), dpi=self.cfg.DPI)
            sns.violinplot(x='plate', y=feature, data=data, order=order,
                           palette=palette, ax=ax, cut=2, bw_adjust=0.7,
                           inner='quart', linewidth=1.2, saturation=0.85)
            ax.axhline(float(data[feature].mean()), color='black', linestyle=':',
                       linewidth=1.5, alpha=0.7, label='Grand mean')
            ax.legend(fontsize=8, loc='upper right')
            unit = self.cfg.FEATURE_UNITS.get(feature, '')
            ax.set_xlabel('Plate (biological replicate)', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{feature} {unit}'.strip(), fontsize=11, fontweight='bold')
            ax.set_title(f'Well-Level Means by Plate: {feature}',
                         fontsize=12, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(out / f'violin_plate_{feature}.png', dpi=self.cfg.DPI, bbox_inches='tight')
            plt.close()

        with open(out / 'bio_rep_summary.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\nBIOLOGICAL REPLICATE (PLATE) VARIABILITY SUMMARY\n"
                    + "=" * 80 + "\n\n")
            for _, row in summary.iterrows():
                f.write(f"{row['feature']}:\n"
                        f"  Between-plate CV: {row['cv_pct']:.2f}%\n"
                        f"  Grand mean:       {row['grand_mean']:.4f}\n")
                for p in plates:
                    f.write(f"  [{p}] {row.get(f'mean_{p}', np.nan):.4f}\n")
                f.write("\n")

        print(f"  [OK] Bio replicate summary → {out}")
        return summary


# ============================================================================
# 4.  PLATE EFFECT ANALYZER
# (Between-plate batch effect: how much does the plate mean deviate from grand mean?)
# ============================================================================
class PlateEffectAnalyzer:
    def __init__(self, df, output_base, cfg):
        self.df, self.output_base, self.cfg = df, output_base, cfg

    def run(self):
        print("\n" + "=" * 80)
        print("PLATE EFFECT ANALYSIS")
        print("=" * 80)
        out = self.output_base / 'plate_effects'
        out.mkdir(parents=True, exist_ok=True)

        feat_cols = [f for f in self.cfg.MORPHOLOGY_FEATURES if f in self.df.columns]
        plates    = sorted(self.df['plate'].unique())
        records   = []

        for feature in feat_cols:
            all_vals    = self.df[feature].values
            grand_mean  = float(np.mean(all_vals))
            ss_total    = float(np.sum((all_vals - grand_mean)**2))
            plate_groups = [self.df.loc[self.df['plate'] == p, feature].values for p in plates]
            eta_sq      = _ss_between(plate_groups) / ss_total if ss_total > 0 else np.nan
            for plate, pgrp in self.df.groupby('plate'):
                m = float(np.mean(pgrp[feature].values))
                records.append({
                    'plate': plate, 'feature': feature, 'n_cells': len(pgrp),
                    'mean': m, 'sd': float(np.std(pgrp[feature].values, ddof=1)),
                    'deviation_from_grand': m - grand_mean,
                    'eta_squared': eta_sq,
                })

        summary = pd.DataFrame(records)
        summary.to_csv(out / 'plate_effect_summary.csv', index=False)

        for feature in feat_cols:
            feat_df = summary[summary['feature'] == feature]
            fig, ax = plt.subplots(figsize=(max(6, len(feat_df)), 5), dpi=self.cfg.DPI)
            colors  = ['#C0392B' if d > 0 else '#2980B9' for d in feat_df['deviation_from_grand']]
            ax.bar(feat_df['plate'], feat_df['deviation_from_grand'],
                   color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
            ax.axhline(0, color='black', linewidth=1.2)
            unit = self.cfg.FEATURE_UNITS.get(feature, '')
            ax.set_ylabel(f'Deviation from grand mean {unit}'.strip(), fontsize=11, fontweight='bold')
            ax.set_title(f'Plate-Level Mean Deviation: {feature}', fontsize=12, fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            ax.grid(True, axis='y', alpha=0.3, linestyle='--', color=_COL_GRID)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(out / f'plate_deviation_{feature}.png', dpi=self.cfg.DPI, bbox_inches='tight')
            plt.close()

        with open(out / 'plate_effect_summary.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\nPLATE EFFECT SUMMARY\n" + "=" * 80 + "\n\n"
                    "eta^2 = fraction of total SS attributable to plate differences.\n\n")
            for feature in feat_cols:
                sub = summary[summary['feature'] == feature]
                if sub.empty: continue
                plate_means = sub['mean'].unique()
                cv = float(np.std(plate_means, ddof=1) / np.mean(plate_means) * 100)
                f.write(f"{feature}:\n"
                        f"  Between-plate CV: {cv:.2f}%\n"
                        f"  eta^2:            {sub['eta_squared'].iloc[0]:.4f}\n")
                for _, row in sub.iterrows():
                    f.write(f"  [{row['plate']}] mean={row['mean']:.4f}  "
                            f"dev={row['deviation_from_grand']:+.4f}  n={row['n_cells']:,}\n")
                f.write("\n")

        print(f"  [OK] Plate effect summary → {out}")
        return summary


# ============================================================================
# 5.  VARIANCE DECOMPOSITION ANALYZER
# ============================================================================
class VarianceDecompositionAnalyzer:
    def __init__(self, df, output_base, cfg):
        self.df, self.output_base, self.cfg = df, output_base, cfg

    def run(self):
        print("\n" + "=" * 80)
        print("VARIANCE DECOMPOSITION ANALYSIS")
        print("=" * 80)
        out = self.output_base / 'variance_decomposition'
        out.mkdir(parents=True, exist_ok=True)

        records = []
        for feature in self.cfg.MORPHOLOGY_FEATURES:
            if feature not in self.df.columns:
                continue
            print(f"  Computing: {feature}")
            comp            = compute_variance_decomposition(self.df, feature)
            comp['feature'] = feature
            records.append(comp)

        summary = pd.DataFrame(records)
        summary.to_csv(out / 'variance_decomposition_summary.csv', index=False)
        self._plot_stacked_bars(summary, out)
        self._write_text_summary(summary, out)
        print(f"  [OK] Variance decomposition → {out}")
        return summary

    def _plot_stacked_bars(self, summary, out):
        level_cols   = ['frac_bio_rep', 'frac_well', 'frac_within_well']
        level_labels = ['Between plates (bio_rep)', 'Between wells (same plate)',
                        'Within well (cell-to-cell)']
        level_colors = ['#C0392B', '#27AE60', '#2980B9']
        features = summary['feature'].tolist()
        y = np.arange(len(features))

        for suffix, cols, labels, colors, title_suf in [
            ('', level_cols, level_labels, level_colors, ''),
            ('_upper', level_cols[:2], level_labels[:2], level_colors[:2],
             ' — Upper Levels (within-well excluded)'),
        ]:
            sub_data = summary[cols].fillna(0).values
            fig, ax  = plt.subplots(figsize=(10, max(5, len(features) * 0.8)), dpi=self.cfg.DPI)
            left = np.zeros(len(features))
            for i, (lbl, col) in enumerate(zip(labels, colors)):
                vals = sub_data[:, i]
                ax.barh(y, vals * 100, left=left * 100, label=lbl,
                        color=col, alpha=0.9, edgecolor='white', linewidth=0.5)
                for j, (v, lft) in enumerate(zip(vals, left)):
                    if v > 0.02:
                        ax.text((lft + v / 2) * 100, y[j], f'{v*100:.1f}%',
                                ha='center', va='center', fontsize=7.5,
                                fontweight='bold', color='white')
                left += vals
            ax.set_yticks(y); ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('Fraction of total variance (%)', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 100)
            ax.set_title(f'Variance Decomposition by Hierarchy Level{title_suf}',
                         fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9, title='Source of variation', framealpha=0.9)
            ax.grid(True, axis='x', alpha=0.25, linestyle='--', color=_COL_GRID)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(out / f'variance_decomposition{suffix}.png',
                        dpi=self.cfg.DPI, bbox_inches='tight')
            plt.close()

    def _write_text_summary(self, summary, out):
        with open(out / 'variance_decomposition_summary.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\nVARIANCE DECOMPOSITION SUMMARY\n" + "=" * 80 + "\n\n"
                    "Hierarchy: plate (bio_rep) -> well -> within well (cells).\n\n")
            for _, row in summary.iterrows():
                f.write(f"Feature: {row['feature']}\n")
                for col, lbl in [('frac_bio_rep',     'Between plates '),
                                  ('frac_well',        'Between wells  '),
                                  ('frac_within_well', 'Within well    ')]:
                    f.write(f"  {lbl}: {row.get(col, np.nan)*100:6.2f}%\n")
                f.write("\n")


# ============================================================================
# 6.  FEATURE RELIABILITY ANALYZER
# ============================================================================
class FeatureReliabilityAnalyzer:
    def __init__(self, df, output_base, cfg, decomp_summary=None):
        self.df, self.output_base, self.cfg = df, output_base, cfg
        self.decomp_summary = decomp_summary

    def run(self):
        print("\n" + "=" * 80)
        print("FEATURE RELIABILITY ANALYSIS")
        print("=" * 80)
        out = self.output_base / 'feature_reliability'
        out.mkdir(parents=True, exist_ok=True)

        feat_cols = [f for f in self.cfg.MORPHOLOGY_FEATURES if f in self.df.columns]
        global_cv = self.df[feat_cols].apply(
            lambda s: s.std(ddof=1) / s.mean() * 100 if s.mean() != 0 else np.nan)
        plate_cv  = self.df.groupby('plate')[feat_cols].mean().apply(
            lambda s: s.std(ddof=1) / s.mean() * 100 if s.mean() != 0 else np.nan)
        well_cv   = self.df.groupby(['plate', 'well'])[feat_cols].mean().apply(
            lambda s: s.std(ddof=1) / s.mean() * 100 if s.mean() != 0 else np.nan)

        records = []
        for feature in feat_cols:
            if self.decomp_summary is not None and feature in self.decomp_summary['feature'].values:
                comp = self.decomp_summary[self.decomp_summary['feature'] == feature].iloc[0].to_dict()
            else:
                comp = compute_variance_decomposition(self.df, feature)
            # Plate-level signal fraction: how much variation is at the bio_rep level?
            f_bio   = comp.get('frac_bio_rep', 0) or 0
            f_well  = comp.get('frac_well', 0) or 0
            # Signal fraction = bio_rep / (bio_rep + well) — how much is plate vs within-plate noise
            sig_frac = f_bio / (f_bio + f_well) if (f_bio + f_well) > 0 else np.nan
            records.append({
                'feature': feature,
                'cv_global_pct': float(global_cv[feature]),
                'cv_plate_pct':  float(plate_cv[feature]),
                'cv_well_pct':   float(well_cv[feature]),
                'signal_fraction_bio_rep': sig_frac,
                **{k: comp.get(k, np.nan) for k in
                   ['frac_bio_rep', 'frac_well', 'frac_within_well']},
            })

        summary = pd.DataFrame(records).sort_values('signal_fraction_bio_rep', ascending=False)
        summary.to_csv(out / 'feature_reliability_summary.csv', index=False)

        # Signal fraction bar chart
        df_s   = summary.sort_values('signal_fraction_bio_rep', ascending=True)
        colors = [_COL_OK if v >= 0.5 else _COL_WARN1 if v >= 0.25 else _COL_WARN2
                  for v in df_s['signal_fraction_bio_rep'].fillna(0)]
        fig, ax = plt.subplots(figsize=(9, max(3, len(df_s) * 0.7)), dpi=self.cfg.DPI)
        bars = ax.barh(df_s['feature'], df_s['signal_fraction_bio_rep'],
                       color=colors, alpha=0.88, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, df_s['signal_fraction_bio_rep']):
            if not np.isnan(val):
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
        ax.axvline(0.5,  color=_COL_OK,    linestyle='--', linewidth=1.3, alpha=0.85,
                   label='0.5 [Good — between-plate drives variation]')
        ax.axvline(0.25, color=_COL_WARN1, linestyle='--', linewidth=1.3, alpha=0.85,
                   label='0.25 [Moderate]')
        ax.set_xlabel('Signal fraction  (bio_rep SS / (bio_rep + well SS))',
                      fontsize=10, fontweight='bold')
        ax.set_title('Feature Reliability: Fraction of Upper-Level Variance at Plate Level',
                     fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1.12)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(out / 'feature_signal_fraction.png', dpi=self.cfg.DPI, bbox_inches='tight')
        plt.close()

        # Feature correlation matrix
        well_means = self.df.groupby(['plate', 'well'])[feat_cols].mean()
        corr = well_means.corr()
        _, ax = plt.subplots(figsize=(len(feat_cols) + 1, len(feat_cols)), dpi=self.cfg.DPI)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                    cbar_kws={'label': 'Pearson r (well means)'})
        ax.set_title('Feature Correlation Matrix (well-level means, all plates)',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out / 'feature_correlation_matrix.png', dpi=self.cfg.DPI, bbox_inches='tight')
        plt.close()
        corr.to_csv(out / 'feature_correlation_matrix.csv')

        with open(out / 'feature_reliability_summary.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\nFEATURE RELIABILITY SUMMARY\n" + "=" * 80 + "\n\n"
                    "signal_fraction = frac_bio_rep / (frac_bio_rep + frac_well)\n"
                    "  → how much of upper-level variance is driven by plate (biology)\n"
                    "    vs. within-plate well-to-well noise.\n\n")
            for _, row in summary.iterrows():
                sf  = row['signal_fraction_bio_rep']
                tag = 'GOOD' if (not np.isnan(sf) and sf >= 0.5) else \
                      'MODERATE' if (not np.isnan(sf) and sf >= 0.25) else 'POOR'
                f.write(f"{row['feature']}: signal_fraction={sf:.3f} [{tag}]\n"
                        f"  Global CV:        {row['cv_global_pct']:.2f}%\n"
                        f"  Between-plate CV: {row['cv_plate_pct']:.2f}%\n"
                        f"  Between-well CV:  {row['cv_well_pct']:.2f}%\n\n")

        print(f"  [OK] Feature reliability summary → {out}")
        return summary


# ============================================================================
# 7.  OUTLIER DETECTOR
# ============================================================================
class OutlierDetector:
    def __init__(self, df, output_base, cfg):
        self.df, self.output_base, self.cfg = df, output_base, cfg

    def run(self):
        print("\n" + "=" * 80)
        print("OUTLIER DETECTION")
        print("=" * 80)
        out = self.output_base / 'outliers'
        out.mkdir(parents=True, exist_ok=True)

        feat_cols  = [f for f in self.cfg.MORPHOLOGY_FEATURES if f in self.df.columns]
        well_stats = (
            self.df.groupby(['plate', 'well'])[feat_cols].agg(['mean', 'count'])
        )
        well_stats.columns = ['_'.join(c) for c in well_stats.columns]
        well_stats = well_stats.reset_index()
        count_col  = f'{feat_cols[0]}_count'
        well_stats = well_stats[well_stats[count_col] >= self.cfg.MIN_CELLS_PER_WELL]

        # Well z-scores within each plate
        z_records = []
        for feat in feat_cols:
            mean_col = f'{feat}_mean'
            for plate, pgrp in well_stats.groupby('plate'):
                m, s = pgrp[mean_col].mean(), pgrp[mean_col].std(ddof=1)
                for _, row in pgrp.iterrows():
                    z = (row[mean_col] - m) / s if s > 0 else 0.0
                    z_records.append({
                        'plate': plate, 'well': row['well'],
                        'feature': feat, 'well_mean': row[mean_col], 'z_score': z,
                        'is_outlier': abs(z) > self.cfg.WELL_ZSCORE_THRESHOLD,
                    })
        z_df          = pd.DataFrame(z_records)
        outlier_wells = z_df[z_df['is_outlier']].copy()
        outlier_wells.to_csv(out / 'outlier_wells.csv', index=False)

        # Plate-level z-scores
        plate_records = []
        for feat in feat_cols:
            plate_means = self.df.groupby('plate')[feat].mean()
            gm, gs      = plate_means.mean(), plate_means.std(ddof=1)
            for plate, pm in plate_means.items():
                z = (pm - gm) / gs if gs > 0 else 0.0
                plate_records.append({
                    'plate': plate, 'feature': feat, 'plate_mean': pm, 'z_score': z,
                    'is_outlier': abs(z) > self.cfg.PLATE_ZSCORE_THRESHOLD,
                })
        plate_z_df     = pd.DataFrame(plate_records)
        outlier_plates = plate_z_df[plate_z_df['is_outlier']].copy()
        outlier_plates.to_csv(out / 'outlier_plates.csv', index=False)

        # Cell count anomalies
        if count_col in well_stats.columns:
            anomalies = []
            for plate, pgrp in well_stats.groupby('plate'):
                m, s = pgrp[count_col].mean(), pgrp[count_col].std(ddof=1)
                for _, row in pgrp.iterrows():
                    z = (row[count_col] - m) / s if s > 0 else 0.0
                    if abs(z) > self.cfg.WELL_ZSCORE_THRESHOLD:
                        anomalies.append({'plate': plate, 'well': row['well'],
                                          'n_cells': row[count_col], 'z_score': z})
            if anomalies:
                pd.DataFrame(anomalies).to_csv(out / 'cell_count_anomalies.csv', index=False)

        n_w = outlier_wells['well'].nunique()   if not outlier_wells.empty   else 0
        n_p = outlier_plates['plate'].nunique() if not outlier_plates.empty  else 0
        print(f"  Outlier wells: {n_w}  |  Outlier plates: {n_p}")
        print(f"  [OK] Outlier results → {out}")
        return outlier_wells, outlier_plates


# ============================================================================
# 8.  SPATIAL / FOV-LEVEL ANALYZER
# ============================================================================
class SpatialAnalyzer:
    def __init__(self, df, fov_data, output_base, cfg):
        self.df, self.fov_data = df, fov_data
        self.output_base, self.cfg = output_base, cfg

    def run(self):
        print("\n" + "=" * 80)
        print("SPATIAL / FOV-LEVEL ANALYSIS")
        print("=" * 80)
        out = self.output_base / 'spatial_analysis'
        out.mkdir(parents=True, exist_ok=True)

        N_FOV = 12
        csv_records = []
        for idx in range(N_FOV):
            pos = fov_index_to_position(idx)
            csv_records.append({
                'fov_index': idx,
                'fov_row': pos[0] if pos else None,
                'fov_col': pos[1] if pos else None,
            })

        for feature in self.cfg.MORPHOLOGY_FEATURES:
            if feature not in self.df.columns:
                continue
            feat_col = f'{feature}_mean'
            feat_data: Dict[int, List[float]] = {i: [] for i in range(N_FOV)}
            for _, row in self.fov_data.iterrows():
                try:
                    idx = int(row['fov'])
                    if 0 <= idx < N_FOV and feat_col in row.index and pd.notna(row[feat_col]):
                        feat_data[idx].append(float(row[feat_col]))
                except (ValueError, TypeError):
                    continue
            for rec in csv_records:
                vals = feat_data[rec['fov_index']]
                rec[f'{feature}_mean'] = float(np.mean(vals)) if vals else float('nan')
            self._heatmap(self.fov_data, feature,
                          out / f'{feature}_fov_heatmap.png',
                          title='All Plates Combined')

        pd.DataFrame(csv_records).to_csv(out / 'fov_summary.csv', index=False)
        print(f"  [OK] Spatial analysis → {out}")

    def _heatmap(self, fov_df, feature, save_path, title):
        N_FOV     = 12
        feat_col  = f'{feature}_mean'
        feat_data: Dict[int, List[float]] = {i: [] for i in range(N_FOV)}
        cnt_data:  Dict[int, List[int]]   = {i: [] for i in range(N_FOV)}
        for _, row in fov_df.iterrows():
            try:
                idx = int(row['fov'])
                if 0 <= idx < N_FOV:
                    if feat_col in row.index and pd.notna(row[feat_col]):
                        feat_data[idx].append(float(row[feat_col]))
                    if pd.notna(row['n_cells']):
                        cnt_data[idx].append(int(row['n_cells']))
            except (ValueError, TypeError):
                continue

        n_rows, n_cols = 5, 5
        feat_grid  = np.full((n_rows, n_cols), np.nan)
        pos_to_idx = {fov_index_to_position(k): k for k in range(N_FOV)
                      if fov_index_to_position(k) is not None}
        for idx in range(N_FOV):
            pos = fov_index_to_position(idx)
            if pos is not None and feat_data[idx]:
                feat_grid[pos] = np.mean(feat_data[idx])

        fig = plt.figure(figsize=(13, 5), dpi=self.cfg.DPI)
        gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        cmap = mpl_cm.get_cmap(CMAP_SEQ).copy()
        cmap.set_bad('#D5D8DC')
        finite = feat_grid[~np.isnan(feat_grid)]
        f_vmin = float(np.percentile(finite, 5))  if len(finite) else 0.0
        f_vmax = float(np.percentile(finite, 95)) if len(finite) else 1.0
        im = ax1.imshow(feat_grid, cmap=cmap, aspect='equal',
                        vmin=f_vmin, vmax=f_vmax, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label(f'{feature} {self.cfg.FEATURE_UNITS.get(feature,"")}'.strip(), fontsize=9)
        for i in range(n_rows):
            for j in range(n_cols):
                fi = pos_to_idx.get((i, j))
                if not np.isnan(feat_grid[i, j]):
                    ax1.text(j, i + 0.2, f'{feat_grid[i,j]:.3g}',
                             ha='center', va='center', color='black',
                             fontweight='bold', fontsize=8)
                if fi is not None and not np.isnan(feat_grid[i, j]):
                    ax1.text(j, i - 0.3, f'FOV{fi}',
                             ha='center', va='center', color='black', fontsize=7)
        ax1.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
        ax1.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
        ax1.grid(which='minor', color='white', linestyle='-', linewidth=2)
        ax1.tick_params(which='minor', size=0)
        ax1.set_title(f'Mean {feature} per FOV\n{title}', fontsize=10, fontweight='bold')

        ax2   = fig.add_subplot(gs[1])
        valid = [(k, np.mean(cnt_data[k]), np.std(cnt_data[k]))
                 for k in range(N_FOV) if cnt_data[k]]
        if valid:
            idxs, means, stds = zip(*valid)
            ax2.bar(range(len(idxs)), means, yerr=stds, color=_COL_FOV,
                    alpha=0.85, edgecolor='white', linewidth=0.5, capsize=3)
            ax2.axhline(float(np.mean(means)), color=_COL_REF, linestyle='--', linewidth=1.5)
            ax2.set_xticks(range(len(idxs)))
            ax2.set_xticklabels([f'FOV{k}' for k in idxs], rotation=45, ha='right', fontsize=8)
        ax2.set_xlabel('FOV', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Average cell count', fontsize=10, fontweight='bold')
        ax2.set_title('Cell Count by FOV', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--', color=_COL_GRID)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=self.cfg.DPI, bbox_inches='tight')
        plt.close()


# ============================================================================
# 9.  DENSITY–FEATURE CORRELATION ANALYZER
# ============================================================================
class DensityCorrelationAnalyzer:
    """Does image-level cell density correlate with feature values (per FOV)?"""

    def __init__(self, fov_data, output_base, cfg):
        self.fov_data, self.output_base, self.cfg = fov_data, output_base, cfg

    def run(self):
        print("\n" + "=" * 80)
        print("DENSITY-FEATURE CORRELATION ANALYSIS")
        print("=" * 80)
        out = self.output_base / 'density_correlation'
        out.mkdir(parents=True, exist_ok=True)

        records = []

        for feature in self.cfg.MORPHOLOGY_FEATURES:
            feat_col = f'{feature}_mean'
            if feat_col not in self.fov_data.columns:
                continue
            all_sub = self.fov_data[['n_cells', feat_col]].dropna()
            all_sub = all_sub[all_sub['n_cells'] >= 100]
            if len(all_sub) >= 3:
                x_a, y_a = all_sub['n_cells'].values, all_sub[feat_col].values
                r_a, p_a = stats.pearsonr(x_a, y_a)
                records.append({'feature': feature, 'plate': 'all',
                                 'pearson_r': float(r_a), 'p_value': float(p_a),
                                 'n_fovs': len(x_a)})
                self._plot_scatter(x_a, y_a, f'All plates — {feature}',
                                   out / f'density_vs_{feature}_all_plates.png',
                                   feature, _COL_FOV, r_a, p_a)

        pd.DataFrame(records).to_csv(out / 'density_correlation_summary.csv', index=False)
        print(f"  [OK] Density correlation → {out}")

    def _plot_scatter(self, x, y, title, out_path, feature, colour, r, p):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=self.cfg.DPI)
        ax.scatter(x, y, color=colour, alpha=0.45, s=18, edgecolors='none')
        if len(x) >= 2:
            coef   = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, np.poly1d(coef)(x_line), color='black', linewidth=1.4, alpha=0.8)
        p_str = f'{p:.2e}' if p < 0.001 else f'{p:.3f}'
        ax.text(0.97, 0.97, f'r = {r:.3f}\np = {p_str}\nn = {len(x):,}',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85))
        unit = self.cfg.FEATURE_UNITS.get(feature, '')
        ax.set_xlabel('Number of cells (image)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{feature} {unit}'.strip(), fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.25, linestyle='--', color=_COL_GRID)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=self.cfg.DPI, bbox_inches='tight')
        plt.close()


# ============================================================================
# 10.  PLATE REPRODUCIBILITY ANALYZER
#
# Computes Cohen's d (gene vs WT) separately on each plate, then quantifies
# how reproducibly each gene shows the same effect across plates.
#
# Outputs per (gene, feature):
#   d_<plate>:       per-plate Cohen's d
#   d_pooled:        pooled Cohen's d (all plates combined)
#   sign_concordance: fraction of plates matching pooled d direction
#   cv_d:            SD(d_plates) / |mean(d_plates)| — lower = more stable
# ============================================================================
class PlateReproducibilityAnalyzer:
    def __init__(self, df, output_base, cfg):
        self.df, self.output_base, self.cfg = df, output_base, cfg

    def run(self):
        print("\n" + "=" * 80)
        print("PLATE REPRODUCIBILITY ANALYSIS")
        print("=" * 80)
        out = self.output_base / 'plate_reproducibility'
        out.mkdir(parents=True, exist_ok=True)

        feat_cols = [f for f in self.cfg.MORPHOLOGY_FEATURES if f in self.df.columns]
        plates    = sorted(self.df['plate'].unique())
        genes     = sorted(self.df.loc[self.df['gene'] != self.cfg.WT_LABEL,
                                        'gene'].dropna().unique())
        wt_label  = self.cfg.WT_LABEL

        # Per-plate d matrix per feature
        per_plate_d: Dict[str, pd.DataFrame] = {}
        for feature in feat_cols:
            wt_all  = self.df.loc[self.df['gene'] == wt_label, feature].values
            records = []
            for plate in plates:
                pl_df   = self.df[self.df['plate'] == plate]
                wt_vals = pl_df.loc[pl_df['gene'] == wt_label, feature].values
                if len(wt_vals) < 10:
                    wt_vals = wt_all  # fall back to pooled WT if plate has too few WT cells
                row = {'plate': plate}
                for gene in genes:
                    gene_vals = pl_df.loc[pl_df['gene'] == gene, feature].values
                    row[gene] = EffectSizeCalculator.cohens_d(gene_vals, wt_vals) \
                                if len(gene_vals) >= 10 else np.nan
                records.append(row)
            per_plate_d[feature] = pd.DataFrame(records).set_index('plate')

        # Reproducibility summary
        repro_records = []
        for feature in feat_cols:
            d_mat = per_plate_d[feature]
            for gene in genes:
                d_vals = d_mat[gene].values.astype(float)
                valid  = d_vals[np.isfinite(d_vals)]
                if len(valid) < 2:
                    continue
                d_pool  = EffectSizeCalculator.cohens_d(
                    self.df.loc[self.df['gene'] == gene, feature].values,
                    self.df.loc[self.df['gene'] == wt_label, feature].values)
                sign_c  = float(np.mean(np.sign(valid) == np.sign(d_pool))) \
                          if not np.isnan(d_pool) else np.nan
                mean_d  = float(np.mean(valid))
                sd_d    = float(np.std(valid, ddof=1))
                cv_d    = sd_d / abs(mean_d) if abs(mean_d) > 0.01 else np.nan
                repro_records.append({
                    'gene': gene, 'feature': feature,
                    'd_pooled':          round(d_pool, 4) if not np.isnan(d_pool) else np.nan,
                    'mean_d_plates':     round(mean_d, 4),
                    'sd_d_plates':       round(sd_d, 4),
                    'cv_d':              round(cv_d, 4)    if not np.isnan(cv_d) else np.nan,
                    'sign_concordance':  round(sign_c, 3)  if not np.isnan(sign_c) else np.nan,
                    'n_plates_valid':    int(len(valid)),
                    **{f'd_{p}': round(float(d_mat.loc[p, gene]), 4)
                       if p in d_mat.index and np.isfinite(d_mat.loc[p, gene]) else np.nan
                       for p in plates},
                })

        repro_df = pd.DataFrame(repro_records)
        repro_df.to_csv(out / 'plate_reproducibility_summary.csv', index=False)

        # Per-feature heatmap: genes × plates
        for feature in feat_cols:
            d_mat = per_plate_d[feature]
            if d_mat.empty: continue
            vmax = max(float(np.nanpercentile(np.abs(d_mat.values), 95)), 0.5)
            fig, ax = plt.subplots(
                figsize=(max(6, len(plates) * 1.2), max(5, len(genes) * 0.35)),
                dpi=self.cfg.DPI)
            im = ax.imshow(d_mat[genes].T.values, cmap='RdBu_r', aspect='auto',
                           vmin=-vmax, vmax=vmax, interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03).set_label(
                "Cohen's d (gene vs WT)", fontsize=9)
            ax.set_xticks(range(len(plates)))
            ax.set_xticklabels(plates, rotation=45, ha='right', fontsize=9)
            ax.set_yticks(range(len(genes)))
            ax.set_yticklabels(genes, fontsize=8)
            ax.set_title(f"Per-Plate Cohen's d: {feature}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(out / f'per_plate_d_{feature}.png', dpi=self.cfg.DPI, bbox_inches='tight')
            plt.close()
            d_mat[genes].to_csv(out / f'per_plate_d_{feature}.csv')

        # Concordance bar chart (mean across features per gene)
        if not repro_df.empty:
            conc_mean = repro_df.groupby('gene')['sign_concordance'].mean().sort_values()
            fig, ax   = plt.subplots(figsize=(max(8, len(conc_mean) * 0.5), 5), dpi=self.cfg.DPI)
            colors    = [_COL_OK if v >= 0.8 else _COL_WARN1 if v >= 0.5 else _COL_WARN2
                         for v in conc_mean]
            ax.bar(conc_mean.index, conc_mean.values, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5)
            ax.axhline(0.8, color=_COL_OK,    linestyle='--', linewidth=1.2, alpha=0.8,
                       label='0.8 (good concordance)')
            ax.axhline(0.5, color=_COL_WARN1, linestyle='--', linewidth=1.2, alpha=0.8,
                       label='0.5 (chance level for 2-plate sets)')
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('Mean sign concordance across features', fontsize=11, fontweight='bold')
            ax.set_title('Per-Gene Sign Concordance Across Plates\n'
                         '(fraction of plates matching pooled effect direction)',
                         fontsize=12, fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--', color=_COL_GRID)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(out / 'sign_concordance_by_gene.png', dpi=self.cfg.DPI, bbox_inches='tight')
            plt.close()

        self._write_text_summary(repro_df, plates, out)
        print(f"  [OK] Plate reproducibility summary → {out}")
        return repro_df

    def _write_text_summary(self, repro_df, plates, out):
        with open(out / 'plate_reproducibility_summary.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n"
                    "PLATE REPRODUCIBILITY SUMMARY\n"
                    + "=" * 80 + "\n\n"
                    "sign_concordance: fraction of plates whose d sign matches the pooled d.\n"
                    "cv_d:             SD(d_plates) / |mean(d_plates)| — lower = more stable.\n\n")
            for gene in sorted(repro_df['gene'].unique()):
                sub = repro_df[repro_df['gene'] == gene]
                f.write(f"Gene: {gene}\n")
                for _, row in sub.iterrows():
                    per_plate = '  '.join(
                        f"{p}={row.get(f'd_{p}', np.nan):+.2f}"
                        for p in plates
                        if not np.isnan(row.get(f'd_{p}', np.nan)))
                    f.write(f"  {row['feature']:<14}: "
                            f"pooled={row['d_pooled']:+.3f}  "
                            f"CV={row['cv_d']:.2f}  "
                            f"conc={row['sign_concordance']:.2f}  "
                            f"[{per_plate}]\n")
                f.write("\n")


# ============================================================================
# DISTRIBUTION VISUALIZER
# ============================================================================
class DistributionVisualizer:
    def __init__(self, df, output_base, cfg):
        self.df, self.output_base, self.cfg = df, output_base, cfg

    def run(self):
        print("\n" + "=" * 80)
        print("DISTRIBUTION VISUALIZATION")
        print("=" * 80)
        out = self.output_base / 'visualization'
        out.mkdir(parents=True, exist_ok=True)

        feat_cols   = [f for f in self.cfg.MORPHOLOGY_FEATURES if f in self.df.columns]
        sample_size = min(80_000, len(self.df))
        df_sample   = self.df.sample(sample_size, random_state=42)

        for feature in feat_cols:
            q_lo, q_hi = df_sample[feature].quantile([0.005, 0.995])
            data   = df_sample[(df_sample[feature] >= q_lo) & (df_sample[feature] <= q_hi)]
            ylabel = f'{feature} {self.cfg.FEATURE_UNITS.get(feature,"")}'.strip()
            order  = sorted(data['plate'].unique())
            palette = get_plate_palette(order)
            fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.2), 5), dpi=self.cfg.DPI)
            sns.violinplot(x='plate', y=feature, data=data, order=order,
                           palette=palette, ax=ax, cut=2, bw_adjust=0.7,
                           inner='quart', linewidth=1.2, saturation=0.85)
            ax.set_title(f'{feature} by Plate (bio_rep)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Plate', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            plt.savefig(out / f'{feature}_by_plate_violin.png',
                        dpi=self.cfg.DPI, bbox_inches='tight')
            plt.close()

        print(f"  [OK] Distribution plots → {out}")


# ============================================================================
# REPLICATE SIMILARITY VISUALIZER
# ============================================================================
class ReplicateSimilarityVisualizer:
    def __init__(self, df, output_base, cfg):
        self.df, self.output_base, self.cfg = df, output_base, cfg

    def run(self):
        print("\n" + "=" * 80)
        print("REPLICATE SIMILARITY VISUALIZATION")
        print("=" * 80)
        out = self.output_base / 'visualization'
        out.mkdir(parents=True, exist_ok=True)

        feat_cols   = [f for f in self.cfg.MORPHOLOGY_FEATURES if f in self.df.columns]
        plate_means = self.df.groupby('plate')[feat_cols].mean()
        plate_z     = (plate_means - plate_means.mean()) / plate_means.std().replace(0, 1)
        plates_list = plate_z.index.tolist()
        n           = len(plates_list)
        dist        = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = float(np.sqrt(np.sum((plate_z.iloc[i] - plate_z.iloc[j])**2)))
        dist_df = pd.DataFrame(dist, index=plates_list, columns=plates_list)
        fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)), dpi=self.cfg.DPI)
        sns.heatmap(dist_df, annot=True, fmt='.2f', cmap='YlOrRd_r', vmin=0, ax=ax,
                    linewidths=1, cbar_kws={'label': 'Euclidean distance (z-scored features)'})
        ax.set_title('Plate-Level Feature Distance\n(0 = identical)', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out / 'plate_similarity_matrix.png', dpi=self.cfg.DPI, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Plate distance matrix → {out}")


# ============================================================================
# RUN SUMMARY
# ============================================================================
def write_run_summary(df: pd.DataFrame, output_base: Path, cfg: Config,
                      decomp_summary=None, outlier_wells=None) -> None:
    plates    = sorted(df['plate'].unique())
    feat_cols = [f for f in cfg.MORPHOLOGY_FEATURES if f in df.columns]
    plate_cvs = (df.groupby('plate')[feat_cols].mean()
                   .std(ddof=1) / df.groupby('plate')[feat_cols].mean().mean() * 100).to_dict()

    path = output_base / 'run_summary.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\nMULTI-PLATE RUN SUMMARY\n" + "=" * 80 + "\n\n")
        f.write(f"Data directory: {cfg.ROOT_DATA_DIR}\n"
                f"Output:         {output_base}\n\n")
        f.write(f"DATASET\n"
                f"  Plates ({len(plates)}): {', '.join(plates)}\n"
                f"  Total cells:  {len(df):,}\n"
                f"  Genes:        {df['gene'].nunique()}\n")
        for p in plates:
            n = (df['plate'] == p).sum()
            w = df.loc[df['plate'] == p, 'well'].nunique()
            f.write(f"  [{p}] {n:>10,} cells,  {w} wells\n")
        f.write("\nBETWEEN-PLATE CV (batch effect size)\n")
        for feat in feat_cols:
            f.write(f"  {feat:<15}: {plate_cvs.get(feat, np.nan):.2f}%\n")
        if decomp_summary is not None and not decomp_summary.empty:
            f.write("\nVARIANCE DECOMPOSITION (% of total SS)\n")
            f.write(f"  {'Feature':<15}  {'Plate':>7}  {'Well':>6}  {'Within':>7}\n")
            for _, row in decomp_summary.iterrows():
                f.write(f"  {row['feature']:<15}  "
                        f"{row.get('frac_bio_rep',0)*100:7.2f}  "
                        f"{row.get('frac_well',0)*100:6.2f}  "
                        f"{row.get('frac_within_well',0)*100:7.2f}\n")
        f.write(f"\nOUTLIER WELLS: "
                f"{outlier_wells['well'].nunique() if outlier_wells is not None and not outlier_wells.empty else 0} "
                f"unique wells flagged\n")
    print(f"  [OK] Run summary → {path}")


# ============================================================================
# PIPELINE RUNNERS
# ============================================================================
def run_variability_pipeline(df, fov_data, output_base, cfg):
    """Run all plate-level QC and variability analyzers."""
    print('\n' + '=' * 80)
    print('VARIABILITY ANALYSIS PIPELINE')
    print('=' * 80)

    WithinWellAnalyzer      (df, fov_data, output_base, cfg).run()
    WellVariabilityAnalyzer (df,           output_base, cfg).run()
    BioReplicateAnalyzer    (df,           output_base, cfg).run()
    PlateEffectAnalyzer     (df,           output_base, cfg).run()

    decomp_summary = VarianceDecompositionAnalyzer(df, output_base, cfg).run()
    FeatureReliabilityAnalyzer(
        df, output_base, cfg,
        decomp_summary=decomp_summary,
    ).run()
    outlier_wells, _ = OutlierDetector(df, output_base, cfg).run()
    SpatialAnalyzer(df, fov_data, output_base, cfg).run()
    DensityCorrelationAnalyzer(fov_data, output_base, cfg).run()
    PlateReproducibilityAnalyzer(df, output_base, cfg).run()

    DistributionVisualizer(df, output_base, cfg).run()
    ReplicateSimilarityVisualizer(df, output_base, cfg).run()

    return decomp_summary, outlier_wells


# ============================================================================
# CLI + MAIN
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Multi-plate CRISPRi morphology analysis pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--root-data-dir', default=Config.ROOT_DATA_DIR,
        help=(
            'Root folder containing plate subdirectories (P1/, P2/, ...) '
            'AND plate map CSVs (P_1_plate_map.csv, P_2_plate_map.csv, ...).'
        ),
    )
    parser.add_argument('--dpi', type=int, default=Config.DPI,
                        help='DPI for all saved figures')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg               = Config()
    cfg.ROOT_DATA_DIR = args.root_data_dir
    cfg.DPI           = args.dpi

    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(cfg.ROOT_DATA_DIR) / f'MultiPlateAnalysis_{timestamp}'
    output_base.mkdir(exist_ok=True)

    sep = '=' * 80
    print(f'{sep}\nMULTI-PLATE MORPHOLOGY ANALYSIS PIPELINE\n{sep}\n'
          f'Root:   {cfg.ROOT_DATA_DIR}\n'
          f'Output: {output_base}\n'
          f'DPI:    {cfg.DPI}\n{sep}')

    df = load_all_plates(cfg)

    all_features = list(set(cfg.MORPHOLOGY_FEATURES + cfg.FEATURES))
    fov_data     = aggregate_fov(df, all_features)

    # The single-plate pipeline uses groupby('well') throughout.  On pooled
    # multi-plate data, wells from different plates sharing the same position
    # (e.g. P1 A01 and P2 A01) would be merged, corrupting every well-level
    # statistic.  Prefix 'plate_' to all well IDs so each (plate, well) pair
    # becomes a unique well for the single-plate pipeline.
    df_sp       = df.copy()
    df_sp['well'] = df_sp['plate'].astype(str) + '_' + df_sp['well'].astype(str)
    df_sp['Well'] = df_sp['well']
    fov_sp      = fov_data.copy()
    fov_sp['well'] = fov_sp['plate'].astype(str) + '_' + fov_sp['well'].astype(str)

    # Part 1: standard single-plate pipeline run on pooled multi-plate data
    run_histogram_pipeline(df_sp, fov_sp, output_base, cfg)
    run_visualization_pipeline(df_sp, output_base, cfg)

    # Part 2: plate-level variability and reproducibility analyses
    decomp_summary, outlier_wells = run_variability_pipeline(
        df, fov_data, output_base, cfg)

    write_run_summary(df, output_base, cfg,
                      decomp_summary=decomp_summary,
                      outlier_wells=outlier_wells)

    print('\n'.join([
        sep, 'ANALYSIS COMPLETE', sep,
        f'Results → {output_base}',
        '  ├── <feature>/wt_comparisons/      pooled effect sizes vs WT',
        '  ├── plate_reproducibility/         per-plate d + cross-plate concordance',
        '  ├── variance_decomposition/        plate / well / within-well SS',
        '  ├── bio_replicate/                plate-level violins',
        '  ├── plate_effects/                plate deviation + eta²',
        '  ├── feature_reliability/           signal fraction per feature',
        '  ├── outliers/                     well/plate z-score flags',
        '  ├── spatial_analysis/             FOV-level positional heatmaps',
        '  ├── density_correlation/          n_cells vs feature per FOV',
        '  ├── within_well/                 within-well CV heatmaps',
        '  ├── well_variability/            between-well CV + edge effects',
        '  ├── visualization/              plate violins + plate distance',
        '  └── run_summary.txt',
    ]))


if __name__ == '__main__':
    main()
