# ============================================================================
# MORPHOLOGY ANALYSIS PIPELINE
# Combines: histogram/statistical analyses + visualization analyses
# ============================================================================

import argparse
import gc
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as mpl_cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, t as scipy_t
from tqdm import tqdm

# Suppress only cosmetic rendering warnings; keep RuntimeWarning/UserWarning
# from NumPy (e.g. divide-by-zero in CV) and pandas visible.
warnings.filterwarnings('ignore', category=UserWarning,  module='seaborn')
warnings.filterwarnings('ignore', category=UserWarning,  module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths
    DATA_FOLDER    = r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P1\CellposeSAM Segmentation results"
    AGGREGATED_FILE = "micromorph_cell_measurements.parquet"

    # Features for histogram/statistical analyses — now all 8 features
    MORPHOLOGY_FEATURES = [
        'roundness', 'area_um2', 'length_um', 'width_um', 'perimeter_um',
        'aspect_ratio', 'solidity', 'eccentricity',
    ]

    # Features for visualization analyses (same superset)
    FEATURES = [
        'area_um2', 'perimeter_um', 'length_um', 'width_um',
        'aspect_ratio', 'roundness', 'solidity', 'eccentricity',
    ]

    # Histogram settings
    BIN_WIDTHS = {
        'roundness': 0.02, 'area_um2': 0.2, 'length_um': 0.05,
        'width_um': 0.02, 'perimeter_um': 0.2,
        'aspect_ratio': 0.2, 'solidity': 0.02, 'eccentricity': 0.02,
    }
    FEATURE_UNITS = {
        'roundness': '', 'area_um2': 'µm²', 'length_um': 'µm',
        'width_um': 'µm', 'perimeter_um': 'µm',
        'aspect_ratio': '', 'solidity': '', 'eccentricity': '',
    }
    HISTOGRAM_ALPHA        = 0.8
    FIGURE_SIZE            = (12, 6)
    EFFECT_SIZE_THRESHOLDS = {'small': 0.2, 'medium': 0.5, 'large': 0.8}

    # Single authoritative DPI used by every plot in both pipelines.
    DPI = 300

    # Visualization settings
    FIGURE_SIZE_STANDARD = (13.333, 7.5)
    FIGURE_SIZE_SUBGROUP = (8, 6)

    # Gene color mapping by mechanism of action
    GENE_COLORS = {
        'mrcA': '#E57373', 'mrcB': '#EF5350', 'mrdA': '#F06292', 'ftsI': '#EC407A',
        'mreB': '#FF8A65', 'murA': '#FFB74D', 'murC': '#FFA726',
        'lpxA': '#4DB6AC', 'lpxC': '#26A69A', 'lptA': '#4DD0E1', 'lptC': '#26C6DA', 'msbA': '#80DEEA',
        'gyrA': '#5C6BC0', 'gyrB': '#3F51B5', 'parC': '#7986CB', 'parE': '#9FA8DA',
        'dnaE': '#9575CD', 'dnaB': '#B39DDB',
        'rpoA': '#81C784', 'rpoB': '#66BB6A',
        'rpsA': '#FFF176', 'rpsL': '#FFEE58', 'rplA': '#FFD54F', 'rplC': '#FFCA28',
        'folA': '#AED581', 'folP': '#9CCC65',
        'secY': '#80CBC4', 'secA': '#4DB6AC',
        'ftsZ': '#F06292', 'minC': '#F48FB1',
        'WT':   '#424242',
    }

    # Pathway / mechanism-of-action groupings — matches Selection of Genes reference plate
    PATHWAY_GROUPS: Dict[str, List[str]] = {
        'Peptidoglycan Synthesis':   ['mrcA', 'mrcB', 'mrdA', 'ftsI', 'mreB', 'murA', 'murC'],
        'Outer Membrane & Division': ['lpxA', 'lpxC', 'lptA', 'lptC', 'msbA', 'ftsZ'],
        'DNA Replication':           ['gyrA', 'gyrB', 'parC', 'parE', 'dnaE', 'dnaB'],
        'Transcription':             ['rpoA', 'rpoB'],
        'Translation':               ['rpsA', 'rpsL', 'rplA', 'rplC'],
        'Metabolism':                ['folA', 'folP'],
        'Secretion':                 ['secA', 'secY'],
    }

    # ── Control labels ───────────────────────────────────────────────────────
    # 'WT'  — wells labeled "WT NC_X": the primary morphological reference.
    # 'NC'  — wells labeled "NC_X": a genuinely SEPARATE experimental condition
    #         (different gRNA backbone / induction context / plate position).
    #         Observed d≈0.2–0.4 vs WT on multiple features; treated as the
    #         assay noise floor, not as an alias for WT.
    WT_LABEL  = 'WT'
    NC_LABEL  = 'NC'

    # ── Survivor-selection / depletion flag ──────────────────────────────────
    # Genes with mean cells-per-well below this threshold likely reflect severe
    # growth arrest or CRISPRi lethality.  Morphology estimates from such
    # conditions describe survivor sub-populations, not the depleted population.
    DEPLETION_THRESHOLD_CELLS_PER_WELL = 5_000

    # ── Known segmentation failure modes ────────────────────────────────────
    # Genes whose segmentation is known to be unreliable with CellposeSAM.
    # ftsZ produces filaments that exceed the model's shape prior and are
    # collapsed into circular blobs — geometric features (aspect_ratio, length)
    # are therefore not trustworthy for this gene.  Cell count (n_cells) and
    # area remain valid gross indicators of filamentation.
    KNOWN_SEGMENTATION_ISSUES: Dict[str, str] = {
        'ftsZ': 'Filaments collapsed to blobs by CellposeSAM; '
                'aspect_ratio/length unreliable. Use area & n_cells only.',
    }


# ============================================================================
# UTILITIES
# ============================================================================
def extract_fov_from_filename(filename: str) -> str:
    try:
        parts = str(filename).split('_')
        for i, part in enumerate(parts):
            if part.startswith('Point') and i + 1 < len(parts):
                img_num = parts[i + 1]
                if img_num.isdigit():
                    return img_num
    except Exception:
        pass
    return 'unknown'


_WT_RE = re.compile(r'^\s*WT\s+NC_(\d+)\s*$', re.IGNORECASE)
_SG_RE = re.compile(r'^\s*([A-Za-z]\w*?)_(\d+)\s*$')


def parse_gene_subgroup(label: str) -> Tuple[str, Optional[str]]:
    """Parse gene label into (base_name, subgroup).

    'WT NC_3' -> ('WT', '3')   'NC_5' -> ('NC', '5')   'mrcA_2' -> ('mrcA', '2')

    Label semantics
    ---------------
    WT NC_1..6  — "WT" wells: the primary morphological reference baseline.
                  These are non-cutting gRNA wells in the WT strain background.
    NC_1..6     — "NC" wells: a GENUINELY DIFFERENT experimental condition.
                  Despite also carrying a non-cutting gRNA, NC wells differ from
                  WT NC wells in observed morphology (d≈0.2–0.4 on 5/8 features,
                  concordance=1.0).  Likely causes: different plate position,
                  induction batch, or gRNA construct.  NC is NOT an alias for WT;
                  it defines the assay noise floor and should be retained as its
                  own condition in all comparisons.
    gene_N      — CRISPRi knockdown with gRNA subgroup N (typically 1–3).
    """
    s = str(label).strip()
    m = _WT_RE.match(s)
    if m:
        return 'WT', m.group(1)
    if s.upper().startswith('WT'):
        return 'WT', None
    m = _SG_RE.match(s)
    if m:
        return m.group(1), m.group(2)
    return s, None


def get_grouped_gene_name(label: str) -> str:
    """Return the base gene name without any subgroup suffix."""
    base_gene, _ = parse_gene_subgroup(label)
    return base_gene


def label_sort_key(label: str) -> Tuple[int, object]:
    """Sort numbers before strings, then alphabetically."""
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def get_gene_color(gene: str, default: str = 'gray') -> str:
    """Look up a gene's display color from Config."""
    return Config.GENE_COLORS.get(gene, default)


def fov_index_to_position(fov_idx: int) -> Optional[Tuple[int, int]]:
    """Convert FOV index to (row, col) grid position for the meandering acquisition pattern."""
    fov_positions = {
        0:  (0, 1), 1:  (0, 2), 2:  (0, 3),
        3:  (1, 4), 4:  (1, 3), 5:  (1, 2), 6:  (1, 1), 7:  (1, 0),
        8:  (2, 0), 9:  (2, 1), 10: (2, 2), 11: (2, 3), 12: (2, 4),
        13: (3, 4), 14: (3, 3), 15: (3, 2), 16: (3, 1), 17: (3, 0),
        18: (4, 1), 19: (4, 2), 20: (4, 3),
    }
    return fov_positions.get(fov_idx)


def _clip_q(data: pd.DataFrame, feature: str, lo: float = 0.005, hi: float = 0.995) -> pd.DataFrame:
    """Return data clipped to the [lo, hi] quantile range of `feature`."""
    q_low, q_high = data[feature].quantile([lo, hi])
    return data[(data[feature] >= q_low) & (data[feature] <= q_high)]


def _bins_for(df: pd.DataFrame, feature: str, cfg: 'Config') -> np.ndarray:
    """Uniform-width histogram bins spanning the central 99.8% of `feature`."""
    bin_width = cfg.BIN_WIDTHS.get(feature, 0.1)
    return np.arange(np.percentile(df[feature], 0.1),
                     np.percentile(df[feature], 99.9) + bin_width, bin_width)


def _save_fig(fig, path, cfg: 'Config', facecolor: str = 'white'):
    fig.tight_layout()
    fig.savefig(str(path), dpi=cfg.DPI, bbox_inches='tight', facecolor=facecolor)
    plt.close(fig)


def _banner(title: str, width: int = 80, leading_nl: bool = True):
    bar = '=' * width
    print(f'{chr(10) if leading_nl else ""}{bar}\n{title}\n{bar}')


def _draw_violin(ax, data, x_col, feature, order, palette, title, xlabel='Gene'):
    sns.violinplot(x=x_col, y=feature, data=data, order=order,
                   palette=palette, ax=ax, cut=2, bw_adjust=0.7, gridsize=150,
                   inner='quart', scale='width', linewidth=1.2, saturation=0.85)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(feature, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)


def _draw_count_bars(ax, df_counts, label_col, cfg, xlabel, title):
    """Shared bar-chart drawing for cell-count plots (by label and by gene)."""
    x_pos = np.arange(len(df_counts))
    colors = [get_gene_color(parse_gene_subgroup(lbl)[0], '#4A90E2')
              for lbl in df_counts[label_col]]
    bars = ax.bar(x_pos, df_counts['cells_per_well'], color=colors,
                  alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Cells per Well', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_counts[label_col], rotation=45, ha='right', fontweight='bold')
    max_h = df_counts['cells_per_well'].max()
    for bar, cpw, lbl, wells in zip(bars, df_counts['cells_per_well'],
                                    df_counts[label_col], df_counts['wells']):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + max_h * 0.01,
                f'{cpw:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=45)
        ax.text(bar.get_x() + bar.get_width() / 2., h * 0.5,
                str(lbl)[:6], ha='center', va='center', fontsize=9, alpha=0.4, color='white', fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2., h * 0.15,
                f'w={wells}', ha='center', va='center', fontsize=8, alpha=0.7, color='white', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _stats(values) -> dict:
    """Summary stats dict shared by StatisticsCache entries."""
    return {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)),
            'values': values, 'n': len(values)}


def _draw_abs_bars(ax, genes, data, feature, cfg, colors, compact=False):
    """Mean ± SD bar chart vs WT reference band. Shared by plot_feature/plot_overview."""
    means = [float(np.mean(data.loc[data['Gene'] == g, feature].values)) for g in genes]
    sds   = [float(np.std( data.loc[data['Gene'] == g, feature].values, ddof=1)) for g in genes]
    unit  = cfg.FEATURE_UNITS.get(feature, '')
    wt_mean = float(np.mean(data.loc[data['Gene'] == 'WT', feature].values))
    wt_sd   = float(np.std( data.loc[data['Gene'] == 'WT', feature].values, ddof=1))

    lw, cap, elw = (0.4, 2, 1) if compact else (0.8, 4, 1.5)
    ax.bar(range(len(genes)), means, yerr=sds, color=colors, alpha=0.8 if compact else 0.85,
           edgecolor='black', linewidth=lw, capsize=cap,
           error_kw={'linewidth': elw, 'ecolor': 'black', **({} if compact else {'capthick': 1.5})})
    ax.axhline(wt_mean, color='black', linestyle='--',
               linewidth=1.2 if compact else 1.8, alpha=0.7,
               label=None if compact else f'WT: {wt_mean:.3f} ± {wt_sd:.3f} {unit}'.strip())
    ax.fill_between([-0.5, len(genes) - 0.5],
                    wt_mean - wt_sd, wt_mean + wt_sd,
                    color='gray', alpha=0.12, zorder=0)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return unit


# ============================================================================
# STATISTICS CACHE
# Scoped to the histogram pipeline — passed explicitly to each analyzer.
# Never a global singleton, so the visualization pipeline cannot accidentally
# read stale data if it were ever extended to use the cache.
# ============================================================================
class StatisticsCache:
    """Memoize per-gene / per-feature statistics to avoid redundant data passes."""

    def __init__(self):
        self.cell_level_stats = {}
        self.well_level_stats = {}
        self.grouped_stats    = {}

    def clear(self):
        """Release all cached arrays (call after each feature to free memory)."""
        self.cell_level_stats.clear()
        self.well_level_stats.clear()
        self.grouped_stats.clear()

    def get_cell_level_stats(self, df: pd.DataFrame, gene: str, feature: str) -> dict:
        key = (gene, feature)
        if key not in self.cell_level_stats:
            self.cell_level_stats[key] = _stats(
                df.loc[df['gene'] == gene, feature].values)
        return self.cell_level_stats[key]

    def get_grouped_stats(self, df: pd.DataFrame, grouped_gene: str, feature: str) -> dict:
        key = (grouped_gene, feature)
        if key not in self.grouped_stats:
            mask = df['gene'].map(get_grouped_gene_name) == grouped_gene
            self.grouped_stats[key] = _stats(df.loc[mask, feature].values)
        return self.grouped_stats[key]

    def get_well_level_stats(self, df: pd.DataFrame, gene: str, feature: str) -> list:
        key = (gene, feature)
        if key not in self.well_level_stats:
            self.well_level_stats[key] = [
                {'well': well, 'mean': float(np.mean(g[feature].values)),
                 'sd': float(np.std(g[feature].values, ddof=1)), 'n': len(g)}
                for well, g in df[df['gene'] == gene].groupby('well') if len(g) >= 2
            ]
        return self.well_level_stats[key]


# ============================================================================
# DATA LOADING  (single load — shared by both pipelines)
# ============================================================================
def load_and_prepare_data(cfg: Config) -> pd.DataFrame:
    """
    Load the parquet + plate-map exactly once and return a single DataFrame
    containing all columns required by both pipelines:

      Histogram pipeline:     gene (Categorical), well, fov
      Visualization pipeline: Label, Gene, Subgroup, Well

    gene == Gene == base gene name without subgroup suffix (e.g. 'mrcA', 'WT')
    Label          == full label string with subgroup index (e.g. 'mrcA_1', 'WT NC_3')
    well == Well == original well identifier
    """
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)

    datapath     = Path(cfg.DATA_FOLDER) / cfg.AGGREGATED_FILE

    # Derive plate number from DATA_FOLDER (e.g. '…\P1\…' → 'P_1_plate_map.csv')
    _plate_folder = Path(cfg.DATA_FOLDER).parent.name          # e.g. 'P1'
    _m = re.search(r'(?i)^P(\d+)', _plate_folder)
    if _m is None:
        raise ValueError(
            f"Cannot parse plate number from folder '{_plate_folder}'; "
            f"expected a name starting with P<N> (e.g. 'P1', 'P12').")
    _plate_num   = _m.group(1)                                 # e.g. '1'
    platemappath = Path(cfg.DATA_FOLDER).parent.parent / f"P_{_plate_num}_plate_map.csv"

    all_features      = list(set(cfg.MORPHOLOGY_FEATURES + cfg.FEATURES))
    requested_columns = ['Well', 'filename'] + all_features

    # ── Parquet: read only the columns the pipeline uses ─────────────────────
    print(f"  Loading {len(requested_columns)} columns...")
    try:
        df = pd.read_parquet(datapath, columns=requested_columns)
    except Exception:
        # Graceful fallback: load everything then trim to what exists
        df = pd.read_parquet(datapath)
        requested_columns = [c for c in requested_columns if c in df.columns]
        df = df[requested_columns]

    # Cast feature columns to float32 immediately — halves memory and
    # accelerates every downstream NumPy operation.
    feat_cols = [f for f in all_features if f in df.columns]
    df[feat_cols] = df[feat_cols].astype(np.float32)

    plate_map = pd.read_csv(platemappath, header=None)
    print(f"  Loaded: {len(df):,} cells  |  plate map: {plate_map.shape}")

    # ── FOV column ───────────────────────────────────────────────────────────
    if 'filename' in df.columns:
        df['fov'] = df['filename'].apply(extract_fov_from_filename)
        df.drop(columns='filename', inplace=True)
    else:
        df['fov'] = 'unknown'

    # ── Well → label mapping ─────────────────────────────────────────────────
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

    # Build a dict over unique well values only (not one call per row),
    # then use a single vectorised map — ~100× faster on large datasets.
    well_map     = {w: _well_to_label(w) for w in df['Well'].unique()}
    label_series = df['Well'].map(well_map)

    # ── Plate-map / parquet reconciliation (loud — previously silent) ────────
    plate_wells = {
        f'{chr(65 + r)}{c + 1:02d}'
        for r in range(plate_map.shape[0])
        for c in range(plate_map.shape[1])
        if pd.notna(plate_map.iat[r, c])
    }
    data_wells = {str(w) for w in df['Well'].dropna().unique()}
    missing_in_map  = data_wells - plate_wells
    missing_in_data = plate_wells - data_wells
    if missing_in_map:
        print(f'  WARN: {len(missing_in_map)} wells in parquet absent from plate map '
              f'(cells will be dropped): {sorted(missing_in_map)}')
    if missing_in_data:
        print(f'  WARN: {len(missing_in_data)} plate-map wells have no imaged cells: '
              f'{sorted(missing_in_data)}')
    dropped_rows = int(label_series.isna().sum())
    if dropped_rows:
        print(f'  WARN: {dropped_rows:,} rows have no plate-map label and will be dropped')

    # Parse Gene/Subgroup over unique label values only, then map.
    unique_labels = [l for l in label_series.unique() if pd.notna(l)]
    parsed        = {lbl: parse_gene_subgroup(str(lbl)) for lbl in unique_labels}
    gene_map      = {lbl: v[0] for lbl, v in parsed.items()}
    subgroup_map  = {lbl: v[1] for lbl, v in parsed.items()}

    # Single label derivation step — serves both pipelines.
    df['gene']     = pd.Categorical(label_series.map(gene_map))  # histogram pipeline
    df['well']     = df['Well']                                   # lowercase alias
    df['Label']    = label_series                                 # visualization pipeline
    df['Gene']     = label_series.map(gene_map)
    df['Subgroup'] = label_series.map(subgroup_map)

    # ── Feature clipping and validation ──────────────────────────────────────
    for feat, (lo, hi) in [('roundness', (0, 1)), ('solidity', (0, 1)),
                            ('eccentricity', (0, 1)), ('aspect_ratio', (1, 20))]:
        if feat in df.columns:
            df[feat] = df[feat].clip(lo, hi)

    for feat in all_features:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
            df = df[np.isfinite(df[feat])]

    # Per-gene quantile clipping in a single pass.
    # Previously: global, cumulative 1st/99th percentile trimming over 8 features.
    # That disproportionately removed morphologically distinct knockdowns
    # (e.g. filamentous ftsZ) — exactly the signal a classifier needs. Now each
    # gene is trimmed against its own distribution, and bounds are computed once
    # on the pre-filter data rather than recomputed after each cut.
    initial_count = len(df)
    df = df[df['gene'].notna()]
    feat_cols = [f for f in cfg.MORPHOLOGY_FEATURES if f in df.columns]
    if feat_cols:
        # Clip lower tail only, using WT-derived bounds applied uniformly to all
        # conditions.  Per-gene trimming would clip the upper tail of filamentous
        # knockdowns (e.g. ftsZ) — exactly the signal we want to keep.  Cells
        # below WT 1st percentile are segmentation debris; cells above WT max are
        # biological signal and must not be removed.
        wt_mask  = df['gene'] == 'WT'
        wt_lo    = df.loc[wt_mask, feat_cols].quantile(0.01)
        keep     = np.ones(len(df), dtype=bool)
        for feat in feat_cols:
            keep &= df[feat].values >= float(wt_lo[feat])
        df = df.loc[keep]

    del plate_map
    gc.collect()

    print(f"  After filtering: {len(df):,} cells  ({initial_count - len(df):,} removed)")
    print(f"  Unique genes: {df['gene'].nunique()}  |  "
          f"WT cells: {len(df[df['gene'] == 'WT']):,}  |  "
          f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df


# ============================================================================
# FOV AGGREGATION  (module-level, called exactly once in main)
# ============================================================================
def aggregate_fov(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Aggregate cell-level measurements to FOV level (well × fov × gene).

    Called exactly once in main(); the resulting DataFrame is passed directly
    to VariabilityAnalyzer and SpatialAnalyzer so neither class re-groups the
    full cell DataFrame by FOV on every feature iteration.

    Columns: well, fov, gene, n_cells, {feature}_mean, {feature}_std
    """
    feat_cols = [f for f in features if f in df.columns]
    fov_data = (
        df.groupby(['well', 'fov', 'gene'], observed=True, sort=False)
          .agg(
              n_cells=('well', 'count'),
              **{f'{f}_mean': (f, 'mean') for f in feat_cols},
              **{f'{f}_std':  (f, 'std')  for f in feat_cols},
          )
          .reset_index()
    )
    return fov_data


# ============================================================================
# EFFECT SIZE CALCULATOR
# ============================================================================
class EffectSizeCalculator:

    @staticmethod
    def cohens_d(group1, group2) -> float:
        """Pooled-SD Cohen's d using float32 for speed."""
        g1 = np.asarray(group1, dtype=np.float64)
        g2 = np.asarray(group2, dtype=np.float64)
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2:
            return np.nan
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        pooled_sd  = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return float((np.mean(g1) - np.mean(g2)) / pooled_sd) if pooled_sd != 0 else np.nan

    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        if np.isnan(d):  return 'undefined'
        abs_d = abs(d)
        if abs_d < 0.2:  return 'negligible'
        if abs_d < 0.5:  return 'small'
        if abs_d < 0.8:  return 'medium'
        return 'large'

    @staticmethod
    def bh_correction(p_values: np.ndarray) -> np.ndarray:
        """Benjamini-Hochberg FDR correction. Returns q-values (same length as input)."""
        n = len(p_values)
        if n == 0:
            return p_values.copy()
        rank  = np.argsort(p_values)
        q     = p_values[rank] * n / (np.arange(1, n + 1))
        # enforce monotonicity right-to-left
        for i in range(n - 2, -1, -1):
            q[i] = min(q[i], q[i + 1])
        result        = np.empty(n)
        result[rank]  = np.minimum(q, 1.0)
        return result

    @staticmethod
    def sig_stars(p: float) -> str:
        """Convert p-value to significance stars."""
        if np.isnan(p): return ''
        if p < 0.001:   return '***'
        if p < 0.01:    return '**'
        if p < 0.05:    return '*'
        return ''

    @staticmethod
    def cohens_d_ci(d: float, n1: int, n2: int,
                    ci: float = 0.95) -> Tuple[float, float]:
        """Approximate 95% CI for Cohen's d via the standard-error formula.

        SE(d) ≈ sqrt((n1+n2)/(n1*n2) + d²/(2*(n1+n2-2)))
        CI = d ± t_{df, alpha/2} * SE(d)
        """
        if np.isnan(d) or n1 < 2 or n2 < 2:
            return np.nan, np.nan
        df_   = n1 + n2 - 2
        se    = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * df_))
        alpha = 1.0 - ci
        t_crit = scipy_t.ppf(1.0 - alpha / 2.0, df_)
        return float(d - t_crit * se), float(d + t_crit * se)


# ============================================================================
# VARIABILITY ANALYZER
# ============================================================================
class VariabilityAnalyzer:
    """Plate-, well-, and FOV-level variability analysis."""

    def __init__(self, df: pd.DataFrame, fov_data: pd.DataFrame,
                 output_base: Path, cfg: Config):
        self.df          = df
        self.fov_data    = fov_data   # pre-aggregated by aggregate_fov()
        self.output_base = output_base
        self.cfg         = cfg

    def analyze_feature(self, feature: str):
        print(f"\n{'='*80}\nVARIABILITY QUANTIFICATION: {feature}\n{'='*80}")

        output_folder = self.output_base / feature / 'variability_analysis'
        output_folder.mkdir(parents=True, exist_ok=True)

        print(' -> Computing plate-level WT variability...')
        plate_metrics = self._compute_plate_level_variability(feature)

        print(' -> Computing well-level variability...')
        # wt_total_var passed in so _compute_well_level_variability does NOT
        # re-scan WT data from scratch for every gene (was O(n_genes) before).
        well_metrics = self._compute_well_level_variability(feature, plate_metrics['total_var'])

        print(' -> Computing FOV-level variability...')
        fov_metrics = self._compute_fov_level_variability(feature)

        print(' -> Generating cross-level summary...')
        self._generate_cross_level_summary(feature, plate_metrics, well_metrics, fov_metrics, output_folder)

        print(' -> Generating CV comparison plot...')
        self._generate_cv_comparison_plot(feature, plate_metrics, output_folder)

        print(' [OK] Variability analysis complete')

    def _compute_plate_level_variability(self, feature: str) -> dict:
        """Compute WT variability metrics across the entire plate (called once per feature)."""
        wt_df     = self.df[self.df['gene'] == 'WT']
        wt_values = wt_df[feature].values
        mean_val  = float(np.mean(wt_values))
        sd_val    = float(np.std(wt_values, ddof=1))

        well_means, within_well_vars = [], []
        for _, group in wt_df.groupby('well'):
            v = group[feature].values
            if len(v) >= 2:
                well_means.append(np.mean(v))
                within_well_vars.append(np.var(v, ddof=1))

        wt_fov = self.fov_data[(self.fov_data['gene'] == 'WT') &
                               (self.fov_data['n_cells'] >= 2)]
        fov_variations = [np.std(g[f'{feature}_mean'].values, ddof=1)
                          for _, g in wt_fov.groupby('well')
                          if len(g[f'{feature}_mean'].values) >= 2]

        q25 = float(np.percentile(wt_values, 25))
        q75 = float(np.percentile(wt_values, 75))
        return {
            'mean': mean_val, 'sd': sd_val,
            'cv': (sd_val / mean_val * 100) if mean_val != 0 else np.nan,
            'min': float(np.min(wt_values)), 'max': float(np.max(wt_values)),
            'median': float(np.median(wt_values)),
            'q25': q25, 'q75': q75, 'iqr': q75 - q25,
            'between_well_var': (float(np.var(well_means, ddof=1))
                                 if len(well_means) > 1 else 0.0),
            'within_well_var': (float(np.mean(within_well_vars))
                                if within_well_vars else 0.0),
            'total_var': float(np.var(wt_values, ddof=1)),
            'avg_fov_variation': (float(np.mean(fov_variations))
                                  if fov_variations else np.nan),
            'n_wells': len(well_means), 'n_cells': len(wt_values),
        }

    def _compute_well_level_variability(self, feature: str, wt_total_var: float) -> pd.DataFrame:
        """
        Per-gene well-level variability.
        Accepts pre-computed wt_total_var to avoid repeating the WT plate scan
        once per gene (the O(n_genes) redundancy in the original code).
        """
        results = []

        for gene in self.df['gene'].unique():
            gene_df = self.df[self.df['gene'] == gene]

            well_sds = [np.std(group[feature].values, ddof=1)
                        for _, group in gene_df.groupby('well') if len(group) >= 2]
            avg_intra_well_sd = float(np.mean(well_sds)) if well_sds else np.nan

            base_gene, subgroup = parse_gene_subgroup(gene)
            if subgroup is not None:
                sg_genes = [g for g in self.df['gene'].unique()
                            if get_grouped_gene_name(g) == base_gene]
                if len(sg_genes) > 1:
                    rep_means = [float(np.mean(self.df.loc[self.df['gene'] == sg, feature].values))
                                 for sg in sg_genes]
                    replicate_cv    = (np.std(rep_means, ddof=1) / np.mean(rep_means) * 100
                                       if len(rep_means) > 1 else np.nan)
                    replicate_range = float(np.max(rep_means) - np.min(rep_means))
                else:
                    replicate_cv = replicate_range = np.nan
            else:
                replicate_cv = replicate_range = np.nan

            gene_var  = float(np.var(gene_df[feature].values, ddof=1))
            var_ratio = (gene_var / wt_total_var) if wt_total_var != 0 else np.nan

            results.append({
                'gene':               gene,
                'avg_intra_well_sd':  avg_intra_well_sd,
                'replicate_cv':       replicate_cv,
                'replicate_range':    replicate_range,
                'var_ratio_vs_wt':    var_ratio,
                'n_wells':            len(well_sds),
            })

        return pd.DataFrame(results)

    def _compute_fov_level_variability(self, feature: str) -> pd.DataFrame:
        """Per-gene FOV-level variability using pre-aggregated fov_data."""
        results  = []
        mean_col = f'{feature}_mean'

        for gene in self.df['gene'].unique():
            gene_fov = self.fov_data[
                (self.fov_data['gene'] == gene) & (self.fov_data['n_cells'] >= 2)
            ]
            fov_heterogeneity, fov_counts, cells_per_fov = [], [], []

            for _, well_group in gene_fov.groupby('well'):
                means = well_group[mean_col].values
                if len(means) >= 2:
                    fov_heterogeneity.append(np.std(means, ddof=1))
                    fov_counts.append(len(means))
                cells_per_fov.extend(well_group['n_cells'].tolist())

            results.append({
                'gene':                  gene,
                'avg_fov_heterogeneity': float(np.mean(fov_heterogeneity)) if fov_heterogeneity else np.nan,
                'avg_fovs_per_well':     float(np.mean(fov_counts))         if fov_counts         else np.nan,
                'avg_cells_per_fov':     float(np.mean(cells_per_fov))      if cells_per_fov      else np.nan,
            })

        return pd.DataFrame(results)

    def _generate_cross_level_summary(self, feature, plate_metrics, well_metrics,
                                      fov_metrics, output_folder: Path):
        unit = self.cfg.FEATURE_UNITS.get(feature, '')
        pm, sep, dash = plate_metrics, '=' * 80, '-' * 80
        bw_pct = pm['between_well_var'] / pm['total_var'] * 100
        ww_pct = pm['within_well_var'] / pm['total_var'] * 100
        wt_row = well_metrics[well_metrics['gene'] == 'WT']
        wt_fov = fov_metrics[fov_metrics['gene'] == 'WT']

        lines = [
            sep, f'CROSS-LEVEL VARIABILITY SUMMARY: {feature}', sep, '',
            'PLATE LEVEL (WT Baseline Variability)', dash,
            f" Mean ± SD: {pm['mean']:.3f} ± {pm['sd']:.3f} {unit}",
            f" Coefficient of Variation: {pm['cv']:.2f}%",
            f" Range: {pm['min']:.3f} - {pm['max']:.3f} {unit}",
            f" Median [IQR]: {pm['median']:.3f} [{pm['q25']:.3f} - {pm['q75']:.3f}] {unit}",
            f" Sample: n={pm['n_cells']:,} cells from {pm['n_wells']} wells", '',
            ' Variance Decomposition:',
            f"  Total:        {pm['total_var']:.4f}",
            f"  Between-well: {pm['between_well_var']:.4f} ({bw_pct:.1f}%)",
            f"  Within-well:  {pm['within_well_var']:.4f} ({ww_pct:.1f}%)",
            f"  Avg FOV-to-FOV variation: {pm['avg_fov_variation']:.4f}", '',
            'WELL LEVEL VARIABILITY', dash,
        ]
        if len(wt_row) > 0:
            lines.append(f" WT average within-well SD: "
                         f"{wt_row.iloc[0]['avg_intra_well_sd']:.4f} {unit}")
        lines.append('')
        lines.append(' Top 5 genes by within-well variability:')
        for _, row in well_metrics.nlargest(5, 'avg_intra_well_sd').iterrows():
            lines.append(f"  {row['gene']}: SD = {row['avg_intra_well_sd']:.4f} {unit}")
        lines += ['', 'FOV LEVEL VARIABILITY', dash]
        if len(wt_fov) > 0:
            r = wt_fov.iloc[0]
            lines += [
                f" WT average FOV heterogeneity: {r['avg_fov_heterogeneity']:.4f} {unit}",
                f" Average FOVs per well: {r['avg_fovs_per_well']:.1f}",
                f" Average cells per FOV: {r['avg_cells_per_fov']:.1f}",
            ]

        (output_folder / f'cross_level_summary_{feature}.txt').write_text(
            '\n'.join(lines) + '\n', encoding='utf-8')
        well_metrics.to_csv(output_folder / f'well_level_variability_{feature}.csv', index=False)
        fov_metrics.to_csv(output_folder / f'fov_level_variability_{feature}.csv', index=False)

    def _generate_cv_comparison_plot(self, feature: str, plate_metrics: dict, output_folder: Path):
        """CV bar chart across hierarchical levels using WT data only."""
        wt_df = self.df[self.df['gene'] == 'WT']

        def _mean_cv_well(groups) -> float:
            cvs = []
            for _, group in groups:
                if len(group) >= 2:
                    m = np.mean(group[feature].values)
                    s = np.std(group[feature].values, ddof=1)
                    if m != 0:
                        cvs.append(s / m * 100)
            return float(np.mean(cvs)) if cvs else np.nan

        # FOV-level CV: use pre-aggregated fov_data (mean and std already computed)
        wt_fov  = self.fov_data[(self.fov_data['gene'] == 'WT') &
                                 (self.fov_data['n_cells'] >= 2)]
        fov_cvs = (wt_fov[f'{feature}_std'] / wt_fov[f'{feature}_mean'] * 100).dropna()
        fov_cv  = float(fov_cvs.mean()) if len(fov_cvs) > 0 else np.nan

        levels = ['Plate\n(All WT)', 'Well\n(Avg)', 'FOV\n(Avg)']
        cvs    = [plate_metrics['cv'], _mean_cv_well(wt_df.groupby('well')), fov_cv]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.cfg.DPI)
        bars = ax.bar(levels, cvs, color=['#2E86AB', '#A23B72', '#F18F01'],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        for bar, cv in zip(bars, cvs):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{cv:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Variability Across Hierarchical Levels: {feature}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        valid_cvs = [v for v in cvs if not np.isnan(v)]
        if valid_cvs:
            ax.set_ylim(0, max(valid_cvs) * 1.2)
        _save_fig(fig, output_folder / f'cv_comparison_{feature}.png', self.cfg, facecolor='white')


# ============================================================================
# SPATIAL ANALYZER
# ============================================================================
class SpatialAnalyzer:
    """Spatial distribution of cell counts and effect sizes within wells."""

    def __init__(self, df: pd.DataFrame, fov_data: pd.DataFrame,
                 output_base: Path, cfg: Config):
        self.df          = df
        self.fov_data    = fov_data   # pre-aggregated by aggregate_fov()
        self.output_base = output_base
        self.cfg         = cfg

    def analyze_feature(self, feature: str):
        print(f"\n{'='*80}\nSPATIAL DISTRIBUTION ANALYSIS: {feature}\n{'='*80}")

        output_folder = self.output_base / feature / 'spatial_analysis'
        output_folder.mkdir(parents=True, exist_ok=True)

        wt_values = self.df.loc[self.df['gene'] == 'WT', feature].values

        tasks = []
        for gene in self.df['gene'].unique():
            gene_folder = output_folder / gene
            gene_folder.mkdir(exist_ok=True)
            for well, well_group in self.df[self.df['gene'] == gene].groupby('well'):
                tasks.append((well_group, well, gene, feature, wt_values, gene_folder))

        print(' -> Generating spatial plots for each well...')
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(self._generate_spatial_plots_for_well, *t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=' Processing wells', leave=False):
                try:
                    future.result()
                except Exception as e:
                    print(f'   Error processing well: {e}')

        # Build positional_data from pre-aggregated fov_data — avoids any
        # re-groupby on the full cell DataFrame.
        positional_data = {i: [] for i in range(21)}
        for _, row in self.fov_data.iterrows():
            try:
                fov_idx = int(row['fov'])
                if 0 <= fov_idx <= 20:
                    positional_data[fov_idx].append(int(row['n_cells']))
            except (ValueError, TypeError):
                continue

        print(' -> Generating averaged positional heatmap...')
        self._generate_averaged_positional_heatmap(positional_data, feature, output_folder)
        gc.collect()
        print(' [OK] Spatial analysis complete')

    def _generate_spatial_plots_for_well(self, well_df: pd.DataFrame, well: str,
                                          gene: str, feature: str,
                                          wt_values: np.ndarray, output_folder: Path):
        fov_stats = [
            {
                'fov':      fov,
                'n_cells':  len(fg),
                'cohens_d': EffectSizeCalculator.cohens_d(fg[feature].values, wt_values),
            }
            for fov, fg in well_df.groupby('fov')
        ]
        if not fov_stats:
            return

        fov_df     = pd.DataFrame(fov_stats).sort_values('fov')
        fov_labels = fov_df['fov'].astype(str)

        fig      = Figure(figsize=(12, 8), dpi=self.cfg.DPI)
        ax1, ax2 = fig.subplots(2, 1)

        ax1.bar(range(len(fov_df)), fov_df['n_cells'], color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('FOV', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Cells', fontsize=11, fontweight='bold')
        ax1.set_title(f'{gene} - Well {well}: Cell Count Distribution', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(fov_df)))
        ax1.set_xticklabels(fov_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        mean_cells = fov_df['n_cells'].mean()
        ax1.axhline(y=mean_cells, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_cells:.1f}')
        ax1.legend()

        bar_colors = ['green' if d > 0 else 'red' for d in fov_df['cohens_d']]
        ax2.bar(range(len(fov_df)), fov_df['cohens_d'], color=bar_colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('FOV', fontsize=11, fontweight='bold')
        ax2.set_ylabel("Cohen's d vs WT", fontsize=11, fontweight='bold')
        ax2.set_title(f'{gene} - Well {well}: Effect Size Distribution', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(fov_df)))
        ax2.set_xticklabels(fov_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        for threshold, color, label in [(0.2, 'orange', 'Small effect'), (0.5, 'red', 'Medium effect')]:
            ax2.axhline( threshold, color=color, linestyle='--', linewidth=1, alpha=0.5, label=label)
            ax2.axhline(-threshold, color=color, linestyle='--', linewidth=1, alpha=0.5)
        ax2.legend()

        _save_fig(fig, output_folder / f'well_{well}_spatial.png', self.cfg)

    def _generate_averaged_positional_heatmap(self, positional_data: dict,
                                               feature: str, output_folder: Path):
        n_rows, n_cols = 5, 5
        grid_mean = np.full((n_rows, n_cols), np.nan)
        grid_std  = np.full((n_rows, n_cols), np.nan)

        for fov_idx in range(21):
            counts = positional_data.get(fov_idx, [])
            if counts:
                pos = fov_index_to_position(fov_idx)
                if pos:
                    grid_mean[pos] = np.mean(counts)
                    grid_std[pos]  = np.std(counts)

        fig = plt.figure(figsize=(14, 7), dpi=self.cfg.DPI)
        gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)

        ax1  = fig.add_subplot(gs[0])
        cmap = mpl_cm.viridis.copy()
        cmap.set_bad(color='lightgray')
        im   = ax1.imshow(grid_mean, cmap=cmap, aspect='equal', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Average Cell Count', fontsize=11, fontweight='bold')

        for i in range(n_rows):
            for j in range(n_cols):
                if not np.isnan(grid_mean[i, j]):
                    fov_idx = next((k for k in range(21) if fov_index_to_position(k) == (i, j)), None)
                    ax1.text(j, i + 0.15, f'{grid_mean[i,j]:.0f}\n±{grid_std[i,j]:.0f}',
                             ha='center', va='center', color='white', fontweight='bold', fontsize=9)
                    if fov_idx is not None:
                        ax1.text(j, i - 0.35, f'FOV{fov_idx}',
                                 ha='center', va='center', color='yellow', fontsize=7, alpha=0.9)

        ax1.set_xticks(np.arange(n_cols))
        ax1.set_yticks(np.arange(n_rows))
        ax1.set_xticklabels(range(n_cols))
        ax1.set_yticklabels(range(n_rows))
        ax1.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
        ax1.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
        ax1.grid(which='minor', color='white', linestyle='-', linewidth=2)
        ax1.tick_params(which='minor', size=0)
        ax1.set_xlabel('Column', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Row',    fontsize=11, fontweight='bold')
        ax1.set_title('Averaged FOV Cell Counts Across All Wells\n(Meandering Pattern: 21 FOVs)',
                      fontsize=12, fontweight='bold')

        ax2 = fig.add_subplot(gs[1])
        valid = [(k, np.mean(positional_data[k]), np.std(positional_data[k]))
                 for k in range(21) if positional_data.get(k)]
        if valid:
            idxs, means, stds = zip(*valid)
            ax2.bar(range(len(idxs)), means, yerr=stds,
                    color='steelblue', alpha=0.7, edgecolor='black', capsize=3)
            ax2.axhline(y=np.mean(means), color='red', linestyle='--', linewidth=2,
                        label=f'Overall Mean: {np.mean(means):.1f}')
            ax2.legend()
            ax2.set_xticks(range(len(idxs)))
            ax2.set_xticklabels(idxs, rotation=45, ha='right')
        ax2.set_xlabel('FOV Index',          fontsize=11, fontweight='bold')
        ax2.set_ylabel('Average Cell Count', fontsize=11, fontweight='bold')
        ax2.set_title('Cell Count by FOV Position', fontsize=12, fontweight='bold')
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

        _save_fig(fig, output_folder / f'averaged_positional_heatmap_{feature}.png', self.cfg)


# ============================================================================
# FOV-LEVEL ANALYZER
# ============================================================================
class FOVLevelAnalyzer:
    def __init__(self, df: pd.DataFrame, output_base: Path, cfg: Config, cache: StatisticsCache):
        self.df          = df
        self.output_base = output_base
        self.cfg         = cfg
        self.cache       = cache

    def analyze_feature(self, feature: str) -> pd.DataFrame:
        print(f"\n{'='*80}\nFOV-LEVEL ANALYSIS: {feature}\n{'='*80}")

        output_folder = self.output_base / feature / 'fov_level'
        output_folder.mkdir(parents=True, exist_ok=True)

        print(' -> Computing effect sizes...')
        effect_df = self._compute_fov_effect_sizes(feature)
        effect_df.to_csv(output_folder / f'effect_size_fov_summary_{feature}.csv', index=False)

        print(' -> Generating FOV effect size strip plot...')
        self._plot_fov_effect_sizes(effect_df, feature, output_folder)

        print(' [OK] FOV-level analysis complete')
        return effect_df

    def _compute_fov_effect_sizes(self, feature: str) -> pd.DataFrame:
        wt_values = self.cache.get_cell_level_stats(self.df, 'WT', feature)['values']
        results   = []

        for gene in self.df['gene'].unique():
            if gene == 'WT':
                continue
            for (well, fov), group in self.df[self.df['gene'] == gene].groupby(['well', 'fov']):
                fov_values = group[feature].values
                if len(fov_values) < 10:
                    continue
                d = EffectSizeCalculator.cohens_d(fov_values, wt_values)
                results.append({
                    'gene':           gene,
                    'well':           well,
                    'fov':            fov,
                    'n_cells':        len(fov_values),
                    'mean':           float(np.mean(fov_values)),
                    'sd':             float(np.std(fov_values, ddof=1)),
                    'cohens_d':       d,
                    'interpretation': EffectSizeCalculator.interpret_cohens_d(d),
                })

        return pd.DataFrame(results)

    def _plot_fov_effect_sizes(self, effect_df: pd.DataFrame, feature: str, output_folder: Path):
        """Strip plot of per-FOV Cohen's d values sorted by median effect size."""
        if effect_df.empty:
            return

        genes = sorted(effect_df['gene'].unique(), key=label_sort_key)
        gene_medians = (
            effect_df.groupby('gene')['cohens_d']
            .median()
            .reindex(genes)
            .fillna(0)
        )
        sorted_genes = gene_medians.sort_values().index.tolist()

        fig, ax = plt.subplots(figsize=(max(12, len(sorted_genes) * 0.65), 6), dpi=self.cfg.DPI)

        for i, gene in enumerate(sorted_genes):
            vals  = effect_df.loc[effect_df['gene'] == gene, 'cohens_d'].dropna().values
            color = get_gene_color(gene)
            # jitter x slightly so overlapping dots are visible
            jitter = np.random.default_rng(42).uniform(-0.25, 0.25, len(vals))
            ax.scatter(i + jitter, vals, color=color, alpha=0.55, s=18, zorder=3)
            median = float(np.median(vals))
            ax.plot([i - 0.35, i + 0.35], [median, median], color=color, linewidth=2.5, zorder=4)

        ax.axhline(0, color='black', linewidth=1.5, zorder=2)
        ax.set_xticks(range(len(sorted_genes)))
        ax.set_xticklabels(sorted_genes, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Cohen's d vs WT (per FOV)", fontsize=11, fontweight='bold')
        ax.set_title(f'{feature} — Per-FOV Effect Sizes\n'
                     f'(dots = individual FOVs, bar = median; sorted by median)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        _save_fig(fig, output_folder / f'fov_effect_sizes_strip_{feature}.png', self.cfg)


# ============================================================================
# WELL-LEVEL ANALYZER
# ============================================================================
class WellLevelAnalyzer:
    def __init__(self, df: pd.DataFrame, output_base: Path, cfg: Config, cache: StatisticsCache):
        self.df          = df
        self.output_base = output_base
        self.cfg         = cfg
        self.cache       = cache

    def analyze_feature(self, feature: str) -> pd.DataFrame:
        print(f"{'='*80}\nWELL-LEVEL ANALYSIS — {feature}\n{'='*80}")

        output_folder = self.output_base / feature / 'well_level'
        output_folder.mkdir(parents=True, exist_ok=True)

        print(' -> Computing well-level statistics...')
        effect_df = self._compute_well_effect_sizes(feature)

        print(' -> Generating density plots...')
        self._generate_well_density_plots(feature, output_folder)

        effect_df.to_csv(output_folder / f'effect_size_well_summary_{feature}.csv', index=False)
        self._write_well_summary(effect_df, feature, output_folder / f'effect_size_well_summary_{feature}.txt')
        print(' OK Saved summaries')
        return effect_df

    def _generate_well_density_plots(self, feature: str, output_folder: Path):
        bins  = _bins_for(self.df, feature, self.cfg)
        genes = list(self.df['gene'].unique())

        def _plot_gene(gene: str):
            gene_folder = output_folder / gene
            gene_folder.mkdir(exist_ok=True)
            for well, group in self.df[self.df['gene'] == gene].groupby('well'):
                values = group[feature].values
                if len(values) < 30:
                    continue
                fig = Figure(figsize=(8, 4), dpi=self.cfg.DPI)
                ax  = fig.add_subplot(1, 1, 1)
                ax.hist(values, bins=bins, alpha=self.cfg.HISTOGRAM_ALPHA, density=True,
                        color='steelblue', edgecolor='black', linewidth=0.5)
                unit = self.cfg.FEATURE_UNITS.get(feature, '')
                ax.set_xlabel(f'{feature} {unit}'.strip(), fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title(f'{gene} - Well {well}\nn={len(values)}', fontsize=12, weight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                _save_fig(fig, gene_folder / f'{gene}_well_{well}_{feature}.png', self.cfg)

        with ThreadPoolExecutor(max_workers=12) as executor:
            list(tqdm(executor.map(_plot_gene, genes),
                      total=len(genes), desc=' Genes', leave=False))

    def _compute_well_effect_sizes(self, feature: str) -> pd.DataFrame:
        wt_values = self.cache.get_cell_level_stats(self.df, 'WT', feature)['values']
        results   = []

        for gene in self.df['gene'].unique():
            if gene == 'WT':
                continue
            for well, group in self.df[self.df['gene'] == gene].groupby('well'):
                well_values = group[feature].values
                if len(well_values) < 30:
                    continue
                d = EffectSizeCalculator.cohens_d(well_values, wt_values)
                results.append({
                    'gene':           gene,
                    'well':           well,
                    'n_cells':        len(well_values),
                    'mean':           float(np.mean(well_values)),
                    'sd':             float(np.std(well_values, ddof=1)),
                    'cohens_d':       d,
                    'interpretation': EffectSizeCalculator.interpret_cohens_d(d),
                })

        return pd.DataFrame(results)

    def _write_well_summary(self, effect_df: pd.DataFrame, feature: str, output_path: Path):
        wt_stats = self.cache.get_cell_level_stats(self.df, 'WT', feature)
        unit    = self.cfg.FEATURE_UNITS.get(feature, '')
        sep, dash = '=' * 80, '-' * 80
        lines = [sep, f'WELL-LEVEL ANALYSIS SUMMARY — {feature}', sep,
                 'WT REFERENCE',
                 f'  {feature}: {wt_stats["mean"]:.3f} ± {wt_stats["sd"]:.3f} {unit}',
                 f'  n={wt_stats["n"]} cells',
                 sep + ' COMPARISONS TO WT ' + sep]
        for _, row in effect_df.sort_values('cohens_d', key=abs, ascending=False).iterrows():
            lines += [dash, f'{row["gene"]}, Well {row["well"]}', dash,
                      f'  {feature}: {row["mean"]:.3f} ± {row["sd"]:.3f} {unit}',
                      f'  n={row["n_cells"]} cells',
                      f"  Cohen's d vs WT: {row['cohens_d']:.3f} ({row['interpretation']})"]
        Path(output_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ============================================================================
# WT COMPARISON ANALYZER
# ============================================================================
class WTComparisonAnalyzer:
    def __init__(self, df: pd.DataFrame, output_base: Path, cfg: Config, cache: StatisticsCache):
        self.df          = df
        self.output_base = output_base
        self.cfg         = cfg
        self.cache       = cache

    def analyze_feature(self, feature: str) -> pd.DataFrame:
        print(f"\n{'='*80}\nWT COMPARISON ANALYSIS: {feature}\n{'='*80}")

        output_folder = self.output_base / feature / 'wt_comparisons'
        output_folder.mkdir(parents=True, exist_ok=True)

        print(' -> Computing gene-level effect sizes + p-values...')
        effect_df = self._compute_gene_effect_sizes(feature)

        print(' -> Generating comparison plots...')
        self._generate_comparison_plots(feature, output_folder)

        print(' -> Generating grouped matrix plot...')
        self._generate_grouped_matrix_plot(feature, output_folder, effect_df)

        effect_df.to_csv(output_folder / f'effect_size_gene_summary_{feature}.csv', index=False)
        self._write_gene_summary(effect_df, feature, output_folder / f'effect_size_gene_summary_{feature}.txt')
        print(' [OK] Saved summaries')
        return effect_df

    def _compute_gene_effect_sizes(self, feature: str) -> pd.DataFrame:
        wt_values = self.cache.get_cell_level_stats(self.df, 'WT', feature)['values']
        n_wt      = len(wt_values)

        # Well-level WT means — the valid statistical unit for the hypothesis test.
        wt_well_means = np.array([
            grp[feature].mean()
            for _, grp in self.df[self.df['gene'] == 'WT'].groupby('well')
            if len(grp) >= 5
        ])

        results = []

        for gene in self.df['gene'].unique():
            if gene == 'WT':
                continue
            gene_stats = self.cache.get_cell_level_stats(self.df, gene, feature)
            if gene_stats['n'] < 30:
                continue

            # Cell-level Cohen's d (magnitude descriptor) with 95% CI.
            d = EffectSizeCalculator.cohens_d(gene_stats['values'], wt_values)
            d_lo, d_hi = EffectSizeCalculator.cohens_d_ci(d, gene_stats['n'], n_wt)

            # Per-well means for this gene — the actual unit of replication.
            gene_well_means = np.array([
                grp[feature].mean()
                for _, grp in self.df[self.df['gene'] == gene].groupby('well')
                if len(grp) >= 5
            ])

            # Well-level Mann-Whitney U — statistically valid test.
            # p-values here are honest: n_wells ≈ 3 vs 6, not thousands of cells.
            if len(gene_well_means) >= 2 and len(wt_well_means) >= 2:
                try:
                    _, p = mannwhitneyu(gene_well_means, wt_well_means,
                                        alternative='two-sided')
                except Exception:
                    p = np.nan
            else:
                p = np.nan

            # gRNA concordance: fraction of wells agreeing in direction with
            # the pooled Cohen's d.  Requires ≥2/3 wells to flag as concordant.
            n_wells = len(gene_well_means)
            if n_wells > 0 and not np.isnan(d):
                wt_grand = float(np.mean(wt_well_means)) if len(wt_well_means) > 0 else 0.0
                n_agree  = int(np.sum(np.sign(gene_well_means - wt_grand) == np.sign(d)))
                concordance      = n_agree / n_wells
                concordance_flag = n_agree >= max(2, round(n_wells * 0.67))
            else:
                concordance = concordance_flag = np.nan

            # Depletion: low mean cells-per-well → survivor-selection bias warning.
            # Genes below DEPLETION_THRESHOLD reflect a sub-population that survived
            # knockdown, not the depleted population.
            mean_cpw       = gene_stats['n'] / n_wells if n_wells > 0 else np.nan
            depletion_flag = (
                bool(mean_cpw < self.cfg.DEPLETION_THRESHOLD_CELLS_PER_WELL)
                if not np.isnan(mean_cpw) else False
            )

            results.append({
                'gene':             gene,
                'n_cells':          gene_stats['n'],
                'n_wells':          n_wells,
                'mean':             gene_stats['mean'],
                'sd':               gene_stats['sd'],
                'cohens_d':         d,
                'cohens_d_ci_lo':   d_lo,
                'cohens_d_ci_hi':   d_hi,
                'interpretation':   EffectSizeCalculator.interpret_cohens_d(d),
                'p_value':          p,
                'concordance':      concordance,
                'concordance_flag': concordance_flag,
                'mean_cells_per_well': mean_cpw,
                'depletion_flag':      depletion_flag,
            })

        df_out = pd.DataFrame(results)

        # BH-FDR correction on well-level p-values, within this feature.
        # Note: correction is per-feature (~30 tests); see heatmap title for caveat.
        if len(df_out) > 1:
            valid = df_out['p_value'].notna()
            if valid.sum() > 1:
                q_vals = EffectSizeCalculator.bh_correction(
                    df_out.loc[valid, 'p_value'].values
                )
                df_out.loc[valid, 'q_value'] = q_vals
            else:
                df_out['q_value'] = df_out['p_value']
        elif len(df_out) == 1:
            df_out['q_value'] = df_out['p_value']

        return df_out

    def _generate_comparison_plots(self, feature: str, output_folder: Path):
        wt_values = self.cache.get_cell_level_stats(self.df, 'WT', feature)['values']
        bins      = _bins_for(self.df, feature, self.cfg)
        genes     = [g for g in self.df['gene'].unique() if g != 'WT']

        def _plot(gene: str):
            gene_stats = self.cache.get_cell_level_stats(self.df, gene, feature)
            if gene_stats['n'] < 30:
                return
            d   = EffectSizeCalculator.cohens_d(gene_stats['values'], wt_values)
            fig = Figure(figsize=self.cfg.FIGURE_SIZE, dpi=self.cfg.DPI)
            ax  = fig.add_subplot(1, 1, 1)
            ax.hist(wt_values,           bins=bins, alpha=0.5, density=True, label='WT',
                    color='gray', edgecolor='black', linewidth=0.5)
            ax.hist(gene_stats['values'], bins=bins, alpha=0.5, density=True, label=gene,
                    color='red',  edgecolor='black', linewidth=0.5)
            unit = self.cfg.FEATURE_UNITS.get(feature, '')
            ax.set_xlabel(f'{feature} {unit}'.strip(), fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(
                f"{gene} vs WT  |  Cohen's d={d:.2f}  {EffectSizeCalculator.interpret_cohens_d(d)}",
                fontsize=12, weight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            _save_fig(fig, output_folder / f'{gene}_vs_WT_{feature}.png', self.cfg)

        with ThreadPoolExecutor(max_workers=12) as executor:
            list(tqdm(executor.map(_plot, genes), total=len(genes), desc=' Genes', leave=False))

    def _write_gene_summary(self, effect_df: pd.DataFrame, feature: str, output_path: Path):
        wt = self.cache.get_cell_level_stats(self.df, 'WT', feature)
        unit = self.cfg.FEATURE_UNITS.get(feature, '')
        bar = '=' * 80

        # NC noise floor: expected d even for non-cutting controls
        nc_rows = effect_df[effect_df['gene'] == self.cfg.NC_LABEL]
        nc_floor_str = ''
        if len(nc_rows):
            nc_d = float(nc_rows['cohens_d'].iloc[0])
            nc_floor_str = (f'\nNC ASSAY NOISE FLOOR: |d|={abs(nc_d):.3f} for NC vs WT.\n'
                            f'  Effects ≤ this magnitude may reflect plate/batch noise, '
                            f'not true biology.\n'
                            f'  NC is a genuinely different condition from WT NC wells '
                            f'(concordance=1.0 on most features).')

        lines = [bar, f'GENE-LEVEL ANALYSIS SUMMARY: {feature}', bar, '',
                 'WT REFERENCE:',
                 f" {feature}: {wt['mean']:.3f} ± {wt['sd']:.3f} {unit}",
                 f" (n={wt['n']:,} cells)", '']
        if nc_floor_str:
            lines += [nc_floor_str, '']
        lines += [
            'NOTE: p/q-values use well-level Mann-Whitney (n_wells per gene vs n_wells WT).',
            '  Cohen\'s d and 95% CI are cell-level descriptors (magnitude only).',
            '  With n_wells=3, MWU p-floor = 0.0238 — all ordered triplets share this p-value.',
            '  Use effect size (Cohen\'s d) as the primary ranking criterion.',
            '  † concordance_flag=False means <2/3 gRNAs agree in direction.',
            '  ‡ depletion_flag=True means <5,000 cells/well (survivor-selection bias).',
            bar, 'GENE COMPARISONS TO WT (sorted by |d|):', bar, '']

        for _, row in effect_df.sort_values('cohens_d', key=abs, ascending=False).iterrows():
            gene = row['gene']
            ci_lo = row.get('cohens_d_ci_lo', np.nan)
            ci_hi = row.get('cohens_d_ci_hi', np.nan)
            ci_str = (f" [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]"
                      if not (pd.isna(ci_lo) or pd.isna(ci_hi)) else '')
            concordance = row.get('concordance', np.nan)
            conc_flag = row.get('concordance_flag', np.nan)
            conc_str = (
                f" concordance={concordance:.2f}"
                f"{' †' if conc_flag is False or conc_flag == 0 else ''}"
                if not pd.isna(concordance) else ''
            )

            dep_flag = row.get('depletion_flag', False)
            mean_cpw = row.get('mean_cells_per_well', np.nan)
            depletion_str = (
                f' ‡ DEPLETION WARNING: {mean_cpw:.0f} cells/well '
                f'(threshold {self.cfg.DEPLETION_THRESHOLD_CELLS_PER_WELL:,}). '
                f'Effect sizes reflect survivor sub-population.'
                if dep_flag else ''
            )

            seg_note = self.cfg.KNOWN_SEGMENTATION_ISSUES.get(gene, '')
            seg_str = f' ⚠ SEGMENTATION NOTE: {seg_note}' if seg_note else ''

            header_markers = (''.join([
                ' ‡' if dep_flag else '',
                ' ⚠' if seg_note else '',
            ])).strip()
            lines += ['', '-' * 80,
                      f"Gene: {gene}{(' ' + header_markers) if header_markers else ''}  "
                      f"(n_wells={int(row.get('n_wells', 0))})",
                      '-' * 80,
                      f" {feature}: {row['mean']:.3f} ± {row['sd']:.3f} {unit}",
                      f" (n={row['n_cells']:,} cells)",
                      f" Cohen's d vs WT: {row['cohens_d']:.3f} ({row['interpretation']})"
                      f"{ci_str}{conc_str}"]
            if 'p_value' in row and not pd.isna(row['p_value']):
                lines.append(f" Well-level MWU p = {row['p_value']:.2e}  |  "
                             f"q (BH-FDR) = {row.get('q_value', np.nan):.2e}")
            if depletion_str:
                lines.append(depletion_str)
            if seg_str:
                lines.append(seg_str)

        output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def _generate_grouped_matrix_plot(self, feature: str, output_folder: Path,
                                       effect_df: pd.DataFrame):
        grouped = sorted({get_grouped_gene_name(g) for g in self.df['gene'].unique() if g != 'WT'})
        if not grouped:
            return

        q_lookup: dict = {}
        if 'q_value' in effect_df.columns:
            for gg in grouped:
                rows = effect_df[effect_df['gene'].map(get_grouped_gene_name) == gg]
                if len(rows):
                    q_lookup[gg] = float(rows['q_value'].min())

        wt_values = self.cache.get_cell_level_stats(self.df, 'WT', feature)['values']
        bins = _bins_for(self.df, feature, self.cfg)
        unit = self.cfg.FEATURE_UNITS.get(feature, '')
        n_cols = len(grouped)
        fig, axes = plt.subplots(1, n_cols, figsize=(max(16, 3 * n_cols), 4), dpi=self.cfg.DPI)
        axes = np.array(axes).reshape(1, -1)

        for idx, gg in enumerate(grouped):
            ax = axes[0, idx]
            gene_values = self.cache.get_grouped_stats(self.df, gg, feature)['values']
            d = EffectSizeCalculator.cohens_d(gene_values, wt_values)
            stars = EffectSizeCalculator.sig_stars(q_lookup.get(gg, np.nan))
            for vals, lbl, col in [(wt_values, 'WT', 'gray'), (gene_values, gg, 'red')]:
                ax.hist(vals, bins=bins, alpha=0.5, density=True, label=lbl,
                        color=col, edgecolor='black', linewidth=0.3)
            ax.set_xlabel(f'{feature} {unit}'.strip(), fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title(f'{gg} | d={d:.2f} ({EffectSizeCalculator.interpret_cohens_d(d)}) {stars}',
                         fontsize=10, weight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.2, linestyle='--')

        _save_fig(fig, output_folder / f'grouped_genes_matrix_{feature}.png', self.cfg)


# ============================================================================
# VISUALIZER  (by label)
# ============================================================================
class Visualizer:
    def __init__(self, analysis_folder: str, cfg: Config):
        self.analysis_folder = analysis_folder
        self.cfg             = cfg

    def plot_violin(self, data: pd.DataFrame, feature: str):
        print(f'  Generating {feature} violin plot...', end=' ')
        unique_labels = sorted(data['Label'].unique(), key=label_sort_key)
        color_dict    = {lbl: get_gene_color(parse_gene_subgroup(lbl)[0]) for lbl in unique_labels}

        fig, ax = plt.subplots(1, 1, figsize=self.cfg.FIGURE_SIZE_STANDARD)
        _draw_violin(ax, _clip_q(data, feature), 'Label', feature,
                     unique_labels, color_dict, f'{feature} Distribution per Gene')
        _save_fig(fig, os.path.join(self.analysis_folder, f'{feature}_violin.png'), self.cfg)
        print('✓')

    def plot_cell_counts_normalized(self, data: pd.DataFrame, wells_per_gene: pd.Series) -> pd.DataFrame:
        print('  Generating normalized cell counts...', end=' ')
        raw       = data['Label'].value_counts()
        df_counts = pd.DataFrame({
            'Label':         raw.index,
            'total_cells':   raw.values,
            'wells':         wells_per_gene.reindex(raw.index).fillna(1).astype(int),
            'cells_per_well':raw.values / wells_per_gene.reindex(raw.index).fillna(1),
        }).sort_values('cells_per_well', ascending=False)

        wt_mask   = df_counts['Label'].str.startswith('WT')
        df_counts = pd.concat([df_counts[wt_mask], df_counts[~wt_mask]])

        fig, ax = plt.subplots(1, 1, figsize=self.cfg.FIGURE_SIZE_STANDARD)
        _draw_count_bars(ax, df_counts, 'Label', self.cfg,
                         'Gene Knockdown', 'CRISPRi Cell Counts per Well')
        _save_fig(fig, os.path.join(self.analysis_folder, 'cell_counts_per_well.png'), self.cfg)
        df_counts.to_excel(os.path.join(self.analysis_folder, 'cell_counts_per_well.xlsx'), index=False)
        print('✓')
        return df_counts


# ============================================================================
# GENE AGGREGATED VISUALIZER  (by gene)
# ============================================================================
class GeneAggregatedVisualizer:
    def __init__(self, analysis_folder: str, cfg: Config):
        self.analysis_folder = analysis_folder
        self.cfg             = cfg

    def plot_violin_by_gene(self, data: pd.DataFrame, feature: str):
        print(f'  Generating {feature} violin plot (by gene)...', end=' ')
        gene_means   = data.groupby('Gene')[feature].mean()
        unique_genes = gene_means.sort_values(ascending=True).index.tolist()
        color_dict   = {gene: get_gene_color(gene) for gene in unique_genes}

        fig, ax = plt.subplots(1, 1, figsize=self.cfg.FIGURE_SIZE_STANDARD)
        _draw_violin(ax, _clip_q(data, feature), 'Gene', feature,
                     unique_genes, color_dict, f'{feature} Distribution per Gene (Aggregated)')
        _save_fig(fig, os.path.join(self.analysis_folder, f'{feature}_violin_by_gene.png'), self.cfg)
        print('✓')

    def plot_cell_counts_by_gene(self, data: pd.DataFrame) -> pd.DataFrame:
        print('  Generating cell counts by gene...', end=' ')
        data_no_wt = data[data['Gene'] != 'WT']
        raw       = data_no_wt.groupby('Gene').size()
        wells     = data_no_wt.groupby('Gene')['Well'].nunique()
        df_counts = pd.DataFrame({
            'Gene':          raw.index,
            'total_cells':   raw.values,
            'wells':         wells.values,
            'cells_per_well':raw.values / wells.values,
        }).sort_values('cells_per_well', ascending=False)

        wt_mask   = df_counts['Gene'] == 'WT'
        df_counts = pd.concat([df_counts[wt_mask], df_counts[~wt_mask]])

        fig, ax = plt.subplots(1, 1, figsize=self.cfg.FIGURE_SIZE_STANDARD)
        _draw_count_bars(ax, df_counts, 'Gene', self.cfg,
                         'Gene', 'CRISPRi Cell Counts per Well (By Gene)')
        _save_fig(fig, os.path.join(self.analysis_folder, 'cell_counts_per_well_by_gene.png'), self.cfg)
        df_counts.to_excel(os.path.join(self.analysis_folder, 'cell_counts_per_well_by_gene.xlsx'), index=False)
        print('✓')
        return df_counts


# ============================================================================
# SUBGROUP VISUALIZER  (subgroups 1-3 vs WT)
# ============================================================================
class SubgroupVisualizer:
    def __init__(self, analysis_folder: str, cfg: Config):
        self.analysis_folder = analysis_folder
        self.cfg             = cfg
        self.subgroup_folder = os.path.join(analysis_folder, 'subgroup_comparisons')
        os.makedirs(self.subgroup_folder, exist_ok=True)

    def plot_subgroup_vs_wt(self, data: pd.DataFrame, gene: str, feature: str):
        # Use the subgroups actually present in the data (3 for most genes, 6 for NC/WT).
        gene_subgroups = sorted(
            data.loc[data['Gene'] == gene, 'Label'].dropna().unique(),
            key=label_sort_key,
        )
        if not gene_subgroups:
            return
        subset_data = data[data['Label'].isin(gene_subgroups) | (data['Gene'] == 'WT')].copy()
        if len(subset_data) == 0:
            return
        # Pool all WT NC_X variants into a single 'WT' group for the plot.
        subset_data['PlotLabel'] = np.where(subset_data['Gene'] == 'WT', 'WT', subset_data['Label'])
        label_order = [l for l in ['WT'] + list(gene_subgroups)
                       if l in subset_data['PlotLabel'].unique()]
        if len(label_order) < 2:
            return

        gene_color = get_gene_color(gene)
        color_dict = {lbl: gene_color for lbl in gene_subgroups}
        color_dict['WT'] = '#424242'

        fig, ax    = plt.subplots(1, 1, figsize=self.cfg.FIGURE_SIZE_SUBGROUP)
        data_clean = _clip_q(subset_data, feature)
        _draw_violin(ax, data_clean, 'PlotLabel', feature, label_order, color_dict,
                     f'{gene} Subgroups vs WT - {feature}', xlabel='Label')
        _save_fig(fig, os.path.join(self.subgroup_folder,
                                    f'{gene}_{feature}_subgroups_vs_wt.png'), self.cfg)

    def generate_all_subgroup_plots(self, data: pd.DataFrame):
        print('\nSubgroup Comparison Plots (Subgroups vs WT)')
        print('-' * 80)
        genes           = sorted([g for g in data['Gene'].unique() if g != 'WT'], key=label_sort_key)
        violin_features = [
            'roundness', 'area_um2', 'length_um', 'width_um',
            'perimeter_um', 'aspect_ratio', 'solidity', 'eccentricity',
        ]
        total_plots = 0
        for gene in genes:
            for feature in violin_features:
                if feature in data.columns:
                    self.plot_subgroup_vs_wt(data, gene, feature)
                    total_plots += 1
                    if total_plots % 10 == 0:
                        print(f'  Generated {total_plots} plots...', end='\r')
        print(f'  ✓ Generated {total_plots} subgroup comparison plots')
        print(f'  → Saved to: {self.subgroup_folder}')


# ============================================================================
# ABSOLUTE VALUE PLOTTER
# ============================================================================
class AbsoluteValuePlotter:
    """
    Bar charts showing mean ± SD of each feature in absolute units for every condition.
    Answers the question: "What is the actual size/shape of cells in each knockdown?"
    """

    def __init__(self, viz_folder: str, cfg: Config):
        self.viz_folder   = viz_folder
        self.cfg          = cfg
        self.output_folder = Path(viz_folder) / 'absolute_values'
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def plot_feature(self, data: pd.DataFrame, feature: str):
        """One bar chart per feature: all conditions with mean ± SD, WT reference line."""
        print(f'  Absolute value plot: {feature}...', end=' ')
        genes  = sorted(data['Gene'].unique(), key=label_sort_key)
        colors = [get_gene_color(g) for g in genes]

        fig, ax = plt.subplots(figsize=self.cfg.FIGURE_SIZE_STANDARD, dpi=self.cfg.DPI)
        unit = _draw_abs_bars(ax, genes, data, feature, self.cfg, colors, compact=False)
        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(f'{feature}{" (" + unit + ")" if unit else ""}', fontsize=12, fontweight='bold')
        ax.set_title(f'{feature} — Absolute Values per Condition\n(mean ± SD)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        _save_fig(fig, self.output_folder / f'{feature}_absolute_values.png', self.cfg)
        print('✓')

    def plot_overview(self, data: pd.DataFrame, features: List[str]):
        """Multi-panel overview: all features × all conditions in one figure."""
        print('  Absolute values overview (all features)...', end=' ')
        valid   = [f for f in features if f in data.columns]
        genes   = sorted(data['Gene'].unique(), key=label_sort_key)
        colors  = [get_gene_color(g) for g in genes]
        n_cols  = 4
        n_rows  = (len(valid) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 4.5, n_rows * 4), dpi=self.cfg.DPI)
        axes = np.array(axes).reshape(n_rows, n_cols)

        for idx, feature in enumerate(valid):
            ax   = axes[idx // n_cols, idx % n_cols]
            unit = _draw_abs_bars(ax, genes, data, feature, self.cfg, colors, compact=True)
            ax.set_xticks(range(len(genes)))
            ax.set_xticklabels(genes, rotation=90, ha='center', fontsize=6)
            ax.set_ylabel(unit or 'value', fontsize=8)
            ax.set_title(feature, fontsize=9, fontweight='bold')

        for idx in range(len(valid), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].axis('off')

        plt.suptitle('Absolute Feature Values — All Conditions (mean ± SD)\nGray band = WT ± 1 SD',
                     fontsize=13, fontweight='bold', y=1.01)
        _save_fig(fig, self.output_folder / 'all_features_absolute_overview.png', self.cfg)
        print('✓')


# ============================================================================
# EFFECT SIZE HEATMAP PLOTTER
# ============================================================================
class EffectSizeHeatmapPlotter:
    """
    Heatmap of Cohen's d for all genes × features — the phenotypic fingerprint.
    Accumulate per-feature effect_dfs during the histogram pipeline loop, then
    call generate() once after all features have been processed.
    """

    def __init__(self, output_base: Path, cfg: Config):
        self.output_base = output_base
        self.cfg         = cfg
        self._records: List[dict] = []

    def add_feature(self, feature: str, effect_df: pd.DataFrame):
        """Called once per feature with the gene-level effect_df from WTComparisonAnalyzer."""
        for _, row in effect_df.iterrows():
            rec = {'gene': row['gene'], 'feature': feature, 'cohens_d': row['cohens_d']}
            if 'q_value'             in row: rec['q_value']             = row['q_value']
            if 'concordance_flag'    in row: rec['concordance_flag']    = row['concordance_flag']
            if 'depletion_flag'      in row: rec['depletion_flag']      = row['depletion_flag']
            if 'mean_cells_per_well' in row: rec['mean_cells_per_well'] = row['mean_cells_per_well']
            self._records.append(rec)

    def generate(self):
        """Build and save the heatmap. Call after all features have been added."""
        if not self._records:
            return

        print('\n -> Generating Cohen\'s d phenotypic fingerprint heatmap...')
        df     = pd.DataFrame(self._records)
        matrix = df.pivot_table(index='gene', columns='feature',
                                values='cohens_d', aggfunc='mean')

        # Sort genes by total absolute effect size (most affected first)
        row_order = matrix.abs().sum(axis=1).sort_values(ascending=False).index
        matrix    = matrix.loc[row_order]

        # q-value matrix for significance annotations
        q_matrix = None
        if 'q_value' in df.columns:
            q_matrix = df.pivot_table(index='gene', columns='feature',
                                      values='q_value', aggfunc='min')
            q_matrix = q_matrix.reindex(index=row_order, columns=matrix.columns)

        # concordance_flag: True = ≥2/3 gRNAs agree in direction
        concordance_matrix = None
        if 'concordance_flag' in df.columns:
            concordance_matrix = df.pivot_table(index='gene', columns='feature',
                                                values='concordance_flag', aggfunc='min')
            concordance_matrix = concordance_matrix.reindex(
                index=row_order, columns=matrix.columns)

        # Transpose: features on y-axis (rows), genes on x-axis (columns)
        matrix_T = matrix.T  # rows=features, columns=genes
        if concordance_matrix is not None:
            concordance_T = concordance_matrix.T.reindex(
                index=matrix_T.index, columns=matrix_T.columns)
        else:
            concordance_T = None

        # Depletion: genes where mean_cells_per_well < threshold on any feature
        depleted_genes: set = set()
        if 'depletion_flag' in df.columns:
            depleted_genes = set(
                df.loc[df['depletion_flag'] == True, 'gene'].unique()
            )

        segmentation_issue_genes: set = set(self.cfg.KNOWN_SEGMENTATION_ISSUES.keys())

        # NC noise floor: use NC row from matrix if present
        nc_label = self.cfg.NC_LABEL
        nc_noise_floor: Optional[float] = None
        if nc_label in matrix.index:
            nc_noise_floor = float(matrix.loc[nc_label].abs().mean())

        vmax = max(0.5, float(np.nanpercentile(np.abs(matrix_T.values), 95)))

        fig_w = max(8, len(matrix_T.columns) * 1.1)
        fig_h = fig_w * 9 / 16
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=self.cfg.DPI)

        im = ax.imshow(matrix_T.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        cbar = plt.colorbar(im, ax=ax, label="Cohen's d vs WT", fraction=0.035, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

        # x-tick labels: bold + markers for depleted (‡) and segmentation issues (⚠)
        col_labels = []
        for gene in matrix_T.columns:
            label = gene
            if gene in segmentation_issue_genes:
                label += ' ⚠'
            if gene in depleted_genes:
                label += ' ‡'
            col_labels.append(label)

        ax.set_xticks(range(len(matrix_T.columns)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(matrix_T.index)))
        ax.set_yticklabels(matrix_T.index, fontsize=10)

        # Hatching overlay for depleted gene columns (survivor-selection bias)
        for j, gene in enumerate(matrix_T.columns):
            if gene in depleted_genes:
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, -0.5), 1, len(matrix_T.index),
                    fill=False, hatch='////', edgecolor='gray',
                    linewidth=0, alpha=0.4, zorder=3))

        # Cell annotations: d value + concordance marker only.
        # q-value stars are NOT shown: with n_wells=3, the MWU p-value floor is
        # 2/84≈0.024 for ANY monotonically ordered triplet, making stars uninformative.
        # Significance is implied by effect size magnitude; see per-feature CSV for p/q.
        for i in range(len(matrix_T.index)):
            for j in range(len(matrix_T.columns)):
                val = matrix_T.values[i, j]
                if np.isnan(val):
                    continue
                text_color = 'white' if abs(val) > vmax * 0.55 else 'black'
                low_concordance = (
                    concordance_T is not None
                    and not bool(concordance_T.iloc[i, j])
                )
                dagger = '†' if low_concordance else ''
                ax.text(j, i, f'{val:.2f}{dagger}',
                        ha='center', va='center', fontsize=7, color=text_color)

        # NC noise floor reference lines on colorbar (data coordinates = d-value)
        if nc_noise_floor is not None:
            for sign in (1, -1):
                cbar.ax.axhline(y=sign * nc_noise_floor,
                                color='gold', linewidth=1.5, linestyle='--')
            cbar.ax.text(1.5, nc_noise_floor,
                         f'NC |d|≈{nc_noise_floor:.2f}',
                         transform=cbar.ax.transData, fontsize=7,
                         color='darkgoldenrod', va='center')

        legend_parts = [
            "† <2/3 gRNAs concordant",
            "‡ depleted (<5k cells/well; survivor bias)",
            "⚠ known segmentation issue (unreliable features)",
            "Stars omitted: MWU p-floor = 0.024 for n_wells=3 — see per-feature CSV for p/q",
        ]
        if nc_noise_floor is not None:
            legend_parts.insert(0, f"Gold dashes = NC assay noise floor (|d|≈{nc_noise_floor:.2f})")

        ax.set_title(
            "Phenotypic Fingerprint: Cohen's d vs WT\n"
            "Genes sorted by total |d| (post-hoc; not a validated ranking).\n"
            + "  |  ".join(legend_parts),
            fontsize=9, fontweight='bold')
        ax.set_xlabel('Gene / Condition', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
        out_path = self.output_base / 'cohens_d_heatmap.png'
        _save_fig(fig, out_path, self.cfg)
        print(f' [OK] Saved: {out_path}')

        # Also export the numeric table
        matrix.to_csv(self.output_base / 'cohens_d_matrix.csv')


# ============================================================================
# PATHWAY ANALYZER
# ============================================================================
class PathwayAnalyzer:
    """
    Aggregates genes by mechanism of action (pathway) and generates:
      • Horizontal bar chart of Cohen's d per pathway vs WT
      • Violin plot of per-pathway feature distributions
    """

    def __init__(self, output_base: Path, cfg: Config):
        self.output_base = output_base
        self.cfg         = cfg

    def analyze_feature(self, df: pd.DataFrame, feature: str):
        """Generate pathway-level plots for one feature."""
        output_folder = self.output_base / feature / 'pathway_analysis'
        output_folder.mkdir(parents=True, exist_ok=True)

        unit      = self.cfg.FEATURE_UNITS.get(feature, '')
        wt_values = df.loc[df['Gene'] == 'WT', feature].values
        wt_mean   = float(np.mean(wt_values))
        wt_sd     = float(np.std(wt_values, ddof=1))

        # Only include pathways that have ≥10 cells present in the data
        present_genes = set(df['Gene'].unique())
        pathway_stats = []
        for pathway, genes in self.cfg.PATHWAY_GROUPS.items():
            active = [g for g in genes if g in present_genes]
            if not active:
                continue
            vals = df.loc[df['Gene'].isin(active), feature].values
            if len(vals) < 10:
                continue
            try:
                _, p = mannwhitneyu(vals, wt_values, alternative='two-sided')
            except Exception:
                p = np.nan
            pathway_stats.append({
                'pathway':  pathway,
                'genes':    ', '.join(active),
                'mean':     float(np.mean(vals)),
                'sd':       float(np.std(vals, ddof=1)),
                'n':        len(vals),
                'cohens_d': EffectSizeCalculator.cohens_d(vals, wt_values),
                'p_value':  p,
            })

        if not pathway_stats:
            return

        pathway_df = pd.DataFrame(pathway_stats).sort_values('cohens_d', ascending=True)

        # BH-FDR correction across all pathway tests within this feature.
        valid_p = pathway_df['p_value'].notna()
        if valid_p.sum() > 1:
            q_vals = EffectSizeCalculator.bh_correction(
                pathway_df.loc[valid_p, 'p_value'].values)
            pathway_df.loc[valid_p, 'q_value'] = q_vals
        else:
            pathway_df['q_value'] = pathway_df['p_value']

        pathway_df.to_csv(output_folder / f'{feature}_pathway_effect_sizes.csv', index=False)

        # ── Horizontal bar chart: Cohen's d per pathway ───────────────────────
        fig_h = max(5, len(pathway_df) * 0.55 + 2)
        fig, ax = plt.subplots(figsize=(10, fig_h), dpi=self.cfg.DPI)
        bar_colors = ['#C62828' if d > 0 else '#1565C0' for d in pathway_df['cohens_d']]
        bars = ax.barh(pathway_df['pathway'], pathway_df['cohens_d'],
                       color=bar_colors, alpha=0.82, edgecolor='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=1.5)
        for bar, row in zip(bars, pathway_df.itertuples()):
            xpos = bar.get_width() + 0.03 * (1 if bar.get_width() >= 0 else -1)
            q_col = row.q_value if hasattr(row, 'q_value') else row.p_value
            stars = EffectSizeCalculator.sig_stars(q_col)
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f'd={row.cohens_d:.2f}  n={row.n:,} {stars}',
                    va='center', fontsize=8,
                    ha='left' if bar.get_width() >= 0 else 'right')
        ax.set_xlabel("Cohen's d vs WT", fontsize=11, fontweight='bold')
        ax.set_title(f'{feature} — Pathway-Level Effect Sizes vs WT\n'
                     f'(WT: {wt_mean:.3f} ± {wt_sd:.3f} {unit}'.strip() + ')',
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        _save_fig(fig, output_folder / f'{feature}_pathway_cohens_d.png', self.cfg)

        # ── Violin plot: per-pathway distributions ────────────────────────────
        rows = []
        for _, prow in pathway_df.iterrows():
            active = [g for g in self.cfg.PATHWAY_GROUPS[prow['pathway']] if g in present_genes]
            for v in df.loc[df['Gene'].isin(active), feature].values:
                rows.append({'pathway': prow['pathway'], feature: float(v)})
        for v in wt_values:
            rows.append({'pathway': 'WT', feature: float(v)})

        long_df      = pd.DataFrame(rows).dropna()
        pathway_order = ['WT'] + pathway_df.sort_values('cohens_d', ascending=False)['pathway'].tolist()
        pathway_order = [p for p in pathway_order if p in long_df['pathway'].unique()]

        color_dict = {'WT': '#424242'}
        for pathway, genes in self.cfg.PATHWAY_GROUPS.items():
            color_dict[pathway] = get_gene_color(genes[0]) if genes else 'gray'

        clean = _clip_q(long_df, feature)

        fig_w = max(12, len(pathway_order) * 1.3)
        fig, ax = plt.subplots(figsize=(fig_w, 7), dpi=self.cfg.DPI)
        ax.set_facecolor('white')
        sns.violinplot(x='pathway', y=feature, data=clean, order=pathway_order,
                       palette=color_dict, ax=ax, cut=2, bw_adjust=0.7, inner='quart',
                       scale='width', linewidth=1.2, saturation=0.85)
        ax.axhline(wt_mean, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='WT mean')
        ax.set_xlabel('Pathway / MOA', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{feature}{" (" + unit + ")" if unit else ""}', fontsize=11, fontweight='bold')
        ax.set_title(f'{feature} — Pathway-Level Distribution', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=35)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        _save_fig(fig, output_folder / f'{feature}_pathway_violin.png', self.cfg)


# ============================================================================
# PIPELINE RUNNERS
# ============================================================================
def run_histogram_pipeline(df: pd.DataFrame, fov_data: pd.DataFrame,
                            output_base: Path, cfg: Config):
    """Statistical analyses: FOV / well / WT-comparison / variability / spatial +
    cross-feature Cohen's d heatmap + pathway-level analysis."""
    _banner('PART 1: HISTOGRAM & STATISTICAL ANALYSES')

    # Cache is local to this pipeline — not shared with or accessible by Part 2.
    cache = StatisticsCache()

    fov_analyzer         = FOVLevelAnalyzer(df, output_base, cfg, cache)
    well_analyzer        = WellLevelAnalyzer(df, output_base, cfg, cache)
    wt_analyzer          = WTComparisonAnalyzer(df, output_base, cfg, cache)
    variability_analyzer = VariabilityAnalyzer(df, fov_data, output_base, cfg)
    spatial_analyzer     = SpatialAnalyzer(df, fov_data, output_base, cfg)
    heatmap_plotter      = EffectSizeHeatmapPlotter(output_base, cfg)
    pathway_analyzer     = PathwayAnalyzer(output_base, cfg)

    for i, feature in enumerate(cfg.MORPHOLOGY_FEATURES, 1):
        print(f'\n[{i}/{len(cfg.MORPHOLOGY_FEATURES)}] Processing: {feature}')
        print('-' * 80)
        fov_analyzer.analyze_feature(feature)
        well_analyzer.analyze_feature(feature)
        effect_df = wt_analyzer.analyze_feature(feature)
        heatmap_plotter.add_feature(feature, effect_df)

        # Positive-control QC: ftsZ knockdown must produce filamentation.
        # If length_um Cohen's d < 0.5, the assay likely failed.
        if feature == 'length_um' and 'ftsZ' in df['gene'].unique():
            ftsz_rows = effect_df[effect_df['gene'] == 'ftsZ']
            ftsz_d    = float(ftsz_rows['cohens_d'].iloc[0]) if len(ftsz_rows) else float('nan')
            if np.isnan(ftsz_d) or abs(ftsz_d) < 0.5:
                warnings.warn(
                    f'\n{"!" * 80}\n'
                    f'ASSAY QC FAILED: ftsZ Cohen\'s d for length_um = '
                    f'{ftsz_d:.3f} (expected |d| > 0.5).\n'
                    f'If ftsZ knockdown is not producing filamentation, the assay '
                    f'may have failed or the segmentation is not capturing filaments.\n'
                    f'{"!" * 80}',
                    stacklevel=2,
                )

        variability_analyzer.analyze_feature(feature)
        spatial_analyzer.analyze_feature(feature)
        pathway_analyzer.analyze_feature(df, feature)

        cache.clear()   # release cached arrays before the next feature
        gc.collect()

    # Cross-feature summaries — generated once after all features are processed
    _banner('CROSS-FEATURE SUMMARIES')
    heatmap_plotter.generate()


def run_visualization_pipeline(df: pd.DataFrame, output_base: Path, cfg: Config):
    """Visualization analyses: violin plots + cell counts + absolute value charts."""
    _banner('PART 2: VISUALIZATION ANALYSES')
    print(f"✓ Genes detected: {sorted(df['Gene'].unique(), key=label_sort_key)}")
    print(f"✓ Labels: {df['Label'].nunique()} unique  |  Total cells: {len(df):,}\n")

    viz_folder = str(output_base / 'visualization')
    os.makedirs(viz_folder, exist_ok=True)

    visualizer      = Visualizer(viz_folder, cfg)
    gene_visualizer = GeneAggregatedVisualizer(viz_folder, cfg)
    subgroup_viz    = SubgroupVisualizer(viz_folder, cfg)
    abs_plotter     = AbsoluteValuePlotter(viz_folder, cfg)

    wells_per_gene  = df.groupby('Label')['Well'].nunique()
    violin_features = [
        'roundness', 'area_um2', 'length_um', 'width_um',
        'perimeter_um', 'aspect_ratio', 'solidity', 'eccentricity',
    ]

    print('Cell Count Analysis (By Label)')
    print('-' * 80)
    cc_df = visualizer.plot_cell_counts_normalized(df, wells_per_gene)
    print(f"  → {cc_df['total_cells'].sum():,} total cells  |  {len(cc_df)} labels\n")

    print('Cell Count Analysis (By Gene — Aggregated)')
    print('-' * 80)
    cc_gene_df = gene_visualizer.plot_cell_counts_by_gene(df)
    print(f"  → {cc_gene_df['total_cells'].sum():,} total cells  |  {len(cc_gene_df)} genes\n")

    print('Violin Plots (By Gene — Aggregated)')
    print('-' * 80)
    for feature in violin_features:
        if feature in df.columns:
            gene_visualizer.plot_violin_by_gene(df, feature)
        else:
            print(f'  ⚠ Skipping {feature} (not in data)')

    print('\nAbsolute Value Plots (mean ± SD per condition)')
    print('-' * 80)
    for feature in violin_features:
        if feature in df.columns:
            abs_plotter.plot_feature(df, feature)
    abs_plotter.plot_overview(df, violin_features)

    subgroup_viz.generate_all_subgroup_plots(df)


# ============================================================================
# CLI + MAIN
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='CRISPRi morphology analysis pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-folder', default=Config.DATA_FOLDER,
                        help='Path to the folder containing the parquet and plate-map files')
    parser.add_argument('--dpi', type=int, default=Config.DPI,
                        help='DPI for all saved figures')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg             = Config()
    cfg.DATA_FOLDER = args.data_folder
    cfg.DPI         = args.dpi

    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(cfg.DATA_FOLDER) / f'Analysis_{timestamp}'
    output_base.mkdir(exist_ok=True)

    sep = '=' * 80
    print(f'{sep}\nMORPHOLOGY ANALYSIS PIPELINE\n{sep}\n'
          f'Output: {output_base}\nDPI:    {cfg.DPI}\n{sep}')

    df = load_and_prepare_data(cfg)

    # Aggregate to FOV level exactly once; pass the result to both analyzers
    # that need it — VariabilityAnalyzer and SpatialAnalyzer — so neither
    # re-groups the full cell DataFrame on every feature iteration.
    all_features = list(set(cfg.MORPHOLOGY_FEATURES + cfg.FEATURES))
    fov_data     = aggregate_fov(df, all_features)

    run_histogram_pipeline(df, fov_data, output_base, cfg)
    run_visualization_pipeline(df, output_base, cfg)

    sep = '=' * 80
    print('\n'.join([
        sep, 'ANALYSIS COMPLETE', sep, f'Results → {output_base}',
        '  ├── <feature>/wt_comparisons/       histogram comparisons vs WT + p-values',
        '  ├── <feature>/well_level/            per-well density plots & stats',
        '  ├── <feature>/fov_level/             per-FOV effect sizes + strip plot',
        '  ├── <feature>/variability_analysis/  CV & variance decomposition',
        '  ├── <feature>/spatial_analysis/      spatial heatmaps',
        '  ├── <feature>/pathway_analysis/      pathway-level bar chart & violin',
        '  ├── cohens_d_heatmap.png             phenotypic fingerprint (genes × features)',
        '  ├── cohens_d_matrix.csv              numeric effect size table',
        '  └── visualization/',
        '        ├── absolute_values/           mean ± SD per condition per feature',
        '        ├── *_violin.png               violin by label',
        '        ├── *_violin_by_gene.png       violin aggregated by gene',
        '        └── subgroup_comparisons/      subgroup vs WT violins',
    ]))


if __name__ == '__main__':
    main()
