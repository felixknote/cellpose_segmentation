# ============================================================================
# OD600 × MORPHOLOGY CORRELATION ANALYSIS
#
# Correlates per-well OD600 measurements (optical density at imaging time)
# with per-well mean morphological features from CellposeSAM segmentation.
#
# Plates: P1_1, P1_2, P2_1, P2_2, P3_1, P3_2  (nested WT replicates)
#
# Outputs (under Analysis_{timestamp}/od600_correlation/):
#   per_plate/                  — scatter panels per plate, well labels shown
#   summary/                    — combined scatter per feature (all plates)
#   summary/by_biorep/          — per bio-replicate scatter (P1 / P2 / P3)
#   correlation_heatmap.png     — raw Pearson r  +  within-plate z-normalised Overall
#   partial_correlation_heatmap.png — partial r controlling for edge distance
#   od600_heatmaps/             — 96-well OD600 plate layouts
# ============================================================================

import gc
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    ROOT_DATA_DIR        = r"D:\2026_03_14_WT_Comparison"
    SEGMENTATION_SUBPATH = r"TIFOCUS\CellposeSAM Segmentation results"
    AGGREGATED_FILE      = "micromorph_cell_measurements.parquet"
    OD600_SUBDIR         = "OD600"

    MORPHOLOGY_FEATURES = ['roundness', 'area_um2', 'length_um', 'width_um', 'perimeter_um']

    FEATURE_UNITS = {'roundness': '', 'area_um2': 'µm²', 'length_um': 'µm',
                     'width_um': 'µm', 'perimeter_um': 'µm'}
    DPI = 300

    MIN_CELLS_PER_WELL = 1000


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


def parse_well_metadata(well_name: str) -> Tuple[str, str]:
    """'P1_2' → ('P1', '2').  Falls back to (well_name, 'unknown')."""
    if pd.isna(well_name):
        return ('unknown', 'unknown')
    s = str(well_name).strip()
    if '_' in s:
        idx = s.rfind('_')
        return (s[:idx], s[idx + 1:])
    return (s, 'unknown')


def parse_well_position(well_name: str) -> Optional[Tuple[int, int]]:
    """'A01' → (0, 0),  'H12' → (7, 11).  Returns None on parse failure."""
    s = str(well_name).strip()
    if len(s) >= 2 and s[0].isalpha() and s[1:].isdigit():
        r = ord(s[0].upper()) - ord('A')
        c = int(s[1:]) - 1
        if 0 <= r < 8 and 0 <= c < 12:
            return (r, c)
    return None


def is_edge_well(well_name: str) -> bool:
    pos = parse_well_position(well_name)
    if pos is None:
        return False
    r, c = pos
    return r == 0 or r == 7 or c == 0 or c == 11


# ============================================================================
# COLOUR SYSTEM
# ============================================================================
_BIO_COL: dict = {
    'P1': '#4878D0',
    'P2': '#EF6548',
    'P3': '#6ACC65',
}

_PLATE_COL: dict = {
    'P1_1': '#84A9E6',  'P1_2': '#1C4B9E',
    'P2_1': '#F5A08C',  'P2_2': '#B53D1E',
    'P3_1': '#98DE96',  'P3_2': '#2D8B3A',
}

_FEAT_COL: dict = {
    'roundness':    '#9467BD',
    'area_um2':     '#1F77B4',
    'length_um':    '#FF7F0E',
    'width_um':     '#2CA02C',
    'perimeter_um': '#D62728',
}

# Edge-distance colours — same as plot_od600.py
_EDGE_COL: dict = {0: '#c0392b', 1: '#e07070', 2: '#a8c8e8', 3: '#2980b9'}


# ============================================================================
# DATA LOADING — MORPHOLOGY
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
        raise FileNotFoundError(f"No parquet files found under {root}")
    return found


def _load_single_plate(plate_name: str, parquet_path: Path,
                        all_features: List[str]) -> pd.DataFrame:
    requested = ['Well', 'filename'] + all_features
    try:
        df = pd.read_parquet(parquet_path, columns=requested)
    except Exception:
        df = pd.read_parquet(parquet_path)
        df = df[[c for c in requested if c in df.columns]]

    df['plate'] = plate_name
    df['fov']   = df['filename'].apply(extract_fov_from_filename) if 'filename' in df.columns else 'unknown'
    if 'filename' in df.columns:
        df.drop(columns='filename', inplace=True)

    bio_rep, tech_rep = parse_well_metadata(plate_name)
    df['bio_replicate']  = bio_rep
    df['tech_replicate'] = tech_rep
    df['well']           = df['Well'].astype(str)
    df.drop(columns='Well', inplace=True)
    return df


def load_all_plates(cfg: Config) -> pd.DataFrame:
    print("=" * 80)
    print("LOADING MULTI-PLATE MORPHOLOGY DATA")
    print("=" * 80)

    all_features = cfg.MORPHOLOGY_FEATURES
    plate_files  = find_plate_files(cfg)
    print(f"  Found {len(plate_files)} plates:")
    for name, path in plate_files:
        print(f"    {name}: {path}")

    frames = []
    for plate_name, parquet_path in plate_files:
        df_plate = _load_single_plate(plate_name, parquet_path, all_features)
        frames.append(df_plate)
        print(f"  Loaded {plate_name}: {len(df_plate):,} cells")

    df = pd.concat(frames, ignore_index=True)

    feat_cols = [f for f in all_features if f in df.columns]
    df[feat_cols] = df[feat_cols].astype(np.float32)

    for feat, (lo, hi) in [('roundness', (0, 1)), ('solidity', (0, 1)),
                            ('eccentricity', (0, 1)), ('aspect_ratio', (1, 20))]:
        if feat in df.columns:
            df[feat] = df[feat].clip(lo, hi)

    for feat in all_features:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
            df = df[np.isfinite(df[feat])]

    initial_count = len(df)
    for feat in cfg.MORPHOLOGY_FEATURES:
        if feat in df.columns:
            q1, q99 = df[feat].quantile(0.01), df[feat].quantile(0.99)
            df = df[(df[feat] >= q1) & (df[feat] <= q99)]

    df = df[df['well'].notna() & (df['well'] != 'nan')]
    df.reset_index(drop=True, inplace=True)
    gc.collect()

    print(f"\n  After filtering: {len(df):,} cells  ({initial_count - len(df):,} removed)")
    print(f"  Plates:          {df['plate'].nunique()}")
    print(f"  Bio replicates:  {sorted(df['bio_replicate'].unique())}")
    print(f"  Unique wells:    {df['well'].nunique()}")
    print(f"  Memory:          {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df


# ============================================================================
# DATA LOADING — OD600
# Tidy-DataFrame construction mirrors plot_od600.py (zone_grid, row/col mapping).
# ============================================================================
def load_od600(cfg: Config) -> Tuple[pd.DataFrame, dict]:
    """Load OD600 CSVs.  Returns (tidy_df_with_well_column, raw_plates_dict)."""
    print("\n" + "=" * 80)
    print("LOADING OD600 DATA")
    print("=" * 80)

    od600_dir   = Path(cfg.ROOT_DATA_DIR) / cfg.OD600_SUBDIR
    plate_files = sorted(od600_dir.glob("P*.csv"))

    if not plate_files:
        raise FileNotFoundError(f"No OD600 CSVs found in {od600_dir}")

    plates: dict = {}
    for f in plate_files:
        df_csv = pd.read_csv(f, sep=";", header=None, decimal=",")
        plates[f.stem] = df_csv.values.astype(float)   # shape (8, 12)
        print(f"  Loaded {f.name}:  OD range "
              f"[{plates[f.stem].min():.3f}, {plates[f.stem].max():.3f}]")

    # Edge-distance grid — identical to plot_od600.py
    #   0 = outer edge,  1–2 = intermediate,  3 = centre
    rows_g, cols_g = np.meshgrid(np.arange(8), np.arange(12), indexing="ij")
    zone_grid = np.minimum(np.minimum(rows_g, 7 - rows_g),
                           np.minimum(cols_g, 11 - cols_g))

    records = []
    for name, data in plates.items():
        bio, tech = name.split("_")
        for r in range(8):
            for c in range(12):
                records.append({
                    "plate":     name,
                    "bio_rep":   bio,
                    "tech_rep":  tech,
                    "row":       r,
                    "col":       c,
                    "edge_dist": int(zone_grid[r, c]),
                    "OD600":     data[r, c],
                    # Well name in A01 format — same key as morphology data
                    "well":      f"{chr(65 + r)}{c + 1:02d}",
                })

    df_od = pd.DataFrame(records)
    print(f"\n  Total wells: {len(df_od)}  ({df_od['plate'].nunique()} plates)")
    return df_od, plates


# ============================================================================
# AGGREGATION & MERGE
# ============================================================================
def aggregate_to_well(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Average cell-level measurements to per-well means (same key used for OD join)."""
    feat_cols = [f for f in cfg.MORPHOLOGY_FEATURES if f in df.columns]
    df_well = (
        df.groupby(['plate', 'well', 'bio_replicate', 'tech_replicate'],
                   observed=True, sort=True)
          .agg(n_cells=('well', 'count'),
               **{f: (f, 'mean') for f in feat_cols})
          .reset_index()
    )
    print(f"\n  Well-level aggregation: {len(df_well)} wells  "
          f"(median {df_well.n_cells.median():.0f} cells/well)")
    return df_well


def merge_od_morphology(df_well: pd.DataFrame, df_od: pd.DataFrame,
                         cfg: Config) -> pd.DataFrame:
    """Inner-join per-well morphology with OD600 on (plate, well)."""
    df_merged = df_well.merge(
        df_od[['plate', 'well', 'OD600', 'edge_dist', 'bio_rep', 'tech_rep']],
        on=['plate', 'well'],
        how='inner',
    )
    missing = len(df_well) - len(df_merged)
    if missing:
        print(f"  WARNING: {missing} morphology wells had no OD600 match")

    before = len(df_merged)
    df_merged = df_merged[df_merged.n_cells >= cfg.MIN_CELLS_PER_WELL]
    excluded = before - len(df_merged)
    if excluded:
        print(f"  Excluded {excluded} wells with < {cfg.MIN_CELLS_PER_WELL} cells")

    print(f"  Merged dataset: {len(df_merged)} wells across "
          f"{df_merged['plate'].nunique()} plates")
    return df_merged


# ============================================================================
# SHARED HELPERS
# ============================================================================
def _sig_label(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def _pearsonr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Pearson r and two-tailed p-value via NumPy + t-distribution.
    Avoids scipy stub type errors that arise from unpacking LinregressResult."""
    if len(x) < 3:
        return np.nan, np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    denom = max(1.0 - r ** 2, 1e-15)
    t_stat = r * np.sqrt((len(x) - 2) / denom)
    p = float(2 * stats.t.sf(abs(t_stat), df=len(x) - 2))
    return r, p




# ============================================================================
# PLOT A — PER-PLATE SCATTER PANELS
# per_plate/{plate}/{feature}.png  — one PNG per feature per plate
# ============================================================================
def plot_per_plate_correlations(df: pd.DataFrame, output_base: Path,
                                 cfg: Config) -> None:
    """
    For each plate a subfolder is created; each feature gets its own PNG.
    X = OD600, Y = mean feature.  Points coloured by edge_dist, labelled by well.
    """
    print("\n── Per-plate correlation panels ─────────────────────────────────")

    from matplotlib.patches import Patch

    feat_cols = [f for f in cfg.MORPHOLOGY_FEATURES if f in df.columns]

    for plate, grp in df.groupby('plate', sort=True):
        plate_dir = output_base / 'per_plate' / str(plate)
        plate_dir.mkdir(parents=True, exist_ok=True)

        bio_rep   = str(grp['bio_rep'].iloc[0])
        tech_rep  = str(grp['tech_rep'].iloc[0])
        bio_label = bio_rep[1:] if bio_rep.startswith('P') else bio_rep

        x_raw = grp['OD600'].values
        e_raw = grp['edge_dist'].values
        w_raw = grp['well'].values

        for feat in feat_cols:
            y_raw = grp[feat].values
            mask  = np.isfinite(x_raw) & np.isfinite(y_raw)
            x, y  = x_raw[mask], y_raw[mask]
            e_vals   = e_raw[mask]
            w_labels = w_raw[mask]

            fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=cfg.DPI)
            ax.set_title(
                f'{feat}  —  Plate {plate}\n'
                f'Bio rep {bio_label}  ·  Tech rep {tech_rep}',
                fontsize=10, fontweight='bold',
            )

            for dist_val in sorted(_EDGE_COL):
                sel = e_vals == dist_val
                if sel.any():
                    ax.scatter(x[sel], y[sel],
                               color=_EDGE_COL[dist_val], s=30, alpha=0.85,
                               linewidths=0, zorder=3)

            for xi, yi, wl in zip(x, y, w_labels):
                ax.annotate(wl, (xi, yi),
                            xytext=(0, 3), textcoords='offset points',
                            fontsize=4, ha='center', va='bottom',
                            color='#333333', zorder=4)

            if len(x) >= 3:
                r_val, p_val = _pearsonr_safe(x, y)
                coeffs  = np.polyfit(x, y, 1)
                x_line  = np.linspace(x.min(), x.max(), 200)
                ax.plot(x_line, np.polyval(coeffs, x_line),
                        color='black', linewidth=1.5, zorder=5)
                # Stats box — anchored outside the axes, top-right of the figure
                ax.text(1.02, 1.0,
                        f'r = {r_val:.3f}  {_sig_label(p_val)}',
                        transform=ax.transAxes, fontsize=8,
                        va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.35', fc='white',
                                  ec='#CCCCCC', alpha=0.95))

            unit = cfg.FEATURE_UNITS.get(feat, '')
            ax.set_xlabel('OD600', fontsize=10)
            ax.set_ylabel(f'{feat} [{unit}]' if unit else feat, fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=9)

            # Legend outside the axes, below the stats box
            handles = [Patch(facecolor=_EDGE_COL[d], alpha=0.85,
                             label=f'dist {d}' + (' (edge)' if d == 0
                                                   else ' (centre)' if d == 3 else ''))
                       for d in sorted(_EDGE_COL)]
            ax.legend(handles=handles, fontsize=7,
                      bbox_to_anchor=(1.02, 0.72), loc='upper left',
                      borderaxespad=0, framealpha=0.95,
                      title='Edge dist', title_fontsize=7)

            fig.tight_layout()
            fig.savefig(str(plate_dir / f'{feat}.png'), dpi=cfg.DPI, bbox_inches='tight')
            plt.close(fig)

        print(f'  per_plate/{plate}/  ({len(feat_cols)} plots)')


# ============================================================================
# PLOT B — COMBINED SCATTER  (all plates as one group, one regression line)
# ============================================================================
def plot_summary_per_feature(df: pd.DataFrame, output_base: Path,
                              cfg: Config) -> None:
    """
    One figure per feature: all wells pooled as one group.
    Single colour, single regression line, single Pearson r.
    """
    print("\n── Combined scatter per feature ─────────────────────────────────")
    out = output_base / 'summary'
    out.mkdir(parents=True, exist_ok=True)

    feat_cols = [f for f in cfg.MORPHOLOGY_FEATURES if f in df.columns]

    for feat in feat_cols:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=cfg.DPI)

        x_all = df['OD600'].values
        y_all = df[feat].values
        mask  = np.isfinite(x_all) & np.isfinite(y_all)
        x_all, y_all = x_all[mask], y_all[mask]

        col = _FEAT_COL.get(feat, '#555555')
        ax.scatter(x_all, y_all, color=col, s=22, alpha=0.55,
                   linewidths=0, zorder=3)

        if len(x_all) >= 3:
            r_val, p_val = _pearsonr_safe(x_all, y_all)
            coeffs  = np.polyfit(x_all, y_all, 1)
            x_line  = np.linspace(x_all.min(), x_all.max(), 200)
            ax.plot(x_line, np.polyval(coeffs, x_line),
                    color='black', linewidth=2, zorder=5)
            ax.text(0.97, 0.05,
                    f'r = {r_val:.3f}  {_sig_label(p_val)}\nn = {len(x_all)} wells',
                    transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8))

        unit = cfg.FEATURE_UNITS.get(feat, '')
        ax.set_xlabel('OD600', fontsize=10)
        ax.set_ylabel(f'{feat} [{unit}]' if unit else feat, fontsize=10)
        ax.set_title(f'OD600 vs {feat}  (all plates combined)',
                     fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)

        fig.tight_layout()
        save_path = out / f'od600_vs_{feat}.png'
        fig.savefig(save_path, dpi=cfg.DPI, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {save_path.name}')


# ============================================================================
# PLOT B2 — NORMALIZED-OD SUMMARY  (OD600 z-scored per plate, all wells pooled)
# ============================================================================
def plot_summary_normalized(df: pd.DataFrame, output_base: Path,
                             cfg: Config) -> None:
    """
    Same layout as plot_summary_per_feature but with OD600 z-scored within
    each plate before pooling.  Removes between-plate OD scale differences so
    only within-plate OD variation (relative growth density) is correlated.
    """
    print("\n── Normalized-OD summary per feature ───────────────────────────")
    out = output_base / 'summary'
    out.mkdir(parents=True, exist_ok=True)

    feat_cols = [f for f in cfg.MORPHOLOGY_FEATURES if f in df.columns]

    # Z-score OD600 per plate
    df_z = df.copy()
    for plate, idx in df.groupby('plate').groups.items():
        vals = df_z.loc[idx, 'OD600']
        sd   = float(vals.std())
        df_z.loc[idx, 'OD600_z'] = (vals - float(vals.mean())) / sd if sd > 0 else 0.0

    for feat in feat_cols:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=cfg.DPI)

        x_all = df_z['OD600_z'].values
        y_all = df_z[feat].values
        mask  = np.isfinite(x_all) & np.isfinite(y_all)
        x_all, y_all = x_all[mask], y_all[mask]

        col = _FEAT_COL.get(feat, '#555555')
        ax.scatter(x_all, y_all, color=col, s=22, alpha=0.55,
                   linewidths=0, zorder=3)

        if len(x_all) >= 3:
            r_val, p_val = _pearsonr_safe(x_all, y_all)
            coeffs  = np.polyfit(x_all, y_all, 1)
            x_line  = np.linspace(x_all.min(), x_all.max(), 200)
            ax.plot(x_line, np.polyval(coeffs, x_line),
                    color='black', linewidth=2, zorder=5)
            ax.text(0.97, 0.05,
                    f'r = {r_val:.3f}  {_sig_label(p_val)}\nn = {len(x_all)} wells',
                    transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8))

        unit = cfg.FEATURE_UNITS.get(feat, '')
        ax.set_xlabel('OD600  (z-scored per plate)', fontsize=10)
        ax.set_ylabel(f'{feat} [{unit}]' if unit else feat, fontsize=10)
        ax.set_title(f'OD600 (normalised) vs {feat}',
                     fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)

        fig.tight_layout()
        save_path = out / f'od600_norm_vs_{feat}.png'
        fig.savefig(save_path, dpi=cfg.DPI, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {save_path.name}')


# ============================================================================
# MAIN
# ============================================================================
def main():
    cfg         = Config()
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(cfg.ROOT_DATA_DIR) / f'Analysis_{timestamp}' / 'od600_correlation'
    output_base.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('OD600 × MORPHOLOGY CORRELATION ANALYSIS')
    print('=' * 80)
    print(f'Data:   {cfg.ROOT_DATA_DIR}')
    print(f'Output: {output_base}')
    print('=' * 80)

    df             = load_all_plates(cfg)
    df_od, _       = load_od600(cfg)
    df_well        = aggregate_to_well(df, cfg)
    df_merged      = merge_od_morphology(df_well, df_od, cfg)

    plot_per_plate_correlations(df_merged, output_base, cfg)
    plot_summary_per_feature(df_merged, output_base, cfg)
    plot_summary_normalized(df_merged, output_base, cfg)

    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)
    print(f'Results → {output_base}')
    print('  ├── per_plate/{plate}/            one PNG per feature per plate')
    print('  ├── summary/od600_vs_*.png        combined scatter, raw OD600')
    print('  └── summary/od600_norm_vs_*.png   combined scatter, OD z-scored per plate')


if __name__ == '__main__':
    main()
