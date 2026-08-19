# ============================================================================
# CRISPRi CONTROL PLATE ANALYSIS PIPELINE  (v2 — factorial redesign)
#
# 2×2×2 factorial: cell_line × ATC × plasmid_status  (N=6 biological replicate plates)
# Primary question: how much do the 8 control conditions differ morphologically?
# Statistics operate on per-plate means (N=6); individual cells are not independent.
# ============================================================================

import argparse
import gc
import importlib.util
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.cm
import matplotlib.colors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ── Load single-plate module (digit prefix prevents standard import) ──────────
_sp_path = Path(__file__).parent / "02_morphological_analysis.py"
_spec    = importlib.util.spec_from_file_location("_sp", _sp_path)
_sp      = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sp)

extract_fov_from_filename = _sp.extract_fov_from_filename


# ============================================================================
# CONFIGURATION
# ============================================================================
PLATE_FOLDERS: List[str] = [
    r"D:\2026_06_17_CRISPRi Control Plate\P1",
    r"D:\2026_06_17_CRISPRi Control Plate\P2",
    r"D:\2026_06_17_CRISPRi Control Plate\P3",
    r"D:\2026_06_17_CRISPRi Control Plate\P4",
    r"D:\2026_06_17_CRISPRi Control Plate\P5",
    r"D:\2026_06_17_CRISPRi Control Plate\P6",
]
VALID_CELL_LINES = {'MG1655', 'ACE-1'}
VALID_PLASMIDS   = {f'NC_{i}' for i in range(1, 7)}
VALID_ATC        = {'+ATC', '-ATC'}


class Config(_sp.Config):
    ROOT_DATA_DIR        = r"D:\2026_06_17_CRISPRi Control Plate"
    SEGMENTATION_SUBPATH = r"CellposeSAM Segmentation results"
    AGGREGATED_FILE      = "cell_measurements.parquet"

    FEATURE_UNITS = {
        'roundness': '', 'area_um2': 'µm²', 'length_um': 'µm',
        'width_um': 'µm', 'perimeter_um': 'µm',
        'aspect_ratio': '', 'solidity': '', 'eccentricity': '',
    }
    WELL_ZSCORE_THRESHOLD  = 3.0
    PLATE_ZSCORE_THRESHOLD = 2.0
    MIN_CELLS_PER_WELL     = 500
    N_FOV                  = 15



# ============================================================================
# LABEL PARSING
# ============================================================================
def parse_crisprI_label(raw: str) -> Dict[str, Optional[str]]:
    """Parse semicolon-delimited plate-map label into structured metadata.

    Example: 'MG1655; NC_2; + ATC' → gene='MG1655_+ATC_+plasmid'
    """
    parts = [p.strip() for p in str(raw).split(';')]
    cell_line = parts[0].strip() if parts else 'Unknown'
    plasmid: Optional[str] = None
    atc_condition: Optional[str] = None

    for part in parts[1:]:
        norm = re.sub(r'\s+', '', part)
        if re.match(r'^NC_\d+$', norm, re.IGNORECASE):
            plasmid = norm.upper()
        elif norm in ('+ATC', '-ATC'):
            atc_condition = norm

    plasmid_status = '+plasmid' if plasmid else '-plasmid'
    return {
        'cell_line':      cell_line,
        'plasmid':        plasmid or 'No plasmid',
        'plasmid_status': plasmid_status,
        'atc_condition':  atc_condition,
        'gene':           f'{cell_line}_{atc_condition}_{plasmid_status}',
    }


# ============================================================================
# PRE-FLIGHT VALIDATION
# ============================================================================
def run_preflight_validation(plate_folders: List[str], cfg: Config) -> None:
    sep = '=' * 80
    OK, FAIL, WARN = '[OK]  ', '[FAIL]', '[WARN]'
    issues: List[str] = []
    print(f'\n{sep}\nPRE-FLIGHT VALIDATION\n{sep}')

    # ── 1. Folder & file existence ─────────────────────────────────────────
    print('\n-- 1. Folder & file existence --')
    for folder in plate_folders:
        p = Path(folder)
        if not p.exists():
            print(f'  {FAIL} {p.name}: not found')
            issues.append(f'{p.name}: folder missing')
            continue
        print(f'  {OK} {p.name}: exists')
        seg_dir = p / cfg.SEGMENTATION_SUBPATH
        parquet  = seg_dir / cfg.AGGREGATED_FILE
        m        = re.search(r'(?i)^P(\d+)', p.name)
        pm_path  = seg_dir / f'P_{m.group(1)}_Plate_Map.xlsx' if m else None
        for path, label in [(seg_dir, cfg.SEGMENTATION_SUBPATH),
                             (parquet,  cfg.AGGREGATED_FILE),
                             (pm_path,  pm_path.name if pm_path else '')]:
            if path is None:
                continue
            if not path.exists():
                print(f'  {FAIL}   {label}')
                issues.append(f'{p.name}: {label} missing')
            else:
                print(f'  {OK}   {label}')

    # ── 2. Parquet schema ──────────────────────────────────────────────────
    print('\n-- 2. Parquet schema --')
    REQUIRED = {'well', 'filename'} | set(cfg.MORPHOLOGY_FEATURES)
    first    = True
    for folder in plate_folders:
        p       = Path(folder)
        parquet = p / cfg.SEGMENTATION_SUBPATH / cfg.AGGREGATED_FILE
        if not parquet.exists():
            continue
        try:
            df = pd.read_parquet(parquet)
        except Exception as e:
            print(f'  {FAIL} {p.name}: unreadable — {e}')
            issues.append(f'{p.name}: parquet unreadable')
            continue
        missing = REQUIRED - set(df.columns)
        if missing:
            print(f'  {FAIL} {p.name}: missing {sorted(missing)}')
            issues.append(f'{p.name}: missing columns')
        else:
            print(f'  {OK} {p.name}: {len(df):,} rows')
        if first:
            first = False
            bad = [f for f in cfg.MORPHOLOGY_FEATURES
                   if f in df.columns and not pd.api.types.is_numeric_dtype(df[f])]
            if bad:
                print(f'  {WARN}   Non-numeric features: {bad}')
            else:
                print(f'  {OK}   All morphology features are numeric')
            if 'well' in df.columns:
                print(f'  {OK}   Well sample: {sorted(df["well"].dropna().unique())[:6]}')
            if 'point' in df.columns:
                print(f'  {OK}   Point sample: {sorted(df["point"].dropna().unique())[:5]}')
        del df; gc.collect()

    # ── 3. Plate-map validation ────────────────────────────────────────────
    print('\n-- 3. Plate map validation --')
    all_conditions: Dict[str, int] = {}
    for folder in plate_folders:
        p = Path(folder)
        m = re.search(r'(?i)^P(\d+)', p.name)
        if not m:
            continue
        seg_dir  = p / cfg.SEGMENTATION_SUBPATH
        platemap = seg_dir / f'P_{m.group(1)}_Plate_Map.xlsx'
        if not platemap.exists():
            platemap = seg_dir / f'P_{m.group(1)}_plate_map.xlsx'
        if not platemap.exists():
            continue
        try:
            pm = pd.read_excel(platemap, header=None)
        except Exception as e:
            print(f'  {FAIL} {p.name}: {e}')
            continue
        errs = []
        for r in range(pm.shape[0]):
            for c in range(pm.shape[1]):
                val = pm.iloc[r, c]
                if pd.isna(val):
                    continue
                parsed = parse_crisprI_label(str(val))
                gene   = parsed['gene'] or 'unknown'
                all_conditions[gene] = all_conditions.get(gene, 0) + 1
                if parsed['cell_line'] not in VALID_CELL_LINES:
                    errs.append(f'unknown cell_line at {chr(65+r)}{c+1:02d}')
                if parsed['atc_condition'] not in VALID_ATC:
                    errs.append(f'unknown ATC at {chr(65+r)}{c+1:02d}')
        tag = f'{WARN} {len(errs)} issues' if errs else f'{OK} clean'
        print(f'  {tag}  ({p.name})')

    if all_conditions:
        print(f'\n  Detected conditions ({len(all_conditions)} unique):')
        print(f'  {"Condition":<45}  {"Total wells":>12}')
        print(f'  {"-"*45}  {"-"*12}')
        for cond, n in sorted(all_conditions.items()):
            print(f'  {cond:<45}  {n:>12}')

    # ── 4. Filename parsing (spot-check) ───────────────────────────────────
    print('\n-- 4. Filename parsing --')
    for folder in plate_folders[:1]:
        p       = Path(folder)
        parquet = p / cfg.SEGMENTATION_SUBPATH / cfg.AGGREGATED_FILE
        if not parquet.exists():
            continue
        try:
            sample  = pd.read_parquet(parquet, columns=['filename', 'point', 'well'])
            fnames  = sample['filename'].dropna().unique()[:5]
            fn_errs = sum(1 for fn in fnames if extract_fov_from_filename(fn) == 'unknown')
            print(f'  {OK} {p.name}: {fn_errs}/{len(fnames)} failed FOV extraction')
            if len(fnames):
                print(f'       Sample: {fnames[0]}')
        except Exception as e:
            print(f'  {WARN} {p.name}: filename check failed — {e}')

    print(f'\n{sep}')
    if issues:
        print(f'PRE-FLIGHT FAILED — {len(issues)} issue(s):')
        for i, iss in enumerate(issues, 1):
            print(f'  {i}. {iss}')
        print(sep)
        sys.exit(1)
    print('PRE-FLIGHT PASSED — all checks OK')
    print(sep)


# ============================================================================
# COLOUR SYSTEM
# ============================================================================
# Factor-level colours (for main-effects violin panels)
_CL_COL  = {'MG1655': '#1A8C7A', 'ACE-1': '#7A3B1E'}
_ATC_COL = {'+ATC':   '#C00000', '-ATC':  '#70AD47'}
_PS_COL  = {'+plasmid': '#7030A0', '-plasmid': '#9DC3E6'}

_PLATE_COL: Dict[str, str] = {
    f'P{i+1}': matplotlib.colors.to_hex(matplotlib.colormaps['plasma'](v))
    for i, v in enumerate(np.linspace(0.10, 0.88, 8))
}
_FEAT_COL: Dict[str, str] = {
    'roundness':    '#4477AA', 'area_um2':     '#EE6677',
    'length_um':    '#228833', 'width_um':     '#CCBB44',
    'perimeter_um': '#66CCEE', 'aspect_ratio': '#AA3377',
    'solidity':     '#BBBBBB', 'eccentricity': '#999933',
}
_FACTOR_COLS = {'cell_line': _CL_COL, 'atc_condition': _ATC_COL, 'plasmid_status': _PS_COL}
_FACTOR_TITLES = {
    'cell_line':      'Cell line  (MG1655 vs ACE-1)',
    'atc_condition':  'ATC induction  (−ATC vs +ATC)',
    'plasmid_status': 'Plasmid carriage  (−plasmid vs +plasmid)',
}
_COL_GRID = '#E0E0E0'


# Per-feature y-axis limits for plasmid QC plots (None = auto)
_PLASMID_QC_YLIM: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    'area_um2':    (None, 15.0),
    'perimeter_um': (None, 20.0),
    'solidity':    (None, 1.0),
    'width_um':    (None, 3.0),
}



# ============================================================================
# WELL / PLATE-MAP UTILITIES
# ============================================================================
def parse_well_position(well: str) -> Optional[Tuple[int, int]]:
    s = str(well).strip()
    if len(s) >= 2 and s[0].isalpha() and s[1:].isdigit():
        r, c = ord(s[0].upper()) - ord('A'), int(s[1:]) - 1
        if 0 <= r < 8 and 0 <= c < 12:
            return r, c
    return None


def _find_platemap(plate_dir: Path, cfg: Config) -> Optional[Path]:
    m = re.search(r'(?i)^P(\d+)', plate_dir.name)
    if m is None:
        return None
    seg = plate_dir / cfg.SEGMENTATION_SUBPATH
    for cand in [seg / f'P_{m.group(1)}_Plate_Map.xlsx',
                 seg / f'P_{m.group(1)}_plate_map.xlsx']:
        if cand.exists():
            return cand
    return None


# ============================================================================
# DATA LOADING
# ============================================================================
def _load_single_plate(plate_name: str, parquet_path: Path,
                        cfg: Config) -> Optional[pd.DataFrame]:
    all_feats = list(set(cfg.MORPHOLOGY_FEATURES + cfg.FEATURES))
    cols      = ['well', 'filename', 'point'] + all_feats
    try:
        df = pd.read_parquet(parquet_path, columns=cols)
    except Exception:
        df = pd.read_parquet(parquet_path)
        df = df[[c for c in cols if c in df.columns]]

    if 'Well' in df.columns and 'well' not in df.columns:
        df = df.rename(columns={'Well': 'well'})
    if 'well' not in df.columns:
        print(f'  WARN: No well column in {plate_name} — skipping')
        return None

    platemap_path = _find_platemap(parquet_path.parent.parent, cfg)
    if platemap_path is None:
        print(f'  WARN: Plate map not found for {plate_name} — skipping')
        return None

    pm = (pd.read_excel(platemap_path, header=None)
          if platemap_path.suffix.lower() in ('.xlsx', '.xls')
          else pd.read_csv(platemap_path, header=None))

    def _well_meta(well):
        if pd.isna(well) or len(str(well)) < 2:
            return None
        try:
            ri = ord(well[0].upper()) - ord('A')
            ci = int(well[1:]) - 1
            if 0 <= ri < pm.shape[0] and 0 <= ci < pm.shape[1]:
                val = pm.iloc[ri, ci]
                if pd.notna(val):
                    return parse_crisprI_label(str(val))
        except Exception:
            pass
        return None

    well_meta = {w: _well_meta(w) for w in df['well'].unique()}
    for key in ('gene', 'cell_line', 'plasmid', 'plasmid_status', 'atc_condition'):
        df[key] = df['well'].map(lambda w, k=key: (well_meta.get(w) or {}).get(k))
    df['Label'] = df['Gene'] = df['gene']
    df['Subgroup'] = df['plasmid_status']
    df['gene'] = pd.Categorical(df['gene'])
    df = df[df['gene'].notna()].copy()
    df['plate'] = df['bio_replicate'] = plate_name

    if 'point' in df.columns:
        df['fov'] = (pd.to_numeric(df['point'].astype(str).str.extract(r'(\d+)', expand=False),
                                    errors='coerce').fillna(-1).astype(int))
        df.drop(columns='point', inplace=True)
    elif 'filename' in df.columns:
        df['fov'] = df['filename'].apply(extract_fov_from_filename)
    else:
        df['fov'] = -1

    df.drop(columns='filename', errors='ignore', inplace=True)
    return df


def load_all_plates(cfg: Config) -> pd.DataFrame:
    print('=' * 80 + '\nLOADING MULTI-PLATE DATA\n' + '=' * 80)
    plate_files = [
        (Path(f).name, Path(f) / cfg.SEGMENTATION_SUBPATH / cfg.AGGREGATED_FILE)
        for f in cfg.PLATE_FOLDERS
        if (Path(f) / cfg.SEGMENTATION_SUBPATH / cfg.AGGREGATED_FILE).exists()
    ]
    if not plate_files:
        raise FileNotFoundError('No parquet files found. Check PLATE_FOLDERS.')
    print(f'  Found {len(plate_files)} plates:')
    for name, path in plate_files:
        print(f'    {name}: {path}')

    frames = []
    for name, path in plate_files:
        df_p = _load_single_plate(name, path, cfg)
        if df_p is not None:
            frames.append(df_p)
            print(f'  Loaded {name}: {len(df_p):,} cells')
    if not frames:
        raise RuntimeError('No plates loaded successfully.')

    df = pd.concat(frames, ignore_index=True)

    feat_cols = [f for f in set(cfg.MORPHOLOGY_FEATURES + cfg.FEATURES) if f in df.columns]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors='coerce')
    for feat, (lo, hi) in [('roundness', (0, 1)), ('solidity', (0, 1)),
                            ('eccentricity', (0, 1)), ('aspect_ratio', (1, 20))]:
        if feat in df.columns:
            df[feat] = df[feat].clip(lo, hi)

    n_before = len(df)
    # Global 1st-percentile debris filter — thresholds computed on original df,
    # then applied as a single joint mask so all features share the same cell set.
    thresholds = {feat: float(df[feat].quantile(0.01)) for feat in feat_cols}
    joint_mask = pd.Series(True, index=df.index)
    for feat, lo_val in thresholds.items():
        joint_mask &= df[feat] >= lo_val
    df = df[joint_mask]

    df = df[df['well'].notna() & (df['well'] != 'nan')]
    df.reset_index(drop=True, inplace=True)
    gc.collect()

    print(f'\n  After filtering: {len(df):,} cells  ({n_before - len(df):,} removed)')
    print(f'  Plates:  {sorted(df["plate"].unique())}')
    print(f'  Genes:   {df["gene"].nunique()}')
    print('  Factorial conditions (2×2×2):')
    cond = (df.groupby(['cell_line', 'atc_condition', 'plasmid_status'], observed=True)
              .size().reset_index(name='n_cells'))
    for _, row in cond.iterrows():
        print(f'    {row["cell_line"]:<8}  {str(row["atc_condition"]):<6}  '
              f'{str(row["plasmid_status"]):<12}  {row["n_cells"]:>10,} cells')
    print(f'  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
    return df


# ============================================================================
# FACTORIAL ANALYSIS VISUALIZER
# ============================================================================
class FactorialAnalysisVisualizer:
    PLOT_CELLS_PER_COND = 5_000

    FACTORS = [
        ('cell_line',      'MG1655',   'ACE-1',     'ACE-1 − MG1655'),
        ('atc_condition',  '-ATC',     '+ATC',       '+ATC − −ATC'),
        ('plasmid_status', '-plasmid', '+plasmid',   '+plasmid − −plasmid'),
    ]
    def __init__(self, df: pd.DataFrame, output_base: Path, cfg: Config):
        self.df, self.output_base, self.cfg = df, output_base, cfg

    @staticmethod
    def _sample_per_factor(df, factor, n_per, seed=42):
        rng   = np.random.default_rng(seed)
        parts = []
        for _, grp in df.groupby(factor, observed=True):
            idx = rng.choice(len(grp), min(n_per, len(grp)), replace=False)
            parts.append(grp.iloc[idx])
        return pd.concat(parts, ignore_index=True)

    def run(self):
        print('\n' + '=' * 80 + '\nFACTORIAL ANALYSIS (2×2×2)\n' + '=' * 80)
        out = self.output_base / 'factorial_analysis'
        out.mkdir(parents=True, exist_ok=True)
        feat_cols = [f for f in self.cfg.MORPHOLOGY_FEATURES if f in self.df.columns]

        self._export_factorial_summary(feat_cols, out)

        for i, feat in enumerate(feat_cols, 1):
            print(f'  [{i}/{len(feat_cols)}] {feat}')
            self._plot_main_effects_violin(feat, out)

        self._plot_plate_reproducibility(feat_cols, out)
        self._plot_plasmid_qc(feat_cols, out)
        print(f'  [OK] Factorial analysis → {out}')

    # ── Main effects violin (3 panels per feature) ─────────────────────────
    def _plot_main_effects_violin(self, feature: str, out: Path):
        sub = out / 'main_effects'; sub.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False, dpi=self.cfg.DPI)
        unit = self.cfg.FEATURE_UNITS.get(feature, '')

        for ax, (factor, lvl_lo, lvl_hi, _) in zip(axes, self.FACTORS):
            col   = _FACTOR_COLS[factor]
            order = [lvl_lo, lvl_hi]
            samp  = self._sample_per_factor(
                self.df[[feature, factor]].dropna(), factor, self.PLOT_CELLS_PER_COND)

            sns.violinplot(x=factor, y=feature, data=samp, order=order,
                           palette={k: col[k] for k in order if k in col},
                           ax=ax, cut=2, bw_adjust=0.8, inner='quart',
                           linewidth=1.2, saturation=0.9)

            # Overlay plate means (the true inferential unit, N=6)
            for xi, lvl in enumerate(order):
                pm = (self.df[self.df[factor] == lvl]
                      .groupby('plate', observed=True)[feature].mean().values)
                rng = np.random.default_rng(xi + 7)
                ax.scatter(xi + rng.uniform(-0.09, 0.09, len(pm)), pm,
                           s=36, color='white', edgecolors='#333333',
                           linewidths=1.0, zorder=6, alpha=0.95)

            # Clip y-axis to 1–99th percentile to suppress long tails
            q_lo, q_hi = samp[feature].quantile([0.01, 0.99])
            pad = (q_hi - q_lo) * 0.15
            ax.set_ylim(q_lo - pad, q_hi + pad)

            ax.set_title(_FACTOR_TITLES[factor], fontsize=9, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(f'{feature} {unit}'.strip() if ax is axes[0] else '')
            ax.grid(True, axis='y', alpha=0.25, linestyle='--', color=_COL_GRID)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        fig.suptitle(f'{feature}: main effects of cell line, ATC, and plasmid',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(sub / f'{feature}_main_effects.png', dpi=self.cfg.DPI, bbox_inches='tight')
        plt.close()

    # ── Plate reproducibility (8 conditions, N=6 plates) ───────────────────
    def _plot_plate_reproducibility(self, feat_cols: List[str], out: Path):
        sub = out / 'plate_reproducibility'; sub.mkdir(exist_ok=True)
        plate_means = (self.df.groupby(['plate', 'gene', 'cell_line',
                                         'atc_condition', 'plasmid_status'],
                                        observed=True)[feat_cols]
                       .mean().reset_index())
        plates  = sorted(self.df['plate'].unique())
        palette = {str(x): _PLATE_COL.get(str(x), '#999999') for x in plates}

        def _key(g):
            r = plate_means[plate_means['gene'] == g].iloc[0]
            return (r['cell_line'], r['atc_condition'], r['plasmid_status'])

        for feature in feat_cols:
            data = plate_means[['gene', 'cell_line', 'atc_condition',
                                  'plasmid_status', 'plate', feature]].dropna()
            if data.empty:
                continue
            order       = sorted(data['gene'].unique(), key=_key)
            grand_means = data.groupby('gene', observed=True)[feature].mean()
            rng         = np.random.default_rng(0)

            fig, ax = plt.subplots(figsize=(max(7, len(order) * 0.9), 5), dpi=self.cfg.DPI)
            for plate in plates:
                p_data = data[data['plate'] == plate]
                xs     = [order.index(g) + rng.uniform(-0.15, 0.15)
                          for g in p_data['gene']]
                ax.scatter(xs, p_data[feature].values,
                           color=palette.get(plate, '#999'), s=50, alpha=0.85,
                           zorder=3, label=plate,
                           edgecolors='white', linewidths=0.5)

            for g, gm in grand_means.items():
                xi = order.index(g)
                ax.plot([xi - 0.35, xi + 0.35], [gm, gm],
                        color='#333333', linewidth=2.2, zorder=4)

            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha='right', fontsize=8)
            unit = self.cfg.FEATURE_UNITS.get(feature, '')
            ax.set_ylabel(f'{feature} {unit} (plate mean)'.strip(),
                          fontsize=11, fontweight='bold')
            ax.set_title(f'{feature}: plate reproducibility  (N=6; bar = grand mean)',
                         fontsize=10, fontweight='bold')
            ax.legend(title='Plate', fontsize=8, loc='upper right',
                      framealpha=0.9, ncol=2)
            ax.grid(True, axis='y', alpha=0.25, linestyle='--', color=_COL_GRID)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(sub / f'{feature}_plate_reproducibility.png',
                        dpi=self.cfg.DPI, bbox_inches='tight')
            plt.close()

    # ── Plasmid QC: NC_1–6 spread (QC only, separate subfolder) ───────────
    def _plot_plasmid_qc(self, feat_cols: List[str], out: Path):
        sub = out / 'plasmid_qc'; sub.mkdir(parents=True, exist_ok=True)
        df_plus = self.df[self.df['plasmid_status'] == '+plasmid'].copy()
        if df_plus.empty:
            return
        cl_vals  = sorted(df_plus['cell_line'].unique())
        atc_vals = sorted(df_plus['atc_condition'].unique())

        for feature in feat_cols:
            data = df_plus[[feature, 'plasmid', 'cell_line', 'atc_condition']].dropna()
            if data.empty:
                continue
            samp     = data.sample(min(30_000, len(data)), random_state=42)
            nc_order = sorted(samp['plasmid'].unique())

            fig, axes = plt.subplots(len(cl_vals), len(atc_vals),
                                     figsize=(4 * len(atc_vals), 4 * len(cl_vals)),
                                     sharey=True, squeeze=False, dpi=self.cfg.DPI)
            for ri, cl in enumerate(cl_vals):
                for ci, atc in enumerate(atc_vals):
                    ax    = axes[ri][ci]
                    sub_d = samp[(samp['cell_line'] == cl) & (samp['atc_condition'] == atc)]
                    if sub_d.empty:
                        ax.set_visible(False); continue
                    sns.boxplot(x='plasmid', y=feature, data=sub_d,
                                order=[p for p in nc_order if p in sub_d['plasmid'].values],
                                color=_PS_COL['+plasmid'], ax=ax,
                                linewidth=0.8, flierprops={'markersize': 2})
                    ax.set_title(f'{cl}  {atc}', fontsize=9, fontweight='bold')
                    ax.set_xlabel('')
                    unit = self.cfg.FEATURE_UNITS.get(feature, '')
                    ax.set_ylabel(f'{feature} {unit}'.strip() if ci == 0 else '')
                    ax.tick_params(axis='x', rotation=45, labelsize=7)
                    ylim = _PLASMID_QC_YLIM.get(feature)
                    if ylim:
                        ax.set_ylim(ylim[0], ylim[1])
                    ax.grid(True, axis='y', alpha=0.25, linestyle='--', color=_COL_GRID)
                    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

            fig.suptitle(f'{feature}: NC plasmid spread  (QC only — +plasmid cells)',
                         fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(sub / f'{feature}_nc_spread.png', dpi=self.cfg.DPI, bbox_inches='tight')
            plt.close()

    # ── Summary CSV ────────────────────────────────────────────────────────
    def _export_factorial_summary(self, feat_cols: List[str], out: Path):
        s = (self.df.groupby(['cell_line', 'atc_condition', 'plasmid_status', 'gene'],
                              observed=True)[feat_cols]
             .agg(['mean', 'std', 'count']).reset_index())
        s.columns = ['_'.join(c).strip('_') for c in s.columns]
        s.to_csv(out / 'factorial_summary.csv', index=False)


# ============================================================================
# RUN SUMMARY
# ============================================================================
def write_run_summary(df, output_base):
    plates    = sorted(df['plate'].unique())
    path      = output_base / 'run_summary.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\nCRISPRi CONTROL PLATE RUN SUMMARY\n' + '=' * 80 + '\n\n')
        f.write(f'Plates ({len(plates)}): {", ".join(plates)}\n'
                f'Total cells:    {len(df):,}\n'
                f'Conditions:     {df["gene"].nunique()}\n\n')
        f.write('FACTORIAL CONDITION BREAKDOWN (2×2×2)\n')
        cond = (df.groupby(['cell_line', 'atc_condition', 'plasmid_status', 'gene'],
                            observed=True).size().reset_index(name='n_cells'))
        for _, row in cond.iterrows():
            f.write(f'  {row["gene"]:<42}  {row["n_cells"]:>10,} cells\n')
    print(f'  [OK] Run summary → {path}')


# ============================================================================
# CLI + MAIN
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='CRISPRi Control Plate morphology analysis pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dpi', type=int, default=Config.DPI)
    p.add_argument('--skip-preflight', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    cfg.DPI = args.dpi
    cfg.PLATE_FOLDERS = PLATE_FOLDERS

    ts          = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(cfg.ROOT_DATA_DIR) / f'CRISPRiControlAnalysis_{ts}'
    output_base.mkdir(exist_ok=True)

    sep = '=' * 80
    print(f'{sep}\nCRISPRi CONTROL PLATE ANALYSIS PIPELINE\n{sep}\n'
          f'Plates:  {len(PLATE_FOLDERS)}\n'
          f'Output:  {output_base}\n{sep}')

    if not args.skip_preflight:
        run_preflight_validation(PLATE_FOLDERS, cfg)

    df = load_all_plates(cfg)

    FactorialAnalysisVisualizer(df, output_base, cfg).run()
    write_run_summary(df, output_base)

    print('\n'.join([
        sep, 'ANALYSIS COMPLETE', sep,
        f'Results → {output_base}',
        '  factorial_analysis/',
        '    main_effects/          violin per feature with plate means overlaid',
        '    plate_reproducibility/ N=6 plates × 8 conditions',
        '    plasmid_qc/            NC_1–6 spread (QC only)',
        '    factorial_summary.csv  per-condition cell counts and feature stats',
        '  run_summary.txt',
    ]))


if __name__ == '__main__':
    main()
