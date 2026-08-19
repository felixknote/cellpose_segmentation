# ============================================================================
# DEEP LEARNING MORPHOLOGY CLASSIFIER
#
# Classifies CRISPRi gene knockdowns from FOV-level morphology feature vectors.
#
# Input:  median + std of 8 morphology features per FOV  (→ 16-dim vector)
# Target: gene base name from plate_map CSV (NC_4/NC_5/NC_6 → NC, etc.)
#
# Architecture:  ResidualMLP  — deep MLP with residual connections, BatchNorm,
#                GELU activation; residual design principles from EfficientNet
#                MBConv, adapted for 1-D tabular feature vectors.
#
# Optimizer:     SAM (Foret et al. 2021) with AdamW base optimizer
# Scheduler:     CosineAnnealingLR
# Hardware:      Optimised for NVIDIA RTX 4500 Ada (TF32, BF16, torch.compile)
#
# Split strategy (Campos et al. 2018 style):
#   • Plates discovered under ROOT_DATA_DIR / P{1..6} subfolders
#   • Sorted alphabetically → indices 0-5
#   • Train: indices 0-3  |  Val: index 4  |  Test: index 5
#
# References:
#   Campos et al. 2018 — morphology classification benchmark
#   ai4ab (krentzd/ai4ab) — DL on bacterial microscopy images
#   Foret et al. 2021 — Sharpness-Aware Minimization (SAM)
# ============================================================================

import gc
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, RobustScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    from bokeh.embed import file_html
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure as bk_figure
    from bokeh.resources import CDN
    from bokeh.transform import factor_cmap
    BOKEH = True
except ImportError:
    BOKEH = False
    print("bokeh not found — HTML UMAP output skipped.  pip install bokeh")


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # ── Data ─────────────────────────────────────────────────────────────────
    ROOT_DATA_DIR        = r"D:\2025_12_19 CRISPRi Reference Plate Imaging"
    SEGMENTATION_SUBPATH = r"CellposeSAM Segmentation results"
    AGGREGATED_FILE      = "micromorph_cell_measurements.parquet"
    # Plate map CSVs: P_1_plate_map.csv … P_6_plate_map.csv  in ROOT_DATA_DIR

    # Plates sorted alphabetically; 0-indexed
    VAL_PLATE_IDX  = 4   # → P5  validation
    TEST_PLATE_IDX = 5   # → P6  held-out test

    MORPHOLOGY_FEATURES = [
        'roundness', 'area_um2', 'length_um', 'width_um',
        'perimeter_um', 'aspect_ratio', 'solidity', 'eccentricity',
    ]
    # Feature vector = 8×median + 8×std = 16-dim  (n_cells excluded)

    MIN_CELLS_PER_FOV = 100   # discard FOVs with fewer cells (FOV-level pipeline only)

    # ── Label strategy ────────────────────────────────────────────────────────
    # When True, guideRNA replicates (_1/_2/_3) are merged into their base gene
    # before training: 96 guideRNA classes → ~28 gene classes.
    # Benefits: 3× more cells per class, cleaner signal, fewer confusion sources.
    COLLAPSE_REPLICATES = True

    # ── Model ─────────────────────────────────────────────────────────────────
    # Scaled for millions of cells: wide + deep residual MLP.
    HIDDEN_DIM = 512
    N_BLOCKS   = 6
    DROPOUT    = 0.2    # less dropout — regularised by sheer data volume

    # ── Training ──────────────────────────────────────────────────────────────
    # Data lives on GPU — large batches have near-zero overhead.
    EPOCHS          = 80
    BATCH_SIZE      = 8192   # smaller batches → higher gradient diversity
    LR              = 1e-3   # higher LR matched to smaller batch size
    WEIGHT_DECAY    = 1e-4
    LABEL_SMOOTHING = 0.05
    NOISE_STD       = 0.01   # small noise — signal is strong with millions of cells
    PATIENCE        = 10
    RANDOM_STATE    = 42

    # ── Output ────────────────────────────────────────────────────────────────
    DPI = 300

    # ── Subgroup / Bokeh marker maps ──────────────────────────────────────────
    SUBGROUP_MARKERS = {"1": "o", "2": "s", "3": "^"}
    BOKEH_MARKERS    = {"o": "circle", "s": "square", "^": "triangle"}

    # ── Gene base colours (same palette as 03_morphological_map.py) ──────────
    # Variants (_1, _2, _3) are rendered as shades of their base gene colour.
    GENE_COLORS = {
        # Cell wall synthesis
        "mrcA": "#E57373", "mrcB": "#EF5350", "mrdA": "#F06292",
        "ftsI": "#EC407A", "mreB": "#FF8A65", "murA": "#FFB74D", "murC": "#FFA726",
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
        # Controls
        "NC":    "#9E9E9E",
        "WT NC": "#424242",
        "WT":    "#000000",
    }


# ============================================================================
# UTILITIES  (adapted from 02_morphological_analysis.py)
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


def parse_gene_label(label: str) -> str:
    """
    Return the full label unchanged — each guideRNA variant is its own class.

    mrcA_1, mrcA_2, mrcA_3  →  kept as separate classes (3 guideRNAs per gene)
    NC_1 … NC_6             →  kept as separate classes (6 control wells/plate)
    WT NC_1 … WT NC_6       →  kept as separate classes
    """
    return str(label).strip()


def _base_gene(label: str) -> str:
    """Extract base gene name for colour lookup: 'mrcA_1' → 'mrcA', 'WT NC_3' → 'WT NC'."""
    s = str(label).strip()
    if '_' in s:
        base, _, suffix = s.rpartition('_')
        if suffix.isdigit():
            return base
    return s


def _shade_hex(hex_color: str, variant_idx: int, n_variants: int) -> str:
    """
    Return a lighter or darker shade of hex_color based on variant index.
    variant_idx=0 → base, positive → lighter, negative → darker.
    Shading range is ±25% lightness.
    """
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if n_variants <= 1:
        return hex_color
    # map idx 0..n-1 → factor -0.25 .. +0.25
    factor = ((variant_idx / (n_variants - 1)) - 0.5) * 0.5
    r2 = int(max(0, min(255, r + factor * 255)))
    g2 = int(max(0, min(255, g + factor * 255)))
    b2 = int(max(0, min(255, b + factor * 255)))
    return f'#{r2:02x}{g2:02x}{b2:02x}'


def label_color_map(labels: List[str], gene_colors: dict) -> dict:
    """
    Build label → hex color mapping.
    Variants of the same base gene get distinct shades of the base color.
    """
    from collections import defaultdict
    # Group labels by base gene
    groups: dict = defaultdict(list)
    for lbl in sorted(set(labels)):
        groups[_base_gene(lbl)].append(lbl)

    result = {}
    for base, variants in groups.items():
        base_col = gene_colors.get(base, '#999999')
        for i, lbl in enumerate(sorted(variants)):
            result[lbl] = _shade_hex(base_col, i, len(variants))
    return result


# ============================================================================
# DATA LOADING
# ============================================================================
def _find_plate_dirs(cfg: Config) -> List[Tuple[int, str, Path]]:
    """Return sorted list of (plate_idx, plate_name, parquet_path)."""
    root = Path(cfg.ROOT_DATA_DIR)
    if not root.exists():
        raise FileNotFoundError(f"Root data directory not found: {root}")

    found = []
    for plate_dir in sorted(root.iterdir()):
        if not plate_dir.is_dir():
            continue
        parquet = plate_dir / cfg.SEGMENTATION_SUBPATH / cfg.AGGREGATED_FILE
        if parquet.exists():
            found.append((plate_dir.name, parquet))

    if not found:
        raise FileNotFoundError(f"No parquet files found under {root}")

    return [(i, name, path) for i, (name, path) in enumerate(found)]


def _load_plate(plate_idx: int, plate_name: str, parquet_path: Path,
                cfg: Config) -> pd.DataFrame:
    """Load one plate: parquet + plate_map CSV → cell-level DataFrame with gene labels."""
    features  = cfg.MORPHOLOGY_FEATURES
    requested = ['Well', 'filename'] + features

    try:
        df = pd.read_parquet(parquet_path, columns=requested)
    except Exception:
        df = pd.read_parquet(parquet_path)
        df = df[[c for c in requested if c in df.columns]]

    # ── Plate map: P_1_plate_map.csv … lives in ROOT_DATA_DIR ────────────
    plate_num      = plate_name.replace('P', '').replace('_', '')
    csv_name       = f"P_{plate_num}_plate_map.csv"
    plate_map_path = Path(cfg.ROOT_DATA_DIR) / csv_name
    if not plate_map_path.exists():
        raise FileNotFoundError(
            f"Plate map not found: {plate_map_path}\n"
            f"  Expected: P_{{N}}_plate_map.csv in {cfg.ROOT_DATA_DIR}"
        )
    plate_map = pd.read_csv(plate_map_path, header=None)

    # ── Well → raw label → base gene name ────────────────────────────────
    def _well_to_raw_label(well) -> Optional[str]:
        if pd.isna(well) or len(str(well)) < 2:
            return None
        try:
            row_idx = ord(str(well)[0].upper()) - ord('A')
            col_idx = int(str(well)[1:]) - 1
            if 0 <= row_idx < plate_map.shape[0] and 0 <= col_idx < plate_map.shape[1]:
                val = plate_map.iloc[row_idx, col_idx]
                return str(val) if pd.notna(val) else None
        except Exception:
            return None

    well_map     = {w: _well_to_raw_label(w) for w in df['Well'].unique()}
    label_series = df['Well'].map(well_map)

    # Strip replicate suffixes (_1 … _6) → base gene class
    unique_raw  = [l for l in label_series.unique() if pd.notna(l)]
    gene_lookup = {raw: (_base_gene(raw) if cfg.COLLAPSE_REPLICATES else parse_gene_label(raw))
                   for raw in unique_raw}

    df['gene']      = label_series.map(gene_lookup)
    df['fov']       = df['filename'].apply(extract_fov_from_filename) \
                      if 'filename' in df.columns else 'unknown'
    df['well']      = df['Well'].astype(str)
    df['plate_idx'] = plate_idx
    df['plate']     = plate_name

    drop_cols = [c for c in ('Well', 'filename') if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # ── Feature clipping + validation ─────────────────────────────────────
    feat_cols = [f for f in features if f in df.columns]
    df[feat_cols] = df[feat_cols].astype(np.float32)

    for feat, (lo, hi) in [('roundness', (0, 1)), ('solidity', (0, 1)),
                             ('eccentricity', (0, 1)), ('aspect_ratio', (1, 20))]:
        if feat in df.columns:
            df[feat] = df[feat].clip(lo, hi)

    for feat in feat_cols:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        df = df[np.isfinite(df[feat])]

    # 1–99th percentile filter
    for feat in feat_cols:
        q1, q99 = df[feat].quantile(0.01), df[feat].quantile(0.99)
        df = df[(df[feat] >= q1) & (df[feat] <= q99)]

    df = df[df['gene'].notna()]
    return df


def load_all_plates(cfg: Config) -> pd.DataFrame:
    print("=" * 70)
    print("LOADING MULTI-PLATE DATA")
    print("=" * 70)

    plates = _find_plate_dirs(cfg)
    print(f"  Found {len(plates)} plates:")
    for idx, name, path in plates:
        role = "train"
        if idx == cfg.VAL_PLATE_IDX:  role = "val"
        if idx == cfg.TEST_PLATE_IDX: role = "test"
        print(f"    [{idx}] {name}  ({role})  →  {path}")

    frames = []
    for idx, name, path in plates:
        df_plate = _load_plate(idx, name, path, cfg)
        print(f"  Loaded [{idx}] {name}: {len(df_plate):,} cells  "
              f"| {df_plate['gene'].nunique()} gene classes")
        frames.append(df_plate)

    df = pd.concat(frames, ignore_index=True)
    gc.collect()
    print(f"\n  Total: {len(df):,} cells  |  "
          f"{df['plate'].nunique()} plates  |  "
          f"{df['gene'].nunique()} unique gene classes")
    return df


# ============================================================================
# FOV AGGREGATION
# ============================================================================
def aggregate_to_fov(df: pd.DataFrame, features: List[str],
                     min_cells: int) -> pd.DataFrame:
    """
    Aggregate cell-level measurements to individual FOV level.

    Each well contains 21 FOVs; each FOV is treated as one sample.
    Groups by (plate_idx, plate, well, fov, gene).

    Output feature vector per FOV = 8×median + 8×std = 16-dim.
    n_cells is retained as metadata but NOT included in the feature vector.
    FOVs with n_cells < min_cells are discarded.
    """
    feat_cols  = [f for f in features if f in df.columns]
    group_keys = ['plate_idx', 'plate', 'well', 'fov', 'gene']

    agg_funcs = {}
    for f in feat_cols:
        agg_funcs[f'{f}_median'] = pd.NamedAgg(column=f, aggfunc='median')
        agg_funcs[f'{f}_std']    = pd.NamedAgg(column=f, aggfunc='std')
    agg_funcs['n_cells'] = pd.NamedAgg(column=feat_cols[0], aggfunc='count')

    fov_df = (
        df.groupby(group_keys, observed=True)
          .agg(**agg_funcs)
          .reset_index()
    )

    # std is NaN when a FOV has only one cell — fill with 0
    std_cols = [f'{f}_std' for f in feat_cols]
    fov_df[std_cols] = fov_df[std_cols].fillna(0.0)

    fov_df = fov_df[fov_df['n_cells'] >= min_cells].reset_index(drop=True)

    print(f"\n  FOV aggregation: {len(fov_df):,} FOVs  "
          f"(min {min_cells} cells/FOV filter applied)")
    return fov_df


# ============================================================================
# PREPROCESSING & SPLITS
# ============================================================================
def prepare_splits(fov_df: pd.DataFrame, cfg: Config):
    """
    Plate-stratified split → scale → encode labels.

    Feature vector = [f_median for f in features] + [f_std for f in features]
    = 16-dim.  n_cells is excluded.

    Scaler is fitted on train FOVs only and applied to val/test unchanged.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test,
             label_encoder, (robust_scaler, quantile_scaler),
             feat_vec_cols, test_meta_df
    """
    features = cfg.MORPHOLOGY_FEATURES
    feat_vec_cols = (
        [f'{f}_median' for f in features] +
        [f'{f}_std'    for f in features]
    )
    feat_vec_cols = [c for c in feat_vec_cols if c in fov_df.columns]

    mask_val   = fov_df['plate_idx'] == cfg.VAL_PLATE_IDX
    mask_test  = fov_df['plate_idx'] == cfg.TEST_PLATE_IDX
    mask_train = ~mask_val & ~mask_test

    train_df = fov_df[mask_train]
    val_df   = fov_df[mask_val]
    test_df  = fov_df[mask_test]

    print(f"\n  Split sizes  →  "
          f"train: {len(train_df):,}  |  "
          f"val: {len(val_df):,}  |  "
          f"test: {len(test_df):,}  FOVs")

    # ── FOVs per gene diagnostic ───────────────────────────────────────────
    print("\n  FOVs per gene:")
    all_genes = sorted(set(train_df['gene']) | set(val_df['gene']) | set(test_df['gene']))
    header = f"  {'Gene':<12}  {'Train':>6}  {'Val':>6}  {'Test':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for g in all_genes:
        tr = (train_df['gene'] == g).sum()
        va = (val_df['gene']   == g).sum()
        te = (test_df['gene']  == g).sum()
        print(f"  {g:<12}  {tr:>6}  {va:>6}  {te:>6}")
    print(f"  {'TOTAL':<12}  {len(train_df):>6}  {len(val_df):>6}  {len(test_df):>6}")

    # ── Label encoding (fit on train classes only) ─────────────────────────
    le = LabelEncoder()
    le.fit(train_df['gene'])

    val_df  = val_df[val_df['gene'].isin(le.classes_)]
    test_df = test_df[test_df['gene'].isin(le.classes_)]

    y_train = le.transform(train_df['gene'])
    y_val   = le.transform(val_df['gene'])
    y_test  = le.transform(test_df['gene'])

    # ── Per-plate median normalization ────────────────────────────────────────
    # Each plate subtracts its own per-feature median.  This is purely technical
    # (removes plate-level batch offsets); it is NOT label-dependent so applying
    # it to val/test plates independently does not cause data leakage.
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    for df_split in (train_df, val_df, test_df):
        for pid in df_split['plate_idx'].unique():
            mask = df_split['plate_idx'] == pid
            plate_med = df_split.loc[mask, feat_vec_cols].median()
            df_split.loc[mask, feat_vec_cols] -= plate_med.values

    # ── Scaling: RobustScaler → QuantileTransformer (fit on train) ─────────
    X_train_raw = train_df[feat_vec_cols].values.astype(np.float32)
    X_val_raw   = val_df[feat_vec_cols].values.astype(np.float32)
    X_test_raw  = test_df[feat_vec_cols].values.astype(np.float32)

    rs = RobustScaler()
    X_train_rs = rs.fit_transform(X_train_raw)
    X_val_rs   = rs.transform(X_val_raw)
    X_test_rs  = rs.transform(X_test_raw)

    n_q = min(1_000, len(X_train_rs))
    qt  = QuantileTransformer(n_quantiles=n_q, output_distribution='normal',
                               random_state=cfg.RANDOM_STATE)
    X_train = qt.fit_transform(X_train_rs).astype(np.float32)
    X_val   = qt.transform(X_val_rs).astype(np.float32)
    X_test  = qt.transform(X_test_rs).astype(np.float32)

    print(f"  Feature vector dim: {X_train.shape[1]}")
    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")

    # Retain metadata for Bokeh hover tooltips
    test_meta = test_df[['plate', 'well', 'fov', 'gene', 'n_cells']].reset_index(drop=True)

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            le, (rs, qt), feat_vec_cols, test_meta)


def prepare_splits_cell_level(df: pd.DataFrame, cfg: Config):
    """
    Cell-level plate-stratified split → per-plate normalise → scale.

    All available cells are used (no subsampling).  Class balance during
    training is handled by WeightedRandomSampler + inverse-frequency loss
    weights in train_model().

    Feature vector = 8-dim raw morphology features (one row per cell).
    Returns same interface as prepare_splits():
        X_train, y_train, X_val, y_val, X_test, y_test,
        label_encoder, (robust_scaler, quantile_scaler),
        feat_cols, test_meta_df
    test_meta_df retains plate / well / fov / gene for FOV-level majority vote.
    """
    features  = cfg.MORPHOLOGY_FEATURES
    feat_cols = [f for f in features if f in df.columns]

    mask_val   = df['plate_idx'] == cfg.VAL_PLATE_IDX
    mask_test  = df['plate_idx'] == cfg.TEST_PLATE_IDX
    mask_train = ~mask_val & ~mask_test

    train_df = df[mask_train].copy()
    val_df   = df[mask_val].copy()
    test_df  = df[mask_test].copy()

    print(f"\n  Split sizes  →  "
          f"train: {len(train_df):,}  |  val: {len(val_df):,}  |  test: {len(test_df):,}  cells")

    # ── Cells per gene diagnostic ──────────────────────────────────────────
    print("\n  Cells per gene:")
    all_genes = sorted(set(train_df['gene']) | set(val_df['gene']) | set(test_df['gene']))
    header = f"  {'Gene':<16}  {'Train':>8}  {'Val':>8}  {'Test':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for g in all_genes:
        tr = (train_df['gene'] == g).sum()
        va = (val_df['gene']   == g).sum()
        te = (test_df['gene']  == g).sum()
        print(f"  {g:<16}  {tr:>8}  {va:>8}  {te:>8}")
    print(f"  {'TOTAL':<16}  {len(train_df):>8}  {len(val_df):>8}  {len(test_df):>8}")

    # ── Label encoding (fit on train classes only) ─────────────────────────
    le = LabelEncoder()
    le.fit(train_df['gene'])

    val_df  = val_df[val_df['gene'].isin(le.classes_)]
    test_df = test_df[test_df['gene'].isin(le.classes_)]

    y_train = le.transform(train_df['gene'])
    y_val   = le.transform(val_df['gene'])
    y_test  = le.transform(test_df['gene'])

    # ── Per-plate median normalisation ────────────────────────────────────
    for df_split in (train_df, val_df, test_df):
        for pid in df_split['plate_idx'].unique():
            mask      = df_split['plate_idx'] == pid
            plate_med = df_split.loc[mask, feat_cols].median()
            df_split.loc[mask, feat_cols] -= plate_med.values

    # ── Scaling: RobustScaler → QuantileTransformer (fit on train) ────────
    X_train_raw = train_df[feat_cols].values.astype(np.float32)
    X_val_raw   = val_df[feat_cols].values.astype(np.float32)
    X_test_raw  = test_df[feat_cols].values.astype(np.float32)

    rs = RobustScaler()
    X_train_rs = rs.fit_transform(X_train_raw)
    X_val_rs   = rs.transform(X_val_raw)
    X_test_rs  = rs.transform(X_test_raw)

    n_q = min(10_000, len(X_train_rs))
    qt  = QuantileTransformer(n_quantiles=n_q, output_distribution='normal',
                               random_state=cfg.RANDOM_STATE)
    X_train = qt.fit_transform(X_train_rs).astype(np.float32)
    X_val   = qt.transform(X_val_rs).astype(np.float32)
    X_test  = qt.transform(X_test_rs).astype(np.float32)

    print(f"\n  Feature vector dim: {X_train.shape[1]}")
    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")

    # Retain cell-level metadata for FOV majority vote + hover tooltips
    meta_cols = [c for c in ('plate', 'plate_idx', 'well', 'fov', 'gene')
                 if c in test_df.columns]
    test_meta = test_df[meta_cols].reset_index(drop=True)

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            le, (rs, qt), feat_cols, test_meta)


# ============================================================================
# PYTORCH DATASET
# ============================================================================
class MorphologyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ============================================================================
# MODEL  — ResidualMLP
# ============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ResidualMLP(nn.Module):
    """
    Input stem  → N residual blocks → classification head.

    Each residual block: Linear→BN→GELU→Dropout→Linear→BN + skip → GELU
    Design follows EfficientNet compound scaling / MBConv principles for 1-D
    tabular feature vectors.
    """
    def __init__(self, in_features: int, hidden_dim: int,
                 n_blocks: int, n_classes: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

    def embed(self, x):
        """Return penultimate (pre-head) representation for UMAP."""
        return self.blocks(self.stem(x))


# ============================================================================
# SAM OPTIMIZER  (Foret et al. 2021 — inline, no external dependency)
# ============================================================================
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization.

    Two-step update:
        loss = criterion(model(x), y); loss.backward()
        optimizer.first_step(zero_grad=True)
        criterion(model(x), y).backward()
        optimizer.second_step(zero_grad=True)
    """
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05,
                 adaptive: bool = False, **kwargs):
        assert rho >= 0, "rho must be non-negative"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group['adaptive'] else torch.ones_like(p)) \
                      * p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("Use first_step / second_step explicitly.")

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norms = [
            ((torch.abs(p) if group['adaptive'] else torch.ones_like(p)) * p.grad)
            .norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ============================================================================
# CLASS-IMBALANCE UTILITIES
# ============================================================================
def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Soft log-inverse-frequency class weights for CrossEntropyLoss.

    weight_i = 1 / log(1 + count_i)

    Log damping keeps the max weight ratio ~1.7× on this dataset — a gentle
    nudge on top of balanced GPULoader sampling (which already equalises class
    frequencies).  Normalised so mean(weights over active classes) = 1.
    Classes with zero training samples receive weight=0.
    """
    counts  = np.bincount(y, minlength=n_classes).astype(np.float64)
    w       = np.where(counts > 0, 1.0 / np.log1p(counts), 0.0).astype(np.float32)
    active  = w > 0
    if active.sum() > 0:
        w[active] *= float(active.sum()) / float(w[active].sum())  # mean = 1
    return torch.from_numpy(w)


class GPULoader:
    """
    Stores the entire dataset on the GPU as a single tensor.
    No CPU↔GPU transfers per batch — eliminates DataLoader overhead entirely.

    For training, balanced class sampling is done with numpy (no torch.multinomial
    limit) and the sampled indices are passed as a GPU index tensor each epoch.

    Memory estimate: 20M cells × 8 features × float32 = ~640 MB train set.
    Fits easily within 24 GB VRAM alongside model activations.
    """
    N_PER_CLASS = 50_000  # balanced samples drawn per class per epoch (train only)
    # 50k × 96 classes = 4.8M cells/epoch — covers ~25% of training data each epoch

    def __init__(self, X: np.ndarray, y: np.ndarray, device: torch.device,
                 n_classes: int = 0, balanced: bool = False, seed: int = 42):
        self.X        = torch.from_numpy(X).to(device)
        self.y        = torch.from_numpy(y).long().to(device)
        self.balanced = balanced
        self.device   = device
        self._rng     = np.random.RandomState(seed)
        if balanced and n_classes > 0:
            y_np = y  # numpy array for indexing
            self._class_idx = [np.where(y_np == c)[0] for c in range(n_classes)]
        else:
            self._class_idx = []

    def iter_batches(self, batch_size: int):
        """Yield (X_batch, y_batch) GPU tensors, balanced if requested."""
        if self.balanced and self._class_idx:
            parts = [
                self._rng.choice(ci, self.N_PER_CLASS, replace=True)
                for ci in self._class_idx if len(ci) > 0
            ]
            idx = np.concatenate(parts)
            self._rng.shuffle(idx)
            idx_t = torch.from_numpy(idx).to(self.device)
        else:
            idx_t = torch.randperm(len(self.y), device=self.device)

        for start in range(0, len(idx_t), batch_size):
            batch_idx = idx_t[start:start + batch_size]
            yield self.X[batch_idx], self.y[batch_idx]

    def __len__(self) -> int:
        if self.balanced and self._class_idx:
            n_active = sum(1 for ci in self._class_idx if len(ci) > 0)
            return self.N_PER_CLASS * n_active
        return len(self.y)


# ============================================================================
# TRAINING
# ============================================================================
def build_loaders(X_train, y_train, X_val, y_val, n_classes: int, cfg: Config,
                  device: torch.device):
    train_loader = GPULoader(X_train, y_train, device, n_classes=n_classes,
                             balanced=True, seed=cfg.RANDOM_STATE)
    val_loader   = GPULoader(X_val,   y_val,   device, balanced=False)
    return train_loader, val_loader


def train_model(model: nn.Module, train_loader: GPULoader,
                val_loader: GPULoader, optimizer: torch.optim.Optimizer,
                class_weights: torch.Tensor,
                n_classes: int, cfg: Config, output_dir: Path) -> Dict:
    device          = next(model.parameters()).device
    n_classes_total = n_classes
    # GPULoader balanced sampling already equalises class frequency; the log-
    # inverse-frequency loss weights add only a mild additional correction (~1.7×).
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=cfg.LABEL_SMOOTHING,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    use_amp   = device.type == 'cuda'
    amp_dtype = torch.bfloat16  # BF16 native on Ada Lovelace

    best_val_loss  = float('inf')
    patience_count = 0
    history        = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    n_train_batches = max(1, len(train_loader) // cfg.BATCH_SIZE)
    n_val_batches   = max(1, len(val_loader)   // cfg.BATCH_SIZE)

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    epoch_bar = tqdm(range(1, cfg.EPOCHS + 1), desc="Epochs", unit="ep",
                     dynamic_ncols=True)

    for epoch in epoch_bar:
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_train      = 0
        train_bar    = tqdm(train_loader.iter_batches(cfg.BATCH_SIZE),
                            total=n_train_batches,
                            desc=f"  Train {epoch:3d}", unit="batch",
                            leave=False, dynamic_ncols=True)
        for X_batch, y_batch in train_bar:
            # Gaussian noise augmentation on features
            noise   = torch.randn_like(X_batch) * cfg.NOISE_STD
            X_noisy = X_batch + noise

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                 enabled=use_amp):
                loss = criterion(model(X_noisy), y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y_batch)
            n_train      += len(y_batch)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(n_train, 1)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss      = 0.0
        total         = 0
        class_correct = np.zeros(n_classes_total, dtype=np.int64)
        class_total   = np.zeros(n_classes_total, dtype=np.int64)
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader.iter_batches(cfg.BATCH_SIZE * 4),
                                          total=n_val_batches,
                                          desc=f"  Val   {epoch:3d}", unit="batch",
                                          leave=False, dynamic_ncols=True):
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                     enabled=use_amp):
                    logits = model(X_batch)
                val_loss += criterion(logits.float(), y_batch).item() * len(y_batch)
                preds    = logits.argmax(dim=1)
                total   += len(y_batch)
                for c in range(n_classes_total):
                    mask             = y_batch == c
                    class_total[c]  += mask.sum().item()
                    class_correct[c]+= (preds[mask] == c).sum().item()

        val_loss /= total
        # Macro accuracy: mean per-class recall — matches what BalancedSampler optimises
        per_class_acc = np.where(class_total > 0,
                                  class_correct / class_total, np.nan)
        val_macro_acc = float(np.nanmean(per_class_acc))
        val_acc_overall = class_correct.sum() / total   # kept for logging

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_macro_acc)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            macro_acc=f"{val_macro_acc:.3f}",
        )

        # ── Checkpoint & early stopping on macro accuracy ──────────────────
        # Use -macro_acc as the quantity to minimise (higher is better).
        monitor = -val_macro_acc
        if monitor < best_val_loss:
            best_val_loss  = monitor
            patience_count = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_count += 1
            if patience_count >= cfg.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg.PATIENCE} epochs)")
                break

    print(f"\n  Best macro val_acc: {-best_val_loss:.4f}")
    return history


# ============================================================================
# EVALUATION & VISUALISATION
# ============================================================================

# ── Confusion matrix helpers ─────────────────────────────────────────────────

def _rand_acc(support: np.ndarray) -> float:
    """Expected accuracy of a stratified random classifier: sum(p_i^2)."""
    p = support / support.sum()
    return float((p ** 2).sum())


def _plot_confusion_matrix(cm_norm: np.ndarray, labels: List[str],
                            title: str, path: Path,
                            acc: float, rand_acc: float, dpi: int):
    """Save normalised confusion matrix heatmap annotated with accuracy metrics."""
    n_cls = len(labels)
    fig_h = max(8, n_cls * 0.4)
    fig, ax = plt.subplots(figsize=(fig_h * 1.2, fig_h))
    sns.heatmap(cm_norm, annot=(n_cls <= 40),
                fmt='.2f' if n_cls <= 40 else '',
                cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.3 if n_cls <= 40 else 0)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True',      fontsize=11)
    fold = acc / rand_acc if rand_acc > 0 else float('nan')
    ax.set_title(
        f'{title}\n'
        f'acc={acc:.3f}  rand_acc={rand_acc:.3f}  fold/rand={fold:.1f}x',
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _binarize_cm(cm: np.ndarray) -> np.ndarray:
    """Winner-takes-all per row: mark 1 at highest-count column, 0 elsewhere."""
    cm_bin = np.zeros_like(cm)
    for i in range(len(cm)):
        if cm[i].sum() > 0:
            cm_bin[i, int(cm[i].argmax())] = 1
    return cm_bin


def _gene_level_cm(y_true_labels: np.ndarray, y_pred_labels: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Aggregate guideRNA predictions to base-gene level and return (cm, cm_norm, classes)."""
    from sklearn.metrics import confusion_matrix as _skl_cm
    y_true_gene  = np.array([_base_gene(l) for l in y_true_labels])
    y_pred_gene  = np.array([_base_gene(l) for l in y_pred_labels])
    gene_classes = sorted(set(y_true_gene))
    cm           = _skl_cm(y_true_gene, y_pred_gene, labels=gene_classes)
    cm_norm      = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    return cm, cm_norm, gene_classes
def _bokeh_umap(emb2d: np.ndarray, gene_labels: np.ndarray,
                meta: pd.DataFrame, output_dir: Path, cfg: Config):
    """Save interactive Bokeh HTML of UMAP embeddings with hover tooltips.

    guideRNA variant suffix (_1/_2/_3) is mapped to distinct Bokeh markers
    via Config.SUBGROUP_MARKERS / Config.BOKEH_MARKERS.
    """
    if not BOKEH:
        return

    gene_colors = cfg.GENE_COLORS
    color_map   = label_color_map(gene_labels.tolist(), gene_colors)

    def _bokeh_marker(label: str) -> str:
        s = str(label).strip()
        if '_' in s:
            suffix = s.rpartition('_')[2]
            if suffix.isdigit():
                mpl_sym = cfg.SUBGROUP_MARKERS.get(suffix, 'o')
                return cfg.BOKEH_MARKERS.get(mpl_sym, 'circle')
        return 'circle'

    n_cells_col = meta['n_cells'].tolist() if 'n_cells' in meta.columns else ['—'] * len(meta)

    source = ColumnDataSource(dict(
        x       = emb2d[:, 0].tolist(),
        y       = emb2d[:, 1].tolist(),
        gene    = gene_labels.tolist(),
        plate   = meta['plate'].tolist(),
        well    = meta['well'].tolist(),
        fov     = meta['fov'].astype(str).tolist(),
        n_cells = n_cells_col,
        color   = [color_map[g] for g in gene_labels],
        marker  = [_bokeh_marker(g) for g in gene_labels],
    ))

    p = bk_figure(
        title  = "UMAP — Penultimate Layer Embeddings (test set)",
        width  = 900, height = 700,
        tools  = "pan,wheel_zoom,box_zoom,reset,save",
    )
    p.add_tools(HoverTool(tooltips=[
        ("Gene",    "@gene"),
        ("Plate",   "@plate"),
        ("Well",    "@well"),
        ("FOV",     "@fov"),
        ("n_cells", "@n_cells"),
        ("UMAP 1",  "@x{0.2f}"),
        ("UMAP 2",  "@y{0.2f}"),
    ]))
    p.scatter('x', 'y', source=source, color='color', marker='marker',
              size=6, alpha=0.7, line_color=None)
    p.xaxis.axis_label = "UMAP 1"
    p.yaxis.axis_label = "UMAP 2"

    html_path = output_dir / 'embeddings_umap.html'
    with open(html_path, 'w') as f:
        f.write(file_html(p, CDN, "UMAP Embeddings"))
    print(f"  Saved embeddings_umap.html  (interactive Bokeh)")


def evaluate_and_save(model: nn.Module, X_test: np.ndarray,
                      y_test: np.ndarray, label_encoder: LabelEncoder,
                      test_meta: pd.DataFrame, history: Dict,
                      output_dir: Path, cfg: Config):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, accuracy_score)

    device  = next(model.parameters()).device
    use_amp = device.type == 'cuda'
    model.eval()

    # ── Inference in chunks to avoid OOM on large test sets ───────────────
    chunk     = 32_768
    all_logits, all_embeds = [], []
    for start in range(0, len(X_test), chunk):
        X_chunk = torch.from_numpy(X_test[start:start + chunk]).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type,
                                              dtype=torch.bfloat16, enabled=use_amp):
            logits_c = model(X_chunk)
            embed_c  = model.embed(X_chunk).float()
        all_logits.append(logits_c.cpu())
        all_embeds.append(embed_c.cpu())
    logits     = torch.cat(all_logits, dim=0)
    embeddings = torch.cat(all_embeds, dim=0).numpy()

    preds = logits.argmax(dim=1).numpy()
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average='macro', zero_division=0)
    print(f"\n  Cell-level test accuracy: {acc:.4f}  |  Macro-F1: {f1:.4f}")

    # ── FOV-level majority vote ────────────────────────────────────────────
    fov_acc          = None
    y_fov_true       = None
    y_fov_pred       = None
    fov_true_labels  = None
    fov_pred_labels  = None
    fov_classes      = list(label_encoder.classes_)

    if {'plate', 'well', 'fov'}.issubset(test_meta.columns):
        meta_preds         = test_meta.copy()
        meta_preds['pred'] = preds
        meta_preds['true'] = label_encoder.inverse_transform(y_test)
        fov_vote = (
            meta_preds.groupby(['plate', 'well', 'fov'])['pred']
            .agg(lambda x: int(pd.Series(x).mode().iloc[0]))
            .reset_index()
        )
        fov_true = (
            meta_preds.groupby(['plate', 'well', 'fov'])['true']
            .agg(lambda x: x.iloc[0])
            .reset_index()
        )
        fov_true['true_class'] = label_encoder.transform(fov_true['true'])
        y_fov_true      = fov_true['true_class'].values
        y_fov_pred      = fov_vote['pred'].values
        fov_acc         = accuracy_score(y_fov_true, y_fov_pred)
        fov_true_labels = label_encoder.inverse_transform(y_fov_true)
        fov_pred_labels = label_encoder.inverse_transform(y_fov_pred)
        print(f"  FOV-level majority-vote accuracy:  {fov_acc:.4f}")

    # ── Classification report (cell-level) ────────────────────────────────
    report    = classification_report(y_test, preds,
                                       target_names=label_encoder.classes_,
                                       zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / 'classification_report.csv')
    print(f"  Saved classification_report.csv")

    # ── Confusion matrices — FOV majority-vote level ───────────────────────
    # Using FOV predictions eliminates per-cell noise and directly reflects
    # image-level classification quality (one vote per field of view).
    if y_fov_true is not None:
        cm_fov      = confusion_matrix(y_fov_true, y_fov_pred)
        cm_fov_norm = cm_fov.astype(float) / cm_fov.sum(axis=1, keepdims=True).clip(1)
        support_fov = cm_fov.sum(axis=1).astype(float)
        ra_fov      = _rand_acc(support_fov)

        _plot_confusion_matrix(
            cm_fov_norm, fov_classes,
            'Confusion Matrix — FOV majority-vote',
            output_dir / 'confusion_matrix_fov.png',
            fov_acc, ra_fov, cfg.DPI,
        )
        print(f"  Saved confusion_matrix_fov.png")

        # binarized
        cm_fov_bin  = _binarize_cm(cm_fov)
        bin_acc_fov = float(np.diag(cm_fov_bin).sum()) / max(len(fov_classes), 1)
        _plot_confusion_matrix(
            cm_fov_bin.astype(float), fov_classes,
            'Confusion Matrix — FOV majority-vote (binarized)',
            output_dir / 'confusion_matrix_fov_binary.png',
            bin_acc_fov, ra_fov, cfg.DPI,
        )
        print(f"  Saved confusion_matrix_fov_binary.png")

        # When replicates are NOT collapsed, also produce a gene-level CM
        # by aggregating FOV-vote predictions to base gene.
        if not cfg.COLLAPSE_REPLICATES:
            cm_gene, cm_gene_norm, gene_classes = _gene_level_cm(
                fov_true_labels, fov_pred_labels)
            support_gene  = cm_gene.sum(axis=1).astype(float)
            ra_gene       = _rand_acc(support_gene)
            gene_true_enc = np.array([gene_classes.index(_base_gene(l))
                                      for l in fov_true_labels])
            gene_pred_enc = np.array([gene_classes.index(_base_gene(l))
                                      for l in fov_pred_labels])
            acc_gene = accuracy_score(gene_true_enc, gene_pred_enc)
            _plot_confusion_matrix(
                cm_gene_norm, gene_classes,
                'Confusion Matrix — gene level (FOV vote)',
                output_dir / 'confusion_matrix_gene.png',
                acc_gene, ra_gene, cfg.DPI,
            )
            cm_gene_bin  = _binarize_cm(cm_gene)
            bin_acc_gene = float(np.diag(cm_gene_bin).sum()) / max(len(gene_classes), 1)
            _plot_confusion_matrix(
                cm_gene_bin.astype(float), gene_classes,
                'Confusion Matrix — gene level binarized (FOV vote)',
                output_dir / 'confusion_matrix_gene_binary.png',
                bin_acc_gene, ra_gene, cfg.DPI,
            )
            print(f"  Saved confusion_matrix_gene.png + _binary.png")
    else:
        # Fallback: cell-level CM when no FOV metadata
        true_labels  = label_encoder.inverse_transform(y_test)
        pred_labels  = label_encoder.inverse_transform(preds)
        cm           = confusion_matrix(y_test, preds)
        cm_norm      = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        support      = cm.sum(axis=1).astype(float)
        ra           = _rand_acc(support)
        _plot_confusion_matrix(cm_norm, fov_classes,
                               'Confusion Matrix — cell level',
                               output_dir / 'confusion_matrix_cell.png',
                               acc, ra, cfg.DPI)
        print(f"  Saved confusion_matrix_cell.png")

    # ── Training curves ────────────────────────────────────────────────────
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, history['train_loss'], label='Train loss')
    axes[0].plot(ep, history['val_loss'],   label='Val loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss'); axes[0].legend()
    axes[1].plot(ep, history['val_acc'])
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Macro Accuracy')
    axes[1].set_title('Validation Macro Accuracy (mean per-class recall)')
    plt.tight_layout()
    fig.savefig(output_dir / 'training_curves.png', dpi=cfg.DPI)
    plt.close(fig)
    print(f"  Saved training_curves.png")

    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)

    # ── UMAP of penultimate embeddings ─────────────────────────────────────
    try:
        import umap
        # Sub-sample for UMAP speed: max 50k points
        n_umap = min(50_000, len(embeddings))
        if n_umap < len(embeddings):
            rng_umap = np.random.RandomState(cfg.RANDOM_STATE)
            idx_umap = rng_umap.choice(len(embeddings), n_umap, replace=False)
            emb_sub  = embeddings[idx_umap]
            lab_sub  = y_test[idx_umap]
            meta_sub = test_meta.iloc[idx_umap].reset_index(drop=True)
        else:
            emb_sub, lab_sub, meta_sub = embeddings, y_test, test_meta

        print(f"\n  Computing UMAP of penultimate layer embeddings "
              f"({len(emb_sub):,} points)…")
        reducer     = umap.UMAP(n_neighbors=15, min_dist=0.1,
                                random_state=cfg.RANDOM_STATE, verbose=False)
        emb2d       = reducer.fit_transform(emb_sub)
        gene_labels = label_encoder.inverse_transform(lab_sub)

        # Static PNG — gene-colours with shades per guideRNA variant
        color_map    = label_color_map(gene_labels.tolist(), cfg.GENE_COLORS)
        unique_genes = sorted(set(gene_labels))
        fig, ax = plt.subplots(figsize=(12, 9))
        for gene in unique_genes:
            mask = gene_labels == gene
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                       c=color_map[gene], label=gene,
                       s=8, alpha=0.6, linewidths=0)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=6, ncol=max(1, len(unique_genes) // 35))
        ax.set_title('UMAP — Penultimate Layer Embeddings (test set)')
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
        plt.tight_layout()
        fig.savefig(output_dir / 'embeddings_umap.png', dpi=cfg.DPI,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved embeddings_umap.png")

        # Interactive Bokeh HTML
        _bokeh_umap(emb2d, gene_labels, meta_sub, output_dir, cfg)

    except ImportError:
        print("  umap-learn not found — skipping UMAP.  pip install umap-learn")


# ============================================================================
# MAIN
# ============================================================================
def main():
    # ── Ada 4500 optimisations ─────────────────────────────────────────────
    torch.set_float32_matmul_precision('high')   # TF32 on Ada tensor cores
    torch.backends.cudnn.benchmark = True

    cfg = Config()
    torch.manual_seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.ROOT_DATA_DIR) / f"DL_Classification_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # ── Load raw cell-level data ───────────────────────────────────────────
    df = load_all_plates(cfg)

    # ── Cell-level splits (no FOV aggregation) ────────────────────────────
    print("\n" + "=" * 70)
    print("PREPROCESSING — CELL-LEVEL SPLITS")
    print("=" * 70)
    (X_train, y_train, X_val, y_val, X_test, y_test,
     le, scalers, feat_cols, test_meta) = prepare_splits_cell_level(df, cfg)
    del df; gc.collect()

    # ── Save label encoder & scaler ────────────────────────────────────────
    with open(output_dir / 'label_encoder.json', 'w') as f:
        json.dump({int(i): str(c) for i, c in enumerate(le.classes_)}, f, indent=2)

    rs, qt = scalers
    np.savez(output_dir / 'scaler_params.npz',
             rs_center=rs.center_, rs_scale=rs.scale_,
             qt_references_=qt.references_,
             qt_quantiles_=qt.quantiles_,
             feature_names=np.array(feat_cols))
    print(f"  Saved label_encoder.json and scaler_params.npz")

    # ── Build model ────────────────────────────────────────────────────────
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes   = len(le.classes_)
    in_features = X_train.shape[1]   # 8 (raw morphology features)

    print(f"\n  Device:  {device}")
    print(f"  Model:   ResidualMLP  in={in_features}  "
          f"hidden={cfg.HIDDEN_DIM}  blocks={cfg.N_BLOCKS}  classes={n_classes}")

    model = ResidualMLP(
        in_features=in_features,
        hidden_dim=cfg.HIDDEN_DIM,
        n_blocks=cfg.N_BLOCKS,
        n_classes=n_classes,
        dropout=cfg.DROPOUT,
    ).to(device)

    # torch.compile requires Triton, which is Linux-only — disabled on Windows
    print(f"  torch.compile: disabled (Triton not available on Windows)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── AdamW optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )

    # ── Log-inverse-frequency class weights ────────────────────────────────
    cw = compute_class_weights(y_train, n_classes)
    print(f"  Class weight range: {cw[cw > 0].min():.3f} – {cw.max():.3f}")

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n  Pinning train/val tensors to GPU…")
    train_loader, val_loader = build_loaders(X_train, y_train, X_val, y_val,
                                              n_classes, cfg, device)
    history = train_model(model, train_loader, val_loader, optimizer, cw,
                          n_classes, cfg, output_dir)

    # ── Evaluate on test set ───────────────────────────────────────────────
    model.load_state_dict(
        torch.load(output_dir / 'best_model.pt', map_location=device,
                   weights_only=True))
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    evaluate_and_save(model, X_test, y_test, le, test_meta,
                      history, output_dir, cfg)

    print(f"\nDone.  All outputs saved to:\n  {output_dir}")


if __name__ == '__main__':
    main()