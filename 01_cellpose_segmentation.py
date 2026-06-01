import os
import re
import io
import contextlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage as ndi
from scipy.stats import skew as _skew, kurtosis as _kurtosis
from skimage.filters import sobel_h, sobel_v, laplace
from tqdm import tqdm

from cellpose import models
from cp_measure.bulk import get_core_measurements

warnings.filterwarnings('ignore')

# ── Folders to process sequentially ──────────────────────────────────────────
input_folders = [
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P1",
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P2",
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P3",
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P4",
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P5",
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P6",
]

# ── Environment & threading ───────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"]         = "8"
os.environ["MKL_NUM_THREADS"]         = "8"

# ── Physical calibration ──────────────────────────────────────────────────────
PIXEL_SIZE_UM = np.float32(0.108)

# ── Segmentation parameters ───────────────────────────────────────────────────
DIAMETER     = None
BATCH_SIZE   = 128
TILE_OVERLAP = 0.1
BSIZE        = 256

# ── cp_measure prefixes (sizeshape/feret/texture are unprefixed in raw output) ─
_PREFIXES = {
    "sizeshape": "AreaShape_",
    "feret":     "Feret_",
    "texture":   "Texture_",
}


def parse_filename(filename):
    """Extract Well (e.g. A01) and Point (e.g. Z0003) from filename."""
    well_match = re.search(r'Well([A-H]\d{2})', filename)
    seq_match  = re.search(r'_Seq(\d+)_', filename)
    if not well_match or not seq_match:
        return None, None
    return well_match.group(1), f"Z{seq_match.group(1)}"


def load_dic(stack_path):
    """Read TIFF and return the DIC channel as a 2-D array."""
    stack = tifffile.imread(stack_path)
    if stack.ndim == 2:
        return stack
    if stack.ndim == 3:
        return stack[0]
    raise ValueError(f"Unexpected image dimensions {stack.shape} in {stack_path}")


def segment(model, dic_img):
    """Run Cellpose-SAM. Returns 2-D label mask."""
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf), \
         torch.inference_mode():
        masks_padded, _, _ = model.eval(
            dic_img,
            diameter=DIAMETER,
            batch_size=BATCH_SIZE,
            augment=False,
            normalize=True,
            tile_overlap=TILE_OVERLAP,
            bsize=BSIZE,
        )
    masks = np.squeeze(masks_padded)
    if masks.ndim != 2 or masks.shape != dic_img.shape:
        raise ValueError(f"Unexpected masks shape {masks.shape} for input {dic_img.shape}")
    return masks


def _normalize01(img):
    """Scale a 2D image into [0, 1] float32 — required for cp_measure texture."""
    img = img.astype(np.float32, copy=False)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn:
        return np.zeros_like(img)
    return ((img - mn) / (mx - mn)).astype(np.float32)


def _handcrafted_features(mask_kept, labels, dic_img, grad_mag, lap_img, bg_mean, bg_std):
    """Per-object DIC + mask features from pre-built full-image arrays."""
    n = len(labels)
    out = {
        "DIC_Contrast":      np.zeros(n, dtype=np.float32),
        "DIC_Skewness":      np.zeros(n, dtype=np.float32),
        "DIC_Kurtosis":      np.zeros(n, dtype=np.float32),
        "DIC_Gradient_Mean": np.zeros(n, dtype=np.float32),
        "DIC_Gradient_Std":  np.zeros(n, dtype=np.float32),
        "DIC_Gradient_p95":  np.zeros(n, dtype=np.float32),
        "DIC_Laplacian_Var": np.zeros(n, dtype=np.float32),
        "DIC_BgRel_Mean":    np.zeros(n, dtype=np.float32),
        "DIC_BgRel_Std":     np.zeros(n, dtype=np.float32),
        "Frame_Distance":    np.zeros(n, dtype=np.float32),
    }
    H, W = mask_kept.shape
    slices = ndi.find_objects(mask_kept)
    inv_bg_std = 1.0 / bg_std if bg_std > 0 else 0.0
    for i, lab in enumerate(labels):
        sl = slices[lab - 1] if lab - 1 < len(slices) else None
        if sl is None:
            continue
        sub_mask = (mask_kept[sl] == lab)
        if not sub_mask.any():
            continue
        sub_img  = dic_img[sl][sub_mask]
        sub_grad = grad_mag[sl][sub_mask]
        sub_lap  = lap_img[sl][sub_mask]
        p1, p99 = np.percentile(sub_img, (1, 99))
        out["DIC_Contrast"][i]      = float(p99 - p1)
        if sub_img.size > 2:
            out["DIC_Skewness"][i]  = float(_skew(sub_img, bias=False))
        if sub_img.size > 3:
            out["DIC_Kurtosis"][i]  = float(_kurtosis(sub_img, bias=False))
        out["DIC_Gradient_Mean"][i] = float(sub_grad.mean())
        out["DIC_Gradient_Std"][i]  = float(sub_grad.std())
        out["DIC_Gradient_p95"][i]  = float(np.percentile(sub_grad, 95))
        out["DIC_Laplacian_Var"][i] = float(sub_lap.var())
        mean_in = float(sub_img.mean())
        std_in  = float(sub_img.std())
        out["DIC_BgRel_Mean"][i]    = (mean_in - bg_mean) * inv_bg_std
        out["DIC_BgRel_Std"][i]     = std_in * inv_bg_std
        ys, xs = np.nonzero(sub_mask)
        cy = float(ys.mean()) + sl[0].start
        cx = float(xs.mean()) + sl[1].start
        out["Frame_Distance"][i] = float(min(cy, cx, H - 1 - cy, W - 1 - cx))
    return out


def measure(funcs, masks_full, dic_img):
    """Edge-touch exclusion + cp_measure (8 funcs) + handcrafted DIC features.
    Returns a per-object DataFrame (one row per kept cell)."""
    edge_labels = set(
        np.unique(np.concatenate([
            masks_full[0, :], masks_full[-1, :], masks_full[:, 0], masks_full[:, -1]
        ]))
    ) - {0}
    all_labels = set(np.unique(masks_full)) - {0}
    keep_labels = sorted(all_labels - edge_labels)
    if not keep_labels:
        return pd.DataFrame()

    # Remap kept labels to contiguous 1..N (cp_measure assumes no gaps).
    remap = np.zeros(int(masks_full.max()) + 1, dtype=np.int32)
    for new_lab, old_lab in enumerate(keep_labels, start=1):
        remap[old_lab] = new_lab
    mask_kept = remap[masks_full]
    labels = np.arange(1, len(keep_labels) + 1, dtype=np.int32)

    dic = dic_img.astype(np.float32, copy=False)
    dic_norm = _normalize01(dic)
    grad_mag = np.hypot(sobel_h(dic), sobel_v(dic)).astype(np.float32)
    lap_img  = laplace(dic).astype(np.float32)

    bg_pixels = dic[masks_full == 0]
    bg_mean = float(bg_pixels.mean()) if bg_pixels.size else 0.0
    bg_std  = float(bg_pixels.std())  if bg_pixels.size else 0.0

    mask_only_keys = ("sizeshape", "zernike", "feret")
    mask_img_keys  = ("intensity", "radial_distribution", "radial_zernikes",
                      "granularity", "texture")

    parts = []
    for key in mask_only_keys:
        d = funcs[key](mask_kept, None)
        df = pd.DataFrame(d)
        pref = _PREFIXES.get(key)
        if pref:
            df = df.add_prefix(pref)
        parts.append(df)
    for key in mask_img_keys:
        img_in = dic_norm if key == "texture" else dic
        d = funcs[key](mask_kept, img_in)
        df = pd.DataFrame(d)
        pref = _PREFIXES.get(key)
        if pref:
            df = df.add_prefix(pref)
        parts.append(df)

    df = pd.concat(parts, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df.insert(0, "Label", labels)

    hc = _handcrafted_features(mask_kept, labels, dic, grad_mag, lap_img, bg_mean, bg_std)
    for k, v in hc.items():
        df[k] = v

    # Derived shape ratios from cp_measure AreaShape_ columns.
    area_px        = df["AreaShape_Area"].to_numpy(np.float32)
    perim_px       = df["AreaShape_Perimeter"].to_numpy(np.float32)
    major_px       = df["AreaShape_MajorAxisLength"].to_numpy(np.float32)
    minor_px       = df["AreaShape_MinorAxisLength"].to_numpy(np.float32)
    bbox_area_px   = df["AreaShape_BoundingBoxArea"].to_numpy(np.float32)
    convex_area_px = df["AreaShape_ConvexArea"].to_numpy(np.float32)

    safe_perim     = np.where(perim_px       > 0, perim_px,       1e-9)
    safe_minor     = np.where(minor_px       > 0, minor_px,       1e-9)
    safe_bbox      = np.where(bbox_area_px   > 0, bbox_area_px,   1e-9)
    safe_convex    = np.where(convex_area_px > 0, convex_area_px, 1e-9)
    safe_area_sqrt = np.where(area_px        > 0, np.sqrt(area_px), 1e-9)

    df["Roundness"]           = (4 * np.pi * area_px / (safe_perim ** 2)).astype(np.float32)
    df["Aspect_Ratio"]        = (major_px / safe_minor).astype(np.float32)
    df["Bbox_Area_Ratio"]     = (area_px / safe_bbox).astype(np.float32)
    df["Convex_Defect_Ratio"] = ((convex_area_px - area_px) / safe_convex).astype(np.float32)
    df["Perim_to_Sqrt_Area"]  = (perim_px / safe_area_sqrt).astype(np.float32)

    # ── Pixel → physical unit conversion ─────────────────────────────────────
    _px  = float(PIXEL_SIZE_UM)
    _px2 = _px ** 2
    df["AreaShape_Area_um2"]              = (area_px  * _px2).astype(np.float32)
    df["AreaShape_Perimeter_um"]          = (perim_px * _px ).astype(np.float32)
    df["AreaShape_MajorAxisLength_um"]    = (major_px * _px ).astype(np.float32)
    df["AreaShape_MinorAxisLength_um"]    = (minor_px * _px ).astype(np.float32)
    df["AreaShape_EquivalentDiameter_um"] = (df["AreaShape_EquivalentDiameter"].to_numpy(np.float32) * _px).astype(np.float32)
    df["Feret_MinFeretDiameter_um"]       = (df["Feret_MinFeretDiameter"].to_numpy(np.float32) * _px).astype(np.float32)
    df["Feret_MaxFeretDiameter_um"]       = (df["Feret_MaxFeretDiameter"].to_numpy(np.float32) * _px).astype(np.float32)

    df["Background_Mean"] = np.float32(bg_mean)
    df["Background_Std"]  = np.float32(bg_std)
    return df


def write_outputs(df, masks, fname, output_folder):
    stem = os.path.splitext(fname)[0]
    df.to_csv(os.path.join(output_folder, f"{stem}_results.csv"), index=False)
    tifffile.imwrite(os.path.join(output_folder, f"{stem}_masks.tif"), masks.astype(np.uint16))


def aggregate_for_parquet(df, well, z_slice, fname):
    if df.empty:
        return None
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    out = df[numeric_cols].copy()
    out["Well"]     = well
    out["Point"]    = z_slice
    out["filename"] = fname
    out[numeric_cols] = out[numeric_cols].astype("float32")
    out["Well"]     = out["Well"].astype("category")
    out["Point"]    = out["Point"].astype("category")
    out["filename"] = out["filename"].astype("category")
    return out


def main():
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    print(f"PyTorch using {torch.get_num_threads()} threads")

    model = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model='cpsam')
    funcs = get_core_measurements()

    for input_folder in input_folders:
        output_folder = os.path.join(input_folder, "CellposeSAM Segmentation results")
        os.makedirs(output_folder, exist_ok=True)

        tiff_files = sorted(f for f in os.listdir(input_folder)
                            if f.lower().endswith((".tif", ".tiff")))
        print(f"\n{'='*70}\nFolder : {input_folder}\nFound  : {len(tiff_files)} TIFF files")

        aggregated_frames = []

        for fname in tqdm(tiff_files, desc="Processing", dynamic_ncols=True):
            fpath = os.path.join(input_folder, fname)
            try:
                dic_img = load_dic(fpath)
                masks   = segment(model, dic_img)
                df      = measure(funcs, masks, dic_img)
                if not df.empty:
                    write_outputs(df, masks, fname, output_folder)
                    well, z_slice = parse_filename(fname)
                    if well is None:
                        print(f"  [warn] Could not parse Well/Point from {fname!r}")
                    else:
                        agg = aggregate_for_parquet(df, well, z_slice, fname)
                        if agg is not None:
                            aggregated_frames.append(agg)
            except Exception as e:
                print(f"Error processing {fname}: {e}")
            finally:
                torch.cuda.empty_cache()

        if aggregated_frames:
            aggregated_df = pd.concat(aggregated_frames, ignore_index=True)
            ts = datetime.now().strftime("%Y%m%d,%H%M%S")
            output_parquet = os.path.join(output_folder, f"{ts}_cell_measurements.parquet")
            aggregated_df.to_parquet(output_parquet, index=False, engine="pyarrow")
            file_size = os.path.getsize(output_parquet) / 1024**2
            print(f"Parquet saved → {output_parquet} ({file_size:.1f} MB, "
                  f"{len(aggregated_df):,} cells, {aggregated_df.shape[1]} cols)")
            aggregated_df.head(10000).to_csv(
                output_parquet.replace(".parquet", "_sample_10k.csv"), index=False)
        else:
            print("No data aggregated; Parquet not written.")

    print("\nAll folders processed.")


if __name__ == "__main__":
    main()
