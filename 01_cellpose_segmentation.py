import os
import re
import gc
import io
import sys
from datetime import datetime
import contextlib
import warnings
import torch
import numpy as np
import tifffile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from cellpose import models
from skimage.measure import regionprops_table
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Folders to process ────────────────────────────────────────────────────────
input_folders = [
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P1",
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P2",
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P3",
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P4",
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P5",
    r"D:\2025_12_19 CRISPRi Reference Plate Imaging\P6",
    r"D:\2026_04_28_Antibiotics Reference Set\P1",
    r"D:\2026_04_28_Antibiotics Reference Set\P2",
    r"D:\2026_04_28_Antibiotics Reference Set\P3",
    r"D:\2026_04_28_Antibiotics Reference Set\P4",
    r"D:\2026_04_28_Antibiotics Reference Set\P5",
    r"D:\2026_04_28_Antibiotics Reference Set\P6",
]

# ── Environment & threading ───────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"]         = "36"
os.environ["MKL_NUM_THREADS"]         = "36"
torch.set_num_threads(36)
print(f"PyTorch using {torch.get_num_threads()} threads")
torch.backends.cudnn.benchmark = True

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model='cpsam')

# ── Physical calibration ──────────────────────────────────────────────────────
PIXEL_SIZE_UM = np.float32(0.108)  # µm per pixel

# ── Segmentation parameters ───────────────────────────────────────────────────
diameter        = None
BATCH_SIZE_INIT = 128
BSIZE_INIT      = 256
tile_overlap    = 0.1
batch_size      = BATCH_SIZE_INIT
bsize           = BSIZE_INIT


def parse_filename(filename):
    well_match = re.search(r'Well([A-H]\d{2})', filename)
    seq_match  = re.search(r'_Seq(\d+)_', filename)
    if not well_match or not seq_match:
        return None, None
    return well_match.group(1), f"Z{seq_match.group(1)}"


_RP_RENAME = {
    "label":             "Label",
    "area":              "Area",
    "major_axis_length": "Length",
    "minor_axis_length": "Width",
    "solidity":          "Solidity",
    "eccentricity":      "Eccentricity",
    "centroid-0":        "Centroid Y",
    "centroid-1":        "Centroid X",
    "mean_intensity":    "Mean Intensity",
}
_RP_BASE = ["label", "area", "perimeter", "major_axis_length",
            "minor_axis_length", "solidity", "eccentricity", "centroid"]


def _load_and_prepare(stack_path):
    stack = tifffile.imread(stack_path)
    if stack.ndim == 3:
        if stack.shape[0] == 2:
            return stack[0], stack[1], True
        elif stack.shape[0] == 1:
            return stack[0], None, False
        else:
            raise ValueError(f"Unexpected channel count {stack.shape[0]} in {stack_path}")
    elif stack.ndim == 2:
        return stack, None, False
    else:
        raise ValueError(f"Unexpected dimensions {stack.shape} in {stack_path}")


def _infer_and_measure(dic_img, fluo_img, fluo_available):
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
        masks_padded, flows, styles = model.eval(
            dic_img,
            diameter=diameter,
            batch_size=batch_size,
            augment=False,
            normalize=True,
            tile_overlap=tile_overlap,
            bsize=bsize,
        )
    del flows, styles

    masks = np.squeeze(masks_padded)
    if masks.ndim != 2 or masks.shape != dic_img.shape:
        raise ValueError(f"Mask shape {masks.shape} != image shape {dic_img.shape}")

    edge_labels = set(np.unique(np.concatenate([
        masks[0, :], masks[-1, :], masks[:, 0], masks[:, -1]
    ]))) - {0}

    intensity_img = fluo_img if fluo_available else dic_img
    _props = _RP_BASE + (["mean_intensity"] if fluo_available else [])
    tbl = regionprops_table(masks, intensity_image=intensity_img, properties=_props)
    df = pd.DataFrame(tbl)

    if edge_labels:
        df = df[~df["label"].isin(edge_labels)].reset_index(drop=True)

    # dimensionless shape features (pixel units cancel)
    perc = df["perimeter"].clip(lower=1e-9)
    df["Roundness"]    = (4 * np.pi * df["area"] / perc ** 2).astype(np.float32)
    df["Aspect Ratio"] = np.where(
        df["minor_axis_length"] != 0,
        df["major_axis_length"] / df["minor_axis_length"], 0.0,
    ).astype(np.float32)

    _px2 = PIXEL_SIZE_UM ** 2
    df["perimeter_um"]      = (df["perimeter"]         * PIXEL_SIZE_UM).astype(np.float32)
    df["area"]              = (df["area"]              * _px2         ).astype(np.float32)
    df["major_axis_length"] = (df["major_axis_length"] * PIXEL_SIZE_UM).astype(np.float32)
    df["minor_axis_length"] = (df["minor_axis_length"] * PIXEL_SIZE_UM).astype(np.float32)

    bg = intensity_img[masks == 0]
    df["Background Mean"] = float(np.mean(bg)) if bg.size > 0 else 0.0
    df["Background Std"]  = float(np.std(bg))  if bg.size > 0 else 0.0

    df = df.drop(columns=["perimeter"]).rename(columns=_RP_RENAME)
    return df, masks.astype(np.uint16)


def get_free_gpu_memory():
    torch.cuda.empty_cache()
    return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()


def adjust_parameters_on_oom(current_batch, current_bsize):
    free_mem = get_free_gpu_memory()
    print(f"  Free GPU: {free_mem / (1024**3):.1f} GB")
    if free_mem < 5 * (1024**3):
        new_batch = max(64, current_batch // 2)
        new_bsize = max(128, current_bsize // 2)
        print(f"  Reducing batch_size {current_batch}→{new_batch}, bsize {current_bsize}→{new_bsize}")
        return new_batch, new_bsize
    return current_batch, current_bsize


def analyze_single_stack_with_retries(stack_path, retries=3):
    global batch_size, bsize
    prepared = _load_and_prepare(stack_path)
    for attempt in range(retries):
        try:
            return _infer_and_measure(*prepared)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"  CUDA OOM (attempt {attempt + 1}) — {os.path.basename(stack_path)}")
                batch_size, bsize = adjust_parameters_on_oom(batch_size, bsize)
                torch.cuda.empty_cache()
            else:
                raise
    raise RuntimeError(f"OOM after {retries} attempts: {stack_path}")


PARQUET_RENAME = {
    "Area":         "area_um2",
    "Length":       "length_um",
    "Width":        "width_um",
    "Roundness":    "roundness",
    "Aspect Ratio": "aspect_ratio",
    "Solidity":     "solidity",
    "Eccentricity": "eccentricity",
    "ZSlice":       "Point",
}
PARQUET_FEATURES = [
    "area_um2", "perimeter_um", "length_um", "width_um",
    "roundness", "aspect_ratio", "solidity", "eccentricity",
]


def _write_outputs(df, masks, fname, output_folder):
    stem = os.path.splitext(fname)[0]
    df.to_csv(os.path.join(output_folder, f"{stem}_results.csv"), index=False)
    tifffile.imwrite(os.path.join(output_folder, f"{stem}_masks.tif"), masks)


# ── Main loop ─────────────────────────────────────────────────────────────────
for input_folder in input_folders:
    parquet_writer = None
    try:
        output_folder = os.path.join(input_folder, "CellposeSAM Segmentation results")
        os.makedirs(output_folder, exist_ok=True)

        batch_size = BATCH_SIZE_INIT
        bsize      = BSIZE_INIT

        tiff_files = sorted(
            f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))
        )
        print(f"\n{'='*70}")
        print(f"Folder : {input_folder}")
        print(f"Found  : {len(tiff_files)} TIFF files", flush=True)

        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_parquet = os.path.join(output_folder, f"{timestamp}_micromorph_cell_measurements.parquet")
        total_cells    = 0

        with tqdm(total=len(tiff_files), desc="Processing", dynamic_ncols=True) as pbar:
            for fname in tiff_files:
                fpath = os.path.join(input_folder, fname)
                try:
                    df, masks = analyze_single_stack_with_retries(fpath)

                    if not df.empty:
                        _write_outputs(df, masks, fname, output_folder)

                    well, z_slice = parse_filename(fname)
                    if not df.empty and well is not None:
                        df_agg = df[["Area", "perimeter_um", "Roundness", "Width", "Length",
                                     "Aspect Ratio", "Solidity", "Eccentricity"]].copy()
                        df_agg["Well"]     = well
                        df_agg["ZSlice"]   = z_slice
                        df_agg["filename"] = fname
                        df_agg = df_agg.rename(columns=PARQUET_RENAME)
                        df_agg[PARQUET_FEATURES] = df_agg[PARQUET_FEATURES].astype("float32")
                        df_agg["Well"]     = df_agg["Well"].astype(str)
                        df_agg["Point"]    = df_agg["Point"].astype(str)
                        df_agg["filename"] = df_agg["filename"].astype(str)
                        table = pa.Table.from_pandas(df_agg, preserve_index=False)
                        if parquet_writer is None:
                            parquet_writer = pq.ParquetWriter(output_parquet, table.schema)
                        parquet_writer.write_table(table)
                        total_cells += len(df_agg)
                        del df_agg, table
                    elif well is None and not df.empty:
                        print(f"  [warn] Cannot parse Well/Point from {fname!r} — skipped")

                    del df, masks

                except Exception as e:
                    print(f"  [skip] {fname}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    pbar.update(1)

        if total_cells > 0:
            file_size = os.path.getsize(output_parquet) / 1024**2
            print(f"Parquet saved → {output_parquet} ({file_size:.1f} MB, {total_cells:,} cells)",
                  flush=True)
        else:
            print("No data aggregated; Parquet not written.", flush=True)

    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] at {input_folder}")
        raise
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {input_folder}: {e}")
        traceback.print_exc()
    finally:
        if parquet_writer is not None:
            parquet_writer.close()
        gc.collect()
        torch.cuda.empty_cache()
        sys.stdout.flush()

print("\nAll folders processed.")
