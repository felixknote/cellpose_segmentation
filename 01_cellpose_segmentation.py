"""
01 — CellposeSAM segmentation + morphological measurement (batch, resumable).

Restored from the original Jupyter pipeline (cellpose_segmentation-GPU.ipynb) and
extended with calibrated measurements, edge-cell exclusion, and crash-resilient
batch processing across many folders.

Design notes
------------
* The CPSAM model is loaded ONCE and images are processed sequentially. This is
  the configuration that was stable in the original notebook.
* Robustness: every image writes a per-file mask TIFF and a per-file parquet
  "shard". On rerun, files whose mask already exists are skipped, so a crash
  (CUDA/host OOM, power loss, Ctrl-C) only costs the in-flight image. The
  combined per-folder parquet is rebuilt from the shards at the end of each
  folder, so it is never left half-written.
* The previous hard crash (no Python traceback) was a native out-of-memory kill
  driven by batch_size=128 on the large CPSAM ViT. BATCH_SIZE_INIT is now
  conservative and the OOM retry shrinks it further if needed.
"""

import os
import re
import gc
import io
import sys
import glob
import contextlib
import warnings

import numpy as np
import tifffile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

warnings.filterwarnings("ignore")


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


# ── Configuration ─────────────────────────────────────────────────────────────
PIXEL_SIZE_UM = np.float32(0.108)   # µm per pixel
SAVE_MASKS    = True                # write per-cell label masks (used as resume marker)
RESUME        = True                # skip images whose mask already exists
N_THREADS     = min(os.cpu_count() or 8, 36)

# Segmentation parameters (conservative batch_size avoids native OOM kills).
DIAMETER        = None              # automatic cell-diameter estimation
TILE_OVERLAP    = 0.1
BATCH_SIZE_INIT = 32                # tiles processed simultaneously through CPSAM
BSIZE_INIT      = 256              # tile edge length (px)
OOM_RETRIES     = 4


# ── Environment & threading ───────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"]         = str(N_THREADS)
os.environ["MKL_NUM_THREADS"]         = str(N_THREADS)

import torch  # noqa: E402  (after env vars are set)
from cellpose import models  # noqa: E402
from skimage.measure import regionprops_table  # noqa: E402

torch.set_num_threads(N_THREADS)
torch.backends.cudnn.benchmark = True
print(f"PyTorch using {torch.get_num_threads()} threads")
print(f"CUDA available: {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))


# ── Model ─────────────────────────────────────────────────────────────────────
model = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model="cpsam")

# Mutable copies adjusted on OOM.
batch_size = BATCH_SIZE_INIT
bsize      = BSIZE_INIT


# ── Filename parsing ──────────────────────────────────────────────────────────
def parse_filename(filename):
    """Return (well, point) e.g. ('A01', 'Z0000'); (None, None) if unparseable."""
    well_match = re.search(r"Well([A-H]\d{2})", filename)
    seq_match  = re.search(r"_Seq(\d+)", filename)
    if not well_match or not seq_match:
        return None, None
    return well_match.group(1), f"Z{seq_match.group(1)}"


# ── Measurement schema ────────────────────────────────────────────────────────
_RP_BASE = ["label", "area", "perimeter", "major_axis_length", "minor_axis_length",
            "solidity", "eccentricity", "extent", "euler_number", "centroid"]

_RP_RENAME = {
    "label":             "label",
    "area":              "area_um2",
    "major_axis_length": "length_um",
    "minor_axis_length": "width_um",
    "perimeter":         "perimeter_um",
    "solidity":          "solidity",
    "eccentricity":      "eccentricity",
    "extent":            "extent",
    "euler_number":      "euler_number",
    "centroid-0":        "centroid_y",
    "centroid-1":        "centroid_x",
    "mean_intensity":    "mean_intensity",
}


def process_stack(stack_path):
    """Segment one TIFF and return (DataFrame of per-cell features, label mask)."""
    global batch_size, bsize

    # load & split channels — these reference plates are single-channel DIC,
    # but a 2-channel (DIC, fluorescence) stack is also supported.
    stack = tifffile.imread(stack_path)
    if stack.ndim == 3:
        if stack.shape[0] == 2:
            dic_img, fluo_img, fluo_available = stack[0], stack[1], True
        elif stack.shape[0] == 1:
            dic_img, fluo_img, fluo_available = stack[0], None, False
        else:
            raise ValueError(f"Unexpected channel count {stack.shape[0]} in {stack_path}")
    elif stack.ndim == 2:
        dic_img, fluo_img, fluo_available = stack, None, False
    else:
        raise ValueError(f"Unexpected dimensions {stack.shape} in {stack_path}")

    # segment with OOM retries (shrink batch/tile size and try again)
    masks_padded = None
    for attempt in range(OOM_RETRIES):
        try:
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
                masks_padded, flows, styles = model.eval(
                    dic_img,
                    diameter=DIAMETER,
                    batch_size=batch_size,
                    augment=False,
                    normalize=True,
                    tile_overlap=TILE_OVERLAP,
                    bsize=bsize,
                )
            del flows, styles
            break
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                batch_size = max(8,   batch_size // 2)
                bsize      = max(128, bsize      // 2)
                print(f"  CUDA OOM (attempt {attempt + 1}) on "
                      f"{os.path.basename(stack_path)} — reduced "
                      f"batch_size→{batch_size}, bsize→{bsize}", flush=True)
            else:
                raise
    if masks_padded is None:
        raise RuntimeError(f"OOM after {OOM_RETRIES} attempts: {stack_path}")

    masks = np.squeeze(masks_padded)
    if masks.ndim != 2 or masks.shape != dic_img.shape:
        raise ValueError(f"Mask shape {masks.shape} != image shape {dic_img.shape}")

    # exclude cells touching the image border (incomplete / biased morphology)
    edge_labels = set(np.unique(np.concatenate([
        masks[0, :], masks[-1, :], masks[:, 0], masks[:, -1]
    ]))) - {0}

    intensity_img = fluo_img if fluo_available else None
    props = _RP_BASE + (["mean_intensity"] if fluo_available else [])
    tbl = regionprops_table(masks, intensity_image=intensity_img, properties=props)
    df = pd.DataFrame(tbl)

    if edge_labels:
        df = df[~df["label"].isin(edge_labels)].reset_index(drop=True)

    if df.empty:
        return df, _cast_mask(masks)

    # dimensionless shape features (pixel units cancel — compute before calibration)
    perc = df["perimeter"].clip(lower=1e-9)
    df["roundness"]    = (4 * np.pi * df["area"] / perc ** 2).astype(np.float32)
    df["aspect_ratio"] = np.where(
        df["minor_axis_length"] != 0,
        df["major_axis_length"] / df["minor_axis_length"], 0.0,
    ).astype(np.float32)

    # physical calibration → micrometres
    _px2 = PIXEL_SIZE_UM ** 2
    df["perimeter"]         = (df["perimeter"]         * PIXEL_SIZE_UM).astype(np.float32)
    df["area"]              = (df["area"]              * _px2         ).astype(np.float32)
    df["major_axis_length"] = (df["major_axis_length"] * PIXEL_SIZE_UM).astype(np.float32)
    df["minor_axis_length"] = (df["minor_axis_length"] * PIXEL_SIZE_UM).astype(np.float32)

    if fluo_available:
        bg = fluo_img[masks == 0]
        df["background_mean"] = np.float32(np.mean(bg)) if bg.size else np.float32(0)
        df["background_std"]  = np.float32(np.std(bg))  if bg.size else np.float32(0)

    df = df.rename(columns=_RP_RENAME)
    return df, _cast_mask(masks)


def _cast_mask(masks):
    """Use uint16 unless the label count overflows it."""
    dtype = np.uint16 if masks.max() <= np.iinfo(np.uint16).max else np.uint32
    return masks.astype(dtype)


# ── Per-folder driver ─────────────────────────────────────────────────────────
def process_folder(input_folder):
    output_folder = os.path.join(input_folder, "CellposeSAM Segmentation results")
    masks_dir     = os.path.join(output_folder, "masks")
    shards_dir    = os.path.join(output_folder, "shards")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(shards_dir, exist_ok=True)

    tiff_files = sorted(
        f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))
    )
    print(f"\n{'=' * 70}")
    print(f"Folder : {input_folder}")
    print(f"Found  : {len(tiff_files)} TIFF files", flush=True)

    processed = skipped = failed = 0
    for fname in tqdm(tiff_files, desc="Processing", dynamic_ncols=True):
        stem      = os.path.splitext(fname)[0]
        mask_path = os.path.join(masks_dir, f"{stem}_masks.tif")
        shard_path = os.path.join(shards_dir, f"{stem}.parquet")

        if RESUME and os.path.exists(mask_path):
            skipped += 1
            continue

        try:
            df, masks = process_stack(os.path.join(input_folder, fname))

            if SAVE_MASKS:
                tifffile.imwrite(mask_path, masks, compression="zlib")

            well, point = parse_filename(fname)
            if not df.empty and well is not None:
                df["well"]     = well
                df["point"]    = point
                df["filename"] = fname
                pq.write_table(pa.Table.from_pandas(df, preserve_index=False), shard_path)
            elif not df.empty and well is None:
                print(f"  [warn] cannot parse Well/Seq from {fname!r} — shard skipped", flush=True)

            processed += 1
            del df, masks
        except Exception as e:
            failed += 1
            print(f"  [skip] {fname}: {e}", flush=True)
        finally:
            torch.cuda.empty_cache()

    # rebuild the combined parquet from all shards (atomic via temp-then-replace)
    shard_files = sorted(glob.glob(os.path.join(shards_dir, "*.parquet")))
    if shard_files:
        combined = os.path.join(output_folder, "cell_measurements.parquet")
        tmp      = combined + ".tmp"
        writer   = None
        total    = 0
        try:
            for sf in shard_files:
                table = pq.read_table(sf)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema)
                writer.write_table(table)
                total += table.num_rows
        finally:
            if writer is not None:
                writer.close()
        os.replace(tmp, combined)
        size_mb = os.path.getsize(combined) / 1024**2
        print(f"Parquet → {combined} ({size_mb:.1f} MB, {total:,} cells)", flush=True)
    else:
        print("No cells measured; parquet not written.", flush=True)

    print(f"Done: {processed} processed, {skipped} skipped, {failed} failed.", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for input_folder in input_folders:
        if not os.path.isdir(input_folder):
            print(f"\n[missing] {input_folder} — skipped", flush=True)
            continue
        try:
            process_folder(input_folder)
        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] at {input_folder} — rerun to resume.", flush=True)
            raise
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {input_folder}: {e}", flush=True)
            traceback.print_exc()
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            sys.stdout.flush()

    print("\nAll folders processed.")


if __name__ == "__main__":
    main()
