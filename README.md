# Cellpose Segmentation & Morphology Pipeline

A GPU-accelerated pipeline for segmenting bacterial cells in DIC microscopy
stacks and analysing their morphology across CRISPRi screening plates.

Segmentation runs on [CellposeSAM](https://github.com/MouseLand/cellpose)
(`cellpose==4.0.6`, `cpsam` weights). Downstream steps take the resulting
per-cell measurement tables and produce statistics, PCA/UMAP embeddings,
OD600 correlations, and a deep-learning knockdown classifier.

---

## Pipeline overview

Each script is a stage. They communicate through **parquet files on disk**, not
through imports, so stages can be re-run independently.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 01 | `01_cellpose_segmentation.py` | Folders of 2-channel `.tif`/`.tiff` stacks | Label masks + per-cell parquet |
| 02 | `02_morphological_analysis.py` | One plate's parquet + plate map | Histograms, per-well stats, Excel counts |
| 02b | `02_morphological_analysis_multi_plate.py` | Several plates (`P1`…`P6`) | Cross-plate comparison |
| 02c | `02_morphological_analysis_CRISPRi_control_plate.py` | Control-plate replicates | 2×2×2 factorial analysis |
| 03 | `03_morphological_map.py` | Plate parquet + plate map | PCA + UMAP maps (JPG + interactive HTML) |
| 04 | `04_od600_morphology_correlation.py` | Plate parquets + OD600 tables | OD600 × morphology correlations |
| 05 | `05_deep_learning_classification.py` | Multi-plate parquets | ResidualMLP gene-knockdown classifier |

`multiplate/` holds two exploratory notebooks (histograms, UMAP) kept for
reference; the maintained equivalents are steps 02b and 03.

---

## Installation

The pipeline is developed against a dedicated conda environment
(`cellposeSAM`, Python 3.8):

```bash
conda create -n cellposeSAM python=3.8
conda activate cellposeSAM
pip install -r requirements.txt
```

CellposeSAM is only practical on a CUDA GPU. Install the matching PyTorch build
**before** the rest, otherwise you get the CPU wheel:

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 01 — Segmentation

Edit the `input_folders` list at the top of `01_cellpose_segmentation.py`, then:

```bash
python 01_cellpose_segmentation.py
```

**Expected input.** Each `.tif`/`.tiff` is a stack where

- **channel 0** = DIC — used for segmentation and all intensity features
- **channel 1** = fluorescence — optional; if absent, intensity/background
  columns are simply omitted

**Key parameters** (module-level constants):

| Constant | Default | Meaning |
|----------|---------|---------|
| `PIXEL_SIZE_UM` | `0.108` | µm per pixel — sets the physical calibration |
| `DIAMETER` | `None` | Cell diameter in px; `None` = auto-estimate |
| `BATCH_SIZE_INIT` | `32` | Tiles pushed through CPSAM at once |
| `BSIZE_INIT` | `256` | Tile edge length (px) |
| `TILE_OVERLAP` | `0.1` | Tile overlap fraction |
| `OOM_RETRIES` | `4` | OOM retry attempts |
| `RESUME` | `True` | Skip images whose mask already exists |

**Outputs**, written to `<input_folder>/CellposeSAM Segmentation results/`:

```
masks/       one uint16 (or uint32) label TIFF per input image
shards/      one parquet per input image
cell_measurements.parquet    combined table, rebuilt from shards
```

**Measured features** (via `skimage.measure.regionprops_table`), one row per
kept cell. Cells touching the image border are excluded, since their morphology
is truncated. Dimensionless ratios are computed *before* calibration; lengths
and areas are then converted to µm and µm².

`area` (µm²), `perimeter` (µm), `major_axis_length` (µm),
`minor_axis_length` (µm), `roundness` (4π·area / perimeter²),
`aspect_ratio` (major/minor), plus `mean_intensity`, `background_mean` and
`background_std` when a fluorescence channel is present.

### Robustness

The script is built to survive long unattended batches:

- **Resumable** — a per-image mask TIFF doubles as a completion marker, so a
  crash (CUDA OOM, host OOM, power loss, Ctrl-C) costs only the in-flight image.
- **Atomic combine** — the per-folder parquet is rebuilt from shards via
  temp-file-then-`os.replace`, so it is never left half-written.
- **OOM backoff** — on CUDA OOM the batch size and tile size are halved and the
  image is retried, up to `OOM_RETRIES`.
- **Per-image isolation** — one bad stack is reported and skipped rather than
  killing the batch.

---

## Steps 02–05 — Analysis

All scripts have a `__main__` guard and run standalone. Only step 02 can take
its input path on the command line; every other script is configured by editing
the constants at the top of the file.

| Script | Available flags |
|--------|-----------------|
| `02_morphological_analysis.py` | `--data-folder`, `--dpi` |
| `02_morphological_analysis_multi_plate.py` | `--dpi` |
| `02_morphological_analysis_CRISPRi_control_plate.py` | `--dpi`, `--skip-preflight` |
| `01`, `03`, `04`, `05` | none — edit the constants |

```bash
python 02_morphological_analysis.py --data-folder "D:\...\P1\CellposeSAM Segmentation results"
python 02_morphological_analysis_multi_plate.py --dpi 300
python 03_morphological_map.py
```

Multi-plate steps expect one plate per subfolder plus a plate map per plate,
rooted at the `ROOT_DATA_DIR` constant in the script:

```
ROOT_DATA_DIR/
  P1/<SEGMENTATION_SUBPATH>/<parquet>
  P2/<SEGMENTATION_SUBPATH>/<parquet>
  ...
  P_1_plate_map.csv
  P_2_plate_map.csv
```

Results are written to timestamped `Analysis_<timestamp>/` folders next to the
data. Interactive HTML plots in step 03 need `bokeh`; without it the script
prints a warning and writes only the static JPGs.

---

## Data contract between stages

Step 01 writes `cell_measurements.parquet`; every downstream script reads that
same name via its `AGGREGATED_FILE` constant. The stages therefore connect
without any manual renaming.

The 8 morphology features steps 02–05 analyse are all produced by step 01:

`area_um2`, `perimeter_um`, `length_um`, `width_um`, `roundness`,
`aspect_ratio`, `solidity`, `eccentricity`

If you change the output filename in step 01, update `AGGREGATED_FILE` in
`02_morphological_analysis.py`, `02_morphological_analysis_CRISPRi_control_plate.py`,
`03_morphological_map.py`, `04_od600_morphology_correlation.py` and
`05_deep_learning_classification.py`. (`02_morphological_analysis_multi_plate.py`
inherits its config from the single-plate module.)

> `PARQUET_SCHEMA_HANDOFF.md` describes a **proposed** `cp_measure` schema with
> ~296 columns that this pipeline does not produce. It is a design document for
> a possible future enhancement, not a description of current output — see the
> status banner at the top of that file.

---

## ⚠ Known limitation

**Data paths are hardcoded.** The `D:\...` paths at the top of each script
point at specific acquisition folders and must be edited for any other dataset.
Only `02_morphological_analysis.py` can take its input path on the command line
(`--data-folder`).

---

## Repository layout

```
01_cellpose_segmentation.py                        segmentation + measurement
02_morphological_analysis.py                       single-plate morphology
02_morphological_analysis_multi_plate.py           multi-plate morphology
02_morphological_analysis_CRISPRi_control_plate.py control-plate factorial
03_morphological_map.py                            PCA + UMAP
04_od600_morphology_correlation.py                 OD600 × morphology
05_deep_learning_classification.py                 ResidualMLP classifier
multiplate/                                        exploratory notebooks
PARQUET_SCHEMA_HANDOFF.md                          target parquet schema
requirements.txt
```

Segmentation outputs (`images/`, `Analysis_*/`, `*.parquet`, mask TIFFs) and
`pyvis`'s vendored `lib/` assets are regenerated from raw data and are excluded
via `.gitignore`.

---

## License

MIT — see [LICENSE](LICENSE).
