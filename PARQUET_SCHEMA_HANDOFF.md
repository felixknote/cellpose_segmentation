# Parquet schema handoff — `01_cellpose_segmentation.py` → downstream

> ## ⚠ STATUS: PROPOSED — NOT IMPLEMENTED
>
> **This document describes a `cp_measure`-based schema that the pipeline does
> not currently produce.** It is a design proposal for a future migration, kept
> for reference. Do not treat it as a description of the current data.
>
> As of 2026-08-19:
>
> - `01_cellpose_segmentation.py` uses `skimage.measure.regionprops_table`,
>   **not** `cp_measure`. It emits ~16 columns, not the ~296 described below.
> - `cp_measure` is **not installed** in any conda environment on this machine.
> - The output file is `cell_measurements.parquet` — no timestamp prefix.
> - Column names are lowercase (`area_um2`, `length_um`, `roundness`), **not**
>   the CellProfiler-style `AreaShape_*` / `Texture_*` / `Zernike_*` names.
>
> For the schema that steps 02–05 actually consume, see the *Step 01* section
> of [README.md](README.md). The 8 morphology features there are already
> consistent across every stage — the "Migration mapping" table below maps
> *from* that working schema *to* the proposed one, so it reads backwards
> relative to the current code.
>
> Nothing needs to change for the pipeline to run. Adopting this schema would
> be a deliberate enhancement: richer texture/Zernike/granularity features at
> the cost of installing `cp_measure` and updating the feature lists in
> steps 02–05.

---

Under the proposed design, the segmentation script writes a `cp_measure`-derived feature parquet per `TIFocus` folder. Files that load this parquet (e.g. `02_morphological_analysis.py`, `02_morphological_analysis_multi_plate.py`, `02_morphological_analysis_WT_multi_plate.py`, `03_morphological_map.py`, `04_od600_morphology_correlation.py`, `05_deep_learning_classification.py`) must be updated to use the new column names. This document is the contract.

## File location & naming

- One parquet per `<input_folder>/CellposeSAM Segmentation results/`.
- Filename: `YYYYMMDD,HHMMSS_cell_measurements.parquet` (was: `micromorph_cell_measurements.parquet`). The comma is intentional, set by the user. Glob with `*_cell_measurements.parquet` to find the latest.
- Sibling: `YYYYMMDD,HHMMSS_cell_measurements_sample_10k.csv` — first 10k rows for quick inspection.

## Row identity

- One row per **kept segmented object** (edge-touching objects excluded as before).
- Index columns (categorical):
  - `Well` — e.g. `A01`
  - `Point` — e.g. `Z0003` (was previously also called `ZSlice`)
  - `filename` — source TIFF filename

## Feature columns

All numeric columns are `float32`. The CellProfiler-style prefix tells you the source. Counts in parentheses are the typical column count; exact count depends on `cp_measure` version (we target 0.1.19).

| Family | Count | Source |
|---|---|---|
| `Label` (int) | 1 | per-object label id within the image (not globally unique) |
| `AreaShape_*` | ~83 | `cp_measure` `sizeshape` — Area, Perimeter, MajorAxisLength, MinorAxisLength, Eccentricity, Solidity, Compactness, FormFactor, Extent, EulerNumber, MaximumRadius, MedianRadius, MeanRadius, FilledArea, SpatialMoment_*, CentralMoment_*, NormalizedMoment_*, HuMoment_0..6, InertiaTensor_*, InertiaTensorEigenvalues_0..1, PerimeterCrofton, Center_X, Center_Y, BoundingBox*_X/Y, EquivalentDiameter, ConvexArea, BoundingBoxArea |
| `AreaShape_*_um`, `AreaShape_*_um2`, `Feret_*_um` | 7 | Same as above with **pixel→µm conversion** applied (PIXEL_SIZE_UM = 0.108): `AreaShape_Area_um2`, `AreaShape_Perimeter_um`, `AreaShape_MajorAxisLength_um`, `AreaShape_MinorAxisLength_um`, `AreaShape_EquivalentDiameter_um`, `Feret_MinFeretDiameter_um`, `Feret_MaxFeretDiameter_um` |
| `Zernike_n_m` | 30 | `cp_measure` `zernike` — pure shape Zernike moments |
| `Feret_*` | 4 | `cp_measure` `feret` (MinFeretDiameter, MaxFeretDiameter) + their `_um` versions |
| `Intensity_*` | 15 | `cp_measure` `intensity` on **DIC** — IntegratedIntensity, MeanIntensity, StdIntensity, MinIntensity, MaxIntensity, MassDisplacement, LowerQuartileIntensity, MedianIntensity, MADIntensity, UpperQuartileIntensity, plus the `*Edge` variants |
| `Location_*` | 6 | Centroid-of-intensity X/Y/Z and Max-intensity X/Y/Z (from `intensity`). Z is always 0 (2-D). |
| `RadialDistribution_*` | 72 | `cp_measure` `radial_distribution` + `radial_zernikes` on DIC — FracAtD/MeanFrac/RadialCV per ring (4 rings) plus ZernikeMagnitude/ZernikePhase (n,m up to 9) |
| `Granularity_1..16` | 16 | `cp_measure` `granularity` on DIC |
| `Texture_<feature>_3_<angle>_256` | 52 | `cp_measure` `texture` on **[0,1]-normalized** DIC — 13 Haralick features × 4 angles (00, 01, 02, 03), distance=3, gray levels=256 |
| `DIC_*` | 9 | **Handcrafted** per-object DIC features (not from cp_measure): `DIC_Contrast` (p99−p1), `DIC_Skewness`, `DIC_Kurtosis`, `DIC_Gradient_Mean/Std/p95` (Sobel magnitude inside object), `DIC_Laplacian_Var`, `DIC_BgRel_Mean/Std` (object stats expressed as z-score against image background) |
| `Frame_Distance` | 1 | Centroid distance to nearest image edge in pixels |
| `Roundness`, `Aspect_Ratio`, `Bbox_Area_Ratio`, `Convex_Defect_Ratio`, `Perim_to_Sqrt_Area` | 5 | Derived shape ratios |
| `Background_Mean`, `Background_Std` | 2 | Per-image background statistics on DIC (broadcast to every row of that image) |

**Total: ~296 numeric columns + 3 categorical (`Well`, `Point`, `filename`).**

## Migration mapping — old → new

The legacy parquet had 8 feature columns. Direct replacements:

| Legacy column        | New column                          | Notes |
|----------------------|-------------------------------------|-------|
| `area_um2`           | `AreaShape_Area_um2`                | identical formula |
| `perimeter_um`       | `AreaShape_Perimeter_um`            | identical |
| `length_um`          | `AreaShape_MajorAxisLength_um`      | identical |
| `width_um`           | `AreaShape_MinorAxisLength_um`      | identical |
| `roundness`          | `Roundness`                         | same formula `4π·A/P²` |
| `aspect_ratio`       | `Aspect_Ratio`                      | same formula |
| `solidity`           | `AreaShape_Solidity`                | identical |
| `eccentricity`       | `AreaShape_Eccentricity`            | identical |
| `Well`, `Point`, `filename` | unchanged                    | still categorical |

**Action items for each downstream file:**

1. **Drop the `PARQUET_FEATURES` / hard-coded 8-column list** if any. Replace with either:
   - The 8 mapped column names above for parity, or
   - A richer selection — recommended: select all `AreaShape_*_um*`, `Roundness`, `Aspect_Ratio`, `AreaShape_Solidity`, `AreaShape_Eccentricity`, `AreaShape_Compactness`, `AreaShape_FormFactor`, `AreaShape_Extent`, `Feret_*_um`, `DIC_*`, `Intensity_Mean/Std/Median/MAD/UpperQuartile/LowerQuartile`, `Granularity_*`. Skip raw moment columns (`SpatialMoment_*`, `CentralMoment_*`, etc.) and `Texture_*`/`Zernike_*` for PCA/UMAP unless explicitly wanted — they are high-dimensional and partly redundant.

2. **Parquet glob**: replace any hard-coded `micromorph_cell_measurements.parquet` reference with a glob:
   ```python
   import glob, os
   candidates = sorted(glob.glob(os.path.join(folder, "*_cell_measurements.parquet")))
   parquet_path = candidates[-1]  # most recent
   ```

3. **Numeric subset for ML**: use `df.select_dtypes(include="number").columns` and drop `Label`, `Background_Mean`, `Background_Std`, `Frame_Distance` from clustering input (they are per-image or positional, not per-cell biology). Standardize before PCA/UMAP.

4. **Units**: anything ending in `_um` or `_um2` is in micrometres; everything else without that suffix is in pixel/dimensionless units. Pick one or the other per axis — don't mix.

5. **Texture/Zernike scale**: `Texture_*` was computed on per-image min-max-normalized DIC, so magnitudes are comparable across images but the raw intensity scale is lost — use them as shape/texture descriptors, not absolute brightness. `Zernike_*` are pure mask shape descriptors and are scale-invariant.

## Environment

The new pipeline runs in the **`cellposeSAMCP`** conda env (Python 3.11, torch 2.5.1+cu121, cellpose 4.0.6, cp_measure 0.1.19). The original `cellposeSAM` (Python 3.8) env is untouched and still works for the legacy script.

## Reproducibility note

`Texture_*` keys depend on the cp_measure version; if you upgrade past 0.1.19 the gray-level / distance / angle suffix in the column name may change. The other families are stable.
