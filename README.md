# clump-DB

A small “database generator” for **irregular particle STL meshes** and **CLUMP** sphere-pack *clumps*, storing each generated case with a **hash-based case ID** and **JSON metadata**.  
This is designed for later binning/filtering (e.g., selecting clumps by shape/roundness metrics for LAMMPS runs).

---

## What this project does

For each case (one parameter set), the pipeline:

1. Generates an ellipsoidal, tapered, superellipsoidal, or rounded-trapezoidal STL mesh (`shape.stl`)
2. Runs CLUMP (Euclidean_3D extended procedure) to create a sphere-pack clump
3. Saves:
   - sphere list `balls_xyzr.txt` (`x y z r`)
   - `meta.json` containing input parameters + derived metrics
4. Stores everything under a hash-based folder:
   - `dataset/shapes/<case_id>/`

---

## Directory layout

- `clumpgen/`  
  Python package with core utilities (e.g., mesh generation used by scripts)
  - `molecule_mc.py` : module script for make molecule file for lammps
  - `shape.py` : module script for make STL shape file

- `scripts/`  
  CLI scripts:
  - `make_shape.py` : generate a single STL mesh (sanity check / debugging)
  - `run_compact_case_hash.py` : generate a full “case” (STL + clump + metadata)
  - `backfill_shape_parameters.py` : measure and update existing cases
  - `view_case.py` : interactive viewer for a case
  - `rebuild_manifest.py` : rebuild "manifest.jsonl" file based on the current dataset

- `dataset/shapes/<case_id>/`  
  Output database (one folder per case)

---

## Installation (recommended)

Create and activate a virtual environment, then install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Install runtime dependencies (if not already installed via your environment):

```bash
python -m pip install numpy pillow scipy trimesh pyvista
```

Install CLUMP Python wrapper:

```bash
python -m pip install clump-python
```

> CLUMP upstream reference: https://github.com/vsangelidakis/CLUMP

---

## 1) `scripts/make_shape.py`

Generate a single particle STL mesh.
This is useful to validate mesh parameters before running the full clump pipeline.

### Usage

```bash
python scripts/make_shape.py   --out outputs/test.stl   --L 10 --e 0.75 --f 0.65   --sub 4 --randomness 0.18 --bias 1.5 --seed 1234
```

Generate a Hime-like rounded trapezoidal particle:

```bash
python scripts/make_shape.py \
  --out outputs/hime_rounded_trapezoid.stl \
  --shape-family rounded_trapezoid \
  --L 10 --e 0.80 --f 0.75 \
  --taper 0.25 --boxiness 3.0 \
  --asymmetry-I 0.12 --asymmetry-S -0.06 \
  --sub 4 --randomness 0.06 --bias 1.5 --seed 1234
```

### Shape parameters

- `L` : historical longest-axis scale (semi-axis; the unperturbed full extent is `2L`)
- `e` : `I/L` (intermediate-to-long axis ratio)
- `f` : `S/I` (short-to-intermediate axis ratio)
- `shape-family` : `ellipsoid`, `tapered_ellipsoid`, `superellipsoid`, or `rounded_trapezoid`
- `taper` : fractional reduction of the smaller end; the sign selects which end is reduced
- `boxiness` : superellipsoid exponent (`2` is ellipsoidal; larger values give flatter faces and sharper shoulders)
- `asymmetry-I` : signed one-sided bulging across the intermediate-axis direction
- `asymmetry-S` : signed one-sided bulging across the short-axis direction
- `sub` : mesh subdivision level (higher → more vertices/faces)
- `randomness` : inward vertex perturbation magnitude
- `bias` : distribution skew for perturbation (higher → many small changes, fewer large ones)
- `seed` : RNG seed for reproducibility

`taper` is active for `tapered_ellipsoid` and `rounded_trapezoid`.
`boxiness` is active for `superellipsoid` and `rounded_trapezoid`. Controls
that do not apply to the selected family are normalized before the shape ID is
calculated, so they do not create duplicate geometries with different IDs.
The transverse asymmetry controls apply to every family. Zero preserves mirror
symmetry; the sign selects the enlarged side, and the magnitude controls the
coherent departure from a centered cross-section. Values must remain between
`-0.5` and `0.5`.

The generated metadata includes source-mesh descriptors under
`metrics.source_shape`:

- `ellipsoid_reference_rms` : normalized departure from the reference ellipsoid
- `taper_ratio` : end-size ratio implied by the taper control
- `boxiness_exponent` : resolved superellipsoid exponent
- `transverse_asymmetry_I`, `transverse_asymmetry_S` : resolved transverse deformation controls
- `convexity_volume_ratio` : particle volume divided by convex-hull volume
- `sphericity_wadell_3d` : surface-area/volume-based 3D Wadell sphericity

These are evaluated on `shape.stl`, before the sphere-clump approximation.

### Measured shape-parameter vector

Every newly generated `meta.json` also contains a top-level
`shape_parameters` record. Unlike `shape_params`, which records generator
inputs, this record is measured from the actual source mesh passed to CLUMP:

```json
{
  "shape_parameters": {
    "schema": "clump-db.shape-parameters.v1",
    "geometry_role": "source_mesh",
    "e": 0.75,
    "f": 0.60,
    "tau": 0.0,
    "p": 2.0,
    "A_perp": 0.0,
    "I_3D": 0.0,
    "R_W1": 0.48,
    "R_W2": 0.48,
    "vector_order": ["e", "f", "tau", "p", "A_perp", "I_3D", "R_W1", "R_W2"],
    "vector": [0.75, 0.60, 0.0, 2.0, 0.0, 0.0, 0.48, 0.48]
  }
}
```

The flat ASCII field names are intended as stable `genSample` filters:

- `e`, `f`: form ratios measured from principal-axis bounding dimensions
- `tau`: unequal-end taper from opposed section areas at `q/L = 0.5`
- `p`: fitted one-exponent superellipsoid blockiness
- `A_perp`: RMS transverse mirror mismatch measured by filled voxels
- `I_3D`: `1 - particle_volume / convex_hull_volume`
- `R_W1`, `R_W2`: three-projection arithmetic and harmonic Wadell roundness

The record also includes component-level values, all three projection results,
and the measurement settings required to reproduce the calculation. Values in
`metrics.wadell` are derived only from component-sphere radii and remain a
clump-approximation proxy; they are not the source-mesh `R_W1` and `R_W2`.

To add the new record to cases generated by an older version:

```bash
python scripts/backfill_shape_parameters.py --root dataset/shapes
python scripts/rebuild_manifest.py --root dataset/shapes --out dataset/manifest.jsonl
```

Use `--dry-run` to inspect backfill values without changing any JSON, or
`--overwrite` to recompute records that already contain `shape_parameters`.

### Projection shape and SAGI

Every source mesh is also measured over deterministic, nearly uniform viewing
axes and stored in a separate top-level `projection_shape` record. This keeps
the two-dimensional dynamic-image-analysis descriptors separate from the 3D
eight-parameter vector:

```json
{
  "projection_shape": {
    "schema": "clump-db.projection-shape.v1",
    "geometry_role": "source_mesh",
    "AR": 0.76,
    "C_x": 0.95,
    "S": 0.90,
    "SAGI": 9.5,
    "SAGI_class": "rounded"
  }
}
```

The measurements follow the definitions used by Altuhafi et al. (2016):

- `AR`: minimum divided by maximum Feret diameter
- `C_x`: projected silhouette area divided by convex-hull area
- `S`: equal-area-circle perimeter divided by silhouette perimeter
- `SAGI`: `-5.4(1-AR) + 67.8(1-C_x) + 77.9(1-S)`

The SAGI signs follow the direction vector in Eq. (4) and reproduce the
positive values in Table 2; Eq. (5) appears in print with the opposite signs.
The record contains per-view measurements plus mean, standard deviation,
range, and 10th/50th/90th percentiles. Defaults are 64 orientations and a
512-pixel silhouette. They can be changed when generating a case with
`--projection-orientations` and `--projection-resolution`, or when backfilling
existing cases with the same options.

clump-DB stores measured geometry only. Material targets such as a Hime-gravel
mean SAGI near 9.5 belong in a downstream genSample profile, not in this
database schema.

---

## 2) `scripts/run_compact_case_hash.py`

Generate a full “case”:
- builds an irregular STL mesh
- runs CLUMP sphere packing
- writes outputs into a hash-based case directory

### Usage

```bash
python scripts/run_compact_case_hash.py \
  --shape-family rounded_trapezoid --taper 0.25 --boxiness 3.0 \
  --asymmetry-I 0.12 --asymmetry-S -0.06 \
  --L 1 --e 0.75 --f 0.65 --sub 1 --randomness 0.10 --seed 1234 \
  --N 30 --rMin 0.0 --div 50 --overlap 0.7 --rMax_ratio 1.0 \
  --Gs 2.65 --samples-volume 2000 --samples-inertia 2000 \
  --update-meta
```

### Parameters

#### STL generation (shape)
- `L` : historical longest-axis scale (semi-axis; the unperturbed full extent is `2L`)
- `e` : `I/L`
- `f` : `S/I`
- `shape-family` : source shape family
- `taper` : end taper for tapered families
- `boxiness` : superellipsoid exponent for boxy/rounded-trapezoidal families
- `asymmetry-I`, `asymmetry-S` : coherent non-symmetry in perpendicular cross-sections
- `sub` : subdivision level
- `randomness` : inward vertex perturbation magnitude
- `seed` : RNG seed

#### CLUMP sphere packing (clump-python)
- `N` : number of spheres (balls)
- `rMin` : minimum sphere radius
- `div` : voxel/grid resolution representing the STL boundary (higher → more accurate, slower)
- `overlap` : allowed overlap ratio between spheres
- `rMax_ratio` : max radius ratio used by the extended procedure  
  (e.g., `1.0` allows large spheres up to the boundary constraint)
- `samples-volume` : number of points to calculate volume and the center of mass using Monte Carlo method. The results make "molcule_mc.data"
- `samples-inertia` : number of points to calculate inertia tensor using Monte Carlo method. The results make "molcule_mc.data"


### Output (per case)

A case is stored under:

`dataset/shapes/<case_id>/`

Typical files:
- `shape.stl` : target mesh
- `balls_xyzr.txt` : sphere centers and radii (`x y z r`)
- `meta.json` : input parameters + derived metrics
- `molecule_mc.data` : molecule file for LAMMPS (using "molecule_mc.py")
- (optional) CLUMP outputs (e.g., `*.txt`, `*.vtk` depending on your script settings)

The `<case_id>` is derived from key parameters (hash-based), which helps caching/reproducibility.
The generated `meta.json` and rebuilt manifest expose the same top-level
`shape_parameters` object, so `genSample` can scan ranges without opening the
STL or recomputing geometry.

---

## 3) `scripts/view_case.py`

Interactive PyVista viewer for a generated case.

### Default behavior

- Shows **ALL balls by default**
- Shows STL as wireframe, and optionally a faint surface (`--surface`)
- Optional overlays:
  - `--largest` : show only the largest sphere (unless `--all-balls` is also set)
  - `--circ` : show circumscribing sphere (transparent surface only; no wire)

### Usage

Open the latest case:

```bash
python scripts/view_case.py --latest
```

Show STL surface + wireframe:

```bash
python scripts/view_case.py --latest --surface
```

Show largest sphere only (hides all spheres unless `--all-balls` is used):

```bash
python scripts/view_case.py --latest --surface --largest
```

Show largest + all spheres together:

```bash
python scripts/view_case.py --latest --surface --largest --all-balls
```

Show circumscribing sphere (transparent surface only):

```bash
python scripts/view_case.py --latest --surface --circ
```

Show largest + circumscribing sphere:

```bash
python scripts/view_case.py --latest --surface --largest --circ
```

Other useful options:
- `--edl` : Eye-Dome Lighting (adds depth cue)
- `--no-grid`, `--no-axes`
- `--no-target`, `--no-balls`
- Case selection:
  - `--latest`
  - `--root <dir>`
  - `--case-id <hash>`
  - `--case-dir <path>`

---

## Notes

- If you run scripts directly (e.g., `python scripts/make_shape.py ...`), installing the project in editable mode (`pip install -e .`) is recommended so that `import clumpgen` works reliably.
- For large clumps, reduce rendering resolution for speed:
  - `--balls-res 16` (or lower)

---

## Dependencies

- Python 3.x
- `numpy`
- `scipy` (convex-hull metric used for source-shape convexity)
- `trimesh`
- `pyvista` (viewer)
- `clump-python` (CLUMP wrapper)

Example install:

```bash
python -m pip install numpy pillow scipy trimesh pyvista clump-python
```
