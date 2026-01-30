# clump-DB

A small “database generator” for **irregular particle STL meshes** and **CLUMP** sphere-pack *clumps*, storing each generated case with a **hash-based case ID** and **JSON metadata**.  
This is designed for later binning/filtering (e.g., selecting clumps by shape/roundness metrics for LAMMPS runs).

---

## What this project does

For each case (one parameter set), the pipeline:

1. Generates an irregular ellipsoid STL mesh (`shape.stl`)
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
python -m pip install numpy trimesh pyvista
```

Install CLUMP Python wrapper:

```bash
python -m pip install clump-python
```

> CLUMP upstream reference: https://github.com/vsangelidakis/CLUMP

---

## 1) `scripts/make_shape.py`

Generate a single irregular ellipsoid STL mesh.  
This is useful to validate mesh parameters before running the full clump pipeline.

### Usage

```bash
python scripts/make_shape.py   --out outputs/test.stl   --L 10 --e 0.75 --f 0.65   --sub 4 --randomness 0.18 --bias 1.5 --seed 1234
```

### Shape parameters

- `L` : longest axis length
- `e` : `I/L` (intermediate-to-long axis ratio)
- `f` : `S/I` (short-to-intermediate axis ratio)
- `sub` : mesh subdivision level (higher → more vertices/faces)
- `randomness` : inward vertex perturbation magnitude
- `bias` : distribution skew for perturbation (higher → many small changes, fewer large ones)
- `seed` : RNG seed for reproducibility

---

## 2) `scripts/run_compact_case_hash.py`

Generate a full “case”:
- builds an irregular STL mesh
- runs CLUMP sphere packing
- writes outputs into a hash-based case directory

### Usage

```bash
python scripts/run_compact_case_hash.py \
  --L 1 --e 0.75 --f 0.65 --sub 1 --randomness 0.30 --seed 1234 \
  --N 40 --rMin 0.0 --div 300 --overlap 0.7 --rMax_ratio 1.0 \
  --Gs 2.65 --samples-volume 200000 --samples-inertia 200000 \
  --update-meta
```

### Parameters

#### STL generation (shape)
- `L` : longest axis length
- `e` : `I/L`
- `f` : `S/I`
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

### Output (per case)

A case is stored under:

`dataset/shapes/<case_id>/`

Typical files:
- `shape.stl` : target mesh
- `balls_xyzr.txt` : sphere centers and radii (`x y z r`)
- `meta.json` : input parameters + derived metrics
- (optional) CLUMP outputs (e.g., `*.txt`, `*.vtk` depending on your script settings)

The `<case_id>` is derived from key parameters (hash-based), which helps caching/reproducibility.

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
- `trimesh`
- `pyvista` (viewer)
- `clump-python` (CLUMP wrapper)

Example install:

```bash
python -m pip install numpy trimesh pyvista clump-python
```
