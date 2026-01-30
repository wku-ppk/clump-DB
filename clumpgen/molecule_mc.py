# clumpgen/molecule_mc.py
"""
Monte Carlo mass properties (volume, COM, inertia) for a CLUMP "union of spheres".

Units:
- coordinates: meters
- density: kg/m^3
- mass: kg
- inertia: kg*m^2

This intentionally does NOT depend on Itasca/PFC.
It uses only numpy and works with sphere centers/radii from the CLUMP Python package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class MCMassProps:
    volume: float
    mass: float
    com: np.ndarray          # (3,)
    inertia: np.ndarray      # (6,) = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    bbox: np.ndarray         # (6,) = [xmin,xmax,ymin,ymax,zmin,zmax]
    density: float
    n_spheres: int
    samples_volume: int
    samples_inertia: int
    seed: int
    sphere_block: int


def _bbox_union_spheres(xyz: np.ndarray, r: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)
    mins = (xyz - r[:, None]).min(axis=0)
    maxs = (xyz + r[:, None]).max(axis=0)
    return np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]], dtype=float)


def _inside_union(points: np.ndarray, xyz: np.ndarray, r: np.ndarray, sphere_block: int = 256) -> np.ndarray:
    """
    points: (P,3)
    xyz: (M,3)
    r: (M,)
    Returns mask (P,) True if inside ANY sphere.
    """
    points = np.asarray(points, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)

    P = points.shape[0]
    inside = np.zeros(P, dtype=bool)

    # Process spheres in blocks to limit memory.
    M = xyz.shape[0]
    for j0 in range(0, M, sphere_block):
        j1 = min(M, j0 + sphere_block)
        c = xyz[j0:j1]                 # (B,3)
        rr = r[j0:j1]                  # (B,)

        # squared distance from points to each center: (P,B)
        d = points[:, None, :] - c[None, :, :]
        d2 = np.einsum("pbi,pbi->pb", d, d)  # (P,B)
        inside |= (d2 <= (rr[None, :] ** 2)).any(axis=1)
        if inside.all():
            break

    return inside


def compute_mc_mass_properties(
    xyz: np.ndarray,
    r: np.ndarray,
    *,
    density: float,
    samples_volume: int = 200_000,
    samples_inertia: int = 200_000,
    seed: int = 1234,
    sphere_block: int = 256,
    points_block: int = 20_000,
) -> MCMassProps:
    """
    Monte Carlo estimate for union-of-spheres mass properties.
    """
    xyz = np.asarray(xyz, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (M,3), got {xyz.shape}")
    if r.shape[0] != xyz.shape[0]:
        raise ValueError(f"r length mismatch: {r.shape[0]} vs {xyz.shape[0]}")
    if xyz.shape[0] == 0:
        raise ValueError("No spheres (empty clump).")

    bbox = _bbox_union_spheres(xyz, r)
    xmin, xmax, ymin, ymax, zmin, zmax = bbox.tolist()
    box_vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    if box_vol <= 0:
        raise RuntimeError("Degenerate bounding box.")

    rng = np.random.default_rng(int(seed))

    # ---- Volume + COM ----
    inside_count = 0
    sum_xyz = np.zeros(3, dtype=float)

    n_left = int(samples_volume)
    while n_left > 0:
        n = min(points_block, n_left)
        pts = np.column_stack([
            rng.uniform(xmin, xmax, size=n),
            rng.uniform(ymin, ymax, size=n),
            rng.uniform(zmin, zmax, size=n),
        ])
        mask = _inside_union(pts, xyz, r, sphere_block=sphere_block)
        k = int(mask.sum())
        if k:
            inside_count += k
            sum_xyz += pts[mask].sum(axis=0)
        n_left -= n

    if inside_count == 0:
        raise RuntimeError("No inside points. Increase samples or check clump scaling.")

    frac_in = inside_count / float(samples_volume)
    volume = frac_in * box_vol
    com = sum_xyz / float(inside_count)

    # ---- Inertia about COM ----
    inside_count2 = 0
    Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0

    n_left = int(samples_inertia)
    while n_left > 0:
        n = min(points_block, n_left)
        pts = np.column_stack([
            rng.uniform(xmin, xmax, size=n),
            rng.uniform(ymin, ymax, size=n),
            rng.uniform(zmin, zmax, size=n),
        ])
        mask = _inside_union(pts, xyz, r, sphere_block=sphere_block)
        if mask.any():
            p = pts[mask]
            rx = p[:, 0] - com[0]
            ry = p[:, 1] - com[1]
            rz = p[:, 2] - com[2]
            inside_count2 += p.shape[0]

            Ixx += np.sum(ry * ry + rz * rz)
            Iyy += np.sum(rx * rx + rz * rz)
            Izz += np.sum(rx * rx + ry * ry)
            Ixy += np.sum(-rx * ry)
            Ixz += np.sum(-rx * rz)
            Iyz += np.sum(-ry * rz)

        n_left -= n

    if inside_count2 == 0:
        raise RuntimeError("No inside points for inertia. Increase samples.")

    mass = float(density) * float(volume)

    # Average integrand over inside volume, then multiply by mass (ρ*V).
    scale = mass / float(inside_count2)
    inertia = np.array([Ixx, Iyy, Izz, Ixy, Ixz, Iyz], dtype=float) * scale

    return MCMassProps(
        volume=float(volume),
        mass=float(mass),
        com=np.asarray(com, dtype=float),
        inertia=np.asarray(inertia, dtype=float),
        bbox=np.asarray(bbox, dtype=float),
        density=float(density),
        n_spheres=int(xyz.shape[0]),
        samples_volume=int(samples_volume),
        samples_inertia=int(samples_inertia),
        seed=int(seed),
        sphere_block=int(sphere_block),
    )


def write_molecule_file(
    path: Path,
    xyz: np.ndarray,
    r: np.ndarray,
    props: MCMassProps,
    *,
    type_id: int = 1,
    include_per_sphere_masses: bool = True,
) -> None:
    """
    Write a LAMMPS-style molecule-like data file (same structure as your example).
    Note: 'Masses' section is per-sphere mass ignoring overlap (optional).
    """
    path = Path(path)
    xyz = np.asarray(xyz, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)
    M = xyz.shape[0]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Monte Carlo Molecule for Clumps\n")
        f.write("# header section:\n")
        f.write(f"{M} atoms\n")
        f.write(f"{props.mass} mass\n")
        f.write(f"{props.com[0]} {props.com[1]} {props.com[2]} com\n")
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = props.inertia.tolist()
        f.write(f"{Ixx} {Iyy} {Izz} {Ixy} {Ixz} {Iyz} inertia\n\n")

        f.write("# body section:\n")
        f.write("Coords\n\n")
        for i in range(M):
            f.write(f"{i+1} {xyz[i,0]} {xyz[i,1]} {xyz[i,2]}\n")

        f.write("\nDiameters\n\n")
        for i in range(M):
            f.write(f"{i+1} {2.0*r[i]}\n")

        f.write("\nTypes\n\n")
        for i in range(M):
            f.write(f"{i+1} {type_id}\n")

        if include_per_sphere_masses:
            f.write("\nMasses\n\n")
            # Ignore overlaps: sphere volume * density
            vol_sphere = (4.0/3.0) * np.pi * (r ** 3)
            m_sphere = props.density * vol_sphere
            for i in range(M):
                f.write(f"{i+1} {m_sphere[i]}\n")


def compute_and_write_molecule_from_xyzr(
    *,
    xyz: np.ndarray,
    r: np.ndarray,
    out_path: Path,
    Gs: float,
    rho_water: float = 1000.0,
    density: Optional[float] = None,
    samples_volume: int = 200_000,
    samples_inertia: int = 200_000,
    seed: int = 1234,
    sphere_block: int = 256,
    points_block: int = 20_000,
) -> Tuple[Path, Dict]:
    """
    Convenience wrapper used by scripts/run_compact_case_hash.py

    Returns: (written_path, mc_dict_for_meta)
    """
    dens = float(density) if density is not None else float(Gs) * float(rho_water)

    props = compute_mc_mass_properties(
        xyz, r,
        density=dens,
        samples_volume=int(samples_volume),
        samples_inertia=int(samples_inertia),
        seed=int(seed),
        sphere_block=int(sphere_block),
        points_block=int(points_block),
    )
    out_path = Path(out_path)
    write_molecule_file(out_path, xyz, r, props)

    mc = {
        "method": "monte_carlo_union_of_spheres",
        "Gs": float(Gs),
        "rho_water": float(rho_water),
        "density": float(props.density),
        "samples_volume": int(props.samples_volume),
        "samples_inertia": int(props.samples_inertia),
        "seed": int(props.seed),
        "sphere_block": int(props.sphere_block),
        "volume_est": float(props.volume),
        "mass_est": float(props.mass),
        "com": [float(x) for x in props.com.tolist()],
        "inertia": [float(x) for x in props.inertia.tolist()],
        "bbox": [float(x) for x in props.bbox.tolist()],
        "molecule_file": out_path.name,
    }
    return out_path, mc
