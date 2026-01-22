# Script for generating STL

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import trimesh


@dataclass(frozen=True)
class ShapeParams:
    L: float = 10.0
    e: float = 0.8           # I/L
    f: float = 0.7           # S/I
    subdivisions: int = 4
    randomness: float = 0.12
    bias: float = 1.5
    seed: int = 1234


def make_irregular_ellipsoid_stl(out_path: str | Path, p: ShapeParams) -> trimesh.Trimesh:
    """
    Create an inward-perturbed ellipsoid-like watertight mesh from an icosphere.

    Mapping:
      I = e * L
      S = f * I  (= f * e * L)

    Randomness:
      vertices scaled inward by (1 - randomness * u), u ~ U(0,1)^bias
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    L = float(p.L)
    I = float(p.e) * L
    S = float(p.f) * I

    m = trimesh.creation.icosphere(subdivisions=int(p.subdivisions), radius=1.0)
    v = m.vertices * np.array([L, I, S], dtype=float)

    rng = np.random.default_rng(int(p.seed))
    u = rng.random(len(v)) ** float(p.bias)
    m.vertices = v * (1.0 - float(p.randomness) * u)[:, None]  # inward-only

    m.export(out.as_posix())
    return m