"""Parametric particle-mesh generation and source-mesh shape metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import trimesh


SHAPE_FAMILIES = (
    "ellipsoid",
    "tapered_ellipsoid",
    "superellipsoid",
    "rounded_trapezoid",
)


@dataclass(frozen=True)
class ShapeParams:
    L: float = 10.0
    e: float = 0.8  # I/L
    f: float = 0.7  # S/I
    family: str = "ellipsoid"
    taper: float = 0.25
    boxiness: float = 3.0
    asymmetry_I: float = 0.0
    asymmetry_S: float = 0.0
    subdivisions: int = 4
    randomness: float = 0.12
    bias: float = 1.5
    seed: int = 1234


def resolve_shape_params(p: ShapeParams) -> ShapeParams:
    """Validate parameters and remove controls that a family does not use."""
    family = str(p.family).lower()
    if family not in SHAPE_FAMILIES:
        choices = ", ".join(SHAPE_FAMILIES)
        raise ValueError(f"unknown shape family {p.family!r}; choose one of: {choices}")
    if float(p.L) <= 0.0:
        raise ValueError("L must be positive")
    if not 0.0 < float(p.e) <= 1.0:
        raise ValueError("e=I/L must be in (0, 1]")
    if not 0.0 < float(p.f) <= 1.0:
        raise ValueError("f=S/I must be in (0, 1]")
    if int(p.subdivisions) < 0:
        raise ValueError("subdivisions must be non-negative")
    if not 0.0 <= float(p.randomness) < 1.0:
        raise ValueError("randomness must be in [0, 1)")
    if float(p.bias) <= 0.0:
        raise ValueError("bias must be positive")
    if abs(float(p.taper)) >= 1.0:
        raise ValueError("taper must be greater than -1 and less than 1")
    if float(p.boxiness) < 2.0:
        raise ValueError("boxiness must be at least 2")
    if abs(float(p.asymmetry_I)) >= 0.5:
        raise ValueError("asymmetry_I must be greater than -0.5 and less than 0.5")
    if abs(float(p.asymmetry_S)) >= 0.5:
        raise ValueError("asymmetry_S must be greater than -0.5 and less than 0.5")

    uses_taper = family in {"tapered_ellipsoid", "rounded_trapezoid"}
    uses_boxiness = family in {"superellipsoid", "rounded_trapezoid"}
    return replace(
        p,
        family=family,
        taper=float(p.taper) if uses_taper else 0.0,
        boxiness=float(p.boxiness) if uses_boxiness else 2.0,
    )


def _superellipsoid_vertices(
    unit_vertices: np.ndarray,
    axes: np.ndarray,
    exponent: float,
) -> np.ndarray:
    """Radially map unit-sphere vertices onto an axis-aligned Lp surface."""
    denominator = np.sum(np.abs(unit_vertices) ** exponent, axis=1) ** (1.0 / exponent)
    return unit_vertices / denominator[:, None] * axes


def _apply_long_axis_taper(vertices: np.ndarray, L: float, taper: float) -> np.ndarray:
    """
    Taper one end along the L axis.

    ``abs(taper)`` is the fractional reduction at the smaller end. A positive
    value shrinks the +L end; a negative value shrinks the -L end.
    """
    if taper == 0.0:
        return vertices

    result = vertices.copy()
    x_normalized = np.clip(result[:, 0] / float(L), -1.0, 1.0)
    progression = (x_normalized + 1.0) / 2.0
    if taper < 0.0:
        progression = 1.0 - progression
    cross_section_scale = 1.0 - abs(float(taper)) * progression
    result[:, 1:] *= cross_section_scale[:, None]
    return result


def _apply_transverse_asymmetry(
    vertices: np.ndarray,
    I: float,
    S: float,
    asymmetry_I: float,
    asymmetry_S: float,
) -> np.ndarray:
    """Create coherent, one-sided bulging in the two transverse directions."""
    if asymmetry_I == 0.0 and asymmetry_S == 0.0:
        return vertices

    result = vertices.copy()
    normalized_I = np.clip(result[:, 1] / float(I), -1.0, 1.0)
    normalized_S = np.clip(result[:, 2] / float(S), -1.0, 1.0)
    result[:, 1] *= 1.0 + float(asymmetry_I) * normalized_I
    result[:, 2] *= 1.0 + float(asymmetry_S) * normalized_S
    return result


def make_particle_mesh(p: ShapeParams) -> trimesh.Trimesh:
    """
    Create a watertight particle mesh from one of the supported families.

    Families
    --------
    ellipsoid
        Backward-compatible inward-perturbed ellipsoid.
    tapered_ellipsoid
        Ellipsoid with one cross-sectionally smaller end.
    superellipsoid
        Untapered rounded/blocky particle; ``boxiness=2`` is an ellipsoid.
    rounded_trapezoid
        Tapered superellipsoid, producing a rounded wedge/trapezoidal profile.

    ``asymmetry_I`` and ``asymmetry_S`` enlarge one transverse side while
    compressing the opposite side. Their signs select the enlarged side.

    The historical project convention is retained: ``L``, ``I=eL`` and
    ``S=fI`` are coordinate scale factors (semi-axis lengths before taper).
    """
    p = resolve_shape_params(p)
    L = float(p.L)
    I = float(p.e) * L
    S = float(p.f) * I

    mesh = trimesh.creation.icosphere(subdivisions=int(p.subdivisions), radius=1.0)
    vertices = _superellipsoid_vertices(
        np.asarray(mesh.vertices, dtype=float),
        np.array([L, I, S], dtype=float),
        float(p.boxiness),
    )
    vertices = _apply_long_axis_taper(vertices, L=L, taper=float(p.taper))
    vertices = _apply_transverse_asymmetry(
        vertices,
        I=I,
        S=S,
        asymmetry_I=float(p.asymmetry_I),
        asymmetry_S=float(p.asymmetry_S),
    )

    rng = np.random.default_rng(int(p.seed))
    perturbation = rng.random(len(vertices)) ** float(p.bias)
    vertices *= (1.0 - float(p.randomness) * perturbation)[:, None]
    mesh.vertices = vertices
    return mesh


def make_particle_stl(out_path: str | Path, p: ShapeParams) -> trimesh.Trimesh:
    """Create and export a particle mesh."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh = make_particle_mesh(p)
    mesh.export(out.as_posix())
    return mesh


def make_irregular_ellipsoid_stl(out_path: str | Path, p: ShapeParams) -> trimesh.Trimesh:
    """Backward-compatible alias for the original public function."""
    return make_particle_stl(out_path, replace(p, family="ellipsoid"))


def source_shape_metrics(mesh: trimesh.Trimesh, p: ShapeParams) -> dict[str, float]:
    """Compute descriptors from the source mesh, before clump approximation."""
    p = resolve_shape_params(p)
    L = float(p.L)
    I = float(p.e) * L
    S = float(p.f) * I
    vertices = np.asarray(mesh.vertices, dtype=float)

    normalized_radius = np.sqrt(
        (vertices[:, 0] / L) ** 2
        + (vertices[:, 1] / I) ** 2
        + (vertices[:, 2] / S) ** 2
    )
    ellipsoid_reference_rms = float(np.sqrt(np.mean((normalized_radius - 1.0) ** 2)))

    volume = abs(float(mesh.volume))
    area = float(mesh.area)
    hull_volume = abs(float(mesh.convex_hull.volume))
    convexity = float(volume / hull_volume) if hull_volume > 0.0 else float("nan")
    sphericity = (
        float(np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area)
        if volume > 0.0 and area > 0.0
        else float("nan")
    )

    return {
        "ellipsoid_reference_rms": ellipsoid_reference_rms,
        "convexity_volume_ratio": convexity,
        "sphericity_wadell_3d": sphericity,
        "taper_ratio": 1.0 - abs(float(p.taper)),
        "boxiness_exponent": float(p.boxiness),
        "transverse_asymmetry_I": float(p.asymmetry_I),
        "transverse_asymmetry_S": float(p.asymmetry_S),
    }
