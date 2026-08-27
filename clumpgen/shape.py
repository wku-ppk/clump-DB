"""Parametric particle-mesh generation and source-mesh shape metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linprog, minimize_scalar
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull


SHAPE_FAMILIES = (
    "ellipsoid",
    "tapered_ellipsoid",
    "superellipsoid",
    "rounded_trapezoid",
)

SHAPE_PARAMETER_SCHEMA = "clump-db.shape-parameters.v1"
PROJECTION_SHAPE_SCHEMA = "clump-db.projection-shape.v1"
SAGI_FORMULA = "-5.4*(1-AR)+67.8*(1-C_x)+77.9*(1-S)"
SHAPE_PARAMETER_ORDER = (
    "e",
    "f",
    "tau",
    "p",
    "A_perp",
    "I_3D",
    "R_W1",
    "R_W2",
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


def _principal_frame(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return volume-centred vertices, L/I/S axes, and full axis extents."""
    if not mesh.is_watertight:
        raise ValueError("shape-parameter measurement requires a watertight mesh")

    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("mesh must contain at least four three-dimensional vertices")

    center = np.asarray(mesh.center_mass, dtype=float)
    inertia = np.asarray(mesh.moment_inertia, dtype=float)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(inertia)):
        raise ValueError("mesh has invalid mass properties")

    _, axes = np.linalg.eigh(inertia)
    local = (vertices - center) @ axes

    # Inertia eigenvalues normally put the geometric axes in L/I/S order, but
    # sorting the resulting spans also handles near-degenerate/equant shapes.
    order = np.argsort(np.ptp(local, axis=0))[::-1]
    axes = axes[:, order]
    local = (vertices - center) @ axes
    extents = np.ptp(local, axis=0)
    if np.any(extents <= np.finfo(float).eps):
        raise ValueError("mesh has a degenerate principal-axis extent")
    return local, axes, extents


def _section_area(
    vertices: np.ndarray,
    faces: np.ndarray,
    x_position: float,
) -> float:
    """Area enclosed by an oriented triangle-mesh section normal to local L."""
    triangles = vertices[np.asarray(faces, dtype=int)]
    scale = max(float(np.ptp(vertices[:, 0])), 1.0)
    tolerance = 1.0e-10 * scale
    plane_normal = np.array([1.0, 0.0, 0.0])
    twice_area = 0.0

    for triangle in triangles:
        distances = triangle[:, 0] - float(x_position)
        if np.all(distances > tolerance) or np.all(distances < -tolerance):
            continue

        points: list[np.ndarray] = []
        for index in range(3):
            a = triangle[index]
            b = triangle[(index + 1) % 3]
            da = distances[index]
            db = distances[(index + 1) % 3]
            if abs(da) <= tolerance:
                points.append(a)
            if da * db < -(tolerance * tolerance):
                fraction = da / (da - db)
                points.append(a + fraction * (b - a))

        unique: list[np.ndarray] = []
        for point in points:
            if not any(np.linalg.norm(point - other) <= tolerance for other in unique):
                unique.append(point)
        if len(unique) < 2:
            continue
        if len(unique) > 2:
            pairs = [
                (np.linalg.norm(unique[i] - unique[j]), unique[i], unique[j])
                for i in range(len(unique))
                for j in range(i + 1, len(unique))
            ]
            _, first, second = max(pairs, key=lambda item: item[0])
        else:
            first, second = unique

        face_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        tangent = np.cross(plane_normal, face_normal)
        if np.dot(second - first, tangent) < 0.0:
            first, second = second, first
        twice_area += first[1] * second[2] - second[1] * first[2]

    return abs(0.5 * float(twice_area))


def _taper_index(
    local_vertices: np.ndarray,
    faces: np.ndarray,
    q_over_L: float,
) -> tuple[float, dict[str, float]]:
    """Measure the unequal-end area-equivalent-radius taper index."""
    if not 0.0 < float(q_over_L) < 1.0:
        raise ValueError("taper_q_over_L must be between 0 and 1")

    lower = float(local_vertices[:, 0].min())
    upper = float(local_vertices[:, 0].max())
    midpoint = 0.5 * (lower + upper)
    half_length = 0.5 * (upper - lower)
    offset = float(q_over_L) * half_length
    area_negative = _section_area(local_vertices, faces, midpoint - offset)
    area_positive = _section_area(local_vertices, faces, midpoint + offset)
    maximum = max(area_negative, area_positive)
    if maximum <= np.finfo(float).eps:
        raise ValueError("taper sections have zero area; choose a smaller taper_q_over_L")
    tau = 1.0 - np.sqrt(min(area_negative, area_positive) / maximum)
    return float(np.clip(tau, 0.0, 1.0)), {
        "q_over_L": float(q_over_L),
        "area_negative": float(area_negative),
        "area_positive": float(area_positive),
    }


def _fit_superellipsoid_exponent(
    local_vertices: np.ndarray,
    extents: np.ndarray,
) -> tuple[float, float]:
    """Fit one equal superellipsoid exponent to principal-frame vertices."""
    midpoints = 0.5 * (
        local_vertices.max(axis=0) + local_vertices.min(axis=0)
    )
    normalized = np.abs(
        (local_vertices - midpoints) / (0.5 * np.asarray(extents, dtype=float))
    )
    normalized = np.clip(normalized, 0.0, 1.0)

    def objective(exponent: float) -> float:
        residual = np.sum(normalized ** float(exponent), axis=1) - 1.0
        return float(np.mean(residual * residual))

    result = minimize_scalar(
        objective,
        bounds=(0.25, 64.0),
        method="bounded",
        options={"xatol": 1.0e-6},
    )
    exponent = float(result.x)
    return exponent, float(np.sqrt(objective(exponent)))


def _row_keys(rows: np.ndarray) -> np.ndarray:
    """Encode integer rows for fast set intersection/union operations."""
    contiguous = np.ascontiguousarray(rows, dtype=np.int64)
    return contiguous.view(
        np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    ).reshape(-1)


def _mirror_mismatch(
    mesh: trimesh.Trimesh,
    local_vertices: np.ndarray,
    resolution: int,
) -> tuple[float, float, float, float]:
    """Approximate transverse volumetric mirror mismatch on a filled voxel grid."""
    if int(resolution) < 16:
        raise ValueError("asymmetry_voxel_resolution must be at least 16")

    local_mesh = trimesh.Trimesh(
        vertices=local_vertices,
        faces=np.asarray(mesh.faces, dtype=int),
        process=False,
    )
    pitch = float(np.ptp(local_vertices, axis=0).max() / int(resolution))
    occupied = np.rint(local_mesh.voxelized(pitch).fill().points / pitch).astype(np.int64)
    occupied_keys = np.unique(_row_keys(occupied))

    mismatches: list[float] = []
    for coordinate in (1, 2):
        reflected = occupied.copy()
        reflected[:, coordinate] *= -1
        reflected_keys = np.unique(_row_keys(reflected))
        intersection = np.intersect1d(
            occupied_keys,
            reflected_keys,
            assume_unique=True,
        ).size
        union = occupied_keys.size + reflected_keys.size - intersection
        mismatch = 1.0 - intersection / float(union) if union else 0.0
        mismatches.append(float(np.clip(mismatch, 0.0, 1.0)))

    A_I, A_S = mismatches
    A_perp = float(np.sqrt((A_I * A_I + A_S * A_S) / 2.0))
    return A_I, A_S, A_perp, pitch


def _resample_closed_polygon(polygon: np.ndarray, count: int) -> np.ndarray:
    """Resample a closed polygon at equally spaced arc-length positions."""
    closed = np.vstack([polygon, polygon[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = segment_lengths > np.finfo(float).eps
    polygon = polygon[keep]
    closed = np.vstack([polygon, polygon[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = float(cumulative[-1])
    if perimeter <= np.finfo(float).eps:
        raise ValueError("projection silhouette has zero perimeter")

    targets = np.arange(int(count), dtype=float) * perimeter / int(count)
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, len(segment_lengths) - 1)
    fractions = (targets - cumulative[indices]) / segment_lengths[indices]
    return closed[indices] + fractions[:, None] * (
        closed[indices + 1] - closed[indices]
    )


def _maximum_inscribed_circle_radius(hull: ConvexHull) -> float:
    """Solve the Chebyshev-centre problem for a convex 2-D projection."""
    equations = np.asarray(hull.equations, dtype=float)
    constraints = np.column_stack([equations[:, :2], np.ones(len(equations))])
    result = linprog(
        c=np.array([0.0, 0.0, -1.0]),
        A_ub=constraints,
        b_ub=-equations[:, 2],
        bounds=[(None, None), (None, None), (0.0, None)],
        method="highs",
    )
    if not result.success or result.x[2] <= np.finfo(float).eps:
        raise ValueError("could not calculate a maximum inscribed projection circle")
    return float(result.x[2])


def _projection_roundness(
    points: np.ndarray,
    sample_count: int,
    smoothing_fraction: float,
) -> dict[str, Any]:
    """Measure Wadell arithmetic/harmonic roundness on one projection."""
    if int(sample_count) < 64:
        raise ValueError("roundness_projection_samples must be at least 64")
    if not 0.0 < float(smoothing_fraction) < 0.1:
        raise ValueError("roundness_smoothing_fraction must be between 0 and 0.1")

    hull = ConvexHull(np.asarray(points, dtype=float))
    polygon = np.asarray(points, dtype=float)[hull.vertices]
    contour = _resample_closed_polygon(polygon, int(sample_count))
    sigma = max(1.0, float(sample_count) * float(smoothing_fraction))
    smooth = np.column_stack(
        [gaussian_filter1d(contour[:, index], sigma=sigma, mode="wrap") for index in range(2)]
    )

    perimeter = float(
        np.linalg.norm(np.roll(smooth, -1, axis=0) - smooth, axis=1).sum()
    )
    spacing = perimeter / int(sample_count)
    first = (np.roll(smooth, -1, axis=0) - np.roll(smooth, 1, axis=0)) / (2.0 * spacing)
    second = (
        np.roll(smooth, -1, axis=0) - 2.0 * smooth + np.roll(smooth, 1, axis=0)
    ) / (spacing * spacing)
    denominator = np.maximum(
        (first[:, 0] ** 2 + first[:, 1] ** 2) ** 1.5,
        np.finfo(float).eps,
    )
    curvature = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / denominator
    positive = curvature[curvature > np.finfo(float).eps]
    if not len(positive):
        raise ValueError("projection curvature is zero everywhere")

    mean_curvature = float(np.mean(positive))
    relative_variation = float(np.ptp(positive) / mean_curvature)
    if relative_variation < 0.08:
        corner_curvatures = np.array([mean_curvature])
    else:
        prominence = max(0.05 * float(curvature.max()), 0.10 * float(curvature.std()))
        peak_indices, _ = find_peaks(
            curvature,
            distance=max(1, int(sample_count) // 12),
            prominence=prominence,
        )
        threshold = float(np.median(curvature) + 0.20 * (curvature.max() - np.median(curvature)))
        peak_indices = peak_indices[curvature[peak_indices] >= threshold]
        if not len(peak_indices):
            peak_indices = np.array([int(np.argmax(curvature))])
        corner_curvatures = curvature[peak_indices]

    maximum_inscribed_radius = _maximum_inscribed_circle_radius(hull)
    normalized_radii = np.clip(
        1.0 / (corner_curvatures * maximum_inscribed_radius),
        0.0,
        1.0,
    )
    R_W1 = float(np.mean(normalized_radii))
    R_W2 = (
        0.0
        if np.any(normalized_radii <= np.finfo(float).eps)
        else float(len(normalized_radii) / np.sum(1.0 / normalized_radii))
    )
    return {
        "R_W1": R_W1,
        "R_W2": R_W2,
        "corner_count": int(len(normalized_radii)),
        "maximum_inscribed_radius": maximum_inscribed_radius,
    }


def calculate_sagi(AR: float, C_x: float, S: float) -> float:
    """Return the Altuhafi et al. Shape-Angularity Group Indicator.

    The signs follow the direction vector in Eq. (4) and reproduce the
    positive values in Table 2 of Altuhafi et al. (2016). Equation (5) in the
    published article appears with all three signs reversed.
    """
    values = np.asarray([AR, C_x, S], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("AR, C_x, and S must be finite")
    return float(
        -5.4 * (1.0 - float(AR))
        + 67.8 * (1.0 - float(C_x))
        + 77.9 * (1.0 - float(S))
    )


def classify_sagi(value: float) -> str:
    """Classify SAGI using the boundaries proposed by Altuhafi et al."""
    if not np.isfinite(value):
        raise ValueError("SAGI must be finite")
    if value < 10.0:
        return "rounded"
    if value < 11.0:
        return "subrounded"
    if value < 12.0:
        return "subangular"
    return "angular"


def _fibonacci_hemisphere_directions(count: int) -> np.ndarray:
    """Return deterministic, nearly uniform unoriented viewing axes."""
    if int(count) < 1:
        raise ValueError("projection orientation_count must be positive")
    index = np.arange(int(count), dtype=float)
    z = (index + 0.5) / float(count)
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * (index + 0.5)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack(
        [radial * np.cos(azimuth), radial * np.sin(azimuth), z]
    )


def _normalize_projection_directions(directions: np.ndarray) -> np.ndarray:
    """Validate and normalize user-supplied projection directions."""
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3 or not len(directions):
        raise ValueError("projection directions must have shape (N, 3)")
    lengths = np.linalg.norm(directions, axis=1)
    if not np.all(np.isfinite(directions)) or np.any(lengths <= np.finfo(float).eps):
        raise ValueError("projection directions must be finite and nonzero")
    return directions / lengths[:, None]


def _projection_coordinates(vertices: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project three-dimensional vertices into a stable orthonormal view plane."""
    direction = np.asarray(direction, dtype=float)
    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(float(direction[2])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    axis_u = np.cross(direction, reference)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(direction, axis_u)
    return np.column_stack([vertices @ axis_u, vertices @ axis_v])


def _rasterized_projection_silhouette(
    points: np.ndarray,
    faces: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Rasterize the union of projected mesh triangles as a binary silhouette."""
    if int(resolution) < 64:
        raise ValueError("projection resolution must be at least 64")
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    lower = points.min(axis=0)
    spans = np.ptp(points, axis=0)
    maximum_span = float(spans.max())
    if maximum_span <= np.finfo(float).eps:
        raise ValueError("projection has zero extent")

    margin = 3.0
    scale = (float(resolution) - 2.0 * margin - 1.0) / maximum_span
    pixels = (points - lower) * scale + margin
    image = Image.new("1", (int(resolution), int(resolution)), color=0)
    draw = ImageDraw.Draw(image)
    for triangle in pixels[faces]:
        draw.polygon([tuple(vertex) for vertex in triangle], fill=1)
    return np.asarray(image, dtype=bool), float(1.0 / scale), pixels


def _crofton_perimeter(mask: np.ndarray, pixel_size: float) -> float:
    """Estimate perimeter from boundary crossings in four line directions."""
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    vertical = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    diagonal_positive = np.count_nonzero(
        padded[1:, 1:] != padded[:-1, :-1]
    )
    diagonal_negative = np.count_nonzero(
        padded[1:, :-1] != padded[:-1, 1:]
    )
    crossings = (
        horizontal
        + vertical
        + (diagonal_positive + diagonal_negative) / np.sqrt(2.0)
    )
    return float(np.pi * crossings * float(pixel_size) / 8.0)


def _feret_diameters(points: np.ndarray, hull: ConvexHull) -> tuple[float, float]:
    """Return minimum caliper width and maximum Feret diameter."""
    polygon = np.asarray(points, dtype=float)[hull.vertices]
    edges = np.roll(polygon, -1, axis=0) - polygon
    edge_lengths = np.linalg.norm(edges, axis=1)
    valid = edge_lengths > np.finfo(float).eps
    if not np.any(valid):
        raise ValueError("projection convex hull has no valid edges")
    edges = edges[valid]
    edge_lengths = edge_lengths[valid]
    normals = np.column_stack([-edges[:, 1], edges[:, 0]]) / edge_lengths[:, None]
    widths = np.ptp(polygon @ normals.T, axis=0)
    feret_minimum = float(widths.min())
    feret_maximum = float(pdist(polygon).max())
    if feret_minimum <= 0.0 or feret_maximum <= 0.0:
        raise ValueError("projection Feret diameter is zero")
    return feret_minimum, feret_maximum


def _single_projection_shape_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    direction: np.ndarray,
    resolution: int,
) -> dict[str, float | list[float]]:
    """Measure DIA-style shape factors on one orthographic projection."""
    points = _projection_coordinates(vertices, direction)
    hull = ConvexHull(points)
    hull_area = float(hull.volume)
    if hull_area <= np.finfo(float).eps:
        raise ValueError("projection convex hull has zero area")

    mask, pixel_size, pixels = _rasterized_projection_silhouette(
        points,
        faces,
        resolution,
    )
    projected_area = float(mask.sum()) * pixel_size * pixel_size
    hull_image = Image.new("1", (int(resolution), int(resolution)), color=0)
    ImageDraw.Draw(hull_image).polygon(
        [tuple(vertex) for vertex in pixels[hull.vertices]],
        fill=1,
    )
    rasterized_hull_area = (
        float(np.asarray(hull_image, dtype=bool).sum()) * pixel_size * pixel_size
    )
    perimeter = _crofton_perimeter(mask, pixel_size)
    if projected_area <= 0.0 or rasterized_hull_area <= 0.0 or perimeter <= 0.0:
        raise ValueError("projection silhouette has zero area or perimeter")

    feret_minimum, feret_maximum = _feret_diameters(points, hull)
    AR = float(np.clip(feret_minimum / feret_maximum, 0.0, 1.0))
    C_x = float(np.clip(projected_area / rasterized_hull_area, 0.0, 1.0))
    S = float(np.clip(2.0 * np.sqrt(np.pi * projected_area) / perimeter, 0.0, 1.0))
    SAGI = calculate_sagi(AR, C_x, S)
    return {
        "direction": [float(value) for value in direction],
        "AR": AR,
        "C_x": C_x,
        "S": S,
        "SAGI": SAGI,
        "feret_minimum": feret_minimum,
        "feret_maximum": feret_maximum,
        "projected_area": projected_area,
        "convex_hull_area": hull_area,
        "rasterized_convex_hull_area": rasterized_hull_area,
        "perimeter": perimeter,
    }


def _projection_metric_summary(values: np.ndarray) -> dict[str, float]:
    """Return stable distribution summaries for one projection metric."""
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def calculate_projection_shape_parameters(
    mesh: trimesh.Trimesh,
    *,
    orientation_count: int = 64,
    resolution: int = 512,
    directions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure projected AR, convexity, sphericity, and SAGI.

    Orthographic silhouettes are sampled over unoriented viewing axes because
    views along ``d`` and ``-d`` have identical scalar shape factors. The
    returned flat fields are orientation means intended for manifest filtering;
    per-view values and distribution summaries retain the observed variability.
    """
    if not mesh.is_watertight:
        raise ValueError("projection-shape measurement requires a watertight mesh")
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if directions is None:
        view_directions = _fibonacci_hemisphere_directions(int(orientation_count))
        orientation_method = "fibonacci_hemisphere_unoriented_axes"
    else:
        view_directions = _normalize_projection_directions(directions)
        orientation_method = "explicit_directions"

    views = [
        _single_projection_shape_metrics(
            vertices,
            faces,
            direction,
            int(resolution),
        )
        for direction in view_directions
    ]
    summaries = {
        name: _projection_metric_summary(
            np.asarray([float(view[name]) for view in views], dtype=float)
        )
        for name in ("AR", "C_x", "S", "SAGI")
    }
    means = {name: summaries[name]["mean"] for name in summaries}
    return {
        "schema": PROJECTION_SHAPE_SCHEMA,
        "geometry_role": "source_mesh",
        **means,
        "SAGI_class": classify_sagi(means["SAGI"]),
        "aggregate": summaries,
        "views": views,
        "measurement": {
            "projection_method": "orthographic_union_of_rasterized_mesh_triangles",
            "orientation_method": orientation_method,
            "orientation_count": int(len(view_directions)),
            "silhouette_resolution": int(resolution),
            "AR_method": "minimum_over_maximum_feret_diameter",
            "C_x_method": "silhouette_area_over_convex_hull_area",
            "S_method": "equivalent_circle_perimeter_over_silhouette_perimeter",
            "perimeter_method": "four_direction_crofton_boundary_crossings",
            "SAGI_formula": SAGI_FORMULA,
            "SAGI_reference": "Altuhafi_et_al_2016_Table_2_sign_convention",
        },
    }


def calculate_shape_parameters(
    mesh: trimesh.Trimesh,
    *,
    taper_q_over_L: float = 0.5,
    asymmetry_voxel_resolution: int = 48,
    roundness_projection_samples: int = 512,
    roundness_smoothing_fraction: float = 0.015,
) -> dict[str, Any]:
    """Calculate the database descriptor vector from an actual source mesh.

    The returned record deliberately uses flat ASCII keys for genSample filters,
    while retaining component and measurement details for reproducibility.
    """
    local_vertices, axes, extents = _principal_frame(mesh)
    L, I, S = (float(value) for value in extents)
    e = I / L
    f = S / I

    tau, taper_details = _taper_index(
        local_vertices,
        np.asarray(mesh.faces, dtype=int),
        taper_q_over_L,
    )
    exponent, exponent_rmse = _fit_superellipsoid_exponent(local_vertices, extents)
    A_I, A_S, A_perp, voxel_pitch = _mirror_mismatch(
        mesh,
        local_vertices,
        asymmetry_voxel_resolution,
    )

    particle_volume = abs(float(mesh.volume))
    hull_volume = abs(float(mesh.convex_hull.volume))
    if hull_volume <= np.finfo(float).eps:
        raise ValueError("mesh convex hull has zero volume")
    I_3D = float(np.clip(1.0 - particle_volume / hull_volume, 0.0, 1.0))

    view_axes = {"LI": (0, 1), "LS": (0, 2), "IS": (1, 2)}
    views = {
        name: _projection_roundness(
            local_vertices[:, indices],
            roundness_projection_samples,
            roundness_smoothing_fraction,
        )
        for name, indices in view_axes.items()
    }
    R_W1 = float(np.mean([view["R_W1"] for view in views.values()]))
    R_W2 = float(np.mean([view["R_W2"] for view in views.values()]))

    values = {
        "e": float(e),
        "f": float(f),
        "tau": float(tau),
        "p": float(exponent),
        "A_perp": float(A_perp),
        "I_3D": float(I_3D),
        "R_W1": float(R_W1),
        "R_W2": float(R_W2),
    }
    return {
        "schema": SHAPE_PARAMETER_SCHEMA,
        "geometry_role": "source_mesh",
        **values,
        "vector_order": list(SHAPE_PARAMETER_ORDER),
        "vector": [values[name] for name in SHAPE_PARAMETER_ORDER],
        "components": {
            "form": {"e": values["e"], "f": values["f"]},
            "family_geometry": {"tau": values["tau"], "p": values["p"]},
            "asymmetry": {"A_I": A_I, "A_S": A_S, "A_perp": values["A_perp"]},
            "concavity": {"I_3D": values["I_3D"]},
            "corner_rounding": {
                "R_W1": values["R_W1"],
                "R_W2": values["R_W2"],
                "views": views,
            },
        },
        "measurement": {
            "axis_method": "volume_inertia_principal_axes_with_bounding_extents",
            "principal_dimensions": {"L": L, "I": I, "S": S},
            "principal_axes_columns": [[float(value) for value in row] for row in axes],
            "taper_method": "opposed_section_area_equivalent_radii",
            "taper": taper_details,
            "p_method": "one_exponent_superellipsoid_vertex_fit",
            "p_fit_bounds": [0.25, 64.0],
            "p_fit_rmse": exponent_rmse,
            "asymmetry_method": "filled_voxel_mirror_jaccard",
            "asymmetry_voxel_resolution": int(asymmetry_voxel_resolution),
            "asymmetry_voxel_pitch": float(voxel_pitch),
            "concavity_method": "one_minus_mesh_volume_over_convex_hull_volume",
            "particle_volume": particle_volume,
            "convex_hull_volume": hull_volume,
            "roundness_method": "three_principal_convex_projection_curvature_peaks",
            "roundness_projection_samples": int(roundness_projection_samples),
            "roundness_smoothing_fraction": float(roundness_smoothing_fraction),
        },
    }
