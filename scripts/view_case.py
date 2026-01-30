#python scripts/view_case.py --latest
#python scripts/view_case.py --latest --surface
#python scripts/view_case.py --latest --surface --largest
#python scripts/view_case.py --latest --surface --largest --circ
#--edl, --no-grid, --no-axes, --no-target, --no-balls
#--latest / --root / --case-id / --case-dir

import argparse
import json
from pathlib import Path

import numpy as np


def load_xyzr_from_molecule_file(path: Path):
    """
    Parse molecule_mc.data (Coords + Diameters) and return xyz, r.
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    def find_section(name: str) -> int:
        for i, ln in enumerate(lines):
            if ln.strip() == name:
                return i
        return -1

    i_coords = find_section("Coords")
    i_diams = find_section("Diameters")
    if i_coords < 0 or i_diams < 0:
        raise ValueError(f"Cannot find Coords/Diameters in {path}")

    coords = {}
    i = i_coords + 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    while i < len(lines) and lines[i].strip() != "":
        parts = lines[i].split()
        if len(parts) >= 4:
            idx = int(parts[0])
            coords[idx] = (float(parts[1]), float(parts[2]), float(parts[3]))
        i += 1

    diams = {}
    i = i_diams + 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    while i < len(lines) and lines[i].strip() != "":
        parts = lines[i].split()
        if len(parts) >= 2:
            idx = int(parts[0])
            diams[idx] = float(parts[1])
        i += 1

    ids = sorted(set(coords.keys()) & set(diams.keys()))
    if not ids:
        raise ValueError(f"No matching ids in Coords/Diameters for {path}")

    xyz = np.array([coords[i] for i in ids], dtype=float)
    r = np.array([0.5 * diams[i] for i in ids], dtype=float)
    return xyz, r
import trimesh
import pyvista as pv


def trimesh_to_pyvista(tm: trimesh.Trimesh) -> pv.PolyData:
    """Convert trimesh.Trimesh -> pyvista.PolyData safely."""
    faces = tm.faces.astype(np.int64)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(tm.vertices, faces_pv)


def load_meta_optional(meta_path: Path) -> dict:
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def find_latest_case_dir(root: Path) -> Path:
    """
    root 아래에서 'shape.stl' 또는 'meta.json'이 있는 케이스 폴더를 찾고,
    가장 최근 수정(mtime)된 폴더를 반환합니다.
    """
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    candidates = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if (d / "shape.stl").exists() or (d / "meta.json").exists():
            key_file = (d / "meta.json") if (d / "meta.json").exists() else (d / "shape.stl")
            candidates.append((key_file.stat().st_mtime, d))

    if not candidates:
        raise FileNotFoundError(f"No case directories found under: {root}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def resolve_case_dir(case_dir: str | None, case_id: str | None, root: str, latest: bool) -> Path:
    """
    우선순위:
      1) --case-dir
      2) --case-id  (root/case-id)
      3) --latest   (root 아래 최신 케이스)
    """
    root_path = Path(root)

    if case_dir:
        return Path(case_dir)
    if case_id:
        return root_path / case_id
    if latest:
        return find_latest_case_dir(root_path)

    raise ValueError("다음 중 하나는 반드시 제공해야 합니다: --case-dir, --case-id, --latest")


def main():
    ap = argparse.ArgumentParser(
        description="View case with STL (wire + optional surface) and balls. "
                    "Default: ALL balls. "
                    "--largest => largest only (unless --all-balls). "
                    "--circ => transparent circumsphere."
    )

    # ---- Case selection ----
    ap.add_argument("--case-dir", default="", help="e.g., dataset/shapes/<case_id>")
    ap.add_argument("--case-id", default="", help="e.g., d4d969c0de48 (will use --root)")
    ap.add_argument("--latest", action="store_true", help="open most recent case under --root")
    ap.add_argument("--root", default="dataset/shapes", help="case-id/latest root dir")

    # ---- Global toggles ----
    ap.add_argument("--no-target", action="store_true", help="hide STL target")
    ap.add_argument("--no-balls", action="store_true", help="hide balls")
    ap.add_argument("--no-axes", action="store_true", help="hide axes")
    ap.add_argument("--no-grid", action="store_true", help="hide grid")
    ap.add_argument("--edl", action="store_true", help="enable Eye-Dome Lighting (depth cue)")

    # ---- STL rendering: wire + optional faint surface ----
    ap.add_argument("--surface", action="store_true", help="also draw faint STL surface (in addition to wire)")
    ap.add_argument("--surface-color", default="yellow")
    ap.add_argument("--surface-opacity", type=float, default=0.18)

    ap.add_argument("--wire-color", default="darkgray")
    ap.add_argument("--wire-opacity", type=float, default=0.65)
    ap.add_argument("--wire-width", type=float, default=2.0)

    # ---- Balls appearance ----
    ap.add_argument("--balls-color", default="steelblue")
    ap.add_argument("--balls-opacity", type=float, default=1.0)
    ap.add_argument("--balls-res", type=int, default=24, help="sphere resolution for ALL balls glyph (lower=faster)")

    # ---- Optional: largest ball behavior ----
    ap.add_argument("--largest", action="store_true",
                    help="show only the largest ball (unless --all-balls is also set)")
    ap.add_argument("--all-balls", action="store_true",
                    help="force draw ALL balls even when --largest is set (largest + all)")
    ap.add_argument("--largest-color", default="red")
    ap.add_argument("--largest-opacity", type=float, default=1.0)
    ap.add_argument("--largest-res", type=int, default=96)

    # ---- Optional: circumscribing sphere (transparent surface only) ----
    ap.add_argument("--circ", action="store_true", help="draw circumscribing sphere (transparent surface only)")
    ap.add_argument("--circ-color", default="white")
    ap.add_argument("--circ-opacity", type=float, default=0.15)
    ap.add_argument("--circ-res", type=int, default=96)

    ap.add_argument("--window", default="1400x1000", help='window size like "1400x1000"')

    args = ap.parse_args()

    # ---- Resolve case dir ----
    case_dir = resolve_case_dir(
        case_dir=args.case_dir.strip() or None,
        case_id=args.case_id.strip() or None,
        root=args.root,
        latest=bool(args.latest),
    )

    print(f"[INFO] Using case-dir: {case_dir}")

    stl_path = case_dir / "shape.stl"
    balls_path = case_dir / "balls_xyzr.txt"
    meta_path = case_dir / "meta.json"

    if not stl_path.exists():
        raise FileNotFoundError(f"Missing STL: {stl_path}")
    if not balls_path.exists() and not args.no_balls:
        raise FileNotFoundError(f"Missing balls file: {balls_path} (or use --no-balls)")

    meta = load_meta_optional(meta_path)

    # ---- Load target mesh ----
    mesh = trimesh.load_mesh(stl_path.as_posix())
    target = trimesh_to_pyvista(mesh)

    # ---- Load balls ----
    xyz = None
    r = None
    if not args.no_balls and balls_path.exists():
        xyzr = np.loadtxt(balls_path.as_posix(), dtype=float)
        if xyzr.ndim == 1:
            xyzr = xyzr.reshape(1, 4)
        xyz = xyzr[:, :3]
        r = xyzr[:, 3]

    # ---- Largest ball (for --largest) ----
    if r is not None and len(r) > 0:
        i_max = int(np.argmax(r))
        c_in = xyz[i_max]
        r_in = float(r[i_max])
    else:
        i_max, c_in, r_in = None, None, None

    # ---- Circumscribing sphere (for --circ): prefer meta, else recompute ----
    center = None
    r_out = None
    if args.circ:
        cs = meta.get("metrics", {}).get("circumsphere", {})
        if "center" in cs and ("r_out" in cs or "D_out" in cs):
            center = np.array(cs["center"], dtype=float)
            r_out = float(cs["r_out"]) if "r_out" in cs else float(cs["D_out"]) / 2.0
        else:
            c2, r2 = trimesh.nsphere.minimum_nsphere(mesh)
            center = np.array(c2, dtype=float)
            r_out = float(r2)

    # ---- Plotter ----
    try:
        w, h = args.window.lower().split("x")
        window_size = (int(w), int(h))
    except Exception:
        window_size = (1400, 1000)

    p = pv.Plotter(lighting="three lights", window_size=window_size)
    if args.edl:
        p.enable_eye_dome_lighting()

    # 1) STL: faint surface (optional) + wireframe
    if not args.no_target:
        if args.surface:
            p.add_mesh(
                target,
                opacity=float(args.surface_opacity),
                color=str(args.surface_color),
                smooth_shading=True,
                show_edges=False,
                ambient=0.10,
                diffuse=0.85,
                specular=0.35,
                specular_power=40,
            )

        p.add_mesh(
            target,
            style="wireframe",
            color=str(args.wire_color),
            line_width=float(args.wire_width),
            opacity=float(args.wire_opacity),
        )

    # 2) Balls
    # 기본: ALL balls 표시
    # --largest: 기본적으로 ALL 숨김, largest만 표시
    # --largest --all-balls: ALL + largest 같이 표시
    if r is not None and xyz is not None:
        draw_all_balls = (not args.no_balls) and (not args.largest or args.all_balls)

        if draw_all_balls:
            pts = pv.PolyData(xyz)
            pts["radius"] = r
            glyph = pts.glyph(
                scale="radius",
                factor=1.0,
                geom=pv.Sphere(
                    radius=1.0,
                    theta_resolution=int(args.balls_res),
                    phi_resolution=int(args.balls_res),
                ),
            )
            p.add_mesh(
                glyph,
                color=str(args.balls_color),
                opacity=float(args.balls_opacity),
                smooth_shading=True,
                lighting=True,
                ambient=0.10,
                diffuse=0.85,
                specular=0.35,
                specular_power=40,
            )

        if args.largest:
            if c_in is None:
                raise RuntimeError("Cannot show largest ball: balls file empty or not loaded.")
            ball = pv.Sphere(
                radius=float(r_in),
                center=tuple(c_in.tolist()),
                theta_resolution=int(args.largest_res),
                phi_resolution=int(args.largest_res),
            )
            p.add_mesh(
                ball,
                opacity=float(args.largest_opacity),
                color=str(args.largest_color),
                smooth_shading=True,
                lighting=True,
            )
    else:
        if args.largest:
            raise RuntimeError("Cannot show largest ball because balls are not available (use without --no-balls).")

    # 3) Optional: circumscribing sphere (transparent surface only, NO wire)
    if args.circ:
        circ = pv.Sphere(
            radius=float(r_out),
            center=tuple(center.tolist()),
            theta_resolution=int(args.circ_res),
            phi_resolution=int(args.circ_res),
        )
        p.add_mesh(
            circ,
            opacity=float(args.circ_opacity),
            color=str(args.circ_color),
            smooth_shading=True,
            lighting=True,
        )

    if not args.no_axes:
        p.add_axes()
    if not args.no_grid:
        p.show_grid()

    p.view_isometric()

    # ---- Info ----
    if r is not None:
        print(f"[INFO] balls M={len(r)}")
        print(f"[INFO] largest r_in={r_in} (use --largest)")
    if args.circ:
        print(f"[INFO] circumsphere r_out={r_out} (use --circ)")

    p.show()


if __name__ == "__main__":
    main()