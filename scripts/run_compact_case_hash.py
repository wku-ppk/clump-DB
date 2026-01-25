#python scripts/run_compact_case_hash.py \
#  --L 1 --e 0.75 --f 0.65 --randomness 0.3 --seed 1234 \
#  --N 30 --rMin 0.0 --div 100 --overlap 0.7 --rMax_ratio 1.0 \
#  --resume

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from CLUMP import GenerateClump_Euclidean_3D
from clumpgen.id import make_hash_id


def spheres_from_clump_object(clump):
    if hasattr(clump, "positions") and hasattr(clump, "radii"):
        xyz = np.asarray(clump.positions, dtype=float)
        r = np.asarray(clump.radii, dtype=float).reshape(-1)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"clump.positions shape unexpected: {xyz.shape}")
        if r.shape[0] != xyz.shape[0]:
            raise ValueError(f"positions/radii length mismatch: {xyz.shape[0]} vs {r.shape[0]}")
        return xyz, r
    raise AttributeError("This CLUMP build does not expose (positions, radii).")


def wadell_roundness_from_radii(r):
    r = np.asarray(r, dtype=float).reshape(-1)
    r_in = float(r.max())
    R1 = float((r / r_in).mean())
    R2 = float(1.0 / (r_in / r).mean())
    return {"R1": R1, "R2": R2, "r_in": r_in, "D_in": 2.0 * r_in}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset/shapes", help="root output dir")
    ap.add_argument("--manifest", default="dataset/manifest.jsonl", help="jsonl index")
    ap.add_argument("--params", default="", help="optional params.json (run_compact style)")
    ap.add_argument("--resume", action="store_true", help="skip if meta.json already exists")

    # geometry (ellipsoid-ish)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--e", type=float, default=0.75)        # I/L
    ap.add_argument("--f", type=float, default=0.65)        # S/I
    ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--randomness", type=float, default=0.18)
    ap.add_argument("--bias", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1234)

    # clump
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--rMin", type=float, default=0.5)
    ap.add_argument("--div", type=int, default=120)
    ap.add_argument("--overlap", type=float, default=0.6)
    ap.add_argument("--rMax_ratio", type=float, default=0.3)
    ap.add_argument("--visualise", action="store_true")

    args = ap.parse_args()

    # ---- optional params.json override ----
    if args.params:
        cfg = json.loads(Path(args.params).read_text(encoding="utf-8"))
        g = cfg.get("geometry", {})
        c = cfg.get("clump", {})
        args.L = float(g.get("L", args.L))
        args.e = float(g.get("I_over_L", args.e))
        args.f = float(g.get("S_over_I", args.f))
        args.sub = int(g.get("subdivisions", args.sub))
        args.randomness = float(g.get("randomness", args.randomness))
        args.bias = float(g.get("bias", args.bias))
        args.seed = int(g.get("seed", args.seed))

        args.N = int(c.get("N", args.N))
        args.rMin = float(c.get("rMin", args.rMin))
        args.div = int(c.get("div", args.div))
        args.overlap = float(c.get("overlap", args.overlap))
        args.rMax_ratio = float(c.get("rMax_ratio", args.rMax_ratio))
        args.visualise = bool(c.get("visualise", args.visualise))

    # ---- build canonical payloads for IDs ----
    shape_payload = {
        "algo_version": "shape_v1",
        "L": float(args.L),
        "e": float(args.e),
        "f": float(args.f),
        "subdivisions": int(args.sub),
        "randomness": float(args.randomness),
        "bias": float(args.bias),
        "seed": int(args.seed),
    }
    shape_id = make_hash_id(shape_payload, n=12)

    clump_payload = {
        "algo_version": "clump_v1",
        "shape_id": shape_id,
        "N": int(args.N),
        "rMin": float(args.rMin),
        "div": int(args.div),
        "overlap": float(args.overlap),
        "rMax_ratio": float(args.rMax_ratio),
        "visualise": bool(args.visualise),
    }
    case_id = make_hash_id(clump_payload, n=12)

    dataset_dir = Path(args.dataset_dir)
    case_dir = dataset_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    stl_path = case_dir / "shape.stl"
    out_txt = case_dir / "clump_output.txt"
    out_vtk = case_dir / "clump_output.vtk"
    balls_txt = case_dir / "balls_xyzr.txt"
    meta_path = case_dir / "meta.json"
    manifest_path = Path(args.manifest)

    if args.resume and meta_path.exists():
        print(f"[SKIP] {case_id} already done: {meta_path}")
        return

    # ---- 1) make irregular ellipsoid STL ----
    L = float(args.L)
    I = float(args.e) * L
    S = float(args.f) * I

    m = trimesh.creation.icosphere(subdivisions=int(args.sub), radius=1.0)
    v = m.vertices * np.array([L, I, S], dtype=float)
    u = np.random.default_rng(int(args.seed)).random(len(v)) ** float(args.bias)
    m.vertices = v * (1.0 - float(args.randomness) * u)[:, None]  # inward-only
    m.export(stl_path.as_posix())

    # ---- 2) CLUMP ----
    mesh, clump = GenerateClump_Euclidean_3D(
        stl_path.as_posix(),
        int(args.N),
        float(args.rMin),
        int(args.div),
        float(args.overlap),
        output=out_txt.as_posix(),
        outputVTK=out_vtk.as_posix(),
        visualise=bool(args.visualise),
        rMax_ratio=float(args.rMax_ratio),
    )

    xyz, r = spheres_from_clump_object(clump)
    np.savetxt(balls_txt.as_posix(), np.column_stack([xyz, r]), fmt="%.8g")

    # ---- 3) metrics ----
    wadell = wadell_roundness_from_radii(r)
    center, r_out = trimesh.nsphere.minimum_nsphere(mesh)
    D_out = 2.0 * float(r_out)
    sphericity = float(np.sqrt(abs(wadell["D_in"] / D_out)))

    # ---- 4) meta.json + manifest.jsonl ----
    meta = {
        "case_id": case_id,
        "shape_id": shape_id,
        "paths": {
            "stl": stl_path.as_posix(),
            "clump_output_txt": out_txt.as_posix(),
            "clump_output_vtk": out_vtk.as_posix(),
            "balls_xyzr": balls_txt.as_posix(),
            "meta": meta_path.as_posix(),
        },
        "shape_params": shape_payload,
        "clump_params": clump_payload,
        "mesh_stats": {
            "watertight": bool(m.is_watertight),
            "n_vertices": int(len(m.vertices)),
            "n_faces": int(len(m.faces)),
        },
        "metrics": {
            "wadell": wadell,
            "circumsphere": {
                "center": [float(x) for x in center],
                "r_out": float(r_out),
                "D_out": float(D_out),
            },
            "sphericity_riley1941_style": float(sphericity),
        },
        "stage": "clump_done",
    }

    write_json(meta_path, meta)
    append_jsonl(manifest_path, meta)

    print("[OK] case_id =", case_id, "shape_id =", shape_id)
    print("[OK] STL:", stl_path, "watertight=", m.is_watertight)
    print("[OK] CLUMP:", out_txt, out_vtk)
    print("[OK] balls:", balls_txt, "M=", len(r))
    print("[OK] meta:", meta_path)
    print("[INFO] sphericity =", sphericity)


if __name__ == "__main__":
    main()