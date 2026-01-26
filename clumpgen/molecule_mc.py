# clumpgen/molecule_mc.py (기존 함수에 인자/블록 추가)

import json
from typing import Optional, Tuple
from pathlib import Path
import numpy as np

# ... (기존 코드 유지)

def compute_and_write_molecule_from_case(
    case_dir: Path,
    Gs: float,
    rho_water: float = 1000.0,
    density: Optional[float] = None,
    samples_volume: int = 200_000,
    samples_inertia: int = 200_000,
    seed: int = 1234,
    sphere_block: int = 256,
    balls_name: str = "balls_xyzr.txt",
    out_name: str = "molecule_mc.data",
    update_meta: bool = False,          # ✅ 추가
) -> Path:
    balls_path = case_dir / balls_name
    xyz, r = load_balls_xyzr(balls_path)

    dens = float(density) if density is not None else float(Gs * rho_water)
    rng = np.random.default_rng(int(seed))

    vol, com, bbox = mc_volume_and_com(
        xyz, r,
        n_samples=int(samples_volume),
        rng=rng,
        sphere_block=int(sphere_block),
    )
    if vol <= 0.0:
        raise RuntimeError("MC volume is zero. Increase samples or check balls_xyzr.txt")

    inertia = mc_inertia_about_com(
        xyz, r,
        com=com,
        volume_est=vol,
        density=dens,
        n_samples=int(samples_inertia),
        rng=rng,
        bbox=bbox,
        sphere_block=int(sphere_block),
    )
    mass = dens * vol

    out_path = case_dir / out_name
    write_molecule_file(out_path, xyz, r, mass, com, inertia, dens)

    # ✅ meta.json 업데이트
    if update_meta:
        meta_path = case_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(meta, dict):
                    meta = {}
            except Exception:
                meta = {}

        mc = meta.get("mc", {})
        if not isinstance(mc, dict):
            mc = {}

        mc.update({
            "method": "monte_carlo_union_of_spheres",
            "Gs": float(Gs),
            "rho_water": float(rho_water),
            "density": float(dens),
            "samples_volume": int(samples_volume),
            "samples_inertia": int(samples_inertia),
            "seed": int(seed),
            "sphere_block": int(sphere_block),
            "volume_est": float(vol),
            "mass_est": float(mass),
            "com": [float(com[0]), float(com[1]), float(com[2])],
            "inertia": [float(inertia[0]), float(inertia[1]), float(inertia[2]),
                        float(inertia[3]), float(inertia[4]), float(inertia[5])],
            "bbox": [float(x) for x in bbox],
            "molecule_file": out_path.name,
            "balls_file": balls_name,
        })

        meta["mc"] = mc
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_path