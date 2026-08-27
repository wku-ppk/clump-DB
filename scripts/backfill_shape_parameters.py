#!/usr/bin/env python3
"""Backfill measured 3D and projection shape records in existing cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh

from clumpgen.shape import (
    calculate_projection_shape_parameters,
    calculate_shape_parameters,
)


def write_json_atomic(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add measured source-mesh shape_parameters and projection_shape "
            "records to existing clump-DB cases."
        )
    )
    parser.add_argument("--root", default="dataset/shapes")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--taper-q-over-L", type=float, default=0.5)
    parser.add_argument("--asymmetry-voxel-resolution", type=int, default=48)
    parser.add_argument("--roundness-projection-samples", type=int, default=512)
    parser.add_argument("--roundness-smoothing-fraction", type=float, default=0.015)
    parser.add_argument("--projection-orientations", type=int, default=64)
    parser.add_argument("--projection-resolution", type=int, default=512)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(f"case root not found: {root}")

    requested = set(args.case_id)
    case_directories = sorted(path for path in root.iterdir() if path.is_dir())
    if requested:
        case_directories = [path for path in case_directories if path.name in requested]
        missing = sorted(requested - {path.name for path in case_directories})
        if missing:
            raise FileNotFoundError(f"case IDs not found under {root}: {', '.join(missing)}")

    updated = skipped = failed = 0
    for case_directory in case_directories:
        stl_path = case_directory / "shape.stl"
        meta_path = case_directory / "meta.json"
        if not stl_path.is_file() or not meta_path.is_file():
            print(f"[SKIP] {case_directory.name}: missing shape.stl or meta.json")
            skipped += 1
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            measure_shape = args.overwrite or "shape_parameters" not in meta
            measure_projection = args.overwrite or "projection_shape" not in meta
            if not measure_shape and not measure_projection:
                print(
                    f"[SKIP] {case_directory.name}: shape_parameters and "
                    "projection_shape already present"
                )
                skipped += 1
                continue

            mesh = trimesh.load_mesh(stl_path, process=True)
            descriptor = (
                calculate_shape_parameters(
                    mesh,
                    taper_q_over_L=args.taper_q_over_L,
                    asymmetry_voxel_resolution=args.asymmetry_voxel_resolution,
                    roundness_projection_samples=args.roundness_projection_samples,
                    roundness_smoothing_fraction=args.roundness_smoothing_fraction,
                )
                if measure_shape
                else meta["shape_parameters"]
            )
            projection_shape = (
                calculate_projection_shape_parameters(
                    mesh,
                    orientation_count=args.projection_orientations,
                    resolution=args.projection_resolution,
                )
                if measure_projection
                else meta["projection_shape"]
            )
            if args.dry_run:
                print(
                    f"[DRY-RUN] {case_directory.name}: "
                    + json.dumps(
                        {
                            "shape_parameters": {
                                key: descriptor[key]
                                for key in descriptor["vector_order"]
                            },
                            "projection_shape": {
                                key: projection_shape[key]
                                for key in ("AR", "C_x", "S", "SAGI", "SAGI_class")
                            },
                        }
                    )
                )
            else:
                if measure_shape:
                    meta["shape_parameters"] = descriptor
                if measure_projection:
                    meta["projection_shape"] = projection_shape
                write_json_atomic(meta_path, meta)
                print(f"[OK] {case_directory.name}: updated {meta_path}")
            updated += 1
        except Exception as error:
            print(f"[ERROR] {case_directory.name}: {error}")
            failed += 1

    print(f"[DONE] measured={updated} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
