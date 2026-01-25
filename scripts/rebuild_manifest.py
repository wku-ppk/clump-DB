#rm -f dataset/manifest.jsonl
# (케이스 폴더도 다 지우려면) rm -rf dataset/shapes
#mkdir -p dataset/shapes
#python scripts/rebuild_manifest.py



#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CASE_FILES = [
    "shape.stl",
    "balls_xyzr.txt",
    "meta.json",
    # (optional)
    "clump_output.txt",
    "clump_output.vtk",
]


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def iso_from_mtime(mtime: float) -> str:
    # local time string
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(mtime))


def build_record(case_dir: Path, root: Path, include_files: bool, require_stl: bool, require_balls: bool) -> Optional[Dict[str, Any]]:
    case_id = case_dir.name

    stl_path = case_dir / "shape.stl"
    balls_path = case_dir / "balls_xyzr.txt"
    meta_path = case_dir / "meta.json"

    if require_stl and not stl_path.exists():
        return None
    if require_balls and not balls_path.exists():
        return None

    meta = read_json(meta_path) if meta_path.exists() else None

    # Use newest mtime among known files if possible
    mtimes = []
    for fn in DEFAULT_CASE_FILES:
        p = case_dir / fn
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    mtime = max(mtimes) if mtimes else case_dir.stat().st_mtime

    rec: Dict[str, Any] = {
        "case_id": case_id,
        "path": str(case_dir.relative_to(root).as_posix()),  # relative path under root
        "abs_path": str(case_dir.resolve().as_posix()),
        "mtime": mtime,
        "mtime_iso": iso_from_mtime(mtime),
        "has": {
            "shape_stl": stl_path.exists(),
            "balls_xyzr": balls_path.exists(),
            "meta_json": meta_path.exists(),
        },
    }

    # If meta exists, keep its useful sections (lightly normalized)
    if isinstance(meta, dict):
        # Common keys we expect from your generator
        # (If your schema differs, this still preserves raw meta under "meta")
        shape = meta.get("shape") or meta.get("geometry") or {}
        clump = meta.get("clump") or {}
        metrics = meta.get("metrics") or {}

        rec["shape"] = shape
        rec["clump"] = clump
        rec["metrics"] = metrics

        # Keep full meta as well (optional but helpful)
        rec["meta"] = meta

    if include_files:
        files = {}
        for fn in DEFAULT_CASE_FILES:
            p = case_dir / fn
            if p.exists():
                st = p.stat()
                files[fn] = {
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "mtime_iso": iso_from_mtime(st.st_mtime),
                }
        rec["files"] = files

    return rec


def main():
    ap = argparse.ArgumentParser(
        description="Rebuild dataset/manifest.jsonl by scanning case directories under dataset/shapes."
    )
    ap.add_argument("--root", default="dataset/shapes", help="Root directory containing case folders")
    ap.add_argument("--out", default="dataset/manifest.jsonl", help="Output manifest.jsonl path")
    ap.add_argument("--include-files", action="store_true", help="Include per-file size/mtime in the manifest")
    ap.add_argument("--strict", action="store_true", help="Fail on any bad case directory instead of skipping")
    ap.add_argument("--dry-run", action="store_true", help="Do not write output; just print summary")
    ap.add_argument("--require-stl", action="store_true", help="Skip cases missing shape.stl")
    ap.add_argument("--require-balls", action="store_true", help="Skip cases missing balls_xyzr.txt")
    ap.add_argument("--sort", default="mtime", choices=["mtime", "case_id"], help="Sort order for output lines")

    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    case_dirs = [p for p in root.iterdir() if p.is_dir()]
    records = []
    skipped = 0
    errors = 0

    for d in case_dirs:
        try:
            rec = build_record(
                case_dir=d,
                root=root,
                include_files=bool(args.include_files),
                require_stl=bool(args.require_stl),
                require_balls=bool(args.require_balls),
            )
            if rec is None:
                skipped += 1
                continue
            records.append(rec)
        except Exception as e:
            errors += 1
            if args.strict:
                raise
            else:
                # skip bad case
                skipped += 1

    # sort
    if args.sort == "mtime":
        records.sort(key=lambda r: (r.get("mtime", 0.0), r.get("case_id", "")), reverse=True)
    else:
        records.sort(key=lambda r: r.get("case_id", ""))

    if args.dry_run:
        print(f"[DRY-RUN] root={root} out={out}")
        print(f"[DRY-RUN] cases_total={len(case_dirs)} records={len(records)} skipped={skipped} errors={errors}")
        if records:
            print("[DRY-RUN] first record preview:")
            print(json.dumps(records[0], ensure_ascii=False, indent=2)[:2000])
        return

    out.parent.mkdir(parents=True, exist_ok=True)

    # write JSONL
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {out} (records={len(records)}, skipped={skipped}, errors={errors})")


if __name__ == "__main__":
    main()