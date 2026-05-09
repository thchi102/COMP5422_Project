#!/usr/bin/env python3
"""Check whether the SSH server is ready to run VLM-Grounder on nr3d_top10."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TOP10_SCENES = [
    "scene0090_00",
    "scene0117_00",
    "scene0143_00",
    "scene0258_00",
    "scene0333_00",
    "scene0443_00",
    "scene0444_00",
    "scene0546_00",
    "scene0558_00",
    "scene0658_00",
]

SCAN_REQUIRED_SUFFIXES = [
    ".sens",
    ".aggregation.json",
    "_vh_clean_2.0.010000.segs.json",
    "_vh_clean_2.labels.ply",
    "_vh_clean_2.ply",
    ".txt",
]


def find_project_root() -> Path:
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "data" / "schema_v1.json").exists():
            return path.resolve()
    raise RuntimeError("Could not find project root. Run this from inside the project.")


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def count_files(path: Path, patterns: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    total = 0
    for pattern in patterns:
        total += len(list(path.rglob(pattern)))
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=Path("data/nr3d_top10.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/nr3d_top10_manifest.json"))
    args = parser.parse_args()

    project = (args.project_root or find_project_root()).resolve()
    csv_path = (project / args.csv).resolve() if not args.csv.is_absolute() else args.csv
    manifest_path = (project / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest

    scans_root = project / "data" / "ScanNet" / "scans"
    vg_root = project / "scripts" / "vlm-grounder-repo"
    vg_scannet = vg_root / "data" / "scannet"
    posed_root = vg_scannet / "posed_images"
    instance_info = vg_scannet / "scannet_instance_data" / "scenes_train_val_info_w_images.pkl"
    match_pkl = vg_scannet / "scannet_match_data" / "exhaustive_matching.pkl"

    failures: list[str] = []
    warnings: list[str] = []

    print(f"project: {project}")
    print()

    csv_rows = count_csv_rows(csv_path)
    if csv_rows == 100:
        print(f"[OK] nr3d_top10 csv: {csv_path} ({csv_rows} rows)")
    elif csv_rows is None:
        failures.append(f"Missing CSV: {csv_path}")
    else:
        failures.append(f"CSV row count is {csv_rows}, expected 100: {csv_path}")

    if manifest_path.exists():
        print(f"[OK] top10 manifest: {manifest_path}")
    else:
        warnings.append(f"Manifest missing; notebook will build it from CSV: {manifest_path}")

    if vg_root.exists():
        print(f"[OK] VLM-Grounder repo: {vg_root}")
    else:
        failures.append(f"Missing VLM-Grounder repo: {vg_root}")

    if instance_info.exists():
        print(f"[OK] scene info pkl: {instance_info}")
    else:
        warnings.append(f"Scene info missing; wrapper can regenerate it if one-time stages run: {instance_info}")

    if match_pkl.exists():
        print(f"[OK] matching cache: {match_pkl}")
    else:
        warnings.append(f"Matching cache missing; keep WRAPPER_SKIP_ONE_TIME=False: {match_pkl}")

    print()
    print("Scene readiness:")
    for scene in TOP10_SCENES:
        scene_dir = scans_root / scene
        missing = []
        for suffix in SCAN_REQUIRED_SUFFIXES:
            name = f"{scene}{suffix}" if suffix.startswith(".") else f"{scene}{suffix}"
            if not (scene_dir / name).exists():
                missing.append(name)

        posed_dir = posed_root / scene
        posed_count = count_files(posed_dir, ("*.jpg", "*.png"))

        scan_ok = scene_dir.exists() and not missing
        posed_ok = posed_count > 0
        status = "OK" if scan_ok and posed_ok else "BAD"
        print(f"  [{status}] {scene}: scan_files={'ok' if scan_ok else 'missing'} posed_images={posed_count}")

        if not scene_dir.exists():
            failures.append(f"Missing ScanNet scene dir: {scene_dir}")
        for name in missing:
            failures.append(f"Missing ScanNet file: {scene_dir / name}")
        if not posed_ok:
            warnings.append(f"No posed images found for {scene}: {posed_dir}")

    print()
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")
        print()

    if failures:
        print("Failures:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Ready enough to run. If matching cache is missing, run without --skip-one-time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
