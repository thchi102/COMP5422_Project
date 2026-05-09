from __future__ import annotations

"""
convert_manifest_to_vg_csv.py
─────────────────────────────
Convert dev_mini_manifest.json (our format) to VLM-Grounder's input CSV.

VLM-Grounder input CSV columns:
  scan_id       — ScanNet scene ID  (e.g. scene0002_00)
  utterance     — natural language query
  instance_type — object class label (e.g. "storage bin")
    target_id     — ScanNet object id for target instance
  query_id      — our NR3D assignmentid, carried through for output mapping

Usage
─────
  python scripts/convert_manifest_to_vg_csv.py \
      --manifest data/dev_mini_manifest.json   \
      --out      /tmp/vg_input.csv             \
      --scenes   scene0002_00 scene0025_00      # optional filter
"""

import argparse
import csv
import json
from pathlib import Path


def convert(manifest_path: str, output_csv_path: str, scenes: list[str] | None = None) -> int:
    # Support UTF-8 JSON with optional BOM (seen in final-test manifest files).
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        manifest = json.load(f)

    queries: list[dict] = manifest["queries"]

    if scenes:
        scenes_set = set(scenes)
        queries = [q for q in queries if q["scene_id"] in scenes_set]
        if not queries:
            raise ValueError(f"No queries found for scenes: {scenes}")

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["scan_id", "utterance", "instance_type", "target_id", "query_id"]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in queries:
            writer.writerow({
                "scan_id":       q["scene_id"],
                "utterance":     q["utterance"],
                "instance_type": q["instance_type"],
                "target_id":     q["target_id"],
                "query_id":      q["query_id"],
            })

    print(f"[convert] Wrote {len(queries)} queries → {output_csv_path}")
    return len(queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert manifest to VLM-Grounder CSV")
    parser.add_argument("--manifest", required=True, help="Path to dev_mini_manifest.json")
    parser.add_argument("--out",      required=True, help="Output CSV path")
    parser.add_argument("--scenes",   nargs="+",     help="Filter to these scene IDs only")
    args = parser.parse_args()

    convert(args.manifest, args.out, args.scenes)
