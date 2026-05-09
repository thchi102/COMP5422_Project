"""
Schema v1 validator for segmentation and reconstruction packages.
Usage:
    python scripts/validate_schema.py --seg path/to/seg.json
    python scripts/validate_schema.py --recon path/to/recon.json
    python scripts/validate_schema.py --seg path/to/seg.json --recon path/to/recon.json
    python scripts/validate_schema.py --examples   # validate examples in schema_v1.json itself
"""

import argparse
import json
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent.parent / "data" / "schema_v1.json"

# ── required fields and their expected types ─────────────────────────────────

SEG_REQUIRED = {
    "schema_version":        str,
    "run_id":                str,
    "sample_id":             str,
    "scene_id":              str,
    "query_id":              str,
    "object_index":          int,
    "keyframe":              (int, list),   # TBD: may be list
    "object_description":    str,
    "mask_sequence":         list,
    "frame_sampling_config": dict,
}

RECON_REQUIRED = {
    "schema_version":    str,
    "run_id":            str,
    "sample_id":         str,
    "scene_id":          str,
    "query_id":          str,
    "predicted_bbox_3d": dict,
    "camera_intrinsics": dict,
    "depth_metadata":    dict,
    "coordinate_frame":  str,
}

RECON_BBOX_REQUIRED = {"center": list, "size": list}
RECON_CAM_REQUIRED  = {"fx": float, "fy": float, "cx": float, "cy": float,
                        "depth_width": int, "depth_height": int}
SEG_FSC_REQUIRED    = {"frame_skip": int}


def _check(errors, condition, msg):
    if not condition:
        errors.append(msg)


def validate_segmentation(data: dict) -> list[str]:
    errors = []
    for field, typ in SEG_REQUIRED.items():
        _check(errors, field in data, f"missing required field: {field}")
        if field in data:
            _check(errors, isinstance(data[field], typ),
                   f"{field}: expected {typ}, got {type(data[field]).__name__}")

    if "schema_version" in data:
        _check(errors, data["schema_version"] == "v1",
               f"schema_version must be 'v1', got '{data['schema_version']}'")

    if "sample_id" in data and "scene_id" in data and "query_id" in data:
        expected = f"{data['scene_id']}__{data['query_id']}"
        _check(errors, data["sample_id"] == expected,
               f"sample_id mismatch: got '{data['sample_id']}', expected '{expected}'")

    if isinstance(data.get("mask_sequence"), list):
        for i, entry in enumerate(data["mask_sequence"]):
            _check(errors, isinstance(entry, dict), f"mask_sequence[{i}] must be a dict")
            if isinstance(entry, dict):
                _check(errors, "frame_idx" in entry, f"mask_sequence[{i}] missing frame_idx")
                has_rle  = "mask_rle"  in entry
                has_path = "mask_path" in entry
                _check(errors, has_rle or has_path,
                       f"mask_sequence[{i}] needs mask_rle or mask_path")

    fsc = data.get("frame_sampling_config", {})
    if isinstance(fsc, dict):
        for field, typ in SEG_FSC_REQUIRED.items():
            _check(errors, field in fsc,
                   f"frame_sampling_config missing: {field}")
        if "frame_skip" in fsc:
            _check(errors, fsc["frame_skip"] == 20,
                   f"frame_skip should be 20, got {fsc.get('frame_skip')} (confirm if intentional)")

    return errors


def validate_reconstruction(data: dict) -> list[str]:
    errors = []
    for field, typ in RECON_REQUIRED.items():
        _check(errors, field in data, f"missing required field: {field}")
        if field in data:
            _check(errors, isinstance(data[field], typ),
                   f"{field}: expected {typ}, got {type(data[field]).__name__}")

    if "schema_version" in data:
        _check(errors, data["schema_version"] == "v1",
               f"schema_version must be 'v1', got '{data['schema_version']}'")

    if "sample_id" in data and "scene_id" in data and "query_id" in data:
        expected = f"{data['scene_id']}__{data['query_id']}"
        _check(errors, data["sample_id"] == expected,
               f"sample_id mismatch: got '{data['sample_id']}', expected '{expected}'")

    bbox = data.get("predicted_bbox_3d", {})
    if isinstance(bbox, dict):
        for field, typ in RECON_BBOX_REQUIRED.items():
            _check(errors, field in bbox, f"predicted_bbox_3d missing: {field}")
        if "center" in bbox:
            _check(errors, len(bbox["center"]) == 3,
                   f"predicted_bbox_3d.center must have 3 elements")
            _check(errors, all(isinstance(v, (int, float)) for v in bbox["center"]),
                   f"predicted_bbox_3d.center must be numeric")
        if "size" in bbox:
            _check(errors, len(bbox["size"]) == 3,
                   f"predicted_bbox_3d.size must have 3 elements")
            _check(errors, all(isinstance(v, (int, float)) and v > 0 for v in bbox["size"]),
                   "predicted_bbox_3d.size must be 3 positive numbers")

    cam = data.get("camera_intrinsics", {})
    if isinstance(cam, dict):
        for field, typ in RECON_CAM_REQUIRED.items():
            _check(errors, field in cam, f"camera_intrinsics missing: {field}")

    if "coordinate_frame" in data:
        _check(errors, data["coordinate_frame"] == "scannet_world",
               f"coordinate_frame must be 'scannet_world', got '{data['coordinate_frame']}'")

    pc = data.get("predicted_point_cloud")
    if pc is not None:
        _check(errors, isinstance(pc, list),
               "predicted_point_cloud must be a list of [x,y,z] or null")
        if isinstance(pc, list) and len(pc) > 0:
            _check(errors, all(isinstance(p, list) and len(p) == 3 for p in pc[:5]),
                   "predicted_point_cloud entries must be [x, y, z] lists")

    return errors


def report(label: str, errors: list[str]) -> bool:
    if errors:
        print(f"FAIL  {label}")
        for e in errors:
            print(f"      [x] {e}")
        return False
    else:
        print(f"PASS  {label}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg",      help="Path to segmentation JSON")
    parser.add_argument("--recon",    help="Path to reconstruction JSON")
    parser.add_argument("--examples", action="store_true",
                        help="Validate the built-in examples in schema_v1.json")
    args = parser.parse_args()

    if not any([args.seg, args.recon, args.examples]):
        parser.print_help()
        sys.exit(0)

    all_passed = True

    if args.examples:
        schema = json.loads(SCHEMA_PATH.read_text())
        seg_ex   = schema["segmentation_package"]["example"]
        recon_ex = schema["reconstruction_package"]["example"]
        all_passed &= report("schema_v1.json segmentation example",   validate_segmentation(seg_ex))
        all_passed &= report("schema_v1.json reconstruction example", validate_reconstruction(recon_ex))

    if args.seg:
        data = json.loads(Path(args.seg).read_text())
        all_passed &= report(args.seg, validate_segmentation(data))

    if args.recon:
        data = json.loads(Path(args.recon).read_text())
        all_passed &= report(args.recon, validate_reconstruction(data))

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
