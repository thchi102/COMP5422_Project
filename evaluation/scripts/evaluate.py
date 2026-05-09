"""
Evaluator — CoT-Guided 3D Language Grounding
Pipeline: parse -> validate -> join -> enrich (NR3D) -> GT bbox -> Mode B match -> report

Usage:
    # From a run directory (expects reconstruction/ and optionally segmentation/):
    python scripts/evaluate.py --run outputs/run_001/ --scannet data/ScanNet/scans

    # Without ScanNet (metrics will be None — V0 mode):
    python scripts/evaluate.py --run outputs/mock_run/

    # From individual files (dev / debugging):
    python scripts/evaluate.py --recon path/recon.json
    python scripts/evaluate.py --seg path/seg.json --recon path/recon.json

Output: <run_dir>/report.json
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

try:
    from lib.gt_bbox import load_scene_gt_bboxes, load_scene_gt_point_clouds
    from lib.matching import mode_b_match, split_accuracy
    from lib.iou import box_iou_3d, scanrefer_accuracy
    from lib.chamfer import chamfer_distance, point_cloud_fscore
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

# ── schema validation (reuse validate_schema logic inline) ───────────────────

SEG_REQUIRED = {
    "schema_version": str,
    "run_id":         str,
    "sample_id":      str,
    "scene_id":       str,
    "query_id":       str,
    "object_index":   int,
    "keyframe":       (int, list),
    "object_description": str,
    "mask_sequence":  list,
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


def _validate_seg(d):
    errs = []
    for f, t in SEG_REQUIRED.items():
        if f not in d:
            errs.append(f"missing: {f}")
        elif not isinstance(d[f], t):
            errs.append(f"{f}: expected {t}, got {type(d[f]).__name__}")
    if d.get("schema_version") != "v1":
        errs.append(f"schema_version must be v1, got {d.get('schema_version')!r}")
    expected_sid = f"{d.get('scene_id')}__{d.get('query_id')}"
    if d.get("sample_id") != expected_sid:
        errs.append(f"sample_id mismatch: {d.get('sample_id')!r} != {expected_sid!r}")
    for i, m in enumerate(d.get("mask_sequence", [])):
        if not isinstance(m, dict) or ("mask_rle" not in m and "mask_path" not in m):
            errs.append(f"mask_sequence[{i}] needs mask_rle or mask_path")
    fsc = d.get("frame_sampling_config", {})
    if isinstance(fsc, dict) and fsc.get("frame_skip") != 20:
        errs.append(f"frame_skip={fsc.get('frame_skip')} (expected 20 — confirm if intentional)")
    return errs


def _validate_recon(d):
    errs = []
    for f, t in RECON_REQUIRED.items():
        if f not in d:
            errs.append(f"missing: {f}")
        elif not isinstance(d[f], t):
            errs.append(f"{f}: expected {t}, got {type(d[f]).__name__}")
    if d.get("schema_version") != "v1":
        errs.append(f"schema_version must be v1, got {d.get('schema_version')!r}")
    expected_sid = f"{d.get('scene_id')}__{d.get('query_id')}"
    if d.get("sample_id") != expected_sid:
        errs.append(f"sample_id mismatch: {d.get('sample_id')!r} != {expected_sid!r}")
    bbox = d.get("predicted_bbox_3d", {})
    if isinstance(bbox, dict):
        c, s = bbox.get("center", []), bbox.get("size", [])
        if len(c) != 3:
            errs.append(f"bbox.center must have 3 elements, got {len(c)}")
        if len(s) != 3 or any(v <= 0 for v in s if isinstance(v, (int, float))):
            errs.append("bbox.size must be 3 positive numbers")
    if d.get("coordinate_frame") != "scannet_world":
        errs.append(f"coordinate_frame must be scannet_world, got {d.get('coordinate_frame')!r}")
    return errs


# ── NR3D loader ──────────────────────────────────────────────────────────────

def load_nr3d(csv_path: Path) -> dict:
    """Returns dict keyed by query_id (assignmentid as string)."""
    nr3d = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parts = row["stimulus_id"].split("-")
            count = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            nr3d[row["assignmentid"]] = {
                "query_id":          row["assignmentid"],
                "scene_id":          row["scan_id"],
                "utterance":         row["utterance"],
                "target_object_id":  int(row["target_id"]),
                "instance_type":     row["instance_type"],
                "easy_hard":         "easy" if count == 2 else "hard",
                "n_distractors":     count - 1,
                "view_dependent":    row["uses_spatial_lang"] == "True",   # proxy
                "scene_discoverable": None,                                 # not in CSV
            }
    return nr3d


# ── discovery: find seg/recon pairs in a run dir ─────────────────────────────

def discover_pairs(run_dir: Path):
    """
    Expects run_dir/reconstruction/*.json and optionally run_dir/segmentation/*.json.
    Matches by filename (sample_id.json).
    Returns list of (seg_path, recon_path, sample_id).
    """
    seg_dir   = run_dir / "segmentation"
    recon_dir = run_dir / "reconstruction"

    if not recon_dir.exists():
        raise FileNotFoundError(f"reconstruction/ not found in {run_dir}")

    seg_files = {
        p.stem: p
        for p in seg_dir.rglob("*.json")
        if not p.name.startswith("_")
    } if seg_dir.exists() else {}
    recon_files = {
        p.stem: p
        for p in recon_dir.rglob("*.json")
        if not p.name.startswith("_")
    }

    all_ids = sorted(set(seg_files) | set(recon_files))
    pairs = []
    for sid in all_ids:
        pairs.append((
            seg_files.get(sid),
            recon_files.get(sid),
            sid,
        ))
    return pairs


def _placeholder_seg_from_recon(recon_data):
    """Create a schema-compatible placeholder when segmentation output is absent."""
    return {
        "schema_version": "v1",
        "run_id": recon_data.get("run_id", "reconstruction_only"),
        "sample_id": recon_data["sample_id"],
        "scene_id": recon_data["scene_id"],
        "query_id": recon_data["query_id"],
        "object_index": -1,
        "keyframe": [],
        "object_description": "",
        "mask_sequence": [],
        "frame_sampling_config": {
            "frame_skip": 20,
            "total_frames": None,
            "source": "reconstruction_only_placeholder",
        },
    }


# ── core evaluation logic ─────────────────────────────────────────────────────

def evaluate_pair(seg_data, recon_data, nr3d_meta, scene_bboxes=None, scene_pcs=None):
    """
    Parse + join + C1 (Mode B + ScanRefer IoU) + C2 (Chamfer) matching.
    """
    predicted_center = recon_data["predicted_bbox_3d"]["center"]
    predicted_size   = recon_data["predicted_bbox_3d"]["size"]
    target_object_id = nr3d_meta["target_object_id"] if nr3d_meta else None

    # ── C1a: Mode B matching (NR3D grounding accuracy)
    if scene_bboxes is not None and target_object_id is not None and _METRICS_AVAILABLE:
        match = mode_b_match(predicted_center, target_object_id, scene_bboxes)
        grounding_correct  = match["grounding_correct"]
        gt_bbox_center     = match["target_bbox_center"]
        bbox_center_dist   = match["bbox_center_dist"]
        matched_object_id  = match["matched_object_id"]
        matched_label      = match["matched_label"]
        target_bbox_dist   = match["target_bbox_dist"]
    else:
        grounding_correct = matched_object_id = matched_label = None
        gt_bbox_center = bbox_center_dist = target_bbox_dist = None

    # ── C1b: ScanRefer IoU (Acc@0.25 and Acc@0.5)
    iou_025 = iou_050 = iou_val = None
    if scene_bboxes is not None and target_object_id is not None and _METRICS_AVAILABLE:
        gt_bbox = scene_bboxes.get(target_object_id)
        if gt_bbox is not None:
            iou_val = box_iou_3d(predicted_center, predicted_size,
                                 gt_bbox["center"],  gt_bbox["size"])
            iou_025 = iou_val >= 0.25
            iou_050 = iou_val >= 0.50

    # ── C2: Reconstruction fidelity (Chamfer + F-score)
    cd = pc_f = None
    pred_pc = recon_data.get("predicted_point_cloud")
    if pred_pc is not None and scene_pcs is not None and target_object_id is not None and _METRICS_AVAILABLE:
        gt_pc = scene_pcs.get(target_object_id)
        if gt_pc is not None and len(pred_pc) > 0:
            cd   = chamfer_distance(pred_pc, gt_pc)
            pc_f = point_cloud_fscore(pred_pc, gt_pc, threshold=0.05)

    return {
        "sample_id":          seg_data["sample_id"],
        "scene_id":           seg_data["scene_id"],
        "query_id":           seg_data["query_id"],
        # query metadata
        "utterance":          nr3d_meta["utterance"]          if nr3d_meta else None,
        "target_object_id":   target_object_id,
        "instance_type":      nr3d_meta["instance_type"]      if nr3d_meta else None,
        "easy_hard":          nr3d_meta["easy_hard"]           if nr3d_meta else None,
        "n_distractors":      nr3d_meta["n_distractors"]       if nr3d_meta else None,
        "view_dependent":     nr3d_meta["view_dependent"]      if nr3d_meta else None,
        "scene_discoverable": nr3d_meta["scene_discoverable"]  if nr3d_meta else None,
        # segmentation fields
        "keyframe":           seg_data.get("keyframe"),
        "object_index":       seg_data.get("object_index"),
        "object_description": seg_data.get("object_description"),
        "n_mask_frames":      len(seg_data.get("mask_sequence", [])),
        # reconstruction fields
        "predicted_bbox_center": predicted_center,
        "predicted_bbox_size":   predicted_size,
        "has_point_cloud":       pred_pc is not None,
        # C1a — Mode B NR3D grounding accuracy
        "grounding_correct":  grounding_correct,
        "matched_object_id":  matched_object_id,
        "matched_label":      matched_label,
        "gt_bbox_center":     gt_bbox_center,
        "bbox_center_dist":   bbox_center_dist,
        "target_bbox_dist":   target_bbox_dist,
        # C1b — ScanRefer IoU
        "iou_with_gt":        iou_val,
        "correct_at_025":     iou_025,
        "correct_at_050":     iou_050,
        # C2 — Object-level reconstruction fidelity
        "chamfer_distance":   cd,
        "pc_fscore":          pc_f,
        # failure taxonomy
        "failure_category":   None,
    }


def compute_summary(results):
    """Compute full summary: C1 (NR3D + ScanRefer) + C2 (Chamfer) + baselines."""
    if not _METRICS_AVAILABLE:
        return {"n_total": len(results), "overall_accuracy": None,
                "note": "metrics lib not available"}

    acc = split_accuracy(results)

    # ScanRefer Acc@0.25 / Acc@0.5
    scored_iou = [r for r in results if r.get("iou_with_gt") is not None]
    n_iou = len(scored_iou)
    acc_025 = round(sum(1 for r in scored_iou if r["correct_at_025"]) / n_iou, 4) if n_iou else None
    acc_050 = round(sum(1 for r in scored_iou if r["correct_at_050"]) / n_iou, 4) if n_iou else None

    # C2 — Chamfer distance summary
    cd_vals = [r["chamfer_distance"] for r in results if r.get("chamfer_distance") is not None]
    pc_f_vals = [r["pc_fscore"]["fscore"] for r in results
                 if r.get("pc_fscore") is not None and r["pc_fscore"].get("fscore") is not None]
    mean_cd = round(sum(cd_vals) / len(cd_vals), 6)        if cd_vals   else None
    mean_pf = round(sum(pc_f_vals) / len(pc_f_vals), 4)   if pc_f_vals else None

    return {
        # C1a — NR3D Mode B
        "n_total":            acc["n_overall"],
        "n_easy":             acc["n_easy"],
        "n_hard":             acc["n_hard"],
        "n_view_dependent":   acc["n_vd"],
        "n_view_independent": acc["n_vid"],
        "overall_accuracy":   acc["overall"],
        "easy_accuracy":      acc["easy"],
        "hard_accuracy":      acc["hard"],
        "vd_accuracy":        acc["view_dependent"],
        "vid_accuracy":       acc["view_independent"],
        # C1b — ScanRefer IoU
        "n_iou_scored":       n_iou,
        "acc_at_025":         acc_025,
        "acc_at_050":         acc_050,
        # C2 — Reconstruction fidelity
        "n_c2_scored":        len(cd_vals),
        "mean_chamfer_dist":  mean_cd,
        "mean_pc_fscore":     mean_pf,
        # baselines
        "baselines":          acc["baselines"],
    }


# ── main ──────────────────────────────────────────────────────────────────────

def _fmt_pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def _fmt_float(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Evaluator — CoT 3D Language Grounding")
    parser.add_argument("--run",     help="Run directory (expects reconstruction/ and optionally segmentation/)")
    parser.add_argument("--seg",     help="Single segmentation JSON (dev mode, optional when --recon is provided)")
    parser.add_argument("--recon",   help="Single reconstruction JSON (dev mode)")
    parser.add_argument("--nr3d",    default=str(ROOT / "data/ReferIt3D/nr3d.csv"),
                        help="Path to NR3D CSV (default: data/ReferIt3D/nr3d.csv)")
    parser.add_argument("--scannet", default=str(ROOT / "data/ScanNet/scans"),
                        help="Path to ScanNet scans dir for GT bboxes (default: data/ScanNet/scans)")
    parser.add_argument("--out",     help="Output report path (default: <run_dir>/report.json)")
    args = parser.parse_args()

    if not args.run and not args.recon:
        parser.print_help()
        sys.exit(1)

    # ── load NR3D
    nr3d_path = Path(args.nr3d)
    if not nr3d_path.exists():
        print(f"[ERROR] NR3D CSV not found: {nr3d_path}")
        sys.exit(1)
    print(f"Loading NR3D from {nr3d_path} ...", end=" ", flush=True)
    nr3d = load_nr3d(nr3d_path)
    print(f"{len(nr3d):,} queries loaded")

    # ── set up GT bbox loader
    scannet_scans = Path(args.scannet)
    use_gt = _METRICS_AVAILABLE and scannet_scans.exists()
    if use_gt:
        print(f"GT bboxes: loading from {scannet_scans}")
    else:
        print("GT bboxes: not available (metrics will be None)")
    scene_bbox_cache = {}    # scene_id -> bboxes dict, loaded lazily
    scene_pc_cache   = {}    # scene_id -> point_clouds dict, loaded lazily

    # ── discover pairs
    if args.run:
        run_dir = Path(args.run)
        pairs = discover_pairs(run_dir)
        run_id = run_dir.name
    else:
        seg_path   = Path(args.seg) if args.seg else None
        recon_path = Path(args.recon)
        sample_id  = recon_path.stem
        pairs      = [(seg_path, recon_path, sample_id)]
        run_id     = "single_pair"
        run_dir    = recon_path.parent

    print(f"Found {len(pairs)} sample(s) to evaluate")

    # ── evaluate each pair
    results          = []
    n_valid          = 0
    n_seg_missing    = 0
    n_recon_missing  = 0
    n_nr3d_missing   = 0
    n_invalid        = 0

    for seg_path, recon_path, sample_id in pairs:
        row_errors = []

        # check files exist
        if recon_path is None:
            print(f"  [SKIP] {sample_id}: reconstruction file missing")
            n_recon_missing += 1
            continue

        # parse JSON
        try:
            recon_data = json.loads(recon_path.read_text(encoding="utf-8"))
            if seg_path is None:
                seg_data = _placeholder_seg_from_recon(recon_data)
                n_seg_missing += 1
            else:
                seg_data = json.loads(seg_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [SKIP] {sample_id}: JSON parse error: {e}")
            n_invalid += 1
            continue

        # validate
        seg_errs   = _validate_seg(seg_data)
        recon_errs = _validate_recon(recon_data)
        row_errors = seg_errs + recon_errs

        if row_errors:
            print(f"  [WARN] {sample_id}: {len(row_errors)} validation error(s)")
            for e in row_errors:
                print(f"         [x] {e}")
            n_invalid += 1
            # still continue — soft failure for V0 (log errors but don't skip)

        # check sample_id consistency between packages
        if seg_data.get("sample_id") != recon_data.get("sample_id"):
            print(f"  [WARN] {sample_id}: sample_id mismatch between seg and recon packages")

        # enrich from NR3D
        query_id  = seg_data.get("query_id", "")
        nr3d_meta = nr3d.get(query_id)
        if nr3d_meta is None:
            print(f"  [WARN] {sample_id}: query_id {query_id!r} not found in NR3D CSV")
            n_nr3d_missing += 1

        # load GT bboxes for scene (cached)
        scene_id = seg_data.get("scene_id", "")
        scene_bboxes = None
        scene_pcs    = None
        if use_gt and scene_id:
            if scene_id not in scene_bbox_cache:
                try:
                    scene_bbox_cache[scene_id] = load_scene_gt_bboxes(scene_id, scannet_scans)
                    print(f"  [GT] {scene_id}: {len(scene_bbox_cache[scene_id])} objects loaded")
                except FileNotFoundError as e:
                    print(f"  [WARN] {scene_id}: GT bboxes unavailable — {e}")
                    scene_bbox_cache[scene_id] = None
            scene_bboxes = scene_bbox_cache[scene_id]

            # load GT point clouds lazily (only if predicted_point_cloud is present)
            pred_pc = recon_data.get("predicted_point_cloud")
            if pred_pc is not None:
                if scene_id not in scene_pc_cache:
                    try:
                        scene_pc_cache[scene_id] = load_scene_gt_point_clouds(
                            scene_id, scannet_scans)
                    except FileNotFoundError:
                        scene_pc_cache[scene_id] = None
                scene_pcs = scene_pc_cache[scene_id]

        # evaluate
        result = evaluate_pair(seg_data, recon_data, nr3d_meta, scene_bboxes, scene_pcs)
        result["validation_errors"] = row_errors
        results.append(result)
        n_valid += 1

    # ── build report
    summary = compute_summary(results)

    report = {
        "evaluator_version": "v0",
        "schema_version":    "v1",
        "run_id":            run_id,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "nr3d_csv":          str(nr3d_path),
        "n_pairs_found":     len(pairs),
        "n_valid":           n_valid,
        "n_seg_missing":     n_seg_missing,
        "n_recon_missing":   n_recon_missing,
        "n_nr3d_missing":    n_nr3d_missing,
        "n_invalid":         n_invalid,
        "summary":           summary,
        "results":           results,
    }

    # ── write report
    if args.out:
        out_path = Path(args.out)
    elif args.run:
        out_path = Path(args.run) / "report.json"
    else:
        out_path = run_dir / f"report_{run_id}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ── print summary
    print()
    n_parse_errors = n_invalid - sum(1 for r in results if r["validation_errors"])
    print(f"Results: {n_valid} evaluated  |  "
          f"{sum(1 for r in results if r['validation_errors'])} schema warnings  |  "
          f"{n_parse_errors} parse errors  |  "
          f"{n_seg_missing + n_recon_missing} skipped")

    acc = summary.get("overall_accuracy")
    if acc is not None:
        print(f"\n--- C1a: NR3D Grounding Accuracy (Mode B) ---")
        print(f"  overall={_fmt_pct(acc)}  easy={_fmt_pct(summary['easy_accuracy'])}  "
              f"hard={_fmt_pct(summary['hard_accuracy'])}  "
              f"VD={_fmt_pct(summary['vd_accuracy'])}  VID={_fmt_pct(summary['vid_accuracy'])}")
        print(f"  split: easy={summary['n_easy']}  hard={summary['n_hard']}  "
              f"VD={summary['n_view_dependent']}  VID={summary['n_view_independent']}")
        baselines = summary.get("baselines", {})
        vlm = baselines.get("vlm_grounder_zero_shot", {})
        ref = baselines.get("referit3dnet_supervised", {})
        if vlm.get("overall"):
            print(f"  vs VLM-Grounder (zero-shot):   overall={vlm['overall']:.1%}  "
                  f"easy={vlm['easy']:.1%}  hard={vlm['hard']:.1%}")
        if ref.get("overall"):
            print(f"  vs ReferIt3DNet (supervised):  overall={ref['overall']:.1%}")
    else:
        print("C1a: no GT bboxes — run with --scannet to enable")

    acc025 = summary.get("acc_at_025")
    if acc025 is not None:
        print(f"\n--- C1b: ScanRefer IoU ---")
        print(f"  Acc@0.25={_fmt_pct(acc025)}  Acc@0.50={_fmt_pct(summary['acc_at_050'])}  "
              f"(n={summary['n_iou_scored']})")

    cd = summary.get("mean_chamfer_dist")
    if cd is not None:
        print(f"\n--- C2: Object-Level Reconstruction Fidelity ---")
        print(f"  mean Chamfer dist={_fmt_float(cd, 4)} m  mean F-score@5cm={_fmt_float(summary['mean_pc_fscore'], 3)}  "
              f"(n={summary['n_c2_scored']})")

    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
