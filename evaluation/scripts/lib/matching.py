"""
Mode B matching for NR3D evaluation.

Protocol (same as VLM-Grounder):
  - Given a predicted 3D bbox center, find the nearest GT bbox center
    among ALL objects in the scene (by Euclidean distance).
  - grounding_correct = True iff the nearest GT bbox belongs to target_object_id.

Reference: VLM-Grounder evaluation uses gt_bbox_distance matching (Mode B).
No GT proposals given to the model — purely predicted bbox vs all GT bboxes.
"""

import math
from typing import Optional


def mode_b_match(
    predicted_center: list[float],
    target_object_id: int,
    scene_bboxes: dict,
) -> dict:
    """
    Run Mode B matching for one query.

    Args:
        predicted_center:  [x, y, z] predicted bbox center in scannet_world
        target_object_id:  ground-truth object ID from NR3D
        scene_bboxes:      dict from load_scene_gt_bboxes (objectId -> bbox info)

    Returns dict:
        grounding_correct:  bool — True if nearest GT box is target_object_id
        matched_object_id:  int  — objectId of the nearest GT box
        matched_label:      str  — label of the nearest GT box
        bbox_center_dist:   float — Euclidean distance (meters) to nearest GT center
        target_bbox_center: list[float] | None — GT center of the target object
        target_bbox_dist:   float | None — distance to target GT center (may != bbox_center_dist)
    """
    if not scene_bboxes:
        return {
            "grounding_correct": None,
            "matched_object_id": None,
            "matched_label": None,
            "bbox_center_dist": None,
            "target_bbox_center": None,
            "target_bbox_dist": None,
            "error": "scene_bboxes is empty",
        }

    px, py, pz = predicted_center

    best_dist = float("inf")
    best_id   = None

    for obj_id, bbox in scene_bboxes.items():
        cx, cy, cz = bbox["center"]
        dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_id   = obj_id

    grounding_correct = (best_id == target_object_id)

    target_bbox = scene_bboxes.get(target_object_id)
    if target_bbox:
        tx, ty, tz = target_bbox["center"]
        target_dist = math.sqrt((px - tx) ** 2 + (py - ty) ** 2 + (pz - tz) ** 2)
        target_center = target_bbox["center"]
    else:
        target_dist   = None
        target_center = None

    return {
        "grounding_correct":  grounding_correct,
        "matched_object_id":  best_id,
        "matched_label":      scene_bboxes[best_id]["label"] if best_id is not None else None,
        "bbox_center_dist":   round(best_dist, 4),
        "target_bbox_center": target_center,
        "target_bbox_dist":   round(target_dist, 4) if target_dist is not None else None,
    }


def compute_accuracy(results: list[dict], key: str = "grounding_correct") -> Optional[float]:
    """Mean of grounding_correct over results where value is not None."""
    scored = [r[key] for r in results if r.get(key) is not None]
    if not scored:
        return None
    return round(sum(scored) / len(scored), 4)


def split_accuracy(results: list[dict]) -> dict:
    """
    Compute accuracy for all NR3D splits.
    Returns dict with overall + Easy/Hard + VD/VID breakdowns.
    Baseline targets: VLM-Grounder 48.0% overall (zero-shot), ReferIt3DNet 35.6% (supervised).
    """
    easy = [r for r in results if r.get("easy_hard") == "easy"]
    hard = [r for r in results if r.get("easy_hard") == "hard"]
    vd   = [r for r in results if r.get("view_dependent") is True]
    vid  = [r for r in results if r.get("view_dependent") is False]

    return {
        "overall":        compute_accuracy(results),
        "easy":           compute_accuracy(easy),
        "hard":           compute_accuracy(hard),
        "view_dependent": compute_accuracy(vd),
        "view_independent": compute_accuracy(vid),
        "n_overall": len([r for r in results if r.get("grounding_correct") is not None]),
        "n_easy":    len([r for r in easy if r.get("grounding_correct") is not None]),
        "n_hard":    len([r for r in hard if r.get("grounding_correct") is not None]),
        "n_vd":      len([r for r in vd   if r.get("grounding_correct") is not None]),
        "n_vid":     len([r for r in vid  if r.get("grounding_correct") is not None]),
        "baselines": {
            # Paper-reported numbers on full NR3D test set.
            # VLM-Grounder must also be run on dev-mini for fair per-query comparison.
            "referit3dnet_supervised":  {"overall": 0.356, "easy": 0.436, "hard": 0.279,
                                         "view_dependent": 0.325, "view_independent": 0.371},
            "vlm_grounder_zero_shot":   {"overall": 0.480, "easy": 0.552, "hard": 0.395,
                                         "view_dependent": 0.458, "view_independent": 0.494},
            "seeground_zero_shot":      {"overall": None, "note": "CVPR 2025 — +7.1% over VLM-Grounder on NR3D; exact split numbers TBD"},
            "z3d_zero_shot":            {"overall": None, "note": "arxiv Feb 2026 — new SOTA zero-shot; exact numbers TBD"},
        },
    }
