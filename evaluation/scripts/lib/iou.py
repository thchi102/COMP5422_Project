"""
3D axis-aligned bounding box IoU for ScanRefer evaluation.

ScanRefer metric: Acc@0.25 and Acc@0.5 — fraction of predictions
where IoU(predicted box, GT box) >= threshold.
"""


def _box_to_minmax(center, size):
    """Convert [cx,cy,cz] + [dx,dy,dz] to (min, max) corners."""
    cx, cy, cz = center
    dx, dy, dz = size
    return (
        (cx - dx / 2, cy - dy / 2, cz - dz / 2),
        (cx + dx / 2, cy + dy / 2, cz + dz / 2),
    )


def box_iou_3d(pred_center, pred_size, gt_center, gt_size) -> float:
    """
    Compute IoU between two axis-aligned 3D boxes.

    Args:
        pred_center: [cx, cy, cz]
        pred_size:   [dx, dy, dz]  (positive)
        gt_center:   [cx, cy, cz]
        gt_size:     [dx, dy, dz]  (positive)

    Returns:
        IoU in [0, 1]
    """
    (p_min, p_max) = _box_to_minmax(pred_center, pred_size)
    (g_min, g_max) = _box_to_minmax(gt_center,   gt_size)

    # Intersection
    inter_min = tuple(max(p_min[i], g_min[i]) for i in range(3))
    inter_max = tuple(min(p_max[i], g_max[i]) for i in range(3))

    inter_dims = tuple(max(0.0, inter_max[i] - inter_min[i]) for i in range(3))
    inter_vol  = inter_dims[0] * inter_dims[1] * inter_dims[2]

    if inter_vol == 0:
        return 0.0

    pred_vol = pred_size[0] * pred_size[1] * pred_size[2]
    gt_vol   = gt_size[0]   * gt_size[1]   * gt_size[2]
    union_vol = pred_vol + gt_vol - inter_vol

    if union_vol <= 0:
        return 0.0

    return round(inter_vol / union_vol, 6)


def scanrefer_accuracy(results: list[dict], threshold: float, scene_bboxes_map: dict) -> dict:
    """
    Compute ScanRefer Acc@threshold over a list of result rows.

    Args:
        results:          list of evaluate.py result dicts
        threshold:        IoU threshold (0.25 or 0.5)
        scene_bboxes_map: dict[scene_id -> scene_bboxes dict from load_scene_gt_bboxes]

    Returns dict:
        correct:  int
        total:    int
        accuracy: float | None
    """
    correct = 0
    total   = 0

    for r in results:
        pred_center = r.get("predicted_bbox_center")
        pred_size   = r.get("predicted_bbox_size")
        scene_id    = r.get("scene_id")
        target_id   = r.get("target_object_id")

        if pred_center is None or pred_size is None:
            continue
        if scene_id is None or target_id is None:
            continue

        scene_bboxes = scene_bboxes_map.get(scene_id)
        if scene_bboxes is None:
            continue

        gt_bbox = scene_bboxes.get(target_id)
        if gt_bbox is None:
            continue

        iou = box_iou_3d(pred_center, pred_size, gt_bbox["center"], gt_bbox["size"])
        total   += 1
        if iou >= threshold:
            correct += 1

    accuracy = round(correct / total, 4) if total > 0 else None
    return {"correct": correct, "total": total, "accuracy": accuracy,
            "threshold": threshold}
