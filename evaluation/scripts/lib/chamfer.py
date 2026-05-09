"""
Chamfer distance and F-score for C2 (Object-Level Reconstruction Fidelity).

Both metrics compare a predicted point cloud against the GT object point cloud
extracted from ScanNet's labeled mesh.

No external baseline exists for this metric — results are reported as absolute
numbers vs GT only.
"""

import math


def chamfer_distance(pred_points: list, gt_points: list) -> float:
    """
    Compute bidirectional Chamfer distance between two point clouds.

    CD = mean_{p in pred} min_{g in gt} dist(p,g)
       + mean_{g in gt}  min_{p in pred} dist(g,p)

    Args:
        pred_points: list of [x, y, z]
        gt_points:   list of [x, y, z]

    Returns:
        Chamfer distance (meters). Lower is better.
    """
    if not pred_points or not gt_points:
        return None

    def min_dist(source, target):
        total = 0.0
        for s in source:
            sx, sy, sz = s[0], s[1], s[2]
            best = float("inf")
            for t in target:
                d = math.sqrt((sx-t[0])**2 + (sy-t[1])**2 + (sz-t[2])**2)
                if d < best:
                    best = d
            total += best
        return total / len(source)

    return round(min_dist(pred_points, gt_points) + min_dist(gt_points, pred_points), 6)


def point_cloud_fscore(pred_points: list, gt_points: list, threshold: float = 0.05) -> dict:
    """
    Compute F-score between two point clouds at a distance threshold.

    Precision = fraction of pred points within threshold of any gt point.
    Recall    = fraction of gt points within threshold of any pred point.
    F-score   = 2 * precision * recall / (precision + recall)

    Args:
        pred_points: list of [x, y, z]
        gt_points:   list of [x, y, z]
        threshold:   distance threshold in meters (default 5 cm)

    Returns dict:
        precision, recall, fscore, threshold
    """
    if not pred_points or not gt_points:
        return {"precision": None, "recall": None, "fscore": None, "threshold": threshold}

    def fraction_within(source, target, thr):
        count = 0
        for s in source:
            sx, sy, sz = s[0], s[1], s[2]
            for t in target:
                d = math.sqrt((sx-t[0])**2 + (sy-t[1])**2 + (sz-t[2])**2)
                if d <= thr:
                    count += 1
                    break
        return count / len(source)

    precision = fraction_within(pred_points, gt_points, threshold)
    recall    = fraction_within(gt_points, pred_points, threshold)
    denom     = precision + recall
    fscore    = round(2 * precision * recall / denom, 4) if denom > 0 else 0.0

    return {
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "fscore":     fscore,
        "threshold":  threshold,
    }
