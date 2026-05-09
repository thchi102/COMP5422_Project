"""
GT bounding box extraction from ScanNet scene files.

For each objectId in a scene, computes an axis-aligned 3D bounding box
by joining: aggregation.json (objectId → segment IDs) +
            segs.json       (vertex index → segment ID) +
            labels.ply      (vertex index → x,y,z)

NR3D target_id maps directly to ScanNet objectId (verified on scene0002_00).
"""

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


def load_scene_gt_bboxes(scene_id: str, scannet_scans_dir: str | Path) -> dict:
    """
    Load all GT axis-aligned bboxes for a scene.

    Returns:
        dict mapping objectId (int) -> {
            "center": [x, y, z],
            "size":   [dx, dy, dz],   # full extents (positive)
            "label":  str,
            "n_vertices": int,
        }
        Returns empty dict if required files are missing.
    """
    scans_dir = Path(scannet_scans_dir)
    scene_dir = scans_dir / scene_id

    agg_path  = scene_dir / f"{scene_id}.aggregation.json"
    segs_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    ply_path  = scene_dir / f"{scene_id}_vh_clean_2.labels.ply"

    missing = [p for p in [agg_path, segs_path, ply_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing ScanNet files for {scene_id}: {[p.name for p in missing]}"
        )

    # Load aggregation: objectId -> label + segment IDs
    agg = json.loads(agg_path.read_text())
    obj_to_segs = {
        g["objectId"]: {"label": g["label"], "seg_ids": set(g["segments"])}
        for g in agg["segGroups"]
    }

    # Load segs: vertex index -> segment ID (as numpy array for fast indexing)
    segs_data   = json.loads(segs_path.read_text())
    seg_indices = np.array(segs_data["segIndices"], dtype=np.int32)

    # Load PLY vertices: x, y, z
    ply  = PlyData.read(str(ply_path))
    verts = ply["vertex"]
    xs = np.asarray(verts["x"], dtype=np.float32)
    ys = np.asarray(verts["y"], dtype=np.float32)
    zs = np.asarray(verts["z"], dtype=np.float32)

    # Compute AABB per object
    bboxes = {}
    for obj_id, info in obj_to_segs.items():
        mask = np.isin(seg_indices, list(info["seg_ids"]))
        n = mask.sum()
        if n == 0:
            continue
        ox, oy, oz = xs[mask], ys[mask], zs[mask]
        cx = (ox.min() + ox.max()) / 2
        cy = (oy.min() + oy.max()) / 2
        cz = (oz.min() + oz.max()) / 2
        dx = float(ox.max() - ox.min())
        dy = float(oy.max() - oy.min())
        dz = float(oz.max() - oz.min())
        bboxes[obj_id] = {
            "center": [float(cx), float(cy), float(cz)],
            "size":   [max(dx, 1e-3), max(dy, 1e-3), max(dz, 1e-3)],
            "label":  info["label"],
            "n_vertices": int(n),
        }

    return bboxes


def get_gt_bbox(object_id: int, scene_bboxes: dict) -> dict | None:
    """Return the GT bbox for a single object, or None if not found."""
    return scene_bboxes.get(object_id)


def load_scene_gt_point_clouds(
    scene_id: str,
    scannet_scans_dir,
    max_points: int = 512,
    seed: int = 42,
) -> dict:
    """
    Load per-object GT point clouds (subsampled vertex arrays) for a scene.
    Used for C2 (Object-Level Reconstruction Fidelity) — Chamfer distance evaluation.

    Args:
        scene_id:          ScanNet scene ID
        scannet_scans_dir: path to ScanNet scans directory
        max_points:        max vertices to keep per object (random subsample)
        seed:              random seed for reproducible subsampling

    Returns:
        dict mapping objectId (int) -> list of [x, y, z] points
        Returns empty dict if required files are missing.
    """
    scans_dir = Path(scannet_scans_dir)
    scene_dir = scans_dir / scene_id

    agg_path  = scene_dir / f"{scene_id}.aggregation.json"
    segs_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    ply_path  = scene_dir / f"{scene_id}_vh_clean_2.labels.ply"

    missing = [p for p in [agg_path, segs_path, ply_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing ScanNet files for {scene_id}: {[p.name for p in missing]}"
        )

    agg = json.loads(agg_path.read_text())
    obj_to_segs = {
        g["objectId"]: set(g["segments"])
        for g in agg["segGroups"]
    }

    segs_data   = json.loads(segs_path.read_text())
    seg_indices = np.array(segs_data["segIndices"], dtype=np.int32)

    ply   = PlyData.read(str(ply_path))
    verts = ply["vertex"]
    xs = np.asarray(verts["x"], dtype=np.float32)
    ys = np.asarray(verts["y"], dtype=np.float32)
    zs = np.asarray(verts["z"], dtype=np.float32)

    rng = np.random.default_rng(seed)
    point_clouds = {}

    for obj_id, seg_ids in obj_to_segs.items():
        mask = np.isin(seg_indices, list(seg_ids))
        n = mask.sum()
        if n == 0:
            continue
        ox = xs[mask]
        oy = ys[mask]
        oz = zs[mask]

        if n > max_points:
            idx = rng.choice(n, max_points, replace=False)
            ox, oy, oz = ox[idx], oy[idx], oz[idx]

        points = [[float(ox[i]), float(oy[i]), float(oz[i])] for i in range(len(ox))]
        point_clouds[obj_id] = points

    return point_clouds
