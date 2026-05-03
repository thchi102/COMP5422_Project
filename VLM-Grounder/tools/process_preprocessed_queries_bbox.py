#!/usr/bin/env python3
"""Batch-generate 3D AABBs from preprocessed query masks/frames/poses.

This script is designed for the directory layout:
  - triangulation_root/<query_id>/{masks,frames,poses}
  - posed_root/<scene_id>/intrinsic/intrinsic_depth.txt (or intrinsic_color.txt)
  - instance_data_root/<scene_id>/aligned_points.npy
  - scene_infos_pkl with axis_align_matrix per scene

For each query:
1) Map query_id -> scene_id from NR3D CSV (`assignmentid` -> `scan_id`)
2) Load scene aligned points
3) Project 3D points into each selected view using intrinsic + pose
4) Keep points that lie inside mask(s) across views (vote threshold)
5) Compute axis-aligned bbox [cx, cy, cz, dx, dy, dz]
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None


@dataclass
class QueryResult:
    query_id: str
    scene_id: str | None
    status: str
    num_views: int
    num_points: int
    bbox: list[float] | None
    message: str | None

    def to_json_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "scene_id": self.scene_id,
            "status": self.status,
            "num_views": self.num_views,
            "num_points": self.num_points,
            "bbox": self.bbox,
            "message": self.message,
        }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Process all preprocessed triangulation queries and produce 3D bbox."
    )
    parser.add_argument(
        "--triangulation-root",
        type=Path,
        default=script_dir / "scannet_top10_triangulation_openai",
        help="Root directory containing per-query folders.",
    )
    parser.add_argument(
        "--nr3d-csv",
        type=Path,
        default=script_dir.parent.parent.parent / "data" / "nr3d_top10.csv",
        help="CSV file with assignmentid -> scan_id mapping.",
    )
    parser.add_argument(
        "--posed-root",
        type=Path,
        default=script_dir / "scannet_top10_posed",
        help="Root containing scene folders with intrinsic files.",
    )
    parser.add_argument(
        "--instance-data-root",
        type=Path,
        default=script_dir / "scannet_top10_instance_data",
        help="Root containing per-scene aligned_points.npy.",
    )
    parser.add_argument(
        "--scene-infos-pkl",
        type=Path,
        default=script_dir
        / "scannet_top10_instance_data"
        / "scenes_train_val_info.pkl",
        help="Scene-info pickle containing axis_align_matrix.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=script_dir / "scannet_top10_triangulation_openai_bbox.json",
        help="Output JSON file for all query results.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help="Minimum number of masks a 3D point must satisfy.",
    )
    parser.add_argument(
        "--z-near",
        type=float,
        default=1e-4,
        help="Minimum camera-space depth for valid projection.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="Optional voxel downsampling (meters). Set <=0 to disable.",
    )
    parser.add_argument(
        "--trim-quantile",
        type=float,
        default=0.01,
        help="Trim each axis at both tails before bbox. Set 0 to disable.",
    )
    parser.add_argument(
        "--query-ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of query IDs to process, e.g. 489 1009.",
    )
    return parser.parse_args()


def load_assignment_to_scan(csv_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assignment_id = (row.get("assignmentid") or "").strip()
            scan_id = (row.get("scan_id") or "").strip()
            if assignment_id and scan_id:
                mapping[assignment_id] = scan_id
    return mapping


def load_scene_infos(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def load_intrinsic_matrix(posed_root: Path, scene_id: str) -> np.ndarray:
    intrinsic_dir = posed_root / scene_id / "intrinsic"
    candidates = [
        intrinsic_dir / "intrinsic_color.txt",
        intrinsic_dir / "intrinsic_depth.txt",
    ]
    for c in candidates:
        if c.is_file():
            mat = np.loadtxt(c, dtype=np.float64)
            if mat.shape == (3, 3):
                mat4 = np.eye(4, dtype=np.float64)
                mat4[:3, :3] = mat
                return mat4
            if mat.shape == (4, 4):
                return mat
            raise ValueError(f"Unexpected intrinsic shape {mat.shape} in {c}")
    raise FileNotFoundError(f"Intrinsic file not found for {scene_id}: {candidates}")


def iter_query_dirs(triangulation_root: Path) -> list[Path]:
    return sorted(
        p for p in triangulation_root.iterdir() if p.is_dir() and p.name.isdigit()
    )


def read_mask(mask_path: Path) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img > 0

    if Image is not None:
        with Image.open(mask_path) as img:
            return np.array(img.convert("L")) > 0

    raise RuntimeError(
        "No image backend available. Install opencv-python or pillow to read masks."
    )


def list_mask_pose_pairs(query_dir: Path) -> list[tuple[Path, Path]]:
    mask_dir = query_dir / "masks"
    pose_dir = query_dir / "poses"
    if not mask_dir.is_dir() or not pose_dir.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        pose_path = pose_dir / f"{mask_path.stem}.txt"
        if pose_path.is_file():
            pairs.append((mask_path, pose_path))
    return pairs


def points_to_homogeneous(points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3]
    ones = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    return np.concatenate([xyz, ones], axis=1)


def project_points_to_mask(
    aligned_points: np.ndarray,
    pose_c2w: np.ndarray,
    intrinsic: np.ndarray,
    axis_align_inv: np.ndarray,
    mask: np.ndarray,
    z_near: float,
) -> np.ndarray:
    # aligned(world-aligned) -> world (unaligned) -> camera
    aligned_h = points_to_homogeneous(aligned_points)  # [N, 4]
    world_h = (axis_align_inv @ aligned_h.T).T
    cam_h = (np.linalg.inv(pose_c2w) @ world_h.T).T

    z = cam_h[:, 2]
    valid = z > z_near
    if not np.any(valid):
        return np.zeros((aligned_points.shape[0],), dtype=bool)

    cam_valid = cam_h[valid]
    pix_h = (intrinsic @ cam_valid.T).T
    u = np.round(pix_h[:, 0] / pix_h[:, 2]).astype(np.int64)
    v = np.round(pix_h[:, 1] / pix_h[:, 2]).astype(np.int64)

    h, w = mask.shape[:2]
    in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    vote = np.zeros((aligned_points.shape[0],), dtype=bool)
    if np.any(in_frame):
        valid_indices = np.where(valid)[0]
        hit_indices = valid_indices[in_frame]
        vote[hit_indices] = mask[v[in_frame], u[in_frame]]
    return vote


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or points.shape[0] == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def trim_outliers_quantile(points: np.ndarray, q: float) -> np.ndarray:
    if q <= 0 or points.shape[0] == 0:
        return points
    q = min(max(q, 0.0), 0.49)
    low = np.quantile(points[:, :3], q, axis=0)
    high = np.quantile(points[:, :3], 1.0 - q, axis=0)
    keep = np.all((points[:, :3] >= low) & (points[:, :3] <= high), axis=1)
    return points[keep]


def calculate_aabb(points: np.ndarray) -> np.ndarray:
    min_corner = np.min(points[:, :3], axis=0)
    max_corner = np.max(points[:, :3], axis=0)
    center = (max_corner + min_corner) / 2.0
    size = max_corner - min_corner
    return np.concatenate([center, size], axis=0)


def process_query(
    query_dir: Path,
    scene_id: str,
    scene_infos: dict,
    intrinsic: np.ndarray,
    aligned_points: np.ndarray,
    min_votes: int,
    z_near: float,
    voxel_size: float,
    trim_q: float,
) -> QueryResult:
    pairs = list_mask_pose_pairs(query_dir)
    if not pairs:
        return QueryResult(
            query_id=query_dir.name,
            scene_id=scene_id,
            status="skip",
            num_views=0,
            num_points=0,
            bbox=None,
            message="Missing masks/poses or no matched mask-pose pairs.",
        )

    axis_align = scene_infos[scene_id]["axis_align_matrix"]
    axis_align_inv = np.linalg.inv(axis_align)
    votes = np.zeros((aligned_points.shape[0],), dtype=np.int32)
    valid_views = 0

    for mask_path, pose_path in pairs:
        mask = read_mask(mask_path)
        pose = np.loadtxt(pose_path, dtype=np.float64)
        if pose.shape != (4, 4):
            return QueryResult(
                query_id=query_dir.name,
                scene_id=scene_id,
                status="error",
                num_views=valid_views,
                num_points=0,
                bbox=None,
                message=f"Invalid pose shape {pose.shape} for {pose_path.name}",
            )
        hit = project_points_to_mask(
            aligned_points=aligned_points,
            pose_c2w=pose,
            intrinsic=intrinsic,
            axis_align_inv=axis_align_inv,
            mask=mask,
            z_near=z_near,
        )
        votes += hit.astype(np.int32)
        valid_views += 1

    if valid_views == 0:
        return QueryResult(
            query_id=query_dir.name,
            scene_id=scene_id,
            status="skip",
            num_views=0,
            num_points=0,
            bbox=None,
            message="No valid view was processed.",
        )

    threshold = max(1, min(min_votes, valid_views))
    selected = aligned_points[votes >= threshold]
    if selected.shape[0] == 0:
        return QueryResult(
            query_id=query_dir.name,
            scene_id=scene_id,
            status="empty",
            num_views=valid_views,
            num_points=0,
            bbox=None,
            message=f"No 3D points pass vote threshold={threshold}.",
        )

    selected = voxel_downsample(selected, voxel_size=voxel_size)
    selected = trim_outliers_quantile(selected, q=trim_q)
    if selected.shape[0] == 0:
        return QueryResult(
            query_id=query_dir.name,
            scene_id=scene_id,
            status="empty",
            num_views=valid_views,
            num_points=0,
            bbox=None,
            message="No 3D points left after downsample/trim.",
        )

    bbox = calculate_aabb(selected)
    return QueryResult(
        query_id=query_dir.name,
        scene_id=scene_id,
        status="ok",
        num_views=valid_views,
        num_points=int(selected.shape[0]),
        bbox=[float(x) for x in bbox.tolist()],
        message=None,
    )


def main() -> None:
    args = parse_args()
    if not args.triangulation_root.is_dir():
        raise FileNotFoundError(f"triangulation root does not exist: {args.triangulation_root}")
    if not args.nr3d_csv.is_file():
        raise FileNotFoundError(f"nr3d csv does not exist: {args.nr3d_csv}")
    if not args.scene_infos_pkl.is_file():
        raise FileNotFoundError(f"scene infos pkl does not exist: {args.scene_infos_pkl}")
    if not args.instance_data_root.is_dir():
        raise FileNotFoundError(f"instance data root does not exist: {args.instance_data_root}")
    if not args.posed_root.is_dir():
        raise FileNotFoundError(f"posed root does not exist: {args.posed_root}")

    assignment_to_scan = load_assignment_to_scan(args.nr3d_csv)
    scene_infos = load_scene_infos(args.scene_infos_pkl)

    query_dirs = iter_query_dirs(args.triangulation_root)
    if args.query_ids:
        selected = set(args.query_ids)
        query_dirs = [q for q in query_dirs if q.name in selected]
    results: list[QueryResult] = []

    scene_points_cache: dict[str, np.ndarray] = {}
    scene_intrinsic_cache: dict[str, np.ndarray] = {}

    for query_dir in query_dirs:
        query_id = query_dir.name
        scene_id = assignment_to_scan.get(query_id)
        if not scene_id:
            results.append(
                QueryResult(
                    query_id=query_id,
                    scene_id=None,
                    status="skip",
                    num_views=0,
                    num_points=0,
                    bbox=None,
                    message="Query id not found in nr3d csv.",
                )
            )
            continue
        if scene_id not in scene_infos:
            results.append(
                QueryResult(
                    query_id=query_id,
                    scene_id=scene_id,
                    status="skip",
                    num_views=0,
                    num_points=0,
                    bbox=None,
                    message="Scene id not found in scene infos.",
                )
            )
            continue

        try:
            if scene_id not in scene_points_cache:
                scene_points_path = args.instance_data_root / scene_id / "aligned_points.npy"
                if not scene_points_path.is_file():
                    raise FileNotFoundError(f"Missing aligned points: {scene_points_path}")
                scene_points_cache[scene_id] = np.load(scene_points_path)
            if scene_id not in scene_intrinsic_cache:
                scene_intrinsic_cache[scene_id] = load_intrinsic_matrix(
                    posed_root=args.posed_root, scene_id=scene_id
                )

            result = process_query(
                query_dir=query_dir,
                scene_id=scene_id,
                scene_infos=scene_infos,
                intrinsic=scene_intrinsic_cache[scene_id],
                aligned_points=scene_points_cache[scene_id],
                min_votes=args.min_votes,
                z_near=args.z_near,
                voxel_size=args.voxel_size,
                trim_q=args.trim_quantile,
            )
            results.append(result)
            print(
                f"[{result.status.upper()}] query={query_id} scene={scene_id} "
                f"views={result.num_views} points={result.num_points}"
            )
        except Exception as exc:
            results.append(
                QueryResult(
                    query_id=query_id,
                    scene_id=scene_id,
                    status="error",
                    num_views=0,
                    num_points=0,
                    bbox=None,
                    message=str(exc),
                )
            )
            print(f"[ERROR] query={query_id} scene={scene_id}: {exc}")

    summary = {
        "total_queries": len(results),
        "ok": sum(r.status == "ok" for r in results),
        "empty": sum(r.status == "empty" for r in results),
        "skip": sum(r.status == "skip" for r in results),
        "error": sum(r.status == "error" for r in results),
        "config": {
            "triangulation_root": str(args.triangulation_root),
            "nr3d_csv": str(args.nr3d_csv),
            "posed_root": str(args.posed_root),
            "instance_data_root": str(args.instance_data_root),
            "scene_infos_pkl": str(args.scene_infos_pkl),
            "min_votes": args.min_votes,
            "z_near": args.z_near,
            "voxel_size": args.voxel_size,
            "trim_quantile": args.trim_quantile,
        },
        "results": [r.to_json_dict() for r in results],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {args.output_json}")
    print(
        f"Done. total={summary['total_queries']} ok={summary['ok']} "
        f"empty={summary['empty']} skip={summary['skip']} error={summary['error']}"
    )


if __name__ == "__main__":
    main()
