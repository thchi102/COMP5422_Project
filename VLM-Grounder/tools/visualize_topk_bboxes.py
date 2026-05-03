#!/usr/bin/env python3
"""Visualize top-k predicted and GT bboxes on ScanNet scene meshes.

This script reads:
- evaluation json (from evaluate_preprocessed_bboxes.py)
- scene infos pkl (for axis_align_matrix)
- ScanNet meshes under scannet_top10/scannet_top10/scans/<scene_id>

For each top-k query:
1) load scene mesh
2) apply axis_align_matrix to mesh vertices
3) build wireframe meshes for predicted bbox (red) and GT bbox (green)
4) export one merged mesh that can be opened in Meshlab / Blender / Open3D
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Visualize top-k bbox predictions in 3D.")
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=script_dir / "scannet_top10_triangulation_openai_bbox_colorK_eval.json",
        help="Evaluation JSON with top_queries and records.",
    )
    parser.add_argument(
        "--scene-infos-pkl",
        type=Path,
        default=script_dir
        / "scannet_top10_instance_data"
        / "scenes_train_val_info.pkl",
        help="Scene info pkl containing axis_align_matrix.",
    )
    parser.add_argument(
        "--scans-root",
        type=Path,
        default=script_dir / "scannet_top10" / "scans",
        help="Root containing scene folders with ScanNet meshes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "bbox_top10_visualizations",
        help="Directory to save output visualizations.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top queries to visualize.",
    )
    parser.add_argument(
        "--edge-radius",
        type=float,
        default=0.015,
        help="Cylinder radius for bbox wireframe edges.",
    )
    return parser.parse_args()


def load_eval(eval_json: Path) -> tuple[list[dict], dict[str, dict]]:
    with eval_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    top_queries = data.get("top_queries", [])
    records = data.get("records", [])
    record_by_query = {str(r["query_id"]): r for r in records}
    return top_queries, record_by_query


def load_scene_infos(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def find_mesh_path(scans_root: Path, scene_id: str) -> Path:
    scene_dir = scans_root / scene_id
    candidates = [
        scene_dir / f"{scene_id}_vh_clean_2.ply",
        scene_dir / f"{scene_id}_vh_clean.ply",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Mesh not found for {scene_id}. Tried: {candidates}")


def bbox_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    cx, cy, cz = center
    dx, dy, dz = size
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    return np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float64,
    )


def edge_indices() -> list[tuple[int, int]]:
    return [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]


def rotation_matrix_from_z_to_vec(vec: np.ndarray) -> np.ndarray:
    """Compute rotation matrix mapping +Z axis to vec direction."""
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(z_axis, vec)
    dot = np.clip(np.dot(z_axis, vec), -1.0, 1.0)
    if np.linalg.norm(cross) < 1e-10:
        if dot > 0:
            return np.eye(3)
        # 180-degree around X if vec is -Z
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    axis = cross / np.linalg.norm(cross)
    angle = float(np.arccos(dot))
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def bbox_wireframe_mesh(
    box6: np.ndarray,
    color: np.ndarray,
    edge_radius: float,
) -> o3d.geometry.TriangleMesh:
    center = box6[:3]
    size = box6[3:6]
    corners = bbox_corners(center, size)
    mesh = o3d.geometry.TriangleMesh()
    for i0, i1 in edge_indices():
        p0 = corners[i0]
        p1 = corners[i1]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length <= 1e-10:
            continue
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=edge_radius, height=length)
        cyl.compute_vertex_normals()
        rot = rotation_matrix_from_z_to_vec(vec)
        cyl.rotate(rot, center=np.array([0.0, 0.0, 0.0]))
        cyl.translate((p0 + p1) / 2.0)
        cyl.paint_uniform_color(color.tolist())
        mesh += cyl
    return mesh


def transform_mesh_vertices(mesh: o3d.geometry.TriangleMesh, transform: np.ndarray) -> None:
    verts = np.asarray(mesh.vertices)
    ones = np.ones((verts.shape[0], 1), dtype=np.float64)
    verts_h = np.concatenate([verts, ones], axis=1)
    verts_t = (transform @ verts_h.T).T[:, :3]
    mesh.vertices = o3d.utility.Vector3dVector(verts_t)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    top_queries, record_by_query = load_eval(args.eval_json)
    scene_infos = load_scene_infos(args.scene_infos_pkl)

    selected = top_queries[: max(1, args.top_k)]
    if not selected:
        raise ValueError("No top_queries found in eval json.")

    for rank, entry in enumerate(selected, start=1):
        query_id = str(entry["query_id"])
        scene_id = str(entry["scene_id"])
        iou = float(entry["iou_3d"])
        rec = record_by_query.get(query_id)
        if rec is None:
            print(f"[Skip] query={query_id}: missing record details.")
            continue
        if rec.get("pred_bbox") is None or rec.get("gt_bbox") is None:
            print(f"[Skip] query={query_id}: missing pred/gt bbox.")
            continue

        pred_box = np.asarray(rec["pred_bbox"], dtype=np.float64)
        gt_box = np.asarray(rec["gt_bbox"], dtype=np.float64)
        axis_align = np.asarray(scene_infos[scene_id]["axis_align_matrix"], dtype=np.float64)

        mesh_path = find_mesh_path(args.scans_root, scene_id)
        scene_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if scene_mesh.is_empty():
            print(f"[Skip] query={query_id}: failed to load mesh {mesh_path}.")
            continue
        if not scene_mesh.has_vertex_normals():
            scene_mesh.compute_vertex_normals()
        transform_mesh_vertices(scene_mesh, axis_align)
        if not scene_mesh.has_vertex_colors():
            scene_mesh.paint_uniform_color([0.75, 0.75, 0.75])

        pred_wire = bbox_wireframe_mesh(
            box6=pred_box,
            color=np.array([1.0, 0.1, 0.1], dtype=np.float64),  # red
            edge_radius=args.edge_radius,
        )
        gt_wire = bbox_wireframe_mesh(
            box6=gt_box,
            color=np.array([0.1, 0.95, 0.1], dtype=np.float64),  # green
            edge_radius=args.edge_radius,
        )
        merged = scene_mesh + pred_wire + gt_wire
        merged.compute_vertex_normals()

        out_prefix = (
            f"rank{rank:02d}_q{query_id}_{scene_id}_iou{str(round(iou, 4)).replace('.', 'p')}"
        )
        out_mesh = args.output_dir / f"{out_prefix}.ply"
        out_meta = args.output_dir / f"{out_prefix}.json"

        ok = o3d.io.write_triangle_mesh(str(out_mesh), merged)
        if not ok:
            print(f"[Warn] failed to write mesh: {out_mesh}")
            continue

        with out_meta.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "rank": rank,
                    "query_id": query_id,
                    "scene_id": scene_id,
                    "target_id": rec["target_id"],
                    "iou_3d": iou,
                    "pred_bbox": rec["pred_bbox"],
                    "gt_bbox": rec["gt_bbox"],
                    "source_scene_mesh": str(mesh_path),
                    "output_mesh": str(out_mesh),
                    "legend": {"pred_bbox": "red", "gt_bbox": "green"},
                },
                f,
                indent=2,
            )
        print(f"[OK] rank={rank} query={query_id} scene={scene_id} -> {out_mesh.name}")

    print(f"Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
