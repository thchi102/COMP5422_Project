#!/usr/bin/env python3
"""Select triangulation-friendly frames from binary masks.

For each query folder under `--input-root`:
1) Check whether `binary_mask_openai` exists.
2) Keep mask frames whose area ratio exceeds a threshold.
3) Subsample up to N candidates evenly across time for viewpoint diversity.
4) Copy selected frame + mask + matching pose txt into a separate output root.
"""

from __future__ import annotations

import argparse
import csv
import struct
import re
import shutil
import zlib
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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MASK_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class SceneAssets:
    image_by_stem: dict[str, Path]
    image_by_num: dict[int, Path]
    pose_by_stem: dict[str, Path]
    pose_by_num: dict[int, Path]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input_root = script_dir / "scannet_top10_output_openai"
    default_csv = default_input_root / "nr3d_subset.csv"
    default_output_root = script_dir / "scannet_top10_triangulation_openai"
    default_posed_roots = [
        script_dir / "scannet_top10_posed",
        script_dir.parent / "scannet" / "posed_images",
    ]

    parser = argparse.ArgumentParser(
        description="Select 6 diverse high-area mask frames for triangulation."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_input_root,
        help=f"Root directory with query folders (default: {default_input_root})",
    )
    parser.add_argument(
        "--nr3d-csv",
        type=Path,
        default=default_csv,
        help=f"CSV file mapping assignment id -> scan id (default: {default_csv})",
    )
    parser.add_argument(
        "--posed-roots",
        type=Path,
        nargs="+",
        default=default_posed_roots,
        help="One or more roots containing scene folders with images/pose.",
    )
    parser.add_argument(
        "--mask-dir-name",
        type=str,
        default="binary_mask_openai",
        help="Mask directory name under each query folder.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help=f"Separate output root for selected data (default: {default_output_root})",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.01,
        help="Minimum foreground ratio in mask to keep candidate frame.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=6,
        help="Number of evenly sampled frames from candidates.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folder if present.",
    )
    return parser.parse_args()


def numeric_id_from_name(name: str) -> int | None:
    matches = re.findall(r"\d+", name)
    if not matches:
        return None
    return int(matches[-1])


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


def gather_scene_assets(scene_dir: Path) -> SceneAssets:
    image_candidates = [scene_dir / "color", scene_dir / "images", scene_dir]
    image_files: list[Path] = []
    for folder in image_candidates:
        if folder.is_dir():
            image_files.extend(
                sorted(
                    p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                )
            )
            if image_files:
                break

    pose_dir = scene_dir / "pose"
    pose_files = []
    if pose_dir.is_dir():
        pose_files = sorted(p for p in pose_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")

    image_by_stem: dict[str, Path] = {}
    image_by_num: dict[int, Path] = {}
    for img in image_files:
        stem = img.stem
        image_by_stem.setdefault(stem, img)
        n = numeric_id_from_name(stem)
        if n is not None:
            image_by_num.setdefault(n, img)

    pose_by_stem: dict[str, Path] = {}
    pose_by_num: dict[int, Path] = {}
    for pose in pose_files:
        stem = pose.stem
        pose_by_stem.setdefault(stem, pose)
        n = numeric_id_from_name(stem)
        if n is not None:
            pose_by_num.setdefault(n, pose)

    return SceneAssets(
        image_by_stem=image_by_stem,
        image_by_num=image_by_num,
        pose_by_stem=pose_by_stem,
        pose_by_num=pose_by_num,
    )


def resolve_scene_dir(scan_id: str, posed_roots: list[Path]) -> Path | None:
    for root in posed_roots:
        scene_dir = root / scan_id
        if scene_dir.is_dir():
            return scene_dir
    return None


def mask_area_ratio(mask_path: Path) -> float:
    try:
        if cv2 is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.size > 0:
                return float(np.count_nonzero(mask)) / float(mask.size)
        if Image is not None:
            with Image.open(mask_path) as img:
                mask = np.array(img.convert("L"))
                if mask.size > 0:
                    return float(np.count_nonzero(mask)) / float(mask.size)
    except Exception:
        pass

    if mask_path.suffix.lower() != ".png":
        return 0.0

    try:
        pixels = load_png_rgba(mask_path)
    except Exception:
        return 0.0

    if pixels.size == 0:
        return 0.0
    rgb = pixels[:, :, :3]
    nonzero = np.any(rgb != 0, axis=2)
    return float(np.count_nonzero(nonzero)) / float(nonzero.size)


def load_png_rgba(path: Path) -> np.ndarray:
    data = path.read_bytes()
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        raise ValueError(f"Invalid PNG file: {path}")

    width = height = None
    bit_depth = color_type = None
    idat_parts: list[bytes] = []
    idx = len(signature)
    while idx < len(data):
        if idx + 8 > len(data):
            break
        length = struct.unpack(">I", data[idx : idx + 4])[0]
        chunk_type = data[idx + 4 : idx + 8]
        chunk_data_start = idx + 8
        chunk_data_end = chunk_data_start + length
        chunk_data = data[chunk_data_start:chunk_data_end]
        idx = chunk_data_end + 4  # skip CRC

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type = parse_ihdr(chunk_data)
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError(f"Malformed PNG (missing IHDR): {path}")
    if bit_depth != 8:
        raise ValueError(f"Unsupported PNG bit depth {bit_depth}: {path}")

    raw = zlib.decompress(b"".join(idat_parts))
    channels = channels_for_color_type(color_type)
    bpp = channels  # bytes per pixel for 8-bit PNG
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) < expected:
        raise ValueError(f"Corrupt PNG stream: {path}")

    out = np.zeros((height, stride), dtype=np.uint8)
    src_idx = 0
    prev = np.zeros(stride, dtype=np.uint8)
    for y in range(height):
        filter_type = raw[src_idx]
        src_idx += 1
        line = np.frombuffer(raw[src_idx : src_idx + stride], dtype=np.uint8).copy()
        src_idx += stride
        out[y, :] = apply_png_filter(line, prev, filter_type, bpp)
        prev = out[y, :]

    pixels = out.reshape(height, width, channels)
    if color_type == 0:  # grayscale
        rgb = np.repeat(pixels, 3, axis=2)
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        return np.concatenate([rgb, alpha], axis=2)
    if color_type == 2:  # RGB
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        return np.concatenate([pixels, alpha], axis=2)
    if color_type == 4:  # grayscale + alpha
        gray = pixels[:, :, :1]
        alpha = pixels[:, :, 1:2]
        rgb = np.repeat(gray, 3, axis=2)
        return np.concatenate([rgb, alpha], axis=2)
    if color_type == 6:  # RGBA
        return pixels
    raise ValueError(f"Unsupported PNG color type {color_type}: {path}")


def parse_ihdr(chunk_data: bytes) -> tuple[int, int, int, int]:
    if len(chunk_data) != 13:
        raise ValueError("Invalid IHDR length")
    width = struct.unpack(">I", chunk_data[0:4])[0]
    height = struct.unpack(">I", chunk_data[4:8])[0]
    bit_depth = chunk_data[8]
    color_type = chunk_data[9]
    compression = chunk_data[10]
    filtering = chunk_data[11]
    interlace = chunk_data[12]
    if compression != 0 or filtering != 0 or interlace != 0:
        raise ValueError("Unsupported PNG compression/filter/interlace")
    return width, height, bit_depth, color_type


def channels_for_color_type(color_type: int) -> int:
    if color_type == 0:
        return 1
    if color_type == 2:
        return 3
    if color_type == 4:
        return 2
    if color_type == 6:
        return 4
    raise ValueError(f"Unsupported PNG color type: {color_type}")


def paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def apply_png_filter(
    line: np.ndarray, prev_line: np.ndarray, filter_type: int, bpp: int
) -> np.ndarray:
    out = line.copy()
    if filter_type == 0:  # None
        return out
    if filter_type == 1:  # Sub
        for i in range(bpp, len(out)):
            out[i] = (out[i] + out[i - bpp]) & 0xFF
        return out
    if filter_type == 2:  # Up
        out = (out + prev_line) & 0xFF
        return out.astype(np.uint8)
    if filter_type == 3:  # Average
        for i in range(len(out)):
            left = out[i - bpp] if i >= bpp else 0
            up = prev_line[i]
            out[i] = (out[i] + ((left + up) // 2)) & 0xFF
        return out
    if filter_type == 4:  # Paeth
        for i in range(len(out)):
            a = out[i - bpp] if i >= bpp else 0
            b = prev_line[i]
            c = prev_line[i - bpp] if i >= bpp else 0
            out[i] = (out[i] + paeth_predictor(int(a), int(b), int(c))) & 0xFF
        return out
    raise ValueError(f"Unsupported PNG filter type: {filter_type}")


def select_evenly(paths: list[Path], num_samples: int) -> list[Path]:
    if len(paths) <= num_samples:
        return paths
    idx = np.linspace(0, len(paths) - 1, num_samples)
    idx = np.round(idx).astype(int)
    idx = np.unique(idx)
    selected = [paths[i] for i in idx.tolist()]
    if len(selected) > num_samples:
        selected = selected[:num_samples]
    while len(selected) < num_samples:
        for path in paths:
            if path not in selected:
                selected.append(path)
            if len(selected) == num_samples:
                break
    return selected


def find_matching_image_and_pose(mask_path: Path, assets: SceneAssets) -> tuple[Path | None, Path | None]:
    stem = mask_path.stem
    num = numeric_id_from_name(stem)

    image = assets.image_by_stem.get(stem)
    if image is None and num is not None:
        image = assets.image_by_num.get(num)

    pose = assets.pose_by_stem.get(stem)
    if pose is None and num is not None:
        pose = assets.pose_by_num.get(num)

    return image, pose


def process_query_folder(
    query_dir: Path,
    scan_id: str,
    args: argparse.Namespace,
) -> tuple[int, int, bool]:
    mask_dir = query_dir / args.mask_dir_name
    if not mask_dir.is_dir():
        return (0, 0, False)

    scene_dir = resolve_scene_dir(scan_id, args.posed_roots)
    if scene_dir is None:
        print(f"[Skip] {query_dir.name}: scene folder not found for {scan_id}")
        return (0, 0, True)

    assets = gather_scene_assets(scene_dir)
    if not assets.image_by_stem and not assets.image_by_num:
        print(f"[Skip] {query_dir.name}: no images found in {scene_dir}")
        return (0, 0, True)
    if not assets.pose_by_stem and not assets.pose_by_num:
        print(f"[Skip] {query_dir.name}: no poses found in {scene_dir / 'pose'}")
        return (0, 0, True)

    masks = sorted(
        [
            p
            for p in mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() in MASK_SUFFIXES
        ],
        key=lambda p: (numeric_id_from_name(p.stem) is None, numeric_id_from_name(p.stem) or 0, p.stem),
    )
    if not masks:
        print(f"[Skip] {query_dir.name}: no mask files in {mask_dir}")
        return (0, 0, True)

    candidates = [m for m in masks if mask_area_ratio(m) >= args.min_area_ratio]
    if not candidates:
        candidates = sorted(masks, key=mask_area_ratio, reverse=True)[: args.num_frames]
        candidates = sorted(candidates, key=lambda p: numeric_id_from_name(p.stem) or 0)

    selected_masks = select_evenly(candidates, args.num_frames)

    out_dir = args.output_root / query_dir.name
    frames_out_dir = out_dir / "frames"
    masks_out_dir = out_dir / "masks"
    poses_out_dir = out_dir / "poses"
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)
    poses_out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for mask_path in selected_masks:
        img_path, pose_path = find_matching_image_and_pose(mask_path, assets)
        if img_path is None or pose_path is None:
            print(
                f"[Warn] {query_dir.name}: cannot match image/pose for mask {mask_path.name}"
            )
            continue
        shutil.copy2(img_path, frames_out_dir / img_path.name)
        shutil.copy2(mask_path, masks_out_dir / mask_path.name)
        shutil.copy2(pose_path, poses_out_dir / pose_path.name)
        copied += 1

    print(
        f"[OK] {query_dir.name}: masks={len(masks)}, candidates={len(candidates)}, "
        f"selected={len(selected_masks)}, copied_triplets={copied}"
    )
    return (len(selected_masks), copied, True)


def main() -> None:
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if not args.nr3d_csv.is_file():
        raise FileNotFoundError(f"CSV does not exist: {args.nr3d_csv}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be > 0")
    if not (0.0 <= args.min_area_ratio <= 1.0):
        raise ValueError("--min-area-ratio must be in [0, 1]")

    assignment_to_scan = load_assignment_to_scan(args.nr3d_csv)
    query_dirs = sorted(
        p for p in args.input_root.iterdir() if p.is_dir() and p.name.isdigit()
    )

    total_selected = 0
    total_copied = 0
    processed = 0
    with_mask_dir = 0
    for query_dir in query_dirs:
        scan_id = assignment_to_scan.get(query_dir.name)
        if not scan_id:
            print(f"[Skip] {query_dir.name}: scan_id not found in CSV")
            continue
        selected, copied, has_mask_dir = process_query_folder(query_dir, scan_id, args)
        if has_mask_dir:
            with_mask_dir += 1
        if selected > 0:
            processed += 1
        total_selected += selected
        total_copied += copied

    print(
        f"Done. query_folders={len(query_dirs)}, folders_with_{args.mask_dir_name}={with_mask_dir}, "
        f"processed_folders={processed}, selected_masks={total_selected}, copied_pairs={total_copied}"
    )


if __name__ == "__main__":
    main()
