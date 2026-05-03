#!/usr/bin/env python3
"""Extract every Nth frame from ScanNet scene videos.

Default behavior:
- Read videos from:  scannet_videos/
- Save frames to:    scannet_frames/<scene_name>/
- Sample interval:   every 20 frames
- Output names:      00000.jpg, 00001.jpg, ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def extract_video_frames(video_path: Path, output_dir: Path, interval: int) -> int:
    """Extract frames from one video every `interval` frames."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return 0

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            output_path = output_dir / f"{saved_idx:05d}.jpg"
            ok = cv2.imwrite(str(output_path), frame)
            if ok:
                saved_idx += 1
            else:
                print(f"[WARN] Failed to write frame: {output_path}")

        frame_idx += 1

    cap.release()
    return saved_idx


def list_video_files(input_dir: Path) -> list[Path]:
    """Return sorted list of videos in `input_dir` (recursive)."""
    videos = [
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "scannet_videos"
    default_output = script_dir / "scannet_frames"

    parser = argparse.ArgumentParser(
        description="Extract every Nth frame from scene videos into per-scene folders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Folder containing scene videos (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Output root folder (default: {default_output})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="Save one frame every N frames (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    interval: int = args.interval

    if interval <= 0:
        raise ValueError("--interval must be a positive integer")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    video_files = list_video_files(input_dir)
    if not video_files:
        print(f"No video files found in: {input_dir}")
        return

    total_saved = 0
    print(f"Found {len(video_files)} videos. Extracting every {interval} frames...")

    for video_path in video_files:
        scene_name = video_path.stem
        scene_output_dir = output_dir / scene_name
        saved = extract_video_frames(video_path, scene_output_dir, interval)
        total_saved += saved
        print(f"[OK] {video_path.name} -> {scene_output_dir} ({saved} frames)")

    print(f"Done. Saved {total_saved} frames into: {output_dir}")


if __name__ == "__main__":
    main()
