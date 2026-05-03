import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from infer.seg_zero import generate_mask
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.util import parse_gpt_output, save_video


COLOR_MAP = [
    (255, 0, 0),
    (0, 255, 255),
    (128, 255, 0),
    (187, 19, 208),
    (222, 148, 80),
    (147, 71, 238),
    (98, 43, 249),
]


def parse_args(args):
    parser = argparse.ArgumentParser(description="Batch CoT-RVS segmentation for ScanNet")
    parser.add_argument(
        "--cot_results_dir",
        required=True,
        type=str,
        help="Path to CoT result folders (e.g. scannet_output)",
    )
    parser.add_argument(
        "--video_frames_root",
        required=True,
        type=str,
        help="Path to ScanNet frames root (supports posed_images layout or scene directories)",
    )
    parser.add_argument("--sam2_model", default="checkpoints/sam2.1_hiera_large.pt", type=str)
    parser.add_argument("--sam2_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml", type=str)
    parser.add_argument("--segzero_model", default="checkpoints/Seg-Zero-7B", type=str)
    parser.add_argument("--num_candidates", default=8, type=int)
    parser.add_argument(
        "--mask_dir_name",
        default="binary_masks",
        type=str,
        help="Directory name under each CoT result folder for binary masks",
    )
    parser.add_argument(
        "--video_name",
        default="segmented.mp4",
        type=str,
        help="Output mp4 name under each CoT result folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing segmented.mp4 and binary masks",
    )
    return parser.parse_args(args)


def sorted_frame_names(video_dir):
    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]

    def frame_sort_key(name):
        stem = os.path.splitext(name)[0]
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            return (0, int(digits), stem)
        return (1, stem)

    frame_names.sort(key=frame_sort_key)
    return frame_names


def find_scene_id(scene_result_dir):
    keyframe_dir = os.path.join(scene_result_dir, "keyframes")
    if not os.path.isdir(keyframe_dir):
        return None

    keyframe_imgs = [
        p
        for p in os.listdir(keyframe_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    if not keyframe_imgs:
        return None

    # Keyframe file is typically sceneXXXX_00.jpg.
    return os.path.splitext(keyframe_imgs[0])[0]


def resolve_video_dir(video_frames_root, scene_id):
    candidates = [
        os.path.join(video_frames_root, scene_id, "color"),
        os.path.join(video_frames_root, "posed_images", scene_id, "color"),
        os.path.join(video_frames_root, scene_id),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def seg_zero_objects(
    args,
    reasoning_model,
    segmentation_model,
    processor,
    image_paths,
    target_objects,
    sample_every,
):
    masks = []
    ann_frame_idx = []
    max_frame = len(image_paths) - 1

    for target in target_objects:
        ann_keyframe = (target["keyframe"] - 1) * sample_every
        ann_keyframe = max(0, min(ann_keyframe, max_frame))
        object_desc = target["object_description"]
        prompt = f"Please segment {object_desc}."
        image_path = image_paths[ann_keyframe]
        mask = generate_mask(
            args,
            reasoning_model,
            segmentation_model,
            processor,
            image_path,
            prompt,
            save_mask=False,
            save_name="",
        )
        masks.append(mask)
        ann_frame_idx.append(ann_keyframe)
    return masks, ann_frame_idx


def ensure_model_runtime(device):
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "Support for MPS devices is preliminary; outputs may differ from CUDA."
        )


def process_scene(
    args,
    scene_result_dir,
    video_dir,
    reasoning_model,
    segmentation_model,
    processor,
    predictor,
):
    answer_path = os.path.join(scene_result_dir, "answer.txt")
    with open(answer_path) as f:
        answer = f.read()

    target_objects = parse_gpt_output(answer)
    if not target_objects:
        raise RuntimeError("No target objects parsed from answer.txt")

    frame_names = sorted_frame_names(video_dir)
    if not frame_names:
        raise RuntimeError(f"No frames found in {video_dir}")

    t_frames = len(frame_names)
    sample_every = (t_frames - 1) // args.num_candidates + 1
    all_image_paths = [os.path.join(video_dir, path) for path in frame_names]

    masks, ann_frame_idx = seg_zero_objects(
        args,
        reasoning_model,
        segmentation_model,
        processor,
        all_image_paths,
        target_objects,
        sample_every,
    )

    inference_state = predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=True,
        async_loading_frames=True,
    )
    predictor.reset_state(inference_state)

    for object_id, (binary_mask, frame_idx) in enumerate(zip(masks, ann_frame_idx), start=1):
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            mask=binary_mask,
        )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, reverse=True
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    seg_video_frames = []
    mask_dir = os.path.join(scene_result_dir, args.mask_dir_name)
    os.makedirs(mask_dir, exist_ok=True)

    for out_frame_idx in range(len(frame_names)):
        image_path = os.path.join(video_dir, frame_names[out_frame_idx])
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        save_img = image_np.copy()

        mask_taken = np.zeros_like(image_np[:, :, 0], dtype=np.bool_)
        for object_idx in range(len(target_objects)):
            frame_masks = video_segments.get(out_frame_idx, {})
            if object_idx + 1 not in frame_masks:
                continue
            pred_mask = frame_masks[object_idx + 1][0]
            pred_mask = np.logical_and(pred_mask, np.logical_not(mask_taken))
            mask_taken = np.logical_or(mask_taken, pred_mask)
            color = np.array(COLOR_MAP[object_idx % len(COLOR_MAP)], dtype=np.float32)
            save_img[pred_mask] = (
                save_img[pred_mask].astype(np.float32) * 0.4 + color * 0.6
            ).astype(np.uint8)

        seg_video_frames.append(save_img.astype(np.uint8))
        mask_img = (mask_taken.astype(np.uint8)) * 255
        frame_stem = os.path.splitext(frame_names[out_frame_idx])[0]
        cv2.imwrite(os.path.join(mask_dir, f"{frame_stem}.png"), mask_img)

    save_video(seg_video_frames, os.path.join(scene_result_dir, args.video_name))


def discover_scene_result_dirs(cot_results_dir):
    root = Path(cot_results_dir)
    return sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir() and (p / "answer.txt").exists()
        ],
        key=lambda p: p.name,
    )


def main(args):
    args = parse_args(args)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    ensure_model_runtime(device)

    print("Loading Seg-Zero ...")
    reasoning_model_path = args.segzero_model
    segmentation_model_path = "facebook/sam2-hiera-large"
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()
    segmentation_model = SAM2ImagePredictor.from_pretrained(segmentation_model_path)
    processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")

    predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_model, device=device)

    scene_dirs = discover_scene_result_dirs(args.cot_results_dir)
    if not scene_dirs:
        raise RuntimeError(f"No scene result directories with answer.txt found in {args.cot_results_dir}")

    print(f"Found {len(scene_dirs)} scene result folders.")
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        output_video_path = os.path.join(str(scene_dir), args.video_name)
        mask_dir = os.path.join(str(scene_dir), args.mask_dir_name)
        if not args.overwrite and os.path.exists(output_video_path) and os.path.isdir(mask_dir):
            continue

        scene_id = find_scene_id(str(scene_dir))
        if scene_id is None:
            print(f"[Skip] {scene_dir.name}: keyframes scene id not found.")
            continue

        video_dir = resolve_video_dir(args.video_frames_root, scene_id)
        if video_dir is None:
            print(f"[Skip] {scene_dir.name}: frames not found for {scene_id}.")
            continue

        try:
            process_scene(
                args=args,
                scene_result_dir=str(scene_dir),
                video_dir=video_dir,
                reasoning_model=reasoning_model,
                segmentation_model=segmentation_model,
                processor=processor,
                predictor=predictor,
            )
        except Exception as exc:
            print(f"[Error] {scene_dir.name}: {exc}")


if __name__ == "__main__":
    main(sys.argv[1:])
