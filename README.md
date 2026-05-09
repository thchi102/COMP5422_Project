# COMP5422 Project Pipeline

This repository combines `VLM-Grounder` and `CoT-RVS` to generate 3D bounding boxes from ScanNet scenes.

## Environment Setup

Setup the environment with the provided `requirements.txt`

## Overview

1. Download ScanNet and preprocess it with `VLM-Grounder`.
2. Build scene metadata (`scene_info`) and extract scene frames.
3. Run CoT-RVS on color frames to obtain mask frames.
4. Use `visual_grounder_new.py` to project masks into 3D and produce bounding boxes.
5. Evaluate and visualize predictions with scripts in `VLM-Grounder/tools`.

## 1) Prepare ScanNet and VLM-Grounder data

First, set up ScanNet and follow the data preprocessing steps in:

- `VLM-Grounder/README.md`

The goal of this stage is to produce:

- preprocessed ScanNet data used by `VLM-Grounder`
- `scene_info` files (for intrinsics/extrinsics and scene metadata)
- extracted per-scene frames (including color frames)

## 2) Generate mask frames with CoT-RVS

Use color frames from the preprocessed scenes as input to CoT-RVS.

Run:

- `CoT-RVS/run_chatgpt_scannet.py`
  ```bash
  python CoT-RVS/run_chatgpt_scannet.py \
      --config CoT-RVS/config.yaml \
      --scannet_path /path/to/scannet \
      --save_path ./scannet_output \
      --nr3d_csv /path/to/nr3d.csv \
      --num_candidates 8 \
      --output_dir ./vis_output \
      --max_tokens 2500 \
      --max_merged_size 1536
  ```

- `CoT-RVS/seg_and_track_scannet.py`
  ```bash
  python CoT-RVS/seg_and_track_scannet.py \
      --cot_results_dir ./scannet_output \
      --video_frames_root /path/to/scannet_frames \
      --sam2_model checkpoints/sam2.1_hiera_large.pt \
      --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
      --segzero_model checkpoints/Seg-Zero-7B \
      --num_candidates 8 \
      --mask_dir_name binary_masks \
      --video_name segmented.mp4 \
      --overwrite
  ```

These scripts produce mask outputs for the target object across the video frames. After video segmentation, run:

- `VLM-Grounder/tools/select_openai_triangulation_frames.py`
  ```bash
  python VLM-Grounder/tools/select_openai_triangulation_frames.py \
      --input-root ./scannet_output \
      --nr3d-csv /path/to/nr3d.csv \
      --posed-roots /path/to/scannet_posed \
      --mask-dir-name binary_masks \
      --output-root ./scannet_top10_triangulation_openai \
      --min-area-ratio 0.01 \
      --num-frames 6 \
      --overwrite
  ```

This script selects 6 frames from the video mask frames which have the object visible.

## 3) Generate 3D bounding boxes

Use the masks from CoT-RVS together with scene calibration/depth information in:

- `VLM-Grounder/vlm_grounder/grounder/visual_grounder_new.py`

This stage projects 2D masked regions into 3D and aggregates them to get the final 3D bounding box.

## 4) Evaluate and visualize results

Use scripts in `VLM-Grounder/tools`

1. Preprocess the Nr3D dataset for evaluation
    
    `VLM-Grounder/tools/process_preprocessed_queries_bbox.py`
    ```bash
    python VLM-Grounder/tools/process_preprocessed_queries_bbox.py \
        --triangulation-root ./scannet_top10_triangulation_openai \
        --nr3d-csv /path/to/nr3d.csv \
        --posed-root ./scannet_top10_posed \
        --instance-data-root ./scannet_top10_instance_data \
        --scene-infos-pkl ./scannet_top10_instance_data/scenes_train_val_info.pkl \
        --output-json ./scannet_top10_triangulation_openai_bbox.json \
        --min-votes 2 \
        --z-near 1e-4 \
        --voxel-size 0.02 \
        --trim-quantile 0.01
    ```

2. Evaluation

    `VLM-Grounder/tools/evaluate_preprocessed_bboxes.py`
    ```bash
    python VLM-Grounder/tools/evaluate_preprocessed_bboxes.py \
        --pred-json ./scannet_top10_triangulation_openai_bbox.json \
        --nr3d-csv /path/to/nr3d.csv \
        --scene-infos-pkl ./scannet_top10_instance_data/scenes_train_val_info.pkl \
        --output-json ./scannet_top10_triangulation_openai_bbox_eval.json \
        --top-k 10
    ```

3. Save the top K visualization result in `.ply` formats

    `VLM-Grounder/tools/visualize_topk_bboxes.py`
    ```bash
    python VLM-Grounder/tools/visualize_topk_bboxes.py \
        --eval-json ./scannet_top10_triangulation_openai_bbox_eval.json \
        --pred-json ./scannet_top10_triangulation_openai_bbox.json \
        --scene-infos-pkl ./scannet_top10_instance_data/scenes_train_val_info.pkl \
        --scans-root ./scannet_top10/scans \
        --output-dir ./bbox_top10_visualizations \
        --top-k 10 \
        --edge-radius 0.015
    ```
4. Evaluation scripts for the evaluation against VLM-grouder can be found in the `evaluation` folder. Follow the `README.md` in there to perform evaluation.

## Notes

- Follow each subproject's environment and dependency instructions in its own `README.md`.
- Keep data paths consistent across both projects (`VLM-Grounder` and `CoT-RVS`) so masks, frames, and scene metadata match.
