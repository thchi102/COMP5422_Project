import argparse
import os
import sys
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from utils.util import merge_keyframe
from infer.prompt_gemma import prompt_gemma, save_answer
from transformers import AutoProcessor, AutoModelForCausalLM

NUM_EXAMPLES = 100

def parse_args(args):
    parser = argparse.ArgumentParser(description="CoT-RVS-Gemma ScanNet")
    parser.add_argument("--gemma3_model", default="google/gemma-4-E4B-it", type=str)
    parser.add_argument("--scannet_path", required=True, type=str, help="Path to ScanNet dataset")
    parser.add_argument("--save_path", required=True, type=str, help="Path to save results")
    parser.add_argument("--nr3d_csv", required=True, type=str, help="Path to nr3d.csv")
    parser.add_argument("--num_candidates", default=8, type=int)
    parser.add_argument("--output_dir", default="./vis_output", type=str)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--max_merged_size", default=1536, type=int)
    parser.add_argument("--gpu_id", default=0, type=int)
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.makedirs(args.save_path, exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    gemma_model_id = args.gemma3_model
    gemma_model = AutoModelForCausalLM.from_pretrained(
        gemma_model_id, device_map="auto"
    ).eval()
    gemma_processor = AutoProcessor.from_pretrained(gemma_model_id)

    
    
    df = pd.read_csv(args.nr3d_csv)
    scene_root = args.scannet_path

    valid_scene_ids = {
        entry
        for entry in os.listdir(scene_root)
        if os.path.isdir(os.path.join(scene_root, entry))
    }
    df["scan_id"] = df["scan_id"].astype(str)
    df = df[df["scan_id"].isin(valid_scene_ids)].copy()
    print(f"Filtered nr3d rows to {len(df)} using scenes in {scene_root}")

    df = df[:NUM_EXAMPLES]
    print(f"Using {NUM_EXAMPLES} examples")
    df.to_csv(os.path.join(args.save_path, "nr3d_subset.csv"), index=False)
    print(f"Saved subset to {os.path.join(args.save_path, 'nr3d_subset.csv')}")
    
    i = 0
    for idx, row in tqdm(df.iterrows(), total=NUM_EXAMPLES):
        if i >= NUM_EXAMPLES:
            break
        assignment_id = str(row['assignmentid'])
        scene_id = str(row['scan_id'])
        utterance = str(row['utterance'])
        
        save_dir = os.path.join(args.save_path, assignment_id)
        if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "answer.txt")):
            continue
            
        # video_dir = os.path.join(args.scannet_path, "posed_images", scene_id, "color")
        # video_dir = os.path.join(args.scannet_path, scene_id)
        video_dir = os.path.join(args.scannet_path, scene_id, "color")
        if not os.path.exists(video_dir):
            # print(f"Warning: Video directory not found: {video_dir}")
            continue
            
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        
        if not frame_names:
            print(f"Warning: No frames found in {video_dir}")
            continue
            
        try:
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except ValueError:
            frame_names.sort()
            
        T = len(frame_names)
        sample_every = max(1, (T - 1) // args.num_candidates + 1)
        
        sampled_frame_names = frame_names[::sample_every][:args.num_candidates]
        keyframes = [Image.open(os.path.join(video_dir, path)) for path in sampled_frame_names]
        
        os.makedirs(save_dir, exist_ok=True)
        
        original_output_dir = args.output_dir
        args.output_dir = save_dir
        
        merged_result_path = merge_keyframe(
            args, scene_id, keyframes, max_size=args.max_merged_size
        )
        
        answer = prompt_gemma(
            model=gemma_model,
            processor=gemma_processor,
            image_path=merged_result_path,
            query=utterance,
            num_keyframes=len(keyframes),
            max_new_tokens=args.max_new_tokens,
        )
        
        save_answer(utterance, answer, os.path.join(save_dir, "answer.txt"))
        
        args.output_dir = original_output_dir

        i+=1

if __name__ == "__main__":
    main(sys.argv[1:])
