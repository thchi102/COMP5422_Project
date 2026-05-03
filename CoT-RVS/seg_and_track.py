import argparse
import os
import sys
import yaml

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

from utils.util import *
from infer.prompt_api import *

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from infer.seg_zero import generate_mask


COLOR_MAP = [(255,0,0),(0,255,255),(128,255,0),(187, 19, 208),(222, 148, 80),(147, 71, 238),(98, 43, 249)]
def parse_args(args):
    parser = argparse.ArgumentParser(description="CoT-RVS chat")
    parser.add_argument("--video_dir", type=str)
    parser.add_argument("--mllm_response_path", type=str)
    parser.add_argument("--sam2_model", default="checkpoints/sam2.1_hiera_large.pt", type=str)
    parser.add_argument("--sam2_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml", type=str)
    parser.add_argument("--segzero_model", default="Ricky06662/Seg-Zero-7B", type=str)
    parser.add_argument("--output_dir", default="./vis_output",type=str)
    parser.add_argument("--num_candidates", default=8, type=int)
    return parser.parse_args(args)

def seg_zero_objects(args,reasoning_model, segmentation_model, processor,image_paths,target_objects,sample_every,save_mask=False,save_name=""):
    masks = []
    ann_frame_idx = []
    for target in target_objects:
        ann_keyframe = (target["keyframe"]-1)*sample_every
        object_desc = target["object_description"]
        prompt = f"Please segment {object_desc}."
        image_path = image_paths[ann_keyframe]
        mask = generate_mask(args,reasoning_model,segmentation_model,processor,image_path,prompt,save_mask=save_mask,save_name=save_name+str(target["object_index"]))
        masks.append(mask)
        ann_frame_idx.append(ann_keyframe)
    return masks, ann_frame_idx
    
def main(args):
    args = parse_args(args)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.makedirs(os.path.join(args.output_dir,"predictions"),exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,"keyframes"),exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    
    print("Loading Seg-Zero ...")
    reasoning_model_path = args.segzero_model
    segmentation_model_path = "facebook/sam2-hiera-large"
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    segmentation_model = SAM2ImagePredictor.from_pretrained(segmentation_model_path)
    reasoning_model.eval()
    processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")
    
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
        
        
    predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_model, device=device)
    
    
    video_dir = args.video_dir
    video_name = video_dir.split("/")[-1] if video_dir[-1]!="/" else video_dir.split("/")[-2]
    
    if (not os.path.exists(video_dir)):
        raise RuntimeError("Input video path is not found.")
    
    mllm_response_path = args.mllm_response_path
    save_name = os.path.split(mllm_response_path)[1][:-4]
    with open(mllm_response_path) as f:
        answer = f.read()
        
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1].split('frame')[-1]))
    
    T = len(frame_names)
    sample_every = (T-1)//args.num_candidates+1
    

    target_objects = parse_gpt_output(answer)
    all_image_paths = [os.path.join(video_dir,path) for path in frame_names]
    masks, ann_frame_idx = seg_zero_objects(args,reasoning_model,segmentation_model,processor,all_image_paths, target_objects, sample_every, save_mask=False, save_name=save_name)
    inference_state = predictor.init_state(video_path=video_dir,offload_video_to_cpu=True,async_loading_frames=True)
    predictor.reset_state(inference_state)
    
    for object_id, (binary_mask,frame_idx) in enumerate(zip(masks,ann_frame_idx)):
        # the frame index we interact with
        
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=object_id+1,
            mask=binary_mask,
        )
        
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    original_imgs = []
    seg_imgs = []
    pbar = tqdm(range(len(frame_names)),desc="processing output masks")
    for out_frame_idx in pbar:
        image_path = os.path.join(video_dir, frame_names[out_frame_idx])
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_imgs.append(image_np)
        
        save_img = image_np.copy()
        mask_taken=np.zeros_like(image_np[:,:,0]).astype(np.bool_)
        for object_idx in range(len(target_objects)):
            pred_mask = video_segments[out_frame_idx][object_idx+1][0]
            pred_mask = np.logical_and(pred_mask, np.logical_not(mask_taken))
            mask_taken = np.logical_or(mask_taken,pred_mask)
            save_img[pred_mask] = (save_img*0.4+pred_mask[:, :, None].astype(np.uint8) * np.array(COLOR_MAP[object_idx]) * 0.6)[pred_mask]
            
        seg_imgs.append(save_img.astype(np.uint8))
    save_video(seg_imgs,os.path.join(args.output_dir,"predictions",f"{save_name}.mp4"))


if __name__ == "__main__":
    main(sys.argv[1:])
    