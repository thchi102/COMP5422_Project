import argparse
import os
import sys

import cv2
import numpy as np
import torch
import os

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from pathlib import Path
from tqdm import tqdm
from utils.util import save_video

from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from infer.prompt_llava import *
from infer.seg_zero import generate_mask
COLOR_MAP = [(255,0,0),(0,255,0),(0,0,255),(222, 148, 80),(147, 71, 238),(187, 19, 208),(98, 43, 249)]

def parse_args(args):
    parser = argparse.ArgumentParser(description="CoT-RVS chat")
    parser.add_argument("--sam2_model", default="checkpoints/sam2.1_hiera_large.pt",type=str)
    parser.add_argument("--sam2_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml", type=str)
    parser.add_argument("--llava_model", default="liuhaotian/llava-v1.5-7b", type=str)    
    parser.add_argument("--segzero_model", default="Ricky06662/Seg-Zero-7B", type=str)    
    parser.add_argument("--output_dir", default="./vis_output",type=str)
    parser.add_argument("--xi",default=10,type=int)
    return parser.parse_args(args)



def main(args):
    args = parse_args(args)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.makedirs(os.path.join(args.output_dir,"online-predictions"),exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,"keyframes"),exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    
    mllm_model_path = args.llava_model
    mllm_model = load_pretrained_model(
        model_path=mllm_model_path,
        model_base=None,
        model_name=get_model_name_from_path(mllm_model_path)
    )
                

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
    
    while True:
        video_dir = input('Please input the directory of image sequence: ')
        user_query = input('Please input the segmentation query: ')
        
        
        if (not os.path.exists(video_dir)):
            print("Input path not found.")
            continue
                
        save_name = input("Please input the save name: ")
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1].split('frame')[-1]))
        
        inference_state = predictor.init_state(video_path=video_dir,offload_video_to_cpu=True,async_loading_frames=True)
        predictor.reset_state(inference_state)
        
        # gpt_device = vl_gpt.device
        original_imgs = []
        seg_imgs = []
        video_segments = {}
        curr_keyframe = None
        for frame_idx in range(len(frame_names)):
            image_path = os.path.join(video_dir, frame_names[frame_idx])
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_imgs.append(image_np.astype(np.uint8))
            save_img = image_np.copy()
            
            ask_gpt = (frame_idx%args.xi==0)
            if ask_gpt:
                answer = ask_llava(model = mllm_model, image_path=image_path, query=user_query, model_path=mllm_model_path)
                # print(answer)
                use_for_keyframe = parse_gpt_output(answer)
                if use_for_keyframe:
                    predictor.reset_state(inference_state)
                    binary_mask = generate_mask(args,reasoning_model,segmentation_model,processor,image_path,user_query,save_mask=False,save_name=save_name+str(frame_idx)).astype(np.uint8)
                    
                    curr_keyframe = binary_mask
                    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=1,
                        mask=binary_mask,
                    )
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,start_frame_idx=frame_idx,max_frame_num_to_track=0):
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    pred_mask = video_segments[frame_idx][1][0]
                    save_img[pred_mask] = (save_img*0.4 + pred_mask[:, :, None].astype(np.uint8) * np.array(COLOR_MAP[0]) * 0.6)[pred_mask]
                    seg_imgs.append(save_img.astype(np.uint8))
                elif curr_keyframe is None:
                    seg_imgs.append(save_img.astype(np.uint8))
                else:
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,start_frame_idx=frame_idx,max_frame_num_to_track=0):
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    pred_mask = video_segments[frame_idx][1][0]
                    save_img[pred_mask] = (save_img*0.4 + pred_mask[:, :, None].astype(np.uint8) * np.array(COLOR_MAP[0]) * 0.6)[pred_mask]
                    seg_imgs.append(save_img.astype(np.uint8))
            else:
                if curr_keyframe is None:
                    seg_imgs.append(save_img.astype(np.uint8))
                else:
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,start_frame_idx=frame_idx,max_frame_num_to_track=0):
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    pred_mask = video_segments[frame_idx][1][0]
                    save_img[pred_mask] = (save_img*0.4 + pred_mask[:, :, None].astype(np.uint8) * np.array(COLOR_MAP[0]) * 0.6)[pred_mask]
                    seg_imgs.append(save_img.astype(np.uint8))
        

            
        save_video(seg_imgs,os.path.join(args.output_dir,"online-predictions",f"{save_name}.mp4"))
    

if __name__ == "__main__":
    main(sys.argv[1:])
