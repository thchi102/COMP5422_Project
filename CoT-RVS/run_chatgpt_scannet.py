import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from openai import AzureOpenAI
from PIL import Image
from tqdm import tqdm

from infer.prompt_api import local_image_to_data_url, prompt_openai
from infer.prompt_gemma import save_answer
from utils.util import merge_keyframe

NUM_EXAMPLES = 100


def parse_args(args):
    parser = argparse.ArgumentParser(description="CoT-RVS ScanNet via Azure OpenAI (vision + CoT prompt)")
    default_config = Path(__file__).resolve().parent / "config" / "openai.yaml"
    parser.add_argument(
        "--config",
        default=str(default_config),
        type=str,
        help="YAML with api_key, api_version, azure_endpoint, and model (deployment name).",
    )
    parser.add_argument(
        "--azure_model",
        default=None,
        type=str,
        help="Override deployment name from config['model'].",
    )
    parser.add_argument("--scannet_path", required=True, type=str, help="Path to ScanNet dataset")
    parser.add_argument("--save_path", required=True, type=str, help="Path to save results")
    parser.add_argument("--nr3d_csv", required=True, type=str, help="Path to nr3d.csv")
    parser.add_argument("--num_candidates", default=8, type=int)
    parser.add_argument("--output_dir", default="./vis_output", type=str)
    parser.add_argument(
        "--max_tokens",
        default=2500,
        type=int,
        help="Azure chat completion max_tokens (maps to CoT length budget).",
    )
    parser.add_argument("--max_merged_size", default=1536, type=int)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.save_path, exist_ok=True)

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Azure config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        openai_docs = yaml.safe_load(f)

    api_key = openai_docs.get("api_key")
    azure_endpoint = openai_docs.get("azure_endpoint")
    if not api_key or not azure_endpoint:
        raise ValueError(f"Set api_key and azure_endpoint in {config_path}")

    deployment = args.azure_model or openai_docs.get("model")
    if not deployment:
        raise ValueError("Set model in config or pass --azure_model (Azure deployment name).")

    client = AzureOpenAI(
        api_key=api_key,
        api_version=openai_docs.get("api_version", "2024-06-01"),
        azure_endpoint=azure_endpoint,
    )
    print(f"Azure OpenAI deployment: {deployment}")

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
        assignment_id = str(row["assignmentid"])
        scene_id = str(row["scan_id"])
        utterance = str(row["utterance"])

        save_dir = os.path.join(args.save_path, assignment_id)
        if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "answer.txt")):
            continue

        video_dir = os.path.join(args.scannet_path, scene_id, "color")
        if not os.path.exists(video_dir):
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

        sampled_frame_names = frame_names[::sample_every][: args.num_candidates]
        keyframes = [Image.open(os.path.join(video_dir, path)) for path in sampled_frame_names]

        os.makedirs(save_dir, exist_ok=True)

        original_output_dir = args.output_dir
        args.output_dir = save_dir

        merged_result_path = merge_keyframe(
            args, scene_id, keyframes, max_size=args.max_merged_size
        )

        data_url = local_image_to_data_url(merged_result_path)
        answer = prompt_openai(
            client=client,
            model=deployment,
            data_url=data_url,
            query=utterance,
            num_keyframes=len(keyframes),
            max_tokens=args.max_tokens,
        )

        save_answer(
            utterance,
            answer,
            os.path.join(save_dir, "answer.txt"),
            response_label="Azure ChatGPT response",
        )

        args.output_dir = original_output_dir
        del keyframes
        i += 1


if __name__ == "__main__":
    main(sys.argv[1:])
