# CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos [ICLR 2026]
> [**CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos**](https://arxiv.org/abs/2505.18561)      
> Shiu-hong Kao, Yu-Wing Tai, Chi-Keung Tang       
> [Project page](https://danielshkao.github.io/cot-rvs.html)

---
<img width="649" height="385" alt="image" src="https://github.com/user-attachments/assets/b462753c-0f75-4a87-9f2b-965696ab134d" />

## News

- [02/2026] We release our code and [T-ReasonVOS](T-ReasonVOS) dataset!
- [01/2026] CoT-RVS is accepted to ICLR 2026! &#127881;&#127881;
- [09/2025] Preprint is abailable on [arXiv](https://arxiv.org/abs/2505.18561).


## Installation

Our code is tested on [Ubuntu 22.04](https://releases.ubuntu.com/jammy/) with with Python 3.10 and CUDA 12.2.
```
git clone https://github.com/DanielSHKao/CoT-RVS.git
cd CoT-RVS
conda create -n cot-rvs -y python=3.10
conda activate cot-rvs

# (Optional) Install LLaVA only for online extension of CoT-RVS
git clone https://github.com/haotian-liu/LLaVA && cd LLaVA
pip install -e .
cd ..

# This is required
pip install -r requirements.txt

# Install flash-attn. Our code is tested on flash_attn==2.7.3.
pip install flash-attn --no-build-isolation
# Install directly from pre-compiled wheel (we use this during the testing):
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Download Models

1. Download [SAM2 checkpoints](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints) as video processor. We use [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) in our paper.
2. Download Seg-Zero checkpoints from [Huggingface](https://huggingface.co/Ricky06662/Seg-Zero-7B).
3. (Optional) Download LLaVA1.5-7B from [Huggingface](https://huggingface.co/liuhaotian/llava-v1.5-7b) for Online Reasoning VOS.
4. (Optional) Download Gemma3-12B from [Huggingface](https://huggingface.co/google/gemma-3-12b-it) as an alternative if you don't have OpenAI API budgets. Note that this may lead to a performance degration.

## Inference

### Chat with CoT-RVS
To chat with CoT-RVS, please first fill in your OpenAI API Key in `config/openai.yaml`. Then, run the following command:
```
python chat_offline.py \
--sam2_model [path to SAM2 checkpoint] \
--segzero_model [path to SegZero checkpoint] \
--output_dir [output directory] \
--num_candidates 8
```

You may adjust the `--num_candidates` hyper-parameter, which changes the number of keyframe candidates as MLLM's input.

We offer a test sample in `test_sample/` and the expected output in `vis_output/`.

### Chat with CoT-RVS-(Online extension)
To chat with CoT-RVS-LLaVA for online Reasoning VOS, run the following command:
```
python chat_online.py \
--sam2_model [path to SAM2 checkpoint] \
--segzero_model [path to SegZero checkpoint] \
--llava_model [path to LLaVA checkpoint] \
--output_dir [output directory] \
--xi 10
```
You may adjust the hyper-parameter `--xi` to change LLaVA's intervention frequency.

### Replace API with Gemma3
For users without OpenAI API budgets, we encourage you to test CoT-RVS on different MLLMs. We provide a baseline with Gemma3-12B in our paper. To start with, we have to upgrade the `transformers` and `timm` library:
```
pip install -U transformers==4.50.0 timm
```
Run the following command to generate Gemma-3's chain of thoughts:
```
python run_gemma.py \
--gemma3_model [Gemma3 model directory] \
--output_dir [output directory] \
--num_candidates 8
```
A sample response using the test case in `test_sample\` is saved in `vis_output\Gemma-3-responses\`. Next, run the following command to segment the target and track over the entire video:
```
python seg_and_track.py \
--video_dir [input video dir] \
--mllm_response_path [path to Gemma3's response] \
--sam2_model [path to SAM2 checkpoint] \
--segzero_model [path to SegZero checkpoint] \
--output_dir [output directory] \
--num_candidates 8
```

## T-ReasonVOS Dataset
Refer to the [T-ReasonVOS](T-ReasonVOS) directory for more details.

## Citation
If you find this repository helpful, please consider citing:
```
@article{CoTRVS,
  title={CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos},
  author={Kao, Shiu-hong and Tai, Yu-Wing and Tang, Chi-Keung},
  journal={arXiv preprint arXiv:2505.18561},
  year={2025}
}
```
