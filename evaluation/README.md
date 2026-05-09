# COMP5422 VLM-Grounder Benchmark Code

This package contains the code used to run VLM-Grounder as the benchmark for our COMP5422 project. Our teammates' novel pipeline is submitted separately; this folder documents and reproduces the VLM-Grounder baseline that we used for evaluation on the shared NR3D top-10 subset.

## Contents

```text
data/
  nr3d_top10.csv                 # 100-query evaluation subset
  nr3d_top10_manifest.json       # Manifest consumed by the wrapper
  schema_v1.json                 # Output schema used by the evaluator
scripts/
  run_vlm_grounder.py            # End-to-end VLM-Grounder wrapper
  check_nr3d_top10_readiness.py  # Dataset / cache readiness checker
  convert_manifest_to_vg_csv.py  # Manifest -> VLM-Grounder CSV converter
  build_referit3d_manifest.py    # Optional ReferIt3D manifest builder
  evaluate.py                    # Unified evaluator
  validate_schema.py             # Schema validator
  lib/                           # IoU, GT bbox, matching, Chamfer helpers
docs/
  evaluation_content.md          # Report-ready evaluation draft
  evaluation_memory.md           # Run IDs, metrics, and caveats
requirements.txt                 # Evaluator Python dependencies
scannet_download.py              # Official ScanNet download helper
```

Large external assets are intentionally not included:

- ScanNet raw scans and annotations
- ReferIt3D full `nr3d.csv`
- extracted posed images
- VLM-Grounder upstream repository clone
- vLLM/Gemma weights and other model checkpoints
- output folders and cache `.pkl` files

These files are too large for code submission and are provided separately here:

https://hkustconnect-my.sharepoint.com/:f:/g/personal/hra_connect_ust_hk/IgBb4RJZUtskRozl5FslQNycAS9cFM_dc2kkCcZ11qgtvok?e=juGzcQ

## Environment

The final benchmark was run on an SSH server with:

- Python 3.10
- PyTorch with CUDA
- VLM-Grounder dependencies
- vLLM serving `google/gemma-4-E4B-it`
- ScanNet scene files under `data/ScanNet/scans`
- ReferIt3D NR3D CSV under `data/ReferIt3D/nr3d.csv`

For the lightweight evaluator only:

```bash
pip install -r requirements.txt
```

For the full VLM-Grounder benchmark, first clone and install the upstream VLM-Grounder repository into:

```text
scripts/vlm-grounder-repo/
```

The wrapper expects this path.

## Required Data Layout

Place the full ReferIt3D NR3D CSV here:

```text
data/ReferIt3D/nr3d.csv
```

Place ScanNet scene assets here:

```text
data/ScanNet/scans/<scene_id>/
```

For each of the 10 evaluation scenes, the wrapper expects the usual ScanNet files:

```text
<scene_id>.sens
<scene_id>.aggregation.json
<scene_id>_vh_clean_2.0.010000.segs.json
<scene_id>_vh_clean_2.labels.ply
<scene_id>_vh_clean_2.ply
<scene_id>.txt
```

The 10 scenes are:

```text
scene0090_00 scene0117_00 scene0143_00 scene0258_00 scene0333_00
scene0443_00 scene0444_00 scene0546_00 scene0558_00 scene0658_00
```

## Readiness Check

From the project root:

```bash
python scripts/check_nr3d_top10_readiness.py
```

This checks the top-10 CSV, manifest, VLM-Grounder clone, ScanNet scene files, posed images, and cached scene/matching files.

## Running the Benchmark

If this is the first time running on a server, run without `--skip-one-time` so the wrapper can prepare posed images, scene info, and PATS matching:

```bash
RUN_ID=run_vlm_nr3d_top10_$(date +%Y%m%d_%H%M%S)
PY=.envs/vlm_grounder/bin/python

$PY scripts/run_vlm_grounder.py \
  --manifest data/nr3d_top10_manifest.json \
  --out outputs/$RUN_ID \
  --run_id $RUN_ID \
  --scenes scene0090_00 scene0117_00 scene0143_00 scene0258_00 scene0333_00 scene0443_00 scene0444_00 scene0546_00 scene0558_00 scene0658_00 \
  --llm-backend vllm \
  --vllm-model google/gemma-4-E4B-it \
  --vllm-quantization bitsandbytes \
  --vllm-gpu-memory-utilization 0.50
```

After one-time assets are prepared, reruns can use:

```bash
RUN_ID=run_vlm_nr3d_top10_jsonretry_$(date +%Y%m%d_%H%M%S)
PY=.envs/vlm_grounder/bin/python

$PY scripts/run_vlm_grounder.py \
  --manifest data/nr3d_top10_manifest.json \
  --out outputs/$RUN_ID \
  --run_id $RUN_ID \
  --scenes scene0090_00 scene0117_00 scene0143_00 scene0258_00 scene0333_00 scene0443_00 scene0444_00 scene0546_00 scene0558_00 scene0658_00 \
  --llm-backend vllm \
  --vllm-model google/gemma-4-E4B-it \
  --vllm-quantization bitsandbytes \
  --vllm-gpu-memory-utilization 0.50 \
  --skip-one-time
```

The wrapper applies compatibility patches to VLM-Grounder at runtime:

- reads API keys from environment variables
- avoids proxy-related `KeyError`
- retries malformed local-VLM JSON responses instead of aborting the full run
- skips unsupported `--max_fallback_images` when using older VLM-Grounder checkouts

These patches do not replace VLM-Grounder's pipeline stages; they make the local benchmark execution robust to vLLM/Gemma backend behavior.

## Evaluating a Run

After a run finishes:

```bash
python scripts/evaluate.py \
  --run outputs/$RUN_ID \
  --nr3d data/ReferIt3D/nr3d.csv \
  --scannet data/ScanNet/scans
```

The evaluator writes:

```text
outputs/<run_id>/report.json
```

## Final Run Used in Report

The final comparable VLM-Grounder run was:

```text
run_vlm_nr3d_top10_jsonretry3_20260508_211150
```

It completed Stage 6 over all 100 queries, produced 100 raw result items, exported 95 valid bounding boxes, and skipped 5 queries without usable `gpt_pred_bbox`.

VLM-Grounder internal metrics:

```text
Acc@0.25 = 10.0%
Acc@0.50 = 4.0%
```

Unified evaluator metrics on the exported boxes:

```text
95 evaluated
NR3D Mode-B grounding accuracy = 3.2%
Acc@0.25 = 0.0%
Acc@0.50 = 0.0%
```

See `docs/evaluation_memory.md` for the comparison against our team's method and the caveat about VLM-Grounder's internal bbox representation versus our exported-schema evaluator.

## Notes

The earlier VLM-Grounder run `run_20260507_075317` should not be used for final comparison because it used a different CSV split (`data/final_test/nr3d_subset_final_test_vg_input.csv`) rather than `data/nr3d_top10.csv`.
