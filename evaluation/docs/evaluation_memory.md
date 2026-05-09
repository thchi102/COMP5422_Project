# Evaluation Memory

This note records the final evaluation facts for the report.

## Dataset

- Evaluation subset: `data/nr3d_top10.csv`
- Size: 100 NR3D queries
- Scenes: `scene0090_00`, `scene0117_00`, `scene0143_00`, `scene0258_00`, `scene0333_00`, `scene0443_00`, `scene0444_00`, `scene0546_00`, `scene0558_00`, `scene0658_00`

## Our Team Method

- Output file: `scannet_top10_triangulation_openai_bbox_colorK_eval.json`
- Queries: 100
- Predictions found: 79
- Valid IoU records: 78
- Mean IoU over valid records: 0.089194646
- Acc@0.25 over valid records: 12.82%
- Acc@0.50 over valid records: 7.69%
- All-query equivalent Acc@0.25: 10.0%
- All-query equivalent Acc@0.50: 6.0%
- Best scene: `scene0333_00`
  - Mean IoU: 0.3803
  - Acc@0.25: 50%
  - Acc@0.50: 40%

## VLM-Grounder Final Run

- Final comparable run ID: `run_vlm_nr3d_top10_jsonretry3_20260508_211150`
- Input: same 100-query `nr3d_top10` subset
- Stage 3 query analysis: completed 100/100
- Stage 4 YOLO-World detection: completed for 10 scenes / 214 images
- Stage 5 view pre-selection: completed 100/100
- Stage 6 visual grounding: completed 100/100 after JSON-retry hotfix
- Raw Stage 6 result items: 100
- Schema bbox exports: 95/100
- Skipped/no usable bbox: 5/100

VLM-Grounder internal metrics:

- Acc@0.25: 10.0%
- Acc@0.50: 4.0%

Unified evaluator metrics on exported VLM-Grounder boxes:

- Evaluated samples: 95
- NR3D Mode-B grounding accuracy: 3.2%
- Acc@0.25: 0.0%
- Acc@0.50: 0.0%

Important caveat: VLM-Grounder's internal evaluator reports nonzero IoU, while our exported-schema evaluator reports 0.0% IoU. This likely reflects a bbox representation/export mismatch between VLM-Grounder's internal result format and our schema/evaluator path. For final comparison, present the unified-evaluator result as the common metric and mention the VLM-Grounder internal metric as diagnostic.

Evidence files on SSH server:

- `outputs/run_vlm_nr3d_top10_jsonretry3_20260508_211150/report.json`
- `outputs/run_vlm_nr3d_top10_jsonretry3_20260508_211150/run_summary.json`
- `outputs/run_vlm_nr3d_top10_jsonretry3_20260508_211150/reconstruction/_skipped_queries.json`
- `scripts/vlm-grounder-repo/outputs/visual_grounding/run_vlm_nr3d_top10_jsonretry3_20260508_211150/vg_input_prompt_v2_updated_with_images_selected_diffconf_and_pkl_google/gemma-4-E4B-it_promptv3_results.json`

## Excluded Earlier VLM-Grounder Run

- Run ID: `run_20260507_075317`
- Reason for exclusion: wrong CSV/split
- It used `data/final_test/nr3d_subset_final_test_vg_input.csv`, not `data/nr3d_top10.csv`
- It had 100 rows but only 8 scenes:
  - `scene0081_00`, `scene0117_00`, `scene0143_00`, `scene0175_00`, `scene0268_00`, `scene0333_00`, `scene0522_00`, `scene0637_00`
- Do not use it for final method comparison.

## Comparison Summary

| Method | Dataset | Completed / valid predictions | Acc@0.25 | Acc@0.50 | Notes |
|---|---:|---:|---:|---:|---|
| Our team | 100 NR3D top10 queries | 79 predictions, 78 valid IoU | 10.0% all-query / 12.82% valid-only | 6.0% all-query / 7.69% valid-only | Lower coverage, better strict IoU |
| VLM-Grounder internal | 100 NR3D top10 queries | 100 completed, 95 exported boxes | 10.0% | 4.0% | Internal VLM-Grounder metric |
| VLM-Grounder unified evaluator | 95 exported boxes | 95 evaluated | 0.0% | 0.0% | Likely export/evaluator bbox mismatch |

## Suggested Report Text

On the shared 100-query NR3D top-10 subset, our method produced 79 predictions with 78 valid IoU records, achieving 10.0% Acc@0.25 and 6.0% Acc@0.50 when normalized over all queries. VLM-Grounder completed all 100 queries after adding retry handling for malformed local-VLM JSON responses and exported 95 valid bounding boxes. Its internal evaluator reported 10.0% Acc@0.25 and 4.0% Acc@0.50, while our unified exported-schema evaluator measured 0.0% Acc@0.25/0.50 on the 95 exported boxes, likely due to a mismatch between VLM-Grounder's internal bbox representation and our evaluator schema. Overall, VLM-Grounder had higher output coverage, while our method matched its loose-threshold internal accuracy and outperformed it at the stricter 0.50 IoU threshold.
