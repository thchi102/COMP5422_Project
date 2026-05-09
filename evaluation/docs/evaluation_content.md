# Evaluation Content Draft

This file is a report-ready draft for `sec/4_experiment.tex`. It is written in Markdown for easier editing, with a LaTeX table included near the end.

## Where This Fits

`main.tex` currently includes:

```tex
\input{sec/4_experiment}
```

and `sec/4_experiment.tex` currently only contains:

```tex
\section{Experiment}
```

The content below can be adapted into that section.

## Experiment Section Draft

### Experimental Setup

We evaluate our video-to-3D grounding pipeline on a curated subset of the Nr3D and ScanNet datasets. The evaluation subset contains 100 free-form Nr3D language queries over 10 ScanNet scenes: `scene0090_00`, `scene0117_00`, `scene0143_00`, `scene0258_00`, `scene0333_00`, `scene0443_00`, `scene0444_00`, `scene0546_00`, `scene0558_00`, and `scene0658_00`. Each query refers to a target object instance in a ScanNet scene, and the task is to predict a 3D axis-aligned bounding box for the described object.

We compare our method against VLM-Grounder, a recent zero-shot 3D visual grounding pipeline that performs image selection, object localization, and multi-view projection using VLM reasoning. To ensure a fair comparison, VLM-Grounder was rerun on the same 100-query subset used by our method. The earlier VLM-Grounder run `run_20260507_075317` was excluded because it used a different CSV split, `data/final_test/nr3d_subset_final_test_vg_input.csv`, rather than `data/nr3d_top10.csv`.

### Metrics

We report 3D localization accuracy using standard IoU thresholds. A prediction is counted as correct at threshold `t` if the 3D IoU between the predicted bounding box and the ground-truth target box is at least `t`. We report `Acc@0.25` and `Acc@0.50`, where `Acc@0.25` measures coarse 3D localization and `Acc@0.50` measures stricter box alignment. For our method, we report both the valid-only accuracy over predictions with valid IoU records and the all-query equivalent accuracy over all 100 queries.

### VLM-Grounder Execution Details

Running VLM-Grounder with the local `google/gemma-4-E4B-it` vLLM backend required a small robustness fix. The original Stage 6 visual grounding loop could terminate early when the local VLM returned malformed JSON. This is a backend compatibility issue rather than a change to the grounding algorithm: VLM-Grounder already retries failed API calls, invalid image selections, and invalid bounding-box selections, but malformed JSON responses were not consistently treated as retryable failures. We therefore added retry handling so that a malformed JSON response fails only the current query attempt instead of aborting the full 100-query benchmark.

After this fix, VLM-Grounder completed all 100 queries on the shared subset. It produced 100 raw Stage 6 result records and exported 95 valid bounding-box predictions, with 5 queries skipped due to missing usable `gpt_pred_bbox`.

### Quantitative Results

Our method produced 79 predictions, of which 78 had valid IoU records. On the all-query denominator, our method achieved 10.0% Acc@0.25 and 6.0% Acc@0.50. Among valid predictions only, it achieved 12.82% Acc@0.25 and 7.69% Acc@0.50, with a mean IoU of 0.0892. The strongest scene-level result was obtained on `scene0333_00`, where our method reached 50% Acc@0.25 and 40% Acc@0.50.

VLM-Grounder produced higher output coverage, exporting 95 valid boxes out of 100 queries. Its internal evaluator reported 10.0% Acc@0.25 and 4.0% Acc@0.50. Under our unified exported-schema evaluator, however, VLM-Grounder's exported boxes obtained 0.0% Acc@0.25 and 0.0% Acc@0.50 over 95 evaluated samples. This discrepancy suggests a mismatch between VLM-Grounder's internal bounding-box representation/evaluation path and our schema-based evaluator. For this reason, we use the unified evaluator for direct comparison where possible, while also reporting VLM-Grounder's internal metrics as diagnostic evidence.

| Method | Queries | Valid Predictions | Acc@0.25 | Acc@0.50 | Notes |
|---|---:|---:|---:|---:|---|
| Ours | 100 | 79 predictions / 78 valid IoU | 10.0% all-query / 12.82% valid-only | 6.0% all-query / 7.69% valid-only | Lower coverage, stronger strict-threshold accuracy |
| VLM-Grounder internal | 100 | 95 exported boxes | 10.0% | 4.0% | Internal VLM-Grounder evaluation |
| VLM-Grounder unified evaluator | 100 | 95 evaluated boxes | 0.0% | 0.0% | Likely export/evaluator bbox mismatch |

### Analysis

The results show a trade-off between coverage and precision. VLM-Grounder generated predictions for more queries, exporting valid boxes for 95% of the benchmark. Our method generated fewer predictions, but achieved a higher strict-threshold score: 6.0% Acc@0.50 over all queries compared with VLM-Grounder's 4.0% internal Acc@0.50. At the looser 0.25 IoU threshold, both methods reached 10.0% when normalized over the full 100-query subset.

Qualitatively, our method benefits when the CoT-RVS stage selects and segments the correct target object, because the downstream multi-view projection can produce tighter 3D boxes from mask evidence. However, the pipeline remains sensitive to segmentation errors, missing masks, and frame-selection failures, which reduce output coverage. VLM-Grounder is more complete in terms of producing predictions, but its dependence on local VLM image and bounding-box selection introduces failure modes such as malformed JSON responses, invalid image IDs, and cases where the selected image contains no detection for the parsed target class.

### Suggested Report Paragraph

On the shared 100-query Nr3D top-10 subset, our method produced 79 predictions with 78 valid IoU records, achieving 10.0% Acc@0.25 and 6.0% Acc@0.50 when normalized over all queries. VLM-Grounder completed all 100 queries after adding retry handling for malformed local-VLM JSON responses and exported 95 valid bounding boxes. Its internal evaluator reported 10.0% Acc@0.25 and 4.0% Acc@0.50, while our unified exported-schema evaluator measured 0.0% Acc@0.25/0.50 on the 95 exported boxes, likely due to a mismatch between VLM-Grounder's internal bbox representation and our evaluator schema. Overall, VLM-Grounder had higher output coverage, while our method matched its loose-threshold internal accuracy and outperformed it at the stricter 0.50 IoU threshold.

## LaTeX Table Draft

```tex
\begin{table}[t]
    \centering
    \small
    \begin{tabular}{lcccc}
        \toprule
        Method & Queries & Valid Pred. & Acc@0.25 & Acc@0.50 \\
        \midrule
        Ours (all-query) & 100 & 79 / 78 IoU & 10.0 & 6.0 \\
        Ours (valid-only) & 78 & 78 IoU & 12.82 & 7.69 \\
        VLM-Grounder (internal) & 100 & 95 boxes & 10.0 & 4.0 \\
        VLM-Grounder (unified eval.) & 95 & 95 boxes & 0.0 & 0.0 \\
        \bottomrule
    \end{tabular}
    \caption{Comparison on the shared 100-query Nr3D top-10 subset. Accuracies are percentages. VLM-Grounder internal metrics are reported from its own evaluator; the unified evaluator results use our exported-schema evaluation path.}
    \label{tab:main_results}
\end{table}
```

## Evidence Files

Our method:

- `scannet_top10_triangulation_openai_bbox_colorK_eval.json`

VLM-Grounder final comparable run:

- Run ID: `run_vlm_nr3d_top10_jsonretry3_20260508_211150`
- `outputs/run_vlm_nr3d_top10_jsonretry3_20260508_211150/report.json`
- `outputs/run_vlm_nr3d_top10_jsonretry3_20260508_211150/run_summary.json`
- `outputs/run_vlm_nr3d_top10_jsonretry3_20260508_211150/reconstruction/_skipped_queries.json`
- `scripts/vlm-grounder-repo/outputs/visual_grounding/run_vlm_nr3d_top10_jsonretry3_20260508_211150/vg_input_prompt_v2_updated_with_images_selected_diffconf_and_pkl_google/gemma-4-E4B-it_promptv3_results.json`
