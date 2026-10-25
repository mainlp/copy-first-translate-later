# Evaluation

This folder provides the evaluation analyses:
- the behavioral analysis of the word-level-translation,
- MultiBLiMP through [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness), and Chinese and Japanese as custom tasks (in `custom_lm_eval_tasks/`),
- the logit-lens analysis.

The ExPLAIND loss decomposition can be found [here](https://github.com/mainlp/explaind/tree/main/scaled_experiments/llama1b).

## Setup

See `requirements.txt`. In order to run custom tasks we need lm_eval>=0.4., which is pre-release:

```
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
```
### Data

The word-level translation dataset is released at https://huggingface.co/datasets/mainlp/word-level-translations. 

To download it to `data/word_level_translations` (`WLT_DATA_PATH` in `src/constants.py`), as well as build the prompts, and prepare BLiMP:

```
python src/prepare_eval_data.py
```

## Evaluation

All 200 checkpoints are on [Hugging Face](https://huggingface.co/mainlp/copy-first-translate-later) (one branch per step). To evaluate all of them:

```
python src/evaluate_all_checkpoints.py --run_name ours --mode hf --model_name mainlp/copy-first-translate-later
```

Use `--step` / `--freq` to evaluate a subset. Each checkpoint is downloaded into the Hugging Face cache (~6.7GB each).

## Plotting

Plots are written to `figures/<run_name>/`, with preliminary results cached in `cache/` (use `--force` to recompute).
Use `--accuracy_mode ci` for the case-insensitive WLT accuracy, as in the paper.

### WLT metrics
```
python src/plot_wlt_metrics.py --run_name <run_name> --all --accuracy_mode ci
  ```
You can specify the metrics via `--metrics accuracy|probability|probability_normalized`, add `--plot_steps 100000` for the accuracy heatmap at a step, `--symmetry` for the symmetry plots, `--plot_output_classification` for the output classification plots).  The option `--plot_output_classification -translation_success_composition` plots the stacked plot which decomposes the output. 
  
### Logit lens
```
python src/plot_logit_lens.py --run_name <run_name>  --plot_steps 100000  --score_type scores_normed  --accuracy_mode ci  --avgs   --groupings collapsed_other
```

Over-time plots use `--plot_grouped_time``, the layer-wise margin uses `--margin_layer_diff`. The subtoken-overlap groups over time use `--groupings jaccard --plot_tf_langs copy,partial,none`. The layer-wise margin uses `--margin_layer_diff`.

### MultiBLiMP
```
python src/plot_mblimp.py --run_name <run_name>
```

See `--help` of each script for all options.

## Environment Variables

- `OUTPUT_DIR`: default output root used by evaluation scripts (default: `""`)
- `MODEL_CONFIG_DIR`: directory containing training model config files, e.g. `llama1b.json` (default: `../training/configs/models/`)
- `WANDB_PROJECT`: W&B project name (default: `eval`)
- `WANDB_ENTITY`: optional W&B entity
