
# Eval

We run our evaluations using the [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) plus custom benchmarks (WLT, MultiBLiMP), and the logit-lens analysis.

`custom_lm_eval_tasks/` holds `lm-evaluation-harness` YAML task defs for BLiMP variants not in the harness by default (Japanese, Chinese).

All commands below are run from this directory (`eval/`); data, cache and figure paths are relative to it.

## Setup

See `requirements.txt`. In order to run custom tasks we need lm_eval>=0.4., which is pre-release

`pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git`

### Data
The word-level translation dataset is released at https://huggingface.co/datasets/mainlp/word-level-translations. The following downloads it to `data/word_level_translations` (see `WLT_DATA_PATH` in `src/constants.py`), builds the prompts, and downloads/prepares MultiBLiMP, JBLiMP and ZhoBLiMP (all written to `data/`):

`python ./src/prepare_eval_data.py`

Optionally, check that the few-shot examples are aligned across language pairs:

`python ./src/check_eval_data.py`

## Evaluation

All 200 checkpoints are on the Hugging Face hub (one branch per step); this evaluates all of them, including logit lens:

`python ./src/evaluate_all_checkpoints.py --run_name ours --mode hf --model_name mainlp/copy-first-translate-later`

Use `--step` / `--freq` to evaluate a subset; each checkpoint is downloaded into the Hugging Face cache (~6.7GB each).

## Plotting

Plots are written to `figures/<run_name>/`; collated results are cached in `cache/` (use `--force` to recompute).

To plot over tokens seen instead of steps (`--x_axis tokens`, as in the paper), first build the step-to-tokens mapping:

`python ./src/collate_fn.py --run_name <run_name> --collate tokens`

This reads a per-checkpoint `toks_processed.csv`, which only our internal training runs wrote; neither the released training code nor the hub checkpoints include it. Otherwise, plot over steps (each checkpoint is 500 steps, ~185M tokens).

Use `--accuracy_mode ci` for the case-insensitive WLT accuracy reported in the paper.

* WLT metrics: `python ./src/plot_wlt_metrics.py --run_name <run_name> --all` (`--metrics accuracy|probability|probability_normalized`; add `--plot_steps 100000` for the accuracy heatmap at a step, `--symmetry` for the symmetry plots, `--plot_output_classification` for the output classification plots)
* Logit lens, averaged over all 72 language pairs as in the paper (per-step plots for the steps in `--plot_steps`, plots over time, and margin heatmaps):

  `python ./src/plot_logit_lens.py --run_name <run_name> --tgt_lang en --src_langs de --avgs --score_type scores_normed --accuracy_mode ci --groupings relative_script,collapsed_other,jaccard --jaccard_target_only --plot_steps 500,100000 --plot_tf_langs target,source,english,other --plot_grouped_time --plot_margins`

  For the subtoken-overlap groups over time use `--groupings jaccard --plot_tf_langs copy,partial,none`; add `--margin_layer_diff` for the layer-wise margin gradient.
* MultiBLiMP: `python ./src/plot_mblimp.py --run_name <run_name>`

See `--help` of each script for all options.

## Tests

`pytest src/test_evaluate.py`

`src/sanity_check_eval.py` is a manual consistency check (batched vs. unbatched metrics, logit lens) on a local checkpoint; it needs a trained model and is not part of the pytest suite.

## Environment Variables

- `OUTPUT_DIR`: default output root used by evaluation scripts (default: `""`)
- `MODEL_CONFIG_DIR`: directory containing training model config files, e.g. `llama1b.json` (default: `../training/configs/models/`)
- `WANDB_PROJECT`: W&B project name (default: `eval`)
- `WANDB_ENTITY`: optional W&B entity
