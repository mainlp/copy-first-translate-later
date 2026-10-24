import os
import argparse

import torch
import wandb

from evaluate import (
    filter_results,
    get_harness_results_paths,
    get_translation_metrics_results_path,
)
from logit_lens import get_logit_lens_results_path
from utils import (
    setup_logging,
    set_seed_all,
    eval_progress,
    get_checkpoint_results_dirs,
)
from constants import (
    DEFAULT_OUTPUT_DIR,
    HARNESS_METRICS,
    TOKENS_KEY,
    BENCHMARKS,
    RELEASED_MODEL_NAME,
)
from checkpoint_strategy import CheckpointStrategy
from hf_utils import HFCheckpointStrategy
from expl_training_utils import ExplTrainingCheckpointStrategy
from evaluate import run_harness_tasks

# Because too many concurrent threads slow down server...
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)


def parse_eval_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints for a completed run."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help=("expl-training: model config type, e.g. 'llama_tiny.json'."),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hf", "expl-training"],
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store outputs and checkpoints.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name for this evaluation run (used for W&B).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (default: off).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Specific checkpoint step to start evaluating from (default: start at 0).",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=1,
        help="Frequency of checkpoint steps to evaluate (default: 1 means evaluate all).",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--dry_run",
        "-d",
        action="store_true",
        help="Do a dry run without actual evaluation.",
    )
    args = parser.parse_args()
    own_model = args.model_name == RELEASED_MODEL_NAME
    if args.benchmarks is not None:
        for b in args.benchmarks:
            if b not in BENCHMARKS:
                parser.error(f"Unsupported benchmark: {b}")
            if args.mode == "hf" and "logit_lens" in b and not own_model:
                # candidate words are tokenized in isolation, which only matches
                # their in-context tokenization for tokenizers like ours
                parser.error(f"Benchmark {b} is not supported in hf mode.")
    else:
        if args.mode != "hf" or own_model:
            args.benchmarks = BENCHMARKS
        else:
            args.benchmarks = [b for b in BENCHMARKS if "logit_lens" not in b]

    if args.model_name is None:
        parser.error(
            "--model_name is required for the selected mode."
            "expl-training: e.g. 'llama_tiny.json'."
        )

    if args.dry_run and args.use_wandb:
        print("Dry run enabled; disabling Weights & Biases logging.")
        args.use_wandb = False
    args.run_dir = os.path.join(args.output_dir, args.run_name)
    return args


def checkpoint_results_exist_for_benchmarks(output_dir, benchmarks):
    if benchmarks is None:
        benchmarks = BENCHMARKS

    for benchmark in benchmarks:
        # same paths the benchmarks write to
        if benchmark.startswith("logit_lens_"):
            paths = [
                get_logit_lens_results_path(
                    output_dir=output_dir, lang=benchmark.split("_")[-1]
                )
            ]
        elif benchmark == "word_level_translation":
            prefix = get_translation_metrics_results_path(output_dir=output_dir)
            paths = [
                prefix + suffix
                for suffix in [
                    "_accuracy.csv",
                    "_probability.csv",
                    "_probability_normalized.csv",
                ]
            ]
        else:
            paths = get_harness_results_paths(benchmark, output_dir=output_dir)
        if not all(os.path.exists(path) for path in paths):
            return False
    return True


def evaluate_all_checkpoints(
    output_dir,
    run_name,
    strategy: CheckpointStrategy,
    seed=42,
    use_wandb=False,
    start_step=None,
    dry_run=False,
    freq=1,
    benchmarks=None,
):
    checkpoints_dir, results_dir = get_checkpoint_results_dirs(output_dir)
    tokenizer, config_obj = strategy.build_initial()
    if use_wandb:
        setup_logging(
            model_config_path_or_obj=config_obj,
            seed=seed,
            run_name=run_name,
        )

    set_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = strategy.list_checkpoints()
    if len(ckpts) == 0:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}!")
    print(f"Found {len(ckpts)} checkpoints!")

    for i, ckpt in enumerate(ckpts):
        if i % freq != 0 and i != len(ckpts) - 1:
            print(f"Skipping checkpoint {i} due to frequency setting (freq={freq}).")
            continue
        epoch = ckpt.get("epoch")
        step = ckpt.get("step")
        path = ckpt.get("path")

        if start_step is not None and int(step) < start_step:
            print(
                f"Skipping checkpoint {i} (step {step}) due to start step (step={start_step})."
            )
            continue
        # eval_progress creates the folder; a dry run or skip should not
        output_dir = os.path.join(results_dir, f"epoch{epoch}-step{step}")
        if checkpoint_results_exist_for_benchmarks(output_dir, benchmarks):
            print(
                f"All results exist for checkpoint step={step}, skipping evaluation entirely."
            )
            if use_wandb:
                cached_results = run_harness_tasks(
                    model=None,
                    tokenizer=None,
                    tasks=[b for b in benchmarks if b in HARNESS_METRICS],
                    output_dir=output_dir,
                    merge_results=True,
                    force=False,
                )
                flat_results = filter_results(
                    {"results": cached_results}, HARNESS_METRICS
                )
                flat_results[TOKENS_KEY] = strategy.step_to_tokens(step)
                wandb.log(flat_results, step=step)

            continue

        print(f"Evaluating checkpoint {path} (epoch={epoch}, step={step})")
        tokens_seen = strategy.step_to_tokens(step)
        print(f"Tokens seen at this checkpoint: {tokens_seen}")
        if dry_run:
            print("Dry run enabled; skipping actual evaluation.")
            continue

        model_ckpt = strategy.load_model_for_checkpoint(ckpt, device)

        eval_progress(
            global_step=step,
            model=model_ckpt,
            tokenizer=tokenizer,
            results_dir=results_dir,
            epoch=epoch,
            step=step,
            tokens=tokens_seen,
            use_wandb=use_wandb,
            benchmarks=benchmarks,
        )
        del model_ckpt
        print(f"Done evaluating checkpoint (epoch={epoch}, step={step})")

    print("All checkpoints evaluated successfully.")


if __name__ == "__main__":
    args = parse_eval_args()
    run_dir = os.path.join(args.output_dir, args.run_name)
    checkpoints_dir, results_dir = get_checkpoint_results_dirs(run_dir)
    if args.mode == "expl-training":
        strategy = ExplTrainingCheckpointStrategy(
            model_config_type=args.model_name,
            checkpoints_dir=checkpoints_dir,
        )
    elif args.mode == "hf":
        strategy = HFCheckpointStrategy(model_name=args.model_name)
    print(f"Running with {vars(args)}")
    evaluate_all_checkpoints(
        output_dir=run_dir,
        run_name=args.run_name,
        strategy=strategy,
        seed=args.seed,
        use_wandb=args.use_wandb,
        start_step=args.step,
        dry_run=args.dry_run,
        freq=args.freq,
        benchmarks=args.benchmarks,
    )
