import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer
import wandb
import os

from constants import TOKENS_KEY
from evaluate import evaluate_all


def setup_logging(model_config_path_or_obj, seed, run_name):

    if isinstance(model_config_path_or_obj, str):
        with open(model_config_path_or_obj) as f:
            model_cfg = json.load(f)
    else:
        model_cfg = model_config_path_or_obj.to_dict()

    wandb_run_name = f"{run_name} post-hoc"
    job_type = "post-hoc"

    wandb_kwargs = {
        "group": run_name,
        "name": wandb_run_name,
        "job_type": job_type,
        "resume": None,  # always start a fresh run
    }

    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT", "eval"),
        config={
            "model_config": model_cfg,
            "seed": seed,
        },
        settings=wandb.Settings(x_disable_stats=True),
        **wandb_kwargs,
    )
    print(f"W&B group: {run_name}")
    print(f"W&B run name: {wandb_run_name}")


def set_seed_all(seed: int):
    """Set random, numpy, and torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] Using seed={seed}")


def get_output_dir_for_checkpoint(results_dir, epoch, step):
    output_dir = os.path.join(results_dir, f"epoch{epoch}-step{step}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def log_results(results, global_step, tokens, use_wandb=False):
    print(f"Step: {global_step}, Tokens seen: {tokens}")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    if use_wandb:
        results[TOKENS_KEY] = tokens
        wandb.log(results, step=global_step)


def eval_progress(
    global_step,
    model,
    tokenizer,
    results_dir,
    epoch,
    step,
    tokens,
    use_wandb=False,
    benchmarks=None,
):
    print(f"[EVAL] Step: {global_step}, Tokens seen: {tokens}")
    output_dir = get_output_dir_for_checkpoint(results_dir, epoch, step)
    results = evaluate_all(
        model,
        tokenizer,
        output_dir=output_dir,
        benchmarks=benchmarks,
    )
    if len(results) > 0:
        log_results(results, global_step, tokens, use_wandb=use_wandb)
    else:
        print(f"No evaluation results to log for {global_step}.")


def prepare_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def get_checkpoint_results_dirs(run_dir):
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return checkpoints_dir, results_dir
