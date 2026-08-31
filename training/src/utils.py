import json
import os
import random
import numpy as np
import torch
from transformers import get_wsd_schedule
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, GradientAccumulationPlugin
from evaluate import evaluate_all
from safetensors.torch import load_file


def setup_logging(
    model_config_path_or_obj,
    optimizer_config_path,
    seed,
    num_epochs,
    run_name,
    restore_last_checkpoint,
    accelerator,
    checkpoints_dir,
    data_name,
    effective_batch_size,
    bs_per_gpu,
    post_hoc=False,
):

    if isinstance(model_config_path_or_obj, str):
        with open(model_config_path_or_obj) as f:
            model_cfg = json.load(f)
    else:
        model_cfg = model_config_path_or_obj.to_dict()

    optim_cfg = dict()
    if optimizer_config_path is not None:
        with open(optimizer_config_path) as f:
            optim_cfg = json.load(f)

    if restore_last_checkpoint:
        latest_epoch, latest_step = find_latest_checkpoint(checkpoints_dir)
        wandb_run_name = f"{run_name} resumed from epoch {latest_epoch}-{latest_step}"
        job_type = "resume"
    elif post_hoc:
        wandb_run_name = f"{run_name} post-hoc"
        job_type = "post-hoc"
    else:
        wandb_run_name = f"{run_name} initial"
        job_type = "train"

    wandb_kwargs = {
        "group": run_name,
        "name": wandb_run_name,
        "job_type": job_type,
        "resume": None,  # always start a fresh run
    }

    accelerator.init_trackers(
        project_name=os.getenv("WANDB_PROJECT", "training"),
        config={
            "model_config": model_cfg,
            "optimizer_config": optim_cfg,
            "num_epochs": num_epochs,
            "data_name": data_name,
            "effective_batch_size": effective_batch_size,
            "batch_size_per_gpu": bs_per_gpu,
            "seed": seed,
        },
        init_kwargs={"wandb": wandb_kwargs},
    )

    if accelerator.is_main_process:
        accelerator.print(f"W&B group: {run_name}")
        accelerator.print(f"W&B run name: {wandb_run_name}")


def store_checkpoint(
    accelerator,
    batch,
    checkpoints_dir,
    epoch,
    step,
    checkpoint_interval,
    global_step,
    dataset_state,
    force=False,
):
    """
    Store model, optimizer, scheduler states and current batch. Note that this function
    needs to be called before the optimizer step, so that the model and optimizer states
    correspond to the state before training on the batch.
    """
    gathered_global_step = (
        accelerator.gather(torch.tensor(global_step, device=accelerator.device))
        .max()
        .item()
    )
    gathered_step = (
        accelerator.gather(torch.tensor(step, device=accelerator.device)).max().item()
    )
    accelerator.wait_for_everyone()
    # gather actual step number
    # checkpoints are stored based on the global step count
    if gathered_global_step % checkpoint_interval == 0 or force:
        # store checkpoint
        checkpoint_dir = os.path.join(
            checkpoints_dir, f"checkpoint-{epoch}-{gathered_step}"
        )
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(batch, os.path.join(checkpoint_dir, "batch.pt"))

        # ensure directory is created before other ranks continue
        accelerator.wait_for_everyone()
        # each rank saves its own streamer state
        rank_path = os.path.join(
            checkpoint_dir, f"dataset_state_rank{accelerator.process_index}.json"
        )
        with open(rank_path, "w") as f:
            json.dump(dataset_state, f, indent=2)

        # optimizer must be saved outside of the main_process due to deepspeed
        accelerator.save_state(checkpoint_dir, safe_serialization=True)
        accelerator.print(f"Stored checkpoint at {checkpoint_dir}")


def restore_checkpoint(
    accelerator,
    optimizer,
    scheduler,
    epoch,
    step,
    checkpoints_dir,
    dataloader,
    streamer=None,
    model=None,
    mode="train",
):
    checkpoint_dir = os.path.join(checkpoints_dir, f"checkpoint-{epoch}-{step}")
    accelerator.print(f"Restoring from checkpoint: {checkpoint_dir}")
    if mode == "train":
        accelerator.load_state(checkpoint_dir)

        # Fix scheduler-optimizer learning rate synchronization after state loading
        # This is necessary because loading scheduler state doesn't automatically update
        # the optimizer's learning rates to match the scheduler's current state
        if scheduler is not None and optimizer is not None:
            current_lr = scheduler.get_last_lr()
            for param_group, lr in zip(optimizer.param_groups, current_lr):
                param_group["lr"] = lr
            accelerator.print(f"Synchronized learning rate to {current_lr[0]:.8f}")

        # load dataset state
        rank = accelerator.process_index
        rank_path = os.path.join(checkpoint_dir, f"dataset_state_rank{rank}.json")
        if os.path.exists(rank_path):
            with open(rank_path) as f:
                dataset_state = json.load(f)
            accelerator.print(f"[data] Loaded dataset_state_rank{rank}.json")
        else:
            accelerator.print(
                f"[data] No dataset_state_rank{rank}.json found, starting from scratch"
            )
            dataset_state = streamer.state  # default empty state

        old_world_size = dataset_state.get("world_size", streamer.world_size)
        if old_world_size != accelerator.num_processes:
            accelerator.print(
                f"[data] World size changed {old_world_size} -> {accelerator.num_processes}"
            )
            streamer.world_size = accelerator.num_processes
            streamer.rank = accelerator.process_index

        # attach restored dataset state
        streamer.state = dataset_state
        accelerator.print(f"[data] Dataset state restored for rank {rank}")

        return accelerator, optimizer, scheduler, dataloader, epoch, step, streamer
    else:
        assert model is not None, "Model must be provided for evaluation mode"
        weights_path = os.path.join(checkpoint_dir, "model.safetensors")
        raw_model = accelerator.unwrap_model(model)
        # simpler than loading through accelerate, as we don't need
        # optimizer/scheduler states for eval
        state_dict = load_file(weights_path, device="cpu")
        raw_model.load_state_dict(state_dict, strict=False)
        raw_model.to(accelerator.device)
        accelerator.print("Model weights loaded for evaluation.")
    return accelerator, optimizer, scheduler, dataloader, epoch, step


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dirs = [
        d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")
    ]

    latest_epoch = 0
    latest_step = 0
    for d in checkpoint_dirs:
        _, epoch, step = d.split("-")
        epoch = int(epoch)
        step = int(step)
        if epoch > latest_epoch or (epoch == latest_epoch and step > latest_step):
            latest_epoch = epoch
            latest_step = step
    return latest_epoch, latest_step


def list_checkpoints(checkpoint_dir):
    ckpts = []
    for d in os.listdir(checkpoint_dir):
        if not d.startswith("checkpoint-"):
            continue
        _, epoch, step = d.split("-")
        ckpts.append(
            {
                "epoch": int(epoch),
                "step": int(step),
                "path": os.path.join(checkpoint_dir, d),
            }
        )

    keys = ["epoch", "step"]
    ckpts.sort(key=lambda x: tuple(x[k] for k in keys))
    return ckpts


def restore_latest_checkpoint(
    accelerator, optimizer, scheduler, checkpoints_dir, dataloader, streamer
):

    if not os.path.exists(checkpoints_dir):
        accelerator.print(f"No checkpoints directory found at {checkpoints_dir}")
        return accelerator, dataloader, 0, 0

    # find latest checkpoint
    latest_epoch, latest_step = find_latest_checkpoint(checkpoints_dir)
    accelerator, optimizer, scheduler, dataloader, epoch, step, streamer = (
        restore_checkpoint(
            accelerator,
            optimizer,
            scheduler,
            latest_epoch,
            latest_step,
            checkpoints_dir,
            dataloader,
            streamer,
        )
    )

    # for future reference
    assert epoch == latest_epoch
    assert step == latest_step

    return (
        accelerator,
        optimizer,
        scheduler,
        dataloader,
        latest_epoch,
        latest_step,
        streamer,
    )


def set_seed_all(seed: int, accelerator):
    """Set random, numpy, and torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if accelerator.is_main_process:
        accelerator.print(f"[seed] Using seed={seed}")


def get_scheduler_and_optimizer(optim_cfg, model, num_training_steps=10000, num_gpus=1):
    sched_cfg = optim_cfg.pop("scheduler", None)

    opt_name = optim_cfg.pop("optimizer")
    OptClass = getattr(torch.optim, opt_name)
    optimizer = OptClass(model.parameters(), **optim_cfg)

    scheduler = None
    if sched_cfg is not None:
        scheduler = get_wsd_schedule(
            optimizer=optimizer,
            num_warmup_steps=int(
                num_training_steps * sched_cfg["warmup_ratio"] * num_gpus
            ),
            num_decay_steps=int(
                num_training_steps * sched_cfg["decay_ratio"] * num_gpus
            ),
            num_training_steps=num_training_steps * num_gpus,
            warmup_type=sched_cfg["warmup_type"],
            decay_type=sched_cfg["decay_type"],
            min_lr_ratio=sched_cfg["min_lr_ratio"],
            num_cycles=sched_cfg.get("num_cycles", 0.5),
        )

    return optimizer, scheduler


def setup_accelerator(deepspeed_config_path, use_wandb, gradient_accumulation_steps=1):
    """Setup accelerator with optional DeepSpeed plugin."""

    deepspeed_plugin = None
    if deepspeed_config_path is not None:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_config_path)

    return Accelerator(
        log_with=("wandb" if use_wandb else None),
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps
        ),
    )


def load_configs(model_config_path, optimizer_config_path):
    """Load model and optimizer configurations."""
    try:
        with open(model_config_path) as f:
            model_cfg = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")

    try:
        with open(optimizer_config_path) as f:
            optim_cfg = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Optimizer config file not found: {optimizer_config_path}"
        )

    return model_cfg, optim_cfg


def eval_progress(
    accelerator,
    scheduler,
    global_step,
    loss,
    model,
    tokenizer,
    results_dir,
    epoch,
    step,
    add_mnt,
    mode="train",
):
    accelerator.wait_for_everyone()
    if mode == "train":
        gathered_global_step = (
            accelerator.gather(torch.tensor(global_step, device=loss.device))
            .max()
            .item()
        )
    elif mode == "eval":
        gathered_global_step = global_step
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print_str = f"[{mode.upper()}] Step {gathered_global_step}"
    if loss is not None:
        print_str += f", loss {loss.item():.4f}"
    if scheduler is not None:
        lr = scheduler.get_last_lr()[0]
        print_str += f", lr {lr:.6e}"

    accelerator.print(print_str)
    output_prefix = os.path.join(results_dir, f"eval-epoch{epoch}-step{step}-")
    results = evaluate_all(
        model,
        tokenizer,
        accelerator,
        output_prefix=output_prefix,
        add_mnt=add_mnt,
        mode=mode,
    )
    if len(results) > 0:
        for k, v in results.items():
            accelerator.print(f"{k}: {v:.4f}")
        accelerator.log(
            results,
            step=gathered_global_step,
        )
    else:
        accelerator.print(f"No evaluation results to log for {gathered_global_step}.")


def log_progress(accelerator, scheduler, global_step, loss, streamer):
    accelerator.wait_for_everyone()
    gathered_loss = accelerator.gather(loss)
    gathered_mean_loss = gathered_loss.mean().item()
    gathered_global_step = (
        accelerator.gather(torch.tensor(global_step, device=loss.device)).max().item()
    )

    if scheduler is not None:
        accelerator.log(
            {
                "train/mean_loss": gathered_mean_loss,
                "train/lr": scheduler.get_last_lr()[0],
            },
            step=gathered_global_step,
        )
    else:
        accelerator.log(
            {
                "train/mean_loss": gathered_mean_loss,
            },
            step=gathered_global_step,
        )

    for lang in streamer.state["langs"]:
        gathered_total = (
            accelerator.gather(
                torch.tensor(
                    streamer.state["lang_progress"][lang]["total"], device=loss.device
                )
            )
            .sum()
            .item()
        )
        accelerator.log(
            {f"data/{lang} example seen": gathered_total}, step=gathered_global_step
        )


def get_checkpoint_results_dirs(run_dir):
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    results_dir = os.path.join(run_dir, "results")
    return checkpoints_dir, results_dir
