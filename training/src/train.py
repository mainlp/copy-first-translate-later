from model import build_model
import argparse
import os

from utils import (
    set_seed_all,
    restore_latest_checkpoint,
    store_checkpoint,
    setup_logging,
    get_scheduler_and_optimizer,
    log_progress,
    eval_progress,
    load_configs,
    setup_accelerator,
    get_checkpoint_results_dirs,
)
from data_utils import (
    get_dataloader,
    get_wikitext,
    get_multilingual_dataloader,
    build_dataset_roots_from_bases,
)


from constants import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OPT_DIR,
    DEFAULT_DEEPSPEED_DIR,
    DEFAULT_OUTPUT_DIR,
    MAX_STEPS_FOR_TESTING,
    DEFAULT_EVAL_FREQUENCY,
    DEFAULT_CHECKPOINT_FREQUENCY,
    DATASET_BASES,
    LANG_WEIGHTS,
    DEFAULT_EFFECTIVE_BATCH_SIZE,
)


def main(
    model_config_path,
    optimizer_config_path,
    seed,
    num_epochs,
    effective_batch_size,
    gradient_accumulation_steps,
    max_grad_norm,
    checkpoint_interval,
    output_dir,
    data_dir,
    data_name,
    deepspeed_config_path=None,
    run_name="not_specified_name",
    verbose=False,
    use_wandb=False,
    restore_last_checkpoint=False,
    max_steps=-1,
    add_mnt=False,
    conv_paths=False,
):

    model_cfg, optim_cfg = load_configs(model_config_path, optimizer_config_path)
    accelerator = setup_accelerator(
        deepspeed_config_path,
        use_wandb,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    checkpoints_dir, results_dir = get_checkpoint_results_dirs(output_dir)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    accelerator.print(
        f"DeepSpeed active: {accelerator.state.deepspeed_plugin is not None}\n"
        f"Distributed type: {accelerator.state.distributed_type}\n"
    )

    accelerator.print(f"Model config: {model_cfg}")
    accelerator.print(f"Optimizer config: {optim_cfg}")
    accelerator.print(f"Effective batch size: {effective_batch_size}")
    accelerator.print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    accelerator.print(f"Num of GPUs: {accelerator.state.num_processes}\n")

    assert (
        effective_batch_size % (accelerator.num_processes * gradient_accumulation_steps)
        == 0
    ), "Effective batch size must be divisible by number of processes x gradient accumulation steps."
    bs_per_gpu = int(
        effective_batch_size / (gradient_accumulation_steps * accelerator.num_processes)
    )
    accelerator.print(f"Batch size per GPU: {bs_per_gpu}")

    if use_wandb:
        setup_logging(
            model_config_path,
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
        )

    set_seed_all(seed, accelerator)

    ## load model and tokenizer
    model, tokenizer = build_model(model_config_path, accelerator)

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=model_cfg["max_position_embeddings"],
        )

    ## load dataset
    if data_name == "wikitext":
        accelerator.print("Loading wikitext")
        dataset = get_wikitext(verbose, accelerator, tokenize_function)
        dataloader = get_dataloader(
            seed,
            model_cfg,
            accelerator,
            tokenizer,
            dataset,
            bs_per_gpu=bs_per_gpu,
            verbose=verbose,
        )
        num_training_steps = len(dataloader) * num_epochs
    elif data_name == "FineWeb2-HQ":
        dataset_roots = build_dataset_roots_from_bases(
            data_dir_root=data_dir,
            dataset_bases=DATASET_BASES,
            accelerator=accelerator,
        )

        dataloader, streamer = get_multilingual_dataloader(
            accelerator=accelerator,
            tok=tokenizer,
            tokenize_function=tokenize_function,
            model_cfg=model_cfg,
            bs_per_gpu=bs_per_gpu,
            dataset_roots=dataset_roots,
            lang_weights=LANG_WEIGHTS,
            seed=seed,
            verbose=verbose,
            streaming=True,
            conv_paths=conv_paths,
        )
        # TODO properly count this
        num_training_steps = MAX_STEPS_FOR_TESTING
    else:
        raise NotImplementedError(f"This data_name '{data_name}' is not implemented.")

    if max_steps > 0:
        # overwrite for debugging and profiling stuff
        num_training_steps = max_steps

    ## init optimizer

    optimizer, scheduler = get_scheduler_and_optimizer(
        optim_cfg,
        model,
        num_training_steps=num_training_steps,
        num_gpus=accelerator.num_processes,
    )
    accelerator.print("num_training steps", num_training_steps)

    if scheduler is None:
        model, optimizer = accelerator.prepare(
            model, optimizer
        )  # we don't prepare the dataloader because we shard the data ourselves
    else:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    global_step = 0
    start_epoch = 0
    step = 0
    restored = False

    if restore_last_checkpoint:
        # continue from latest checkpoint in output_dir
        # option: We can also specify a particular checkpoint to restore and use restore_checkpoint()
        accelerator, optimizer, scheduler, dataloader, start_epoch, step, streamer = (
            restore_latest_checkpoint(
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoints_dir=checkpoints_dir,
                dataloader=dataloader,
                streamer=streamer,
            )
        )
        accelerator.print(f"Resuming from epoch {start_epoch}, step {step}")

        # update global step to correct point in training
        global_step = step + start_epoch * num_training_steps
        restored = True

    accumulation_counter = gradient_accumulation_steps

    ## training loop
    for epoch in range(start_epoch, num_epochs):

        accelerator.print(f"Starting epoch {epoch + 1}/{num_epochs}")
        model.train()

        for batch in dataloader:

            with accelerator.accumulate(model):

                # forward and backward pass
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                # update parameters
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            accumulation_counter -= 1

            if accumulation_counter == 0:
                # we have performed an update step and can update the step counters
                step += 1
                global_step += 1
                accumulation_counter = gradient_accumulation_steps  # reset counter
                # eval and logging
                # TODO: maybe implement a logging interval
                log_progress(accelerator, scheduler, global_step, loss, streamer)
                if step % DEFAULT_EVAL_FREQUENCY == 0 and step > 0:
                    eval_progress(
                        accelerator,
                        scheduler,
                        global_step,
                        loss,
                        model,
                        tokenizer,
                        results_dir,
                        epoch=epoch,
                        step=step,
                        add_mnt=add_mnt,
                    )

                # only actually stores a checkpoint when global_step % checkpoint_interval == 0
                store_checkpoint(
                    accelerator=accelerator,
                    batch=batch,
                    checkpoints_dir=checkpoints_dir,
                    epoch=epoch,
                    step=step,
                    global_step=global_step,
                    checkpoint_interval=checkpoint_interval,
                    dataset_state=streamer.state,
                )

            if (
                step >= num_training_steps or step >= MAX_STEPS_FOR_TESTING
            ):  # for quick testing
                break

            restored = False  # only skip once after restoration

    # final checkpoint at end of training
    store_checkpoint(
        accelerator,
        batch,
        checkpoints_dir,
        epoch,
        step,
        checkpoint_interval,
        global_step,
        dataset_state=streamer.state,
        force=True,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.end_training()

    accelerator.print("Training complete.")

    return accelerator


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--effective_batch_size",
        type=int,
        default=DEFAULT_EFFECTIVE_BATCH_SIZE,
        help="Effective Batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default="llama_tiny.json")
    parser.add_argument("--optimizer_name", type=str, default="adamw.json")
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON (optional).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable extra debug/info printing."
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (default: off)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=DEFAULT_CHECKPOINT_FREQUENCY,
        help="Steps between checkpoints (default: 1000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store outputs and checkpoints.",
    )
    parser.add_argument(
        "--run_name", type=str, default="run-prepare-0", help="Run name"
    )
    parser.add_argument("--data_dir", type=str, default="", help="Data storage path")
    parser.add_argument(
        "--data_name",
        type=str,
        default="wikitext",
        choices=["wikitext", "FineWeb2-HQ"],
        help="Name of the dataset to use. Options: 'wikitext' or 'FineWeb2-HQ'.",
    )
    parser.add_argument(
        "--add_mnt",
        action="store_true",
        help="Enable mount path prefixing for config files (prepends /mnt/ to config paths).",
    )
    parser.add_argument(
        "--conv_paths",
        action="store_true",
        help="Enable conversion of file paths from old prefix /dataset to new prefix.",
    )
    parser.add_argument(
        "--restore_last_checkpoint",
        action="store_true",
        help="Restore from latest checkpoint in output_dir if available.",
    )

    return parser


def prepare_main(args):
    if args.add_mnt:
        model_config_path = "/mnt/" + DEFAULT_MODEL_DIR + args.model_name
        optimizer_config_path = "/mnt/" + DEFAULT_OPT_DIR + args.optimizer_name
        if args.deepspeed_config is not None:
            deepspeed_config_path = (
                "/mnt/" + DEFAULT_DEEPSPEED_DIR + args.deepspeed_config
            )
        else:
            deepspeed_config_path = None
    else:
        model_config_path = DEFAULT_MODEL_DIR + args.model_name
        optimizer_config_path = DEFAULT_OPT_DIR + args.optimizer_name
        if args.deepspeed_config is not None:
            deepspeed_config_path = DEFAULT_DEEPSPEED_DIR + args.deepspeed_config
        else:
            deepspeed_config_path = None

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    return model_config_path, optimizer_config_path, deepspeed_config_path, run_dir


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    model_config_path, optimizer_config_path, deepspeed_config_path, run_dir = (
        prepare_main(args)
    )

    acc = main(
        model_config_path=model_config_path,
        optimizer_config_path=optimizer_config_path,
        seed=args.seed,
        num_epochs=args.num_epochs,
        effective_batch_size=args.effective_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        deepspeed_config_path=deepspeed_config_path,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=run_dir,
        data_dir=args.data_dir,
        run_name=args.run_name,
        data_name=args.data_name,
        verbose=args.verbose,
        use_wandb=args.use_wandb,
        restore_last_checkpoint=args.restore_last_checkpoint,
        add_mnt=args.add_mnt,
        conv_paths=args.conv_paths,
    )
