import argparse
import os
from transformers import AutoTokenizer
from accelerate import Accelerator
from data_utils import (
    get_multilingual_dataloader,
    build_dataset_roots_from_bases,
    unpack_packed_batch,
)
import json
import torch

from constants import ID2LANG, DATASET_BASES, LANG_WEIGHTS


def test_multilingual_loader(
    accelerator,
    dataset_roots,
    tok=None,
    model_cfg=None,
    n_examples=10,
    effective_batch_size=32,
    gradient_accumulation_steps=1,
):
    """
    Quick sanity check for the multilingual dataloader.
    Loads a few batches and prints decoded examples with language tags.
    """
    # default tokenizer if none provided
    if tok is None:
        tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

    # default tokenizer function
    def tokenize_function(examples):
        tokens = tok(
            examples["text"], truncation=True, padding="max_length", max_length=64
        )
        if "language" in examples:
            tokens["language"] = examples["language"]
        return tokens

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # initialize dataloader
    assert (
        effective_batch_size % (accelerator.num_processes * gradient_accumulation_steps)
        == 0
    ), "Effective batch size must be divisible by (num_processes * gradient_accumulation_steps)"
    bs_per_gpu = effective_batch_size // (
        accelerator.num_processes * gradient_accumulation_steps
    )
    dl, streamer = get_multilingual_dataloader(
        accelerator=accelerator,
        dataset_roots=dataset_roots,
        tok=tok,
        bs_per_gpu=bs_per_gpu,
        tokenize_function=tokenize_function,
        lang_weights=LANG_WEIGHTS,
        model_cfg=model_cfg,
        verbose=True,
        streaming=True,
    )

    # inspect some samples
    print(
        f"Rank = {torch.distributed.get_rank()}, World Size = {torch.distributed.get_world_size()}",
        flush=True,
    )
    print(
        f"Accelerator num_processes={accelerator.num_processes}, "
        f"process_index={accelerator.process_index}, "
        f"local_process_index={accelerator.local_process_index}",
        flush=True,
    )
    count = 0

    for batch in dl:
        accelerator.print(f"Batch {count}/{n_examples-1}")
        if model_cfg.get("collate_style") == "packing":
            examples = unpack_packed_batch(batch)
        else:
            examples = [
                (batch["input_ids"][i], int(batch["language_ids"][i]))
                for i in range(batch["input_ids"].size(0))
            ]
        print(len(examples), "examples in the batch")
        for i, (inp, lang_id) in enumerate(examples):
            decoded = tok.decode(inp, skip_special_tokens=True)
            print(
                f"[Rank {accelerator.process_index}] "
                f"(Example {ID2LANG[lang_id]}): {decoded[:60]}...",
                flush=True,
            )
        count += 1
        accelerator.print("State", streamer.state["lang_progress"], "\n")

        if count >= n_examples:
            break

    accelerator.print(
        f"Test complete — printed {n_examples*bs_per_gpu} decoded examples.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--effective_batch_size", type=int, default=32, help="Effective batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    model_config_path = "configs/models/" + "llama_tiny.json"
    with open(model_config_path) as f:
        model_cfg = json.load(f)

    data_dir = args.data_dir
    dataset_roots = build_dataset_roots_from_bases(
        data_dir_root=data_dir,
        dataset_bases=DATASET_BASES,
        accelerator=accelerator,
    )

    test_multilingual_loader(
        accelerator=accelerator,
        dataset_roots=dataset_roots,
        tok=None,
        model_cfg=model_cfg,
        effective_batch_size=args.effective_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    accelerator.end_training()
