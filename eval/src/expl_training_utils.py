import json
import os
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file

from checkpoint_strategy import CheckpointStrategy
from collate_fn import _tokens_from_checkpoint
from utils import prepare_tokenizer


# configs of the training code, next to this folder in the repo
MODEL_CONFIG_DIR = os.getenv("MODEL_CONFIG_DIR", "../training/configs/models/")


def build_tok(model_config_path: str):
    with open(model_config_path, "r") as f:
        cfg_json = json.load(f)

    # Extract tokenizer path before removing the field
    tokenizer_path = cfg_json.pop("tokenizer")
    tokenizer = prepare_tokenizer(tokenizer_path)
    return tokenizer


def build_model(model_config_path: str):
    """
    Build a LLaMA-like model.
    Args:
        model_config_path: path to a JSON config file (e.g. ../training/configs/models/llama1b.json).
    Returns:
        model, tokenizer
    """
    with open(model_config_path, "r") as f:
        cfg_json = json.load(f)

    # Extract tokenizer path before removing the field
    tokenizer_path = cfg_json.pop("tokenizer")
    tokenizer = prepare_tokenizer(tokenizer_path)

    cfg = LlamaConfig.from_dict(cfg_json)
    cfg.vocab_size = max(cfg.vocab_size, tokenizer.vocab_size)
    if cfg.vocab_size != tokenizer.vocab_size:
        print(f"Vocab size mismatch! {cfg.vocab_size} vs {tokenizer.vocab_size}")

    model = LlamaForCausalLM(cfg)

    model.config.attn_implementation = "flash_attention_2"

    model.eval()
    return model, tokenizer


class ExplTrainingCheckpointStrategy(CheckpointStrategy):
    def __init__(
        self,
        model_config_type: str,
        checkpoints_dir: str,
    ):
        self.model_config_path = os.path.join(MODEL_CONFIG_DIR, model_config_type)
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(
                f"Model config file not found at {self.model_config_path}"
            )
        if not os.path.exists(checkpoints_dir):
            raise FileNotFoundError(
                f"No checkpoint directory found at {checkpoints_dir}"
            )
        self.checkpoints_dir = checkpoints_dir
        self.ckpts = None  # init so we can check for existence in list_checkpoints()
        self.ckpts = self.list_checkpoints()
        self.step_to_tokens_cache = {}
        try:
            cumulative = 0
            for ckpt in self.ckpts:
                cumulative += _tokens_from_checkpoint(ckpt["path"])
                self.step_to_tokens_cache[ckpt["step"]] = cumulative
        except FileNotFoundError as e:
            print(f"Warning: {e}; tokens seen will not be reported.")
            self.step_to_tokens_cache = {}

    def list_checkpoints(self):
        if self.ckpts is not None:
            return self.ckpts
        ckpts = []
        for d in os.listdir(self.checkpoints_dir):
            if not d.startswith("checkpoint-"):
                continue
            _, epoch, step = d.split("-")
            ckpts.append(
                {
                    "epoch": int(epoch),
                    "step": int(step),
                    "path": os.path.join(self.checkpoints_dir, d),
                }
            )

        keys = ["epoch", "step"]
        ckpts.sort(key=lambda x: tuple(x[k] for k in keys))
        self.ckpts = ckpts
        return ckpts

    def load_model_for_checkpoint(self, ckpt_info, device):
        model, _ = build_model(self.model_config_path)
        checkpoint_dir = os.path.join(
            self.checkpoints_dir, f"checkpoint-{ckpt_info['epoch']}-{ckpt_info['step']}"
        )
        print(f"Restoring from checkpoint: {checkpoint_dir}")
        weights_path = os.path.join(checkpoint_dir, "model.safetensors")
        state_dict = load_file(weights_path, device="cpu")
        incompatible = model.load_state_dict(state_dict, strict=False)
        # a key mismatch would otherwise silently leave weights randomly initialised
        if incompatible.missing_keys:
            raise ValueError(
                f"{weights_path} is missing weights: {incompatible.missing_keys[:5]}"
            )
        model.to(device)
        model.eval()
        print("Model weights loaded for evaluation.")
        return model

    def build_initial(self):
        tokenizer = build_tok(self.model_config_path)
        return tokenizer, self.model_config_path

    def step_to_tokens(self, step: int):
        """
        Convert training step to number of tokens seen.
        """
        return self.step_to_tokens_cache.get(step)
