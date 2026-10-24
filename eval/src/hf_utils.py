import re

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM

from checkpoint_strategy import CheckpointStrategy
from utils import prepare_tokenizer


def get_all_refs(model_name: str):
    refs = HfApi().list_repo_refs(model_name, repo_type="model")
    ref_names = []
    for branch in refs.branches:
        if "Apertus" in model_name and "longctx" in branch.name:
            continue
        ref_names.append(branch.name)
    # fallback to tag names
    if len(ref_names) < 2:
        for tag in refs.tags:
            if "Apertus" in model_name and "longctx" in tag.name:
                continue
            ref_names.append(tag.name)
    return ref_names


def get_sorted_steps(model_name: str):
    """
    Returns sorted steps, a step -> revision mapping and a step -> tokens mapping.
    Models without known intermediate-checkpoint naming are evaluated at their
    final revision ("main") only.
    """
    name = model_name.lower()
    if "apertus" in name:
        pat = re.compile(r"step(?P<step>\d+)-tokens(?P<tokens>\d+)B")
    elif "bloom" in name:
        pat = re.compile(r"global_step(?P<step>\d+)")
    elif "olmo" in name:
        pat = re.compile(
            r"stage(?P<stage>\d+)-step(?P<step>\d+)-tokens(?P<tokens>\d+)B"
        )
    else:
        # e.g. our released checkpoints (mainlp/copy-first-translate-later)
        pat = re.compile(r"step(?P<step>\d+)$")

    refs = [
        r for r in get_all_refs(model_name) if "stage2" not in r and "longctx" not in r
    ]
    if not any(pat.match(r) for r in refs):
        return [0], {0: "main"}, {0: None}
    parsed = []

    for r in refs:
        m = pat.match(r)
        if not m:
            continue
        step = int(m.group("step"))
        tokens = m.groupdict().get("tokens")
        tokens = int(tokens) * 1_000_000_000 if tokens is not None else None
        parsed.append((step, r, tokens))

    parsed.sort(key=lambda x: x[0])
    sorted_step_keys = [step for step, _, _ in parsed]
    step_to_path = {step: r for step, r, _ in parsed}
    step_to_tokens = {step: tokens for step, _, tokens in parsed}
    return sorted_step_keys, step_to_path, step_to_tokens


class HFCheckpointStrategy(CheckpointStrategy):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._step_to_tokens = {}

    def list_checkpoints(self):
        sorted_keys, step_to_path, self._step_to_tokens = get_sorted_steps(
            self.model_name
        )
        ckpts = [
            {"epoch": 0, "step": step, "path": step_to_path[step]}
            for step in sorted_keys
        ]
        return ckpts

    def load_model_for_checkpoint(self, ckpt_info, device):
        # Loading per-revision checkpoints may increase local Hugging Face cache usage.
        revision = ckpt_info["path"]
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=revision,
        )
        model.to(device)
        model.eval()
        return model

    def step_to_tokens(self, step: int):
        # only known if the revision name encodes it (e.g. Apertus, OLMo)
        return self._step_to_tokens.get(step)

    def build_initial(self):
        config = AutoConfig.from_pretrained(self.model_name)
        tokenizer = prepare_tokenizer(self.model_name)
        return tokenizer, config
