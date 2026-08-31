import json
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def build_model(model_config_path: str, accelerator):
    """
    Build a LLaMA-like model with the Tekken tokenizer.
    Args:
        model_config_path: path to a JSON config file (e.g. configs/model/llama_tiny.json).
    Returns:
        model, tokenizer
    """
    # load tokenizer (Tekkenv3)
    with open(model_config_path, "r") as f:
        cfg_json = json.load(f)

    # Extract tokenizer path before removing the field
    tokenizer_path = cfg_json.pop("tokenizer")
    tokenizer = prepare_tokenizer(tokenizer_path)
    accelerator.print("Vocab size ", tokenizer.vocab_size)

    # load JSON config
    cfg = LlamaConfig.from_dict(cfg_json)
    # adapt vocab size
    cfg.vocab_size = max(cfg.vocab_size, tokenizer.vocab_size)

    # build model
    model = LlamaForCausalLM(cfg)

    # enable efficiency features
    model.gradient_checkpointing_enable()
    model.config.attn_implementation = "flash_attention_2"

    return model, tokenizer


def prepare_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
