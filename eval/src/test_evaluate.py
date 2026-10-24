import os
import sys

import torch
import pytest
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from evaluate import extract_generated_words, split_at_quote
from constants import FEW_SHOT_DELIMITER

TOKENIZERS = [
    "gpt2",
    "EleutherAI/pythia-70m",
    "hf-internal-testing/llama-tokenizer",  # ungated copy of the Llama-2 tokenizer
]


class FakeTok:
    """Minimal tokenizer stub for unit-testing split_at_quote in isolation."""

    def __init__(self, pieces: dict):
        # pieces: token_id -> decoded string
        self._pieces = pieces

    def decode(self, ids, skip_special_tokens=True):
        # join all pieces so multi-token calls work correctly
        return "".join(self._pieces.get(i, "") for i in ids)


@pytest.mark.parametrize("tok_name", TOKENIZERS)
def test_extract_full_text_keeps_tail(tok_name):
    """Full text includes tokens after the quote; extracted word stops before it."""
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    tokenizer.pad_token = tokenizer.eos_token

    prompt = 'English: "'
    continuation = 'apple"\nEXTRA'

    p_ids = tokenizer(prompt, return_tensors="pt").input_ids
    c_ids = tokenizer(
        continuation, return_tensors="pt", add_special_tokens=False
    ).input_ids

    # Simulate left-padded batch output: pad + prompt + continuation
    pad = torch.full((1, 3), tokenizer.pad_token_id, dtype=torch.long)
    gen_out = torch.cat([pad, p_ids, c_ids], dim=1)

    # Attention mask covers only prompt tokens (not pad, not continuation)
    attention_mask = torch.zeros_like(gen_out)
    prompt_start = pad.shape[1]
    attention_mask[:, prompt_start : prompt_start + p_ids.shape[1]] = 1

    texts, words, gen_ids = extract_generated_words(tokenizer, gen_out, attention_mask)

    assert "EXTRA" in texts[0], "full text should include tokens after the quote"
    assert words[0] == "apple", f"expected 'apple', got {words[0]!r}"


@pytest.mark.parametrize("tok_name", TOKENIZERS)
def test_extract_no_quote_returns_full_continuation(tok_name):
    """When there is no quote, the full continuation is returned as the word."""
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    tokenizer.pad_token = tokenizer.eos_token

    prompt = 'English: "'
    continuation = "apple"

    p_ids = tokenizer(prompt, return_tensors="pt").input_ids
    c_ids = tokenizer(
        continuation, return_tensors="pt", add_special_tokens=False
    ).input_ids

    gen_out = torch.cat([p_ids, c_ids], dim=1)
    attention_mask = torch.zeros_like(gen_out)
    attention_mask[:, : p_ids.shape[1]] = 1

    _, words, _ = extract_generated_words(tokenizer, gen_out, attention_mask)

    assert words[0].strip() == "apple"


def test_split_at_quote_basic():
    """Quote token in the middle: content before, quote_lp at quote token."""
    pieces = {1: "apple", 2: '"', 3: "\n"}
    tokenizer = FakeTok(pieces)
    ids = [1, 2, 3]
    lp = [0.1, 0.2, 0.3]

    content_lp, quote_lp = split_at_quote(tokenizer, ids, lp)

    assert content_lp == [0.1]
    assert quote_lp == 0.2


def test_split_at_quote_quote_first():
    """Quote is the very first token → content_lp is empty list."""
    pieces = {1: '"', 2: "apple"}
    tokenizer = FakeTok(pieces)
    ids = [1, 2]
    lp = [0.5, 0.6]

    content_lp, quote_lp = split_at_quote(tokenizer, ids, lp)

    assert content_lp == []
    assert quote_lp == 0.5


def test_split_at_quote_merged_token():
    """Token that contains a quote but isn't solely a quote (e.g. 'apple"')."""
    pieces = {1: 'apple"', 2: "\n"}
    tokenizer = FakeTok(pieces)
    ids = [1, 2]
    lp = [0.9, 0.1]

    content_lp, quote_lp = split_at_quote(tokenizer, ids, lp)
    assert content_lp == []
    assert quote_lp == 0.9


def test_split_at_quote_no_quote():
    """No quote anywhere → returns full lp list and None."""
    pieces = {1: "apple", 2: "s"}
    tokenizer = FakeTok(pieces)
    ids = [1, 2]
    lp = [0.0, 0.0]

    content_lp, quote_lp = split_at_quote(tokenizer, ids, lp)

    assert content_lp == lp
    assert quote_lp is None


def test_split_at_quote_empty_continuation():
    """Empty token list → no quote found, returns empty list and None."""
    tokenizer = FakeTok({})
    content_lp, quote_lp = split_at_quote(tokenizer, [], [])

    assert content_lp == []
    assert quote_lp is None


@pytest.mark.parametrize("tok_name", TOKENIZERS)
def test_split_at_quote_real_tokenizer(tok_name):
    """Content decoded from returned slice never contains a quote."""
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    text = 'apple"\nEXTRA'
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    lp = [float(i) for i in range(len(ids))]

    content_lp, quote_lp = split_at_quote(tokenizer, ids, lp)

    decoded_content = tokenizer.decode(ids[: len(content_lp)])
    assert '"' not in decoded_content, "content slice should not contain a quote"

    quote_token_text = tokenizer.decode([ids[len(content_lp)]])
    assert '"' in quote_token_text, "token at cut should contain a quote"
    assert quote_lp == lp[len(content_lp)]


@pytest.mark.parametrize("tok_name", TOKENIZERS)
def test_split_at_quote_no_quote_real_tokenizer(tok_name):
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    text = "apple"
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    lp = [0.0] * len(ids)

    content_lp, quote_lp = split_at_quote(tokenizer, ids, lp)

    assert content_lp == lp
    assert quote_lp is None


@pytest.mark.parametrize("tok_name", ["gpt2", "EleutherAI/pythia-70m"])
def test_delimiter_slicing_excludes_quote(tok_name):
    """Slicing off delim_len tokens from the end leaves no quote in content."""
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    delim_ids = tokenizer(FEW_SHOT_DELIMITER, add_special_tokens=False)["input_ids"]
    delim_len = len(delim_ids)

    text = "apple" + FEW_SHOT_DELIMITER
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    content_ids = ids[:-delim_len]
    delim_decoded = tokenizer.decode(delim_ids)
    content_decoded = tokenizer.decode(content_ids)

    assert '"' not in content_decoded, "content slice should not contain a quote"
    assert '"' in delim_decoded, "delimiter slice should contain a quote"
    assert delim_len == len(delim_ids), "delim_len should match actual token count"
