import torch
import os
import json
import torch.nn.functional as F
import pandas as pd
from constants import LANGS, WLT_DATA_PATH


def tokenize_target(tokenizer, word):
    return tokenizer(word, add_special_tokens=False)["input_ids"]

@torch.no_grad()
def project_hidden_states(hidden_states, model, token_id=-1):
    # we do not run logit lens just for the last token, because this changes the kernel
    # used and leads to a mismatch of the final logits.

    logits_last_layer = []

    unembed = model.lm_head.weight
    final_norm = model.model.norm
    num_layers = len(hidden_states)

    for i, h in enumerate(hidden_states):
        if i == num_layers - 1:
            h_norm = h
        else:
            h_norm = final_norm(h)
        logits = h_norm @ unembed.T
        logits_last_layer.append(logits[:, token_id, :])

    return logits_last_layer


@torch.no_grad()
def run_logit_lens_last_token(model, input_ids, token_id=-1):
    model.eval()
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    logits = project_hidden_states(hidden_states, model, token_id=token_id)
    return logits


def get_logit_lens_results_path(
    output_dir="results",
    lang="en",
    data_path=WLT_DATA_PATH,
):
    return f"{output_dir}/eval-logit_lens_{lang}.jsonl"


def per_language_per_layer_scores(
    model,
    tokenizer,
    prompt,
    candidates,
    device="cuda",
):
    """
    Returns: scores_word, scores_normed, scores_first
    scores_word[lang]["synonym"][layer] = whole-word log-prob (logit-lens) at that layer
    """
    model.eval()

    # encode prompt
    base_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    scores_word = {}
    scores_first = {}
    token_ids_by_lang = {}
    logits_per_layer_base = run_logit_lens_last_token(model, base_ids)

    # Compute first token scores
    for lang, words in candidates.items():
        scores_first[lang] = {}
        token_ids_by_lang[lang] = {}
        for word in words:
            token_ids = tokenize_target(tokenizer, word)
            token_ids_by_lang[lang][word] = token_ids
            first_tid = token_ids[0]

            layer_logps = []
            for logits in logits_per_layer_base:
                log_probs = F.log_softmax(logits, dim=-1)
                layer_logps.append(log_probs[0, first_tid].item())

            scores_first[lang][word] = layer_logps

    # Compute whole-word scores
    for lang, words in candidates.items():
        scores_word[lang] = {}
        for word in words:
            cur_ids = base_ids.clone()
            token_ids = token_ids_by_lang[lang][word]

            layer_sums = None

            for tid in token_ids:
                logits_per_layer = run_logit_lens_last_token(model, cur_ids)

                # accumulate log-probs
                layer_logps = []
                for logits in logits_per_layer:
                    log_probs = F.log_softmax(logits, dim=-1)
                    layer_logps.append(log_probs[0, tid].item())

                if layer_sums is None:
                    layer_sums = layer_logps
                else:
                    layer_sums = [a + b for a, b in zip(layer_sums, layer_logps)]

                # teacher forcing
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[tid]], device=device)], dim=1
                )

            scores_word[lang][word] = layer_sums

    scores_normed = {}
    for lang, lang_scores in scores_word.items():
        scores_normed[lang] = {}
        for word, layer_scores in lang_scores.items():
            n_tokens = len(token_ids_by_lang[lang][word])
            scores_normed[lang][word] = [s / max(1, n_tokens) for s in layer_scores]

    return scores_word, scores_normed, scores_first


@torch.no_grad()
def per_language_per_layer_scores_batched(
    model,
    tokenizer,
    prompt,
    candidates,  # dict: lang -> [word1, word2, ...] (always list)
    device="cuda",
):
    """
    Batched version of per_language_per_layer_scores.
    """
    model.eval()
    # encode the prompt
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device)
    prompt_len = len(prompt_ids)

    # Encode all candidates and store their IDs
    # We flatten: for each lang, we create separate batch items for each synonym
    cand_ids_list = []
    cand_keys = []  # (lang, word) pairs
    n_tokens_by_lang = {}

    for lang, synonyms in candidates.items():
        n_tokens_by_lang[lang] = {}
        for word in synonyms:
            c_ids = torch.tensor(tokenize_target(tokenizer, word), device=device)
            cand_ids_list.append(c_ids)
            cand_keys.append((lang, word))
            n_tokens_by_lang[lang][word] = len(c_ids)

    num_cands = len(cand_ids_list)
    max_cand_len = max(len(c) for c in cand_ids_list)

    # Create the Batch, pad with pad_token_id
    input_ids = torch.full(
        (num_cands, prompt_len + max_cand_len),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (num_cands, prompt_len + max_cand_len), dtype=torch.long, device=device
    )

    # target tokens for each batch item
    batch_target_tokens = []

    for i, c_ids in enumerate(cand_ids_list):
        c_len = len(c_ids)
        input_ids[i, :prompt_len] = prompt_ids
        # teacher forcing context
        # feed the full word so the model generates predictions for every step
        input_ids[i, prompt_len : prompt_len + c_len] = c_ids

        # attention mask
        attention_mask[i, : prompt_len + c_len] = 1

        batch_target_tokens.append(c_ids)

    # only hidden states are used, so skip the full-sequence logits
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    hidden_states = outputs.hidden_states

    slice_start = prompt_len - 1
    slice_end = prompt_len + max_cand_len

    final_norm = model.model.norm
    unembed = model.lm_head.weight

    # project one layer at a time, the full-vocab logits of all layers don't fit in memory
    layer_scores = []  # per layer: [batch, seq] log-probs of the target tokens
    targets = torch.zeros(
        (num_cands, max_cand_len), dtype=torch.long, device=device
    )
    for i, c_ids in enumerate(batch_target_tokens):
        targets[i, : len(c_ids)] = c_ids
    for layer, h in enumerate(hidden_states):
        hs = h[:, slice_start:slice_end, :]
        if layer < len(hidden_states) - 1:
            hs = final_norm(hs)  # last layer projected without norm
        log_probs = F.log_softmax(hs @ unembed.T, dim=-1)
        layer_scores.append(
            log_probs[:, :max_cand_len, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        )
        del log_probs
    all_scores = torch.stack(layer_scores)  # [layers, batch, seq]

    # gather scores
    scores_word = {}
    scores_first = {}

    for i, (lang, word) in enumerate(cand_keys):
        num_tokens = len(batch_target_tokens[i])
        token_scores = all_scores[:, i, :num_tokens]

        # Store scores per language and synonym (using word as key)
        if lang not in scores_word:
            scores_word[lang] = {}
            scores_first[lang] = {}

        scores_first[lang][word] = token_scores[:, 0].tolist()
        scores_word[lang][word] = token_scores.sum(dim=1).tolist()

    scores_normed = {}
    for lang, lang_scores in scores_word.items():
        scores_normed[lang] = {}
        for word, layer_scores in lang_scores.items():
            n_tokens = n_tokens_by_lang.get(lang, {}).get(word, 0)
            scores_normed[lang][word] = [s / max(1, n_tokens) for s in layer_scores]

    return scores_word, scores_normed, scores_first


def eval_logit_lens_per_lang(
    model,
    tokenizer,
    output_dir="results",
    data_path=WLT_DATA_PATH,
    force=False,
    logit_langs=LANGS,
    debug=False,
):
    for lang in logit_langs:
        eval_logit_lens(
            model,
            tokenizer,
            per_lang_fn=per_language_per_layer_scores_batched,
            data_path=data_path,
            output_dir=output_dir,
            lang=lang,
            force=force,
            debug=debug,
        )


@torch.no_grad()
def eval_logit_lens(
    model,
    tokenizer,
    per_lang_fn=per_language_per_layer_scores_batched,
    output_dir="results",
    data_path=WLT_DATA_PATH,
    lang="en",
    force=False,
    debug=False,
    test=False,
):
    results_path = get_logit_lens_results_path(
        output_dir=output_dir, lang=lang, data_path=data_path
    )
    print("Logit lens results path:", results_path)
    if os.path.exists(results_path) and not force:
        print(
            f"Found existing logit lens results for {lang} at {results_path}, skipping evaluation."
        )
        return {}

    print(f"Evaluating logit lens for lang {lang}...")

    results_by_src = {}
    for src_lang in LANGS:
        src_tgt_path = os.path.join(
            data_path,
            f"{src_lang}_to",
            f"{src_lang}_{lang}.csv",
        )
        # Skip if file doesn't exist
        if not os.path.exists(src_tgt_path):
            print(f"File not found: {src_tgt_path}, skipping.")
            continue

        valid_df = pd.read_csv(f"{data_path}/{src_lang}_to_valid.csv")
        src_tgt_df = pd.read_csv(src_tgt_path)
        if debug:
            src_tgt_df = src_tgt_df.head(10)
        lang_results = []
        for idx, row in src_tgt_df.iterrows():
            en_word = row["en_word"]
            src_word = row["src_word"]
            tgt_word = row["tgt_word"]
            prompt = row["prompt"]

            # Build candidates: always as lists
            candidates = {}
            valid_row = valid_df.loc[valid_df["word_original"] == en_word].iloc[0]
            for tf_lang in LANGS:
                if tf_lang == lang:
                    # tf_lang is the target language
                    if "valid_targets" in row:
                        # Parse pipe-separated valid targets
                        targets = [
                            t.strip() for t in str(row["valid_targets"]).split("|")
                        ]
                        candidates[lang] = targets
                        # the valid file has no column for its own language (repetition pairs)
                        if debug and lang in valid_row:
                            assert (
                                row["valid_targets"] == valid_row[lang]
                            ), f"Mismatch between valid_targets and valid_row for {en_word} in {lang}, row: {row['valid_targets']}, valid_row: {valid_row}"
                    else:
                        # fallback to target word
                        candidates[lang] = [row["tgt_word"]]
                elif tf_lang == src_lang:
                    candidates[tf_lang] = [row["src_word"]]
                else:
                    candidates[tf_lang] = [
                        t.strip() for t in str(valid_row[tf_lang]).split("|")
                    ]
            scores_word, scores_normed, scores_first_token = per_lang_fn(
                model, tokenizer, prompt, candidates, device=model.device
            )

            lang_results.append(
                {
                    "en_word": en_word,
                    "src": src_word,
                    "tgt": tgt_word,
                    "scores_normed": scores_normed,
                    "scores": scores_word,
                    "scores_first_token": scores_first_token,
                }
            )

        # Organize by source language
        if src_lang + "_to" not in results_by_src:
            results_by_src[src_lang + "_to"] = {}
        results_by_src[src_lang + "_to"][lang] = lang_results

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results_by_src, ensure_ascii=False, indent=2) + "\n")
        print(f"Wrote logit lens results to {results_path}")
    if test:
        return results_by_src
    return {}


