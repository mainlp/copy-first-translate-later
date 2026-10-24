import json
import numpy as np
import os
import torch
import gc

import pandas as pd

from constants import (
    BENCHMARKS,
    FEW_SHOT_DELIMITER,
    LANGS,
    HARNESS_METRICS,
    MAX_HARNESS_SAMPLES,
    CUSTOM_TASK_PATH,
    WLT_DATA_PATH,
    WLT_PREFIX,
    TRANSLATION_ACC_KEY,
    TRANSLATION_PROB_KEY,
    COPY_ACC_KEY,
    COPY_PROB_KEY,
    TRANSLATION_PROB_NORMALIZED_KEY,
    COPY_PROB_NORMALIZED_KEY,
)
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

from logit_lens import eval_logit_lens_per_lang


_TASK_MANAGER = None


def get_task_manager():
    """Lazy singleton to avoid re-initialization."""
    global _TASK_MANAGER
    if _TASK_MANAGER is None:
        _TASK_MANAGER = TaskManager(include_path=CUSTOM_TASK_PATH)
    return _TASK_MANAGER


def filter_results(d, task_metrics: dict[str, list[str]], parent_key=""):
    items = {}

    if parent_key == "" and "results" in d:
        d = d["results"]
        parent_key = ""

    for task, metrics in d.items():
        if task not in task_metrics:
            continue
        for metric_key in task_metrics[task]:
            value = metrics.get(metric_key)
            if isinstance(value, (int, float)):
                items[f"{task}/{metric_key}"] = value
    return items


@torch.no_grad()
def evaluate_all(
    model,
    tokenizer,
    output_dir="results",
    benchmarks=BENCHMARKS,
    force=False,
    debug=False,
):
    results = dict()
    model.eval()

    logit_langs = [
        b.split("_")[-1]
        for b in benchmarks
        if b.startswith("logit_lens_")
        and b.split("_")[-1] in LANGS
    ]

    # logit lens eval
    eval_logit_lens_per_lang(
        model,
        tokenizer,
        output_dir=output_dir,
        force=force,
        logit_langs=logit_langs,
        debug=debug,
    )

    if "word_level_translation" in benchmarks:
        # translation metrics
        translation_results = eval_translation_metrics(
            model,
            tokenizer,
            output_dir=output_dir,
            force=force,
            data_path=WLT_DATA_PATH,
            key_prefix=WLT_PREFIX,
        )
        results.update(translation_results)

    # run harness tasks
    # we don't re-validate benchmark tasks here, since we checked in parse_args
    benchmarks = [b for b in benchmarks if b in HARNESS_METRICS]
    harness_results = run_harness_tasks(
        model,
        tokenizer,
        tasks=benchmarks,
        max_samples=20 if debug else MAX_HARNESS_SAMPLES,
        output_dir=output_dir,
        force=force,
    )
    flat_harness = filter_results(harness_results, HARNESS_METRICS)
    results.update(flat_harness)
    return results


def get_harness_results_paths(task, output_dir="results"):
    results_path = f"{output_dir}/{task}.jsonl"
    agg_path = f"{output_dir}/{task}_agg.json"
    return results_path, agg_path


def run_harness_tasks(
    model,
    tokenizer,
    tasks,
    batch_size="auto",
    max_samples=MAX_HARNESS_SAMPLES,
    output_dir="results",
    merge_results=True,
    force=False,
):
    if model is None:
        if merge_results is False:
            raise ValueError("Must set merge_results=True when model is None.")
        force = False  # can't force eval without a model

    max_samples = None if max_samples < 0 else max_samples
    if isinstance(tasks, str):
        tasks = [tasks]

    all_results = {}
    if model is not None:
        harness_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
        )
    for task in tasks:
        results_path, agg_path = get_harness_results_paths(task, output_dir=output_dir)
        if not force and os.path.exists(results_path) and os.path.exists(agg_path):
            if merge_results:
                print(f"Loading existing results for task {task} from {results_path}.")
                with open(agg_path, "r") as f:
                    all_results[task] = json.load(f)
            else:
                print(
                    f"Found existing results for task {task} at {results_path}, skipping."
                )
            continue
        elif model is None:
            print(
                f"No existing results for task {task} at {results_path}, and no model "
                "provided to evaluate, skipping. This should not happen!"
            )
            continue
        print(f"Evaluating task {task}...")
        results = evaluator.simple_evaluate(
            model=harness_model,
            tasks=[task],
            batch_size=batch_size,
            limit=max_samples,
            use_cache=None,
            task_manager=get_task_manager(),
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        agg = results["results"][task]
        all_results[task] = agg
        with open(results_path, "w", encoding="utf-8") as f:
            for task_subset, samples in results["samples"].items():
                for ex in samples:
                    # Retain subset tag explicitly in the serialized sample record.
                    ex["task_subset"] = task_subset
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with open(agg_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(agg, ensure_ascii=False) + "\n")
    return all_results


@torch.no_grad()
def _strip_padding(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    return input_ids[attention_mask.bool()]


def split_at_quote(tokenizer, cont_ids, lp):
    cut = None
    for j, tid in enumerate(cont_ids):
        if '"' in tokenizer.decode([tid], skip_special_tokens=True):
            cut = j
            break
    if cut is None:
        return lp, None
    return lp[:cut], lp[cut]


@torch.no_grad()
def score_continuations(model, tokenizer, prompt_ids_list, cont_ids_list, device):
    batch_size = len(prompt_ids_list)

    p_lens = [len(p) for p in prompt_ids_list]
    c_lens = [len(c) for c in cont_ids_list]

    max_p = max(p_lens)
    max_c = max(c_lens)
    total_len = max_p + max_c

    input_ids = torch.full(
        (batch_size, total_len),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, total_len),
        dtype=torch.long,
        device=device,
    )

    for i, (p, c) in enumerate(zip(prompt_ids_list, cont_ids_list)):
        p_len = len(p)
        c_len = len(c)

        input_ids[i, :p_len] = p
        input_ids[i, p_len : p_len + c_len] = c
        attention_mask[i, : p_len + c_len] = 1

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    all_token_logps = []
    for i, (p_ids, c_ids) in enumerate(zip(prompt_ids_list, cont_ids_list)):
        p_len = len(p_ids)
        c_len = len(c_ids)

        step_logits = logits[i, p_len - 1 : p_len - 1 + c_len]
        step_logps = torch.log_softmax(step_logits, dim=-1)

        token_lp = [step_logps[j, c_ids[j]].item() for j in range(c_len)]
        all_token_logps.append(token_lp)

    return all_token_logps


def extract_generated_words(tokenizer, gen_out, attention_mask):
    texts = []
    words = []
    gen_token_ids = []

    for i in range(gen_out.shape[0]):
        mask = attention_mask[i]
        prompt_len = int(mask.sum())
        first_prompt_idx = (mask == 1).nonzero(as_tuple=True)[0][0].item()
        prompt_end = first_prompt_idx + prompt_len
        cont_ids = gen_out[i, prompt_end:]

        full_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
        q = full_text.find('"')
        if q != -1:
            # find the quote token
            for j in range(len(cont_ids)):
                partial_text = tokenizer.decode(
                    cont_ids[: j + 1], skip_special_tokens=True
                )
                if '"' in partial_text:
                    cont_ids = cont_ids[: j + 1]
                    break

        gen_token_ids.append(cont_ids)
        word = full_text[:q].strip() if q != -1 else full_text

        texts.append(full_text)
        words.append(word)
    return texts, words, gen_token_ids


def score_valid_translations_batch(
    model, tokenizer, all_p_ids, valid_translations_list, device, delim_len
):
    log_probs_dicts = []
    log_prob_sum_dicts = []
    log_prob_quote_dicts = []

    for sample_idx, vt_list in enumerate(valid_translations_list):
        p_ids = all_p_ids[sample_idx]

        vt_texts = [vt + FEW_SHOT_DELIMITER for vt in vt_list]
        tok = tokenizer(
            vt_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)

        vt_ids = [
            _strip_padding(tok.input_ids[j], tok.attention_mask[j])
            for j in range(len(vt_texts))
        ]

        prompt_rep = [p_ids] * len(vt_ids)

        vt_token_logps = score_continuations(
            model, tokenizer, prompt_rep, vt_ids, device
        )

        lp_dict = {}
        lp_sum_dict = {}
        lp_quote_dict = {}

        for vt, lp in zip(vt_list, vt_token_logps):
            lp_dict[vt] = lp[:-delim_len]
            lp_sum_dict[vt] = sum(lp[:-delim_len])
            # pick only the first, covers: merged or multiple, we just care about "
            lp_quote_dict[vt] = lp[-delim_len]

        log_probs_dicts.append(lp_dict)
        log_prob_sum_dicts.append(lp_sum_dict)
        log_prob_quote_dicts.append(lp_quote_dict)

    return log_probs_dicts, log_prob_sum_dicts, log_prob_quote_dicts


@torch.no_grad()
def get_greedy_accuracy_batched(
    src,
    tgt,
    model,
    tokenizer,
    delim_len,
    batch_size=100,
    data_path="data/word_level_translations",
    device="cuda",
    debug=False,
):
    folder_path = os.path.join(data_path, f"{src}_to")
    file_path = os.path.join(folder_path, f"{src}_{tgt}.csv")
    df = pd.read_csv(file_path)

    if debug:
        df = df.head(10)
        batch_size = 10

    all_accuracies = []
    all_probs = []
    all_probs_normalized = []
    all_predictions = []

    max_new_tokens = 16

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]

        prompts = batch_df["prompt"].astype(str).tolist()
        src_words = batch_df["src_word"].astype(str).tolist()
        tgt_words = batch_df["tgt_word"].astype(str).tolist()
        en_words = batch_df["en_word"].astype(str).tolist()

        valid_translations_list = []
        for idx_batch, row in batch_df.iterrows():
            if "valid_targets" in batch_df.columns and pd.notna(row["valid_targets"]):
                valid_translations = str(row["valid_targets"]).split("|")
                valid_translations = [t.strip() for t in valid_translations]
            else:
                valid_translations = [tgt_words[batch_df.index.get_loc(idx_batch)]]
            valid_translations_list.append(valid_translations)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            max_length=128,
            truncation=True,
        ).to(device)

        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        all_p_ids = [
            _strip_padding(inputs.input_ids[j], inputs.attention_mask[j])
            for j in range(len(prompts))
        ]

        generated_texts, gen_words, all_gen_ids = extract_generated_words(
            tokenizer, gen_out, inputs.attention_mask
        )
        gen_token_logps = score_continuations(
            model, tokenizer, all_p_ids, all_gen_ids, device
        )

        log_probs_dicts, log_prob_sum_dicts, log_prob_quote_dicts = (
            score_valid_translations_batch(
                model,
                tokenizer,
                all_p_ids,
                valid_translations_list,
                device,
                delim_len=delim_len,
            )
        )

        for gen_word, valid_trans in zip(gen_words, valid_translations_list):
            # Check if generated word matches any valid translation
            is_correct = any(gen_word == vt for vt in valid_trans)
            all_accuracies.append(1 if is_correct else 0)

        for idx, (
            en_word,
            src_word,
            tgt_word,
            gen_text,
            acc,
            valid_translations,
        ) in enumerate(
            zip(
                en_words,
                src_words,
                tgt_words,
                generated_texts,
                all_accuracies[-len(prompts) :],
                valid_translations_list,
            )
        ):
            log_probs_dict = log_probs_dicts[idx]
            log_prob_sum_dict = log_prob_sum_dicts[idx]
            log_prob_quote_dict = log_prob_quote_dicts[idx]

            best_trans = max(log_prob_sum_dict, key=log_prob_sum_dict.get)
            log_prob_sum = log_prob_sum_dict[best_trans]
            num_tokens = len(log_probs_dict[best_trans])

            prob = np.exp(log_prob_sum)
            prob_normalized = np.exp(log_prob_sum / max(1, num_tokens))

            all_probs.append(prob)
            all_probs_normalized.append(prob_normalized)

            content_lp, quote_lp = split_at_quote(
                tokenizer, all_gen_ids[idx], gen_token_logps[idx]
            )
            generated_log_probs = content_lp
            generated_log_prob_sum = sum(content_lp)
            generated_log_prob_quote = quote_lp

            all_predictions.append(
                {
                    "en_word": en_word,
                    "src_lang": src,
                    "tgt_lang": tgt,
                    "prompt": src_word,
                    "target_word": tgt_word,
                    "valid_translations": valid_translations,
                    "prediction": gen_text.strip(),
                    "accuracy": acc,
                    "log_prob_sum": log_prob_sum_dict,
                    "log_probs": log_probs_dict,
                    "log_prob_quote": log_prob_quote_dict,
                    "num_tokens": {
                        vt: len(log_probs_dict[vt]) for vt in valid_translations
                    },
                    "generated_word": gen_words[idx],
                    "generated_log_prob_sum": generated_log_prob_sum,
                    "generated_num_tokens": len(generated_log_probs),
                    "generated_log_probs": generated_log_probs,
                    "generated_log_prob_quote": generated_log_prob_quote,
                }
            )

    return (
        np.mean(all_accuracies),
        np.mean(all_probs),
        np.mean(all_probs_normalized),
        all_predictions,
    )


@torch.no_grad()
def get_greedy_accuracy(
    src,
    tgt,
    model,
    tokenizer,
    delim_len,
    batch_size=None,  # unused, but for compatibility
    data_path=WLT_DATA_PATH,
    device="cuda",
    debug=False,
):
    """Non-batched version of get_greedy_accuracy_batched, for testing purposes."""
    folder_path = os.path.join(data_path, f"{src}_to")
    file_path = os.path.join(folder_path, f"{src}_{tgt}.csv")
    df = pd.read_csv(file_path)

    if debug:
        df = df.head(10)

    all_accuracies = []
    all_probs = []
    all_probs_normalized = []
    all_predictions = []
    max_new_tokens = 16

    # Iterate through each sample individually
    for idx, row in df.iterrows():
        prompt = str(row["prompt"])
        tgt_word = str(row["tgt_word"])
        en_word = str(row["en_word"])

        # Get valid translations for this word from valid_targets column
        if "valid_targets" in row and pd.notna(row["valid_targets"]):
            valid_translations = str(row["valid_targets"]).split("|")
            valid_translations = [t.strip() for t in valid_translations]
        else:
            # Fallback: use single tgt_word
            valid_translations = [tgt_word.strip()]

        # Generate
        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, max_length=128, truncation=True
        ).to(device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        input_len = inputs.input_ids.shape[1]
        cont_ids = gen_out[0, input_len:]

        full_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
        q = full_text.find('"')

        if q != -1:
            for j in range(len(cont_ids)):
                partial_text = tokenizer.decode(
                    cont_ids[: j + 1], skip_special_tokens=True
                )
                if '"' in partial_text:
                    cont_ids = cont_ids[: j + 1]
                    break

        gen_word = full_text[:q].strip() if q != -1 else full_text.strip()

        # Compute accuracy against any valid translation
        is_correct = any(gen_word == vt.strip() for vt in valid_translations)
        all_accuracies.append(1 if is_correct else 0)

        # Compute log probs
        p_ids = inputs.input_ids[0]

        # Compute log probs for all valid translations
        log_probs_dict = {}
        log_prob_sum_dict = {}
        log_prob_quote_dict = {}

        for valid_trans in valid_translations:
            # Match batched version: word + FEW_SHOT_DELIMITER
            trans_text = valid_trans.strip() + FEW_SHOT_DELIMITER
            trans_ids = tokenizer(
                trans_text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0].to(device)

            full_input_t = torch.cat(
                [p_ids.unsqueeze(0), trans_ids.unsqueeze(0)], dim=1
            )
            out_t = model(full_input_t)

            p_len = len(p_ids)
            trans_len = len(trans_ids)

            trans_logits = out_t.logits[0, p_len - 1 : p_len - 1 + trans_len, :]
            trans_log_probs = torch.log_softmax(trans_logits, dim=-1)

            content_len = trans_len - delim_len

            trans_log_probs_list = [
                trans_log_probs[j, trans_ids[j]].item() for j in range(content_len)
            ]

            log_probs_dict[valid_trans] = trans_log_probs_list
            log_prob_sum_dict[valid_trans] = sum(trans_log_probs_list)

            # first delimiter token (the quote)
            log_prob_quote_dict[valid_trans] = trans_log_probs[
                content_len, trans_ids[content_len]
            ].item()

        best_trans = max(log_prob_sum_dict, key=log_prob_sum_dict.get)
        best_log_probs = log_probs_dict[best_trans]
        log_prob_sum = log_prob_sum_dict[best_trans]
        prob = np.exp(log_prob_sum)
        prob_normalized = np.exp(log_prob_sum / max(1, len(best_log_probs)))
        all_probs.append(prob)
        all_probs_normalized.append(prob_normalized)

        # use the actual generated tokens
        p_len = len(p_ids)
        c_len = len(cont_ids)

        full_input_g = torch.cat([p_ids.unsqueeze(0), cont_ids.unsqueeze(0)], dim=1)
        out_g = model(full_input_g)

        gen_logits = out_g.logits[0, p_len - 1 : p_len - 1 + c_len, :]
        gen_log_probs_sm = torch.log_softmax(gen_logits, dim=-1)

        lp = [gen_log_probs_sm[j, cont_ids[j]].item() for j in range(c_len)]

        content_lp, quote_lp = split_at_quote(tokenizer, cont_ids, lp)

        generated_log_probs = content_lp
        generated_log_prob_sum = sum(content_lp)
        generated_log_prob_quote = quote_lp

        all_predictions.append(
            {
                "en_word": en_word,
                "src_lang": src,
                "tgt_lang": tgt,
                "target_word": tgt_word,
                "prediction": full_text.strip(),
                "accuracy": all_accuracies[-1],
                "valid_translations": valid_translations,
                "log_prob_sum": log_prob_sum_dict,
                "log_probs": log_probs_dict,
                "log_prob_quote": log_prob_quote_dict,
                "num_tokens": {
                    vt: len(log_probs_dict[vt]) for vt in valid_translations
                },
                "generated_word": gen_word,
                "generated_log_prob_sum": generated_log_prob_sum,
                "generated_num_tokens": len(generated_log_probs),
                "generated_log_probs": generated_log_probs,
                "generated_log_prob_quote": generated_log_prob_quote,
            }
        )

    return (
        np.mean(all_accuracies),
        np.mean(all_probs),
        np.mean(all_probs_normalized),
        all_predictions,
    )


def get_translation_metrics_results_path(
    output_dir="results",
    data_path=WLT_DATA_PATH,
):
    return f"{output_dir}/eval-translation"


def eval_translation_metrics(
    model,
    tokenizer,
    output_dir="results",
    data_path=WLT_DATA_PATH,
    key_prefix=WLT_PREFIX,
    get_accuracy_fn=get_greedy_accuracy_batched,
    batch_size=100,
    force=False,
    debug=False,
):

    delim_len = len(
        tokenizer(FEW_SHOT_DELIMITER, add_special_tokens=False)["input_ids"]
    )
    # probability metrics score valid translations tokenized in isolation, which
    # assumes this matches their tokenization after the opening quote (not the case
    # e.g. for sentencepiece tokenizers adding a prefix space); accuracy is computed
    # on decoded text and is unaffected
    enc = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    if enc('"word' + FEW_SHOT_DELIMITER) != enc('"') + enc("word" + FEW_SHOT_DELIMITER):
        print(
            "Warning: this tokenizer tokenizes words differently in isolation than "
            "after a quote; WLT probability metrics will be unreliable "
            "(accuracy is unaffected)."
        )
    results_path = get_translation_metrics_results_path(
        output_dir=output_dir, data_path=data_path
    )
    print("Translation metrics results path:", results_path)
    results_paths = [
        results_path + "_accuracy.csv",
        results_path + "_probability.csv",
        results_path + "_probability_normalized.csv",
    ]
    if all(os.path.exists(p) and not force for p in results_paths):
        print(
            f"Found existing translation accuracy results at {results_path}, skipping evaluation."
        )
        return {}

    matrix_data = []
    for tgt in LANGS:
        predictions_by_src = {src_lang + "_to": [] for src_lang in LANGS}
        for src in LANGS:
            acc, prob, prob_normalized, preds = get_accuracy_fn(
                src,
                tgt,
                model,
                tokenizer,
                batch_size=batch_size,
                device=model.device,
                debug=debug,
                data_path=data_path,
                delim_len=delim_len,
            )
            predictions_by_src[src + "_to"].extend(preds)
            if acc is None:
                acc, prob = np.nan, np.nan
            matrix_data.append(
                {
                    "Source": src,
                    "Target": tgt,
                    "Accuracy": acc,
                    "Probability": prob,
                    "Probability normalized": prob_normalized,
                }
            )

        predictions_path = f"{results_path}_predictions_to_{tgt}.json"
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions_by_src, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions for target language {tgt} to {predictions_path}")

    df_all = pd.DataFrame(matrix_data)
    matrix_acc = df_all.pivot(index="Source", columns="Target", values="Accuracy")
    matrix_prob = df_all.pivot(index="Source", columns="Target", values="Probability")
    matrix_prob_normalized = df_all.pivot(
        index="Source", columns="Target", values="Probability normalized"
    )

    matrix_acc.to_csv(results_path + "_accuracy.csv")
    matrix_prob.to_csv(results_path + "_probability.csv")
    matrix_prob_normalized.to_csv(results_path + "_probability_normalized.csv")

    mask = np.eye(len(LANGS)) == 1
    return {
        key_prefix + TRANSLATION_ACC_KEY: float(np.matrix(matrix_acc)[~mask].mean()),
        key_prefix + TRANSLATION_PROB_KEY: float(np.matrix(matrix_prob)[~mask].mean()),
        key_prefix
        + TRANSLATION_PROB_NORMALIZED_KEY: float(
            np.matrix(matrix_prob_normalized)[~mask].mean()
        ),
        key_prefix + COPY_ACC_KEY: float(np.matrix(matrix_acc)[mask].mean()),
        key_prefix + COPY_PROB_KEY: float(np.matrix(matrix_prob)[mask].mean()),
        key_prefix
        + COPY_PROB_NORMALIZED_KEY: float(
            np.matrix(matrix_prob_normalized)[mask].mean()
        ),
    }
