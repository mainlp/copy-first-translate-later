import csv
import json
import os
import re
from typing import Any, Dict, List
from collections import defaultdict
from tqdm import tqdm

from constants import (
    CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUN_NAME,
    LANGS,
    MULTI_BLIMP_LANGS,
    WLT_DATA_PATH,
)
from utils import get_checkpoint_results_dirs, prepare_tokenizer
from plot_utils import get_lang_from_logit_lens_file_name, load_results_dirs
import numpy as np
import pandas as pd

from constants import ISO2_to_ISO3

SIMILARITY_GROUP_ORDER = [
    "identical_subtokens",
    "partial_subtoken_overlap",
    "no_subtoken_overlap",
]


def _normalize_for_accuracy_mode(
    word: str,
    accuracy_mode: str = "original",
) -> str:
    normalized = (word or "").strip()
    if accuracy_mode == "ci":
        normalized = normalized.lower()
    return normalized


def _tokens_from_checkpoint(checkpoint_path: str) -> int:
    """Sum token counts from toks_processed.csv inside a checkpoint directory."""
    tokens_file = os.path.join(checkpoint_path, "toks_processed.csv")
    if not os.path.isfile(tokens_file):
        raise FileNotFoundError(f"toks_processed.csv not found at {tokens_file}")
    total = 0
    with open(tokens_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if "total" in row:
                continue
            total += int(row[1])
    return total


def _sorted_ckpts(ckpts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure checkpoints are processed in logical (epoch, step) order."""
    return sorted(
        ckpts,
        key=lambda x: (
            x.get("epoch", 0),
            x.get("step", 0),
            x.get("path", ""),
        ),
    )


def get_collate_name_tokens(run_name: str) -> str:
    """Get cache filename for tokens collation."""
    return f"{run_name}_tokens_processed.json"


def get_collate_name_logit_lens(
    run_name: str,
    file_name: str,
    score_type: str = "scores",
    aggregate_probs: bool = False,
) -> str:
    """Get cache filename for logit lens collation."""
    score_suffix = "" if score_type == "scores" else f"_{score_type}"
    agg_suffix = "_agg_probs" if aggregate_probs else ""
    return f"collated_{run_name}_{file_name.replace('.jsonl', '')}{score_suffix}{agg_suffix}.jsonl"


def _sanitize_cache_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def get_collate_name_logit_lens_similarity_groups(
    file_name: str,
    tokenizer_name_or_path: str,
    accuracy_mode: str = "original",
) -> str:
    """Get cache filename for tokenizer-dependent subtoken overlap groups."""
    accuracy_suffix = (
        f"_accuracy_{accuracy_mode}" if accuracy_mode != "original" else ""
    )
    tokenizer_tag = _sanitize_cache_component(tokenizer_name_or_path)
    return (
        f"logit_lens_similarity_groups_{file_name.replace('.jsonl', '')}"
        f"_{tokenizer_tag}{accuracy_suffix}.json"
    )


def get_collate_name_logit_lens_token_sets(
    tokenizer_name_or_path: str,
    accuracy_mode: str = "original",
) -> str:
    """Cache filename for tokenized string sets used by Jaccard grouping."""
    accuracy_suffix = (
        f"_accuracy_{accuracy_mode}" if accuracy_mode != "original" else ""
    )
    tokenizer_tag = _sanitize_cache_component(tokenizer_name_or_path)
    return f"logit_lens_token_sets_{tokenizer_tag}{accuracy_suffix}.json"


def _get_token_set_cached(text, tokenizer, token_set_cache, lowercase=False):
    key = text.lower() if lowercase else text
    cached = token_set_cache.get(key)
    if cached is not None:
        return set(cached)
    token_ids = tokenizer.encode(key, add_special_tokens=False)
    token_set_cache[key] = list(token_ids)
    return set(token_set_cache[key])


def subtoken_jaccard_cached(
    prompt,
    word,
    tokenizer,
    token_set_cache,
    lowercase=False,
):
    src_tokens = _get_token_set_cached(
        prompt,
        tokenizer,
        token_set_cache,
        lowercase=lowercase,
    )
    tgt_tokens = _get_token_set_cached(
        word,
        tokenizer,
        token_set_cache,
        lowercase=lowercase,
    )
    if not src_tokens and not tgt_tokens:
        return 0.0
    return len(src_tokens & tgt_tokens) / len(src_tokens | tgt_tokens)


def _load_token_set_cache(
    tokenizer_name_or_path: str,
    accuracy_mode: str = "original",
):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        CACHE_DIR,
        get_collate_name_logit_lens_token_sets(
            tokenizer_name_or_path,
            accuracy_mode=accuracy_mode,
        ),
    )
    if not os.path.exists(cache_path):
        return {}, cache_path
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    return cache, cache_path


def _save_token_set_cache(token_set_cache, cache_path: str):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(token_set_cache, f, indent=2, ensure_ascii=False)


def _bucket_subtoken_jaccard(score: float) -> str:
    if np.isclose(score, 1.0):
        return "identical_subtokens"
    if score > 0.0:
        return "partial_subtoken_overlap"
    return "no_subtoken_overlap"


def _parse_synonym_list(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split("|") if item and item.strip()]


def _load_logit_lens_tokenizer(
    mode: str,
    model_config_type: str = "llama1b.json",
):
    """Load the tokenizer used for logit-lens evaluation."""
    tokenizer_path = "mistralai/Mistral-Nemo-Base-2407"
    return prepare_tokenizer(tokenizer_path), tokenizer_path


def collate_logit_lens_similarity_groups(
    file_name: str,
    tokenizer,
    tokenizer_name_or_path: str,
    force: bool = False,
    data_path: str = WLT_DATA_PATH,
    accuracy_mode: str = "original",
):
    """Cache subtoken-Jaccard groups for each source-target pair and synonym."""
    effective_lowercase = accuracy_mode == "ci"
    token_set_cache, token_set_cache_path = _load_token_set_cache(
        tokenizer_name_or_path,
        accuracy_mode=accuracy_mode,
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    output_path = os.path.join(
        CACHE_DIR,
        get_collate_name_logit_lens_similarity_groups(
            file_name,
            tokenizer_name_or_path,
            accuracy_mode=accuracy_mode,
        ),
    )
    if os.path.exists(output_path) and not force:
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    tgt_lang = get_lang_from_logit_lens_file_name(file_name)
    pair_groups = {}

    for src_lang in LANGS:
        src_tgt_path = os.path.join(
            data_path, f"{src_lang}_to", f"{src_lang}_{tgt_lang}.csv"
        )
        valid_path = os.path.join(data_path, f"{src_lang}_to_valid.csv")
        if not os.path.exists(src_tgt_path) or not os.path.exists(valid_path):
            continue

        src_tgt_df = pd.read_csv(src_tgt_path)
        valid_df = pd.read_csv(valid_path).set_index("word_original")
        pair_key = f"{src_lang}_to_{tgt_lang}"
        pair_groups[pair_key] = {}

        for _, row in src_tgt_df.iterrows():
            en_word = str(row.get("en_word", "")).strip()
            if not en_word or en_word not in valid_df.index:
                continue

            valid_row = valid_df.loc[en_word]
            prompt = str(row.get("prompt", row.get("src_word", ""))).strip()
            src_word = str(row.get("src_word", "")).strip()
            entry_by_tf = {}

            for tf_lang in LANGS:
                if tf_lang == tgt_lang:
                    synonyms = _parse_synonym_list(row.get("valid_targets"))
                    if not synonyms:
                        tgt_word = str(row.get("tgt_word", "")).strip()
                        synonyms = [tgt_word] if tgt_word else []
                elif tf_lang == src_lang:
                    synonyms = [src_word] if src_word else []
                else:
                    synonyms = _parse_synonym_list(valid_row.get(tf_lang))

                synonym_groups = {}
                for synonym in synonyms:
                    jaccard = subtoken_jaccard_cached(
                        src_word,
                        synonym,
                        tokenizer,
                        token_set_cache,
                        lowercase=effective_lowercase,
                    )
                    synonym_groups[synonym] = {
                        "jaccard": jaccard,
                        "group": _bucket_subtoken_jaccard(jaccard),
                    }
                entry_by_tf[tf_lang] = synonym_groups

            pair_groups[pair_key][en_word] = {
                "prompt": prompt,
                "src_word": src_word,
                "groups_by_tf_lang": entry_by_tf,
            }

    cache_payload = {
        "file_name": file_name,
        "target_lang": tgt_lang,
        "data_path": data_path,
        "tokenizer": tokenizer_name_or_path,
        "accuracy_mode": accuracy_mode,
        "lowercase": effective_lowercase,
        "pair_groups": pair_groups,
        "group_order": list(SIMILARITY_GROUP_ORDER),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, indent=2, ensure_ascii=False)
    _save_token_set_cache(token_set_cache, token_set_cache_path)
    print(
        f"Saved token-set cache ({len(token_set_cache)} strings) to {token_set_cache_path}"
    )
    print(f"Saved logit-lens similarity groups to {output_path}")
    return cache_payload


def _get_selected_word_similarity_group(
    similarity_cache: Dict[str, Any],
    src_lang: str,
    tgt_lang: str,
    en_word: str,
    tf_lang: str,
    selected_word: str,
):
    pair_key = f"{src_lang}_to_{tgt_lang}"
    pair_entry = similarity_cache.get("pair_groups", {}).get(pair_key, {})
    concept_entry = pair_entry.get(en_word, {})
    tf_groups = concept_entry.get("groups_by_tf_lang", {}).get(tf_lang, {})
    if selected_word in tf_groups:
        return tf_groups[selected_word].get("group")

    selected_word_norm = selected_word.strip()
    for synonym, synonym_info in tf_groups.items():
        if synonym.strip() == selected_word_norm:
            return synonym_info.get("group")
    return None


def _aggregate_score_vectors(score_vectors, aggregate_probs=False):
    if not score_vectors:
        return [], []

    score_array = np.array(score_vectors, dtype=float)
    if aggregate_probs:
        score_array = np.exp(score_array)

    return list(np.mean(score_array, axis=0)), list(np.std(score_array, axis=0))


def get_collate_name_wlt(
    run_name: str,
    file_name: str,
    exclude_langs: List[str] | None = None,
    accuracy_mode: str = "original",
) -> str:
    suffix = f"_no-{'_'.join(exclude_langs)}" if exclude_langs else ""

    base_name = file_name.replace(".csv", "")
    if "accuracy" in base_name and accuracy_mode != "original":
        base_name = base_name.replace(
            "accuracy",
            f"accuracy_{accuracy_mode}",
        )

    return f"collated_{run_name}_{base_name}{suffix}.json"


def get_collate_name_predictions(
    run_name: str,
    file_prefix: str,
    exclude_langs: List[str] | None = None,
    metrics: str = "accuracy",
    accuracy_mode: str = "original",
) -> str:
    """Get cache filename for predictions over time collation."""
    metrics_tag = metrics
    if metrics == "accuracy" and accuracy_mode != "original":
        metrics_tag = f"accuracy_{accuracy_mode}"
    suffix = f"_no-{'_'.join(exclude_langs)}" if exclude_langs else ""
    prefix = file_prefix
    return f"collated_{run_name}_{prefix}_{metrics_tag}{suffix}.json"


def collate_tokens_processed(
    checkpoint_strategy,
    run_name: str,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build a list of checkpoints annotated with per-step and cumulative token counts,
    caching the result in CACHE_DIR for reuse.
    If force=True, recomputes from scratch and overwrites cache.

    Reads token counts from toks_processed.csv in checkpoint directories.
    """
    ckpts = _sorted_ckpts(checkpoint_strategy.list_checkpoints())
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, get_collate_name_tokens(run_name))

    if os.path.exists(cache_path) and not force:
        print(
            f"Collated results already exist at {cache_path}. Use --force to recompute."
        )
        with open(cache_path, "r") as f:
            return json.load(f)

    annotated: List[Dict[str, Any]] = []
    cumulative = 0

    for ckpt in tqdm(ckpts, desc="Processing checkpoints"):
        step_tokens = _tokens_from_checkpoint(ckpt["path"])
        cumulative += step_tokens
        annotated.append(
            {
                **ckpt,
                "tokens_step": step_tokens,
                "tokens_cumulative": cumulative,
            }
        )

    with open(cache_path, "w") as f:
        json.dump(annotated, f, indent=2)
    print(f"Saved tokens_processed cache to {cache_path}")
    return annotated


def collate_results_logit_lens(
    results_dirs,
    run_name,
    file_name="eval-logit_lens_en.jsonl",
    force=False,
    mode="expl-training",
    score_type="scores",
    aggregate_probs=False,
    model_config_type="llama1b.json",
    accuracy_mode="original",
):
    """Collate logit lens results across all checkpoints.
    We select the best synonym by final-layer score, then aggregate those selected
    vectors. Additionally, we bucket the selected synonym by tokenizer-dependent
    subtoken overlap with the source word and compute per-bucket mean/std.

    Args:
        aggregate_probs: If True, convert log probabilities to probabilities (exp),
                        compute mean/std in probability space, and keep those values.
                        If False, compute mean/std of log probabilities directly.
    """
    lang = get_lang_from_logit_lens_file_name(file_name)
    print(f"Processing language: {lang} with score_type: {score_type}")

    output_path = os.path.join(
        CACHE_DIR,
        get_collate_name_logit_lens(run_name, file_name, score_type, aggregate_probs),
    )
    if os.path.exists(output_path) and not force:
        # the similarity groups depend on accuracy_mode, which is not part of the file name
        with open(output_path, "r", encoding="utf-8") as f:
            cached_mode = json.load(f)["similarity_groups"]["accuracy_mode"]
        if cached_mode == accuracy_mode:
            print(
                f"Collated results for {file_name} ({score_type}) already exists. Skipping collation."
            )
            return
        print(
            f"Collated results for {file_name} used accuracy_mode={cached_mode}, recollating."
        )

    tokenizer, tokenizer_name_or_path = _load_logit_lens_tokenizer(
        mode=mode,
        model_config_type=model_config_type,
    )
    similarity_cache = collate_logit_lens_similarity_groups(
        file_name=file_name,
        tokenizer=tokenizer,
        tokenizer_name_or_path=tokenizer_name_or_path,
        force=force,
        data_path=WLT_DATA_PATH,
        accuracy_mode=accuracy_mode,
    )

    all_data = {
        "lang": lang,
        "score_type": score_type,
        "aggregate_probs": aggregate_probs,
        "similarity_groups": {
            "tokenizer": tokenizer_name_or_path,
            "accuracy_mode": accuracy_mode,
            "lowercase": accuracy_mode == "ci",
            "group_order": list(SIMILARITY_GROUP_ORDER),
            "cache_file": get_collate_name_logit_lens_similarity_groups(
                file_name,
                tokenizer_name_or_path,
                accuracy_mode=accuracy_mode,
            ),
        },
        "results": {lang_to + "_to": [] for lang_to in LANGS},
    }
    for results_dir in tqdm(results_dirs, desc="Processing files"):
        result_path = results_dir["path"]
        step = results_dir["step"]

        file_path = os.path.join(result_path, file_name)
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            try:
                data = json.loads(first_line)
            except json.JSONDecodeError:
                f.seek(0)
                data = json.load(f)

        for src_lang in LANGS:
            all_scores = {lang_tf: [] for lang_tf in LANGS}
            selected_words = {lang_tf: [] for lang_tf in LANGS}
            selected_similarity_groups = {lang_tf: [] for lang_tf in LANGS}
            grouped_scores = {
                lang_tf: {group: [] for group in SIMILARITY_GROUP_ORDER}
                for lang_tf in LANGS
            }

            src_lang_key = src_lang + "_to"
            if src_lang_key not in data:
                continue

            src_lang_data = data[src_lang_key]
            if isinstance(src_lang_data, dict) and any(
                k in LANGS for k in src_lang_data.keys()
            ):
                all_concepts = []
                for tf_lang_key in src_lang_data.keys():
                    if isinstance(src_lang_data[tf_lang_key], list):
                        all_concepts.extend(src_lang_data[tf_lang_key])
            else:
                all_concepts = src_lang_data if isinstance(src_lang_data, list) else []

            for concept in all_concepts:
                en_word = concept.get("en_word")
                scores_dict = concept.get(score_type, {})

                for lang_tf in LANGS:
                    lang_scores = scores_dict.get(lang_tf, {})
                    selected_group = None

                    if isinstance(lang_scores, dict):
                        if len(lang_scores) > 0:
                            best_word = max(
                                lang_scores.keys(),
                                key=lambda w: (
                                    lang_scores[w][-1]
                                    if lang_scores[w]
                                    else -float("inf")
                                ),
                            )
                            selected_scores = lang_scores[best_word]
                            selected_words[lang_tf].append(best_word)
                            selected_group = _get_selected_word_similarity_group(
                                similarity_cache=similarity_cache,
                                src_lang=src_lang,
                                tgt_lang=lang,
                                en_word=en_word,
                                tf_lang=lang_tf,
                                selected_word=best_word,
                            )
                        else:
                            selected_scores = []
                            selected_words[lang_tf].append(None)
                    else:
                        selected_scores = lang_scores
                        selected_words[lang_tf].append(None)

                    selected_similarity_groups[lang_tf].append(selected_group)
                    all_scores[lang_tf].append(selected_scores)
                    if (
                        selected_group in grouped_scores[lang_tf]
                        and len(selected_scores) > 0
                    ):
                        grouped_scores[lang_tf][selected_group].append(selected_scores)

            means = {}
            stds = {}
            grouped_stats = {}
            grouped_counts = {}
            for lang_tf in LANGS:
                mean_vals, std_vals = _aggregate_score_vectors(
                    all_scores[lang_tf],
                    aggregate_probs=aggregate_probs,
                )
                means[lang_tf] = mean_vals
                stds[lang_tf] = std_vals

                grouped_stats[lang_tf] = {}
                grouped_counts[lang_tf] = {}
                for group_name in SIMILARITY_GROUP_ORDER:
                    group_mean, group_std = _aggregate_score_vectors(
                        grouped_scores[lang_tf][group_name],
                        aggregate_probs=aggregate_probs,
                    )
                    grouped_stats[lang_tf][group_name] = {
                        "mean": group_mean,
                        "std": group_std,
                    }
                    grouped_counts[lang_tf][group_name] = len(
                        grouped_scores[lang_tf][group_name]
                    )

            agg = {
                "mean": means,
                "std": stds,
                "selected_words": selected_words,
                "selected_similarity_groups": selected_similarity_groups,
                "similarity_group_stats": grouped_stats,
                "similarity_group_counts": grouped_counts,
                "step": step,
            }
            all_data["results"][src_lang_key].append(agg)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved collated results ({score_type}) to {output_path}")


def _recompute_accuracy_from_entry(e, mode):
    if mode == "original":
        return e["accuracy"]

    gen = _normalize_for_accuracy_mode(e.get("generated_word", ""), mode)
    valids = e.get("valid_translations", [])
    valids = [_normalize_for_accuracy_mode(v, mode) for v in valids]

    if not gen:
        return 0.0

    return 1.0 if gen in valids else 0.0


def collate_results_wlt(
    results_dirs,
    run_name,
    file_name="eval-translation_accuracy.csv",
    force=False,
    exclude_langs=None,
    accuracy_mode="original",
):
    """Collate WLT accuracy results across all checkpoints."""
    print(f"Processing WLT metrics from {file_name}")

    # Filter languages
    langs = [lang for lang in LANGS if lang not in (exclude_langs or [])]

    output_path = os.path.join(
        CACHE_DIR,
        get_collate_name_wlt(
            run_name, file_name, exclude_langs, accuracy_mode=accuracy_mode
        ),
    )

    if os.path.exists(output_path) and not force:
        print(f"Collated results for {file_name} already exists. Skipping collation.")
        return

    all_data = {
        "file": file_name,
        "languages": list(langs),  # Store filtered language order
        "excluded_languages": list(exclude_langs or []),
        "steps": [],
    }

    for results_dir in tqdm(results_dirs, desc="Processing WLT files"):
        result_path = results_dir["path"]
        step = results_dir["step"]

        if accuracy_mode == "original":
            file_path = os.path.join(result_path, file_name)

            if not os.path.exists(file_path):
                continue

            # Read CSV as pandas DataFrame
            df = pd.read_csv(file_path, index_col=0)

            # Convert to matrix format: rows = source langs, cols = target langs
            # Only include non-excluded languages
            matrix = []
            for src_lang in langs:
                row = []
                for tgt_lang in langs:
                    if src_lang in df.index and tgt_lang in df.columns:
                        row.append(float(df.loc[src_lang, tgt_lang]))
                    else:
                        row.append(0.0)
                matrix.append(row)
        else:
            matrix = [[0.0 for _ in langs] for _ in langs]

            for tgt_lang in langs:
                pred_file = os.path.join(
                    result_path,
                    f"eval-translation_predictions_to_{tgt_lang}.json",
                )
                if not os.path.exists(pred_file):
                    continue

                with open(pred_file, "r", encoding="utf-8") as f:
                    content = json.load(f)

                for src_lang in langs:
                    entries = content.get(f"{src_lang}_to", [])
                    if not entries:
                        continue

                    vals = [
                        _recompute_accuracy_from_entry(e, accuracy_mode)
                        for e in entries
                    ]

                    i = langs.index(src_lang)
                    j = langs.index(tgt_lang)

                    matrix[i][j] = float(np.mean(vals)) if vals else 0.0

        step_data = {
            "step": step,
            "accuracy_matrix": matrix,  # NxN matrix where N = len(langs)
        }

        all_data["steps"].append(step_data)

    # Sort by step
    all_data["steps"].sort(key=lambda x: x["step"])

    # Save collated results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved collated WLT results to {output_path}")


def collate_predictions_over_time(
    results_dirs,
    run_name,
    file_prefix="eval-translation_predictions",
    force=False,
    exclude_langs=None,
    metrics="accuracy",
    mode="expl-training",
    accuracy_mode="original",
):
    """Collate per-checkpoint prediction jsons over time into one joint file."""
    langs = [lang for lang in LANGS if lang not in (exclude_langs or [])]

    # Compute prefix here
    prefix = file_prefix

    output_path = os.path.join(
        CACHE_DIR,
        get_collate_name_predictions(
            run_name,
            file_prefix,
            exclude_langs,
            metrics,
            accuracy_mode=accuracy_mode,
        ),
    )
    print("Saving to " + output_path)

    if os.path.exists(output_path) and not force:
        print(f"Collated predictions already exists at {output_path}. Skipping.")
        return

    all_results = {src_lang + "_to": {lang: {} for lang in langs} for src_lang in langs}
    steps_list = {f"steps_to_{lang}": [] for lang in langs}
    metrics_key = "accuracy" if metrics == "accuracy" else "log_prob_sum"

    # Process checkpoints in order, parsing each file once
    for result_dir in tqdm(results_dirs, desc="Processing prediction files"):
        step = result_dir["step"]

        result_path = result_dir["path"]
        for tgt in langs:
            file_name = f"{prefix}_to_{tgt}.json"
            file_path = os.path.join(result_path, file_name)
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            steps_list[f"steps_to_{tgt}"].append(step)
            # Loop over source languages
            for src_lang in langs:

                entries = content.get(f"{src_lang}_to")
                all_vals = []
                for e in entries:
                    # Use generated_* fields to get scalar values instead of dictionaries
                    if metrics == "accuracy":
                        if accuracy_mode == "original":
                            val = e[metrics_key]
                        else:
                            val = _recompute_accuracy_from_entry(e, accuracy_mode)
                    elif metrics == "probability":
                        val = np.exp(e["generated_log_prob_sum"])
                    elif metrics == "probability_normalized":
                        # Avoid division by zero
                        num_tokens = e.get("generated_num_tokens", 1)
                        if num_tokens <= 0:
                            num_tokens = 1
                        val = np.exp(e["generated_log_prob_sum"] / num_tokens)
                    all_vals.append(val)
                    en_word = e["en_word"]

                    # Track metric values
                    if en_word not in all_results[src_lang + "_to"][tgt]:
                        all_results[src_lang + "_to"][tgt][en_word] = [val]
                    else:
                        all_results[src_lang + "_to"][tgt][en_word].append(val)

                    # Track generated words (which synonym was predicted)
                    gen_word_key = f"_generated_{en_word}"
                    generated_word = e.get("generated_word", "")
                    if gen_word_key not in all_results[src_lang + "_to"][tgt]:
                        all_results[src_lang + "_to"][tgt][gen_word_key] = [
                            generated_word
                        ]
                    else:
                        all_results[src_lang + "_to"][tgt][gen_word_key].append(
                            generated_word
                        )

                    # Track synonym with highest log probability
                    highest_log_prob_key = f"_highest_log_prob_{en_word}"
                    log_prob_sum = e.get("log_prob_sum", {})
                    if log_prob_sum:
                        # Find synonym with highest log probability
                        best_synonym = max(
                            log_prob_sum.keys(), key=lambda k: log_prob_sum[k]
                        )
                    else:
                        best_synonym = ""

                    if highest_log_prob_key not in all_results[src_lang + "_to"][tgt]:
                        all_results[src_lang + "_to"][tgt][highest_log_prob_key] = [
                            best_synonym
                        ]
                    else:
                        all_results[src_lang + "_to"][tgt][highest_log_prob_key].append(
                            best_synonym
                        )

                all_vals = np.array(all_vals)
                mean = float(np.mean(all_vals))
                std = float(np.std(all_vals))
                if "_mean_" + metrics not in all_results[src_lang + "_to"][tgt]:
                    all_results[src_lang + "_to"][tgt]["_mean_" + metrics] = [mean]
                else:
                    all_results[src_lang + "_to"][tgt]["_mean_" + metrics].append(mean)
                if "_std_" + metrics not in all_results[src_lang + "_to"][tgt]:
                    all_results[src_lang + "_to"][tgt]["_std_" + metrics] = [std]
                else:
                    all_results[src_lang + "_to"][tgt]["_std_" + metrics].append(std)
    all_results["step_list"] = steps_list
    all_results["languages"] = langs
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved collated predictions to {output_path}")


def get_collate_name_mblimp(run_name: str, file_prefix: str = "multiblimp") -> str:
    """Get cache filename for BLiMP collation."""
    return f"{run_name}_eval-{file_prefix}_collated.json"

JBLIMP_EASY_PHENOMENA = {
    "argument structure",
    "morphology",
    "nominal structure",
    "ellipsis",
}

# as categorized by authors
ZHOBLIMP_EASY_PHENOMENA = {
    "verb_phrase_left_negation",
    "verb_phrase_left_adverbial",
    "verb_negation_particle",
    "left_adverbial_b",
    "left_adverbial_d",
    "left_adverbial_e",
    "left_adverbial_negation",
    "left_dou",
    "right_yijing_a",
    "right_yijing_b",
    "topicalization_OSV",
    "topicalization_OSV_mei",
    "topicalization_SOV",
    "topicalization_SOV_mei",
    "control_modal_vs_raising_modal",
    "existential_there_subject_raising",
    "modal_raising_hui",
    "modal_raising_topicalization",
    "question_A_not_A",
    "question_A_not_A_daodi_a",
    "question_A_not_A_daodi_b",
    "question_A_not_A_indirect",
    "question_V_not_VP_1",
    "question_V_not_VP_2",
    "question_daodi_nandao_1",
    "question_daodi_nandao_2",
    "question_daodi_nandao_A_not_A_intran",
    "question_daodi_nandao_A_not_A_tran",
    "question_daodi_negation",
    "question_nandao_negation",
    "question_nandao_raising_1_a",
    "question_nandao_raising_1_b",
    "question_nandao_raising_2",
    "question_nandao_raising_3",
    "question_nandao_scope_1",
    "question_nandao_scope_2",
    "question_particle_daodi_choice_tran",
    "question_particle_daodi_choice_intran",
    "question_particle_nandao",
    "BA_verb_le_a",
    "BA_verb_le_b",
    "BA_inversion",
    "BA_deletion",
    "BA_duplicate_argument",
    "BA_negation",
    "BA_no_progressive",
    "BA_no_stative_verb",
    "BA_meiba",
    "BA_BEI_subj_drop",
    "BA_suo_adverbial_a",
    "BA_suo_adverbial_b",
    "relative_operator_intepretation",
    "relative_operator_who",
    "relativization_movement_no_gap",
    "relativization_movement_when_where",
}

def compute_blimp_filtered(samples_path, lang, metric="acc_norm"):
    vals = []
    keep = JBLIMP_EASY_PHENOMENA if lang == "ja" else ZHOBLIMP_EASY_PHENOMENA
    with open(samples_path) as f:
        for line in f:
            e = json.loads(line)
            if e["doc"]["phenomenon"] in keep:
                vals.append(e[metric])
    n = len(vals)
    mean = float(np.mean(vals))
    stderr = float(np.std(vals) / np.sqrt(n))
    return mean, stderr


def collate_results_mblimp(
    results_dirs,
    run_name,
    force=False,
    exclude_langs=None,
):
    """Collate BLiMP aggregated results across all checkpoints.

    Expects per-language files like:
        {file_prefix}_{lang}_agg.json
    e.g.:
        multiblimp_deu_agg.json
    """
    file_prefix = "multiblimp"
    print(f"Processing BLiMP metrics with prefix '{file_prefix}'")

    file_names = {
        lang: f"multiblimp_{ISO2_to_ISO3[lang]}_agg.json" for lang in MULTI_BLIMP_LANGS
    }
    file_names["zh"] = "zhoblimp_subset.jsonl"
    file_names["ja"] = "jblimp.jsonl"

    # only languages with a BLiMP benchmark
    langs = [
        lang
        for lang in LANGS
        if lang in file_names and lang not in (exclude_langs or [])
    ]

    collate_name = get_collate_name_mblimp(run_name, file_prefix)
    output_path = os.path.join(CACHE_DIR, collate_name)

    if os.path.exists(output_path) and not force:
        print("Collated BLiMP results already exist. Skipping.")
        return

    all_data = {
        "file_prefix": file_prefix,
        "languages": list(langs),
        "excluded_languages": list(exclude_langs or []),
        "steps": [],
    }

    for results_dir in tqdm(results_dirs, desc="Processing BLiMP files"):
        result_path = results_dir["path"]
        step = results_dir["step"]

        missing = [
            file_names[lang]
            for lang in langs
            if not os.path.exists(os.path.join(result_path, file_names[lang]))
        ]
        if missing:
            print(f"Skipping step {step}: missing BLiMP results {missing}")
            continue

        lang_scores = {}

        for lang in langs:
            file_name = file_names[lang]
            file_path = os.path.join(result_path, file_name)

            if lang == "ja" or lang == "zh":
                lang_scores[lang] = {}
                for metric in ["acc", "acc_norm"]:
                    mean, stderr = compute_blimp_filtered(file_path, lang, metric=metric)
                    lang_scores[lang][metric] = mean
                    lang_scores[lang][f"{metric}_stderr"] = stderr
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)

            lang_scores[lang] = {
                "acc": float(content.get("acc,none", 0.0)),
                "acc_stderr": float(content.get("acc_stderr,none", 0.0)),
                "acc_norm": float(content.get("acc_norm,none", 0.0)),
                "acc_norm_stderr": float(content.get("acc_norm_stderr,none", 0.0)),
            }

        step_data = {
            "step": step,
            "language_scores": lang_scores,
        }

        all_data["steps"].append(step_data)

    all_data["steps"].sort(key=lambda x: x["step"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"Saved collated BLiMP results to {output_path}")


def get_collate_name_output_classification(
    run_name: str,
    tgt_lang: str = None,
    exclude_langs: List[str] | None = None,
    accuracy_mode: str = "original",
) -> str:
    """Get cache filename for output classification collation, grouped by target language."""
    suffix = f"_no-{'_'.join(exclude_langs)}" if exclude_langs else ""
    accuracy_suffix = (
        f"_accuracy_{accuracy_mode}" if accuracy_mode != "original" else ""
    )
    if tgt_lang:
        return f"collated_{run_name}_output_classification_to_{tgt_lang}{accuracy_suffix}{suffix}.json"
    else:
        return f"collated_{run_name}_output_classification_all_pairs{accuracy_suffix}{suffix}.json"


def get_collate_name_overlap_bin_accuracy(
    run_name: str,
    exclude_langs: List[str] | None = None,
    accuracy_mode: str = "original",
) -> str:
    """Get cache filename for overlap-bin accuracy-over-time collation."""
    suffix = f"_no-{'_'.join(exclude_langs)}" if exclude_langs else ""
    accuracy_suffix = (
        f"_accuracy_{accuracy_mode}" if accuracy_mode != "original" else ""
    )
    return (
        f"collated_{run_name}_overlap_bin_accuracy"
        f"{accuracy_suffix}{suffix}.json"
    )


def _best_case_overlap_group_for_entry(
    src_word: str,
    valid_translations: list,
    tokenizer,
    token_set_cache: dict,
    accuracy_mode: str = "original",
) -> str:
    """Assign a single fixed overlap bin using best-case synonym priority.

    Priority (user-confirmed): identical > partial > no overlap.
    """
    src = _normalize_for_accuracy_mode(src_word, accuracy_mode)
    if not src:
        return "no_subtoken_overlap"

    if not isinstance(valid_translations, list) or not valid_translations:
        return "no_subtoken_overlap"

    best_idx = len(SIMILARITY_GROUP_ORDER) - 1
    seen = set()

    for synonym in valid_translations:
        syn = _normalize_for_accuracy_mode(str(synonym), accuracy_mode)
        if not syn or syn in seen:
            continue
        seen.add(syn)

        score = subtoken_jaccard_cached(
            src,
            syn,
            tokenizer,
            token_set_cache,
            lowercase=(accuracy_mode == "ci"),
        )
        group = _bucket_subtoken_jaccard(score)
        idx = SIMILARITY_GROUP_ORDER.index(group)
        if idx < best_idx:
            best_idx = idx
            if best_idx == 0:
                break

    return SIMILARITY_GROUP_ORDER[best_idx]


def collate_overlap_bin_accuracy_over_time(
    results_dirs,
    run_name,
    file_prefix="eval-translation_predictions",
    force=False,
    exclude_langs=None,
    mode="expl-training",
    accuracy_mode="original",
):
    """Collate accuracy-over-time for fixed overlap bins.

    Each concept in a src->tgt pair is assigned to one fixed overlap bin once
    (best-case across valid target synonyms): identical > partial > no overlap.
    For each step and bin, accuracy is computed as correct/total over assigned concepts.
    """
    langs = [lang for lang in LANGS if lang not in (exclude_langs or [])]
    output_path = os.path.join(
        CACHE_DIR,
        get_collate_name_overlap_bin_accuracy(
            run_name=run_name,
            exclude_langs=exclude_langs,
            accuracy_mode=accuracy_mode,
        ),
    )

    if os.path.exists(output_path) and not force:
        print(
            f"Overlap-bin accuracy collation already exists at {output_path}. Skipping."
        )
        return

    # Use canonical WLT sources/translations for stable per-concept bin assignment.
    wlt_data_path = WLT_DATA_PATH
    wlt_data = load_wlt_data(
        data_path=wlt_data_path,
        accuracy_mode=accuracy_mode,
        active_langs=langs,
    )

    tokenizer, tokenizer_name_or_path = _load_logit_lens_tokenizer(mode=mode)
    token_set_cache, token_set_cache_path = _load_token_set_cache(
        tokenizer_name_or_path,
        accuracy_mode=accuracy_mode,
    )
    prefix = file_prefix

    pair_word_to_bin = {}
    step_to_bin_total = defaultdict(lambda: defaultdict(int))
    step_to_bin_correct = defaultdict(lambda: defaultdict(int))
    step_to_bin_source_copy = defaultdict(lambda: defaultdict(int))
    step_to_bin_context_copy = defaultdict(lambda: defaultdict(int))
    all_steps_set = set()

    for result_dir in tqdm(results_dirs, desc="Processing overlap-bin accuracy"):
        step = int(result_dir["step"])
        result_path = result_dir["path"]

        for tgt_lang in langs:
            file_name = f"{prefix}_to_{tgt_lang}.json"
            file_path = os.path.join(result_path, file_name)
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)

            all_steps_set.add(step)

            for src_lang in langs:
                if src_lang == tgt_lang:
                    # Exclude repetition pairs for this analysis.
                    continue

                entries = content.get(f"{src_lang}_to", [])
                for e in entries:
                    en_word = e.get("en_word")
                    if not en_word:
                        continue

                    key = (src_lang, tgt_lang, en_word)

                    # Resolve canonical source/synonym data per entry for classification.
                    src_word = (
                        e.get("src_word")
                        or wlt_data.get("source_words", {}).get(en_word, {}).get(src_lang, "")
                    )

                    valid_translations = []
                    csv_syns = (
                        wlt_data.get("translations", {})
                        .get(src_lang, {})
                        .get(en_word, {})
                        .get(tgt_lang, set())
                    )
                    if csv_syns:
                        valid_translations = list(csv_syns)
                    elif e.get("valid_translations"):
                        valid_translations = list(e.get("valid_translations", []))
                    elif e.get("target_word"):
                        valid_translations = [e.get("target_word")]

                    if key not in pair_word_to_bin:
                        pair_word_to_bin[key] = _best_case_overlap_group_for_entry(
                            src_word=src_word,
                            valid_translations=valid_translations,
                            tokenizer=tokenizer,
                            token_set_cache=token_set_cache,
                            accuracy_mode=accuracy_mode,
                        )

                    overlap_bin = pair_word_to_bin[key]
                    step_to_bin_total[step][overlap_bin] += 1

                    generated_word = e.get("generated_word", "")
                    gen_norm = _normalize_for_accuracy_mode(generated_word, accuracy_mode)
                    src_norm = _normalize_for_accuracy_mode(src_word, accuracy_mode)
                    is_source_copy = bool(src_norm and gen_norm == src_norm)

                    context_words = (
                        wlt_data.get("context_words", {})
                        .get(src_lang, {})
                        .get(tgt_lang, {})
                        .get(en_word, set())
                    )
                    if not context_words:
                        context_words = _extract_context_words_from_prompt(
                            e.get("prompt", ""),
                            accuracy_mode=accuracy_mode,
                        )

                    classification, _ = classify_output(
                        generated_word=generated_word,
                        src_word=src_word,
                        context_words=context_words,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        en_word=en_word,
                        wlt_data=wlt_data,
                        valid_translations=valid_translations,
                        accuracy_mode=accuracy_mode,
                        candidate_langs=langs,
                    )
                    is_correct = classification in {"correct", "copy_correct"}
                    is_context_copy = classification == "context_copying"

                    step_to_bin_correct[step][overlap_bin] += int(is_correct)
                    step_to_bin_source_copy[step][overlap_bin] += int(is_source_copy)
                    step_to_bin_context_copy[step][overlap_bin] += int(
                        is_context_copy
                    )

    sorted_steps = sorted(all_steps_set)
    steps_payload = []
    curves = {bin_name: [] for bin_name in SIMILARITY_GROUP_ORDER}
    source_word_copying_curves = {bin_name: [] for bin_name in SIMILARITY_GROUP_ORDER}
    context_copying_curves = {bin_name: [] for bin_name in SIMILARITY_GROUP_ORDER}
    totals = {bin_name: [] for bin_name in SIMILARITY_GROUP_ORDER}

    for step in sorted_steps:
        bins_payload = {}
        for bin_name in SIMILARITY_GROUP_ORDER:
            total = int(step_to_bin_total[step][bin_name])
            correct = int(step_to_bin_correct[step][bin_name])
            source_copy = int(step_to_bin_source_copy[step][bin_name])
            context_copy = int(step_to_bin_context_copy[step][bin_name])
            acc = (correct / total) if total > 0 else 0.0
            source_copy_rate = (source_copy / total) if total > 0 else 0.0
            context_copy_rate = (context_copy / total) if total > 0 else 0.0

            bins_payload[bin_name] = {
                "accuracy": float(acc),
                "correct": correct,
                "total": total,
                "source_word_copying_rate": float(source_copy_rate),
                "source_word_copying_count": source_copy,
                "context_copying_rate": float(context_copy_rate),
                "context_copying_count": context_copy,
            }
            curves[bin_name].append(float(acc))
            source_word_copying_curves[bin_name].append(float(source_copy_rate))
            context_copying_curves[bin_name].append(float(context_copy_rate))
            totals[bin_name].append(total)

        steps_payload.append(
            {
                "step": step,
                "bins": bins_payload,
            }
        )

    payload = {
        "run_name": run_name,
        "accuracy_mode": accuracy_mode,
        "tokenizer": tokenizer_name_or_path,
        "bin_priority": list(SIMILARITY_GROUP_ORDER),
        "assignment": "best_case_fixed_per_pair_word",
        "languages": langs,
        "exclude_repetition_pairs": True,
        "n_assigned_pair_words": len(pair_word_to_bin),
        "steps": steps_payload,
        "curves": curves,
        "source_word_copying_curves": source_word_copying_curves,
        "context_copying_curves": context_copying_curves,
        "totals": totals,
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _save_token_set_cache(token_set_cache, token_set_cache_path)
    print(f"Saved overlap-bin accuracy collation to {output_path}")


def _extract_context_words_from_prompt(
    prompt: str,
    accuracy_mode: str = "original",
) -> set[str]:
    """Extract quoted in-context words from a WLT prompt string."""

    if not isinstance(prompt, str) or not prompt:
        return set()

    quoted = re.findall(r'"([^"]+)"', prompt)
    context_words = set()
    for word in quoted:
        normalized = _normalize_for_accuracy_mode(word, accuracy_mode)
        if normalized:
            context_words.add(normalized)
    return context_words


def _load_example_pair_vocab(
    data_path: str,
    accuracy_mode: str = "original",
    active_langs: list[str] | None = None,
) -> dict:
    """Build pair-specific vocab from example_pool.csv once.

    Returns:
        {src_lang: {tgt_lang: set(normalized_target_words_from_example_pool_for_pair)}}
    """
    example_pool_file = os.path.join(data_path, "example_pool.csv")
    langs = active_langs if active_langs is not None else LANGS
    pair_vocab = {src: {tgt: set() for tgt in langs} for src in langs}

    if not os.path.exists(example_pool_file):
        return pair_vocab

    try:
        df = pd.read_csv(example_pool_file)
    except Exception as e:
        print(f"Error loading example_pool.csv: {e}")
        return pair_vocab

    for _, row in df.iterrows():
        for src in langs:
            for tgt in langs:
                tgt_word = row.get(tgt, "")

                if isinstance(tgt_word, str) and tgt_word.strip():
                    normalized_tgt = _normalize_for_accuracy_mode(
                        tgt_word,
                        accuracy_mode,
                    )
                    pair_vocab[src][tgt].add(normalized_tgt)

    return pair_vocab


def load_wlt_data(
    data_path: str = WLT_DATA_PATH,
    accuracy_mode: str = "original",
    active_langs: list[str] | None = None,
):
    """
    Load WLT data from CSV files.
    - query_set.csv: Source words for all languages
    - {src_lang}_to_valid.csv: Valid translations FROM src_lang TO all other languages

    Returns:
        dict: {
            "source_words": {en_word: {lang: word}},  # From query_set.csv
            "translations": {src_lang: {en_word: {tgt_lang: set(synonyms)}}}  # From {src}_to_valid.csv
            "context_words": {src_lang: {tgt_lang: {en_word: set(words)}}}  # From {src}_to/{src}_{tgt}.csv prompt column
        }
    """
    langs = active_langs if active_langs is not None else LANGS
    wlt_data = {"source_words": {}, "translations": {}, "context_words": {}}
    example_pair_vocab = _load_example_pair_vocab(
        data_path=data_path,
        accuracy_mode=accuracy_mode,
        active_langs=langs,
    )

    # Load source words from query_set.csv
    query_set_file = os.path.join(data_path, "query_set.csv")
    if os.path.exists(query_set_file):
        try:
            df = pd.read_csv(query_set_file)
            for _, row in df.iterrows():
                en_word = row.get("word_original") or row.get("en_word")
                if not en_word:
                    continue

                wlt_data["source_words"][en_word] = {}
                for lang in langs:
                    if lang in df.columns:
                        src_word = row.get(lang, "")
                        if isinstance(src_word, str) and src_word:
                            wlt_data["source_words"][en_word][lang] = src_word.strip()
        except Exception as e:
            print(f"Error loading query_set.csv: {e}")

    # Load translations for each source language from {src_lang}_to_valid.csv
    for src_lang in langs:
        csv_file = os.path.join(data_path, f"{src_lang}_to_valid.csv")

        if not os.path.exists(csv_file):
            print(f"Warning: WLT data file not found: {csv_file}")
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

        wlt_data["translations"][src_lang] = {}

        for _, row in df.iterrows():
            en_word = row.get("word_original") or row.get("en_word")

            if not en_word:
                continue

            wlt_data["translations"][src_lang][en_word] = {}

            # Parse translations from each language column
            for tgt_lang in langs:
                if tgt_lang not in df.columns:
                    continue

                synonyms_str = row.get(tgt_lang, "")
                if isinstance(synonyms_str, str) and synonyms_str:
                    # Split by pipe for multiple synonyms
                    syns = set()
                    for syn in synonyms_str.split("|"):
                        syn = syn.strip()
                        if syn:
                            syns.add(syn)
                    wlt_data["translations"][src_lang][en_word][tgt_lang] = syns

    # Load language-pair-specific in-context prompt words from {src}_to/{src}_{tgt}.csv
    for src_lang in langs:
        wlt_data["context_words"][src_lang] = {}
        src_dir = os.path.join(data_path, f"{src_lang}_to")

        for tgt_lang in langs:
            prompt_csv = os.path.join(src_dir, f"{src_lang}_{tgt_lang}.csv")
            if not os.path.exists(prompt_csv):
                continue

            try:
                df = pd.read_csv(prompt_csv)
            except Exception as e:
                print(f"Error loading {prompt_csv}: {e}")
                continue

            concept_to_context_words = {}
            for _, row in df.iterrows():
                en_word = row.get("en_word") or row.get("word_original")
                if not en_word:
                    continue

                prompt = row.get("prompt", "")
                prompt_words = _extract_context_words_from_prompt(
                    prompt,
                    accuracy_mode=accuracy_mode,
                )
                pair_vocab = example_pair_vocab.get(src_lang, {}).get(tgt_lang, set())
                context_words = {w for w in prompt_words if w in pair_vocab}
                concept_to_context_words[en_word] = context_words

            wlt_data["context_words"][src_lang][tgt_lang] = concept_to_context_words

    return wlt_data


def classify_output(
    generated_word: str,
    src_word: str,
    context_words: set[str],
    src_lang: str,
    tgt_lang: str,
    en_word: str,
    wlt_data: dict,
    valid_translations: list = None,
    accuracy_mode: str = "original",
    candidate_langs: list | None = None,
) -> tuple:
    """
    Classify a single generated output.

    Args:
        generated_word: The word generated by the model
        src_word: The source language word (prompt from prediction file)
        src_lang: Source language code
        tgt_lang: Target language code
        en_word: English concept name
        wlt_data: WLT data for wrong-language checks
        valid_translations: List of valid target translations from prediction file
        candidate_langs: Languages to consider for wrong-language detection.
                If None, defaults to all LANGS.

    Returns:
        Tuple of (classification, wrong_lang)
        - classification: One of "correct", "copy_correct", "copy", "context_copying", "concept_wrong_lang", "other"
        - wrong_lang: Language code if concept_wrong_lang, else None
    """
    if not generated_word:
        return ("other", None)

    gen_stripped = _normalize_for_accuracy_mode(generated_word, accuracy_mode)
    src_stripped = _normalize_for_accuracy_mode(src_word, accuracy_mode)

    # Check if exact copy (exact string match like WLT eval)
    is_copy = bool(src_stripped and gen_stripped == src_stripped)

    # Check if correct translation
    if valid_translations is not None:
        # Use valid_translations from prediction file
        tgt_syns_stripped = {
            _normalize_for_accuracy_mode(s, accuracy_mode) for s in valid_translations
        }
        is_correct = gen_stripped in tgt_syns_stripped
    elif (
        src_lang in wlt_data["translations"]
        and en_word in wlt_data["translations"][src_lang]
    ):
        # Fallback to CSV if valid_translations not provided
        tgt_translations = wlt_data["translations"][src_lang][en_word].get(
            tgt_lang, set()
        )
        tgt_syns_stripped = {
            _normalize_for_accuracy_mode(s, accuracy_mode) for s in tgt_translations
        }
        is_correct = gen_stripped in tgt_syns_stripped
    else:
        is_correct = False

    # Classify based on copy + correct status
    if is_copy and is_correct:
        return ("copy_correct", None)
    elif is_copy:
        return ("copy", None)
    elif is_correct:
        return ("correct", None)

    # Check if generated output is copied from in-context examples for this src->tgt prompt
    if gen_stripped in context_words:
        return ("context_copying", None)

    # Check if concept-correct but wrong language
    # Check in translations from the same source language
    if (
        src_lang in wlt_data["translations"]
        and en_word in wlt_data["translations"][src_lang]
    ):
        langs_to_check = candidate_langs if candidate_langs is not None else LANGS
        for lang in langs_to_check:
            if lang == tgt_lang:
                continue
            lang_translations = wlt_data["translations"][src_lang][en_word].get(
                lang, set()
            )
            lang_syns_stripped = {
                _normalize_for_accuracy_mode(s, accuracy_mode)
                for s in lang_translations
            }
            if gen_stripped in lang_syns_stripped:
                return ("concept_wrong_lang", lang)

    return ("other", None)


def _classify_src_generated_subtoken_overlap(
    src_word: str,
    generated_word: str,
    tokenizer,
    token_set_cache: dict,
    accuracy_mode: str = "original",
) -> str:
    """Bucket src-vs-generated subtoken overlap for output classification stats."""
    src = (src_word or "").strip()
    generated = (generated_word or "").strip()
    if not src or not generated:
        return "no_subtoken_overlap"

    score = subtoken_jaccard_cached(
        src,
        generated,
        tokenizer,
        token_set_cache,
        lowercase=(accuracy_mode == "ci"),
    )
    return _bucket_subtoken_jaccard(score)


def collate_output_classification(
    results_dirs,
    run_name,
    file_prefix="eval-translation_predictions",
    force=False,
    exclude_langs=None,
    mode="expl-training",
    tgt_lang=None,
    accuracy_mode="original",
):
    """
    Collate output classifications for WLT over time.
    Groups by target language to minimize file reads.

    Args:
        results_dirs: List of checkpoint result directories
        run_name: Name of the run
        file_prefix: Prefix for prediction files
        force: Force recomputation
        exclude_langs: Languages to exclude
        mode: Checkpoint strategy mode (e.g. expl-training)
        tgt_lang: Target language to collate (or None for all languages)
    """

    langs = [lang for lang in LANGS if lang not in (exclude_langs or [])]

    # Determine target languages to process
    if tgt_lang:
        tgt_langs = [tgt_lang] if tgt_lang in langs else []
    else:
        tgt_langs = langs

    # Pre-filter targets that actually need work to avoid unnecessary tokenizer load.
    pending_tgt_langs = []
    for tgt in tgt_langs:
        output_path = os.path.join(
            CACHE_DIR,
            get_collate_name_output_classification(
                run_name=run_name,
                tgt_lang=tgt,
                exclude_langs=exclude_langs,
                accuracy_mode=accuracy_mode,
            ),
        )
        if os.path.exists(output_path) and not force:
            print(f"Output classification for to_{tgt} already exists. Skipping.")
            continue
        pending_tgt_langs.append(tgt)

    if not pending_tgt_langs:
        print("All requested output-classification files already exist.")
        return

    # Load WLT synonym data once
    print("Loading WLT synonym data...")
    wlt_data_path = WLT_DATA_PATH
    wlt_data = load_wlt_data(
        data_path=wlt_data_path,
        accuracy_mode=accuracy_mode,
        active_langs=langs,
    )
    tokenizer, tokenizer_name_or_path = _load_logit_lens_tokenizer(mode=mode)
    token_set_cache, token_set_cache_path = _load_token_set_cache(
        tokenizer_name_or_path,
        accuracy_mode=accuracy_mode,
    )

    # Compute prefix
    prefix = file_prefix

    # Process each target language that needs (re)computation.
    for tgt in pending_tgt_langs:
        output_path = os.path.join(
            CACHE_DIR,
            get_collate_name_output_classification(
                run_name=run_name,
                tgt_lang=tgt,
                exclude_langs=exclude_langs,
                accuracy_mode=accuracy_mode,
            ),
        )

        print(f"Processing output classifications for all sources → {tgt}...")

        # Aggregate classifications over all source languages

        step_to_counts = defaultdict(lambda: defaultdict(int))
        step_to_wrong_lang_counts = defaultdict(lambda: defaultdict(int))
        step_to_total = defaultdict(int)
        step_to_translation_overlap_counts = defaultdict(lambda: defaultdict(int))
        step_to_translation_total = defaultdict(int)

        # Also track per source language
        src_step_to_counts = {
            src: defaultdict(lambda: defaultdict(int)) for src in langs
        }
        src_step_to_wrong_lang_counts = {
            src: defaultdict(lambda: defaultdict(int)) for src in langs
        }
        src_step_to_total = {src: defaultdict(int) for src in langs}
        src_step_to_translation_overlap_counts = {
            src: defaultdict(lambda: defaultdict(int)) for src in langs
        }
        src_step_to_translation_total = {src: defaultdict(int) for src in langs}

        all_steps_set = set()

        for result_dir in tqdm(
            results_dirs, desc=f"Processing classifications to {tgt}"
        ):
            step = result_dir["step"]
            result_path = result_dir["path"]

            file_name = f"{prefix}_to_{tgt}.json"
            file_path = os.path.join(result_path, file_name)

            if not os.path.exists(file_path):
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)

            all_steps_set.add(step)

            # Process each source language
            for src in langs:
                entries = content.get(f"{src}_to", [])

                for e in entries:
                    en_word = e.get("en_word")
                    generated_word = e.get("generated_word", "")
                    src_word = e.get("src_word") or wlt_data.get(
                        "source_words", {}
                    ).get(en_word, {}).get(src, "")
                    valid_translations = e.get("valid_translations", [])
                    context_words = (
                        wlt_data.get("context_words", {})
                        .get(src, {})
                        .get(tgt, {})
                        .get(en_word, set())
                    )
                    if not context_words:
                        context_words = _extract_context_words_from_prompt(
                            e.get("prompt", ""),
                            accuracy_mode=accuracy_mode,
                        )

                    if not en_word:
                        continue

                    # Classify using valid_translations from prediction file
                    classification, wrong_lang = classify_output(
                        generated_word,
                        src_word,
                        context_words,
                        src,
                        tgt,
                        en_word,
                        wlt_data,
                        valid_translations,
                        accuracy_mode=accuracy_mode,
                        candidate_langs=langs,
                    )

                    # Aggregate across all sources
                    step_to_counts[step][classification] += 1
                    step_to_total[step] += 1

                    # Track translation accuracy split by src/generated subtoken overlap.
                    is_translation = classification in {"correct", "copy_correct"}
                    if is_translation:
                        overlap_group = _classify_src_generated_subtoken_overlap(
                            src_word=src_word,
                            generated_word=generated_word,
                            tokenizer=tokenizer,
                            token_set_cache=token_set_cache,
                            accuracy_mode=accuracy_mode,
                        )
                        step_to_translation_overlap_counts[step][overlap_group] += 1
                        step_to_translation_total[step] += 1

                    if classification == "concept_wrong_lang" and wrong_lang:
                        step_to_wrong_lang_counts[step][wrong_lang] += 1

                    # Track per source language
                    src_step_to_counts[src][step][classification] += 1
                    src_step_to_total[src][step] += 1

                    if is_translation:
                        src_step_to_translation_overlap_counts[src][step][
                            overlap_group
                        ] += 1
                        src_step_to_translation_total[src][step] += 1

                    if classification == "concept_wrong_lang" and wrong_lang:
                        src_step_to_wrong_lang_counts[src][step][wrong_lang] += 1

        # Convert to output format
        sorted_steps = sorted(list(all_steps_set))

        # Aggregated across all sources
        aggregated = {
            "steps": sorted_steps,
            "correct": [],
            "copy": [],
            "copy_correct": [],
            "context_copying": [],
            "concept_wrong_lang": [],
            "concept_wrong_lang_breakdown": {lang: [] for lang in LANGS},
            "other": [],
            "translation": [],
            "translation_exact_subtoken_copy": [],
            "translation_non_exact_subtoken_copy": [],
            "translation_no_subtoken_overlap": [],
            "translation_exact_subtoken_copy_share": [],
            "translation_non_exact_subtoken_copy_share": [],
            "translation_no_subtoken_overlap_share": [],
        }

        for step in sorted_steps:
            total = step_to_total[step]
            counts = step_to_counts[step]
            wrong_lang_counts = step_to_wrong_lang_counts[step]
            translation_total = step_to_translation_total[step]
            exact_subtoken_copy = step_to_translation_overlap_counts[step][
                "identical_subtokens"
            ]
            partial_subtoken_copy = step_to_translation_overlap_counts[step][
                "partial_subtoken_overlap"
            ]
            no_subtoken_overlap = step_to_translation_overlap_counts[step][
                "no_subtoken_overlap"
            ]
            for category in [
                "correct",
                "copy",
                "copy_correct",
                "context_copying",
                "concept_wrong_lang",
                "other",
            ]:
                rate = counts[category] / total if total > 0 else 0
                aggregated[category].append(rate)
            for lang in LANGS:
                rate = wrong_lang_counts[lang] / total if total > 0 else 0
                aggregated["concept_wrong_lang_breakdown"][lang].append(rate)
            translation_rate = translation_total / total if total > 0 else 0
            aggregated["translation"].append(translation_rate)
            aggregated["translation_exact_subtoken_copy"].append(
                exact_subtoken_copy / total if total > 0 else 0
            )
            aggregated["translation_non_exact_subtoken_copy"].append(
                partial_subtoken_copy / total if total > 0 else 0
            )
            aggregated["translation_no_subtoken_overlap"].append(
                no_subtoken_overlap / total if total > 0 else 0
            )
            # Shares among translation outputs (correct/copy_correct)
            aggregated["translation_exact_subtoken_copy_share"].append(
                exact_subtoken_copy / translation_total if translation_total > 0 else 0
            )
            aggregated["translation_non_exact_subtoken_copy_share"].append(
                partial_subtoken_copy / translation_total
                if translation_total > 0
                else 0
            )
            aggregated["translation_no_subtoken_overlap_share"].append(
                no_subtoken_overlap / translation_total if translation_total > 0 else 0
            )

        # Per source language breakdown
        per_source = {}
        for src in langs:
            per_source[src] = {
                "steps": sorted_steps,
                "correct": [],
                "copy": [],
                "copy_correct": [],
                "context_copying": [],
                "concept_wrong_lang": [],
                "concept_wrong_lang_breakdown": {lang: [] for lang in LANGS},
                "other": [],
                "translation": [],
                "translation_exact_subtoken_copy": [],
                "translation_non_exact_subtoken_copy": [],
                "translation_no_subtoken_overlap": [],
                "translation_exact_subtoken_copy_share": [],
                "translation_non_exact_subtoken_copy_share": [],
                "translation_no_subtoken_overlap_share": [],
            }

            for step in sorted_steps:
                total = src_step_to_total[src][step]
                counts = src_step_to_counts[src][step]
                wrong_lang_counts = src_step_to_wrong_lang_counts[src][step]
                translation_total = src_step_to_translation_total[src][step]
                exact_subtoken_copy = src_step_to_translation_overlap_counts[src][step][
                    "identical_subtokens"
                ]
                partial_subtoken_copy = src_step_to_translation_overlap_counts[src][
                    step
                ]["partial_subtoken_overlap"]
                no_subtoken_overlap = src_step_to_translation_overlap_counts[src][step][
                    "no_subtoken_overlap"
                ]

                for category in [
                    "correct",
                    "copy",
                    "copy_correct",
                    "context_copying",
                    "concept_wrong_lang",
                    "other",
                ]:
                    rate = counts[category] / total if total > 0 else 0
                    per_source[src][category].append(rate)

                for lang in LANGS:
                    rate = wrong_lang_counts[lang] / total if total > 0 else 0
                    per_source[src]["concept_wrong_lang_breakdown"][lang].append(rate)

                translation_rate = translation_total / total if total > 0 else 0
                per_source[src]["translation"].append(translation_rate)

                per_source[src]["translation_exact_subtoken_copy"].append(
                    exact_subtoken_copy / total if total > 0 else 0
                )
                per_source[src]["translation_non_exact_subtoken_copy"].append(
                    partial_subtoken_copy / total if total > 0 else 0
                )
                per_source[src]["translation_no_subtoken_overlap"].append(
                    no_subtoken_overlap / total if total > 0 else 0
                )
                per_source[src]["translation_exact_subtoken_copy_share"].append(
                    exact_subtoken_copy / translation_total
                    if translation_total > 0
                    else 0
                )
                per_source[src]["translation_non_exact_subtoken_copy_share"].append(
                    partial_subtoken_copy / translation_total
                    if translation_total > 0
                    else 0
                )
                per_source[src]["translation_no_subtoken_overlap_share"].append(
                    no_subtoken_overlap / translation_total
                    if translation_total > 0
                    else 0
                )

        # Save
        output_data = {
            "tgt_lang": tgt,
            "aggregated": aggregated,
            "per_source": per_source,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved output classifications to {output_path}")

        # Persist incrementally so interrupted runs keep newly tokenized strings.
        _save_token_set_cache(token_set_cache, token_set_cache_path)

    _save_token_set_cache(token_set_cache, token_set_cache_path)


if __name__ == "__main__":
    import argparse
    from expl_training_utils import ExplTrainingCheckpointStrategy

    parser = argparse.ArgumentParser(
        description="Collate results from checkpoints (tokens, logit lens, WLT)."
    )
    parser.add_argument(
        "--mode",
        choices=["expl-training"],
        default="expl-training",
        help="Checkpoint strategy mode.",
    )
    parser.add_argument(
        "--run_name",
        default=DEFAULT_RUN_NAME,
        help="Run name under DEFAULT_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--model-config-type",
        default="llama1b.json",
        help="Model config filename (for expl-training mode).",
    )
    parser.add_argument(
        "--collate",
        choices=["tokens", "logit_lens", "jaccard_groups", "wlt", "mblimp", "all"],
        default="all",
        help="Which collation(s) to run.",
    )
    parser.add_argument(
        "--exclude_lang",
        type=str,
        default=None,
        help="Comma-separated list of languages to exclude (e.g., 'id' or 'id,fr').",
    )
    parser.add_argument(
        "--score_type",
        choices=["scores", "scores_normed", "scores_first_token"],
        default="scores",
        help="Type of scores to collate for logit lens.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation.",
    )
    parser.add_argument(
        "--aggregate_probs",
        action="store_true",
        help="Aggregate probabilities (exp of log probs) before computing means, instead of aggregating log probs directly.",
    )
    parser.add_argument(
        "--accuracy_mode",
        default="original",
        choices=["original", "ci"],
    )
    args = parser.parse_args()

    # Parse exclude_lang
    exclude_langs = (
        [lang.strip() for lang in args.exclude_lang.split(",")]
        if args.exclude_lang
        else []
    )

    # Setup strategy & results_dirs
    run_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.run_name)
    ckpt_dir, _ = get_checkpoint_results_dirs(run_dir)
    if ckpt_dir is None:
        parser.error("No checkpoints directory found.")
    strategy = ExplTrainingCheckpointStrategy(
        model_config_type=args.model_config_type,
        checkpoints_dir=ckpt_dir,
    )

    results_dirs = load_results_dirs(
        os.path.join(DEFAULT_OUTPUT_DIR, args.run_name), run_name=args.run_name
    )

    # Run collations
    do_tokens = args.collate in ("tokens", "all")
    do_logit_lens = args.collate in ("logit_lens", "all")
    do_jaccard_groups = args.collate in ("jaccard_groups", "all")
    do_wlt = args.collate in ("wlt", "all")
    do_mblimp = args.collate in ("mblimp", "all")

    if do_tokens:
        print("\n=== Collating tokens ===")
        annotated = collate_tokens_processed(
            strategy, run_name=args.run_name, force=args.force
        )
        if annotated:
            print(
                f"Processed {len(annotated)} checkpoints. "
                f"Last cumulative tokens: {annotated[-1]['tokens_cumulative']}"
            )

    if do_logit_lens:
        print("\n=== Collating logit lens ===")
        tgt_langs = [lang for lang in LANGS if lang not in exclude_langs]
        for tgt_lang in tgt_langs:
            file_name = (
                f"eval-logit_lens_{tgt_lang}.jsonl"
            )
            collate_results_logit_lens(
                results_dirs,
                run_name=args.run_name,
                file_name=file_name,
                force=args.force,
                mode=args.mode,
                score_type=args.score_type,
                aggregate_probs=args.aggregate_probs,
                model_config_type=args.model_config_type,
                accuracy_mode=args.accuracy_mode,
            )

    if do_jaccard_groups:
        print("\n=== Precomputing Jaccard similarity groups ===")
        tgt_langs = [lang for lang in LANGS if lang not in exclude_langs]
        tokenizer, tokenizer_name_or_path = _load_logit_lens_tokenizer(
            mode=args.mode,
            model_config_type=args.model_config_type,
        )
        data_path = WLT_DATA_PATH
        for tgt_lang in tgt_langs:
            file_name = (
                f"eval-logit_lens_{tgt_lang}.jsonl"
            )
            collate_logit_lens_similarity_groups(
                file_name=file_name,
                tokenizer=tokenizer,
                tokenizer_name_or_path=tokenizer_name_or_path,
                force=args.force,
                data_path=data_path,
                accuracy_mode=args.accuracy_mode,
            )

    if do_wlt:
        print("\n=== Collating WLT ===")
        base = "eval-translation_accuracy.csv"
        collate_results_wlt(
            results_dirs,
            run_name=args.run_name,
            file_name=base,
            force=args.force,
            exclude_langs=exclude_langs,
            accuracy_mode=args.accuracy_mode,
        )
        collate_predictions_over_time(
            results_dirs,
            run_name=args.run_name,
            file_prefix="eval-translation_predictions",
            force=args.force,
            exclude_langs=exclude_langs,
            mode=args.mode,
            accuracy_mode=args.accuracy_mode,
        )

    if do_mblimp:
        results_dirs = load_results_dirs(
            os.path.join(DEFAULT_OUTPUT_DIR, args.run_name),
            run_name=args.run_name,
        )
        collate_results_mblimp(
            results_dirs,
            run_name=args.run_name,
            force=args.force,
            exclude_langs=exclude_langs,
        )
