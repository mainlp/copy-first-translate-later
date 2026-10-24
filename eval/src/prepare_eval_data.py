import re
from datasets import (
    load_dataset,
    get_dataset_config_names,
    concatenate_datasets,
    load_from_disk,
)
import os
from typing import Dict

import pandas as pd
import numpy as np
from constants import (
    LANGS,
    MULTI_BLIMP_LANGS,
    MULTI_BLIMP_PATH,
    WLT_DATA_PATH,
    WLT_HF_NAME,
    LANG_NAME,
    ISO2_to_ISO3,
    ZHO_BLIMP_EX,
)

def prepare_multiblimp():
    # for multiblimp examples not drawn from zhoblimp, jblimp
    # flatten because dataset has nested structure causing trouble with dataloader
    def flatten_ex(ex):
        return {
            "sen": ex["sen"],
            "wrong_sen": ex["wrong_sen"],
        }

    for lang in MULTI_BLIMP_LANGS:
        lang3 = ISO2_to_ISO3[lang]
        lang_specific_path = f"{MULTI_BLIMP_PATH}_{lang}"
        if os.path.exists(lang_specific_path):
            print(
                f"MultiBLiMP dataset for {lang} already exists at {lang_specific_path}"
            )
            continue
        # there is no test split for MultiBLiMP
        dataset = load_dataset("jumelet/multiblimp", lang3, split="train")
        remove_columns = [c for c in dataset.column_names if c != "phenomenon"]
        dataset = dataset.map(flatten_ex, remove_columns=remove_columns)
        os.makedirs(os.path.dirname(lang_specific_path), exist_ok=True)
        dataset.save_to_disk(lang_specific_path)
        print(f"Saved {len(dataset)} MultiBLiMP examples to {lang_specific_path}")


def prepare_zhoblimp(seed=42):
    zho_blimp_path = f"{MULTI_BLIMP_PATH}_zh"
    if os.path.exists(zho_blimp_path):
        print(f"ZhoBLIMP dataset already exists at {zho_blimp_path}")
        return

    def flatten_ex(ex):
        return {
            "sen": ex["sentence_good"],
            "wrong_sen": ex["sentence_bad"],
            "phenomenon": ex["blimp_config"],
        }

    dataset_name = "Junrui1202/zhoblimp"
    configs = get_dataset_config_names(dataset_name)

    all_datasets = []
    for cfg in configs:
        ds = load_dataset(dataset_name, cfg, split="train")
        n = len(ds)
        # deterministic index selection
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)[:ZHO_BLIMP_EX].tolist()

        ds = ds.select(idx)
        ds = ds.add_column("blimp_config", [cfg] * len(ds))

        ds = ds.map(flatten_ex, remove_columns=ds.column_names)
        print(f"Loaded {len(ds)} examples for ZhoBLIMP config {cfg}")
        all_datasets.append(ds)

    merged = concatenate_datasets(all_datasets)
    os.makedirs(os.path.dirname(zho_blimp_path), exist_ok=True)
    merged.save_to_disk(zho_blimp_path)

    print(f"Saved {len(merged)} ZhoBLIMP examples to {zho_blimp_path}")


def prepare_jblimp():
    jblimp_path = f"{MULTI_BLIMP_PATH}_ja"
    if os.path.exists(jblimp_path):
        print(f"JBLiMP dataset already exists at {jblimp_path}")
        return

    def flatten_ex(ex):
        return {
            "sen": ex["good_sentence"],
            "wrong_sen": ex["bad_sentence"],
        }

    dataset_name = "polm-stability/jblimp"
    ds = load_dataset(dataset_name, split="train")
    remove_columns = [c for c in ds.column_names if c != "phenomenon"]
    ds = ds.map(flatten_ex, remove_columns=remove_columns)
    os.makedirs(os.path.dirname(jblimp_path), exist_ok=True)
    ds.save_to_disk(jblimp_path)
    print(f"Saved {len(ds)} JBLiMP examples to {jblimp_path}")


def download_word_level_translations(wlt_data_path):
    # converts the HF release into the csv layout used below
    os.makedirs(wlt_data_path, exist_ok=True)
    for split in ["query_set", "example_pool"]:
        path = f"{wlt_data_path}/{split}.csv"
        if os.path.exists(path):
            continue
        df = load_dataset(WLT_HF_NAME, split, split="data").to_pandas()
        df["pos_set"] = df["pos_set"].apply(lambda s: str(set(s)))
        df.to_csv(path, index=False)
        print("Saved", path)

    alternatives = None
    for src in LANGS:
        path = f"{wlt_data_path}/{src}_to_valid.csv"
        if os.path.exists(path):
            continue
        if alternatives is None:
            alternatives = load_dataset(
                WLT_HF_NAME, "word_alternatives", split="data"
            ).to_pandas()
        src_df = alternatives[alternatives["source_lang"] == src].copy()
        src_df["translations"] = src_df["translations"].apply(
            lambda t: "|".join(t)
        )
        df = src_df.pivot(
            index=["word_original", "prompt"],
            columns="target_lang",
            values="translations",
        ).reset_index()
        df = df[["word_original", "prompt"] + [l for l in LANGS if l != src]]
        df.to_csv(path, index=False)
        print("Saved", path)


def load_valid_outputs_for_language_pair(
    src_lang: str, tgt_lang: str, data_path: str
) -> Dict[str, str]:
    """
    Load valid output synonyms for a specific source->target language pair.

    """
    valid_path = f"{data_path}/{src_lang}_to_valid.csv"
    if not os.path.exists(valid_path):
        raise ValueError(
            f"No valid outputs file found at {valid_path} for {src_lang}->{tgt_lang}."
        )

    df = pd.read_csv(valid_path)
    valid_outputs = {}

    for _, row in df.iterrows():
        word_original = row["word_original"]
        if src_lang == tgt_lang:
            val = row["prompt"]
        else:
            val = row[tgt_lang]
        valid_outputs[word_original] = str(val)

    return valid_outputs


def build_prompt_with_examples(examples, query_row, src_lang, tgt_lang):
    src_label = LANG_NAME[src_lang]
    tgt_label = LANG_NAME[tgt_lang]
    en_word = query_row["word_original"]

    assert en_word not in examples["word_original"].values

    parts = []
    for _, row in examples.iterrows():
        parts.append(
            f'{src_label}: "{row[src_lang]}" - {tgt_label}: "{row[tgt_lang]}"\n'
        )

    # query added last
    parts.append(f'{src_label}: "{query_row[src_lang]}" - {tgt_label}: "')
    return "".join(parts)


def prepare_word_level_translations(wlt_data_path, seed=42, force=False):
    """
    Prepare word-level translation prompts and add valid_targets for each language pair.
    Few-shot examples are aligned across language pairs at the concept level.
    """
    rng = np.random.RandomState(seed)

    example_pool_path = f"{wlt_data_path}/example_pool.csv"
    query_path = f"{wlt_data_path}/query_set.csv"

    download_word_level_translations(wlt_data_path)

    example_pool_df = pd.read_csv(example_pool_path)
    query_df = pd.read_csv(query_path)
    print(f"Loaded existing splits from {wlt_data_path}")

    num_examples = 5
    pool = example_pool_df.copy().reset_index(drop=True)

    aligned_examples = {}

    # working shuffled pool -- to ensure even coverage of selected few-shot examples
    working = pool.sample(frac=1, random_state=rng).reset_index(drop=True)
    ptr = 0
    N = len(working)

    for _, qrow in query_df.iterrows():
        en_word = qrow["en"]
        chosen = []

        while len(chosen) < num_examples:
            if ptr >= N:
                # reshuffle when exhausted
                working = pool.sample(frac=1, random_state=rng).reset_index(drop=True)
                ptr = 0

            row = working.iloc[ptr]
            ptr += 1

            if row["word_original"] == en_word:
                continue
            if row["word_original"] in [r["word_original"] for r in chosen]:
                continue
            chosen.append(row)

        aligned_examples[en_word] = pd.DataFrame(chosen).reset_index(drop=True)

    for src in LANGS:
        src_folder = f"{wlt_data_path}/{src}_to"
        if os.path.exists(src_folder):
            print(
                f"Prompts for source language {src} already exist at {src_folder}, checking individual files."
            )
        else:
            os.makedirs(src_folder, exist_ok=True)

        for tgt in LANGS:
            outpath = f"{src_folder}/{src}_{tgt}.csv"
            if os.path.exists(outpath) and not force:
                continue

            valid_outputs = load_valid_outputs_for_language_pair(
                src, tgt, wlt_data_path
            )

            rows = []
            for _, qrow in query_df.iterrows():
                en_word = qrow["word_original"]
                examples_df = aligned_examples[qrow["en"]]
                prompt = build_prompt_with_examples(
                    examples_df,
                    qrow,
                    src_lang=src,
                    tgt_lang=tgt,
                )
                if en_word in valid_outputs:
                    valid_targets_str = valid_outputs[en_word]
                else:
                    valid_targets_str = qrow[tgt]

                rows.append(
                    {
                        "en_word": en_word,
                        "src_word": qrow[src],
                        "tgt_word": qrow[tgt],
                        "prompt": prompt,
                        "valid_targets": valid_targets_str,
                    }
                )
            pd.DataFrame(rows).to_csv(outpath, index=False)
            print("Saved", outpath)


def convert_multiblimp_custom_to_json():
    for lang in ["zh", "ja"]:
        out_path = re.sub("_arrow", "", MULTI_BLIMP_PATH)
        json_path = f"{out_path}_{lang}.jsonl"
        if os.path.exists(json_path):
            print(f"{json_path} already exists, skipping.")
            continue
        print("Converting MultiBLiMP for", lang)
        in_path = f"{MULTI_BLIMP_PATH}_{lang}"
        dataset = load_from_disk(in_path)
        dataset.to_json(json_path)


if __name__ == "__main__":
    # WLT: download the dataset and build the prompts
    prepare_word_level_translations(wlt_data_path=WLT_DATA_PATH, force=False)

    # for backwards compatibility, we subselect this ourselves, and run a custom lm_eval task on them
    prepare_zhoblimp()

    # for easy analysis, we prepare these as well, though they need to be ported to json format for lm_eval
    prepare_jblimp()
    prepare_multiblimp()
    convert_multiblimp_custom_to_json()
