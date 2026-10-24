import os

LANGS = [
    "de",  # German
    "en",  # English
    "es",  # Spanish
    "fr",  # French
    "id",  # Indonesian
    "it",  # Italian
    "ja",  # Japanese
    "pt",  # Portuguese
    "zh",  # Chinese
]

MULTI_BLIMP_LANGS = ["de", "en", "es", "fr", "it", "pt"]

ISO2_to_ISO3 = {
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "it": "ita",
    "id": "ind",
    "ja": "jpn",
    "pt": "por",
    "zh": "cmn",
}

LANG_NAME = {
    "de": "Deutsch",
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "id": "Bahasa Indonesia",
    "it": "Italiano",
    "ja": "日本語",
    "pt": "Português",
    "zh": "中文",
}
LANG_NAME_EN = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "pt": "Portuguese",
    "zh": "Chinese",
}

MULTI_BLIMP_PATH = "data/multiblimp_arrow"
WLT_DATA_PATH = "data/word_level_translations"
WLT_HF_NAME = "mainlp/word-level-translations"

ZHO_BLIMP_EX = 30

WLT_PREFIX = "wlt/"
TRANSLATION_ACC_KEY = "acc"
TRANSLATION_PROB_KEY = "prob"
COPY_ACC_KEY = "copy_acc"
COPY_PROB_KEY = "copy_prob"
TRANSLATION_PROB_NORMALIZED_KEY = "prob_normalized"
COPY_PROB_NORMALIZED_KEY = "copy_prob_normalized"

DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "")
CACHE_DIR = "cache"

multi_blimp_metrics = ["acc_norm,none"]
# which metrics to log to wandb, all metrics are saved to file
HARNESS_METRICS = {
    "multiblimp_eng": multi_blimp_metrics,
    "multiblimp_deu": multi_blimp_metrics,
    "multiblimp_fra": multi_blimp_metrics,
    "multiblimp_ita": multi_blimp_metrics,
    "multiblimp_por": multi_blimp_metrics,
    "multiblimp_spa": multi_blimp_metrics,
    "jblimp": multi_blimp_metrics,
    "zhoblimp_subset": multi_blimp_metrics,
}

BENCHMARKS = (
    [f"logit_lens_{lang}" for lang in LANGS]
    + ["word_level_translation"]
    + list(HARNESS_METRICS.keys())
)

MAX_HARNESS_SAMPLES = -1
TOKENS_KEY = "tokens_seen"

CUSTOM_TASK_PATH = "custom_lm_eval_tasks"

TOKENIZER_NAME = "mistralai/Mistral-Nemo-Base-2407"
# our released checkpoints, one branch per step
RELEASED_MODEL_NAME = "mainlp/copy-first-translate-later"
# default --run_name of the scripts
DEFAULT_RUN_NAME = "ours"

FEW_SHOT_DELIMITER = '"\n'
