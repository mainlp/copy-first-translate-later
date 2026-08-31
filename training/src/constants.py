DEFAULT_MODEL_DIR = "configs/models/"
DEFAULT_OPT_DIR = "configs/optimizer/"
DEFAULT_DEEPSPEED_DIR = "configs/deepspeed/"
DEFAULT_OUTPUT_DIR = ""
DEFAULT_EFFECTIVE_BATCH_SIZE = 32

MAX_STEPS_FOR_TESTING = 100000
DEFAULT_EVAL_FREQUENCY = 50
DEFAULT_CHECKPOINT_FREQUENCY = 500

# mapping between langs and ids
LANG2ID = {
    "unknown": -1,  # unknown
    "eng_Latn": 0,  # English
    "cmn_Hani": 1,  # Chinese
    "deu_Latn": 2,  # German
    "spa_Latn": 3,  # Spanish
    "jpn_Jpan": 4,  # Japanese
    "fra_Latn": 5,  # French
    "ita_Latn": 6,  # Italian
    "por_Latn": 7,  # Portuguese
    "ind_Latn": 8,  # Indonesian
}

# reverse mapping
ID2LANG = {v: k for k, v in LANG2ID.items()}

# Sample probabilities
LANG_WEIGHTS = {
    "eng_Latn": 0.5,
    "cmn_Hani": 0.0625,
    "deu_Latn": 0.0625,
    "spa_Latn": 0.0625,
    "jpn_Jpan": 0.0625,
    "fra_Latn": 0.0625,
    "ita_Latn": 0.0625,
    "por_Latn": 0.0625,
    "ind_Latn": 0.0625,
}

DATASET_BASES = ["/dataset"]
