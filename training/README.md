# Training

Trained checkpoints: https://huggingface.co/mainlp/copy-first-translate-later

`src/train.py` pretrains a ~1.7B-parameter LLaMA-style model on nine languages with AdamW (Configs: `configs/models/llama1b.json` for model, `configs/optimizer/adamw.json` for optimizer). The data is streamed from [FineWeb-HQ](https://huggingface.co/datasets/epfml/FineWeb-HQ) (english) and [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) (other languages) parquet shards, saved in `<dataset_base>/<lang_code>/*.parquet` (several dataset_bases can be specified in `constants.py`). 

The training can be run with: `accelerate launch src/train.py`. With the flag `--use_wandb`, training is logged via wandb, in this case the project has to be specified via an environment variable `WANDB_PROJECT` (default: `training`).

For requirements, see `requirements.txt`.
