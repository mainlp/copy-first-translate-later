# Training

Pretraining code for **"Copy First, Translate Later: Interpreting Translation Dynamics in Multilingual Pretraining"** (Körner*, Matveev*, Eichin*, Kutyniok†, Plank†, Hedderich†; EMNLP 2026). Preprint: https://arxiv.org/abs/2604.17633

Trained checkpoints: https://huggingface.co/mainlp/copy-first-translate-later

`src/train.py` pretrains a ~1.7B-parameter LLaMA-style model on nine languages with AdamW (Configs: `configs/models/llama1b.json` for model, `configs/optimizer/adamw.json` for optimizer). The data is streamed from [FineWeb-HQ](https://huggingface.co/datasets/epfml/FineWeb-HQ) (english) and [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) (other languages) parquet shards, saved in `<dataset_base>/<lang_code>/*.parquet` (several dataset_bases can be specified in `constants.py`). 

The training can be run with: `accelerate launch src/train.py`. With the flag `--use_wandb`, training is logged via wandb, in this case the project has to be specified via an environment variable `WANDB_PROJECT` (default: `training`).

For requirements, see `requirements.txt`.

## Citation

If you use this code, please consider citing:
```bibtex
@misc{korner2026copyfirst,
  title         = {Copy First, Translate Later: Interpreting Translation Dynamics in Multilingual Pretraining},
  author        = {K{\"o}rner, Felicia and Matveev, Maria and Eichin, Florian and Kutyniok, Gitta and Plank, Barbara and Hedderich, Michael A.},
  year          = {2026},
  eprint        = {2604.17633},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note          = {To appear at EMNLP 2026},
  url           = {https://arxiv.org/abs/2604.17633}
}
```
