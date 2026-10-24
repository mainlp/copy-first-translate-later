# Copy First, Translate Later

This repository contains the training and evaluation code for the paper **"Copy First, Translate Later: Interpreting Translation Dynamics in Multilingual Pretraining"** (Körner*, Matveev*, Eichin*, Kutyniok†, Plank†, Hedderich†; EMNLP 2026, main conference). 

Preprint: https://arxiv.org/abs/2604.17633 

> Large language models exhibit impressive cross-lingual capabilities. However, how cross-lingual generalization emerges has only been analyzed through isolated factors and at sparse points during training, limiting our understanding --- particularly of the early phases of learning. To study the early trajectory of linguistic and translation capabilities, we pretrain a multilingual 1.7B model on nine diverse languages, capturing checkpoints at a much finer granularity. We use word-level translation as a testbed, introducing a novel dataset to trace how translation develops over training through behavioral analyses, model-component analysis, and parameter-based ablations. We find that the model quickly acquires basic linguistic capabilities in parallel with token-level copying, while translation develops in two distinct phases: an initial phase dominated by copying and surface-level similarities, and a second phase in which generalizing translation mechanisms develop and copying is refined. Together, these findings provide a fine-grained view of how cross-lingual generalization develops during multilingual pretraining.

The word-level translation dataset, covering 125 concepts with translations for 72 language pairs (English, German, French, Spanish, Italian, Portuguese, Indonesian, Japanese, and Chinese) can be found at [https://huggingface.co/datasets/mainlp/word-level-translations](https://huggingface.co/datasets/mainlp/word-level-translations).

The model checkpoints studied in the paper (checkpoint every 500 steps/~185M tokens, from 0-37B tokens processed), and a log of the training data seen at each checkpoint can be found at [https://huggingface.co/mainlp/copy-first-translate-later](https://huggingface.co/mainlp/copy-first-translate-later).


## Repository layout

* [`training/`](training/README.md): Pretraining code. 
*  [`eval/`](eval/README.md): Evaluation -- behavioral analysis and logit lens. 
* [ExPLAIND](https://github.com/mainlp/explaind/tree/main/scaled_experiments/llama1b): ExPLAIND loss decomposition.

## License

The code is released under the [MIT License](LICENSE). The model checkpoints and the dataset on Hugging Face have their own licenses, see their pages.

## Citation

If you use this code, the dataset, or the checkpoints, please cite (to be replaced by the ACL Anthology .bib once EMNLP Proceedings are out):

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
