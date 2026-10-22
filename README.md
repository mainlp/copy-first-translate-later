This repository contains the training and evaluation code for [Copy First, Translate Later: Interpreting Translation Dynamics in Multilingual Pretraining](https://arxiv.org/abs/2604.17633),
to appear at EMNLP 2026, Main Conference.

The word-level translation dataset, covering 125 concepts with translations for 72 language pairs (English, German, French, Spanish, Italian, Portuguese, Indonesian, Japanese, and Chinese) can be found at [https://huggingface.co/datasets/mainlp/word-level-translations](https://huggingface.co/datasets/mainlp/word-level-translations).

The model checkpoints studied in the paper (checkpoint every 500 steps/~185M tokens, from 0-37B tokens processed), and a log of the training data seen at each checkpoint can be found at [https://huggingface.co/mainlp/copy-first-translate-later](https://huggingface.co/mainlp/copy-first-translate-later).

The code for the ExPLAIND loss decomposition used in this project can be found at [https://github.com/mainlp/explaind/tree/main/scaled_experiments/llama1b](https://github.com/mainlp/explaind/tree/main/scaled_experiments/llama1b).

If you use this code, the dataset, or the checkpoints, please cite (to be replaced by the ACL Anthology .bib once EMNLP Proceedings are out):

```
@misc{koerner2026copyfirsttranslatelater,
      title={Copy First, Translate Later: Interpreting Translation Dynamics in Multilingual Pretraining}, 
      author={Felicia Körner and Maria Matveev and Florian Eichin and Gitta Kutyniok and Barbara Plank and Michael A. Hedderich},
      year={2026},
      eprint={2604.17633},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.17633}, 
}
```
