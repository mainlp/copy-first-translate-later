import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, IterableDataset
from transformers import DataCollatorWithFlattening
from constants import LANG2ID
import os
import random
import pathlib


class MultiLangStreamer:
    def __init__(
        self,
        dataset_roots,
        lang_weights,
        state,
        accelerator,
        rng,
        file_rng,
        conv_paths=False,
    ):
        self.dataset_roots = dataset_roots
        self.lang_weights = lang_weights
        self.state = state
        self.accelerator = accelerator
        self.rng = rng
        self.file_rng = file_rng

        self.rank = accelerator.process_index
        self.world_size = accelerator.num_processes
        self.conv_paths = conv_paths

    def __call__(self):
        yield from self._make_stream_all_languages()

    def _convert_path(self, fp):
        # Hook for optional path remapping in custom environments.
        return fp

    def _make_stream_all_languages(self):
        """Streams from multiple languages with weighted random sampling."""
        lang_files = {}

        for lang, lang_dir in self.dataset_roots.items():
            if self.state["lang_progress"][lang]["file_order_by_rank"] != 0:
                files = self.state["lang_progress"][lang]["file_order_by_rank"][
                    str(self.rank)
                ]
            else:
                files = sorted(
                    [str(p) for p in pathlib.Path(lang_dir).rglob("*.parquet")]
                )
                files = files[self.rank :: self.world_size]  # Shard files per rank
                self.file_rng.shuffle(files)  # Shuffle files per rank
                self.state["lang_progress"][lang]["file_order_by_rank"] = {}
                self.state["lang_progress"][lang]["file_order_by_rank"][
                    str(self.rank)
                ] = files
            if not files:
                self.accelerator.print(f"[WARN] No parquet files for {lang}")
                continue
            lang_files[lang] = files

        langs = list(lang_files.keys())
        self.accelerator.print(f"[data] Streaming from languages: {langs}")

        # Normalize sampling probabilities
        if self.lang_weights is None:
            probs = [1.0 / len(langs)] * len(langs)
        else:
            weights = [self.lang_weights.get(l, 0.0) for l in langs]
            total = sum(weights)
            probs = [w / total if total > 0 else 0.0 for w in weights]
        self.accelerator.print(
            f"[data] Sampling probabilities: {dict(zip(langs, probs))}"
        )

        def lang_stream(lang, files):
            for fi, f in enumerate(
                files[self.state["lang_progress"][lang]["file_idx"] :],
                start=self.state["lang_progress"][lang]["file_idx"],
            ):
                # keep full path in streaming state, expose a path relative to the language root
                if self.conv_paths:
                    f = self._convert_path(f)
                file_path = f
                try:
                    rel_path = os.path.relpath(f, start=self.dataset_roots[lang])
                except Exception:
                    rel_path = os.path.basename(f)
                file_name = rel_path

                print(
                    f"[data] Rank {self.rank} streaming {lang} from file {file_name} (full: {file_path})"
                )

                ds = load_dataset(
                    "parquet", data_files=f, split="train", streaming=True
                )
                for ei, ex in enumerate(ds):
                    if (
                        fi == self.state["lang_progress"][lang]["file_idx"]
                        and ei < self.state["lang_progress"][lang]["example_idx"]
                    ):
                        continue
                    self.state["lang_progress"][lang]["file_idx"] = fi
                    self.state["lang_progress"][lang]["file_name"] = file_name
                    self.state["lang_progress"][lang]["file_path"] = file_path
                    self.state["lang_progress"][lang]["example_idx"] = ei + 1
                    self.state["lang_progress"][lang]["total"] += 1
                    ex["language_id"] = LANG2ID[lang]
                    yield ex
                self.accelerator.print(
                    f"[data] File {file_name} for {lang} exhausted after {self.state['lang_progress'][lang]['example_idx']} examples."
                )

                self.state["lang_progress"][lang]["file_idx"] = fi + 1
                self.state["lang_progress"][lang]["seen_paths"].append(file_path)
                self.state["lang_progress"][lang]["example_idx"] = 0

        # Create iterators per language
        lang_iters = {
            lang: lang_stream(lang, files) for lang, files in lang_files.items()
        }

        # Weighted random sampling over active languages
        while langs:
            lang = self.rng.choices(langs, weights=probs, k=1)[0]
            try:
                yield next(lang_iters[lang])
            except StopIteration:
                # Language exhausted
                self.accelerator.print(f"[data] {lang} exhausted, restarting stream.")
                self.accelerator.print(
                    f"[data] New sampling probabilities: {dict(zip(langs, probs))}"
                )
                lang_files_new = lang_files[lang].copy()
                self.file_rng.shuffle(lang_files_new)  # Shuffle files per rank
                lang_files[lang].extend(lang_files_new)
                self.state["lang_progress"][lang]["file_order_by_rank"][
                    str(self.rank)
                ].extend(lang_files_new)
                self.state["lang_progress"][lang]["example_idx"] = 0
                lang_iters[lang] = lang_stream(lang, lang_files[lang])


def unpack_packed_batch(batch):
    """
    Split a packed batch back into individual examples by detecting
    resets in position_ids (== 0). Works for both 1D and 2D tensors.
    """
    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]
    lang_ids = batch["language_ids"]

    # Flatten if 2D
    if input_ids.dim() > 1:
        input_ids = input_ids.view(-1)
        position_ids = position_ids.view(-1)

    # indices where new sequences start
    start_idxs = (position_ids == 0).nonzero(as_tuple=True)[0]

    examples = []
    for i, start in enumerate(start_idxs):
        end = start_idxs[i + 1] if i + 1 < len(start_idxs) else len(input_ids)
        ex = input_ids[start:end]
        examples.append((ex, int(lang_ids[i])))

    return examples


def get_collate_fn(tokenizer, style="default"):
    if style == "default":

        def collate(batch):
            input_ids = [
                torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch
            ]
            attention_mask = [
                torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch
            ]
            language_ids = torch.tensor(
                [ex.get("language_id", -1) for ex in batch], dtype=torch.long
            )
            # dynamic padding
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            )
            labels = input_ids.masked_fill(attention_mask == 0, -100)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "language_ids": language_ids,
            }

    elif style == "packing":
        data_collator = DataCollatorWithFlattening(
            return_tensors="pt",
            separator_id=tokenizer.pad_token_id,
            return_flash_attn_kwargs=True,
        )

        def collate(batch):
            packed = data_collator(batch)
            # attach languages
            packed["language_ids"] = torch.tensor(
                [ex.get("language_id", -1) for ex in batch], dtype=torch.long
            )
            packed = {
                "input_ids": packed["input_ids"],
                "labels": packed["labels"],
                "position_ids": packed["position_ids"],
                "cu_seq_lens_q": packed["cu_seq_lens_q"],
                "cu_seq_lens_k": packed["cu_seq_lens_k"],
                "language_ids": packed["language_ids"],
            }

            return packed

    else:
        raise ValueError(f"Unknown collate style: {style}")

    return collate


def get_dataloader(seed, model_cfg, accelerator, tok, ds, bs_per_gpu, verbose=False):
    collate_fn = get_collate_fn(tok, style=model_cfg.get("collate_style", "default"))

    g = torch.Generator()
    g.manual_seed(seed + accelerator.process_index)  # seed unique per GPU process
    dl = DataLoader(
        ds,
        batch_size=bs_per_gpu,  # split per GPU is done by accelerator
        collate_fn=collate_fn,
        generator=g,
        shuffle=True,
    )
    if verbose:
        torch.cuda.synchronize()
        accelerator.print(
            f"First 20 input_ids: {ds[0]['input_ids'][:100]}"
        )  # check decoding
        accelerator.print(f"Decoded: {tok.decode(ds[0]['input_ids'])}")
        accelerator.print(f"Dataset columns: {ds.column_names}")
    return dl


def get_wikitext(verbose, accelerator, tokenize_function):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # remove empty lines
    ds = ds.filter(lambda ex: len(ex["text"].strip()) > 0)

    if verbose:
        sample = ds[0].copy()
        accelerator.print(
            f"Example raw sample:\n{sample['text'][:100]}"
        )  # inspect raw examples

    ds = ds.map(lambda ex: {"language_id": [0] * len(ex["text"])}, batched=True)
    ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])
    return ds


def build_dataset_roots_from_bases(data_dir_root, dataset_bases, accelerator):
    """
    Automatically discovers which base dataset folder contains which language.
    For example:
        /dataset/eng_Latn/
        /dataset1/cmn_Hani/
    """
    dataset_roots = {}
    for base in dataset_bases:
        if data_dir_root == "":
            full_base = base
        else:
            full_base = os.path.join(data_dir_root, base.lstrip("/"))
        if not os.path.isdir(full_base):
            accelerator.print(f"[data] Base folder missing: {full_base}")
            continue
        accelerator.print(f"[data] Scanning base folder: {full_base}")
        for lang in LANG2ID.keys():
            if lang == "unknown":
                continue

            lang_dir = os.path.join(full_base, lang)
            if os.path.isdir(lang_dir) and any(
                pathlib.Path(lang_dir).rglob("*.parquet")
            ):

                dataset_roots[lang] = lang_dir

    # Log discovery summary
    missing = [l for l in LANG2ID.keys() if l not in dataset_roots and l != "unknown"]
    accelerator.print(
        f"[data] Discovered {len(dataset_roots)} languages: {list(dataset_roots.keys())}"
    )
    if missing:
        accelerator.print(f"[data] Missing or empty datasets for: {missing}")
    return dataset_roots


def get_multilingual_dataloader(
    accelerator,
    tok,
    tokenize_function,
    model_cfg,
    bs_per_gpu,
    dataset_roots: dict[str, str],
    lang_weights: dict[str, float] | None = None,
    seed: int = 42,
    verbose: bool = False,
    streaming: bool = True,
    resume_state: dict = None,
    conv_paths: bool = False,
):
    """
    Returns a distributed, tokenized multilingual DataLoader for training.
    Uses the provided Accelerator instance for rank/world information.
    """
    rank = accelerator.process_index
    state = resume_state or {
        "lang_progress": {},
    }

    for lang in dataset_roots.keys():
        if lang not in state["lang_progress"]:
            state["lang_progress"][lang] = {
                "file_idx": 0,
                "example_idx": -1,
                "total": 0,
                "file_name": "",
                "file_path": "",
                "seen_paths": [],
                "file_order_by_rank": 0,
            }
    state["langs"] = list(dataset_roots.keys())

    rng = random.Random(seed + rank)
    file_rng = random.Random(seed + 2 * rank)

    # Create the stateful streamer over multilingual parquet files.
    streamer = MultiLangStreamer(
        dataset_roots=dataset_roots,
        lang_weights=lang_weights,
        state=state,
        accelerator=accelerator,
        rng=rng,
        file_rng=file_rng,
        conv_paths=conv_paths,
    )

    # Wrap streamer in an IterableDataset for efficient online tokenization.
    raw_dataset = IterableDataset.from_generator(streamer)

    dataset = raw_dataset.map(tokenize_function, batched=False)

    # Optional sample inspection for debugging data/packing behavior.
    if verbose and rank == 0:
        first = next(iter(dataset))
        accelerator.print(f"Example raw (rank 0): {first}")
        accelerator.print(f"Decoded: {tok.decode(first['input_ids'])}")
        accelerator.print(f"Language tag: {first.get('language_id', 'N/A')}")

    # Set up per-rank RNG and collator.
    g = torch.Generator()
    g.manual_seed(seed + rank)
    collate_fn = get_collate_fn(tok, style=model_cfg.get("collate_style", "default"))

    dataloader = DataLoader(
        dataset,
        batch_size=bs_per_gpu,
        collate_fn=collate_fn,
        generator=g,
        shuffle=not streaming,
        num_workers=0,
        pin_memory=True,
        persistent_workers=True if not streaming else False,
    )

    return dataloader, streamer
