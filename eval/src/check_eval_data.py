from collections import Counter
from pathlib import Path
import pandas as pd

from constants import LANGS, WLT_DATA_PATH


def _extract_surface_tokens(prompt):
    """
    Extract the source-side surface tokens for the few-shot lines.
    """
    lines = prompt.strip().splitlines()
    example_lines = lines[:-1]

    toks = []
    for line in example_lines:
        left = line.split(" - ")[0]
        tok = left.split('"')[1]
        toks.append(tok)
    return tuple(toks)


data_path = Path(WLT_DATA_PATH)


example_pool = pd.read_csv(data_path / "example_pool.csv")

lookup = {}
for lang in LANGS:
    if lang in example_pool.columns:
        lookup[lang] = dict(zip(example_pool[lang], example_pool["word_original"]))

prompts_by_pair = {}
for src in LANGS:
    for tgt in LANGS:
        path = data_path / f"{src}_to" / f"{src}_{tgt}.csv"
        assert path.exists(), f"Missing {path}"
        prompts_by_pair[(src, tgt)] = pd.read_csv(path)

ref_df = prompts_by_pair[(LANGS[0], LANGS[0])]
n_rows = len(ref_df)

example_counts = Counter()
per_query_examples = {}
for row_idx in range(n_rows):
    aligned = None
    query_concept = None
    for (src, tgt), df in prompts_by_pair.items():
        row = df.iloc[row_idx]
        prompt = row["prompt"]
        if query_concept is None:
            query_concept = row["en_word"]
        else:
            assert query_concept == row["en_word"]
        surface_tokens = _extract_surface_tokens(prompt)
        concept_tokens = tuple(lookup[src][tok] for tok in surface_tokens)

        if aligned is None:
            aligned = concept_tokens
        else:
            assert concept_tokens == aligned, (
                f"Few-shot mismatch at row {row_idx} for {src}->{tgt}. "
                f"Expected {aligned}, got {concept_tokens}"
            )
    per_query_examples[query_concept] = aligned
    example_counts.update(aligned)

print("\n=== Per-query aligned examples ===")
for concept, exs in per_query_examples.items():
    print(f"{concept}: {list(exs)}")

print("\n=== Example frequency counts ===")
for concept, cnt in sorted(example_counts.items(), key=lambda x: (-x[1], x[0])):
    print(f"{concept}: {cnt}")

total_uses = sum(example_counts.values())
unique_examples = len(example_counts)
print("\n=== Summary ===")
print(f"Total example uses: {total_uses}")
print(f"Unique examples used: {unique_examples}")
print(f"Avg uses per example: {total_uses / max(unique_examples,1):.2f}")