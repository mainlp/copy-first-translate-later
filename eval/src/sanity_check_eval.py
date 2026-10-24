import os
import torch
import time
import numpy as np
import pandas as pd

from expl_training_utils import ExplTrainingCheckpointStrategy
from evaluate import (
    eval_translation_metrics,
    get_greedy_accuracy_batched,
    get_greedy_accuracy,
)
from logit_lens import (
    eval_logit_lens,
    run_logit_lens_last_token,
    per_language_per_layer_scores,
    per_language_per_layer_scores_batched,
)
from constants import (
    DEFAULT_RUN_NAME,
    LANGS,
    WLT_DATA_PATH,
    WLT_PREFIX,
    DEFAULT_OUTPUT_DIR,
    FEW_SHOT_DELIMITER,
)
from utils import get_checkpoint_results_dirs


# avoid noisy tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEBUG = True


def prep_test(mod, tok):
    tok.pad_token = tok.eos_token
    mod.config.pad_token_id = mod.config.eos_token_id
    mod.eval()
    mod.to("cuda" if torch.cuda.is_available() else "cpu")
    return mod, tok


def test_logit_lens_correctness(mod, tok):
    prompt = 'Deutsch: "Brücke" - Français: "pont"\nDeutsch: "Hand" - Français: "main"\nDeutsch: "Talent" - Français: "talent"\nDeutsch: "Feld" - Français: "champ"\nDeutsch: "Kopf" - Français: "tête"\nDeutsch: "tanzen" - Français: "'
    device = getattr(mod, "device", None)
    cur_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    mod.eval()
    lens_logits_exact = run_logit_lens_last_token(mod, cur_ids)[-1]
    true_logits = mod(cur_ids).logits[:, -1, :]
    diff = (true_logits - lens_logits_exact).abs().max().item()
    assert (
        diff < 1e-4
    ), f"Logit Lens exact computation differs from true logits by {diff}"
    print("Logit Lens exact computation matches true logits.")


def run_logit_lens(mod, tok, data_path, eval_fn, lang="en", tol=0.0001):
    print(f"Running logit lens eval ({eval_fn.__name__}) for {data_path}...")
    s = time.time()
    batched = eval_fn(
        mod,
        tok,
        per_lang_fn=per_language_per_layer_scores_batched,
        data_path=data_path,
        output_dir="results",
        lang=lang,
        force=True,
        test=True,
        debug=DEBUG,
    )
    e = time.time()
    print(f"Total eval time, batched: {e-s:.1f} seconds")
    s = time.time()
    unbatched = eval_fn(
        mod,
        tok,
        per_lang_fn=per_language_per_layer_scores,
        data_path=data_path,
        output_dir="results",
        lang=lang,
        force=True,
        test=True,
        debug=DEBUG,
    )
    e = time.time()
    print(f"Total eval time, individual: {e-s:.1f} seconds")

    # Compare results
    tgt_lang = lang
    for src_lang in LANGS:
        src_key = f"{src_lang}_to"
        if tgt_lang in unbatched.get(src_key, {}) and tgt_lang in batched.get(
            src_key, {}
        ):
            for a, b in zip(unbatched[src_key][tgt_lang], batched[src_key][tgt_lang]):
                for la in LANGS:
                    # scores_first_token, scores, scores_normed are dicts {word: [layer_scores]}
                    a_first = a["scores_first_token"][la]
                    b_first = b["scores_first_token"][la]
                    a_scores = a["scores"][la]
                    b_scores = b["scores"][la]
                    a_normed = a["scores_normed"][la]
                    b_normed = b["scores_normed"][la]

                    assert set(a_first.keys()) == set(
                        b_first.keys()
                    ), f"Scores_first_token keys mismatch for {la}"
                    assert set(a_scores.keys()) == set(
                        b_scores.keys()
                    ), f"Scores keys mismatch for {la}"
                    assert set(a_normed.keys()) == set(
                        b_normed.keys()
                    ), f"Scores_normed keys mismatch for {la}"

                    for word in a_scores.keys():
                        assert np.allclose(
                            a_first[word], b_first[word], atol=tol
                        ), f"Scores_first_token mismatch for {la}: {word}, diff={np.abs(np.array(a_first[word]) - np.array(b_first[word])).max()}"
                        assert np.allclose(
                            a_scores[word], b_scores[word], atol=tol
                        ), f"Scores mismatch for {la}: {word}, diff={np.abs(np.array(a_scores[word]) - np.array(b_scores[word])).max()}"
                        assert np.allclose(
                            a_normed[word], b_normed[word], atol=tol
                        ), f"Scores_normed mismatch for {la} :{word}, diff={np.abs(np.array(a_normed[word]) - np.array(b_normed[word])).max()}"
        else:
            print(
                f"No data for {src_lang} -> {tgt_lang} in one of the implementations, skipping comparison for this pair."
            )
            if tgt_lang in unbatched.get(src_key, {}):
                print(f"Unbatched has data for {src_lang} -> {tgt_lang}")
            else:
                print(f"Unbatched does NOT have data for {src_lang} -> {tgt_lang}")
                print(f"Unbatched keys: {unbatched.keys()}")
                print(f"Unbatched keys {src_key}: {unbatched.get(src_key, {}).keys()}")

            if tgt_lang in batched.get(src_key, {}):
                print(f"Batched has data for {src_lang} -> {tgt_lang}")
            else:
                print(f"Batched does NOT have data for {src_lang} -> {tgt_lang}")
                print(f"Batched keys: {batched.keys()}")
                print(f"Batched keys {src_key}: {batched.get(src_key, {}).keys()}")

    print(
        f"Logit Lens per-layer scores for {data_path} match between individual and batched implementations."
    )


def run_translation_metrics(mod, tok, data_path=WLT_DATA_PATH, key_prefix=WLT_PREFIX):
    print("Running translation metrics evaluation...")
    s = time.time()
    results = eval_translation_metrics(
        mod,
        tok,
        output_dir="results",
        data_path=data_path,
        key_prefix=key_prefix,
        force=True,
        debug=DEBUG,
    )
    e = time.time()
    print(results)
    print(f"Total eval time, batched: {e-s:.1f} seconds")


def load_from_local(
    step,
    epoch=0,
    model_type="llama1b.json",
    run_name=DEFAULT_RUN_NAME,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    run_dir = os.path.join(output_dir, run_name)
    checkpoints_dir, _ = get_checkpoint_results_dirs(run_dir)
    strategy = ExplTrainingCheckpointStrategy(
        model_config_type=model_type,
        checkpoints_dir=checkpoints_dir,
    )
    path = checkpoints_dir + f"/checkpoint-{epoch}-{step}"
    print(f"Loading model from {path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ckpt = strategy.load_model_for_checkpoint(
        {"epoch": epoch, "step": step, "path": path},
        device=device,
    )
    tokenizer, _ = strategy.build_initial()
    return model_ckpt, tokenizer


def test_greedy_accuracy_batched_vs_unbatched(mod, tok, tol=0.0001, debug=True):
    """Compare batched and unbatched accuracy computations."""
    print("Testing batched vs unbatched greedy accuracy...")
    device = mod.device

    # Test a few language pairs
    test_pairs = [("de", "de"), ("de", "en"), ("en", "de")]
    delim_len = len(tok(FEW_SHOT_DELIMITER, add_special_tokens=False)["input_ids"])
    for src, tgt in test_pairs:
        print(f"Testing {src} -> {tgt}")

        folder_path = os.path.join(WLT_DATA_PATH, f"{src}_to")
        file_path = os.path.join(folder_path, f"{src}_{tgt}.csv")
        df = pd.read_csv(file_path)
        if debug:
            df = df.head(10)

        s = time.time()
        # Run both versions
        acc_batched, prob_batched, prob_normalized_batched, preds_batched = (
            get_greedy_accuracy_batched(
                src,
                tgt,
                mod,
                tok,
                delim_len=delim_len,
                batch_size=10,
                data_path=WLT_DATA_PATH,
                device=device,
                debug=debug,
            )
        )
        e = time.time()
        print(f"Batched computation time: {e-s:.2f} seconds")

        s = time.time()
        acc_unbatched, prob_unbatched, prob_normalized_unbatched, preds_unbatched = (
            get_greedy_accuracy(
                src,
                tgt,
                mod,
                tok,
                delim_len=delim_len,
                batch_size=1,
                data_path=WLT_DATA_PATH,
                device=device,
                debug=debug,
            )
        )
        e = time.time()
        print(f"Unbatched computation time: {e-s:.2f} seconds")
        # Compare results
        print(
            f"Accuracy:  batched={acc_batched:.4f}, unbatched={acc_unbatched:.4f}, "
            f"diff={abs(acc_batched - acc_unbatched):.6f}"
        )
        print(
            f"Prob avg:  batched={prob_batched:.6f}, unbatched={prob_unbatched:.6f}, "
            f"diff={abs(prob_batched - prob_unbatched):.8f}"
        )
        print(
            f"Prob norm avg:  batched={prob_normalized_batched:.6f}, unbatched={prob_normalized_unbatched:.6f}, "
            f"diff={abs(prob_normalized_batched - prob_normalized_unbatched):.8f}"
        )
        assert abs(acc_batched - acc_unbatched) < tol, "Accuracies don't match!"
        assert abs(prob_batched - prob_unbatched) < tol, "Probabilities don't match!"
        assert (
            abs(prob_normalized_batched - prob_normalized_unbatched) < tol
        ), "Normalized probabilities don't match!"

        # Compare predictions in detail
        mismatches = 0
        required_keys = [
            "en_word",
            "src_lang",
            "tgt_lang",
            "target_word",
            "prediction",
            "accuracy",
            "generated_word",
            "log_prob_sum",
            "generated_log_prob_sum",
            "generated_log_prob_quote",
            "log_prob_quote",
            "num_tokens",
            "generated_num_tokens",
        ]
        for i, (pred_b, pred_u, (_, row)) in enumerate(
            zip(preds_batched, preds_unbatched, df.iterrows())
        ):
            # Check all required keys exist
            for key in required_keys:
                assert key in pred_b, f"Missing key '{key}' in batched prediction {i}"
                assert key in pred_u, f"Missing key '{key}' in unbatched prediction {i}"

            # Check string fields
            for key in [
                "en_word",
                "src_lang",
                "tgt_lang",
                "target_word",
                "accuracy",
                "generated_word",
                "prediction",
            ]:
                if pred_b[key] != pred_u[key]:
                    print(
                        f"Mismatch at idx {i}, {key}: {pred_b[key]} (batched) vs {pred_u[key]} (unbatched)"
                    )
                    mismatches += 1

            # Check dict fields
            for key in [
                "log_prob_sum",
                "log_prob_quote",
                "num_tokens",
            ]:
                dict_b = pred_b[key]
                dict_u = pred_u[key]

                assert set(dict_b.keys()) == set(
                    dict_u.keys()
                ), f"Key mismatch for {key} at idx {i}"

                for trans in dict_b.keys():
                    if isinstance(dict_b[trans], (int, float)):
                        diff = abs(dict_b[trans] - dict_u[trans])
                        if diff > tol:
                            print(
                                f"Mismatch at idx {i}, {key}[{trans}]: {dict_b[trans]} (batched) vs {dict_u[trans]} (unbatched) (diff={diff})"
                            )
                            mismatches += 1

            # Check dict of arrays fields
            for key in ["log_probs"]:
                dict_b = pred_b[key]
                dict_u = pred_u[key]

                assert set(dict_b.keys()) == set(
                    dict_u.keys()
                ), f"Key mismatch for {key} at idx {i}"

                for trans in dict_b.keys():
                    if isinstance(dict_b[trans], list):
                        diff = np.abs(
                            np.array(dict_b[trans]) - np.array(dict_u[trans])
                        ).max()
                        if diff > tol:
                            print(f"Mismatch at idx {i}, {key}[{trans}]: diff={diff}")
                            mismatches += 1

            # Check numerical fields
            for key in [
                "generated_log_probs",
                "generated_log_prob_sum",
                "generated_log_prob_quote",
                "generated_num_tokens",
            ]:
                val_b = pred_b[key]
                val_u = pred_u[key]

                if isinstance(val_b, list):
                    diff = np.abs(np.array(val_b) - np.array(val_u)).max()
                    if diff > tol:
                        print(f"Mismatch at idx {i}, {key}: diff={diff}")
                        mismatches += 1
                else:
                    diff = abs(val_b - val_u)
                    if diff > tol:
                        print(f"Mismatch at idx {i}, {key}: diff={diff}")
                        mismatches += 1

            gen_word_b = pred_b["generated_word"]
            gen_word_u = pred_u["generated_word"]

            if "valid_targets" in row and pd.notna(row["valid_targets"]):
                valid_targets = [
                    t.strip() for t in str(row["valid_targets"]).split("|")
                ]
            else:
                valid_targets = [str(row["tgt_word"]).strip()]

            expected_acc_b = any(gen_word_b == vt for vt in valid_targets)
            expected_acc_u = any(gen_word_u == vt for vt in valid_targets)

            assert pred_b["accuracy"] == (1 if expected_acc_b else 0), (
                f"Batched accuracy mismatch at idx {i}: "
                f"generated='{gen_word_b}', valid_targets={valid_targets}, "
                f"expected={expected_acc_b}, got={pred_b['accuracy']}"
            )
            assert pred_u["accuracy"] == (1 if expected_acc_u else 0), (
                f"Unbatched accuracy mismatch at idx {i}: "
                f"generated='{gen_word_u}', valid_targets={valid_targets}, "
                f"expected={expected_acc_u}, got={pred_u['accuracy']}"
            )
        assert mismatches == 0, f"Predictions don't match for {src}->{tgt}"

    print("Batched vs unbatched metrics match")


def test_synonyms_batched_vs_unbatched(mod, tok, tol=0.0001, debug=True):
    """Test that batched and unbatched implementations handle synonyms correctly."""
    print("Testing batched vs unbatched with synonyms...")
    device = mod.device

    # Create test candidates with multiple synonyms per language
    prompt = 'Deutsch: "Brücke" - Français: "pont"\nDeutsch: "Hand" - Français: "main"\nDeutsch: "Talent" - Français: "talent"\nDeutsch: "Feld" - Français: "champ"\nDeutsch: "Kopf" - Français: "tête"\nDeutsch: "tanzen" - Français: "'

    candidates = {
        "en": ["dance", "danced"],  # Multiple synonyms
        "de": ["tanzen"],
        "fr": ["danser", "danse"],  # Multiple synonyms
    }

    print(f"Testing with candidates: {candidates}")

    # Test batched version
    s = time.time()
    (
        scores_word_batched,
        scores_normed_batched,
        scores_first_batched,
    ) = per_language_per_layer_scores_batched(
        mod, tok, prompt, candidates, device=device
    )
    e = time.time()
    print(f"Batched computation time: {e-s:.2f} seconds")

    # Test unbatched version
    s = time.time()
    (
        scores_word_unbatched,
        scores_normed_unbatched,
        scores_first_unbatched,
    ) = per_language_per_layer_scores(mod, tok, prompt, candidates, device=device)
    e = time.time()
    print(f"Unbatched computation time: {e-s:.2f} seconds")
    # Compare batched vs unbatched results
    all_match = True

    for lang in candidates.keys():
        if lang in scores_word_batched:
            batched_scores = scores_word_batched[lang]
            unbatched_scores = scores_word_unbatched[lang]

            # Iterate over words (now keys in the dict)
            for word in batched_scores.keys():
                b_scores = batched_scores[word]
                u_scores = unbatched_scores[word]

                # Compare first few layer scores
                for i in range(len(b_scores)):
                    diff = abs(b_scores[i] - u_scores[i])
                    if diff > tol:
                        print(
                            f"'{word}', Layer {i}: diff = {diff:.2e} (MISMATCH in scores_word)"
                        )
                        all_match = False
        if lang in scores_first_batched:
            batched_scores = scores_first_batched[lang]
            unbatched_scores = scores_first_unbatched[lang]

            # Iterate over words (now keys in the dict)
            for word in batched_scores.keys():
                b_scores = batched_scores[word]
                u_scores = unbatched_scores[word]

                # Compare layer scores
                for i in range(len(b_scores)):
                    diff = abs(b_scores[i] - u_scores[i])
                    if diff > tol:
                        print(
                            f"'{word}', Layer {i}: diff = {diff:.2e} (MISMATCH in scores_first)"
                        )
                        all_match = False

        if lang in scores_normed_batched:
            batched_scores = scores_normed_batched[lang]
            unbatched_scores = scores_normed_unbatched[lang]

            # Iterate over words (now keys in the dict)
            for word in batched_scores.keys():
                b_scores = batched_scores[word]
                u_scores = unbatched_scores[word]

                # Compare layer scores
                for i in range(len(b_scores)):
                    diff = abs(b_scores[i] - u_scores[i])
                    if diff > tol:
                        print(
                            f"'{word}', Layer {i}: diff = {diff:.2e} (MISMATCH in scores_normed)"
                        )
                        all_match = False

    assert all_match, "Some synonyms tests failed!"
    print("All synonyms tests passed!")

    # Compare multi-synonym vs single-synonym inputs
    print("Comparing multi-synonym vs single-synonym results...")

    for lang, words in candidates.items():
        for word in words:
            single_candidates = {lang: [word]}

            (
                single_scores_word_batched,
                single_scores_normed_batched,
                single_scores_first_batched,
            ) = per_language_per_layer_scores_batched(
                mod, tok, prompt, single_candidates, device=device
            )
            (
                single_scores_word_unbatched,
                single_scores_normed_unbatched,
                single_scores_firs,
            ) = per_language_per_layer_scores(
                mod, tok, prompt, single_candidates, device=device
            )

            # Compare multi vs single (batched)
            assert np.allclose(
                scores_word_batched[lang][word],
                single_scores_word_batched[lang][word],
                atol=tol,
            ), f"Multi vs single (batched) mismatch for {lang}: {word}"

            # Compare multi vs single (unbatched)
            if word in scores_word_unbatched.get(lang, {}):
                assert np.allclose(
                    scores_word_unbatched[lang][word],
                    single_scores_word_unbatched[lang][word],
                    atol=tol,
                ), f"Multi vs single (unbatched) mismatch for {lang}: {word}"

            # Compare multi vs single (batched) normalized
            if word in scores_normed_batched.get(lang, {}):
                assert np.allclose(
                    scores_normed_batched[lang][word],
                    single_scores_normed_batched[lang][word],
                    atol=tol,
                ), f"Multi vs single (batched normed) mismatch for {lang}: {word}"

            # Compare multi vs single (unbatched) normalized
            if word in scores_normed_unbatched.get(lang, {}):
                assert np.allclose(
                    scores_normed_unbatched[lang][word],
                    single_scores_normed_unbatched[lang][word],
                    atol=tol,
                ), f"Multi vs single (unbatched normed) mismatch for {lang}: {word}"

            # Compare multi vs single (batched) first
            if word in scores_first_batched.get(lang, {}):
                assert np.allclose(
                    scores_first_batched[lang][word],
                    single_scores_first_batched[lang][word],
                    atol=tol,
                ), f"Multi vs single (batched first) mismatch for {lang}: {word}"

            # Compare multi vs single (unbatched) first
            if word in scores_first_unbatched.get(lang, {}):
                assert np.allclose(
                    scores_first_unbatched[lang][word],
                    single_scores_firs[lang][word],
                    atol=tol,
                ), f"Multi vs single (unbatched first) mismatch for {lang}: {word}"

    print("Multi-synonym vs single-synonym results match.")

    return all_match


def main():
    os.makedirs("results", exist_ok=True)

    step = 100000

    mod, tok = load_from_local(step=step, epoch=0)
    mod, tok = prep_test(mod, tok)

    test_logit_lens_correctness(mod, tok)
    test_synonyms_batched_vs_unbatched(mod, tok)
    run_logit_lens(mod, tok, WLT_DATA_PATH, eval_logit_lens)
    test_greedy_accuracy_batched_vs_unbatched(mod, tok, debug=DEBUG)

    run_translation_metrics(mod, tok, data_path=WLT_DATA_PATH, key_prefix=WLT_PREFIX)


if __name__ == "__main__":
    main()
