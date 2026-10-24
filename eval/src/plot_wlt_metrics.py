import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import seaborn as sns
import argparse
from collections import defaultdict
from constants import DEFAULT_OUTPUT_DIR, DEFAULT_RUN_NAME, LANGS, LANG_NAME_EN, CACHE_DIR
from utils import get_checkpoint_results_dirs
from plot_utils import (
    get_color_lang_mapping,
    get_output_classification_colors,
    prepare_plotting,
    load_results_dirs,
    save_figure,
    _load_token_mapping_for_plotting,
    get_x_values,
    get_x_label,
    get_plot_suffix,
    apply_token_formatter,
)
from collate_fn import (
    get_collate_name_wlt,
    collate_results_wlt,
    collate_predictions_over_time,
    collate_overlap_bin_accuracy_over_time,
    get_collate_name_output_classification,
    get_collate_name_overlap_bin_accuracy,
)

# Global mapping for output classification labels
OUTPUT_CLASSIFICATION_LABELS = {
    "correct": "correct",
    "copy": "copy (wrong)",
    "copy_correct": "copy (correct)",
    "context_copying": "context copying",
    "concept_wrong_lang": "concept wrong language",
    "other": "other",
    "translation": "translation",
    "translation_exact_subtoken_copy": "correct (full token overlap)",
    "translation_non_exact_subtoken_copy": "correct (partial token overlap)",
    "translation_no_subtoken_overlap": "correct (no token overlap)",
    "translation_exact_subtoken_copy_share": "share: full token overlap",
    "translation_non_exact_subtoken_copy_share": "share: partial token overlap",
    "translation_no_subtoken_overlap_share": "share: no token overlap",
}

OUTPUT_CLASSIFICATION_COLORS = get_output_classification_colors()
OUTPUT_CLASSIFICATION_COLORS.update(
    {
        # Keep these within the project palette and align overlap lines with
        # logit-lens Jaccard colors where possible.
        "other": "#FFFFFF",
        "copy": "#F0E442",
        "context_copying": "#56B4E9",
        "translation": "#005A5B",
        "translation_exact_subtoken_copy": "#E69F00",
        "translation_non_exact_subtoken_copy": "#009E73",
        "translation_no_subtoken_overlap": "#CC79A7",
        "translation_exact_subtoken_copy_share": "#E69F00",
        "translation_non_exact_subtoken_copy_share": "#009E73",
        "translation_no_subtoken_overlap_share": "#D55E00",
    }
)

OUTPUT_CLASSIFICATION_AVG_CATEGORIES = [
    "correct",
    "copy_correct",
    "copy",
    "context_copying",
    "concept_wrong_lang",
    "other",
    "translation",
    "translation_exact_subtoken_copy",
    "translation_non_exact_subtoken_copy",
    "translation_no_subtoken_overlap",
    "translation_exact_subtoken_copy_share",
    "translation_non_exact_subtoken_copy_share",
    "translation_no_subtoken_overlap_share",
]


def _average_classification_data(class_data_list):
    """Average classification data across multiple datasets.

    Args:
        class_data_list: List of classification data dicts, each with 'steps' and category keys

    Returns:
        Averaged classification data dict with means and stds, or None if input is empty
    """
    if not class_data_list:
        return None

    step_to_sum = defaultdict(lambda: defaultdict(float))
    step_to_sum_sq = defaultdict(lambda: defaultdict(float))  # For variance calculation
    step_to_count = defaultdict(int)
    step_to_wrong_lang_sum = defaultdict(lambda: defaultdict(float))
    step_to_wrong_lang_sum_sq = defaultdict(
        lambda: defaultdict(float)
    )  # For variance calculation

    for data in class_data_list:
        steps = data.get("steps", [])
        for i, step in enumerate(steps):
            for cat in OUTPUT_CLASSIFICATION_AVG_CATEGORIES:
                if i < len(data.get(cat, [])):
                    val = data[cat][i]
                    step_to_sum[step][cat] += val
                    step_to_sum_sq[step][cat] += val**2
            for lang in LANGS:
                wl_list = data.get("concept_wrong_lang_breakdown", {}).get(lang, [])
                if i < len(wl_list):
                    val = wl_list[i]
                    step_to_wrong_lang_sum[step][lang] += val
                    step_to_wrong_lang_sum_sq[step][lang] += val**2
            step_to_count[step] += 1

    sorted_steps = sorted(step_to_count.keys())
    averaged = {
        "steps": sorted_steps,
        "correct": [],
        "correct_std": [],
        "copy": [],
        "copy_std": [],
        "copy_correct": [],
        "copy_correct_std": [],
        "context_copying": [],
        "context_copying_std": [],
        "concept_wrong_lang": [],
        "concept_wrong_lang_std": [],
        "concept_wrong_lang_breakdown": {lang: [] for lang in LANGS},
        "concept_wrong_lang_breakdown_std": {lang: [] for lang in LANGS},
        "other": [],
        "other_std": [],
        "translation": [],
        "translation_std": [],
        "translation_exact_subtoken_copy": [],
        "translation_exact_subtoken_copy_std": [],
        "translation_non_exact_subtoken_copy": [],
        "translation_non_exact_subtoken_copy_std": [],
        "translation_no_subtoken_overlap": [],
        "translation_no_subtoken_overlap_std": [],
        "translation_exact_subtoken_copy_share": [],
        "translation_exact_subtoken_copy_share_std": [],
        "translation_non_exact_subtoken_copy_share": [],
        "translation_non_exact_subtoken_copy_share_std": [],
        "translation_no_subtoken_overlap_share": [],
        "translation_no_subtoken_overlap_share_std": [],
    }

    for step in sorted_steps:
        n = step_to_count[step]
        for cat in OUTPUT_CLASSIFICATION_AVG_CATEGORIES:
            mean = step_to_sum[step][cat] / n if n > 0 else 0
            variance = (step_to_sum_sq[step][cat] / n - mean**2) if n > 0 else 0
            std = (variance**0.5) if variance > 0 else 0
            averaged[cat].append(mean)
            averaged[f"{cat}_std"].append(std)
        for lang in LANGS:
            mean = step_to_wrong_lang_sum[step][lang] / n if n > 0 else 0
            variance = (
                (step_to_wrong_lang_sum_sq[step][lang] / n - mean**2) if n > 0 else 0
            )
            std = (variance**0.5) if variance > 0 else 0
            averaged["concept_wrong_lang_breakdown"][lang].append(mean)
            averaged["concept_wrong_lang_breakdown_std"][lang].append(std)

    return averaged


def _load_classification_data(
    run_name,
    file_prefix,
    exclude_langs,
    aggregation="all_pairs",
    accuracy_mode="original",
):
    """Load and aggregate classification data based on aggregation mode.

    Args:
        run_name: Run name
        file_prefix: Predictions file prefix
        exclude_langs: List of languages to exclude
        aggregation: "all_pairs", "pair:src_tgt", "to:lang", or "from:lang"

    Returns:
        Tuple of (class_data dict, src_lang, tgt_lang) or (None, None, None) on error
    """
    # Parse aggregation argument
    if aggregation.startswith("pair:"):
        pair = aggregation.split(":")[1]
        src_lang, tgt_lang = pair.split("_")
        collated_file = os.path.join(
            CACHE_DIR,
            get_collate_name_output_classification(
                run_name=run_name,
                tgt_lang=tgt_lang,
                exclude_langs=exclude_langs,
                accuracy_mode=accuracy_mode,
            ),
        )
    elif aggregation.startswith("to:"):
        tgt_lang = aggregation.split(":")[1]
        src_lang = None
        collated_file = os.path.join(
            CACHE_DIR,
            get_collate_name_output_classification(
                run_name=run_name,
                tgt_lang=tgt_lang,
                exclude_langs=exclude_langs,
                accuracy_mode=accuracy_mode,
            ),
        )
    elif aggregation.startswith("from:"):
        src_lang = aggregation.split(":")[1]
        tgt_lang = None
        collated_file = None
    else:
        src_lang, tgt_lang = None, None
        collated_file = None

    if collated_file and not os.path.exists(collated_file):
        print(f"Collated classifications not found at {collated_file}")
        print("Run collation first.")
        return None, None, None

    # Load and extract data based on aggregation
    if aggregation.startswith("pair:"):
        with open(collated_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        class_data = data.get("per_source", {}).get(src_lang)
        if not class_data:
            print(f"No data for {src_lang} -> {tgt_lang}")
            return None, None, None
    elif aggregation.startswith("to:"):
        with open(collated_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        per_source = data.get("per_source", {})
        per_source_data = [v for src, v in per_source.items() if v and src != tgt_lang]
        class_data = _average_classification_data(per_source_data)
        if not class_data:
            class_data = data.get("aggregated")
        if not class_data:
            print(f"No data for all sources -> {tgt_lang}")
            return None, None, None
    elif aggregation.startswith("from:"):
        src_lang = aggregation.split(":")[1]
        tgt_lang = None
        collated_file = None

        # Derive language list directly from LANGS minus exclude_langs
        langs = (
            [lang for lang in LANGS if lang not in exclude_langs]
            if exclude_langs
            else LANGS
        )

        per_target_data = []
        for tgt in langs:
            if tgt == src_lang:
                continue

            tgt_collated_file = os.path.join(
                CACHE_DIR,
                get_collate_name_output_classification(
                    run_name=run_name,
                    tgt_lang=tgt,
                    exclude_langs=exclude_langs,
                    accuracy_mode=accuracy_mode,
                ),
            )
            if not os.path.exists(tgt_collated_file):
                continue

            with open(tgt_collated_file, "r", encoding="utf-8") as f:
                tgt_data = json.load(f)

            src_data = tgt_data.get("per_source", {}).get(src_lang)
            if src_data:
                per_target_data.append(src_data)

        class_data = _average_classification_data(per_target_data)
        if not class_data:
            print(f"No data for {src_lang} -> all targets")
            return None, None, None
    else:
        # All pairs: average across all individual language pairs
        # Collect raw data from each source->target pair to compute correct variance
        src_lang, tgt_lang = None, None
        collated_file = None

        # Derive language list directly from LANGS minus exclude_langs
        langs = (
            [lang for lang in LANGS if lang not in exclude_langs]
            if exclude_langs
            else LANGS
        )

        all_pairs_data = []
        for tgt in langs:
            tgt_collated_file = os.path.join(
                CACHE_DIR,
                get_collate_name_output_classification(
                    run_name=run_name,
                    tgt_lang=tgt,
                    exclude_langs=exclude_langs,
                    accuracy_mode=accuracy_mode,
                ),
            )
            if not os.path.exists(tgt_collated_file):
                continue

            with open(tgt_collated_file, "r", encoding="utf-8") as f:
                tgt_data = json.load(f)

            per_source = tgt_data.get("per_source", {})
            # Collect all raw source->target pairs for this target
            for src, src_data in per_source.items():
                if src_data and src != tgt:
                    all_pairs_data.append(src_data)

        class_data = _average_classification_data(all_pairs_data)
        if not class_data:
            print("No data available for all-pairs aggregation")
            return None, None, None

    return class_data, src_lang, tgt_lang


def _compute_copying_error_fraction_heatmaps(
    run_name,
    exclude_langs=None,
    accuracy_mode="original",
):
    """Compute copying as fraction of total errors: copy/(copy+other+concept_wrong_lang+context_copying) for all language pairs."""
    langs = (
        [lang for lang in LANGS if lang not in exclude_langs]
        if exclude_langs
        else LANGS
    )

    n_langs = len(langs)
    max_error_fraction = np.zeros((n_langs, n_langs))
    max_error_fraction_steps = np.zeros((n_langs, n_langs), dtype=int)
    final_error_fraction = np.zeros((n_langs, n_langs))
    # Track per-step averages across all source->target pairs.
    step_to_error_fractions = defaultdict(list)

    # Initialize with NaN to mark non-computed pairs
    max_error_fraction[:] = np.nan
    max_error_fraction_steps[:] = 0
    final_error_fraction[:] = np.nan

    print(f"Computing copying error fraction heatmaps for {n_langs} languages...")

    for tgt_idx, tgt_lang in enumerate(langs):
        # Load classification data for this target language
        collated_file = os.path.join(
            CACHE_DIR,
            get_collate_name_output_classification(
                run_name=run_name,
                tgt_lang=tgt_lang,
                exclude_langs=exclude_langs,
                accuracy_mode=accuracy_mode,
            ),
        )

        if not os.path.exists(collated_file):
            print(f"Skipping {tgt_lang}: collated file not found")
            continue

        with open(collated_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        per_source = data.get("per_source", {})

        for src_idx, src_lang in enumerate(langs):
            if src_lang == tgt_lang:
                continue

            src_data = per_source.get(src_lang)
            if not src_data:
                continue

            # Compute copying error fraction: copy / (copy + context_copying + other + concept_wrong_lang)
            steps = src_data.get("steps", [])

            copy_rates = list(src_data.get("copy", []))
            context_copying_rates = list(src_data.get("context_copying", []))
            other_rates = list(src_data.get("other", []))
            concept_wrong_lang_rates = list(src_data.get("concept_wrong_lang", []))

            if (
                not steps
                or len(copy_rates) != len(steps)
                or len(context_copying_rates) != len(steps)
                or len(other_rates) != len(steps)
                or len(concept_wrong_lang_rates) != len(steps)
            ):
                continue

            # Compute error fraction: copy / (copy + context_copying + other + concept_wrong_lang)
            error_fraction = []
            for c, cc, o, cwl in zip(
                copy_rates,
                context_copying_rates,
                other_rates,
                concept_wrong_lang_rates,
            ):
                total_errors = c + cc + o + cwl
                if total_errors > 0:
                    error_fraction.append(c / total_errors)
                else:
                    # No errors at this step, set to 0
                    error_fraction.append(0.0)

            # Find max and corresponding step
            if not error_fraction:
                continue

            for step, frac in zip(steps, error_fraction):
                step_to_error_fractions[int(step)].append(float(frac))

            max_val = max(error_fraction)
            max_idx = list(error_fraction).index(max_val)
            max_step = steps[max_idx] if max_idx < len(steps) else 0
            final_val = error_fraction[-1]

            max_error_fraction[src_idx, tgt_idx] = max_val
            max_error_fraction_steps[src_idx, tgt_idx] = max_step
            final_error_fraction[src_idx, tgt_idx] = final_val

    avg_error_fraction_by_step = {
        step: float(np.mean(vals))
        for step, vals in step_to_error_fractions.items()
        if len(vals) > 0
    }
    if avg_error_fraction_by_step:
        max_avg_step = max(
            avg_error_fraction_by_step,
            key=lambda step: avg_error_fraction_by_step[step],
        )
        max_avg_value = avg_error_fraction_by_step[max_avg_step]
    else:
        max_avg_step = None
        max_avg_value = None

    return {
        "max_error_fraction": max_error_fraction,
        "max_error_fraction_steps": max_error_fraction_steps,
        "final_error_fraction": final_error_fraction,
        "avg_error_fraction_by_step": avg_error_fraction_by_step,
        "max_avg_error_fraction": max_avg_value,
        "max_avg_error_fraction_step": max_avg_step,
        "langs": langs,
    }


def _compute_copying_prevalence_heatmaps(
    run_name,
    exclude_langs=None,
    accuracy_mode="original",
):
    """Compute max copying error prevalence and step where max occurs for all language pairs."""
    langs = (
        [lang for lang in LANGS if lang not in exclude_langs]
        if exclude_langs
        else LANGS
    )

    n_langs = len(langs)
    max_copying = np.zeros((n_langs, n_langs))
    max_copying_steps = np.zeros((n_langs, n_langs), dtype=int)
    final_copying = np.zeros((n_langs, n_langs))

    # Initialize with NaN to mark non-computed pairs
    max_copying[:] = np.nan
    max_copying_steps[:] = 0
    final_copying[:] = np.nan

    print(f"Computing copying error prevalence heatmaps for {n_langs} languages...")

    for tgt_idx, tgt_lang in enumerate(langs):
        # Load classification data for this target language
        collated_file = os.path.join(
            CACHE_DIR,
            get_collate_name_output_classification(
                run_name=run_name,
                tgt_lang=tgt_lang,
                exclude_langs=exclude_langs,
                accuracy_mode=accuracy_mode,
            ),
        )

        if not os.path.exists(collated_file):
            print(f"Skipping {tgt_lang}: collated file not found")
            continue

        with open(collated_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        per_source = data.get("per_source", {})

        for src_idx, src_lang in enumerate(langs):
            if src_lang == tgt_lang:
                continue

            src_data = per_source.get(src_lang)
            if not src_data:
                continue

            # Compute copying error prevalence (copy only, not copy_correct)
            steps = src_data.get("steps", [])
            copy_rates = list(src_data.get("copy", []))

            if not steps or len(copy_rates) != len(steps):
                continue

            copying_prevalence = copy_rates

            # Find max and corresponding step
            if not copying_prevalence:
                continue

            max_val = max(copying_prevalence)
            max_idx = list(copying_prevalence).index(max_val)
            max_step = steps[max_idx] if max_idx < len(steps) else 0
            final_val = copying_prevalence[-1]

            max_copying[src_idx, tgt_idx] = max_val
            max_copying_steps[src_idx, tgt_idx] = max_step
            final_copying[src_idx, tgt_idx] = final_val

    return {
        "max_copying": max_copying,
        "max_copying_steps": max_copying_steps,
        "final_copying": final_copying,
        "langs": langs,
    }


def _plot_copying_heatmaps(
    heatmap_data,
    run_name,
    figures_path,
    suffix_prefix,
    title_prefix,
    exclude_langs=None,
    x_axis="steps",
    accuracy_mode="original",
    strip_titles=True,
):
    """Helper function to plot 3 heatmaps (max, steps, final) for copying metrics."""
    max_vals = heatmap_data["max_vals"]
    max_steps = heatmap_data["max_steps"]
    final_vals = heatmap_data["final_vals"]
    langs = heatmap_data["langs"]

    lang_labels = [LANG_NAME_EN[lang] for lang in langs]
    n_langs = len(langs)

    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics="accuracy",
        accuracy_mode=accuracy_mode,
    )

    # Plot 1: Max values heatmap
    max_vals_display = max_vals.copy()
    for i in range(n_langs):
        max_vals_display[i, i] = np.nan

    prepare_plotting()
    fig1, ax1 = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        max_vals_display,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        annot_kws={"fontsize": 15  },
        xticklabels=lang_labels,
        yticklabels=lang_labels,
        vmin=0,
        vmax=1,
        ax=ax1,
    )

    ax1.set_xlabel("Target Language")
    ax1.set_ylabel("Source Language")
    if not strip_titles:
        ax1.set_title(f"Maximum {title_prefix}")

    path1 = os.path.join(figures_path, f"{base_prefix}{suffix_prefix}{suffix}")
    save_figure(fig1, path1, strip_titles=strip_titles)
    print(f"Saved {title_prefix} heatmap to {path1}")

    # Plot 2: Step at max values
    max_steps_display = max_steps.copy().astype(float)
    for i in range(n_langs):
        max_steps_display[i, i] = np.nan

    steps_flat = max_steps[~np.isnan(max_vals)].astype(int)
    if len(steps_flat) > 0:
        max_step_val = np.max(steps_flat)
        min_step_val = np.min(steps_flat)
    else:
        max_step_val = 1
        min_step_val = 0

    fig2, ax2 = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        max_steps_display,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 15},
        cmap="viridis",
        xticklabels=lang_labels,
        yticklabels=lang_labels,
        vmin=min_step_val,
        vmax=max_step_val,
        cbar_kws={"label": "Training Step"},
        ax=ax2,
        mask=np.isnan(max_vals),
    )

    ax2.set_xlabel("Target Language")
    ax2.set_ylabel("Source Language")
    if not strip_titles:
        ax2.set_title(f"Step at Maximum {title_prefix}")

    path2 = os.path.join(figures_path, f"{base_prefix}{suffix_prefix}_step{suffix}")
    save_figure(fig2, path2, strip_titles=strip_titles)
    print(f"Saved step heatmap to {path2}")

    # Plot 3: Final values
    final_vals_display = final_vals.copy()
    for i in range(n_langs):
        final_vals_display[i, i] = np.nan

    fig3, ax3 = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        final_vals_display,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        annot_kws={"fontsize": 15},
        xticklabels=lang_labels,
        yticklabels=lang_labels,
        vmin=0,
        vmax=1,
        ax=ax3,
    )

    ax3.set_xlabel("Target Language")
    ax3.set_ylabel("Source Language")
    if not strip_titles:
        ax3.set_title(f"Final Checkpoint {title_prefix}")

    path3 = os.path.join(
        figures_path,
        f"{base_prefix}{suffix_prefix}_final{suffix}",
    )
    save_figure(fig3, path3, strip_titles=strip_titles)
    print(f"Saved final checkpoint {title_prefix} heatmap to {path3}")


def plot_output_classification_copying_heatmaps(
    run_name,
    figures_path="figures",
    exclude_langs=None,
    accuracy_mode="original",
    x_axis="steps",
    strip_titles=True,
):
    """Plot heatmaps of max copying error prevalence and error fraction for all language pairs."""

    # Compute both prevalence and fraction data
    prevalence_data = _compute_copying_prevalence_heatmaps(
        run_name,
        exclude_langs=exclude_langs,
        accuracy_mode=accuracy_mode,
    )

    fraction_data = _compute_copying_error_fraction_heatmaps(
        run_name,
        exclude_langs=exclude_langs,
        accuracy_mode=accuracy_mode,
    )

    if fraction_data.get("max_avg_error_fraction") is not None:
        max_avg = fraction_data["max_avg_error_fraction"]
        max_avg_step = fraction_data["max_avg_error_fraction_step"]
        print(
            "Max average copying error fraction across all language pairs: "
            f"{max_avg:.4f} at step {max_avg_step}"
        )

    # Plot prevalence heatmaps
    _plot_copying_heatmaps(
        {
            "max_vals": prevalence_data["max_copying"],
            "max_steps": prevalence_data["max_copying_steps"],
            "final_vals": prevalence_data["final_copying"],
            "langs": prevalence_data["langs"],
        },
        run_name,
        figures_path,
        "_copying_prevalence_heatmap",
        "Copying Error Prevalence",
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        accuracy_mode=accuracy_mode,
        strip_titles=strip_titles,
    )

    # Plot fraction heatmaps
    _plot_copying_heatmaps(
        {
            "max_vals": fraction_data["max_error_fraction"],
            "max_steps": fraction_data["max_error_fraction_steps"],
            "final_vals": fraction_data["final_error_fraction"],
            "langs": fraction_data["langs"],
        },
        run_name,
        figures_path,
        "_copying_error_fraction_heatmap",
        "Copying Error Fraction",
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        accuracy_mode=accuracy_mode,
        strip_titles=strip_titles,
    )


def plot_wlt_output_classification_stacked(
    run_name,
    file_prefix="eval-translation_predictions",
    figures_path="figures",
    exclude_langs=None,
    max_steps=None,
    x_axis="steps",
    mode="expl-training",
    aggregation="all_pairs",
    hide_concept_wrong_lang=False,
    translation_success_composition=False,
    accuracy_mode="original",
    y_axis_break_lower=0.65,
    y_axis_break_upper=0.9,
    y_axis_break_upper_ratio=None,
    strip_titles=True,
):
    """Plot stacked area chart of output classifications over training steps.

    Classifies outputs as: correct, copy, copy_correct, context_copying, concept_wrong_lang, other.
    If hide_concept_wrong_lang=True, concept_wrong_lang is merged into other.

    Args:
        run_name: Run name
        aggregation: "all_pairs" for all language pairs, "pair:en_de" for single pair,
                     "to:en" for all sources to target, "from:en" for source to all targets
        hide_concept_wrong_lang: If True, exclude concept_wrong_lang from the plot
        y_axis_break_lower: Lower limit of omitted middle y band. Set to None to disable.
        y_axis_break_upper: Upper limit of omitted middle y band.
        y_axis_break_upper_ratio: Relative height of upper y-axis segment. If None,
            the height is set proportional to the shown y-range widths.
    """
    # Load classification data
    class_data, src_lang, tgt_lang = _load_classification_data(
        run_name,
        file_prefix,
        exclude_langs,
        aggregation,
        accuracy_mode=accuracy_mode,
    )
    if class_data is None:
        return

    steps = class_data["steps"]
    if not steps:
        print("No data to plot")
        return

    # Load token mapping if needed
    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )

    x_vals = get_x_values(steps, step_to_tokens, x_axis)

    prepare_plotting()
    # Create stacked area plot (optionally with a broken y-axis).
    use_y_break = (
        y_axis_break_lower is not None
        and y_axis_break_upper is not None
        and 0 < float(y_axis_break_lower) < float(y_axis_break_upper) < 1
    )
    fig_size = (6,3.7)
    if use_y_break:
        y_break_lower = float(y_axis_break_lower)
        y_break_upper = float(y_axis_break_upper)
        upper_range = 1.0 - y_break_upper
        lower_range = y_break_lower
        if y_axis_break_upper_ratio is None:
            # Keep panel heights proportional to the visible y-ranges.
            upper_ratio = upper_range / lower_range if lower_range > 0 else 0.2
        else:
            upper_ratio = float(y_axis_break_upper_ratio)
        upper_ratio = max(0.05, upper_ratio)

        fig = plt.figure(figsize=fig_size)
        gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[upper_ratio, 1.0],
            hspace=0.08,
        )
        ax_upper = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1], sharex=ax_upper)
        plot_axes = [ax_upper, ax]
    else:
        fig, ax = plt.subplots(figsize=fig_size)
        ax_upper = None
        plot_axes = [ax]

    # Prepare data for stacking
    if translation_success_composition:
        # Replace correct/copy_correct with breakdown by token overlap (as fractions of total)
        plot_series = {
            "translation_exact_subtoken_copy": np.array(
                class_data.get("translation_exact_subtoken_copy", []), dtype=float
            ),
            "translation_non_exact_subtoken_copy": np.array(
                class_data.get("translation_non_exact_subtoken_copy", []), dtype=float
            ),
            "translation_no_subtoken_overlap": np.array(
                class_data.get("translation_no_subtoken_overlap", []), dtype=float
            ),
            "copy": np.array(class_data.get("copy", []), dtype=float),
            "context_copying": np.array(
                class_data.get("context_copying", []), dtype=float
            ),
            "concept_wrong_lang": np.array(
                class_data.get("concept_wrong_lang", []), dtype=float
            ),
            "other": np.array(class_data.get("other", []), dtype=float),
        }
        if hide_concept_wrong_lang:
            plot_series["other"] = (
                plot_series["other"] + plot_series["concept_wrong_lang"]
            )
        categories = [
            "translation_exact_subtoken_copy",
            "translation_non_exact_subtoken_copy",
            "translation_no_subtoken_overlap",
            "copy",
            "context_copying",
            "concept_wrong_lang",
            "other",
        ]
        if hide_concept_wrong_lang:
            categories = [
                "translation_exact_subtoken_copy",
                "translation_non_exact_subtoken_copy",
                "translation_no_subtoken_overlap",
                "copy",
                "context_copying",
                "other",
            ]
    else:
        plot_series = {
            cat: np.array(class_data.get(cat, []), dtype=float)
            for cat in [
                "correct",
                "copy_correct",
                "copy",
                "context_copying",
                "concept_wrong_lang",
                "other",
            ]
        }
        if hide_concept_wrong_lang:
            plot_series["other"] = (
                plot_series["other"] + plot_series["concept_wrong_lang"]
            )
        categories = [
            "correct",
            "copy_correct",
            "copy",
            "context_copying",
            "concept_wrong_lang",
            "other",
        ]
        if hide_concept_wrong_lang:
            categories = [
                "correct",
                "copy_correct",
                "copy",
                "context_copying",
                "other",
            ]

    colors = OUTPUT_CLASSIFICATION_COLORS
    stackdata = np.array([plot_series[cat] for cat in categories])

    labels = [OUTPUT_CLASSIFICATION_LABELS.get(cat, cat) for cat in categories]
    if translation_success_composition:
        # Use compact labels in composition mode; grouping is handled in legend.
        compact_label_overrides = {
            "translation_exact_subtoken_copy": "full overlap",
            "translation_non_exact_subtoken_copy": "partial overlap",
            "translation_no_subtoken_overlap": "no overlap",
            "copy": "source word copying",
            "context_copying": "context copying",
            "concept_wrong_lang": "wrong language",
            "other": "other",
        }
        labels = [
            compact_label_overrides.get(cat, OUTPUT_CLASSIFICATION_LABELS.get(cat, cat))
            for cat in categories
        ]

    for plot_ax in plot_axes:
        plot_ax.stackplot(
            x_vals,
            stackdata,
            labels=labels,
            colors=[colors.get(cat, "#CCCCCC") for cat in categories],
            alpha=0.8,
        )

    if translation_success_composition:
        # Draw a bold divider between correct translation region and wrong outputs.
        correct_boundary = (
            np.array(
                plot_series.get("translation_exact_subtoken_copy", []), dtype=float
            )
            + np.array(
                plot_series.get("translation_non_exact_subtoken_copy", []), dtype=float
            )
            + np.array(
                plot_series.get("translation_no_subtoken_overlap", []), dtype=float
            )
        )
        for plot_ax in plot_axes:
            plot_ax.plot(
                x_vals,
                correct_boundary,
                color="#1f1f1f",
                linewidth=2,
                alpha=0.95,
                label="translation accuracy",
                zorder=5,
            )

    for plot_ax in plot_axes:
        plot_ax.set_xlim(left=min(x_vals), right=max(x_vals))

    ax.set_xlabel(get_x_label(x_axis))

    if aggregation.startswith("pair:"):
        title = f"Output Classification: {LANG_NAME_EN[src_lang]} → {LANG_NAME_EN[tgt_lang]}"
    elif aggregation.startswith("to:"):
        tgt_lang_code = aggregation.split(":")[1]
        title = f"Output Classification: All → {LANG_NAME_EN[tgt_lang_code]}"
    elif aggregation.startswith("from:"):
        src_lang_code = aggregation.split(":")[1]
        title = f"Output Classification: {LANG_NAME_EN[src_lang_code]} → All"
    else:
        title = "Output Classification: All Language Pairs"
    title_ax = ax_upper if use_y_break else ax
    title_ax.set_title(title)

    if use_y_break:
        ax.set_ylim(0, y_break_lower)
        ax_upper.set_ylim(y_break_upper, 1)
        ax_upper.set_yticks([y_break_upper, 1.0])

        # Hide touching spines and duplicate x labels to indicate the y-axis break.
        ax_upper.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax_upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Draw parallel diagonal markers to denote the axis break.
        d = 0.012
        kwargs_upper = dict(transform=ax_upper.transAxes, color="k", clip_on=False)
        kwargs_lower = dict(transform=ax.transAxes, color="k", clip_on=False)
        ax_upper.plot((-d, +d), (-d, +d), **kwargs_upper)
        ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs_upper)
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs_lower)
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_lower)
    else:
        ax.set_ylim(0, 1)

    legend_fontsize = 15  # if translation_success_composition else 10
    if translation_success_composition:
        grouped_legend_handles = [
            plt.Line2D([], [], linestyle="none", label="Correct Outputs"),
            plt.Line2D(
                [],
                [],
                color=colors.get("translation_exact_subtoken_copy", "#E69F00"),
                linewidth=5,
                label="full token overlap",
            ),
            plt.Line2D(
                [],
                [],
                color=colors.get("translation_non_exact_subtoken_copy", "#009E73"),
                linewidth=5,
                label="partial token overlap",
            ),
            plt.Line2D(
                [],
                [],
                color=colors.get("translation_no_subtoken_overlap", "#D55E00"),
                linewidth=5,
                label="no token overlap",
            ),
            plt.Line2D(
                [],
                [],
                color="#1f1f1f",
                linewidth=2,
                label="translation accuracy",
            ),
            plt.Line2D([], [], linestyle="none", label=" "),
            plt.Line2D([], [], linestyle="none", label="Wrong Outputs"),
            plt.Line2D(
                [],
                [],
                color=colors.get("copy", "#F0E442"),
                linewidth=5,
                label="source word copying",
            ),
            plt.Line2D(
                [],
                [],
                color=colors.get("context_copying", "#56B4E9"),
                linewidth=5,
                label="context copying",
            ),
        ]
        if not hide_concept_wrong_lang:
            grouped_legend_handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=colors.get("concept_wrong_lang", "#CCCCCC"),
                    linewidth=5,
                    label="wrong language",
                )
            )
        other_legend_handle = plt.Line2D(
            [],
            [],
            color=colors.get("other", "#FFFFFF"),
            linewidth=5,
            label="other",
        )
        other_legend_handle.set_path_effects(
            [pe.Stroke(linewidth=6.4, foreground="#666666"), pe.Normal()]
        )
        grouped_legend_handles.append(other_legend_handle)
        legend = title_ax.legend(
            handles=grouped_legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=legend_fontsize,
        )
        for text in legend.get_texts():
            raw_label = text.get_text()
            legend_label = text.get_text().strip().lower()
            if raw_label == " ":
                # Keep a small visual group break without a full extra line.
                text.set_fontsize(1)
            elif legend_label.startswith("correct") or legend_label.startswith("wrong"):
                text.set_fontweight("bold")
    else:
        legend = title_ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=legend_fontsize,
        )
        # Ensure white categories remain visible in the legend.
        for text, handle in zip(legend.get_texts(), legend.legend_handles):
            if text.get_text().strip().lower() != "other":
                continue
            if hasattr(handle, "set_edgecolor"):
                handle.set_edgecolor("#666666")
            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(5)
            if hasattr(handle, "set_path_effects"):
                handle.set_path_effects(
                    [pe.Stroke(linewidth=6.4, foreground="#666666"), pe.Normal()]
                )
    for plot_ax in plot_axes:

        ax.grid(True, alpha=0.3, axis="y")


    if max_steps is not None:
        if x_axis == "tokens":
            max_x = step_to_tokens.get(max_steps, max_steps)
        else:
            max_x = max_steps
        for plot_ax in plot_axes:
            plot_ax.set_xlim(left=0, right=max_x)

    if x_axis == "tokens":
        milestone_tokens = 11_000_000_000
        x_min, x_max = ax.get_xlim()
        phase_text_ax = ax_upper if use_y_break else ax
        if x_min <= milestone_tokens <= x_max:
            for plot_ax in plot_axes:
                plot_ax.axvline(
                    milestone_tokens,
                    color="black",
                    zorder=5,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )
        if x_min <= 3_000_000_000 <= x_max:
            phase_text_ax.text(
                3_000_000_000,
                0.08,
                "Phase I",
                transform=phase_text_ax.get_xaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=15,
                color="black",
            )
        if x_min <= 22_500_000_000 <= x_max:
            phase_text_ax.text(
                22_500_000_000,
                0.08,
                "Phase II",
                transform=phase_text_ax.get_xaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=15,
                color="black",
            )

    apply_token_formatter(ax, x_axis)
    plt.tight_layout()

    # Save figure
    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics="accuracy",
        accuracy_mode=accuracy_mode,
    )

    agg_suffix = (
        "" if aggregation == "all_pairs" else f"_{aggregation.replace(':', '_')}"
    )
    comp_suffix = "_translation_composition" if translation_success_composition else ""
    fname = f"{base_prefix}_output_classification{agg_suffix}{comp_suffix}{suffix}"

    os.makedirs(figures_path, exist_ok=True)
    save_figure(fig, os.path.join(figures_path, fname), strip_titles=strip_titles)
    print(f"Saved output classification plot to {os.path.join(figures_path, fname)}")


def plot_wlt_output_classification_lines(
    run_name,
    file_prefix="eval-translation_predictions",
    figures_path="figures",
    exclude_langs=None,
    max_steps=None,
    x_axis="steps",
    mode="expl-training",
    aggregation="all_pairs",
    hide_concept_wrong_lang=False,
    translation_success_composition=False,
    accuracy_mode="original",
    strip_titles=True,
):
    """Plot line chart of output classification proportions over training steps.

    Shows proportions of: correct, copy, copy_correct, context_copying, concept_wrong_lang.
    The line plot never includes "other".

    Args:
        run_name: Run name
        aggregation: "all_pairs" for all language pairs, "pair:en_de" for single pair,
                     "to:en" for all sources to target, "from:en" for source to all targets
        hide_concept_wrong_lang: If True, exclude concept_wrong_lang from the plot
    """
    # Load classification data
    class_data, src_lang, tgt_lang = _load_classification_data(
        run_name,
        file_prefix,
        exclude_langs,
        aggregation,
        accuracy_mode=accuracy_mode,
    )
    if class_data is None:
        return

    steps = class_data["steps"]
    if not steps:
        print("No data to plot")
        return

    # Load token mapping if needed
    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )

    x_vals = get_x_values(steps, step_to_tokens, x_axis)

    # Create line plot
    prepare_plotting()
    fig, ax = plt.subplots()

    if translation_success_composition:
        # Replace correct/copy_correct with mutually exclusive overlap buckets.
        categories = [
            "translation_exact_subtoken_copy",
            "translation_non_exact_subtoken_copy",
            "translation_no_subtoken_overlap",
            "copy",
            "context_copying",
            "concept_wrong_lang",
        ]
        legend_label_overrides = {
            "translation_exact_subtoken_copy": "full overlap",
            "translation_non_exact_subtoken_copy": "partial overlap",
            "translation_no_subtoken_overlap": "no overlap",
            "copy": "source word copying",
            "context_copying": "context copying",
            "concept_wrong_lang": "wrong language",
        }
    else:
        # Non-composition mode: keep correct and valid copy as separate correct outputs.
        categories = [
            "correct",
            "copy_correct",
            "context_copying",
            "copy",
        ]
        legend_label_overrides = {
            "correct": "Correct translation",
            "copy_correct": "Valid copy",
            "context_copying": "Context copying",
            "copy": "Source word copying",
        }

    # Build plotted series for non-"other" categories only.
    plot_series = {
        cat: np.array(class_data.get(cat, []), dtype=float) for cat in categories
    }

    if not translation_success_composition:
        plot_series["copying_total"] = (
            plot_series.get("copy", np.zeros(len(steps), dtype=float))
            + plot_series.get("context_copying", np.zeros(len(steps), dtype=float))
        )

    # Categories to plot (excluding "other")
    if hide_concept_wrong_lang:
        categories = [cat for cat in categories if cat != "concept_wrong_lang"]
    colors = OUTPUT_CLASSIFICATION_COLORS
    linestyles = {
        "correct": "-",
        "copy_correct": "-",
        "copy": "-",
        "context_copying": "-",
        "copying_total": "--",
        "concept_wrong_lang": "-",
        "translation": "-",
        "translation_exact_subtoken_copy": "-",
        "translation_non_exact_subtoken_copy": "-",
        "translation_no_subtoken_overlap": "-",
    }

    plotted_handles = {}
    for category in categories:
        # Plot mean line
        line, = ax.plot(
            x_vals,
            plot_series[category],
            label=legend_label_overrides.get(
                category, OUTPUT_CLASSIFICATION_LABELS.get(category, category)
            ),
            color=colors.get(category, colors.get("correct", "#009E73")),
            linewidth=2.5,
            linestyle=linestyles[category],
            alpha=0.8,
        )
        plotted_handles[category] = line

        # Add shaded region for standard deviation (only if std data exists)
        std_key = f"{category}_std"
        if std_key in class_data:
            std_vals = np.array(class_data[std_key])
            mean_vals = np.array(plot_series[category])
            ax.fill_between(
                x_vals,
                np.maximum(mean_vals - std_vals, 0),
                np.minimum(mean_vals + std_vals, 1),
                color=colors[category],
                alpha=0.15,
            )

    if not translation_success_composition:
        total_label = "Total"# (wrong+context)"
        total_line, = ax.plot(
            x_vals,
            plot_series["copying_total"],
            label=total_label,
            color="gray",
            linewidth=2,
            linestyle="-",#linestyles["copying_total"],
        )
        plotted_handles["copying_total"] = total_line

    # Report peak points for error modes.
    if not translation_success_composition:
        error_modes = ["copy", "context_copying", "copying_total"]
        for mode_cat in error_modes:
            vals = np.array(plot_series.get(mode_cat, []), dtype=float)
            if len(vals) != len(steps) or len(vals) == 0:
                continue
            max_idx = int(np.argmax(vals))
            max_val = float(vals[max_idx])
            max_step = int(steps[max_idx])
            if mode_cat == "copying_total":
                mode_label = "Total copying (wrong+context)"
            else:
                mode_label = legend_label_overrides.get(mode_cat, mode_cat)
            if x_axis == "tokens":
                max_tokens = step_to_tokens.get(max_step, max_step)
                print(
                    f"Max average {mode_label}: {max_val:.4f} at step {max_step} "
                    f"(tokens {max_tokens})"
                )
            else:
                print(f"Max average {mode_label}: {max_val:.4f} at step {max_step}")

            if mode_cat == "copying_total":
                final_val = float(vals[-1])
                print(f"Final average {mode_label}: {final_val:.4f}")

        # Report crossings of Source word copying and Context copying.
        copy_vals = np.array(plot_series.get("copy", []), dtype=float)
        context_vals = np.array(plot_series.get("context_copying", []), dtype=float)
        if len(copy_vals) == len(context_vals) == len(steps) and len(steps) > 1:
            diff = copy_vals - context_vals
            crossings = []
            for i in range(1, len(diff)):
                d0 = float(diff[i - 1])
                d1 = float(diff[i])
                # Ignore a boundary equality at the very first checkpoint.
                if i == 1 and d0 == 0.0:
                    continue

                if d0 == 0.0:
                    crossings.append((i - 1, 0.0))
                    continue

                if d0 * d1 < 0 or d1 == 0.0:
                    denom = (d0 - d1)
                    t = (d0 / denom) if denom != 0 else 0.0
                    t = float(np.clip(t, 0.0, 1.0))
                    crossings.append((i - 1, t))

            if crossings:
                first_i0, first_t = crossings[0]
                first_step_cross = float(
                    steps[first_i0] + first_t * (steps[first_i0 + 1] - steps[first_i0])
                )
                first_x0 = float(x_vals[first_i0])
                first_x1 = float(x_vals[first_i0 + 1])
                first_x_cross = first_x0 + first_t * (first_x1 - first_x0)

                last_i0, last_t = crossings[-1]
                last_step_cross = float(
                    steps[last_i0] + last_t * (steps[last_i0 + 1] - steps[last_i0])
                )
                last_x0 = float(x_vals[last_i0])
                last_x1 = float(x_vals[last_i0 + 1])
                last_x_cross = last_x0 + last_t * (last_x1 - last_x0)

                if x_axis == "tokens":
                    print(
                        "Source/context copying first post-initial cross at "
                        f"~step {first_step_cross:.1f} (tokens {int(round(first_x_cross))})"
                    )
                    if len(crossings) > 1:
                        print(
                            "Source/context copying last cross at "
                            f"~step {last_step_cross:.1f} (tokens {int(round(last_x_cross))})"
                        )
                else:
                    print(
                        "Source/context copying first post-initial cross at "
                        f"~step {first_step_cross:.1f}"
                    )
                    if len(crossings) > 1:
                        print(
                            "Source/context copying last cross at "
                            f"~step {last_step_cross:.1f}"
                        )
            else:
                print("Source/context copying curves do not cross.")
    ax.set_xlim(left=min(x_vals), right=max(x_vals))
    ax.set_xlabel(get_x_label(x_axis))
    ax.set_ylabel("Proportion of outputs")

    if aggregation.startswith("pair:"):
        title = f"Output Classification: {LANG_NAME_EN[src_lang]} → {LANG_NAME_EN[tgt_lang]}"
    elif aggregation.startswith("to:"):
        tgt_lang_code = aggregation.split(":")[1]
        title = f"Output Classification: All Sources → {LANG_NAME_EN[tgt_lang_code]}"
    elif aggregation.startswith("from:"):
        src_lang_code = aggregation.split(":")[1]
        title = f"Output Classification: {LANG_NAME_EN[src_lang_code]} → All Targets"
    else:
        title = "Output Classification: All Language Pairs (Aggregated)"
    ax.set_title(title)

    ax.set_ylim(0, 1)
    if not translation_success_composition:
        grouped_legend_handles = [
            Line2D([], [], linestyle="none", label="Correct Outputs"),
            plotted_handles["correct"],
            plotted_handles["copy_correct"],
            Line2D([], [], linestyle="none", label=" "),
            Line2D([], [], linestyle="none", label="Wrong Outputs"),
            plotted_handles["copy"],
            plotted_handles["context_copying"],
            plotted_handles["copying_total"],
        ]
        legend = ax.legend(
            handles=grouped_legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
        )
        for text in legend.get_texts():
            raw_label = text.get_text()
            label = raw_label.strip().lower()
            if raw_label == " ":
                text.set_fontsize(1)
            elif label in {"correct outputs", "wrong outputs"}:
                text.set_fontweight("bold")
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, alpha=0.3)# axis="y")

    if max_steps is not None:
        if x_axis == "tokens":
            max_x = step_to_tokens.get(max_steps, max_steps)
        else:
            max_x = max_steps
        ax.set_xlim(left=0, right=max_x)

    apply_token_formatter(ax, x_axis)
    plt.tight_layout()

    # Save figure
    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics="accuracy",
        accuracy_mode=accuracy_mode,
    )

    agg_suffix = (
        "" if aggregation == "all_pairs" else f"_{aggregation.replace(':', '_')}"
    )
    comp_suffix = "_translation_composition" if translation_success_composition else ""
    fname = (
        f"{base_prefix}_output_classification_lines{agg_suffix}{comp_suffix}{suffix}"
    )

    os.makedirs(figures_path, exist_ok=True)
    save_figure(fig, os.path.join(figures_path, fname), strip_titles=strip_titles)
    print(
        f"Saved output classification line plot to {os.path.join(figures_path, fname)}"
    )


def _load_repetition_classification_data(
    run_name,
    exclude_langs,
    accuracy_mode="original",
):
    """Load src=tgt (repetition task) classification data across languages."""
    langs = [lang for lang in LANGS if lang not in (exclude_langs or [])]
    repetition_pairs_data = []

    for lang in langs:
        collated_file = os.path.join(
            CACHE_DIR,
            get_collate_name_output_classification(
                run_name=run_name,
                tgt_lang=lang,
                exclude_langs=exclude_langs,
                accuracy_mode=accuracy_mode,
            ),
        )
        if not os.path.exists(collated_file):
            continue

        with open(collated_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        src_eq_tgt = data.get("per_source", {}).get(lang)
        if src_eq_tgt:
            repetition_pairs_data.append(src_eq_tgt)

    return _average_classification_data(repetition_pairs_data)


def plot_wlt_repetition_context_copying_evolution(
    run_name,
    figures_path="figures",
    exclude_langs=None,
    max_steps=None,
    x_axis="steps",
    mode="expl-training",
    accuracy_mode="original",
    strip_titles=True,
):
    """Plot context-copying rate over time for repetition tasks (src=tgt) per language."""
    langs = [lang for lang in LANGS if lang not in (exclude_langs or [])]
    if not langs:
        print("No languages available for repetition context-copying plot.")
        return

    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )

    color_map = get_color_lang_mapping()
    prepare_plotting()
    fig, ax = plt.subplots()

    step_to_values = defaultdict(list)
    plotted_any = False

    for lang in langs:
        collated_file = os.path.join(
            CACHE_DIR,
            get_collate_name_output_classification(
                run_name=run_name,
                tgt_lang=lang,
                exclude_langs=exclude_langs,
                accuracy_mode=accuracy_mode,
            ),
        )
        if not os.path.exists(collated_file):
            continue

        with open(collated_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        src_eq_tgt = data.get("per_source", {}).get(lang)
        if not src_eq_tgt:
            continue

        steps = src_eq_tgt.get("steps", [])
        context_vals = np.array(src_eq_tgt.get("context_copying", []), dtype=float)
        if not steps or len(context_vals) != len(steps):
            continue

        x_vals = get_x_values(steps, step_to_tokens, x_axis)
        ax.plot(
            x_vals,
            context_vals,
            label=LANG_NAME_EN[lang],
            color=color_map[lang],
            linewidth=2,
            alpha=0.9,
        )
        plotted_any = True

        for step, val in zip(steps, context_vals):
            step_to_values[int(step)].append(float(val))

    if not plotted_any:
        print("No repetition context-copying data available to plot.")
        plt.close(fig)
        return

    if step_to_values:
        avg_steps = sorted(step_to_values.keys())
        avg_vals = np.array(
            [np.mean(step_to_values[s]) for s in avg_steps],
            dtype=float,
        )
        x_avg = get_x_values(avg_steps, step_to_tokens, x_axis)
        ax.plot(
            x_avg,
            avg_vals,
            label="Average",
            color="black",
            linewidth=3,
        )

    ax.set_xlabel(get_x_label(x_axis))
    ax.set_ylabel("Context copying rate")
    ax.set_title("Repetition Context Copying (src=tgt)")
    ax.set_ylim(0, 1)
    ax.legend(title="Languages", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    if max_steps is not None:
        if x_axis == "tokens":
            max_x = step_to_tokens.get(max_steps, max_steps)
        else:
            max_x = max_steps
        ax.set_xlim(right=max_x)

    apply_token_formatter(ax, x_axis)

    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics="accuracy",
        accuracy_mode=accuracy_mode,
    )
    fname = f"{base_prefix}_evolution_repetition_context_copying{suffix}"
    os.makedirs(figures_path, exist_ok=True)
    save_figure(fig, os.path.join(figures_path, fname), strip_titles=strip_titles)
    print(
        "Saved repetition context-copying evolution plot to "
        f"{os.path.join(figures_path, fname)}"
    )


def plot_wlt_overlap_bin_accuracy_evolution(
    run_name,
    figures_path="figures",
    exclude_langs=None,
    max_steps=None,
    x_axis="steps",
    mode="expl-training",
    accuracy_mode="original",
    strip_titles=True,
):
    """Plot accuracy-over-time curves for fixed overlap bins.

    Accuracy in each bin is correct/total among concepts assigned to that bin.
    """
    collated_file = os.path.join(
        CACHE_DIR,
        get_collate_name_overlap_bin_accuracy(
            run_name=run_name,
            exclude_langs=exclude_langs,
            accuracy_mode=accuracy_mode,
        ),
    )
    if not os.path.exists(collated_file):
        print(f"Overlap-bin collation not found at {collated_file}, skipping plot.")
        return

    with open(collated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = [s["step"] for s in data.get("steps", [])]
    if not steps:
        print("No overlap-bin accuracy data to plot.")
        return

    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )
    x_vals = get_x_values(steps, step_to_tokens, x_axis)

    curves = data.get("curves", {})
    bin_order = data.get(
        "bin_priority",
        ["identical_subtokens", "partial_subtoken_overlap", "no_subtoken_overlap"],
    )

    label_map = {
        "identical_subtokens": "full (copy)",
        "partial_subtoken_overlap": "partial",
        "no_subtoken_overlap": "none",
    }
    color_map = {
        "identical_subtokens": OUTPUT_CLASSIFICATION_COLORS.get(
            "translation_exact_subtoken_copy", "#E69F00"
        ),
        "partial_subtoken_overlap": OUTPUT_CLASSIFICATION_COLORS.get(
            "translation_non_exact_subtoken_copy", "#009E73"
        ),
        "no_subtoken_overlap": OUTPUT_CLASSIFICATION_COLORS.get(
            "translation_no_subtoken_overlap", "#D55E00"
        ),
    }

    prepare_plotting()
    fig, ax = plt.subplots(figsize=(5,4))

    for bin_name in bin_order:
        y_vals = np.array(curves.get(bin_name, []), dtype=float)
        if len(y_vals) != len(steps):
            continue

        label = label_map.get(bin_name, bin_name)

        ax.plot(
            x_vals,
            y_vals,
            label=label,
            color=color_map.get(bin_name, "#666666"),
            linewidth=2.5,
            alpha=0.9,
        )

    ax.set_xlim(left=min(x_vals), right=max(x_vals))
    ax.set_ylim(0, 1)
    ax.set_xlabel(get_x_label(x_axis))
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Token-Overlap Bin")
    ax.grid(True, alpha=0.3)

    if max_steps is not None:
        if x_axis == "tokens":
            max_x = step_to_tokens.get(max_steps, max_steps)
        else:
            max_x = max_steps
        ax.set_xlim(left=0, right=max_x)

    apply_token_formatter(ax, x_axis)

    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics="accuracy",
        accuracy_mode=accuracy_mode,
    )
    fname = f"{base_prefix}_evolution_overlap_bin_accuracy{suffix}_nolegend"
    os.makedirs(figures_path, exist_ok=True)
    save_figure(fig, os.path.join(figures_path, fname), strip_titles=strip_titles)
    print(
        "Saved overlap-bin accuracy evolution plot to "
        f"{os.path.join(figures_path, fname)}"
    )


def plot_wlt_overlap_bin_copying_rates_evolution(
    run_name,
    figures_path="figures",
    exclude_langs=None,
    max_steps=None,
    x_axis="steps",
    mode="expl-training",
    accuracy_mode="original",
    strip_titles=True,
):
    """Plot source-word copying and context-copying rates per overlap bin in one panel."""
    collated_file = os.path.join(
        CACHE_DIR,
        get_collate_name_overlap_bin_accuracy(
            run_name=run_name,
            exclude_langs=exclude_langs,
            accuracy_mode=accuracy_mode,
        ),
    )
    if not os.path.exists(collated_file):
        print(f"Overlap-bin collation not found at {collated_file}, skipping plot.")
        return

    with open(collated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = [s["step"] for s in data.get("steps", [])]
    if not steps:
        print("No overlap-bin copying-rate data to plot.")
        return

    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )
    x_vals = get_x_values(steps, step_to_tokens, x_axis)

    source_curves = data.get("source_word_copying_curves", {})
    context_curves = data.get("context_copying_curves", {})
    bin_order = data.get(
        "bin_priority",
        ["identical_subtokens", "partial_subtoken_overlap", "no_subtoken_overlap"],
    )

    label_map = {
        "identical_subtokens": "full (copy)",
        "partial_subtoken_overlap": "partial",
        "no_subtoken_overlap": "none",
    }
    color_map = {
        "identical_subtokens": OUTPUT_CLASSIFICATION_COLORS.get(
            "translation_exact_subtoken_copy", "#E69F00"
        ),
        "partial_subtoken_overlap": OUTPUT_CLASSIFICATION_COLORS.get(
            "translation_non_exact_subtoken_copy", "#009E73"
        ),
        "no_subtoken_overlap": OUTPUT_CLASSIFICATION_COLORS.get(
            "translation_no_subtoken_overlap", "#D55E00"
        ),
    }

    prepare_plotting()
    fig, ax = plt.subplots(figsize=(8, 4))

    for bin_name in bin_order:
        source_vals = np.array(source_curves.get(bin_name, []), dtype=float)
        context_vals = np.array(context_curves.get(bin_name, []), dtype=float)
        if len(source_vals) == len(steps):
            ax.plot(
                x_vals,
                source_vals,
                label=f"{label_map.get(bin_name, bin_name)}: source",
                color=color_map.get(bin_name, "#666666"),
                linewidth=2.5,
                alpha=0.9,
                linestyle="-",
            )
        if len(context_vals) == len(steps):
            ax.plot(
                x_vals,
                context_vals,
                label=f"{label_map.get(bin_name, bin_name)}: context",
                color=color_map.get(bin_name, "#666666"),
                linewidth=2.3,
                alpha=0.9,
                linestyle="--",
            )

    ax.set_xlim(left=min(x_vals), right=max(x_vals))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Fraction of Outputs")
    ax.set_xlabel(get_x_label(x_axis))
    ax.set_title("Copying error mode")

    style_handles = [
        Line2D([], [], linestyle="none", label="Copy Variant"),
        Line2D(
            [0],
            [0],
            color="#BDBDBD",
            linewidth=2.6,
            linestyle="-",
            label="Source word copying",
        ),
        Line2D(
            [0],
            [0],
            color="#BDBDBD",
            linewidth=2.6,
            linestyle="--",
            label="Context copying",
        ),
    ]
    category_handles = [
        Line2D(
            [0],
            [0],
            color=color_map.get(bin_name, "#666666"),
            linewidth=2.6,
            linestyle="-",
            label=label_map.get(bin_name, bin_name),
        )
        for bin_name in bin_order
    ]
    token_overlap_header = [Line2D([], [], linestyle="none", label="Token Overlap")]

    ax.legend(
        handles=token_overlap_header
        + category_handles
        + [Line2D([], [], linestyle="none", label=" ")]
        + style_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )

    if max_steps is not None:
        if x_axis == "tokens":
            max_x = step_to_tokens.get(max_steps, max_steps)
        else:
            max_x = max_steps
        ax.set_xlim(left=0, right=max_x)

    apply_token_formatter(ax, x_axis)

    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics="accuracy",
        accuracy_mode=accuracy_mode,
    )
    fname = f"{base_prefix}_evolution_overlap_bin_copying_rates{suffix}"
    os.makedirs(figures_path, exist_ok=True)
    save_figure(fig, os.path.join(figures_path, fname), strip_titles=strip_titles)
    print(
        "Saved overlap-bin copying-rates evolution plot to "
        f"{os.path.join(figures_path, fname)}"
    )


def plot_wlt_overlap_bin_copying_rates_split_evolution(
    run_name,
    figures_path="figures",
    exclude_langs=None,
    max_steps=None,
    x_axis="steps",
    mode="expl-training",
    accuracy_mode="original",
    strip_titles=True,
):
    """Plot source/context copying rates in separate panels by overlap bin."""
    collated_file = os.path.join(
        CACHE_DIR,
        get_collate_name_overlap_bin_accuracy(
            run_name=run_name,
            exclude_langs=exclude_langs,
            accuracy_mode=accuracy_mode,
        ),
    )
    if not os.path.exists(collated_file):
        print(f"Overlap-bin collation not found at {collated_file}, skipping plot.")
        return

    with open(collated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = [s["step"] for s in data.get("steps", [])]
    if not steps:
        print("No overlap-bin copying-rate data to plot.")
        return

    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )
    x_vals = get_x_values(steps, step_to_tokens, x_axis)

    source_curves = data.get("source_word_copying_curves", {})
    context_curves = data.get("context_copying_curves", {})
    bin_order = data.get(
        "bin_priority",
        ["identical_subtokens", "partial_subtoken_overlap", "no_subtoken_overlap"],
    )

    label_map = {
        "identical_subtokens": "full (copy)",
        "partial_subtoken_overlap": "partial",
        "no_subtoken_overlap": "none",
    }
    color_map = {
        "identical_subtokens": OUTPUT_CLASSIFICATION_COLORS.get(
            "translation_exact_subtoken_copy", "#E69F00"
        ),
        "partial_subtoken_overlap": OUTPUT_CLASSIFICATION_COLORS.get(
            "translation_non_exact_subtoken_copy", "#009E73"
        ),
        "no_subtoken_overlap": OUTPUT_CLASSIFICATION_COLORS.get(
            "translation_no_subtoken_overlap", "#D55E00"
        ),
    }

    def _plot_single(curves, metric_label, metric_slug):
        prepare_plotting()
        fig, ax = plt.subplots(figsize=(5, 4))

        for bin_name in bin_order:
            vals = np.array(curves.get(bin_name, []), dtype=float)
            if len(vals) == len(steps):
                ax.plot(
                    x_vals,
                    vals,
                    label=label_map.get(bin_name, bin_name),
                    color=color_map.get(bin_name, "#666666"),
                    linewidth=2.5,
                    alpha=0.9,
                    linestyle="-",
                )

        ax.set_xlim(left=min(x_vals), right=max(x_vals))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Fraction of Outputs")
        ax.set_xlabel(get_x_label(x_axis))
        ax.set_title(f"Copying error mode: {metric_label}")

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=color_map.get(bin_name, "#666666"),
                linewidth=2.6,
                linestyle="-",
                label=label_map.get(bin_name, bin_name),
            )
            for bin_name in bin_order
        ]
 

        if max_steps is not None:
            if x_axis == "tokens":
                max_x = step_to_tokens.get(max_steps, max_steps)
            else:
                max_x = max_steps
            ax.set_xlim(left=0, right=max_x)

        apply_token_formatter(ax, x_axis)

        base_prefix, suffix = get_plot_suffix(
            run_name,
            exclude_langs=exclude_langs,
            x_axis=x_axis,
            metrics="accuracy",
            accuracy_mode=accuracy_mode,
        )
        fname = (
            f"{base_prefix}_evolution_overlap_bin_copying_rates_"
            f"{metric_slug}{suffix}"
        )
        os.makedirs(figures_path, exist_ok=True)
        save_figure(fig, os.path.join(figures_path, fname+"_no_legend"), strip_titles=strip_titles)
        
        ax.legend(
            handles=legend_handles,
            title="Token Overlap",
            loc="upper right",
            frameon=True,
        )
        
        save_figure(fig, os.path.join(figures_path, fname), strip_titles=strip_titles)

        print(
            "Saved overlap-bin copying-rates split evolution plot to "
            f"{os.path.join(figures_path, fname)}"
        )

    _plot_single(source_curves, "source word copying", "source")
    _plot_single(context_curves, "context copying", "context")


def plot_wlt_output_classification_repetition_stacked(
    run_name,
    figures_path="figures",
    exclude_langs=None,
    max_steps=None,
    x_axis="steps",
    mode="expl-training",
    accuracy_mode="original",
    strip_titles=True,
):
    """Plot stacked output composition for repetition tasks (src=tgt only)."""
    class_data = _load_repetition_classification_data(
        run_name=run_name,
        exclude_langs=exclude_langs,
        accuracy_mode=accuracy_mode,
    )
    if class_data is None:
        print("No repetition (src=tgt) output-classification data to plot")
        return

    steps = class_data.get("steps", [])
    if not steps:
        print("No repetition (src=tgt) steps to plot")
        return

    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )
    x_vals = get_x_values(steps, step_to_tokens, x_axis)

    repetition = np.array(class_data.get("copy_correct", []), dtype=float)
    context_copying = np.array(class_data.get("context_copying", []), dtype=float)

    prepare_plotting()
    fig, ax = plt.subplots()

    categories = ["repetition", "context_copying"]
    labels = {
        "repetition": "repetition",
        "context_copying": "context copying",
    }
    colors = {
        "repetition": OUTPUT_CLASSIFICATION_COLORS.get(
            "translation_exact_subtoken_copy", "#E69F00"
        ),
        "context_copying": OUTPUT_CLASSIFICATION_COLORS.get(
            "context_copying", "#56B4E9"
        ),
    }

    plot_series = {
        "repetition": repetition,
        "context_copying": context_copying,
    }
    stackdata = np.array([plot_series[cat] for cat in categories])

    ax.stackplot(
        x_vals,
        stackdata,
        labels=[labels[cat] for cat in categories],
        colors=[colors[cat] for cat in categories],
        alpha=0.8,
    )

    ax.set_xlim(left=min(x_vals), right=max(x_vals))
    ax.set_xlabel(get_x_label(x_axis))
    ax.set_title("Output Classification: Repetition Tasks (src=tgt)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, alpha=0.3, axis="y")

    if max_steps is not None:
        if x_axis == "tokens":
            max_x = step_to_tokens.get(max_steps, max_steps)
        else:
            max_x = max_steps
        ax.set_xlim(left=0, right=max_x)

    apply_token_formatter(ax, x_axis)
    plt.tight_layout()

    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics="accuracy",
        accuracy_mode=accuracy_mode,
    )
    fname = f"{base_prefix}_output_classification_repetition{suffix}"
    os.makedirs(figures_path, exist_ok=True)
    save_figure(fig, os.path.join(figures_path, fname), strip_titles=strip_titles)
    print(
        f"Saved repetition output classification plot to {os.path.join(figures_path, fname)}"
    )


def plot_wlt_output_classification_repetition_lines(
    run_name,
    figures_path="figures",
    exclude_langs=None,
    max_steps=None,
    x_axis="steps",
    mode="expl-training",
    accuracy_mode="original",
    strip_titles=True,
):
    """Plot line chart of output composition for repetition tasks (src=tgt only)."""
    class_data = _load_repetition_classification_data(
        run_name=run_name,
        exclude_langs=exclude_langs,
        accuracy_mode=accuracy_mode,
    )
    if class_data is None:
        print("No repetition (src=tgt) output-classification data to plot")
        return

    steps = class_data.get("steps", [])
    if not steps:
        print("No repetition (src=tgt) steps to plot")
        return

    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )
    x_vals = get_x_values(steps, step_to_tokens, x_axis)

    repetition = np.array(class_data.get("copy_correct", []), dtype=float)
    repetition_std = np.array(class_data.get("copy_correct_std", []), dtype=float)
    context_copying = np.array(class_data.get("context_copying", []), dtype=float)
    context_copying_std = np.array(
        class_data.get("context_copying_std", []), dtype=float
    )

    prepare_plotting()
    fig, ax = plt.subplots()

    labels = {
        "repetition": "Repetition ",
        "context_copying": "Context copying",
    }
    colors = {
        "repetition": OUTPUT_CLASSIFICATION_COLORS.get(
            "translation_exact_subtoken_copy", "#E69F00"
        ),
        "context_copying": OUTPUT_CLASSIFICATION_COLORS.get(
            "context_copying", "#56B4E9"
        ),
    }
    plot_series = {
        "repetition": repetition,
        "context_copying": context_copying,
    }
    plot_stds = {
        "repetition": repetition_std,
        "context_copying": context_copying_std,
    }

    for category in ["repetition", "context_copying"]:
        ax.plot(
            x_vals,
            plot_series[category],
            label=labels[category],
            color=colors[category],
            linewidth=2.5,
            alpha=0.9,
        )

        std_vals = plot_stds[category]
        mean_vals = plot_series[category]
        if len(std_vals) == len(mean_vals):
            ax.fill_between(
                x_vals,
                np.maximum(mean_vals - std_vals, 0),
                np.minimum(mean_vals + std_vals, 1),
                color=colors[category],
                alpha=0.15,
            )

    ax.set_xlim(left=min(x_vals), right=max(x_vals))
    ax.set_xlabel(get_x_label(x_axis))
    ax.set_ylabel("Proportion of outputs")
    ax.set_title("Output Classification: Repetition Tasks (src=tgt)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, alpha=0.3)

    if max_steps is not None:
        if x_axis == "tokens":
            max_x = step_to_tokens.get(max_steps, max_steps)
        else:
            max_x = max_steps
        ax.set_xlim(left=0, right=max_x)

    apply_token_formatter(ax, x_axis)
    plt.tight_layout()

    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics="accuracy",
        accuracy_mode=accuracy_mode,
    )
    fname = f"{base_prefix}_output_classification_repetition_lines{suffix}"
    os.makedirs(figures_path, exist_ok=True)
    save_figure(fig, os.path.join(figures_path, fname), strip_titles=strip_titles)
    print(
        "Saved repetition output classification line plot to "
        f"{os.path.join(figures_path, fname)}"
    )


def plot_wlt_heatmap_at_step(
    step,
    run_name,
    file_name="eval-translation_accuracy.csv",
    figures_path="figures",
    exclude_langs=None,
    accuracy_mode="original",
    strip_titles=True,
):
    """Plot heatmap of WLT accuracies for a specific step."""
    collated_file = os.path.join(
        CACHE_DIR,
        get_collate_name_wlt(
            run_name, file_name, exclude_langs, accuracy_mode=accuracy_mode
        ),
    )

    if not os.path.exists(collated_file):
        print(f"Collated file not found at {collated_file}, skipping.")
        return

    with open(collated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Find data for this step
    step_data = None
    for s in data["steps"]:
        if s["step"] == step:
            step_data = s
            break

    if step_data is None:
        print(f"No data found for step {step}, skipping.")
        return

    print(f"Plotting WLT heatmap for step {step}")

    # Get matrix directly
    matrix = np.array(step_data["accuracy_matrix"])
    lang_labels = [LANG_NAME_EN[lang] for lang in data["languages"]]

    # Create heatmap

    prepare_plotting()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    # Detect normalized-probability separately from plain probability
    is_prob_norm = "probability_normalized" in file_name
    is_prob = ("probability" in file_name) and (not is_prob_norm)

    if is_prob_norm:
        cbar_label = "Translation Probability (normalized)"
    else:
        cbar_label = "Translation Probability" if is_prob else "Translation Accuracy"

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=lang_labels,
        yticklabels=lang_labels,
                annot_kws={"fontsize": 15},

        vmin=0,
        vmax=1,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )

    ax.set_xlabel("Target Language")
    ax.set_ylabel("Source Language")

    # Get standard suffix components - pass file_name as metrics to extract prob type
    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        metrics=file_name,
        accuracy_mode=accuracy_mode,
    )

    path = os.path.join(figures_path, f"{base_prefix}_heatmap_step{step}{suffix}")
    save_figure(fig, path, strip_titles=strip_titles)
    print(f"Saved WLT heatmap to {path}")


def _calculate_joint_average(data, n_langs):
    """Helper to calculate joint average across all non-diagonal entries."""
    all_steps = []
    joint_avg = []
    for step_data in data["steps"]:
        all_steps.append(step_data["step"])
        matrix = np.array(step_data["accuracy_matrix"])

        # Average all non-diagonal entries
        total = sum(
            matrix[i][j] for i in range(n_langs) for j in range(n_langs) if i != j
        )
        count = n_langs * (n_langs - 1)
        joint_avg.append(total / count if count > 0 else 0)

    return all_steps, joint_avg


def plot_wlt_per_language_aggregated(
    run_name,
    direction="to",
    file_name="eval-translation_accuracy.csv",
    figures_path="figures",
    exclude_langs=None,
    max_steps=None,
    x_axis="steps",
    mode="expl-training",
    accuracy_mode="original",
    strip_titles=True,
):
    """Plot aggregated view showing languages (average across source/target languages)."""
    collated_file = os.path.join(
        CACHE_DIR,
        get_collate_name_wlt(
            run_name, file_name, exclude_langs, accuracy_mode=accuracy_mode
        ),
    )

    if not os.path.exists(collated_file):
        print(f"Collated file not found at {collated_file}, skipping.")
        return

    with open(collated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    prepare_plotting()
    color_map = get_color_lang_mapping()
    langs = data["languages"]
    n_langs = len(langs)

    # Load token mapping if x_axis is tokens
    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )

    print(f"Plotting average accuracy {direction} each language")

    fig, ax = plt.subplots(figsize=(5,3.5))

   

    # Detect normalized probability vs plain probability
    is_prob_norm = "probability_normalized" in file_name
    is_prob = ("probability" in file_name) and (not is_prob_norm)

    # Calculate joint average
    all_steps, joint_avg = _calculate_joint_average(data, n_langs)

    # Transpose matrix if direction is "from"
    for step_data in data["steps"]:
        matrix = np.array(step_data["accuracy_matrix"])
        if direction == "from":
            matrix = matrix.T
        step_data["_matrix"] = matrix

    # Plot lines for all languages
    for lang_idx, lang in enumerate(langs):
        steps = []
        avg_accuracies = []
        std_accuracies = []

        for step_data in data["steps"]:
            steps.append(step_data["step"])
            matrix = step_data["_matrix"]

            # Collect all non-diagonal entries to this language
            values = [matrix[i][lang_idx] for i in range(n_langs) if i != lang_idx]
            avg_accuracies.append(np.mean(values))
            std_accuracies.append(np.std(values))

        avg_accuracies = np.array(avg_accuracies)
        std_accuracies = np.array(std_accuracies)

        x_vals = get_x_values(steps, step_to_tokens, x_axis)
        ax.plot(
            x_vals,
            avg_accuracies,
            label=LANG_NAME_EN[lang],
            color=color_map[lang],
            linewidth=2,
        )
        ax.fill_between(
            x_vals,
            np.maximum(avg_accuracies - std_accuracies, 0),
            np.minimum(avg_accuracies + std_accuracies, 1),
            color=color_map[lang],
            alpha=0.15,
            linewidth=0.5,  # Remove the default border of the fill
            zorder=0,
        )

    # Plot joint average
    x_vals_joint = get_x_values(all_steps, step_to_tokens, x_axis)
    ax.plot(x_vals_joint, joint_avg, label="Average", color="black", linewidth=3)

    ax.set_xlabel(get_x_label(x_axis))
    ylabel = (
        "Translation Probability (normalized)"
        if is_prob_norm
        else ("Translation Probability" if is_prob else "Accuracy")
    )
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=min(x_vals), right=max(x_vals))

    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    if max_steps is not None:
        if x_axis == "tokens":
            # Convert the max_steps (in steps) to the corresponding token value for the x-axis limit
            right_limit = get_x_values([max_steps], step_to_tokens, x_axis)[0]
        else:
            right_limit = max_steps
        ax.set_xlim(right=right_limit)


    print("plotting line")
    if x_axis == "tokens":
            milestone_tokens = 11_000_000_000

            ax.axvline(
                milestone_tokens,
                color="black",
                zorder=5,
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )
       
            ax.text(
                2_000_000_000,
                0.95,
                "Phase I",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=16,
                color="#2B2B2B",
            )
      
            ax.text(
                20_000_000_000,
                0.95,
                "Phase II",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=16,
                color="#2B2B2B",
            )
    apply_token_formatter(ax, x_axis)

    # Get standard suffix components - pass file_name as metrics to extract prob type
    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics=file_name,
        accuracy_mode=accuracy_mode,
    )

    path = os.path.join(
        figures_path,
        f"{base_prefix}_evolution_{direction}_all_languages{suffix}",
    )
    save_figure(fig, path, strip_titles=strip_titles)
    print(f"Saved {direction.upper()} all languages evolution plot to {path}")


def plot_wlt_copying_evolution(
    run_name,
    file_name="eval-translation_accuracy.csv",
    figures_path="figures",
    exclude_langs=None,
    max_steps=None,
    x_axis="steps",
    mode="expl-training",
    accuracy_mode="original",
    strip_titles=True,
):
    """Plot copying/self-translation accuracy over time (diagonal entries)."""
    collated_file = os.path.join(
        CACHE_DIR,
        get_collate_name_wlt(
            run_name, file_name, exclude_langs, accuracy_mode=accuracy_mode
        ),
    )

    if not os.path.exists(collated_file):
        print(f"Collated file not found at {collated_file}, skipping.")
        return

    with open(collated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Plotting repetition accuracy evolution")

    color_map = get_color_lang_mapping()
    prepare_plotting()
    langs = data["languages"]
    n_langs = len(langs)

    # Load token mapping if x_axis is tokens
    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )

    fig, ax = plt.subplots(figsize=(5, 3.5))
    # Detect normalized probability vs plain probability
    is_prob_norm = "probability_normalized" in file_name
    is_prob = ("probability" in file_name) and (not is_prob_norm)

    # Calculate joint average of diagonal entries
    all_steps = []
    joint_avg = []
    for step_data in data["steps"]:
        all_steps.append(step_data["step"])
        matrix = np.array(step_data["accuracy_matrix"])

        # Average diagonal entries
        total = sum(matrix[i][i] for i in range(n_langs))
        joint_avg.append(total / n_langs)

    for lang_idx, lang in enumerate(langs):
        steps = []
        accuracies = []

        for step_data in data["steps"]:
            steps.append(step_data["step"])
            matrix = np.array(step_data["accuracy_matrix"])
            # Diagonal entry: lang to itself
            accuracies.append(matrix[lang_idx][lang_idx])

        x_vals = get_x_values(steps, step_to_tokens, x_axis)
        ax.plot(
            x_vals,
            accuracies,
            label=LANG_NAME_EN[lang],
            color=color_map[lang],
            linewidth=2,
        )

    # Plot joint average
    x_vals_joint = get_x_values(all_steps, step_to_tokens, x_axis)
    ax.plot(x_vals_joint, joint_avg, label="Average", color="black", linewidth=3)

    ax.set_xlabel(get_x_label(x_axis))
    ylabel = (
        "Probability (normalized)"
        if is_prob_norm
        else ("Probability" if is_prob else "Accuracy")
    )
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=min(x_vals), right=max(x_vals))
    ax.set_ylim(bottom=0, top=1)
    if x_axis == "tokens":
        milestone_tokens = 11_000_000_000
        ax.axvline(
            milestone_tokens,
            color="black",
            zorder=5,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
    ax.grid(True, alpha=0.3)
    if max_steps is not None:
        if x_axis == "tokens" and step_to_tokens:
            # Convert max_steps (in steps) to corresponding token value for x-axis limit
            max_x = step_to_tokens.get(max_steps, max_steps)
            ax.set_xlim(right=max_x)
        else:
            ax.set_xlim(right=max_steps)
    apply_token_formatter(ax, x_axis)

    # Get standard suffix components - pass file_name as metrics to extract prob type
    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics=file_name,
        accuracy_mode=accuracy_mode,
    )

    path = os.path.join(figures_path, f"{base_prefix}_evolution_copying{suffix}")
    save_figure(fig, path, strip_titles=strip_titles)
    print(f"Saved copying evolution plot to {path}")


def plot_wlt_symmetry_evolution(
    run_name,
    file_name="eval-translation_accuracy.csv",
    figures_path="figures",
    exclude_langs=None,
    max_steps=None,
    absolute=True,
    x_axis="steps",
    mode="expl-training",
    accuracy_mode="original",
    strip_titles=True,
):
    """Plot symmetry/asymmetry metrics over time."""
    collated_file = os.path.join(
        CACHE_DIR,
        get_collate_name_wlt(
            run_name, file_name, exclude_langs, accuracy_mode=accuracy_mode
        ),
    )

    if not os.path.exists(collated_file):
        print(f"Collated file not found at {collated_file}, skipping.")
        return

    with open(collated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    color_map = get_color_lang_mapping()
    prepare_plotting()
    langs = data["languages"]
    n_langs = len(langs)

    is_prob_norm = "probability_normalized" in file_name
    is_prob = ("probability" in file_name) and (not is_prob_norm)

    # Load token mapping if x_axis is tokens
    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )

    steps = []
    per_lang_mean = {lang: [] for lang in langs}
    per_lang_std = {lang: [] for lang in langs}

    for step_data in data["steps"]:
        steps.append(step_data["step"])
        M = np.array(step_data["accuracy_matrix"]).astype(float)

        diff = M - M.T
        if absolute:
            diff = np.abs(diff)
        for k in range(n_langs):
            if n_langs > 1:
                vals = np.delete(diff[k, :], k)
                mean_k = np.mean(vals)
                std_k = np.std(vals)
            else:
                mean_k = 0.0
                std_k = 0.0
            per_lang_mean[langs[k]].append(mean_k)
            per_lang_std[langs[k]].append(std_k)

    fig, ax = plt.subplots()
    all_means = []
    x_vals = get_x_values(steps, step_to_tokens, x_axis)
    for lang in langs:
        mean_arr = np.array(per_lang_mean[lang])
        std_arr = np.array(per_lang_std[lang])
        all_means.append(mean_arr)

        ax.plot(
            x_vals,
            mean_arr,
            label=LANG_NAME_EN[lang],
            color=color_map[lang],
            linewidth=1.5,
        )
        if absolute:
            lower = np.maximum(mean_arr - std_arr, 0)
            upper = np.minimum(mean_arr + std_arr, 1)
        else:
            lower = mean_arr - std_arr
            upper = mean_arr + std_arr
        ax.fill_between(x_vals, lower, upper, color=color_map[lang], alpha=0.08)
    plt.xlim(left=min(x_vals), right=max(x_vals))
    # Plot average across languages for the absolute case, because otherwise it cancels to zero
    if absolute and len(all_means) > 0:
        avg_across_langs = np.mean(np.vstack(all_means), axis=0)
        ax.plot(
            x_vals,
            avg_across_langs,
            label="Average (all languages)",
            color="black",
            linewidth=3,
        )

    ax.set_xlabel(get_x_label(x_axis))
    if not absolute:
        ylabel_base = "Symmetry"
    else:
        ylabel_base = "Absolute Symmetry"
    if is_prob_norm:
        ax.set_ylabel(f"{ylabel_base} of Normalized Probability")
    elif is_prob:
        ax.set_ylabel(f"{ylabel_base} of Probability")
    else:
        ax.set_ylabel(f"{ylabel_base} of Accuracy")

    ax.legend(title="Languages", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    if max_steps is not None:
        if x_axis == "tokens":
            token_mapping = _load_token_mapping_for_plotting(run_name)
            max_tokens = token_mapping.get(max_steps, max_steps)
            ax.set_xlim(right=max_tokens)
        else:
            ax.set_xlim(right=max_steps)
    apply_token_formatter(ax, x_axis)

    # Get standard suffix components - pass file_name as metrics to extract prob type
    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics=file_name,
        accuracy_mode=accuracy_mode,
    )

    file_suffix = "_signed" if (not absolute) else ""

    path = os.path.join(
        figures_path, f"{base_prefix}_evolution_symmetry{file_suffix}{suffix}"
    )
    save_figure(fig, path, strip_titles=strip_titles)
    print(f"Saved symmetry evolution plot to {path}")


def parse_wlt_args():
    parser = argparse.ArgumentParser(
        description="Plot WLT (Word-Level Translation) metrics."
    )
    parser.add_argument(
        "--figures_path",
        type=str,
        default="figures",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Name of the run to plot results for.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["expl-training"],
        default="expl-training",
        help="Training mode.",
    )
    parser.add_argument(
        "--plot_steps",
        type=str,
        default=None,
        help="Comma-separated list of steps to plot heatmaps (e.g., '5000,10000').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of collated results.",
    )
    parser.add_argument(
        "--single_lang",
        type=str,
        default=None,
        help="With --plot_output_classification: one language (to/from plots) or two (one pair), e.g. 'de' or 'de,en'.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Plot all aggregated language views (TO and FROM for all {len(LANGS)} languages).",
    )
    parser.add_argument(
        "--exclude_lang",
        type=str,
        default=None,
        help="Comma-separated list of languages to exclude from analysis (e.g., 'id' for Indonesian).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="If set, restrict plots to steps up to this value (x-axis limit).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        choices=["accuracy", "probability", "probability_normalized"],
        default="accuracy",
        help=(
            "Metric type to plot. Default: accuracy. For probability, uses "
            "eval-translation_probability.csv. For normalized "
            "probability, use 'probability_normalized' which expects files "
            "ending with '_probability_normalized.csv'."
        ),
    )
    parser.add_argument(
        "--symmetry",
        action="store_true",
        help="If set, generate symmetry/asymmetry evolution plots.",
    )
    parser.add_argument(
        "--x_axis",
        type=str,
        choices=["steps", "tokens"],
        default="steps",
        help="X-axis for time-series plots: steps or tokens processed.",
    )
    parser.add_argument(
        "--plot_output_classification",
        action="store_true",
        help="Plot stacked area chart of output classifications (correct, copy, wrong language, other).",
    )
    parser.add_argument(
        "--hide_concept_wrong_lang",
        action="store_true",
        help="Hide 'concept wrong language' category from output classification plots.",
    )
    parser.add_argument(
        "--translation_success_composition",
        action="store_true",
        help=(
            "In output-classification plots, replace correct/copy-correct with "
            "three translation-accuracy overlap buckets: full, partial, and none."
        ),
    )
    parser.add_argument(
        "--accuracy_mode",
        default="original",
        choices=["original", "ci"],
        help="Accuracy definition used when recomputing from predictions.",
    )
    parser.add_argument(
        "--keep_titles",
        action="store_true",
        help="Keep plot titles when saving figures (by default, titles are removed for paper plots).",
    )
    args = parser.parse_args()

    # Parse comma-separated values
    args.plot_steps = (
        [int(s.strip()) for s in args.plot_steps.split(",")] if args.plot_steps else []
    )

    # Parse single_lang into list of languages
    args.single_langs = (
        [lang.strip() for lang in args.single_lang.split(",")]
        if args.single_lang
        else []
    )

    # Parse exclude_lang into list
    args.exclude_langs = (
        [lang.strip() for lang in args.exclude_lang.split(",")]
        if args.exclude_lang
        else []
    )

    # Validate languages
    for lang in args.single_langs:
        if lang not in LANGS:
            parser.error(f"Unsupported language: {lang}. Must be one of {LANGS}")

    # Validate excluded languages
    for lang in args.exclude_langs:
        if lang not in LANGS:
            parser.error(
                f"Unsupported excluded language: {lang}. Must be one of {LANGS}"
            )

    if args.single_langs and not args.plot_output_classification:
        parser.error("--single_lang only applies to --plot_output_classification")

    # Validate single_langs don't include excluded languages
    for lang in args.single_langs:
        if lang in args.exclude_langs:
            parser.error(
                f"Cannot use {lang} in --single_lang when it's excluded via --exclude_lang"
            )

    return args


if __name__ == "__main__":
    # Usage example:
    # python src/plot_wlt_metrics.py --run_name ours --plot_steps 5000,10000,15000 --all
    # python src/plot_wlt_metrics.py --run_name ours --plot_output_classification --single_lang de
    args = parse_wlt_args()
    os.makedirs(CACHE_DIR, exist_ok=True)

    output_dir = DEFAULT_OUTPUT_DIR
    run_dir = os.path.join(output_dir, args.run_name)
    checkpoints_dir, results_dir = get_checkpoint_results_dirs(run_dir)
    print(f"Results directory: {results_dir}")
    results_dirs = load_results_dirs(run_dir, run_name=args.run_name)

    # Set file name based on selected metric
    base = "eval-translation_"
    if args.metrics == "probability_normalized":
        base += "probability_normalized.csv"
    elif args.metrics == "probability":
        base += "probability.csv"
    else:
        base += "accuracy.csv"
    file_name = base
    subfolder = "wlt_metrics"

    # Create figures directory: figures/{run_name}/wlt_metrics
    model_figures_path = os.path.join(args.figures_path, args.run_name, subfolder)
    os.makedirs(model_figures_path, exist_ok=True)

    # Collate results
    collate_results_wlt(
        results_dirs,
        run_name=args.run_name,
        file_name=file_name,
        force=args.force,
        exclude_langs=args.exclude_langs,
        accuracy_mode=args.accuracy_mode,
    )

    collate_predictions_over_time(
        results_dirs,
        run_name=args.run_name,
        file_prefix="eval-translation_predictions",
        force=args.force,
        exclude_langs=args.exclude_langs,
        metrics=args.metrics,
        accuracy_mode=args.accuracy_mode,
    )

    collate_overlap_bin_accuracy_over_time(
        results_dirs,
        run_name=args.run_name,
        file_prefix="eval-translation_predictions",
        force=args.force,
        exclude_langs=args.exclude_langs,
        mode=args.mode,
        accuracy_mode=args.accuracy_mode,
    )

    # Collate output classifications (needed for repetition context-copying evolution plot).
    from collate_fn import collate_output_classification

    langs = [lang for lang in LANGS if lang not in args.exclude_langs]
    for tgt_lang in langs:
        collate_output_classification(
            results_dirs,
            run_name=args.run_name,
            file_prefix="eval-translation_predictions",
            force=args.force,
            exclude_langs=args.exclude_langs,
            mode=args.mode,
            tgt_lang=tgt_lang,
            accuracy_mode=args.accuracy_mode,
        )

    # Plot heatmaps for specific steps
    for step in args.plot_steps:
        plot_wlt_heatmap_at_step(
            step=step,
            run_name=args.run_name,
            file_name=file_name,
            figures_path=model_figures_path,
            exclude_langs=args.exclude_langs,
            accuracy_mode=args.accuracy_mode,
            strip_titles=not args.keep_titles,
        )

    # Plot copying accuracy evolution
    plot_wlt_copying_evolution(
        run_name=args.run_name,
        file_name=file_name,
        figures_path=model_figures_path,
        exclude_langs=args.exclude_langs,
        max_steps=args.max_steps,
        x_axis=args.x_axis,
        mode=args.mode,
        accuracy_mode=args.accuracy_mode,
        strip_titles=not args.keep_titles,
    )

    output_classification_dir = os.path.join(model_figures_path, "output_classification")
    os.makedirs(output_classification_dir, exist_ok=True)
    plot_wlt_repetition_context_copying_evolution(
        run_name=args.run_name,
        figures_path=output_classification_dir,
        exclude_langs=args.exclude_langs,
        max_steps=args.max_steps,
        x_axis=args.x_axis,
        mode=args.mode,
        accuracy_mode=args.accuracy_mode,
        strip_titles=not args.keep_titles,
    )
    plot_wlt_overlap_bin_accuracy_evolution(
        run_name=args.run_name,
        figures_path=output_classification_dir,
        exclude_langs=args.exclude_langs,
        max_steps=args.max_steps,
        x_axis=args.x_axis,
        mode=args.mode,
        accuracy_mode=args.accuracy_mode,
        strip_titles=not args.keep_titles,
    )
    plot_wlt_overlap_bin_copying_rates_evolution(
        run_name=args.run_name,
        figures_path=output_classification_dir,
        exclude_langs=args.exclude_langs,
        max_steps=args.max_steps,
        x_axis=args.x_axis,
        mode=args.mode,
        accuracy_mode=args.accuracy_mode,
        strip_titles=not args.keep_titles,
    )
    plot_wlt_overlap_bin_copying_rates_split_evolution(
        run_name=args.run_name,
        figures_path=output_classification_dir,
        exclude_langs=args.exclude_langs,
        max_steps=args.max_steps,
        x_axis=args.x_axis,
        mode=args.mode,
        accuracy_mode=args.accuracy_mode,
        strip_titles=not args.keep_titles,
    )

    # Plot symmetry/asymmetry evolution: absolute and signed versions
    if args.symmetry:
        plot_wlt_symmetry_evolution(
            run_name=args.run_name,
            file_name=file_name,
            figures_path=model_figures_path,
            exclude_langs=args.exclude_langs,
            max_steps=args.max_steps,
            absolute=False,
            x_axis=args.x_axis,
            mode=args.mode,
            accuracy_mode=args.accuracy_mode,
            strip_titles=not args.keep_titles,
        )
        plot_wlt_symmetry_evolution(
            run_name=args.run_name,
            file_name=file_name,
            figures_path=model_figures_path,
            exclude_langs=args.exclude_langs,
            max_steps=args.max_steps,
            absolute=True,
            x_axis=args.x_axis,
            mode=args.mode,
            accuracy_mode=args.accuracy_mode,
            strip_titles=not args.keep_titles,
        )

    # Plot per-language aggregated views
    if args.all:
        plot_wlt_per_language_aggregated(
            run_name=args.run_name,
            direction="to",
            file_name=file_name,
            figures_path=model_figures_path,
            exclude_langs=args.exclude_langs,
            max_steps=args.max_steps,
            x_axis=args.x_axis,
            mode=args.mode,
            accuracy_mode=args.accuracy_mode,
            strip_titles=not args.keep_titles,
        )
        plot_wlt_per_language_aggregated(
            run_name=args.run_name,
            direction="from",
            file_name=file_name,
            figures_path=model_figures_path,
            exclude_langs=args.exclude_langs,
            max_steps=args.max_steps,
            x_axis=args.x_axis,
            mode=args.mode,
            accuracy_mode=args.accuracy_mode,
            strip_titles=not args.keep_titles,
        )

    # Plot output classification stacked areas
    if args.plot_output_classification:
        print("\n=== Plotting Output Classification ===")
        output_classification_dir = os.path.join(
            model_figures_path, "output_classification"
        )
        os.makedirs(output_classification_dir, exist_ok=True)

        # Derive available languages from LANGS minus exclude_langs
        available_langs = (
            [lang for lang in LANGS if lang not in args.exclude_langs]
            if args.exclude_langs
            else LANGS
        )

        # Build list of aggregations to plot
        aggregations = []
        if args.single_langs:
            if len(args.single_langs) == 2:
                src_lang, tgt_lang = args.single_langs[0], args.single_langs[1]
                aggregations = [
                    f"pair:{src_lang}_{tgt_lang}",
                    f"pair:{tgt_lang}_{src_lang}",
                ]
            elif len(args.single_langs) == 1:
                single_lang = args.single_langs[0]
                # Pairs with this language
                for other_lang in available_langs:
                    if other_lang != single_lang:
                        aggregations.append(f"pair:{single_lang}_{other_lang}")
                        aggregations.append(f"pair:{other_lang}_{single_lang}")
                # Aggregated to/from
                aggregations.extend([f"to:{single_lang}", f"from:{single_lang}"])
            else:
                print(
                    f"Warning: --single_lang expects 1 or 2 languages, got {len(args.single_langs)}"
                )
        else:
            aggregations = ["all_pairs"]

        # Plot each aggregation
        plot_kwargs = {
            "run_name": args.run_name,
            "file_prefix": "eval-translation_predictions",
            "figures_path": output_classification_dir,
            "exclude_langs": args.exclude_langs,
            "max_steps": args.max_steps,
            "x_axis": args.x_axis,
            "mode": args.mode,
            "hide_concept_wrong_lang": args.hide_concept_wrong_lang,
            "translation_success_composition": args.translation_success_composition,
            "accuracy_mode": args.accuracy_mode,
            "strip_titles": not args.keep_titles,
        }

        for aggregation in aggregations:
            plot_wlt_output_classification_stacked(
                aggregation=aggregation, **plot_kwargs
            )
            plot_wlt_output_classification_lines(aggregation=aggregation, **plot_kwargs)

        plot_wlt_output_classification_repetition_stacked(
            run_name=args.run_name,
            figures_path=output_classification_dir,
            exclude_langs=args.exclude_langs,
            max_steps=args.max_steps,
            x_axis=args.x_axis,
            mode=args.mode,
            accuracy_mode=args.accuracy_mode,
            strip_titles=not args.keep_titles,
        )
        plot_wlt_output_classification_repetition_lines(
            run_name=args.run_name,
            figures_path=output_classification_dir,
            exclude_langs=args.exclude_langs,
            max_steps=args.max_steps,
            x_axis=args.x_axis,
            mode=args.mode,
            accuracy_mode=args.accuracy_mode,
            strip_titles=not args.keep_titles,
        )

        # Plot copying prevalence heatmaps for all language pairs
        print("\n=== Plotting Output Classification Copying Heatmaps ===")
        plot_output_classification_copying_heatmaps(
            run_name=args.run_name,
            figures_path=output_classification_dir,
            exclude_langs=args.exclude_langs,
            accuracy_mode=args.accuracy_mode,
            x_axis=args.x_axis,
            strip_titles=not args.keep_titles,
        )
