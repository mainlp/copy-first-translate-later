import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D

from constants import DEFAULT_OUTPUT_DIR, DEFAULT_RUN_NAME, LANG_NAME_EN, LANGS, CACHE_DIR
from utils import get_checkpoint_results_dirs
from plot_utils import (
    prepare_plotting,
    load_results_dirs,
    save_figure,
    get_lang_from_logit_lens_file_name,
    _load_token_mapping_for_plotting,
    get_x_values,
    get_x_label,
    get_plot_suffix,
    apply_token_formatter,
)

from collate_fn import (
    collate_results_logit_lens,
    get_collate_name_logit_lens,
)


LANG_SCRIPT = {
    "de": "Latn",
    "en": "Latn",
    "es": "Latn",
    "fr": "Latn",
    "id": "Latn",
    "it": "Latn",
    "ja": "Jpan",
    "pt": "Latn",
    "zh": "Hani",
}

STEP_GROUP_ORDER = [
    "target",
    "source",
    "english",
    "other_cjk",
    "other_latin",
]

STEP_GROUP_LABELS = {
    "target": "Target language",
    "source": "Source language",
    "english": "English",
    "other_cjk": "Other (CJK)",
    "other_latin": "Other (Latin)",
}

STEP_GROUP_COLORS = {
    "target": "#56B4E9",
    "source": "#E69F00",
    "english": "#009E73",
    "other_cjk": "#CC79A7",
    "other_latin": "#D55E00",
}

OTHER_GROUP_COLOR = "#005A5B"

STEP_GROUP_ORDER_COLLAPSED = [
    "target",
    "source",
    "english",
    "other",
]

STEP_GROUP_LABELS_COLLAPSED = {
    "target": "Target",# language",
    "source": "Source",# language",
    "english": "English",
    "other": "Other",
}

STEP_GROUP_COLORS_COLLAPSED = {
    "target": "#56B4E9",
    "source": "#E69F00",
    "english": "#009E73",
    "other": OTHER_GROUP_COLOR,
}

STEP_GROUP_ORDER_RELATIVE = [
    "target",
    "source",
    "english",
    "other_same_as_target",
    "other_diff_from_target",
]

STEP_GROUP_LABELS_RELATIVE = {
    "target": "Target language",
    "source": "Source language",
    "english": "English",
    "other_same_as_target": "Same script group",
    "other_diff_from_target": "Different script group",
}

STEP_GROUP_COLORS_RELATIVE = {
    "target": "#56B4E9",
    "source": "#E69F00",
    "english": "#009E73",
    "other_same_as_target": "#CC79A7",
    "other_diff_from_target": "#D55E00",
}

STEP_GROUP_ORDER_JACCARD = [
    "identical_subtokens",
    "partial_subtoken_overlap",
    "no_subtoken_overlap",
]

STEP_GROUP_LABELS_JACCARD = {
    "identical_subtokens": "full (copy)",
    "partial_subtoken_overlap": "partial",
    "no_subtoken_overlap": "none",
}

STEP_GROUP_COLORS_JACCARD = {
    "identical_subtokens": "#E69F00",
    "partial_subtoken_overlap": "#009E73",
    "no_subtoken_overlap": "#D55E00",
}
JACCARD_TARGET_FULL_KEY = "target_full"
JACCARD_TARGET_FULL_LABEL = "target (full avg)"

CJK_LANGS = {"ja", "zh"}
LATIN_LANGS = {"de", "en", "es", "fr", "id", "it", "pt"}

VALID_GROUPING_VARIANTS = [
    "cjk_latin",
    "relative_script",
    "collapsed_other",
    "jaccard",
]

VALID_GROUPED_MARGIN_REFERENCES = [
    "auto",
    "target",
    "no_subtoken_overlap",
]


def _resolve_grouping_variants(groupings_arg, include_jaccard_groups):
    """Resolve grouping variants to use in grouped plotting/margin routines."""
    if not groupings_arg:
        variants = ["cjk_latin", "relative_script"]
        if include_jaccard_groups:
            variants.append("jaccard")
        return variants

    requested = [
        item.strip().lower() for item in groupings_arg.split(",") if item.strip()
    ]
    if not requested:
        raise ValueError("--groupings must not be empty when provided")

    variants = []
    for item in requested:
        if item == "all":
            for variant in VALID_GROUPING_VARIANTS:
                if variant not in variants:
                    variants.append(variant)
            continue
        if item not in VALID_GROUPING_VARIANTS:
            raise ValueError(
                f"Unsupported grouping variant: {item}. Must be one of {VALID_GROUPING_VARIANTS} or 'all'."
            )
        if item not in variants:
            variants.append(item)
    return variants


def _grouping_title_suffix(grouping_variant):
    if grouping_variant == "relative_script":
        return " (script-relative)"
    if grouping_variant == "collapsed_other":
        return " (collapsed other)"
    if grouping_variant == "jaccard":
        return " (Jaccard groups)"
    return ""


def _resolve_grouped_margin_reference(grouping_variant, grouped_margin_reference):
    """Resolve and validate grouped margin reference bucket for a grouping variant."""
    if grouped_margin_reference == "auto":
        return "no_subtoken_overlap" if grouping_variant == "jaccard" else "target"

    if grouping_variant == "jaccard":
        if grouped_margin_reference != "no_subtoken_overlap":
            raise ValueError(
                "For jaccard grouping, grouped margin reference must be 'no_subtoken_overlap' (or 'auto')."
            )
        return grouped_margin_reference

    if grouped_margin_reference != "target":
        raise ValueError(
            f"For {grouping_variant} grouping, grouped margin reference must be 'target' (or 'auto')."
        )
    return grouped_margin_reference


def _stack_and_nanmean(vectors):
    """Pad vectors with NaN to max length and compute nanmean/nanstd over axis 0."""
    if not vectors:
        return None, None

    arrs = [np.asarray(v, dtype=float) for v in vectors if v is not None and len(v) > 0]
    if not arrs:
        return None, None

    max_len = max(len(v) for v in arrs)
    stacked = np.full((len(arrs), max_len), np.nan)
    for idx, vec in enumerate(arrs):
        stacked[idx, : len(vec)] = vec

    return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def _combine_mean_std(mean_vecs, std_vecs=None):
    """Combine per-language means into a group mean and std-of-averages.

    Returns mean(μ_k) and std(μ_k) — the spread of per-language means.
    std_vecs is accepted for signature compatibility but not used here.
    """
    if not mean_vecs:
        return None, None

    mean_arrs = [
        np.asarray(v, dtype=float) for v in mean_vecs if v is not None and len(v) > 0
    ]
    if not mean_arrs:
        return None, None

    max_len = max(len(v) for v in mean_arrs)
    stacked_means = np.full((len(mean_arrs), max_len), np.nan)
    for idx, vec in enumerate(mean_arrs):
        stacked_means[idx, : len(vec)] = vec

    return np.nanmean(stacked_means, axis=0), np.nanstd(stacked_means, axis=0)


def _vectors_to_padded_matrix(vectors):
    """Convert a list of optional vectors into a padded 2D matrix with NaNs."""
    max_len = max((len(v) for v in vectors if v is not None), default=0)
    if max_len == 0:
        return None

    matrix = np.full((len(vectors), max_len), np.nan)
    for idx, vec in enumerate(vectors):
        if vec is not None:
            arr = np.asarray(vec, dtype=float)
            matrix[idx, : len(arr)] = arr
    return matrix


def _get_group_langs_for_pair(src_lang, tgt_lang, grouping_variant="cjk_latin"):
    """Return teacher-forcing language buckets for one src→tgt pair."""
    used = {src_lang, tgt_lang, "en"}
    remaining = [lang for lang in LANGS if lang not in used]

    if grouping_variant == "relative_script":
        tgt_script = LANG_SCRIPT.get(tgt_lang, "Unknown")
        same_as_target = [
            lang for lang in remaining if LANG_SCRIPT.get(lang, "Unknown") == tgt_script
        ]
        diff_from_target = [lang for lang in remaining if lang not in same_as_target]

        return {
            "target": [tgt_lang],
            "source": [src_lang],
            "english": ["en"] if "en" not in {src_lang, tgt_lang} else [],
            "other_same_as_target": same_as_target,
            "other_diff_from_target": diff_from_target,
        }

    if grouping_variant == "collapsed_other":
        return {
            "target": [tgt_lang],
            "source": [src_lang],
            "english": ["en"] if "en" not in {src_lang, tgt_lang} else [],
            "other": remaining,
        }

    other_cjk = [lang for lang in remaining if lang in CJK_LANGS]
    other_latin = [lang for lang in remaining if lang in LATIN_LANGS]

    return {
        "target": [tgt_lang],
        "source": [src_lang],
        "english": ["en"] if "en" not in {src_lang, tgt_lang} else [],
        "other_cjk": other_cjk,
        "other_latin": other_latin,
    }


def plot_per_step_grouped_aggregates(
    step,
    run_name,
    figures_path="figures",
    score_type="scores",
    aggregate_probs=False,
    pairs=None,
    scope_label="all",
    title_scope="All directed pairs",
    exclude_langs=None,
    strip_titles=True,
    grouping_variant="cjk_latin",
    jaccard_target_only=False,
):
    """
    Plot per-step grouped teacher-forcing aggregates over translation pairs.

    Grouping per pair:
    - target language
    - source language
    - English
    - other languages with same script as target
    - other languages with other script than target

    Then aggregate those grouped vectors over selected translation pairs.
    """
    pairs = pairs or []
    if not pairs:
        print(f"No pairs found for per-step grouped aggregation (scope={scope_label}).")
        return

    if grouping_variant == "relative_script":
        step_group_order = STEP_GROUP_ORDER_RELATIVE
        step_group_labels = STEP_GROUP_LABELS_RELATIVE
        step_group_colors = STEP_GROUP_COLORS_RELATIVE
    elif grouping_variant == "collapsed_other":
        step_group_order = STEP_GROUP_ORDER_COLLAPSED
        step_group_labels = STEP_GROUP_LABELS_COLLAPSED
        step_group_colors = STEP_GROUP_COLORS_COLLAPSED
    elif grouping_variant == "jaccard":
        step_group_order = STEP_GROUP_ORDER_JACCARD
        step_group_labels = STEP_GROUP_LABELS_JACCARD
        step_group_colors = STEP_GROUP_COLORS_JACCARD
    else:
        step_group_order = STEP_GROUP_ORDER
        step_group_labels = STEP_GROUP_LABELS
        step_group_colors = STEP_GROUP_COLORS

    grouped_values = {group: [] for group in step_group_order}
    grouped_stds = {group: [] for group in step_group_order}

    # Cache collated files per target language to avoid repeated IO
    collated_by_tgt = {}

    for src, tgt in pairs:
        file_name = f"eval-logit_lens_{tgt}.jsonl"
        if tgt not in collated_by_tgt:
            collated_file = os.path.join(
                CACHE_DIR,
                get_collate_name_logit_lens(
                    run_name, file_name, score_type, aggregate_probs
                ),
            )
            if not os.path.exists(collated_file):
                print(f"  Missing collated file for target {tgt}: {collated_file}")
                collated_by_tgt[tgt] = None
            else:
                with open(collated_file, "r", encoding="utf-8") as f:
                    collated_by_tgt[tgt] = json.load(f)

        data = collated_by_tgt.get(tgt)
        if data is None:
            continue

        results = data.get("results", {}).get(f"{src}_to", [])
        if not results:
            continue

        result_step = next((r for r in results if r.get("step") == step), None)
        if result_step is None:
            continue

        if grouping_variant == "jaccard":
            # Use pre-collated per-tf-language Jaccard bucket means, then aggregate.
            jaccard_stats = result_step.get("similarity_group_stats", {})
            jaccard_counts = result_step.get("similarity_group_counts", {})
            tf_langs = [tgt] if jaccard_target_only else LANGS
            for tf_lang in tf_langs:
                tf_stats = jaccard_stats.get(tf_lang, {})
                tf_counts = jaccard_counts.get(tf_lang, {})
                for group_name in step_group_order:
                    if tf_counts.get(group_name, 0) <= 0:
                        continue
                    group_mean = tf_stats.get(group_name, {}).get("mean")
                    group_std = tf_stats.get(group_name, {}).get("std")
                    if group_mean is None:
                        continue
                    grouped_values[group_name].append(group_mean)
                    grouped_stds[group_name].append(group_std)
        else:
            mean_dict = result_step.get("mean", {})
            std_dict = result_step.get("std", {})
            group_langs = _get_group_langs_for_pair(
                src, tgt, grouping_variant=grouping_variant
            )

            for group_name in step_group_order:
                langs = group_langs[group_name]
                for lang in langs:
                    mean_vec = mean_dict.get(lang)
                    if mean_vec is None:
                        continue
                    grouped_values[group_name].append(mean_vec)
                    grouped_stds[group_name].append(
                        std_dict.get(lang)
                    )  # None if absent

    prepare_plotting()
    if grouping_variant=="collapsed_other":
        fig, ax = plt.subplots(figsize=(4, 4))
    elif jaccard_target_only:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig, ax = plt.subplots(figsize=(8, 4))

    plotted_any = False
    for group_name in step_group_order:
        mean_curve, std_curve = _combine_mean_std(
            grouped_values[group_name], grouped_stds[group_name]
        )
        if mean_curve is None:
            continue
        plotted_any = True
        layers = np.arange(len(mean_curve))
        color = step_group_colors[group_name]
        ax.plot(
            layers[:-1],
            mean_curve[1:],
            label=step_group_labels[group_name],
            linewidth=2,
            color=color,
        )

        if (
            not grouping_variant == "collapsed_other"
            and std_curve is not None
            and len(std_curve) == len(layers)
        ):
            ax.fill_between(
                layers[:-1],
                mean_curve[1:] - std_curve[1:],
                mean_curve[1:] + std_curve[1:],
                color=color,
                alpha=0.12,
            )

    contribution_summary = ", ".join(
        [f"{group}={len(grouped_values[group])}" for group in step_group_order]
    )
    print(
        f"Grouped contributions at step {step} ({scope_label}, {grouping_variant}) [individual lang vectors]: {contribution_summary}"
    )

    if grouping_variant == "collapsed_other":
        plt.axvline(
            9.1, color="black", zorder=0, linestyle="--", linewidth=1, alpha=0.7
        )
        plt.axvline(
            15.1, color="black", zorder=0, linestyle="--", linewidth=1, alpha=0.7
        )
        ax.text(
            4.55,
            0.9,
            "Bottom",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=15,
        )
        ax.text(
            12.1,
            0.9,
            "Interm.",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=15,
        )
        ax.text(
            19.05,
            0.9,
            "Top",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=15,
        )
    else:
        if not aggregate_probs:
            ax.set_ylim(-12.2, 0)
    if not plotted_any:
        print(f"No data available for grouped per-step aggregation at step {step}.")
        plt.close(fig)
        return
    ax.set_xlim(0, 23)
    ax.set_xlabel("Layer")
    ylabel_prefix = "p" if aggregate_probs else "log p"
    ax.set_ylabel(f"{ylabel_prefix}")
    ax.set_title(f"Step {step}: grouped TF aggregates ({title_scope})")
    legend_title = "Token overlap" if grouping_variant == "jaccard" else "Language"
    if grouping_variant == "collapsed_other":
        ax.legend(title=legend_title, loc="upper left")
        #pass
    else:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
    
    ax.grid(True, alpha=0.3)

    metrics_name = "eval-logit_lens_grouped_step_agg"
    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis="steps",
        metrics=metrics_name,
        score_type=score_type,
        aggregate_probs=aggregate_probs,
    )
    mode_suffix = (
        "_tgt_only" if grouping_variant == "jaccard" and jaccard_target_only else ""
    )

    path = os.path.join(
        figures_path,
        f"{base_prefix}_logit_lens_step{step}_grouped_tf_{scope_label}_{grouping_variant}{mode_suffix}{suffix}",
    )
    save_figure(fig, path, strip_titles=strip_titles)
    print(f"Saved grouped per-step aggregation to {path}")


def _compute_grouped_time_averages_over_pairs(
    run_name,
    pairs,
    score_type,
    aggregate_probs,
    grouping_variant,
    jaccard_target_only=False,
):
    """Compute average grouped TF matrices over time for a set of directed pairs."""
    if grouping_variant == "relative_script":
        step_group_order = STEP_GROUP_ORDER_RELATIVE
        step_group_labels = STEP_GROUP_LABELS_RELATIVE
    elif grouping_variant == "collapsed_other":
        step_group_order = STEP_GROUP_ORDER_COLLAPSED
        step_group_labels = STEP_GROUP_LABELS_COLLAPSED
    elif grouping_variant == "jaccard":
        step_group_order = STEP_GROUP_ORDER_JACCARD
        step_group_labels = STEP_GROUP_LABELS_JACCARD
    else:
        step_group_order = STEP_GROUP_ORDER
        step_group_labels = STEP_GROUP_LABELS

    grouped_values = {group: [] for group in step_group_order}
    reference_steps = None
    collated_by_tgt = {}

    for src, tgt in pairs:
        file_name = f"eval-logit_lens_{tgt}.jsonl"
        if tgt not in collated_by_tgt:
            collated_file = os.path.join(
                CACHE_DIR,
                get_collate_name_logit_lens(
                    run_name, file_name, score_type, aggregate_probs
                ),
            )
            if not os.path.exists(collated_file):
                print(f"  Missing collated file for target {tgt}: {collated_file}")
                collated_by_tgt[tgt] = None
            else:
                with open(collated_file, "r", encoding="utf-8") as f:
                    collated_by_tgt[tgt] = json.load(f)

        data = collated_by_tgt.get(tgt)
        if data is None:
            continue

        results = data.get("results", {}).get(f"{src}_to", [])
        if not results:
            continue

        results = sorted(results, key=lambda r: r.get("step", -1))
        pair_steps = []
        grouped_series = {group: [] for group in step_group_order}
        group_langs = None
        if grouping_variant != "jaccard":
            group_langs = _get_group_langs_for_pair(
                src, tgt, grouping_variant=grouping_variant
            )

        for result in results:
            step = result.get("step")
            pair_steps.append(step)

            if grouping_variant == "jaccard":
                jaccard_stats = result.get("similarity_group_stats", {})
                jaccard_counts = result.get("similarity_group_counts", {})
                for group_name in step_group_order:
                    group_means = []
                    tf_langs = [tgt] if jaccard_target_only else LANGS
                    for tf_lang in tf_langs:
                        if jaccard_counts.get(tf_lang, {}).get(group_name, 0) <= 0:
                            continue
                        tf_group_mean = (
                            jaccard_stats.get(tf_lang, {})
                            .get(group_name, {})
                            .get("mean")
                        )
                        if tf_group_mean is not None:
                            group_means.append(tf_group_mean)
                    if not group_means:
                        grouped_series[group_name].append(None)
                        continue
                    group_mean, _ = _stack_and_nanmean(group_means)
                    grouped_series[group_name].append(group_mean)
            else:
                mean_dict = result.get("mean", {})

                for group_name in step_group_order:
                    langs = group_langs[group_name]
                    group_means = [
                        mean_dict.get(lang)
                        for lang in langs
                        if mean_dict.get(lang) is not None
                    ]
                    if not group_means:
                        grouped_series[group_name].append(None)
                        continue
                    group_mean, _ = _stack_and_nanmean(group_means)
                    grouped_series[group_name].append(group_mean)

        if reference_steps is None:
            reference_steps = pair_steps
        elif pair_steps != reference_steps:
            print(
                f"  Skipping {src} -> {tgt}: step grid mismatch for grouped time view"
            )
            continue

        for group_name in step_group_order:
            matrix = _vectors_to_padded_matrix(grouped_series[group_name])
            if matrix is None:
                continue
            grouped_values[group_name].append(matrix)

    if reference_steps is None:
        return None, None, step_group_order, step_group_labels

    avg_by_group = {}
    for group_name in step_group_order:
        if not grouped_values[group_name]:
            avg_by_group[group_name] = None
        else:
            avg_by_group[group_name] = np.nanmean(
                np.stack(grouped_values[group_name], axis=0), axis=0
            )

    return reference_steps, avg_by_group, step_group_order, step_group_labels


def plot_grouped_time_layer_group_plot(
    run_name,
    figures_path="figures",
    x_axis="steps",
    mode="expl-training",
    score_type="scores",
    aggregate_probs=False,
    pairs=None,
    scope_label="all",
    title_scope="All directed pairs",
    exclude_langs=None,
    strip_titles=True,
    grouping_variant="cjk_latin",
    layer_groups=None,
    collapse_other_groups=True,
    jaccard_target_only=False,
):
    """Plot grouped TF over-time lines using layer-band averages.

    Default bands include broad + fine-grained final-layer views.
    """
    pairs = pairs or []
    if not pairs:
        print(f"No pairs found for grouped layer-group plot (scope={scope_label}).")
        return

    if grouping_variant == "relative_script":
        step_group_colors = STEP_GROUP_COLORS_RELATIVE
    elif grouping_variant == "collapsed_other":
        step_group_colors = STEP_GROUP_COLORS_COLLAPSED
    elif grouping_variant == "jaccard":
        step_group_colors = STEP_GROUP_COLORS_JACCARD
    else:
        step_group_colors = STEP_GROUP_COLORS

    if layer_groups is None:
        # Default to non-overlapping bands with finer late-layer split.
        layer_groups = [(1, 10), (11, 16), (17, 24)]
    layer_styles = [":", "--", "-"]

    reference_steps, avg_by_group, step_group_order, step_group_labels = (
        _compute_grouped_time_averages_over_pairs(
            run_name=run_name,
            pairs=pairs,
            score_type=score_type,
            aggregate_probs=aggregate_probs,
            grouping_variant=grouping_variant,
            jaccard_target_only=jaccard_target_only,
        )
    )

    if reference_steps is None:
        print(f"No valid grouped layer-group data available for scope={scope_label}.")
        return

    # Optional compact mode: merge "other_*" buckets into a single "other" group.
    plot_order = list(step_group_order)
    plot_labels = dict(step_group_labels)
    plot_colors = dict(step_group_colors)
    plot_values = dict(avg_by_group)
    if collapse_other_groups and grouping_variant not in {"jaccard", "collapsed_other"}:
        other_groups = [g for g in step_group_order if g.startswith("other")]
        if other_groups:
            other_mats = [
                avg_by_group[g] for g in other_groups if avg_by_group[g] is not None
            ]
            if other_mats:
                plot_values["other"] = np.nanmean(np.stack(other_mats, axis=0), axis=0)
            else:
                plot_values["other"] = None

            plot_order = [g for g in step_group_order if not g.startswith("other")]
            plot_order.append("other")
            plot_labels["other"] = "Other"
            plot_colors["other"] = OTHER_GROUP_COLOR

    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )
    x_vals = get_x_values(reference_steps, step_to_tokens, x_axis)
    prepare_plotting()

    fig, ax = plt.subplots(figsize=(8, 4))
    plotted_any = False
    plotted_groups = []
    plotted_layer_band_indices = []

    for group_name in plot_order:
        matrix = plot_values[group_name]
        if matrix is None or matrix.size == 0 or np.all(np.isnan(matrix)):
            continue

        n_layers = matrix.shape[1]
        for idx, (start_layer, end_layer) in enumerate(layer_groups):
            start = max(0, start_layer)
            end = min(n_layers - 1, end_layer)
            if start > end:
                continue

            series = np.nanmean(matrix[:, start : end + 1], axis=1)
            if np.all(np.isnan(series)):
                continue

            plotted_any = True
            if group_name not in plotted_groups:
                plotted_groups.append(group_name)
            if idx not in plotted_layer_band_indices:
                plotted_layer_band_indices.append(idx)
            ax.plot(
                x_vals,
                series,
                color=plot_colors[group_name],
                linestyle=layer_styles[idx % len(layer_styles)],
                linewidth=2,
                alpha=0.95,
                label=None,
            )

    if not plotted_any:
        print(f"No plottable grouped layer-group lines for scope={scope_label}.")
        plt.close(fig)
        return

    prob_str = "p" if aggregate_probs else "log p"
    ax.set_xlabel(get_x_label(x_axis))
    ax.set_ylabel(f"{prob_str}")  # (grouped TF aggregate)")
    ax.set_xlim(min(x_vals), max(x_vals))
    ax.set_title(
        f"Grouped TF layer group plot over time ({title_scope}, {grouping_variant})"
    )
    ax.grid(True, alpha=0.3)
    apply_token_formatter(ax, x_axis)

    # Single combined legend: language is encoded by color, layer band by linestyle.
    group_handles = [
        Line2D(
            [0],
            [0],
            color=plot_colors[group_name],
            linestyle="-",
            linewidth=2,
            label=plot_labels[group_name],
        )
        for group_name in plotted_groups
    ]

    layer_handles = []
    for idx in plotted_layer_band_indices[::-1]:
        start_layer, end_layer = layer_groups[idx]
        layer_label = f"{start_layer-1}-{end_layer-1}"
        layer_handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle=layer_styles[idx % len(layer_styles)],
                linewidth=2,
                label=layer_label,
            )
        )
    if x_axis == "tokens":
        plt.axvline(
            11_000_000_000,
            color="black",
            zorder=0,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        ax.text(
            3_000_000_000,
            0.02,
            "Phase I",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=15,
        )
        ax.text(
            22_500_000_000,
            0.02,
            "Phase II",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=15,
        )

    ax.legend(
        handles=group_handles + layer_handles,
        title=(
            "Token overlap / Layers"
            if grouping_variant == "jaccard"
            else "Languages/ Layers"
        ),
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    metrics_name = "eval-logit_lens_grouped_time_layer_group_plot"
    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics=metrics_name,
        score_type=score_type,
        aggregate_probs=aggregate_probs,
    )
    mode_suffix = (
        "_tgt_only" if grouping_variant == "jaccard" and jaccard_target_only else ""
    )
    path = os.path.join(
        figures_path,
        f"{base_prefix}_logit_lens_grouped_tf_layer_group_plot_{scope_label}_{grouping_variant}{mode_suffix}{suffix}",
    )
    save_figure(fig, path, strip_titles=strip_titles)
    print(f"Saved grouped TF layer group plot to {path}")


def _resolve_requested_tf_groups(requested_groups, grouping_variant):
    """Map user-provided group aliases to internal grouped TF bucket names."""
    if grouping_variant == "relative_script":
        step_group_order = STEP_GROUP_ORDER_RELATIVE
    elif grouping_variant == "collapsed_other":
        step_group_order = STEP_GROUP_ORDER_COLLAPSED
    elif grouping_variant == "jaccard":
        step_group_order = STEP_GROUP_ORDER_JACCARD
    else:
        step_group_order = STEP_GROUP_ORDER

    if not requested_groups:
        return list(step_group_order)

    aliases = {
        "target": ["target"],
        "tgt": ["target"],
        "source": ["source"],
        "src": ["source"],
        "english": ["english"],
        "en": ["english"],
    }

    if grouping_variant == "relative_script":
        aliases.update(
            {
                "same": ["other_same_as_target"],
                "same_script": ["other_same_as_target"],
                "different": ["other_diff_from_target"],
                "diff": ["other_diff_from_target"],
                "different_script": ["other_diff_from_target"],
                "other": ["other_same_as_target", "other_diff_from_target"],
            }
        )
    elif grouping_variant == "collapsed_other":
        aliases.update(
            {
                "other": ["other"],
                "collapsed_other": ["other"],
            }
        )
    elif grouping_variant == "jaccard":
        aliases.update(
            {
                "target": [JACCARD_TARGET_FULL_KEY],
                "tgt": [JACCARD_TARGET_FULL_KEY],
                "full": [JACCARD_TARGET_FULL_KEY],
                "copy": ["identical_subtokens"],
                "identical": ["identical_subtokens"],
                "exact": ["identical_subtokens"],
                "partial": ["partial_subtoken_overlap"],
                "overlap": ["partial_subtoken_overlap"],
                "none": ["no_subtoken_overlap"],
                "zero": ["no_subtoken_overlap"],
                "identical_subtokens": ["identical_subtokens"],
                "partial_subtoken_overlap": ["partial_subtoken_overlap"],
                "no_subtoken_overlap": ["no_subtoken_overlap"],
            }
        )
    else:
        aliases.update(
            {
                "cjk": ["other_cjk"],
                "latin": ["other_latin"],
                "other": ["other_cjk", "other_latin"],
            }
        )

    resolved = []
    for item in requested_groups:
        key = item.strip().lower()
        if key in aliases:
            for g in aliases[key]:
                if (
                    g in step_group_order
                    or (grouping_variant == "jaccard" and g == JACCARD_TARGET_FULL_KEY)
                ) and g not in resolved:
                    resolved.append(g)
        elif key in step_group_order and key not in resolved:
            resolved.append(key)

    if not resolved:
        print(
            f"No valid grouped TF names in --plot_tf_langs for {grouping_variant}; using all groups."
        )
        return list(step_group_order)

    return resolved


def _compute_target_time_average_over_pairs(
    run_name,
    pairs,
    score_type,
    aggregate_probs,
):
    """Compute average target-language TF matrix over time for a set of directed pairs."""
    grouped_values = []
    reference_steps = None
    collated_by_tgt = {}

    for src, tgt in pairs:
        file_name = f"eval-logit_lens_{tgt}.jsonl"
        if tgt not in collated_by_tgt:
            collated_file = os.path.join(
                CACHE_DIR,
                get_collate_name_logit_lens(
                    run_name, file_name, score_type, aggregate_probs
                ),
            )
            if not os.path.exists(collated_file):
                print(f"  Missing collated file for target {tgt}: {collated_file}")
                collated_by_tgt[tgt] = None
            else:
                with open(collated_file, "r", encoding="utf-8") as f:
                    collated_by_tgt[tgt] = json.load(f)

        data = collated_by_tgt.get(tgt)

        results = data.get("results", {}).get(f"{src}_to", [])
        if not results:
            continue

        results = sorted(results, key=lambda r: r.get("step", -1))
        pair_steps = []
        grouped_series = []
        for result in results:
            step = result.get("step")
            pair_steps.append(step)
            mean_dict = result.get("mean", {})
            grouped_series.append(mean_dict.get(tgt))

        if reference_steps is None:
            reference_steps = pair_steps
        elif pair_steps != reference_steps:
            print(f"  Skipping {src} -> {tgt}: step grid mismatch for target TF view")
            continue

        matrix = _vectors_to_padded_matrix(grouped_series)
        if matrix is not None:
            grouped_values.append(matrix)

    if reference_steps is None or not grouped_values:
        return None, None

    return reference_steps, np.nanmean(np.stack(grouped_values, axis=0), axis=0)


def plot_grouped_tf_lang_aggregates(
    run_name,
    pairs,
    requested_groups,
    figures_path="figures",
    x_axis="steps",
    mode="expl-training",
    score_type="scores",
    aggregate_probs=False,
    grouping_variant="cjk_latin",
    scope_label="grouped",
    title_scope="Grouped",
    exclude_langs=None,
    strip_titles=True,
    jaccard_target_only=False,
):
    """Plot grouped TF aggregates in the same style as per-tf-language plots."""
    pairs = pairs or []
    if not pairs:
        print(f"No pairs found for grouped TF-language plotting (scope={scope_label}).")
        return

    reference_steps, avg_by_group, step_group_order, step_group_labels = (
        _compute_grouped_time_averages_over_pairs(
            run_name=run_name,
            pairs=pairs,
            score_type=score_type,
            aggregate_probs=aggregate_probs,
            grouping_variant=grouping_variant,
            jaccard_target_only=jaccard_target_only,
        )
    )
    if reference_steps is None:
        print(f"No grouped TF data available for scope={scope_label}.")
        return

    groups_to_plot = _resolve_requested_tf_groups(requested_groups, grouping_variant)

    if (
        grouping_variant == "jaccard"
        and JACCARD_TARGET_FULL_KEY in groups_to_plot
        and JACCARD_TARGET_FULL_KEY not in avg_by_group
    ):
        target_steps, target_avg = _compute_target_time_average_over_pairs(
            run_name=run_name,
            pairs=pairs,
            score_type=score_type,
            aggregate_probs=aggregate_probs,
        )
        if target_steps == reference_steps and target_avg is not None:
            avg_by_group[JACCARD_TARGET_FULL_KEY] = target_avg
            step_group_labels[JACCARD_TARGET_FULL_KEY] = JACCARD_TARGET_FULL_LABEL
        else:
            print(
                "Could not compute full target TF average for jaccard grouped plot; skipping requested 'target' overlay."
            )

    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )
    x_vals = get_x_values(reference_steps, step_to_tokens, x_axis)
    prepare_plotting()

    metrics_name = "eval-logit_lens_grouped_tf_lang"
    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics=metrics_name,
        score_type=score_type,
        aggregate_probs=aggregate_probs,
    )
    mode_suffix = (
        "_tgt_only" if grouping_variant == "jaccard" and jaccard_target_only else ""
    )

    for group_name in groups_to_plot:
        values_per_layer = avg_by_group.get(group_name)
        if values_per_layer is None or np.all(np.isnan(values_per_layer)):
            continue

        values_per_layer = np.asarray(values_per_layer, dtype=float)
        n_layers = values_per_layer.shape[1]
        norm = plt.Normalize(vmin=0, vmax=n_layers - 1)
        cmap = plt.cm.viridis

        fig, ax = plt.subplots(figsize=(8, 5))
        for layer in range(1, n_layers):
            values = values_per_layer[:, layer]
            color = cmap(norm(layer))
            ax.plot(x_vals, values, color=color, linewidth=2)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Layer")
        if not aggregate_probs:
            ax.set_ylim(-12.2, 0)

        ax.set_xlabel(get_x_label(x_axis))
        ax.set_ylabel("p" if aggregate_probs else "log p")
        ax.set_title(
            f"Grouped TF language: {step_group_labels[group_name]} ({title_scope}, {grouping_variant})"
        )
        ax.grid(True, alpha=0.3)
        apply_token_formatter(ax, x_axis)

        path = os.path.join(
            figures_path,
            f"{base_prefix}_logit_lens_grouped_tf_{scope_label}_{group_name}_{grouping_variant}{mode_suffix}{suffix}",
        )
        save_figure(fig, path, strip_titles=strip_titles)
        print(f"Saved grouped TF-language plot for {group_name} to {path}")


def compute_margin_matrix(
    run_name,
    file_name="eval-logit_lens_en.jsonl",
    src_lang="de",
    score_type="scores",
    tf_lang_margin=None,
    aggregate_probs=False,
):
    """
    Compute margin matrix: m[step, layer] = mean(first_lang) - mean(second_lang)
    from existing collated logit lens data.

    If tf_lang_margin is None, uses src_lang as the baseline (copy).
    Otherwise, uses tf_lang_margin as the baseline.
    """
    tgt_lang = get_lang_from_logit_lens_file_name(file_name)
    baseline_lang = tf_lang_margin if tf_lang_margin is not None else src_lang

    collated_file = os.path.join(
        CACHE_DIR,
        get_collate_name_logit_lens(run_name, file_name, score_type, aggregate_probs),
    )
    if not os.path.exists(collated_file):
        raise FileNotFoundError(f"Collated file not found at {collated_file}")

    with open(collated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {}).get(f"{src_lang}_to", [])
    if not results:
        raise ValueError(f"No results found for {src_lang}_to in {collated_file}")

    # Extract target and baseline scores
    steps = []
    margins_list = []

    for result in results:
        step = result.get("step")
        mean_dict = result.get("mean", {})

        # First language (target translation)
        first_lang_mean = mean_dict.get(tgt_lang)
        # Second language (baseline - copy or specified teacher forcing language)
        second_lang_mean = mean_dict.get(baseline_lang)

        if first_lang_mean is not None and second_lang_mean is not None:
            steps.append(step)
            # Compute margin per layer
            margin_vec = np.array(first_lang_mean, dtype=float) - np.array(
                second_lang_mean, dtype=float
            )
            margins_list.append(margin_vec)

    if not steps:
        raise ValueError(f"No valid data found for {src_lang} → {tgt_lang}")

    # Stack into matrix (n_steps, n_layers)
    max_layers = max((len(m) for m in margins_list), default=0)
    if max_layers == 0:
        raise ValueError("No layers found")

    margins = np.full((len(margins_list), max_layers), np.nan)
    for idx, margin_vec in enumerate(margins_list):
        margins[idx, : len(margin_vec)] = margin_vec

    return {
        "steps": steps,
        "margins": margins,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "baseline_lang": baseline_lang,
    }


def compute_average_margin_matrix_both_directions(
    run_name,
    src_langs,
    score_type="scores",
    tf_lang_margin=None,
    tgt_langs=None,
    aggregate_probs=False,
):
    """Compute average margin matrix across multiple source AND target languages."""
    langs_to_use_tgt = tgt_langs if tgt_langs is not None else LANGS

    all_margins = []
    reference_steps = None
    successful_pairs = []

    for tgt_lang in langs_to_use_tgt:
        file_name = f"eval-logit_lens_{tgt_lang}.jsonl"
        for src_lang in src_langs:
            if tgt_lang == src_lang:
                continue
            try:
                margin_data = compute_margin_matrix(
                    run_name,
                    file_name,
                    src_lang,
                    score_type,
                    tf_lang_margin=tf_lang_margin,
                    aggregate_probs=aggregate_probs,
                )
                if reference_steps is None:
                    reference_steps = margin_data["steps"]
                all_margins.append(margin_data["margins"])
                successful_pairs.append((src_lang, tgt_lang))
            except Exception as e:
                print(f"  Skipping {src_lang} → {tgt_lang}: {e}")
                continue

    if not all_margins:
        raise ValueError("No valid data found for any source-target pairs")

    margins_stack = np.stack(all_margins, axis=0)  # (n_pairs, n_steps, n_layers)
    avg_margins = np.nanmean(margins_stack, axis=0)  # (n_steps, n_layers)

    return {
        "steps": reference_steps,
        "margins": avg_margins,
        "src_langs": src_langs,
        "tgt_langs": [lang for _, lang in set(successful_pairs)],
        "baseline_lang": tf_lang_margin if tf_lang_margin is not None else src_langs[0],
    }


def compute_grouped_margin_matrix(
    run_name,
    file_name="eval-logit_lens_en.jsonl",
    src_lang="de",
    score_type="scores",
    aggregate_probs=False,
    grouping_variant="cjk_latin",
    jaccard_target_only=False,
):
    """
    Compute margin matrices for each language group.

    For non-Jaccard groupings, for each group (target, source, english, other_cjk, etc.):
        margin[step, layer] = mean(target_lang) - mean(group_langs)

    For Jaccard groupings, inter-bucket comparison:
        margin[step, layer] = mean(group_bucket) - mean(no_subtoken_overlap_bucket)
    """
    tgt_lang = get_lang_from_logit_lens_file_name(file_name)

    if grouping_variant == "relative_script":
        step_group_order = STEP_GROUP_ORDER_RELATIVE
    elif grouping_variant == "collapsed_other":
        step_group_order = STEP_GROUP_ORDER_COLLAPSED
    elif grouping_variant == "jaccard":
        step_group_order = STEP_GROUP_ORDER_JACCARD
    else:
        step_group_order = STEP_GROUP_ORDER

    collated_file = os.path.join(
        CACHE_DIR,
        get_collate_name_logit_lens(run_name, file_name, score_type, aggregate_probs),
    )

    if not os.path.exists(collated_file):
        raise FileNotFoundError(f"Collated file not found at {collated_file}")

    with open(collated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {}).get(f"{src_lang}_to", [])
    if not results:
        raise ValueError(f"No results found for {src_lang}_to")

    # Initialize: for each group, collect margin vectors across steps
    grouped_margins = {group: [] for group in step_group_order}
    steps = []

    for result in results:
        step = result.get("step")
        mean_dict = result.get("mean", {})

        # Get target language mean
        tgt_mean = mean_dict.get(tgt_lang)
        if tgt_mean is None:
            continue

        steps.append(step)

        if grouping_variant == "jaccard":
            # Grouped over Jaccard buckets from pre-collated selected-synonym stats.
            # Margin = group_mean - no_subtoken_overlap_mean (inter-bucket comparison).
            jaccard_stats = result.get("similarity_group_stats", {})
            jaccard_counts = result.get("similarity_group_counts", {})
            tf_langs = [tgt_lang] if jaccard_target_only else LANGS

            # First pass: compute mean vector for each bucket.
            bucket_mean_by_group = {}
            for group_name in step_group_order:
                bucket_means = []
                for tf_lang in tf_langs:
                    if jaccard_counts.get(tf_lang, {}).get(group_name, 0) <= 0:
                        continue
                    vec = jaccard_stats.get(tf_lang, {}).get(group_name, {}).get("mean")
                    if vec is not None:
                        bucket_means.append(vec)
                if bucket_means:
                    mean_vec, _ = _stack_and_nanmean(bucket_means)
                    bucket_mean_by_group[group_name] = mean_vec
                else:
                    bucket_mean_by_group[group_name] = None

            # Reference: no_subtoken_overlap bucket.
            ref_mean = bucket_mean_by_group.get("no_subtoken_overlap")

            # Second pass: compute margin = group_mean - ref_mean.
            for group_name in step_group_order:
                group_mean = bucket_mean_by_group.get(group_name)
                if group_mean is None or ref_mean is None:
                    grouped_margins[group_name].append(None)
                    continue
                # Align lengths (ref may differ from group due to padding).
                max_len = max(len(group_mean), len(ref_mean))
                gm = np.full(max_len, np.nan)
                rm = np.full(max_len, np.nan)
                gm[: len(group_mean)] = group_mean
                rm[: len(ref_mean)] = ref_mean
                grouped_margins[group_name].append(gm - rm)
        else:
            # Get groups for this pair
            group_langs = _get_group_langs_for_pair(
                src_lang,
                tgt_lang,
                grouping_variant,
            )

            # For each group, compute margin: target - group_mean
            for group_name in step_group_order:
                langs = group_langs[group_name]
                # Collect means for languages in this group
                group_means = [
                    mean_dict.get(lang)
                    for lang in langs
                    if mean_dict.get(lang) is not None
                ]

                if not group_means:
                    grouped_margins[group_name].append(None)
                    continue

                # Average group means using the existing _stack_and_nanmean logic
                group_mean, _ = _stack_and_nanmean(group_means)

                if group_mean is None:
                    grouped_margins[group_name].append(None)
                else:
                    # Compute margin: target - group
                    tgt_arr = np.array(tgt_mean, dtype=float)
                    margin = tgt_arr - group_mean
                    grouped_margins[group_name].append(margin)

    # Convert to matrices
    result_matrices = {}
    for group_name in step_group_order:
        margins_list = grouped_margins[group_name]
        max_layers = max((len(m) for m in margins_list if m is not None), default=0)

        if max_layers == 0:
            result_matrices[group_name] = None
        else:
            margins = np.full((len(margins_list), max_layers), np.nan)
            for idx, margin_vec in enumerate(margins_list):
                if margin_vec is not None:
                    margins[idx, : len(margin_vec)] = margin_vec
            result_matrices[group_name] = margins

    return {
        "steps": steps,
        "margins_by_group": result_matrices,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "grouping_variant": grouping_variant,
    }


def compute_average_grouped_margin_matrix(
    run_name,
    file_name,
    src_langs,
    score_type="scores",
    aggregate_probs=False,
    grouping_variant="cjk_latin",
    grouped_margin_reference="auto",
    jaccard_target_only=False,
):
    """
    Compute average grouped margin matrices across multiple source languages.

    For each source language, computes grouped margins, then averages across sources.
    """
    tgt_lang = get_lang_from_logit_lens_file_name(file_name)
    successful_langs = [src for src in src_langs if src != tgt_lang]
    if not successful_langs:
        raise ValueError(f"No valid data found for any source languages to {tgt_lang}")

    pairs = [(src, tgt_lang) for src in successful_langs]
    reference_steps, avg_by_group, step_group_order, _ = (
        _compute_grouped_time_averages_over_pairs(
            run_name=run_name,
            pairs=pairs,
            score_type=score_type,
            aggregate_probs=aggregate_probs,
            grouping_variant=grouping_variant,
            jaccard_target_only=jaccard_target_only,
        )
    )

    if reference_steps is None:
        raise ValueError(f"No valid data found for any source languages to {tgt_lang}")

    reference_group = _resolve_grouped_margin_reference(
        grouping_variant, grouped_margin_reference
    )
    reference_values = avg_by_group.get(reference_group)

    result_margins = {}
    for group_name in step_group_order:
        group_values = avg_by_group.get(group_name)
        if group_values is None or reference_values is None:
            result_margins[group_name] = None
            continue
        if grouping_variant == "jaccard":
            result_margins[group_name] = group_values - reference_values
        else:
            result_margins[group_name] = reference_values - group_values

    return {
        "steps": reference_steps,
        "margins_by_group": result_margins,
        "src_langs": successful_langs,
        "tgt_lang": tgt_lang,
        "grouping_variant": grouping_variant,
        "reference_group": reference_group,
    }


def compute_average_grouped_margin_matrix_both_directions(
    run_name,
    src_langs,
    score_type="scores",
    tgt_langs=None,
    aggregate_probs=False,
    grouping_variant="cjk_latin",
    grouped_margin_reference="auto",
    jaccard_target_only=False,
):
    """Compute average grouped margins across multiple source AND target languages."""
    langs_to_use_tgt = tgt_langs if tgt_langs is not None else LANGS
    pairs = [
        (src_lang, tgt_lang)
        for tgt_lang in langs_to_use_tgt
        for src_lang in src_langs
        if tgt_lang != src_lang
    ]

    if not pairs:
        raise ValueError("No valid data found for any grouped source-target pairs")

    reference_steps, avg_by_group, step_group_order, _ = (
        _compute_grouped_time_averages_over_pairs(
            run_name=run_name,
            pairs=pairs,
            score_type=score_type,
            aggregate_probs=aggregate_probs,
            grouping_variant=grouping_variant,
            jaccard_target_only=jaccard_target_only,
        )
    )

    if reference_steps is None:
        raise ValueError("No valid data found for any grouped source-target pairs")

    reference_group = _resolve_grouped_margin_reference(
        grouping_variant, grouped_margin_reference
    )
    reference_values = avg_by_group.get(reference_group)

    result_margins = {}
    for group_name in step_group_order:
        group_values = avg_by_group.get(group_name)
        if group_values is None or reference_values is None:
            result_margins[group_name] = None
            continue
        if grouping_variant == "jaccard":
            result_margins[group_name] = group_values - reference_values
        else:
            result_margins[group_name] = reference_values - group_values

    successful_pairs = pairs

    return {
        "steps": reference_steps,
        "margins_by_group": result_margins,
        "src_langs": sorted({src for src, _ in successful_pairs}),
        "tgt_langs": sorted({tgt for _, tgt in successful_pairs}),
        "grouping_variant": grouping_variant,
        "reference_group": reference_group,
    }


def plot_grouped_margin_heatmaps(
    run_name,
    file_name="eval-logit_lens_en.jsonl",
    src_langs=None,
    tgt_langs=None,
    figures_path="figures",
    x_axis="steps",
    mode="expl-training",
    score_type="scores",
    aggregate_probs=False,
    grouping_variant="cjk_latin",
    exclude_langs=None,
    simple_margin_label=False,
    margin_layer_diff=False,
    margin_time_diff=False,
    strip_titles=True,
    grouped_margin_reference="auto",
    jaccard_target_only=False,
):
    """
    Plot margin heatmaps for target language vs each language group.

    Creates one heatmap per group showing: margin = mean(target) - mean(group)
    Reuses _plot_margin_heatmap_from_data for consistent rendering.
    """
    tgt_lang = get_lang_from_logit_lens_file_name(file_name)

    if src_langs is None or len(src_langs) == 0:
        raise ValueError("src_langs must be provided for grouped margin plotting")

    if grouping_variant == "relative_script":
        step_group_order = STEP_GROUP_ORDER_RELATIVE
        step_group_labels = STEP_GROUP_LABELS_RELATIVE
    elif grouping_variant == "collapsed_other":
        step_group_order = STEP_GROUP_ORDER_COLLAPSED
        step_group_labels = STEP_GROUP_LABELS_COLLAPSED
    elif grouping_variant == "jaccard":
        step_group_order = STEP_GROUP_ORDER_JACCARD
        step_group_labels = STEP_GROUP_LABELS_JACCARD
    else:
        step_group_order = STEP_GROUP_ORDER
        step_group_labels = STEP_GROUP_LABELS

    reference_group = _resolve_grouped_margin_reference(
        grouping_variant, grouped_margin_reference
    )
    reference_label = step_group_labels.get(reference_group, reference_group)
    print(
        "  Grouped margin reference "
        f"({grouping_variant}): {reference_group} [{reference_label}]"
    )

    # Optional all-to-all mode: average grouped margins across both source and target pools
    if tgt_langs is not None:
        grouped_margin_data = compute_average_grouped_margin_matrix_both_directions(
            run_name=run_name,
            src_langs=src_langs,
            score_type=score_type,
            tgt_langs=tgt_langs,
            aggregate_probs=aggregate_probs,
            grouping_variant=grouping_variant,
            grouped_margin_reference=grouped_margin_reference,
            jaccard_target_only=jaccard_target_only,
        )

        prob_str = "p" if aggregate_probs else "log p"
        src_label_for_path = "all_to_all"
        reference_suffix = f"_ref_{reference_group}"
        mode_suffix = (
            "_tgt_only" if grouping_variant == "jaccard" and jaccard_target_only else ""
        )

        for group_name in step_group_order:
            margins = grouped_margin_data["margins_by_group"][group_name]
            if margins is None or np.all(np.isnan(margins)):
                continue

            is_reference_group = group_name == grouped_margin_data.get(
                "reference_group"
            )
            finite_vals = margins[np.isfinite(margins)]
            is_effectively_zero = finite_vals.size > 0 and np.allclose(
                finite_vals, 0.0, atol=1e-12
            )
            if is_reference_group:
                print(
                    f"    NOTE: '{group_name}' is the explicit reference group; skipping save because the map is zero by definition."
                )
                continue
            elif is_effectively_zero:
                print(
                    f"    NOTE: '{group_name}' margin map is numerically near-zero; skipping empty heatmap save."
                )
                continue

            group_label_lower = step_group_labels[group_name].lower()
            margin_data_for_group = {
                "steps": grouped_margin_data["steps"],
                "margins": margins,
                "src_lang": "all",
                "tgt_lang": "en",
                "baseline_lang": group_name,
                "grouping_variant": grouping_variant,
            }
            custom_colorbar_label = None
            if not simple_margin_label:
                if grouping_variant == "jaccard":
                    custom_colorbar_label = f"Average [{prob_str}({group_label_lower}) - {prob_str}({reference_label.lower()})]"
                else:
                    custom_colorbar_label = f"Average [{prob_str}({reference_label.lower()}) - {prob_str}({group_label_lower})]"

            title_prefix = "[Reference] " if is_reference_group else ""

            _plot_margin_heatmap_from_data(
                margin_data=margin_data_for_group,
                run_name=run_name,
                figures_path=figures_path,
                x_axis=x_axis,
                mode=mode,
                score_type=score_type,
                title_lang=f"{title_prefix}All -> All (avg) vs {step_group_labels[group_name]}",
                src_lang_for_copy="all",
                src_lang_label=f"grouped_{src_label_for_path}_{grouping_variant}{reference_suffix}{mode_suffix}",
                metrics_for_suffix="eval-logit_lens_en.jsonl",
                source_averaged=False,
                tf_lang_margin=None,
                custom_colorbar_label=custom_colorbar_label,
                simple_margin_label=simple_margin_label,
                exclude_langs=exclude_langs,
                aggregate_probs=aggregate_probs,
                margin_layer_diff=margin_layer_diff,
                margin_time_diff=margin_time_diff,
                strip_titles=strip_titles,
            )
            print(f"    Saved all-to-all grouped margin heatmap for {group_name}")
        return

    # For single source, compute directly; for multiple, average
    if len(src_langs) == 1:
        src_lang = src_langs[0]
        grouped_margin_data = compute_grouped_margin_matrix(
            run_name,
            file_name,
            src_lang,
            score_type,
            aggregate_probs,
            grouping_variant,
            jaccard_target_only,
        )
        src_label_for_title = LANG_NAME_EN[src_lang]
        src_label_for_path = src_lang
    else:
        grouped_margin_data = compute_average_grouped_margin_matrix(
            run_name,
            file_name,
            src_langs,
            score_type,
            aggregate_probs,
            grouping_variant,
            grouped_margin_reference,
            jaccard_target_only,
        )

        # Check if we're using all available languages for this target
        expected_all_langs = {
            lang
            for lang in LANGS
            if lang != tgt_lang and lang not in (exclude_langs or [])
        }
        actual_langs = set(grouped_margin_data["src_langs"])
        is_all_langs = actual_langs == expected_all_langs

        if is_all_langs:
            src_label_for_title = "All"
            src_label_for_path = "all"
        else:
            src_label_for_title = " & ".join(
                [LANG_NAME_EN[s] for s in grouped_margin_data["src_langs"]]
            )
            src_label_for_path = "_".join(grouped_margin_data["src_langs"])

    # Plot each group's margins as a separate heatmap
    mode_suffix = (
        "_tgt_only" if grouping_variant == "jaccard" and jaccard_target_only else ""
    )
    reference_suffix = f"_ref_{reference_group}"
    for group_name in step_group_order:
        margins = grouped_margin_data["margins_by_group"][group_name]
        if margins is None or np.all(np.isnan(margins)):
            continue  # Skip groups with no data

        is_reference_group = group_name == grouped_margin_data.get("reference_group")
        finite_vals = margins[np.isfinite(margins)]
        is_effectively_zero = finite_vals.size > 0 and np.allclose(
            finite_vals, 0.0, atol=1e-12
        )
        if is_reference_group:
            print(
                f"    NOTE: '{group_name}' is the explicit reference group; skipping save because the map is zero by definition."
            )
            continue
        elif is_effectively_zero:
            print(
                f"    NOTE: '{group_name}' margin map is numerically near-zero; skipping empty heatmap save."
            )
            continue

        # Wrap group margins in format expected by _plot_margin_heatmap_from_data
        group_label_lower = step_group_labels[group_name].lower()
        prob_str = "p" if aggregate_probs else "log p"

        margin_data_for_group = {
            "steps": grouped_margin_data["steps"],
            "margins": margins,
            "src_lang": src_label_for_path,
            "tgt_lang": tgt_lang,
            "baseline_lang": group_name,  # Use group name as the "baseline" for labeling
            "grouping_variant": grouping_variant,
        }

        # Custom colorbar label: for Jaccard, show group vs no_overlap; otherwise target vs group.
        custom_colorbar_label = None
        if not simple_margin_label:
            if grouping_variant == "jaccard":
                custom_colorbar_label = f"{prob_str}({group_label_lower}) - {prob_str}({reference_label.lower()})"
            else:
                custom_colorbar_label = f"{prob_str}({reference_label.lower()}) - {prob_str}({group_label_lower})"

        title_prefix = "[Reference] " if is_reference_group else ""

        if grouping_variant == "jaccard":
            heatmap_title_lang = f"{title_prefix}{step_group_labels[group_name]} − {reference_label}: {src_label_for_title}"
        else:
            heatmap_title_lang = f"{title_prefix}{reference_label} vs {step_group_labels[group_name]}: {src_label_for_title}"

        _plot_margin_heatmap_from_data(
            margin_data=margin_data_for_group,
            run_name=run_name,
            figures_path=figures_path,
            x_axis=x_axis,
            mode=mode,
            score_type=score_type,
            title_lang=heatmap_title_lang,
            src_lang_for_copy=src_label_for_path,
            src_lang_label=f"grouped_{src_label_for_path}_to_{tgt_lang}_{grouping_variant}{reference_suffix}{mode_suffix}",
            metrics_for_suffix=file_name,
            source_averaged=False,
            tf_lang_margin=None,
            custom_colorbar_label=custom_colorbar_label,
            simple_margin_label=simple_margin_label,
            exclude_langs=exclude_langs,
            aggregate_probs=aggregate_probs,
            margin_layer_diff=margin_layer_diff,
            margin_time_diff=margin_time_diff,
            strip_titles=strip_titles,
        )
        print(f"    Saved grouped margin heatmap for {group_name}")


def _plot_margin_heatmap_from_data(
    margin_data,
    run_name,
    figures_path,
    x_axis,
    mode,
    score_type,
    title_lang,
    src_lang_for_copy,
    src_lang_label,
    metrics_for_suffix,
    source_averaged=False,
    tf_lang_margin=None,
    custom_colorbar_label=None,
    simple_margin_label=False,
    exclude_langs=None,
    aggregate_probs=False,
    margin_layer_diff=False,
    margin_time_diff=False,
    strip_titles=True,
):
    """Shared renderer for margin heatmaps from precomputed margin_data."""
    steps = margin_data["steps"]
    margins = margin_data["margins"]

    # Logit-lens vectors include embedding at index 0; always drop it for margin plots.
    if margins.shape[1] < 2:
        raise ValueError(
            "Cannot plot margins with fewer than 2 values (embedding + at least 1 model layer)"
        )
    margins = margins[:, 1:]

    if margin_layer_diff and margin_time_diff:
        raise ValueError(
            "Use only one diff mode: margin_layer_diff or margin_time_diff"
        )

    if margin_layer_diff:
        if margins.shape[1] < 2:
            raise ValueError(
                "Cannot compute layer-wise margin differences with fewer than 2 model layers"
            )
        margins = np.diff(margins, axis=1)
    elif margin_time_diff:
        if margins.shape[0] < 2:
            raise ValueError(
                "Cannot compute time-wise margin differences with fewer than 2 steps"
            )
        margins = np.diff(margins, axis=0)
        steps = steps[1:]

    is_asymmetry = margin_data.get("asymmetry_type", False)

    if not is_asymmetry:
        baseline_lang = margin_data["baseline_lang"]
        grouping_variant = margin_data.get("grouping_variant", "")

        # Determine which group labels to use if this is a grouped margin
        if grouping_variant == "relative_script":
            group_labels = STEP_GROUP_LABELS_RELATIVE
        elif grouping_variant == "collapsed_other":
            group_labels = STEP_GROUP_LABELS_COLLAPSED
        elif grouping_variant == "jaccard":
            group_labels = STEP_GROUP_LABELS_JACCARD
        else:
            group_labels = STEP_GROUP_LABELS

        # Handle group names (from grouped margins), "copy" sentinel, or language codes
        if baseline_lang in group_labels:
            baseline_lang_name = group_labels[baseline_lang]
        elif baseline_lang == "copy":
            baseline_lang_name = "Copy"
        else:
            baseline_lang_name = LANG_NAME_EN.get(baseline_lang, baseline_lang)

        if custom_colorbar_label is not None:
            colorbar_label = custom_colorbar_label
        else:
            # Use "p" for aggregated probabilities, "log p" for log probabilities
            prob_str = "p" if aggregate_probs else "log p"
            if simple_margin_label:
                colorbar_label = f"{prob_str} difference"
            else:
                colorbar_label = f"{prob_str}({LANG_NAME_EN[margin_data['tgt_lang']]}) - {prob_str}({baseline_lang_name})"
    else:
        # For asymmetry, use the custom colorbar label if provided
        colorbar_label = (
            custom_colorbar_label if custom_colorbar_label is not None else "Asymmetry"
        )

    if margin_layer_diff:
        if simple_margin_label:
            colorbar_label = "Layerwise margin change"
        else:
            colorbar_label = f"Layerwise change in [{colorbar_label}]"
    elif margin_time_diff:
        if simple_margin_label:
            colorbar_label = "Timewise margin change"
        else:
            colorbar_label = f"Timewise change in [{colorbar_label}]"

    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )
    x_vals = get_x_values(steps, step_to_tokens, x_axis)
    prepare_plotting()
    fig, ax = plt.subplots()
    n_layers = margins.shape[1]
    layer_indices = np.arange(n_layers) if margin_layer_diff else np.arange(n_layers)
    X, Y = np.meshgrid(x_vals, layer_indices)
    colormap = "RdBu_r"
    max_abs = np.nanmax(np.abs(margins)) if np.any(np.isfinite(margins)) else 1.0
    im = ax.pcolormesh(
        X, Y, margins.T, cmap=colormap, shading="auto", vmin=-max_abs, vmax=max_abs
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, rotation=270, labelpad=20)

    # the zero contour needs at least two checkpoints
    if not margin_layer_diff and not margin_time_diff and len(x_vals) > 1:
        ax.contour(
            X,
            Y,
            margins.T,
            levels=[0],
            colors="gray",
            linewidths=1,
            linestyles="-",
        )

    ax.set_xlabel(get_x_label(x_axis))
    if margin_layer_diff:
        ax.set_ylabel("Layer transition (L → L+1)")
    else:
        ax.set_ylabel("Layer (embedding skipped)")

    # Determine title
    if is_asymmetry:
        # For asymmetry, use the title_lang directly (e.g., "Asymmetry: German↔English")
        ax.set_title(title_lang)
    else:
        baseline_lang = margin_data["baseline_lang"]
        grouping_variant = margin_data.get("grouping_variant", "")

        # For grouped margins, title_lang already includes the group information
        # So just use it directly instead of reconstructing
        if grouping_variant:
            ax.set_title(title_lang)
        else:
            # Determine baseline name for title (non-grouped margins)
            if baseline_lang == "copy" or baseline_lang == src_lang_for_copy:
                baseline_name = "Copy"
            else:
                baseline_name = LANG_NAME_EN.get(baseline_lang, baseline_lang)
            ax.set_title(f"Target vs {baseline_name} Margin: {title_lang}")

    apply_token_formatter(ax, x_axis)

    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=exclude_langs,
        x_axis=x_axis,
        metrics=metrics_for_suffix,
        score_type=score_type,
        aggregate_probs=aggregate_probs,
    )

    if source_averaged:
        baseline_suffix = f"_vs_{baseline_lang}" if tf_lang_margin is not None else ""
    else:
        baseline_suffix = (
            f"_vs_{baseline_lang}"
            if baseline_lang and baseline_lang != src_lang_for_copy
            else ""
        )
    if margin_layer_diff:
        gradient_suffix = "_margin_gradient"
    elif margin_time_diff:
        gradient_suffix = "_margin_time_gradient"
    else:
        gradient_suffix = ""

    path = os.path.join(
        figures_path,
        f"{base_prefix}_margin_heatmap_{src_lang_label}{baseline_suffix}{gradient_suffix}{suffix}",
    )
    save_figure(fig, path, strip_titles=strip_titles)
    print(f"Saved margin heatmap to {path}")


def parse_plot_args():
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Plot results from logit lens evaluations."
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
        "--model_name",
        type=str,
        default="llama1b.json",
        help="Model config type for expl-training (e.g., 'llama1b.json').",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="en",
        help="Target language for logit lens evaluation file (e.g., 'en').",
    )
    parser.add_argument(
        "--src_langs",
        type=str,
        default="de",
        help="Comma-separated list of source languages to plot from (e.g., 'de,fr').",
    )
    parser.add_argument(
        "--plot_steps",
        type=str,
        default=None,
        help="Comma-separated list of steps to plot (e.g., '15000,50000,100000').",
    )
    parser.add_argument(
        "--plot_tf_langs",
        type=str,
        default=None,
        help="Comma-separated list of teacher forcing languages to plot (e.g., 'fr,es').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of collated results.",
    )
    parser.add_argument(
        "--x_axis",
        type=str,
        choices=["steps", "tokens"],
        default="steps",
        help="X-axis for time-series plots: steps or tokens processed (not used for per-step plots).",
    )
    parser.add_argument(
        "--score_type",
        type=str,
        choices=["scores", "scores_normed", "scores_first_token"],
        default="scores",
        help="Type of scores to use for logit lens plots.",
    )
    parser.add_argument(
        "--plot_margins",
        action="store_true",
        help="Plot margin visualizations heatmap .",
    )
    parser.add_argument(
        "--tf_lang_margin",
        type=str,
        default=None,
        help="Teacher forcing language to use as baseline for margin computation (default: copy/source language). E.g., 'fr'.",
    )
    parser.add_argument(
        "--simple_margin_label",
        action="store_true",
        help="Use generic margin colorbar labels (e.g., 'log p difference') instead of explicit language formulas.",
    )
    parser.add_argument(
        "--margin_layer_diff",
        action="store_true",
        help="For margin heatmaps, plot consecutive layer differences of margins (margin[L] - margin[L-1]) instead of raw margins.",
    )
    parser.add_argument(
        "--margin_time_diff",
        action="store_true",
        help="For margin heatmaps, plot consecutive time differences over training steps/tokens.",
    )
    parser.add_argument(
        "--avgs",
        action="store_true",
        help="Generate averaged margin plots over selected languages: all selected sources to selected target, and each selected source to selected target(s).",
    )
    parser.add_argument(
        "--exclude_langs",
        type=str,
        default="",
        help="Comma-separated languages to exclude from average pools (applied everywhere LANGS is used in --avgs).",
    )
    parser.add_argument(
        "--aggregate_probs",
        action="store_true",
        help="Aggregate probabilities (exp of log probs) before computing means, instead of aggregating log probs directly.",
    )
    parser.add_argument(
        "--accuracy_mode",
        default="original",
        choices=["original", "ci"],
        help="Accuracy/case mode used for tokenizer-overlap grouping caches.",
    )
    parser.add_argument(
        "--keep_titles",
        action="store_true",
        help="Keep plot titles when saving figures (by default, titles are removed for paper plots).",
    )
    parser.add_argument(
        "--include_jaccard_groups",
        action="store_true",
        help="In --avgs per-step grouped plots, additionally plot Jaccard overlap groups (copy/partial/none).",
    )
    parser.add_argument(
        "--groupings",
        type=str,
        default=None,
        help=(
            "Comma-separated grouping variants to run for grouped analyses in --avgs mode "
            "(cjk_latin,relative_script,collapsed_other,jaccard,all). If omitted, uses cjk_latin+relative_script "
            "and respects --include_jaccard_groups."
        ),
    )
    parser.add_argument(
        "--plot_grouped_time",
        action="store_true",
        help=(
            "In --avgs mode, generate grouped over-time views (TF time heatmaps and layer-group plots). "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--jaccard_target_only",
        action="store_true",
        help=(
            "For jaccard grouped plots, only use target-language teacher forcing "
            "instead of averaging over all teacher-forcing languages."
        ),
    )
    parser.add_argument(
        "--grouped_margin_reference",
        type=str,
        choices=VALID_GROUPED_MARGIN_REFERENCES,
        default="auto",
        help=(
            "Reference bucket for grouped margin plots. "
            "auto=target for non-jaccard and no_subtoken_overlap for jaccard."
        ),
    )
    args = parser.parse_args()

    args.src_langs_explicit = any(
        item == "--src_langs" or item.startswith("--src_langs=") for item in argv
    )
    args.tgt_lang_explicit = any(
        item == "--tgt_lang" or item.startswith("--tgt_lang=") for item in argv
    )

    if args.margin_layer_diff and args.margin_time_diff:
        parser.error("Use only one of --margin_layer_diff or --margin_time_diff")

    # Validate target language (in file name)
    if args.tgt_lang not in LANGS:
        parser.error(
            f"Unsupported target language: {args.tgt_lang}. Must be one of {LANGS}"
        )

    # Parse comma-separated values
    args.plot_steps = (
        [int(step.strip()) for step in args.plot_steps.split(",")]
        if args.plot_steps
        else []
    )
    args.src_langs = [lang.strip() for lang in args.src_langs.split(",")]
    if args.plot_tf_langs:
        args.plot_tf_langs = [lang.strip() for lang in args.plot_tf_langs.split(",")]
    else:
        args.plot_tf_langs = []
    args.exclude_langs = (
        [lang.strip() for lang in args.exclude_langs.split(",") if lang.strip()]
        if args.exclude_langs
        else []
    )
    # Validate source and plot target languages
    for lang in args.src_langs:
        if lang not in LANGS:
            parser.error(f"Unsupported source language: {lang}. Must be one of {LANGS}")
    if args.avgs:
        valid_group_aliases = {
            "target",
            "tgt",
            "source",
            "src",
            "english",
            "en",
            "other",
            "collapsed_other",
            "cjk",
            "latin",
            "same",
            "same_script",
            "different",
            "diff",
            "different_script",
            "other_cjk",
            "other_latin",
            "other_same_as_target",
            "other_diff_from_target",
            "copy",
            "identical",
            "exact",
            "partial",
            "overlap",
            "none",
            "zero",
            "identical_subtokens",
            "partial_subtoken_overlap",
            "no_subtoken_overlap",
        }
        for lang in args.plot_tf_langs:
            if lang not in LANGS and lang.lower() not in valid_group_aliases:
                parser.error(
                    "Unsupported --plot_tf_langs entry in --avgs mode: "
                    f"{lang}. Use a language code or group alias "
                    "(target, source, english, other, cjk, latin, same_script, different_script)."
                )
    else:
        for lang in args.plot_tf_langs:
            if lang not in LANGS:
                parser.error(
                    f"Unsupported plot teacher forcing language: {lang}. Must be one of {LANGS}"
                )
    for lang in args.exclude_langs:
        if lang not in LANGS:
            parser.error(
                f"Unsupported excluded avg language: {lang}. Must be one of {LANGS}"
            )
    # Validate tf_lang_margin if provided
    if args.tf_lang_margin and args.tf_lang_margin not in LANGS:
        parser.error(
            f"Unsupported margin baseline language: {args.tf_lang_margin}. Must be one of {LANGS}"
        )

    try:
        args.grouping_variants = _resolve_grouping_variants(
            args.groupings,
            args.include_jaccard_groups,
        )
    except ValueError as exc:
        parser.error(str(exc))

    for grouping_variant in args.grouping_variants:
        try:
            _resolve_grouped_margin_reference(
                grouping_variant, args.grouped_margin_reference
            )
        except ValueError as exc:
            parser.error(str(exc))

    return args


if __name__ == "__main__":
    # Usage: see the README for the command producing the paper's plots.
    # adding --force will recompute collated results even if they exist.

    args = parse_plot_args()
    os.makedirs(CACHE_DIR, exist_ok=True)

    output_dir = DEFAULT_OUTPUT_DIR
    run_dir = os.path.join(output_dir, args.run_name)
    checkpoints_dir, results_dir = get_checkpoint_results_dirs(run_dir)
    print(f"Results directory: {results_dir}")

    results_dirs = load_results_dirs(run_dir, run_name=args.run_name)

    file_name = f"eval-logit_lens_{args.tgt_lang}.jsonl"
    subfolder = "wlt"

    # Create model-specific figures directory with subfolder
    model_figures_path = os.path.join(args.figures_path, args.run_name, subfolder)

    os.makedirs(model_figures_path, exist_ok=True)
    # Regular analysis (always run)
    collate_results_logit_lens(
        results_dirs,
        run_name=args.run_name,
        file_name=file_name,
        force=args.force,
        mode=args.mode,
        score_type=args.score_type,
        aggregate_probs=args.aggregate_probs,
        model_config_type=args.model_name,
        accuracy_mode=args.accuracy_mode,
    )

    # For grouped aggregate plots under --avgs, ensure all needed target files are collated
    if args.avgs:
        excluded_langs = set(args.exclude_langs)
        langs_pool = [lang for lang in LANGS if lang not in excluded_langs]
        primary_src = (
            args.src_langs[0]
            if (args.src_langs_explicit and args.src_langs)
            else None
        )
        if primary_src not in langs_pool:
            primary_src = None
        primary_tgt = (
            args.tgt_lang
            if args.tgt_lang_explicit and args.tgt_lang in langs_pool
            else None
        )

        pairs_all = [
            (src, tgt) for src in langs_pool for tgt in langs_pool if src != tgt
        ]
        pairs_src_to_all = [
            (src, tgt)
            for src in ([primary_src] if primary_src is not None else [])
            for tgt in langs_pool
            if src != tgt
        ]
        pairs_all_to_tgt = [
            (src, tgt)
            for src in langs_pool
            for tgt in ([primary_tgt] if primary_tgt is not None else [])
            if src != tgt
        ]

        grouped_pairs = pairs_all + pairs_src_to_all + pairs_all_to_tgt
        needed_tgt_langs_grouped = sorted({tgt for _, tgt in grouped_pairs})
        for tgt_lang_needed in needed_tgt_langs_grouped:
            grouped_file_name = f"eval-logit_lens_{tgt_lang_needed}.jsonl"
            collate_results_logit_lens(
                results_dirs,
                run_name=args.run_name,
                file_name=grouped_file_name,
                force=args.force,
                mode=args.mode,
                score_type=args.score_type,
                aggregate_probs=args.aggregate_probs,
                model_config_type=args.model_name,
                accuracy_mode=args.accuracy_mode,
            )

    if args.avgs and args.plot_steps:
        print("Generating grouped per-step aggregate plots (--avgs)...")
        excluded_langs = set(args.exclude_langs)
        langs_pool = [lang for lang in LANGS if lang not in excluded_langs]
        pairs_all = [
            (src, tgt) for src in langs_pool for tgt in langs_pool if src != tgt
        ]
        primary_src = (
            args.src_langs[0]
            if (args.src_langs_explicit and args.src_langs)
            else None
        )
        if primary_src is None:
            pairs_src_to_all = []
        elif primary_src not in langs_pool:
            print(
                "Skipping 'one src → all' grouped plot: source not in available language pool."
            )
            pairs_src_to_all = []
        else:
            if len(args.src_langs) > 1:
                print(
                    f"Using first source language only for grouped plot: {primary_src}"
                )
            pairs_src_to_all = [
                (primary_src, tgt) for tgt in langs_pool if primary_src != tgt
            ]

        primary_tgt = args.tgt_lang if args.tgt_lang_explicit else None
        if primary_tgt is None:
            pairs_all_to_tgt = []
        elif primary_tgt not in langs_pool:
            print(
                "Skipping 'all → one tgt' grouped plot: target not in available language pool."
            )
            pairs_all_to_tgt = []
        else:
            pairs_all_to_tgt = [
                (src, primary_tgt) for src in langs_pool if src != primary_tgt
            ]

        for step in args.plot_steps:
            for grouping_variant in args.grouping_variants:
                suffix = _grouping_title_suffix(grouping_variant)
                plot_per_step_grouped_aggregates(
                    step=step,
                    run_name=args.run_name,
                    figures_path=model_figures_path,
                    score_type=args.score_type,
                    aggregate_probs=args.aggregate_probs,
                    pairs=pairs_all,
                    scope_label="all72",
                    title_scope=f"All directed pairs{suffix}",
                    exclude_langs=args.exclude_langs,
                    strip_titles=not args.keep_titles,
                    grouping_variant=grouping_variant,
                    jaccard_target_only=args.jaccard_target_only,
                )

                if pairs_src_to_all:
                    plot_per_step_grouped_aggregates(
                        step=step,
                        run_name=args.run_name,
                        figures_path=model_figures_path,
                        score_type=args.score_type,
                        aggregate_probs=args.aggregate_probs,
                        pairs=pairs_src_to_all,
                        scope_label=f"src_{primary_src}_to_all",
                        title_scope=f"{LANG_NAME_EN[primary_src]} → all targets{suffix}",
                        exclude_langs=args.exclude_langs,
                        strip_titles=not args.keep_titles,
                        grouping_variant=grouping_variant,
                        jaccard_target_only=args.jaccard_target_only,
                    )

                if pairs_all_to_tgt:
                    plot_per_step_grouped_aggregates(
                        step=step,
                        run_name=args.run_name,
                        figures_path=model_figures_path,
                        score_type=args.score_type,
                        aggregate_probs=args.aggregate_probs,
                        pairs=pairs_all_to_tgt,
                        scope_label=f"all_to_tgt_{primary_tgt}",
                        title_scope=f"All sources → {LANG_NAME_EN[primary_tgt]}{suffix}",
                        exclude_langs=args.exclude_langs,
                        strip_titles=not args.keep_titles,
                        grouping_variant=grouping_variant,
                        jaccard_target_only=args.jaccard_target_only,
                    )

    if args.avgs and (args.plot_tf_langs or args.plot_grouped_time):
        print(
            "Generating grouped TF-language aggregate plots over time (--avgs)..."
        )
        excluded_langs = set(args.exclude_langs)
        langs_pool = [lang for lang in LANGS if lang not in excluded_langs]
        pairs_all = [
            (src, tgt) for src in langs_pool for tgt in langs_pool if src != tgt
        ]

        primary_src = (
            args.src_langs[0]
            if (args.src_langs_explicit and args.src_langs)
            else None
        )
        if primary_src not in langs_pool:
            pairs_src_to_all = []
        else:
            pairs_src_to_all = [
                (primary_src, tgt) for tgt in langs_pool if primary_src != tgt
            ]

        primary_tgt = args.tgt_lang if args.tgt_lang_explicit else None
        if primary_tgt not in langs_pool:
            pairs_all_to_tgt = []
            pairs_selected_to_tgt = []
        else:
            pairs_all_to_tgt = [
                (src, primary_tgt) for src in langs_pool if src != primary_tgt
            ]
            pairs_selected_to_tgt = [
                (src, primary_tgt)
                for src in args.src_langs
                if src in langs_pool and src != primary_tgt
            ]

        for grouping_variant in args.grouping_variants:
            if args.plot_tf_langs:
                plot_grouped_tf_lang_aggregates(
                    run_name=args.run_name,
                    pairs=pairs_all,
                    requested_groups=args.plot_tf_langs,
                    figures_path=model_figures_path,
                    x_axis=args.x_axis,
                    mode=args.mode,
                    score_type=args.score_type,
                    aggregate_probs=args.aggregate_probs,
                    grouping_variant=grouping_variant,
                    scope_label="all72",
                    title_scope="All directed pairs",
                    exclude_langs=args.exclude_langs,
                    strip_titles=not args.keep_titles,
                    jaccard_target_only=args.jaccard_target_only,
                )

            if args.plot_grouped_time:
                plot_grouped_time_layer_group_plot(
                    run_name=args.run_name,
                    figures_path=model_figures_path,
                    x_axis=args.x_axis,
                    mode=args.mode,
                    score_type=args.score_type,
                    aggregate_probs=args.aggregate_probs,
                    pairs=pairs_all,
                    scope_label="all72",
                    title_scope="All directed pairs",
                    exclude_langs=args.exclude_langs,
                    strip_titles=not args.keep_titles,
                    grouping_variant=grouping_variant,
                    jaccard_target_only=args.jaccard_target_only,
                )

            if pairs_src_to_all:
                if args.plot_tf_langs:
                    plot_grouped_tf_lang_aggregates(
                        run_name=args.run_name,
                        pairs=pairs_src_to_all,
                        requested_groups=args.plot_tf_langs,
                        figures_path=model_figures_path,
                        x_axis=args.x_axis,
                        mode=args.mode,
                        score_type=args.score_type,
                        aggregate_probs=args.aggregate_probs,
                        grouping_variant=grouping_variant,
                        scope_label=f"src_{primary_src}_to_all",
                        title_scope=f"{LANG_NAME_EN[primary_src]} -> all targets",
                        exclude_langs=args.exclude_langs,
                        strip_titles=not args.keep_titles,
                        jaccard_target_only=args.jaccard_target_only,
                    )

                if args.plot_grouped_time:
                    plot_grouped_time_layer_group_plot(
                        run_name=args.run_name,
                        figures_path=model_figures_path,
                        x_axis=args.x_axis,
                        mode=args.mode,
                        score_type=args.score_type,
                        aggregate_probs=args.aggregate_probs,
                        pairs=pairs_src_to_all,
                        scope_label=f"src_{primary_src}_to_all",
                        title_scope=f"{LANG_NAME_EN[primary_src]} -> all targets",
                        exclude_langs=args.exclude_langs,
                        strip_titles=not args.keep_titles,
                        grouping_variant=grouping_variant,
                        jaccard_target_only=args.jaccard_target_only,
                    )
            if pairs_all_to_tgt:
                if args.plot_tf_langs:
                    plot_grouped_tf_lang_aggregates(
                        run_name=args.run_name,
                        pairs=pairs_all_to_tgt,
                        requested_groups=args.plot_tf_langs,
                        figures_path=model_figures_path,
                        x_axis=args.x_axis,
                        mode=args.mode,
                        score_type=args.score_type,
                        aggregate_probs=args.aggregate_probs,
                        grouping_variant=grouping_variant,
                        scope_label=f"all_to_tgt_{primary_tgt}",
                        title_scope=f"All sources -> {LANG_NAME_EN[primary_tgt]}",
                        exclude_langs=args.exclude_langs,
                        strip_titles=not args.keep_titles,
                        jaccard_target_only=args.jaccard_target_only,
                    )

                if args.plot_grouped_time:
                    plot_grouped_time_layer_group_plot(
                        run_name=args.run_name,
                        figures_path=model_figures_path,
                        x_axis=args.x_axis,
                        mode=args.mode,
                        score_type=args.score_type,
                        aggregate_probs=args.aggregate_probs,
                        pairs=pairs_all_to_tgt,
                        scope_label=f"all_to_tgt_{primary_tgt}",
                        title_scope=f"All sources -> {LANG_NAME_EN[primary_tgt]}",
                        exclude_langs=args.exclude_langs,
                        strip_titles=not args.keep_titles,
                        grouping_variant=grouping_variant,
                        jaccard_target_only=args.jaccard_target_only,
                    )

            if pairs_selected_to_tgt:
                selected_srcs_label = "_".join(
                    src for src, _ in pairs_selected_to_tgt
                )
                if args.plot_tf_langs:
                    plot_grouped_tf_lang_aggregates(
                        run_name=args.run_name,
                        pairs=pairs_selected_to_tgt,
                        requested_groups=args.plot_tf_langs,
                        figures_path=model_figures_path,
                        x_axis=args.x_axis,
                        mode=args.mode,
                        score_type=args.score_type,
                        aggregate_probs=args.aggregate_probs,
                        grouping_variant=grouping_variant,
                        scope_label=f"src_{selected_srcs_label}_to_tgt_{primary_tgt}",
                        title_scope=f"Selected sources -> {LANG_NAME_EN[primary_tgt]}",
                        exclude_langs=args.exclude_langs,
                        strip_titles=not args.keep_titles,
                        jaccard_target_only=args.jaccard_target_only,
                    )

                if args.plot_grouped_time:
                    plot_grouped_time_layer_group_plot(
                        run_name=args.run_name,
                        figures_path=model_figures_path,
                        x_axis=args.x_axis,
                        mode=args.mode,
                        score_type=args.score_type,
                        aggregate_probs=args.aggregate_probs,
                        pairs=pairs_selected_to_tgt,
                        scope_label=f"src_{selected_srcs_label}_to_tgt_{primary_tgt}",
                        title_scope=f"Selected sources -> {LANG_NAME_EN[primary_tgt]}",
                        exclude_langs=args.exclude_langs,
                        strip_titles=not args.keep_titles,
                        grouping_variant=grouping_variant,
                        jaccard_target_only=args.jaccard_target_only,
                    )

    # Generate averaged margin plots
    if args.plot_margins and args.avgs:
        print("Generating averaged margin plots...")
        excluded_langs = set(args.exclude_langs)

        avg_langs = [lang for lang in LANGS if lang not in excluded_langs]
        avg_src_langs_to_tgt = [lang for lang in avg_langs if lang != args.tgt_lang]

        if not avg_langs:
            print(
                "  Skipping all-sources→target average: source pool empty after exclusions."
            )

        # Generate grouped margin plots (target vs language groups)
        print("  Generating grouped language margin plots...")
        if args.src_langs_explicit and args.tgt_lang_explicit:
            for grouping_variant in args.grouping_variants:
                plot_grouped_margin_heatmaps(
                    run_name=args.run_name,
                    file_name=file_name,
                    src_langs=args.src_langs,
                    figures_path=model_figures_path,
                    x_axis=args.x_axis,
                    mode=args.mode,
                    score_type=args.score_type,
                    aggregate_probs=args.aggregate_probs,
                    grouping_variant=grouping_variant,
                    exclude_langs=args.exclude_langs,
                    simple_margin_label=args.simple_margin_label,
                    margin_layer_diff=args.margin_layer_diff,
                    margin_time_diff=args.margin_time_diff,
                    strip_titles=not args.keep_titles,
                    grouped_margin_reference=args.grouped_margin_reference,
                    jaccard_target_only=args.jaccard_target_only,
                )
        else:
            print(
                "  Skipping grouped source->target margin plots: provide both --src_langs and --tgt_lang explicitly."
            )

        # Generate grouped margin plots for all-sources aggregation
        print("  Generating grouped language margin plots (all sources)...")
        if args.tgt_lang_explicit and avg_src_langs_to_tgt:
            for grouping_variant in args.grouping_variants:
                plot_grouped_margin_heatmaps(
                    run_name=args.run_name,
                    file_name=file_name,
                    src_langs=avg_src_langs_to_tgt,
                    figures_path=model_figures_path,
                    x_axis=args.x_axis,
                    mode=args.mode,
                    score_type=args.score_type,
                    aggregate_probs=args.aggregate_probs,
                    grouping_variant=grouping_variant,
                    exclude_langs=args.exclude_langs,
                    simple_margin_label=args.simple_margin_label,
                    margin_layer_diff=args.margin_layer_diff,
                    margin_time_diff=args.margin_time_diff,
                    strip_titles=not args.keep_titles,
                    grouped_margin_reference=args.grouped_margin_reference,
                    jaccard_target_only=args.jaccard_target_only,
                )
        elif not args.tgt_lang_explicit:
            print(
                "  Skipping grouped all-sources->target plots: provide --tgt_lang explicitly."
            )

        margin_data_all_to_all = compute_average_margin_matrix_both_directions(
            run_name=args.run_name,
            src_langs=avg_langs,
            score_type=args.score_type,
            tf_lang_margin=args.tf_lang_margin,
            tgt_langs=avg_langs,
            aggregate_probs=args.aggregate_probs,
        )
        baseline_lang = margin_data_all_to_all["baseline_lang"]
        _plot_margin_heatmap_from_data(
            margin_data=margin_data_all_to_all,
            run_name=args.run_name,
            figures_path=model_figures_path,
            x_axis=args.x_axis,
            mode=args.mode,
            score_type=args.score_type,
            title_lang="All → All (avg)",
            src_lang_for_copy=avg_langs[0],
            src_lang_label="all_to_all",
            metrics_for_suffix=file_name,
            source_averaged=False,
            tf_lang_margin=args.tf_lang_margin,
            custom_colorbar_label=(
                (
                    "Average p difference"
                    if args.aggregate_probs
                    else "Average log p difference"
                )
                if args.simple_margin_label
                else f"Average [log p(target) - log p({LANG_NAME_EN[baseline_lang]})]"
            ),
            simple_margin_label=args.simple_margin_label,
            exclude_langs=args.exclude_langs,
            aggregate_probs=args.aggregate_probs,
            margin_layer_diff=args.margin_layer_diff,
            margin_time_diff=args.margin_time_diff,
            strip_titles=not args.keep_titles,
        )

        # Grouped all-to-all margins (true all source-target directions)
        print("  Generating grouped language margin plots (all->all)...")
        for grouping_variant in args.grouping_variants:
            plot_grouped_margin_heatmaps(
                run_name=args.run_name,
                file_name=file_name,
                src_langs=avg_langs,
                tgt_langs=avg_langs,
                figures_path=model_figures_path,
                x_axis=args.x_axis,
                mode=args.mode,
                score_type=args.score_type,
                aggregate_probs=args.aggregate_probs,
                grouping_variant=grouping_variant,
                exclude_langs=args.exclude_langs,
                simple_margin_label=args.simple_margin_label,
                margin_layer_diff=args.margin_layer_diff,
                margin_time_diff=args.margin_time_diff,
                strip_titles=not args.keep_titles,
                grouped_margin_reference=args.grouped_margin_reference,
                jaccard_target_only=args.jaccard_target_only,
            )

