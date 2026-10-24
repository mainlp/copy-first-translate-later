import json
import os

from matplotlib import pyplot as plt
import numpy as np

from collate_fn import (
    collate_results_mblimp,
    get_collate_name_mblimp,
)
from constants import (
    CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUN_NAME,
    LANG_NAME_EN,
    MULTI_BLIMP_LANGS,
)
from plot_utils import (
    get_color_lang_mapping,
    _load_token_mapping_for_plotting,
    apply_token_formatter,
    get_plot_suffix,
    get_x_label,
    get_x_values,
    load_results_dirs,
    prepare_plotting,
    save_figure,
)

LANG_TO_BLIMP = {
    "en": "MultiBLiMP",
    "de": "MultiBLiMP",
    "fr": "MultiBLiMP",
    "it": "MultiBLiMP",
    "es": "MultiBLiMP",
    "pt": "MultiBLiMP",
    "zh": "ZhoBLiMP",
    "ja": "JBLiMP",
}

DATASET_STYLE = {
    "MultiBLiMP": "-",
    "ZhoBLiMP": "-",
    "JBLiMP": "-",
}


def plot_mblimp_per_language(
    run_name,
    metric="acc_norm",
    figures_path="figures",
    x_axis="steps",
    mode="expl-training",
    strip_titles=True,
):
    pred_collated_file = os.path.join(
        CACHE_DIR,
        get_collate_name_mblimp(run_name),
    )
    if not os.path.exists(pred_collated_file):
        print(f"Collated predictions file not found at {pred_collated_file}, skipping.")
        return

    with open(pred_collated_file, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    color_map = get_color_lang_mapping()
    prepare_plotting()
    langs = pred_data["languages"]

    step_to_tokens = (
        _load_token_mapping_for_plotting(run_name) if x_axis == "tokens" else {}
    )

    fig, ax = plt.subplots( figsize=(5, 3.5))

    pred_means = {lang: [] for lang in langs}
    pred_stds = {lang: [] for lang in langs}
    steps = []

    for entry in pred_data["steps"]:
        steps.append(entry["step"])
        for lang in langs:
            lang_data = entry["language_scores"][lang]
            pred_means[lang].append(lang_data.get(metric, np.nan))
            pred_stds[lang].append(lang_data.get(f"{metric}_stderr", np.nan))

    x_vals = get_x_values(steps, step_to_tokens, x_axis)

    lines = {}
    for lang in langs:
        dataset = LANG_TO_BLIMP.get(lang, "multiblimp")
        linestyle = DATASET_STYLE.get(dataset, "-")

        means = np.array(pred_means[lang], dtype=float)
        stds = np.array(pred_stds[lang], dtype=float)

        lower = np.clip(means - stds, 0, 1)
        upper = np.clip(means + stds, 0, 1)

        (line,) = ax.plot(
            x_vals,
            means,
            color=color_map[lang],
            linewidth=1.5,
            linestyle=linestyle,
        )
        ax.fill_between(x_vals, lower, upper, color=color_map[lang], alpha=0.12)
        lines[lang] = line

    ax.set_xlabel(get_x_label(x_axis))
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)

    multi_langs = MULTI_BLIMP_LANGS
    non_multi_langs = [lang for lang in langs if lang not in multi_langs]

    multi_langs = sorted(multi_langs, key=lambda lang: LANG_NAME_EN[lang])
    non_multi_langs = sorted(non_multi_langs, key=lambda lang: LANG_NAME_EN[lang])

    handles = []
    labels = []

    for lang in multi_langs:
        handles.append(lines[lang])
        labels.append(LANG_NAME_EN[lang])

    for lang in non_multi_langs:
        handles.append(lines[lang])
        labels.append(LANG_NAME_EN[lang])

    ax.set_xlim(left=min(x_vals), right=max(x_vals))
    ax.set_ylim(bottom=0.5, top=1.0)
    apply_token_formatter(ax, x_axis)


    if x_axis == "tokens":
        milestone_tokens = 11_000_000_000
        x_min, x_max = ax.get_xlim()
        if x_min <= milestone_tokens <= x_max:
            ax.axvline(
                milestone_tokens,
                color="black",
                zorder=5,
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )
    base_prefix, suffix = get_plot_suffix(
        run_name,
        exclude_langs=None,
        x_axis=x_axis,
        metrics=metric,
        benchmark="mblimp",
    )

    path = os.path.join(figures_path, f"{base_prefix}{suffix}")
    save_figure(fig, path, strip_titles=strip_titles)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot MBLIMP results.")
    parser.add_argument(
        "--run_name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Name of the run to plot results for.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of collated results.",
    )
    parser.add_argument(
        "--figures_path",
        type=str,
        default="figures",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="acc_norm",
        help="Metric to plot (default: acc_norm).",
        choices=["acc", "acc_norm"],
    )
    parser.add_argument(
        "--x_axis",
        type=str,
        choices=["steps", "tokens"],
        default="steps",
        help="X-axis for time-series plots: steps or tokens processed.",
    )
    parser.add_argument(
        "--keep_titles",
        action="store_true",
        help="Keep plot titles when saving figures (by default, titles are removed for paper plots).",
    )
    args = parser.parse_args()
    output_dir = DEFAULT_OUTPUT_DIR
    run_dir = os.path.join(output_dir, args.run_name)
    figures_path = os.path.join(args.figures_path, args.run_name, "mblimp")
    os.makedirs(figures_path, exist_ok=True)
    results_dirs = load_results_dirs(
        run_dir, run_name=args.run_name, 
    )
    collate_results_mblimp(
        results_dirs,
        run_name=args.run_name,
        force=args.force,
    )
    plot_mblimp_per_language(
        args.run_name,
        figures_path=figures_path,
        metric=args.metric,
        x_axis=args.x_axis,
        strip_titles=not args.keep_titles,
    )


