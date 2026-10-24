import os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from constants import LANGS, CACHE_DIR

def get_9_lang_palette():
    # Standard Okabe-Ito + Wine + Dark Teal - Black (for avgs)
    return [
        "#56B4E9",  # Sky Blue
        "#E69F00",  # Orange
        "#009E73",  # Bluish Green
        "#D55E00",  # Vermillion
        "#882255",  # Wine
        "#0072B2",  # Blue
        "#CC79A7",  # Reddish Purple
        "#005A5B",  # Dark Teal
        "#F0E442",  # Yellow (Amber)
    ]

def get_color_lang_mapping():
    colors = get_9_lang_palette()
    return {lang: colors[i % len(colors)] for i, lang in enumerate(LANGS)}


def get_output_classification_colors():
    """Shared colors for output classification plots.

    Matches categorical classification plots:
    other=black, copy=orange, copy_correct=wine,
    context_copying=blue, concept_wrong_lang=vermillion, correct=bluish green.
    """
    return {
        "correct": "#009E73",
        "copy": "#E69F00",
        "copy_correct": "#882255",
        "context_copying": "#0072B2",
        "concept_wrong_lang": "#D55E00",
        "other": "#000000",
    }


def prepare_plotting():

    plt.rcParams.update(
        {
            "figure.figsize": (8, 4),
            "font.family": "serif",
            "font.size": 17,
            "axes.labelsize": 17,
            "legend.fontsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "lines.linewidth": 1,  #
            "lines.markersize": 4,
            "figure.autolayout": True,  # Same as tight_layout()
            "pdf.fonttype": 42,  #  Embeds fonts properly
        }
    )
    plt.rcParams["font.serif"] = ["Liberation Serif"]

def _load_token_mapping_for_plotting(run_name: str) -> dict:
    """Load step-to-tokens mapping from collated tokens file for plotting."""
    from collate_fn import get_collate_name_tokens

    cache_path = os.path.join(CACHE_DIR, get_collate_name_tokens(run_name))
    if not os.path.exists(cache_path):
        raise ValueError(
            f"Token mapping cache not found at {cache_path}; run collate_fn.py --run_name {run_name} --collate tokens first."
        )
    with open(cache_path, "r") as f:
        annotated = json.load(f)
    return {a["step"]: a["tokens_cumulative"] for a in annotated}


def get_x_values(steps: list, step_to_tokens: dict, x_axis: str = "steps") -> list:
    """Convert steps to tokens if x_axis='tokens'."""
    if x_axis == "tokens":
        return [step_to_tokens.get(s, s) for s in steps]
    return steps


def get_x_label(x_axis: str = "steps") -> str:
    """Get x-axis label."""
    return "Tokens Processed" if x_axis == "tokens" else "Training Step"


def list_results_dirs(results_dir):
    """List all result directories with epoch and step information."""
    ckpts = []
    for d in os.listdir(results_dir):
        if not d.startswith("epoch"):
            continue
        epochstr, stepstr = d.split("-")
        ckpts.append(
            {
                "epoch": int(epochstr[5:]),
                "step": int(stepstr[4:]),
                "path": os.path.join(results_dir, d),
            }
        )
    keys = ["epoch", "step"]
    ckpts.sort(key=lambda x: tuple(x[k] for k in keys))
    return ckpts


def load_results_dirs(run_dir, run_name, results_name="results"):
    """List the per-checkpoint results dirs of a run (rescanned every time, so newly
    evaluated checkpoints are always included)."""
    results_dir = os.path.join(run_dir, results_name)
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found at {results_dir}")
    return list_results_dirs(results_dir)


def save_figure(
    fig, path, formats=None, dpi=300, bbox_inches="tight", strip_titles=True
):
    """
    Save figure in multiple formats (PNG and/or PDF).
    """
    if formats is None:
        formats = ["png", "pdf"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Remove titles for paper plots unless explicitly disabled with --keep-titles
    if strip_titles:
        for ax in fig.get_axes():
            ax.set_title("")

    for fmt in formats:
        full_path = f"{path}.{fmt}"
        fig.savefig(full_path, format=fmt, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved figure to {full_path}")
    plt.close(fig)


def get_lang_from_logit_lens_file_name(file_name):
    step1 = file_name.split("_")
    lang = step1[-1].split(".")[0]
    return lang


def get_plot_suffix(
    run_name: str,
    exclude_langs: list = None,
    x_axis: str = "steps",
    benchmark: str = "wlt",
    metrics: str = None,
    score_type: str = "scores",
    aggregate_probs: bool = False,
    accuracy_mode: str = "original",
) -> tuple:
    """
    Construct standard filename components for plots.
    Returns:
        Tuple of (base_prefix, suffix) where suffix includes score_type, prob_suffix, x_axis, exclude_langs, aggregate_probs
    """
    model_prefix = "ours"
    base_prefix =  f"{model_prefix}_{benchmark}"
    

    # Detect probability type from metrics
    prob_suffix = ""
    if metrics:
        is_prob_norm = "probability_normalized" in metrics
        is_prob = ("probability" in metrics) and (not is_prob_norm)
        if is_prob_norm:
            prob_suffix = "_prob_normalized"
        elif is_prob:
            prob_suffix = "_prob"
        elif "accuracy" in metrics and accuracy_mode != "original":
            prob_suffix = f"_accuracy_{accuracy_mode}"
        elif metrics in ["acc_norm", "acc"]:
            prob_suffix = f"_{metrics}"

    score_type_suffix = "" if score_type == "scores" else f"_{score_type}"
    x_axis_suffix = f"_{x_axis}" if x_axis == "tokens" else ""
    exclude_suffix = f"_no-{'-'.join(exclude_langs)}" if exclude_langs else ""
    agg_probs_suffix = "_agg_probs" if aggregate_probs else ""

    suffix = f"{score_type_suffix}{prob_suffix}{x_axis_suffix}{exclude_suffix}{agg_probs_suffix}"

    return base_prefix, suffix


def format_token_value(val: float) -> str:
    """Format large token counts into compact units (B/M/K)."""
    abs_val = abs(val)
    if abs_val >= 1e15:
        return f"{val/1e15:.1f}P"
    if abs_val >= 1e12:
        return f"{val/1e12:.1f}T"
    if abs_val >= 1e9:
        return f"{val/1e9:.1f}B"
    if abs_val >= 1e6:
        return f"{val/1e6:.1f}M"
    if abs_val >= 1e3:
        return f"{val/1e3:.0f}K"
    return str(int(val))


def apply_token_formatter(ax, x_axis: str = "steps"):
    """Apply compact formatter to x-axis when plotting tokens."""
    if x_axis == "tokens":
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, pos: format_token_value(v))
        )
