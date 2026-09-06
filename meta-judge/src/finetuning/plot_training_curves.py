"""Plot training curves (reward, loss, KL, completion length, grad norm, etc.)
for all trained models.

Reads trainer_state.json from the last checkpoint of each model to check
convergence. Produces one PNG per dataset with subplots for each reward mode.

For 'sum' mode models, an additional plot is produced showing how each
individual reward component (BLEU, chrF, BERTScore, BLEURT, COMET, METEOR,
ROUGE) evolves during training, plus cross-mode meta-rewards if available.

Usage
-----
    python plot_training_curves.py [--models_dir models_trained] [--output_dir training_plots]
    python plot_training_curves.py --filter cus_qa_nonbin_sk_en  # single dataset
    python plot_training_curves.py --split-metrics                # one file per metric
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Thesis-quality style
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-safe palette
PALETTE = [
    "#E69F00", "#56B4E9", "#009E73",
    "#0072B2", "#D55E00", "#CC79A7",
    "#F0E442", "#000000",
]

THESIS_RC = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.8",
    "lines.linewidth": 1.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

# Folder name used in --split-metrics output (matches LaTeX \includegraphics paths)
METRIC_FOLDER: dict[str, str] = {
    "reward":               "reward",
    "reward_cv":            "reward_cv",
    "kl":                   "kl_div",
    "frac_reward_zero_std": "frac_reward_zero_std",
    "completion_length":    "completion_length",
    "clipped_ratio":        "clipped_ratio",
    "loss":                 "loss",
    "grad_norm":            "grad_norm",
    "terminated_length":    "terminated_length",
    "learning_rate":        "learning_rate",
}

# Short y-axis label (units only; title carries the description)
METRIC_YLABEL: dict[str, str] = {
    "reward":               "Reward",
    "reward_cv":            "CoV (σ / |μ|)",
    "kl":                   "KL divergence",
    "frac_reward_zero_std": "Fraction",
    "completion_length":    "Tokens",
    "clipped_ratio":        "Fraction",
    "loss":                 "Loss",
    "grad_norm":            "Grad norm",
    "terminated_length":    "Tokens",
    "learning_rate":        "LR",
    "clip_ratio/high_mean": "Fraction",
    "clip_ratio/low_mean":  "Fraction",
    "clip_ratio/region_mean": "Fraction",
    "completions/max_length": "Tokens",
    "completions/min_length": "Tokens",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_last_trainer_state(model_dir: Path) -> Path | None:
    """Find trainer_state.json from the highest-numbered checkpoint."""
    checkpoints = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not checkpoints:
        return None
    state_file = checkpoints[-1] / "trainer_state.json"
    return state_file if state_file.exists() else None


def _get(entry: dict, *keys, default=0):
    for k in keys:
        if k in entry:
            return entry[k]
    return default


def extract_curves(log_history: list) -> dict:
    """Extract key training metrics from log_history."""
    steps, rewards, reward_stds, kls = [], [], [], []
    comp_lens, comp_term_lens, frac_zero_std, clipped_ratios = [], [], [], []
    grad_norms, losses, learning_rates = [], [], []
    clip_high, clip_low, clip_region = [], [], []
    comp_max_lens, comp_min_lens = [], []

    component_keys: set[str] = set()

    for entry in log_history:
        if "reward" not in entry or "step" not in entry:
            continue
        steps.append(entry.get("step", 0))
        rewards.append(entry.get("reward", 0))
        reward_stds.append(entry.get("reward_std", 0))
        kls.append(entry.get("kl", 0))
        comp_lens.append(_get(entry, "completion_length", "completions/mean_length"))
        comp_term_lens.append(entry.get("completions/mean_terminated_length", 0))
        frac_zero_std.append(entry.get("frac_reward_zero_std", 0))
        clipped_ratios.append(entry.get("completions/clipped_ratio", 0))
        grad_norms.append(entry.get("grad_norm", float("nan")))
        losses.append(entry.get("loss", float("nan")))
        learning_rates.append(entry.get("learning_rate", float("nan")))
        clip_high.append(entry.get("clip_ratio/high_mean", float("nan")))
        clip_low.append(entry.get("clip_ratio/low_mean", float("nan")))
        clip_region.append(entry.get("clip_ratio/region_mean", float("nan")))
        comp_max_lens.append(entry.get("completions/max_length", float("nan")))
        comp_min_lens.append(entry.get("completions/min_length", float("nan")))

        for k in entry:
            if k.startswith("rewards/") and k.endswith("/mean"):
                component_keys.add(k)

    curves: dict = {
        "steps": np.array(steps),
        "reward": np.array(rewards),
        "reward_std": np.array(reward_stds),
        "kl": np.array(kls),
        "completion_length": np.array(comp_lens),
        "terminated_length": np.array(comp_term_lens),
        "frac_reward_zero_std": np.array(frac_zero_std),
        "clipped_ratio": np.array(clipped_ratios),
        "grad_norm": np.array(grad_norms),
        "loss": np.array(losses),
        "learning_rate": np.array(learning_rates),
        "clip_ratio/high_mean": np.array(clip_high),
        "clip_ratio/low_mean": np.array(clip_low),
        "clip_ratio/region_mean": np.array(clip_region),
        "completions/max_length": np.array(comp_max_lens),
        "completions/min_length": np.array(comp_min_lens),
        "_component_keys": sorted(component_keys),
    }

    for ck in component_keys:
        vals = [entry.get(ck, float("nan")) for entry in log_history if "reward" in entry and "step" in entry]
        curves[ck] = np.array(vals)

    return curves


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving average for noisy curves."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _plot_metric(ax, steps, values, label, color, smooth_window):
    """Plot raw (faint) + smoothed curve for one metric."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return
    ax.plot(steps, values, alpha=0.15, color=color, linewidth=0.6)
    smoothed = smooth(values, smooth_window)
    offset = (len(steps) - len(smoothed)) // 2
    smoothed_steps = steps[offset: offset + len(smoothed)]
    ax.plot(smoothed_steps, smoothed, label=label, color=color)


# Map dataset directory names to human-readable display names
_DATASET_DISPLAY: dict[str, str] = {
    "cus_qa_cz_orig": "CUS-QA Czech (orig)",
    "cus_qa_sk_orig": "CUS-QA Slovak (orig)",
    "cus_qa_ua_orig": "CUS-QA Ukrainian (orig)",
    "cus_qa_cz_en": "CUS-QA Czech (en)",
    "cus_qa_sk_en": "CUS-QA Slovak (en)",
    "cus_qa_ua_en": "CUS-QA Ukrainian (en)",

    "rose": "RoSE",
    "mocha": "MOCHA",
    
    "wmt_cs-uk": "WMT 24 Czech-Ukrainian",
    "wmt_en-cs": "WMT 24 English-Czech",
    "wmt_en-is": "WMT 24 English-Icelandic",
}


def _dataset_display_name(dataset_name: str) -> str:
    return _DATASET_DISPLAY.get(dataset_name, dataset_name)


# Map model directory base names to human-readable display names
_MODEL_DISPLAY = {
    "gemma-3-1b-it": "Gemma 3 1B",
    "gemma-3-4b-it": "Gemma 3 4B",
    "gemma-3-12b-it": "Gemma 3 12B",
}

# Reward mode tokens recognised in directory names
_REWARD_MODES = ("pca", "reg", "sum")


def _short_label(model_name: str, dataset_name: str) -> str:
    """Return a clean legend label: '<Model display name> <mode>'.

    E.g. 'gemma-3-1b-it_pca_s2000_lp0.5_kl0.05' -> 'Gemma 3 1B pca'
    Falls back to stripping only the dataset prefix if the pattern is unrecognised.
    """
    # Strip dataset prefix first
    stem = model_name
    if stem.startswith(dataset_name + "_"):
        stem = stem[len(dataset_name) + 1:]

    # Try to find a known base model and reward mode
    for base, display in _MODEL_DISPLAY.items():
        if stem.startswith(base):
            remainder = stem[len(base):].lstrip("_")
            # First token of remainder is the reward mode
            mode = remainder.split("_")[0] if remainder else ""
            if mode in _REWARD_MODES:
                return f"{display} {mode}"
            return display

    # Fallback: just return the stripped stem
    return stem


def _short_component_name(key: str) -> str:
    """Make a human-readable short name from a rewards/.../mean key."""
    name = key.removeprefix("rewards/").removesuffix("/mean").removesuffix("_reward")
    name = re.sub(r"_device_(cuda|None)", "", name)
    name = re.sub(r"model_name_", "", name)
    name = re.sub(r"model_type_None_", "", name)
    name = re.sub(r"rescale_with_baseline_False_", "", name)
    name = re.sub(r"_?checkpoint_", "-", name)
    name = re.sub(r"bleu_max_order_(\d)_smooth_True", r"BLEU-\1", name)
    name = re.sub(r"bertscore_lang_(\w+)_num_layers_(\w+)", r"BERTScore-\1-L\2", name)
    name = re.sub(r"chrf_char_order_(\d)_word_order_(\d)_beta_\d_.*", r"chrF\1\2", name)
    name = re.sub(r"meteor_alpha_([\d.]+)_beta_\d_gamma_([\d.]+)", r"METEOR-a\1-g\2", name)
    name = re.sub(r"rouge_rouge_type_(rouge\w+)", r"\1", name)
    name = re.sub(r"bleurt-", "BLEURT-", name)
    name = re.sub(r"BLEURT-20-D12", "BLEURT-20", name)
    name = re.sub(r"comet_", "COMET-", name)
    name = re.sub(r"Unbabel/", "", name)
    name = re.sub(r"per_dataset_pca_reward", "PCA-meta", name)
    name = re.sub(r"per_dataset_regression_reward", "Reg-meta", name)
    name = re.sub(r"pca_meta_reward", "PCA-global-meta", name)
    name = re.sub(r"regression_meta_reward", "Reg-global-meta", name)
    return name.strip("_")


ALL_METRICS = [
    "reward",
    "reward_cv",
    "frac_reward_zero_std",
    "kl",
    "loss",
    "grad_norm",
    "completion_length",
    "clipped_ratio",
    "terminated_length",
    "learning_rate",
    "clip_ratio/high_mean",
    "clip_ratio/low_mean",
    "clip_ratio/region_mean",
    "completions/max_length",
    "completions/min_length",
]

METRIC_TITLES = {
    "reward":                   "Mean reward",
    "reward_cv":                "Reward CoV (σ / |μ|)  —  normalised spread, comparable across reward modes",
    "frac_reward_zero_std":     "Zero-variance batch fraction  —  steps with no gradient signal",
    "kl":                       "KL divergence from reference policy",
    "loss":                     "GRPO policy loss",
    "grad_norm":                "Gradient L2 norm",
    "completion_length":        "Mean completion length (tokens, incl. truncated)",
    "clipped_ratio":            "Clipped completion ratio  —  fraction truncated at max length",
    "terminated_length":        "Mean EOS-terminated completion length (tokens)",
    "learning_rate":            "Learning rate",
    "clip_ratio/high_mean":     "PPO upper-clip ratio  —  policy update too large",
    "clip_ratio/low_mean":      "PPO lower-clip ratio",
    "clip_ratio/region_mean":   "PPO clipped-region ratio",
    "completions/max_length":   "Max completion length per batch (tokens)",
    "completions/min_length":   "Min completion length per batch (tokens)",
}

_LOG_SCALE_METRICS = {"kl"}
_UNIT_SCALE_METRICS = {
    "frac_reward_zero_std",
    "clip_ratio/high_mean", "clip_ratio/low_mean", "clip_ratio/region_mean",
}
# clipped_ratio is intentionally excluded: when most values are near zero the
# fixed [0, 1] range makes the plot look empty, so we let matplotlib auto-scale.


def _apply_axis_style(ax, metric: str, title: str, xlabel: bool = False):
    """Apply consistent axis styling for a single metric panel."""
    ylabel = METRIC_YLABEL.get(metric, metric)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4, loc="left")
    if metric in _LOG_SCALE_METRICS:
        ax.set_yscale("symlog", linthresh=0.1)
    if metric in _UNIT_SCALE_METRICS:
        ax.set_ylim(0, 1)
    if xlabel:
        ax.set_xlabel("Training step")
    ax.legend(loc="best")


# ---------------------------------------------------------------------------
# Multi-metric stacked plot (original behaviour)
# ---------------------------------------------------------------------------

def plot_dataset(dataset_name: str, model_curves: dict, output_dir: Path,
                 smooth_window: int = 10, metric_filter: list[str] | None = None,
                 show_title: bool = False):
    """Plot training curves for all models within one dataset (all metrics stacked)."""
    for curves in model_curves.values():
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(curves["reward"] != 0, curves["reward_std"] / np.abs(curves["reward"]), 0.0)
        curves["reward_cv"] = cv

    default_metrics = [
        "reward", "reward_cv", "frac_reward_zero_std", "kl",
        "loss", "grad_norm", "completion_length", "clipped_ratio",
    ]
    metrics = metric_filter if metric_filter else default_metrics
    titles = [f"{METRIC_TITLES.get(m, m)}  —  {_dataset_display_name(dataset_name)}" for m in metrics]

    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 2.6 * len(metrics)), sharex=False)
        if len(metrics) == 1:
            axes = [axes]
        if show_title:
            fig.suptitle(f"Training curves: {dataset_name}", fontsize=13, fontweight="bold")

        for ax, metric, title in zip(axes, metrics, titles):
            for i, (model_name, curves) in enumerate(sorted(model_curves.items())):
                steps = curves["steps"]
                values = curves.get(metric)
                if values is None or len(values) == 0:
                    continue
                color = PALETTE[i % len(PALETTE)]
                label = _short_label(model_name, dataset_name)
                _plot_metric(ax, steps, values, label, color, smooth_window)
            _apply_axis_style(ax, metric, title, xlabel=False)

        axes[-1].set_xlabel("Training step")
        plt.tight_layout()
        output_path = output_dir / f"{dataset_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Split-metrics plot (one file per metric, one subdirectory per metric)
# ---------------------------------------------------------------------------

def plot_dataset_split(dataset_name: str, model_curves: dict, output_dir: Path,
                       smooth_window: int = 10, metric_filter: list[str] | None = None,
                       show_title: bool = False):
    """Save one PNG per metric for a dataset, each in its own subdirectory.

    Output layout matches LaTeX \\includegraphics paths, e.g.
        output_dir/reward_cv/cus_qa_cz_en.png
        output_dir/kl_div/cus_qa_cz_en.png
    """
    for curves in model_curves.values():
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(curves["reward"] != 0, curves["reward_std"] / np.abs(curves["reward"]), 0.0)
        curves["reward_cv"] = cv

    default_metrics = [
        "reward", "reward_cv", "frac_reward_zero_std", "kl",
        "loss", "grad_norm", "completion_length", "clipped_ratio",
    ]
    metrics = metric_filter if metric_filter else default_metrics

    for metric in metrics:
        folder = METRIC_FOLDER.get(metric, metric.replace("/", "_"))
        metric_dir = output_dir / folder
        metric_dir.mkdir(parents=True, exist_ok=True)

        title = f"{METRIC_TITLES.get(metric, metric)}  —  {_dataset_display_name(dataset_name)}"

        with plt.rc_context(THESIS_RC):
            fig, ax = plt.subplots(figsize=(7, 2.0))

            has_data = False
            for i, (model_name, curves) in enumerate(sorted(model_curves.items())):
                steps = curves["steps"]
                values = curves.get(metric)
                if values is None or len(values) == 0:
                    continue
                color = PALETTE[i % len(PALETTE)]
                label = _short_label(model_name, dataset_name)
                _plot_metric(ax, steps, values, label, color, smooth_window)
                has_data = True

            if not has_data:
                plt.close()
                continue

            # Always show metric title in single-metric plots; show_title only
            # controls the dataset-level suptitle in the stacked plot.
            _apply_axis_style(ax, metric, title, xlabel=True)

            plt.tight_layout()
            output_path = metric_dir / f"{dataset_name}.png"
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close()
        print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Component reward plot (sum mode)
# ---------------------------------------------------------------------------

def plot_component_rewards(model_name: str, curves: dict, output_dir: Path, smooth_window: int = 10):
    """For a 'sum' mode model, plot each individual reward component over training."""
    ck_list = curves.get("_component_keys", [])
    if not ck_list:
        return

    steps = curves["steps"]
    n = len(ck_list)

    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(n, 1, figsize=(7, 2.6 * n), sharex=True)
        if n == 1:
            axes = [axes]
        fig.suptitle(f"Component rewards: {model_name}", fontsize=12, fontweight="bold")

        colors = plt.cm.tab20.colors

        for ax, ck, color in zip(axes, ck_list, colors):
            values = curves.get(ck, np.array([]))
            if len(values) == 0:
                continue
            short = _short_component_name(ck)
            _plot_metric(ax, steps, values, short, color, smooth_window)
            ax.set_ylabel(short, fontsize=9)
            ax.legend(loc="best")

        axes[-1].set_xlabel("Training step")
        plt.tight_layout()
        safe_name = model_name.replace("/", "_")
        output_path = output_dir / f"components_{safe_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

_GROUP_PREFIXES = ("cus_qa_nonbin", "cus_qa", "wmt", "rose", "mocha")

def _dataset_group(ds_name: str) -> str:
    for prefix in _GROUP_PREFIXES:
        if ds_name.startswith(prefix):
            return prefix
    return ds_name


def _run_matches(model_dir_name: str, run_filter: list[str] | None,
                 run_exclude: list[str] | None = None) -> bool:
    """Return True if the model dir matches any run_filter and none of run_exclude."""
    if run_exclude and any(ex in model_dir_name for ex in run_exclude):
        return False
    if not run_filter:
        return True
    return any(
        (not re.search(r"_s\d+", model_dir_name)) if f == "base" else (f in model_dir_name)
        for f in run_filter
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training curves from checkpoints")
    parser.add_argument("--models_dir", type=str, default="models_trained",
                        help="Root directory containing trained models")
    parser.add_argument("--output_dir", type=str, default="models_training_plots",
                        help="Output directory for PNG plots")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only plot datasets matching this substring")
    parser.add_argument("--exclude", type=str, nargs="+", default=None, metavar="SUBSTR",
                        help="Skip datasets whose name contains any of these substrings")
    parser.add_argument("--smooth", type=int, default=10,
                        help="Smoothing window size (default: 10)")
    parser.add_argument("--no-components", action="store_true",
                        help="Skip per-component reward plots for sum mode")
    parser.add_argument("--metrics", type=str, nargs="+", default=None, metavar="METRIC",
                        help=f"Only plot these metrics. Available: {', '.join(ALL_METRICS)}")
    parser.add_argument("--run", type=str, nargs="+", default=None,
                        help="Filter model dirs by run variant substrings (e.g. --run s1000 s2000 lp0.5). "
                             "Use 'base' to select only models without any _s<N> parameter suffix.")
    parser.add_argument("--exclude-run", type=str, nargs="+", default=None, metavar="SUBSTR",
                        help="Skip model dirs whose name contains any of these substrings "
                             "(e.g. --exclude-run parall split). Applied after --run.")
    parser.add_argument("--merge", action="store_true",
                        help="Also produce merged plots grouping datasets by prefix "
                             f"({', '.join(_GROUP_PREFIXES)}). Model labels are prefixed with the dataset name.")
    parser.add_argument("--title", action="store_true",
                        help="Show title on plots (hidden by default).")
    parser.add_argument("--split-metrics", action="store_true",
                        help="Save one PNG per metric in per-metric subdirectories "
                             "(e.g. reward_cv/cus_qa_cz_en.png). Recommended for thesis figures.")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_fn = plot_dataset_split if args.split_metrics else plot_dataset

    dataset_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])

    if args.filter:
        dataset_dirs = [d for d in dataset_dirs if args.filter in d.name]
    if args.exclude:
        dataset_dirs = [d for d in dataset_dirs if not any(ex in d.name for ex in args.exclude)]

    if not dataset_dirs:
        print(f"No dataset directories found in {models_dir}")
        return

    group_curves: dict[str, dict[str, dict]] = {}

    for ds_dir in dataset_dirs:
        dataset_name = ds_dir.name
        model_dirs = sorted([d for d in ds_dir.iterdir() if d.is_dir()])

        model_curves = {}
        for model_dir in model_dirs:
            if "final_model" in model_dir.name:
                continue
            if not _run_matches(model_dir.name, args.run, args.exclude_run):
                continue
            state_file = find_last_trainer_state(model_dir)
            if state_file is None:
                print(f"  Skipping {model_dir.name} (no trainer_state.json)")
                continue
            with open(state_file) as f:
                state = json.load(f)
            curves = extract_curves(state["log_history"])
            if len(curves["steps"]) == 0:
                print(f"  Skipping {model_dir.name} (empty log history)")
                continue
            model_curves[model_dir.name] = curves

        if not model_curves:
            print(f"Skipping dataset {dataset_name} (no valid models)")
            continue

        if args.merge:
            for model_name, curves in model_curves.items():
                group = _dataset_group(dataset_name)
                merged_label = f"{dataset_name}/{model_name}"
                group_curves.setdefault(group, {})[merged_label] = curves
        else:
            print(f"Plotting {dataset_name} ({len(model_curves)} models)...")
            plot_fn(dataset_name, model_curves, output_dir, smooth_window=args.smooth,
                    metric_filter=args.metrics, show_title=args.title)

        if not args.no_components:
            for model_name, curves in model_curves.items():
                if curves.get("_component_keys"):
                    full_name = f"{dataset_name}_{model_name}"
                    print(f"  Component rewards: {model_name} ({len(curves['_component_keys'])} metrics)...")
                    plot_component_rewards(full_name, curves, output_dir, smooth_window=args.smooth)

    if args.merge:
        print("\nGenerating merged group plots...")
        for group, merged in sorted(group_curves.items()):
            print(f"Plotting merged group: {group} ({len(merged)} series)...")
            plot_fn(f"merged_{group}", merged, output_dir, smooth_window=args.smooth,
                    metric_filter=args.metrics, show_title=args.title)

    print("\nGenerating per-reward-mode plots...")
    reward_mode_curves: dict[str, dict[str, dict]] = {}
    for ds_dir in dataset_dirs:
        dataset_name = ds_dir.name
        for model_dir in sorted(ds_dir.iterdir()):
            if not model_dir.is_dir() or "final_model" in model_dir.name:
                continue
            if not _run_matches(model_dir.name, args.run, args.exclude_run):
                continue
            state_file = find_last_trainer_state(model_dir)
            if state_file is None:
                continue
            parts = model_dir.name.rsplit("_", 1)
            if len(parts) < 2:
                continue
            mode = parts[-1]
            if mode not in ("sum", "reg", "pca"):
                continue
            with open(state_file) as f:
                state = json.load(f)
            curves = extract_curves(state["log_history"])
            if len(curves["steps"]) == 0:
                continue
            reward_mode_curves.setdefault(mode, {})[dataset_name] = curves

    for mode, ds_curves in sorted(reward_mode_curves.items()):
        if not ds_curves:
            continue
        print(f"Plotting reward mode: {mode} ({len(ds_curves)} datasets)...")
        plot_fn(f"all_datasets_{mode}", ds_curves, output_dir, smooth_window=args.smooth,
                metric_filter=args.metrics, show_title=args.title)

    print(f"\nDone. Plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
