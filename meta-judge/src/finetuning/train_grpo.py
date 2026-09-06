import os
import time

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from unsloth import FastLanguageModel

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import logging
import warnings

import transformers
# import wandb

warnings.filterwarnings("ignore", message="The `srun` command is available on your system", category=UserWarning)

transformers.logging.set_verbosity_info()
# Suppress PyTorch Lightning INFO logs (keeps WARNING and ERROR logs)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# Sometimes COMET/Lightning uses this specific rank_zero logger too:
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA as _PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from transformers import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from datasets import Dataset
from human_feedback_datasets.processed_dataset import Example, ProcessedDataset

# --- NEW PROMPT IMPORTS ---
from human_feedback_datasets.prompts import (
    CUS_QA_ZERO_SHOT_PROMPT,
    MOCHA_ZERO_SHOT_PROMPT,
    ROSE_ZERO_SHOT_PROMPT,
    WMT_ZERO_SHOT_PROMPT,
)
from metrics.hf_metrics import METRIC_FNS, all_kwargs, kwargs_str, preload_bleurt, preload_comet, set_reward_device

# ---------------------------------------------------------
# GLOBALS & TEMPLATES
# ---------------------------------------------------------
# Pattern to strip thinking tokens (e.g. <think>...</think>) from model output
THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

ROSE_USER_TEMPLATE = "source_text: {source}\nreference_summary: {reference}\ndamage_level: {damage_level}"
MOCHA_USER_TEMPLATE = "passage: {context}\nquestion: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
CUS_QA_USER_TEMPLATE = "question: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
WMT_USER_TEMPLATE = "source_sentence: {source}\nreference_translation: {reference}\ndamage_level: {damage_level}"


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def to_conv_format(dataset: ProcessedDataset, system_prompt: str, user_template: str, dataset_id: str = "default") -> List[Dict[str, Any]]:
    """Formats the dataset using the dynamically selected prompts."""
    out_list = []
    for data in dataset:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_template.format(
                    context=data.context if data.context else "", source=data.input, reference=data.reference, damage_level=data.processed_human_score
                ),
            },
        ]
        out_list.append({"prompt": messages, "reference": data.reference, "damage_level": data.processed_human_score, "dataset_id": dataset_id})
    return out_list


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return THINK_PATTERN.sub("", text).strip()


def extract_response(text: str) -> str:
    """Extracts the generated text, stripping thinking tokens."""
    return strip_thinking(text).strip()


def create_dynamic_metric_reward(metric_name: str, metric_kwargs: dict, metric_min: float, metric_max: float):
    """Factory to generate a reward function for a specific metric and its kwargs.

    metric_min/metric_max are the observed range of this metric on the training data,
    used to normalize scores to [0, 1] instead of hard-clamping.
    damage_level is always 0-5 (0=no damage/high quality, 5=max damage/low quality).
    """
    metric_range = max(metric_max - metric_min, 1e-8)
    safe_kwargs = kwargs_str(metric_kwargs).replace("=", "_").replace(",", "_")
    func_name = f"{metric_name}_{safe_kwargs}_reward".strip("_")

    def dynamic_reward_fn(completions, reference, damage_level, **kwargs):
        scores = [0.0] * len(completions)
        valid_indices, valid_refs, valid_preds, valid_levels = [], [], [], []

        for i, (completion, ref, level) in enumerate(zip(completions, reference, damage_level)):
            response = completion[0]["content"] if isinstance(completion, list) else completion
            extracted_text = extract_response(response)

            if extracted_text:
                valid_indices.append(i)
                valid_refs.append(ref)
                valid_preds.append(extracted_text)
                valid_levels.append(level)

        if valid_refs:
            _t0 = time.perf_counter()
            actual_scores = METRIC_FNS[metric_name](valid_refs, valid_preds, **metric_kwargs)
            print(f"[REWARD_TIMING] {func_name}: {time.perf_counter() - _t0:.2f}s ({len(valid_refs)} samples)", flush=True)
            for idx, actual_score, level in zip(valid_indices, actual_scores, valid_levels):
                norm_score = np.clip((float(actual_score) - metric_min) / metric_range, 0.0, 1.0)
                # damage_level 0 → quality target 1.0, damage_level 5 → quality target 0.0
                target_score = max(0.0, 1.0 - float(level) / 5.0)
                error = abs(target_score - norm_score)
                reward = 3.0 * (1.0 - error)
                scores[idx] = reward

        return scores

    print(f"Created reward function: {func_name} (metric range: [{metric_min:.4f}, {metric_max:.4f}])")
    dynamic_reward_fn.__name__ = func_name
    return dynamic_reward_fn


def get_max_prompt_length(dataset, tokenizer):
    max_len = 0
    for example in dataset:
        prompt_str = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
        if len(tokens) > max_len:
            max_len = len(tokens)
    return max_len


# ---------------------------------------------------------
# REGRESSION LOGIC
# ---------------------------------------------------------
def train_cv(X, y, n_splits=5, prune_threshold=0.0):
    feature_names = list(X.keys())
    X_matrix = np.array([X[name] for name in feature_names]).T
    y_array = np.array(y)

    valid_indices = [i for i, val in enumerate(y_array) if val is not None]
    X_matrix = X_matrix[valid_indices]
    y_array = y_array[valid_indices].astype(float)

    print("\n--- Data Prep ---")
    print(f"Training on {len(y_array)} valid examples using {X_matrix.shape[1]} metrics.")
    print(f"Evaluating with {n_splits}-Fold Cross-Validation.")

    individual_corrs = []
    for i, name in enumerate(feature_names):
        single_metric_preds = X_matrix[:, i]
        sp_corr, _ = spearmanr(y_array, single_metric_preds)
        pe_corr, _ = pearsonr(y_array, single_metric_preds)
        individual_corrs.append((name, sp_corr, pe_corr))

    individual_corrs.sort(key=lambda x: x[1], reverse=True)
    print("\n--- Individual Metric Performance (Baseline) ---")
    for name, sp, pe in individual_corrs:
        print(f"Spearman: {sp:+.4f} | Pearson: {pe:+.4f} | {name}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_predictions = np.zeros_like(y_array)

    for train_idx, test_idx in kf.split(X_matrix):
        X_train, X_test = X_matrix[train_idx], X_matrix[test_idx]
        y_train = y_array[train_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        cv_predictions[test_idx] = model.predict(X_test_scaled)

    spearman_corr, _ = spearmanr(y_array, cv_predictions)
    pearson_corr, _ = pearsonr(y_array, cv_predictions)

    print("\n--- META-METRIC Model Performance (Out-of-Fold CV) ---")
    print(f"Spearman Correlation: {spearman_corr:+.4f}")
    print(f"Pearson Correlation:  {pearson_corr:+.4f}")

    best_single_sp = individual_corrs[0][1]
    improvement = spearman_corr - best_single_sp
    print(f"Improvement over best single metric: {improvement:+.4f}")

    print("\n--- Final Learned Weights (Fit on all data) ---")
    final_scaler = StandardScaler()
    X_scaled_all = final_scaler.fit_transform(X_matrix)

    final_model = Ridge(alpha=1.0)
    final_model.fit(X_scaled_all, y_array)

    sorted_weights = sorted(zip(feature_names, final_model.coef_), key=lambda x: abs(x[1]), reverse=True)
    for name, weight in sorted_weights:
        print(f"{weight:+.4f} : {name}")
    print(f"{final_model.intercept_:+.4f} : (Intercept)\n")

    # Prune low-weight features and refit
    if prune_threshold > 0.0:
        keep_mask = np.abs(final_model.coef_) >= prune_threshold
        kept_names = [n for n, keep in zip(feature_names, keep_mask) if keep]
        dropped = [n for n, keep in zip(feature_names, keep_mask) if not keep]
        print(f"\n--- Pruning {len(dropped)} metrics with |weight| < {prune_threshold} ---")
        for name in dropped:
            print(f"  dropped: {name}")
        print(f"  kept: {len(kept_names)} / {len(feature_names)} metrics")

        X_pruned = X_matrix[:, keep_mask]
        final_scaler = StandardScaler()
        X_pruned_scaled = final_scaler.fit_transform(X_pruned)
        final_model = Ridge(alpha=1.0)
        final_model.fit(X_pruned_scaled, y_array)
        feature_names = kept_names

    return final_model, final_scaler, feature_names


_CPU_METRICS = {"bleu", "chrf", "rouge", "meteor"}


def _train_single_model(ds_data: ProcessedDataset, prune_threshold: float):
    """Train a Ridge regression model on one dataset.

    Returns (model, scaler, feature_configs, y_min, y_range).
    """
    X: Dict[str, list] = {}
    y, refs, preds = [], [], []

    for d in ds_data:
        refs.append(d.reference)
        preds.append(d.prediction)
        y.append(d.processed_human_score)

    y_min = min(y)
    y_max = max(y)
    y_range = max(y_max - y_min, 1e-8)
    print(f"Label range: [{y_min:.4f}, {y_max:.4f}]")

    feature_configs = []
    for m_name, m_kwargs in all_kwargs():
        safe_kwargs = kwargs_str(m_kwargs).replace("=", "_").replace(",", "_")
        feature_name = f"{m_name}_{safe_kwargs}"
        feature_configs.append((feature_name, m_name, m_kwargs))
        scores = METRIC_FNS[m_name](refs, preds, **m_kwargs)
        X[feature_name] = scores

    final_model, final_scaler, kept_feature_names = train_cv(X, y, prune_threshold=prune_threshold)

    kept_set = set(kept_feature_names)
    feature_configs = [(fn, mn, mk) for fn, mn, mk in feature_configs if fn in kept_set]

    return final_model, final_scaler, feature_configs, y_min, y_range


def _build_metric_groups(feature_configs):
    """Split feature_configs into CPU-threadable and GPU-sequential groups."""
    groups: dict = defaultdict(list)
    for j, (_, m_name, m_kwargs) in enumerate(feature_configs):
        groups[m_name].append((j, m_name, m_kwargs))
    cpu = {k: v for k, v in groups.items() if k in _CPU_METRICS}
    gpu = {k: v for k, v in groups.items() if k not in _CPU_METRICS}
    return cpu, gpu


def _score_batch(valid_refs, valid_preds, valid_levels, valid_indices, model_artefacts, scores_out):
    """Score one group of examples using a trained Ridge model, writing into scores_out."""
    model, scaler, feature_configs, y_min, y_range = model_artefacts
    cpu_groups, gpu_groups = _build_metric_groups(feature_configs)

    X_batch = np.zeros((len(valid_refs), len(feature_configs)))

    def _run_group(tasks):
        results = []
        for j, m_name, m_kwargs in tasks:
            results.append((j, METRIC_FNS[m_name](valid_refs, valid_preds, **m_kwargs)))
        return results

    # CPU metrics: run each metric name's variants in a dedicated thread
    with ThreadPoolExecutor(max_workers=max(1, len(cpu_groups))) as executor:
        futures = [executor.submit(_run_group, tasks) for tasks in cpu_groups.values()]
        for future in as_completed(futures):
            for j, m_scores in future.result():
                X_batch[:, j] = m_scores

    # GPU metrics: run sequentially to avoid CUDA contention
    for tasks in gpu_groups.values():
        for j, m_name, m_kwargs in tasks:
            X_batch[:, j] = METRIC_FNS[m_name](valid_refs, valid_preds, **m_kwargs)

    X_batch_scaled = scaler.transform(X_batch)
    predicted_scores = model.predict(X_batch_scaled)

    for idx, pred_score, true_level in zip(valid_indices, predicted_scores, valid_levels):
        norm_pred = np.clip((pred_score - y_min) / y_range, 0.0, 1.0)
        # damage_level 0 → quality target 1.0, damage_level 5 → quality target 0.0
        norm_true = max(0.0, 1.0 - float(true_level) / 5.0)
        error = abs(norm_pred - norm_true)
        scores_out[idx] = 3.0 * (1.0 - error)


# ---------------------------------------------------------
# PCA LOGIC (unsupervised — no human judgments)
# ---------------------------------------------------------
class _PC1Scorer:
    """Wraps PCA to expose a .predict() interface compatible with _score_batch."""

    def __init__(self, pca):
        self.pca = pca

    def predict(self, X):
        return self.pca.transform(X)[:, 0]


def _train_pca_model(ds_data: ProcessedDataset):
    """Fit PCA on metric scores (unsupervised — no human judgments used).

    Returns (scorer, scaler, feature_configs, pc1_min, pc1_range) matching
    the artefact tuple from _train_single_model so _score_batch can be reused.
    """
    X: Dict[str, list] = {}
    refs, preds = [], []

    for d in ds_data:
        refs.append(d.reference)
        preds.append(d.prediction)

    feature_configs = []
    for m_name, m_kwargs in all_kwargs():
        safe_kwargs = kwargs_str(m_kwargs).replace("=", "_").replace(",", "_")
        feature_name = f"{m_name}_{safe_kwargs}"
        feature_configs.append((feature_name, m_name, m_kwargs))
        scores = METRIC_FNS[m_name](refs, preds, **m_kwargs)
        X[feature_name] = scores

    feature_names = list(X.keys())
    X_matrix = np.array([X[name] for name in feature_names]).T

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_matrix)

    pca = _PCA(n_components=1)
    pc1_scores = pca.fit_transform(X_scaled)[:, 0]

    print(f"\n--- PCA Analysis (unsupervised) ---")
    print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.4f}")

    # Ensure PC1 is oriented so higher = better quality.
    # Convention: if PC1 correlates negatively with the mean of all
    # standardized metrics, flip it.
    mean_metrics = X_scaled.mean(axis=1)
    if np.corrcoef(pc1_scores, mean_metrics)[0, 1] < 0:
        pca.components_ *= -1
        pc1_scores *= -1
        print("(Flipped PC1 to align with positive quality direction)")

    print("\n--- PC1 Loadings ---")
    loadings = sorted(zip(feature_names, pca.components_[0]), key=lambda x: abs(x[1]), reverse=True)
    for name, loading in loadings:
        print(f"  {loading:+.4f} : {name}")

    pc1_min = float(pc1_scores.min())
    pc1_max = float(pc1_scores.max())
    pc1_range = max(pc1_max - pc1_min, 1e-8)
    print(f"\nPC1 score range: [{pc1_min:.4f}, {pc1_max:.4f}]")

    scorer = _PC1Scorer(pca)
    return scorer, scaler, feature_configs, pc1_min, pc1_range


def build_pca_reward(
    dataset_objects: ProcessedDataset,
    per_dataset_map: Dict[str, ProcessedDataset] | None = None,
):
    """Build a PCA-based reward function (unsupervised — no human judgments).

    Uses the first principal component of standardized metrics as a quality proxy.
    """
    if per_dataset_map:
        artefacts: Dict[str, tuple] = {}
        for ds_id, ds_data in per_dataset_map.items():
            print(f"\n{'='*60}")
            print(f"Fitting PCA model for dataset: {ds_id}")
            print(f"{'='*60}")
            artefacts[ds_id] = _train_pca_model(ds_data)

        def per_dataset_pca_reward(completions, reference, damage_level, dataset_id, **kwargs):
            scores_out = [0.0] * len(completions)
            buckets: Dict[str, tuple] = defaultdict(lambda: ([], [], [], []))
            for i, (completion, ref, level, ds) in enumerate(zip(completions, reference, damage_level, dataset_id)):
                response = completion[0]["content"] if isinstance(completion, list) else completion
                extracted_text = extract_response(response)
                if extracted_text:
                    idx_list, ref_list, pred_list, lvl_list = buckets[ds]
                    idx_list.append(i)
                    ref_list.append(ref)
                    pred_list.append(extracted_text)
                    lvl_list.append(level)

            for ds_id, (indices, refs, preds, levels) in buckets.items():
                if ds_id not in artefacts:
                    print(f"WARNING: No PCA model for dataset_id={ds_id}, skipping {len(indices)} examples")
                    continue
                _score_batch(refs, preds, levels, indices, artefacts[ds_id], scores_out)

            return scores_out

        return per_dataset_pca_reward

    else:
        print("--- Pre-computing metrics for PCA Model ---")
        single_artefacts = _train_pca_model(dataset_objects)

        def pca_meta_reward(completions, reference, damage_level, **kwargs):
            scores_out = [0.0] * len(completions)
            valid_indices, valid_refs, valid_preds, valid_levels = [], [], [], []

            for i, (completion, ref, level) in enumerate(zip(completions, reference, damage_level)):
                response = completion[0]["content"] if isinstance(completion, list) else completion
                extracted_text = extract_response(response)
                if extracted_text:
                    valid_indices.append(i)
                    valid_refs.append(ref)
                    valid_preds.append(extracted_text)
                    valid_levels.append(level)

            if valid_refs:
                _score_batch(valid_refs, valid_preds, valid_levels, valid_indices, single_artefacts, scores_out)

            return scores_out

        return pca_meta_reward


def build_regression_reward(
    dataset_objects: ProcessedDataset,
    prune_threshold: float = 0.0,
    per_dataset_map: Dict[str, ProcessedDataset] | None = None,
):
    """Build a Ridge-regression-based reward function.

    If per_dataset_map is provided, trains a separate model per dataset_id.
    Otherwise trains a single model on the combined dataset_objects.
    """
    if per_dataset_map:
        # --- Per-dataset mode: one model per dataset file ---
        artefacts: Dict[str, tuple] = {}
        for ds_id, ds_data in per_dataset_map.items():
            print(f"\n{'='*60}")
            print(f"Training regression model for dataset: {ds_id}")
            print(f"{'='*60}")
            artefacts[ds_id] = _train_single_model(ds_data, prune_threshold)

        def per_dataset_regression_reward(completions, reference, damage_level, dataset_id, **kwargs):
            scores_out = [0.0] * len(completions)

            # Group examples by dataset_id
            buckets: Dict[str, tuple] = defaultdict(lambda: ([], [], [], []))
            for i, (completion, ref, level, ds) in enumerate(zip(completions, reference, damage_level, dataset_id)):
                response = completion[0]["content"] if isinstance(completion, list) else completion
                extracted_text = extract_response(response)
                if extracted_text:
                    idx_list, ref_list, pred_list, lvl_list = buckets[ds]
                    idx_list.append(i)
                    ref_list.append(ref)
                    pred_list.append(extracted_text)
                    lvl_list.append(level)

            for ds_id, (indices, refs, preds, levels) in buckets.items():
                if ds_id not in artefacts:
                    print(f"WARNING: No regression model for dataset_id={ds_id}, skipping {len(indices)} examples")
                    continue
                _score_batch(refs, preds, levels, indices, artefacts[ds_id], scores_out)

            return scores_out

        return per_dataset_regression_reward

    else:
        # --- Single-model mode: one model on all data ---
        print("--- Pre-computing metrics for Regression Model ---")
        single_artefacts = _train_single_model(dataset_objects, prune_threshold)

        def regression_meta_reward(completions, reference, damage_level, **kwargs):
            scores_out = [0.0] * len(completions)
            valid_indices, valid_refs, valid_preds, valid_levels = [], [], [], []

            for i, (completion, ref, level) in enumerate(zip(completions, reference, damage_level)):
                response = completion[0]["content"] if isinstance(completion, list) else completion
                extracted_text = extract_response(response)
                if extracted_text:
                    valid_indices.append(i)
                    valid_refs.append(ref)
                    valid_preds.append(extracted_text)
                    valid_levels.append(level)

            if valid_refs:
                _score_batch(valid_refs, valid_preds, valid_levels, valid_indices, single_artefacts, scores_out)

            return scores_out

        return regression_meta_reward


def build_length_penalty_reward(max_ratio: float = 2.0, alpha: float = 1.0):
    """Return a reward function that penalises completions much longer than their reference.

    Penalty = clip(-alpha * max(0, pred_words / ref_words - max_ratio), min=-3.0).
    Clipped to -3.0 (the max task reward) so a single outlier doesn't dominate the
    group variance and drown out the quality signal after sum_then_normalize.
    """
    def length_penalty_reward(completions, reference, damage_level, **kwargs):
        scores = []
        for completion, ref in zip(completions, reference):
            response = completion[0]["content"] if isinstance(completion, list) else completion
            extracted = extract_response(response)
            if not extracted:
                scores.append(0.0)
                continue
            pred_words = len(extracted.split())
            ref_words = max(1, len(ref.split()))
            ratio = pred_words / ref_words
            penalty = -alpha * max(0.0, ratio - max_ratio)
            scores.append(max(-3.0, penalty))  # cap at -3.0 to match task reward range
        return scores

    length_penalty_reward.__name__ = f"length_penalty_r{max_ratio}_a{alpha}"
    print(f"Length penalty reward: max_ratio={max_ratio}, alpha={alpha}")
    return length_penalty_reward


# ---------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Fine-Tuning Script")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a run_config.json from a previous run. "
        "All settings are loaded from the file; any additional CLI flags override them.",
    )
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.2-1B-Instruct", help="Model name or local path (e.g., unsloth/Llama-3.2-3B-Instruct)")
    parser.add_argument("--dataset", type=str, choices=["rose", "cus_qa", "cus_qa_nonbin", "mocha", "wmt"], default="rose", help="Select the dataset group to load.")
    parser.add_argument(
        "--wmt_pair",
        type=str,
        choices=["en-is", "cs-uk", "en-cs"],
        default=None,
        help="Language pair for WMT dataset training. Required when --dataset=wmt. "
        "Maps to datasets/wmt_2025/wmt25_{pair}.jsonl.",
    )
    parser.add_argument(
        "--cus_qa_lang",
        type=str,
        choices=["cz_orig", "cz_en", "sk_orig", "sk_en", "ua_orig", "ua_en"],
        default=None,
        help="Hold-out language for cus_qa/cus_qa_nonbin dataset. Trains on all language files EXCEPT "
        "the specified one (leave-one-out). If omitted, all files are loaded.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Custom base output dir. Defaults to trained_models/<dataset>")
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=["sum", "reg", "pca"],
        default="sum",
        help="Reward combination mode: 'sum' (individual min-max normalized metrics), "
        "'reg' (Ridge regression on human judgments), "
        "'pca' (first principal component, unsupervised).",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional name for this run. Replaces the auto-generated folder name.")
    parser.add_argument(
        "--prune_metrics",
        type=float,
        default=0.0,
        help="When using --reward_mode=reg, drop metrics whose absolute Ridge weight "
        "is below this threshold. 0.0 keeps all metrics. "
        "Example: --prune_metrics 0.01 drops near-zero-weight features.",
    )
    parser.add_argument("--max_prompt_tokens", type=int, default=None, help="Drop training examples whose prompt exceeds this many tokens. Defaults to max_seq_length - max_completion_length (256) - 64 padding.")
    parser.add_argument("--reward_device", type=str, default=None, help="Device for GPU-based reward metrics (e.g. 'cuda:1'). Defaults to the training device.")
    parser.add_argument("--length_penalty_alpha", type=float, default=0.0, help="Strength of length penalty reward. 0.0 disables it. Recommended starting value: 0.5.")
    parser.add_argument("--length_penalty_max_ratio", type=float, default=2.0, help="Pred/ref word ratio above which the length penalty kicks in (default: 2.0).")
    parser.add_argument("--kl_beta", type=float, default=0.0, help="KL penalty coefficient (beta) against the reference model. 0.0 disables it. Recommended starting value: 0.01.")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device train batch size. Must be divisible by num_generations (4).")
    parser.add_argument("--grad_acc", type=int, default=2, help="Gradient accumulation steps. Effective batch = batch_size * grad_acc.")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt in GRPO.")
    parser.add_argument(
        "--wmt_parallel_examples",
        type=int,
        default=None,
        help="When --dataset=wmt, number of parallel bitext examples to add to training. "
        "Defaults to None (use all available). Set to 0 to disable parallel data.",
    )
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps. -1 means use num_train_epochs instead.")
    parser.add_argument("--save_steps", type=int, default=250, help="Save a checkpoint every N steps.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs (ignored when --max_steps > 0).")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help="Path to a checkpoint directory to resume from, or 'latest' to auto-detect the latest checkpoint in output_dir. Defaults to 'latest' (starts fresh if no checkpoint exists).",
    )
    parser.add_argument(
        "--wmt_split",
        action="store_true",
        default=False,
        help="When set, use sentence-split WMT data (*_split.jsonl) for GRPO training prompts "
        "while keeping the document-level data for regression/PCA fitting.",
    )

    # If --config is provided, load the saved config and inject as argparse defaults
    # so that CLI flags still take priority over config values.
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        config_path = Path(preliminary.config)
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")
        with open(config_path) as f:
            saved = json.load(f)
        print(f"Loaded config from: {config_path}")

        # Only override keys that our argparse actually defines
        config_keys = {"model", "dataset", "output_dir", "reward_mode",
                       "prune_metrics", "run_name", "wmt_pair", "cus_qa_lang",
                       "max_prompt_tokens", "batch_size", "grad_acc", "num_generations",
                       "wmt_parallel_examples", "wmt_split"}
        defaults = {k: v for k, v in saved.items() if k in config_keys}
        # Backward compat: old configs with use_regression → reward_mode
        if "use_regression" in saved and "reward_mode" not in saved:
            defaults["reward_mode"] = "reg" if saved["use_regression"] else "sum"

        # Auto-resume from latest checkpoint unless explicitly overridden
        if preliminary.resume_from_checkpoint is None:
            defaults["resume_from_checkpoint"] = "latest"

        parser.set_defaults(**defaults)

    return parser.parse_args()


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    # --- PYTORCH & vLLM STABILITY FIXES ---
    # os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"
    # os.environ["TORCH_COMPILE_DISABLE"] = "1"

    args = parse_args()

    # --- DYNAMIC FOLDER NAMING ---
    # Handle both HF IDs ("unsloth/Llama-3.2-3B-Instruct") and local cache paths
    # (".../models--unsloth--Llama-3.2-1B-Instruct/snapshots/abc123/")
    _cache_match = re.search(r"models--[^/]+--([^/]+)", args.model)
    short_model_name = _cache_match.group(1) if _cache_match else args.model.rstrip("/").split("/")[-1]
    reg_str = args.reward_mode
    folder_name = f"{short_model_name}_{reg_str}"
    if args.dataset == "wmt" and args.wmt_parallel_examples != 0:
        par_tag = f"par{args.wmt_parallel_examples}" if args.wmt_parallel_examples else "parall"
        folder_name = f"{folder_name}_{par_tag}"
    if args.max_steps > 0:
        folder_name = f"{folder_name}_s{args.max_steps}"
    else:
        folder_name = f"{folder_name}_e{args.num_train_epochs}"
    if args.run_name:
        folder_name = f"{folder_name}_{args.run_name}"
    if args.length_penalty_alpha > 0.0:
        folder_name = f"{folder_name}_lp{args.length_penalty_alpha}"
    if args.kl_beta > 0.0:
        folder_name = f"{folder_name}_kl{args.kl_beta}"
    if args.dataset == "wmt" and args.wmt_split:
        folder_name = f"{folder_name}_split"

    if args.output_dir:
        base_out_dir = Path(args.output_dir)
    elif args.dataset == "wmt":
        base_out_dir = Path("models_trained") / f"wmt_{args.wmt_pair}" / folder_name
    elif args.dataset in ("cus_qa", "cus_qa_nonbin") and args.cus_qa_lang:
        base_out_dir = Path("models_trained") / f"{args.dataset}_{args.cus_qa_lang}" / folder_name
    else:
        base_out_dir = Path("models_trained") / args.dataset / folder_name
    base_out_dir = base_out_dir.resolve()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    final_save_path = Path(base_out_dir) / f"{folder_name}_final_model"
    if final_save_path.exists():
        print(f"Final model already exists at {final_save_path}. Skipping training.")
        return

    print("--- Starting GRPO Training ---")
    print(f"Model: {args.model}")
    print(f"Dataset Group: {args.dataset}")
    if args.dataset == "wmt":
        print(f"WMT Language Pair: {args.wmt_pair}")
    if args.dataset in ("cus_qa", "cus_qa_nonbin"):
        print(f"CUS QA Hold-Out: {args.cus_qa_lang or 'none (all included)'}")
    print(f"Output Directory: {base_out_dir}")
    print(f"Reward Mode: {args.reward_mode}")
    print(f"Prune Metrics Below Weight: {args.prune_metrics}")
    print("------------------------------\n")

    max_seq_length = 3_000 + 64 + 256
    lora_rank = 32

    # --- SAVE CONFIG EARLY (so it's available for resume on crash) ---
    def _save_run_config(save_dir: Path):
        run_config = {
            "model": args.model,
            "dataset": args.dataset,
            "wmt_pair": args.wmt_pair,
            "cus_qa_lang": args.cus_qa_lang,
            "output_dir": str(base_out_dir),
            "reward_mode": args.reward_mode,
            "prune_metrics": args.prune_metrics,
            "run_name": args.run_name,
            "batch_size": args.batch_size,
            "grad_acc": args.grad_acc,
            "num_generations": args.num_generations,
            "max_prompt_tokens": args.max_prompt_tokens,
            "wmt_parallel_examples": args.wmt_parallel_examples,
            "wmt_split": args.wmt_split,
            "lora_rank": lora_rank,
            "max_seq_length": max_seq_length,
        }
        with open(save_dir / "run_config.json", "w") as f:
            json.dump(run_config, f, indent=2, default=str)

    _save_run_config(base_out_dir)

    # --- DYNAMIC TEMPLATE SELECTION ---
    if args.dataset == "rose":
        current_system_prompt = ROSE_ZERO_SHOT_PROMPT
        current_user_template = ROSE_USER_TEMPLATE
    elif args.dataset in ("cus_qa", "cus_qa_nonbin"):
        current_system_prompt = CUS_QA_ZERO_SHOT_PROMPT
        current_user_template = CUS_QA_USER_TEMPLATE
    elif args.dataset == "mocha":
        current_system_prompt = MOCHA_ZERO_SHOT_PROMPT
        current_user_template = MOCHA_USER_TEMPLATE
    elif args.dataset == "wmt":
        if not args.wmt_pair:
            raise ValueError("--wmt_pair is required when --dataset=wmt (choices: en-is, cs-uk, en-cs)")
        current_system_prompt = WMT_ZERO_SHOT_PROMPT
        current_user_template = WMT_USER_TEMPLATE
    else:
        raise ValueError(f"No prompt template defined for dataset: {args.dataset!r}")

    base_dir = Path("/mnt/proj1/fta-25-74/dev/master_thesis/datasets")

    dataset_mappings = {
        "rose": [
            base_dir / "rose_cnndm_test.jsonl",
            base_dir / "rose_cnndm_protocol.jsonl",
            base_dir / "rose_cnndm_protocol_gpt3.jsonl",
            base_dir / "rose_xsum.jsonl",
        ],
        "cus_qa": [
            base_dir / "qa_text-CZ_dev_orig_binary.jsonl",
            base_dir / "qa_text-CZ_dev_en_binary.jsonl",
            base_dir / "qa_text-SK_dev_orig_binary.jsonl",
            base_dir / "qa_text-SK_dev_en_binary.jsonl",
            base_dir / "qa_text-UA_dev_orig_binary.jsonl",
            base_dir / "qa_text-UA_dev_en_binary.jsonl",
        ],
        "cus_qa_nonbin": [
            base_dir / "qa_text-CZ_dev_orig.jsonl",
            base_dir / "qa_text-CZ_dev_en.jsonl",
            base_dir / "qa_text-SK_dev_orig.jsonl",
            base_dir / "qa_text-SK_dev_en.jsonl",
            base_dir / "qa_text-UA_dev_orig.jsonl",
            base_dir / "qa_text-UA_dev_en.jsonl",
        ],
        "mocha": [base_dir / "mocha_train.jsonl"],
    }

    # WMT maps language pair to the specific file
    # wmt_parallel_file is loaded separately — used for training only, not PCA/regression fitting
    wmt_parallel_file = None
    wmt_split_training_paths = None
    if args.dataset == "wmt":
        wmt_file_map = {
            "en-is": base_dir / "wmt_2025" / "wmt25_en-is_IS.jsonl",
            "cs-uk": base_dir / "wmt_2025" / "wmt25_cs-uk_UA.jsonl",
            "en-cs": base_dir / "wmt_2025" / "wmt25_en-cs_CZ.jsonl",
        }
        dataset_mappings["wmt"] = [wmt_file_map[args.wmt_pair]]

        if args.wmt_split:
            wmt_split_file_map = {
                "en-is": base_dir / "wmt_2025" / "wmt25_en-is_IS_split.jsonl",
                "cs-uk": base_dir / "wmt_2025" / "wmt25_cs-uk_UA_split.jsonl",
                "en-cs": base_dir / "wmt_2025" / "wmt25_en-cs_CZ_split.jsonl",
            }
            wmt_split_training_paths = [(wmt_file_map[args.wmt_pair].stem, wmt_split_file_map[args.wmt_pair])]

        wmt_parallel_map = {
            "en-is": base_dir / "wmt_2025" / "wmt25_parallel_eng-isl.jsonl",
            "cs-uk": base_dir / "wmt_2025" / "wmt25_parallel_ces-ukr.jsonl",
            "en-cs": base_dir / "wmt_2025" / "wmt25_parallel_eng-ces.jsonl",
        }
        if args.wmt_parallel_examples != 0:
            pf = wmt_parallel_map[args.wmt_pair]
            if pf.exists():
                wmt_parallel_file = pf
            else:
                print(f"WARNING: parallel bitext file not found: {pf}, skipping parallel data")

    # cus_qa / cus_qa_nonbin: leave-one-out — hold out the specified language, train on the rest
    if args.dataset in ("cus_qa", "cus_qa_nonbin") and args.cus_qa_lang:
        suffix = "_binary" if args.dataset == "cus_qa" else ""
        cus_qa_file_map = {
            "cz_orig": base_dir / f"qa_text-CZ_dev_orig{suffix}.jsonl",
            "cz_en":   base_dir / f"qa_text-CZ_dev_en{suffix}.jsonl",
            "sk_orig": base_dir / f"qa_text-SK_dev_orig{suffix}.jsonl",
            "sk_en":   base_dir / f"qa_text-SK_dev_en{suffix}.jsonl",
            "ua_orig": base_dir / f"qa_text-UA_dev_orig{suffix}.jsonl",
            "ua_en":   base_dir / f"qa_text-UA_dev_en{suffix}.jsonl",
        }
        held_out = cus_qa_file_map[args.cus_qa_lang]
        all_paths_before = dataset_mappings[args.dataset]
        assert held_out in all_paths_before, (
            f"Hold-out file {held_out} not found in dataset_mappings['{args.dataset}']. "
            f"Available: {[p.name for p in all_paths_before]}"
        )
        dataset_mappings[args.dataset] = [p for p in all_paths_before if p != held_out]
        assert held_out not in dataset_mappings[args.dataset], f"Hold-out file {held_out} still present after filtering!"
        assert len(dataset_mappings[args.dataset]) == len(all_paths_before) - 1, (
            f"Expected {len(all_paths_before) - 1} training files after hold-out, got {len(dataset_mappings[args.dataset])}"
        )
        print(f"{args.dataset} leave-one-out: holding out {args.cus_qa_lang} ({held_out.name}), training on {len(dataset_mappings[args.dataset])} files")

    dataset_paths = dataset_mappings[args.dataset]

    # Load each file separately so we can track dataset_id per file
    raw_datasets_by_id: Dict[str, ProcessedDataset] = {}
    raw_dataset = ProcessedDataset()
    for path in dataset_paths:
        ds_id = path.stem  # e.g. "qa_text-CZ_dev_orig_binary"
        loaded = ProcessedDataset.load(path)
        raw_datasets_by_id[ds_id] = loaded
        raw_dataset: ProcessedDataset = raw_dataset + loaded

    # When --wmt_split is set, load the sentence-split dataset for GRPO training prompts.
    # The non-split raw_dataset/raw_datasets_by_id is still used for reg/PCA fitting.
    if wmt_split_training_paths:
        training_datasets_by_id: Dict[str, ProcessedDataset] = {}
        for ds_id, path in wmt_split_training_paths:
            print(f"Loading split training data from: {path}")
            training_datasets_by_id[ds_id] = ProcessedDataset.load(path)
    else:
        training_datasets_by_id = raw_datasets_by_id

    # Deduplicate on (input, reference) per dataset_id and expand each pair into levels 0-5
    damage_levels = [0, 1, 2, 3, 4, 5]
    all_conv_rows = []
    total_unique = 0

    for ds_id, ds_data in training_datasets_by_id.items():
        unique_pairs = {}
        for d in ds_data:
            key = (d.input, d.reference)
            if key not in unique_pairs:
                unique_pairs[key] = d

        total_unique += len(unique_pairs)
        expanded = ProcessedDataset()
        for d in unique_pairs.values():
            for level in damage_levels:
                expanded.append(
                    Example(
                        id=d.id,
                        input=d.input,
                        context=d.context,
                        reference=d.reference,
                        prediction=d.prediction,
                        model_name=d.model_name,
                        original_human_score=d.original_human_score,
                        processed_human_score=float(level),
                        extra=d.extra,
                    )
                )
        all_conv_rows.extend(to_conv_format(expanded, current_system_prompt, current_user_template, dataset_id=ds_id))

    print(f"Unique (input, reference) pairs: {total_unique}")
    print(f"Expanded training examples (x{len(damage_levels)} levels): {len(all_conv_rows)}")

    # --- Load WMT parallel bitext for additional training data ---
    if wmt_parallel_file is not None:
        print(f"\nLoading parallel bitext from: {wmt_parallel_file}")
        parallel_ds = ProcessedDataset.load(wmt_parallel_file)
        max_par = args.wmt_parallel_examples
        if max_par is not None and len(parallel_ds) > max_par:
            parallel_ds = ProcessedDataset(parallel_ds[:max_par])
            print(f"  Capped to {max_par} examples")

        # Deduplicate against the human-evaluated pairs already loaded
        existing_pairs = set()
        for ds_data in raw_datasets_by_id.values():
            for d in ds_data:
                existing_pairs.add((d.input, d.reference))

        parallel_unique = {}
        for d in parallel_ds:
            key = (d.input, d.reference)
            if key not in existing_pairs and key not in parallel_unique:
                parallel_unique[key] = d

        # Expand to damage levels and add to training rows
        parallel_expanded = ProcessedDataset()
        for d in parallel_unique.values():
            for level in damage_levels:
                parallel_expanded.append(
                    Example(
                        id=d.id,
                        input=d.input,
                        context=d.context,
                        reference=d.reference,
                        prediction=d.prediction,
                        model_name=d.model_name,
                        original_human_score=d.original_human_score,
                        processed_human_score=float(level),
                        extra=d.extra,
                    )
                )
        ds_id_parallel = wmt_parallel_file.stem
        all_conv_rows.extend(to_conv_format(parallel_expanded, current_system_prompt, current_user_template, dataset_id=ds_id_parallel))
        print(f"  Parallel unique pairs (after dedup): {len(parallel_unique)}")
        print(f"  Parallel expanded examples: {len(parallel_expanded)}")
        print(f"  Total training examples: {len(all_conv_rows)}")

    # --- Filter out examples with prompts exceeding max_prompt_tokens ---
    # Default: max_seq_length - max_completion_length(256) - 64(padding)
    if args.max_prompt_tokens is None:
        args.max_prompt_tokens = max_seq_length - 256 - 64
        print(f"max_prompt_tokens not set, defaulting to {args.max_prompt_tokens} (max_seq_length - 256 - 64)")

    # Also computes the max prompt length to avoid a second tokenization pass later.
    _cached_max_prompt_length = None
    if args.max_prompt_tokens:
        from transformers import AutoTokenizer as _AT
        _tok = _AT.from_pretrained(args.model)
        before = len(all_conv_rows)
        filtered_rows = []
        _max_len = 0
        for row in all_conv_rows:
            n = len(_tok.apply_chat_template(row["prompt"], tokenize=True))
            if n <= args.max_prompt_tokens:
                filtered_rows.append(row)
                _max_len = max(_max_len, n)
        all_conv_rows = filtered_rows
        _cached_max_prompt_length = _max_len
        dropped = before - len(all_conv_rows)
        print(f"Filtered prompts > {args.max_prompt_tokens} tokens: dropped {dropped}/{before}, kept {len(all_conv_rows)}")
        del _tok

    dataset = Dataset.from_list(all_conv_rows)

    # --- REWARD DEVICE ---
    if args.reward_device:
        set_reward_device(args.reward_device)

    # --- REWARD FUNCTION SETUP ---
    reward_functions_list = []

    per_ds = raw_datasets_by_id if (args.dataset in ("cus_qa", "cus_qa_nonbin") and len(raw_datasets_by_id) > 1) else None

    if args.reward_mode == "reg":
        if per_ds:
            print("\n[Regression] Training separate Ridge model per dataset file (cus_qa)...")
            reward_fn = build_regression_reward(raw_dataset, prune_threshold=args.prune_metrics, per_dataset_map=per_ds)
        else:
            print("\n[Regression] Training single Ridge model on combined dataset...")
            reward_fn = build_regression_reward(raw_dataset, prune_threshold=args.prune_metrics)
        reward_functions_list.append(reward_fn)

    elif args.reward_mode == "pca":
        if per_ds:
            print("\n[PCA] Fitting separate PCA model per dataset file (cus_qa)...")
            reward_fn = build_pca_reward(raw_dataset, per_dataset_map=per_ds)
        else:
            print("\n[PCA] Fitting single PCA model on combined dataset...")
            reward_fn = build_pca_reward(raw_dataset)
        reward_functions_list.append(reward_fn)

    else:  # sum
        print("\n[Sum] Using individual min-max normalized metric rewards...")
        refs = [d.reference for d in raw_dataset]
        preds = [d.prediction for d in raw_dataset]
        print("Pre-computing metric ranges on training data...")
        for m_name, m_kwargs in all_kwargs():
            m_scores = METRIC_FNS[m_name](refs, preds, **m_kwargs)
            m_min, m_max = float(min(m_scores)), float(max(m_scores))
            reward_functions_list.append(create_dynamic_metric_reward(
                m_name, m_kwargs, m_min, m_max
            ))

    if args.length_penalty_alpha > 0.0:
        reward_functions_list.append(
            build_length_penalty_reward(
                max_ratio=args.length_penalty_max_ratio,
                alpha=args.length_penalty_alpha,
            )
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        load_in_fp8=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    if _cached_max_prompt_length is not None:
        actual_max_length = _cached_max_prompt_length
    else:
        actual_max_length = get_max_prompt_length(dataset, tokenizer)
    print(f"\nThe longest prompt in dataset is: {actual_max_length} tokens.")

    per_device_train_batch_size = args.batch_size
    gradient_accumulation_steps = args.grad_acc
    num_generations = args.num_generations
    assert per_device_train_batch_size % num_generations == 0, \
        f"batch_size ({per_device_train_batch_size}) must be divisible by num_generations ({num_generations})."
    print(f"per_device_train_batch_size: {per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"num_generations: {num_generations}")
    print(f"effective_batch_size: {per_device_train_batch_size * gradient_accumulation_steps}")
    assert per_device_train_batch_size * gradient_accumulation_steps == 64, "Effective batch size (batch_size * grad_acc) should ideally be around 64 for stable training."

    max_prompt_length = actual_max_length + 64
    max_completion_length = 256
    total_needed = max_prompt_length + max_completion_length
    if total_needed > max_seq_length:
        raise ValueError(
            f"max_prompt_length ({max_prompt_length}) + max_completion_length ({max_completion_length}) "
            f"= {total_needed} exceeds max_seq_length ({max_seq_length}). "
            f"Use --max_prompt_tokens to filter long examples or increase max_seq_length."
        )

    training_args = GRPOConfig(
        temperature=0.8,
        learning_rate=5e-6,
        weight_decay=0.01,
        optim="adamw_8bit",
        # steps_per_generation=steps_per_generation,
        logging_steps=5,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        # save_strategy = "epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        # save_total_limit=10,
        # report_to="no", # "wandb"
        output_dir=base_out_dir,  # Intermediate epochs save here automatically
        seed=42,
        # --- length / repetition mitigations ---
        mask_truncated_completions=True,  # exclude clipped completions from the loss
        beta=args.kl_beta,                # KL penalty against reference model
        use_vllm=True,
        vllm_gpu_memory_utilization=0.4,
        # eval_strategy="steps",
        # eval_steps=100
    )

    # Split dataset
    # dataset = dataset.train_test_split(test_size=0.1, seed=42)
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["test"]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions_list,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset = eval_dataset
    )

    trainer.remove_callback(PrinterCallback)
    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = get_last_checkpoint(base_out_dir)  # None if no checkpoint exists
    original_dir = os.getcwd()
    os.chdir(base_out_dir)
    try:
        trainer.train(resume_from_checkpoint=resume)
    finally:
        os.chdir(original_dir)

    # --- EXPLICIT FINAL SAVE ---
    # This creates the clean "models_trained/rose/final_model" folder
    # instead of just throwing it loosely into the root directory.
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    # Save run config alongside the final model
    _save_run_config(final_save_path)

    print(f"\n✅ Training complete! Final model successfully saved to: {final_save_path}")

    # vLLM registers a custom CUDA allocator that conflicts with PyTorch's GC order
    # during interpreter shutdown, causing a spurious crash with nonzero exit code.
    # os._exit bypasses the destructor chain since all work is already done.
    os._exit(0)


if __name__ == "__main__":
    main()