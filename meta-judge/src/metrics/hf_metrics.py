
import os

# ---- Env toggles: must be set before importing transformers / evaluate / HF hub
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# os.environ.setdefault("HF_HUB_OFFLINE", "1")      # prevent hub API calls (models must be cached)
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1") # prevents is_base_mistral() hub lookups

from typing import Any, Dict, List, Literal, Optional, Callable, Tuple
from evaluate import load, EvaluationModule
from datasets import DownloadConfig
from itertools import product
import sacrebleu as _sacrebleu
from nltk.translate.meteor_score import meteor_score as _nltk_meteor
from nltk import word_tokenize as _word_tokenize
    # if metric_name in {"bleu", "chrf", "meteor", "rouge"}:
    #     metric = evaluate.load(metric_name)
    # elif metric_name == "bertscore":
    #     metric = evaluate.load("bertscore")
    # elif metric_name == "bleurt":
    #     metric = evaluate.load("bleurt", 'bleurt-large-512')

import torch
import contextlib

# Device for GPU-based reward metrics (bertscore, bleurt, comet).
# Defaults to the LAST visible CUDA device so training (which uses cuda:0) and
# metrics land on different GPUs when multiple are allocated. Override with
# set_reward_device() from the training script.
def _default_reward_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    return f"cuda:{n - 1}" if n >= 1 else "cpu"

REWARD_DEVICE: str = _default_reward_device()


@contextlib.contextmanager
def _cuda_device_guard():
    """Restore the active CUDA device after the block exits.

    GPU-based metrics (BERTScore, BLEURT, COMET) may internally call
    torch.cuda.set_device() when running on a non-default GPU, which
    changes the global default and can cause device-mismatch errors
    in the training loop.
    """
    if torch.cuda.is_available():
        prev = torch.cuda.current_device()
        try:
            yield
        finally:
            torch.cuda.set_device(prev)
    else:
        yield


def set_reward_device(device: str) -> None:
    """Set the device used by GPU-based reward metrics (bertscore, bleurt, comet)."""
    global REWARD_DEVICE
    REWARD_DEVICE = device
    print(f"[hf_metrics] Reward metrics device set to: {REWARD_DEVICE}")


def _parse_cuda_index(device: str) -> Optional[int]:
    """Extract GPU index from a device string like 'cuda:1'. Returns None for 'cpu' or bare 'cuda'."""
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    return None


VERSION_HASH: Dict[str, str] = {
    "bleu": "9e0985c1200e367cce45605ce0ecb5ede079894e0f24f54613fca08eeb8aff76",
    "chrf": "d244bab9383988714085a8dacc4871986d9f025398581c33d6b2ee22836b4069",
    "rouge": "6e5315f72865c2eaa764c8361360bb938740b9c120a2cf3a7ad218aa0ce452ed",
    "meteor": "e7ed321a1b44c34fa4679192809db2cee7e3bd4bba0fe8b76061d807706c2374",
    "bleurt": "88cdcafd9cccc9d5927ab758a370025ca107402fa4e0cccccc70fa2add645f41",
    "bertscore": "cf4907b18f8f741f202232c0f8009a3bd49ff98802c245abcb6ea51a37a8c05b",
    "comet": "2760a223ac957f30acfb18c8aa649b01cf1d75f2",
}
CACHED_METRICS: Dict[str, EvaluationModule] = {}
def cached_load(metric_name: str, *args: Any,  cache_key: Optional[str] = None, **kwargs: Dict[str, Any]) -> EvaluationModule:
    cache_key = cache_key or metric_name
    if cache_key not in CACHED_METRICS:
        CACHED_METRICS[cache_key] = load(metric_name, *args, keep_in_memory=True, revision=VERSION_HASH[metric_name], download_config=DownloadConfig(local_files_only=True), **kwargs)
    return CACHED_METRICS[cache_key]

def bleu(ref: List[str], pred: List[str], max_order: int = 4, smooth: bool = False) -> List[Any]:
    bleu = cached_load("bleu")
    # Calling the pinned metric's calculation directly preserves its sentence
    # BLEU formula while avoiding evaluate's Arrow/cache setup for every row.
    compute = getattr(bleu, "_compute", bleu.compute)
    return [compute(predictions=[p], references=[r], max_order=max_order, smooth=smooth)['bleu'] if (p and p.strip() and r) else 0.0 for p,r in zip(pred, ref)]

def chrf(ref: List[str], pred: List[str], char_order: int = 6, word_order: int = 0, beta: int = 2, lowercase: bool = False, whitespace: bool = False, eps_smoothing: bool = False) -> List[Any]:
    metric = _sacrebleu.CHRF(char_order=char_order, word_order=word_order, beta=beta, lowercase=lowercase, whitespace=whitespace, eps_smoothing=eps_smoothing)
    return [metric.sentence_score(p, [r]).score / 100.0 if (p and r) else 0.0 for p, r in zip(pred, ref)]
def rouge(ref: List[str], pred: List[str], rouge_type: Literal["rouge1", "rouge2", "rougeL", "rougeLsum"], use_aggregator: bool = False, use_stemmer: bool = False) -> List[Any]:
    assert use_aggregator is False, "use_aggregator=True not supported in this wrapper"
    rouge = cached_load("rouge")
    scores = rouge.compute(predictions=pred, references=ref, rouge_types=[rouge_type], use_aggregator=use_aggregator, use_stemmer=use_stemmer)[rouge_type]
    assert isinstance(scores, list), f"Expected list but got {type(scores)}"
    return scores

def meteor(ref: List[str], pred: List[str], alpha: float = 0.9, beta: int = 3, gamma: float = 0.5) -> List[Any]:
    return [_nltk_meteor([_word_tokenize(r)], _word_tokenize(p), alpha=alpha, beta=beta, gamma=gamma) if (p and r) else 0.0 for p, r in zip(pred, ref)]

_BLEURT_SCORERS: Dict[str, Any] = {}
_BERTSCORE_SCORERS: Dict[str, Any] = {}

def _find_bleurt_local(checkpoint_name: str) -> str:
    """Find a pre-extracted BLEURT checkpoint across Evaluate cache layouts."""
    from pathlib import Path

    cache_home = Path(
        os.environ.get(
            "HF_HOME",
            os.path.join(
                os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                "huggingface",
            ),
        )
    )
    # Evaluate stores metric assets below HF_HOME/metrics. Keep the older
    # datasets/evaluate roots as fallbacks for caches made by earlier releases.
    search_roots = [
        cache_home / "metrics",
        cache_home / "evaluate",
        cache_home / "datasets" / "downloads" / "extracted",
    ]
    matches = []
    for root in search_roots:
        if root.is_dir():
            matches.extend(
                path
                for path in root.rglob(checkpoint_name)
                if path.is_dir() and (path / "bleurt_config.json").is_file()
            )
    if not matches:
        raise FileNotFoundError(
            f"BLEURT checkpoint '{checkpoint_name}' not found below "
            f"{', '.join(map(str, search_roots))}. "
            "Run once without HF_HUB_OFFLINE=1 to download it."
        )
    return str(sorted(matches)[0])

_BLEURT_TF_DEVICE: Dict[str, str] = {}  # per-checkpoint tf.device scope string

def bleurt(ref: List[str], pred: List[str], checkpoint: Literal["bleurt-tiny-128", "bleurt-tiny-512", "bleurt-base-128", "bleurt-base-512", "bleurt-large-128", "bleurt-large-512", "BLEURT-20-D3", "BLEURT-20-D6", "BLEURT-20-D12", "BLEURT-20"] = "bleurt-tiny-128") -> List[Any]:
    from bleurt import score as bleurt_score
    import tensorflow as tf
    gpu_idx = _parse_cuda_index(REWARD_DEVICE)
    if checkpoint not in _BLEURT_SCORERS:
        # Restrict TF to the reward GPU. set_visible_devices must be called before
        # TF initializes its GPUs; if it has, fall back to tf.device scoping.
        phys_gpus = tf.config.list_physical_devices("GPU")
        tf_device = "/CPU:0"
        if gpu_idx is not None and gpu_idx < len(phys_gpus):
            try:
                tf.config.set_visible_devices([phys_gpus[gpu_idx]], "GPU")
                tf.config.experimental.set_memory_growth(phys_gpus[gpu_idx], True)
                tf_device = "/GPU:0"  # after restriction, the reward GPU becomes logical 0
            except RuntimeError as e:
                print(f"[hf_metrics] BLEURT: TF already initialized, using tf.device scope ({e})")
                tf_device = f"/GPU:{gpu_idx}"
        _BLEURT_TF_DEVICE[checkpoint] = tf_device
        local_path = _find_bleurt_local(checkpoint)
        print(f"Loading BLEURT from local cache: {local_path} (device: {REWARD_DEVICE} → TF {tf_device})")
        with tf.device(tf_device):
            _BLEURT_SCORERS[checkpoint] = bleurt_score.BleurtScorer(local_path)
    scorer = _BLEURT_SCORERS[checkpoint]
    with _cuda_device_guard(), tf.device(_BLEURT_TF_DEVICE.get(checkpoint, "/CPU:0")):
        scores = scorer.score(references=ref, candidates=pred)
    assert isinstance(scores, list), f"Expected list but got {type(scores)}"
    return scores

def bertscore(ref: List[str], pred: List[str], lang: Optional[Literal["en", 'cz', 'sk', 'uk']] = None,
              model_type: Optional[Literal[
                  # EN
                  "roberta-large", "microsoft/deberta-xlarge-mnli", "bert-base-uncased", "bert-base-multilingual-cased"
                  # CZ
                  "ufal/robeczech-base", "UWB-AIR/Czert-A-base-uncased", "fav-kky/FERNET-C5", "DeepPavlov/bert-base-bg-cs-pl-ru-cased",
                  # SK
                  "gerulata/slovakbert",
                  # UK
                  "youscan/ukr-roberta-base", "mshamrai/bert-base-ukr-eng-rus-uncased"
                  ]] = None,
              rescale_with_baseline: bool = False, num_layers: Optional[int] = None, device: Optional[str] = None) -> List[Any]:
    from bert_score import BERTScorer
    device = device or REWARD_DEVICE
    assert (lang is None or model_type is None) and (lang is not None or model_type is not None), "Specify either lang or model_type, not both."
    cache_key = f"bertscore_{lang}_{model_type}_{num_layers}_{rescale_with_baseline}"
    if cache_key not in _BERTSCORE_SCORERS:
        _BERTSCORE_SCORERS[cache_key] = BERTScorer(
            lang=lang, model_type=model_type, num_layers=num_layers,
            rescale_with_baseline=rescale_with_baseline, device=device,
        )
    scorer = _BERTSCORE_SCORERS[cache_key]
    with _cuda_device_guard():
        _, _, F1 = scorer.score(pred, ref)
    return F1.tolist()

def comet(ref: List[str], pred: List[str], src: Optional[List[str]] = None, model_name: str = "Unbabel/wmt22-comet-da", device: Optional[str] = None) -> List[Any]:
    from comet import load_from_checkpoint
    from vn_meta_judge.metrics.comet_cache import download_comet_checkpoint
    device = device or REWARD_DEVICE
    if f"comet_{model_name}" not in CACHED_METRICS:
        ckpt = download_comet_checkpoint(model_name)
        CACHED_METRICS[f"comet_{model_name}"] = load_from_checkpoint(ckpt)
    metric = CACHED_METRICS[f"comet_{model_name}"]
    if src is None:
        src = [""] * len(ref)
    # COMET checkpoints do not all use the same input schema.  Use the
    # checkpoint's declared input segments when available; this handles both
    # reference-based and QE models without relying on the model name.
    input_segments = getattr(getattr(metric, "hparams", None), "input_segments", None)
    values = {"src": src, "mt": pred, "ref": ref}
    if input_segments:
        batches = [
            {name: values[name][index] for name in input_segments if name in values}
            for index in range(len(pred))
        ]
    elif "qe" in model_name.lower():
        batches = [{"src": s, "mt": p} for s, p in zip(src, pred)]
    else:
        batches = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(src, pred, ref)]

    # Route the PyTorch-Lightning trainer to the specific GPU. Passing gpus=1
    # without a device list lets PL grab cuda:0 — which is where training lives.
    gpu_idx = _parse_cuda_index(device)
    if gpu_idx is not None:
        pl_kwargs = {"accelerator": "gpu", "devices": [gpu_idx], "gpus": 1}
    elif device == "cuda":
        pl_kwargs = {"accelerator": "gpu", "gpus": 1}
    else:
        pl_kwargs = {"accelerator": "cpu", "gpus": 0}

    with _cuda_device_guard():
        prediction = metric.predict(
            batches, batch_size=32, progress_bar=False, **pl_kwargs
        )
    if isinstance(prediction, dict):
        scores = prediction["scores"]
    else:
        scores = prediction.scores
    assert isinstance(scores, list), f"Expected list but got {type(scores)}"
    return scores

def preload_comet() -> None:
    """Eagerly load all configured COMET models into the process cache.

    COMET's encoder (xlm-roberta-large) triggers a HuggingFace API call when its
    tokenizer is first loaded.  Pre-loading here in the main process means forked
    reward-function workers inherit the already-loaded models from CACHED_METRICS
    and never make that API call themselves.
    """
    from comet import load_from_checkpoint
    from vn_meta_judge.metrics.comet_cache import download_comet_checkpoint
    for _, m_kwargs in all_kwargs(only=["comet"]):
        model_name = m_kwargs["model_name"]
        cache_key = f"comet_{model_name}"
        if cache_key not in CACHED_METRICS:
            ckpt = download_comet_checkpoint(model_name)
            CACHED_METRICS[cache_key] = load_from_checkpoint(ckpt)


def preload_bleurt() -> None:
    """Eagerly load all configured BLEURT checkpoints into the process cache.

    Uses BleurtScorer directly from local cache to avoid network calls
    and race conditions from concurrent extraction.
    """
    from bleurt import score as bleurt_score
    for _, m_kwargs in all_kwargs(only=["bleurt"]):
        checkpoint = m_kwargs["checkpoint"]
        if checkpoint not in _BLEURT_SCORERS:
            local_path = _find_bleurt_local(checkpoint)
            print(f"Pre-loading BLEURT checkpoint: {local_path}")
            _BLEURT_SCORERS[checkpoint] = bleurt_score.BleurtScorer(local_path)


METRIC_FNS: Dict[str, Callable[..., List[Any]]] = {
    "bleu": bleu,
    "chrf": chrf,
    "rouge": rouge,
    "meteor": meteor,
    "bleurt": bleurt,
    "bertscore": bertscore,
    "comet": comet,
}

# NEW VERSION
METRICS_KWARGS: Dict[str, Dict[str, List[Any]]] = {
    "bleu": {"max_order": [1,2,3,4], "smooth": [True]},
    "chrf": {"char_order": [4,6], "word_order": [0, 2], "beta": [2], "lowercase": [False], "whitespace": [False], "eps_smoothing": [False]},
    "rouge": {"rouge_type": ["rouge1", "rouge2", "rouge4", "rougeL"]},
    "meteor": {"alpha": [0.2, 0.9], "beta": [3], "gamma": [0.0, 0.5]},
    "bleurt": {"checkpoint": ["bleurt-tiny-128", "bleurt-base-512", "bleurt-large-512", "BLEURT-20-D12"]},
    "bertscore": {"lang": ["en", "cs"], "model_type": [None], "rescale_with_baseline": [False], "num_layers": [None, 5], "device": [None]},
    "comet": {"model_name": ["Unbabel/wmt22-comet-da", "eamt22-cometinho-da", "Unbabel/wmt20-comet-da", "Unbabel/wmt20-comet-qe-da"], "device": [None]},
}

def kwargs_product(metric_name: str) -> List[Dict[str, Any]]:
    assert metric_name in METRICS_KWARGS, f"Metric {metric_name} not supported. Supported metrics: {list(METRICS_KWARGS.keys())}"
    keys = list(METRICS_KWARGS[metric_name].keys())
    values = list(METRICS_KWARGS[metric_name].values())
    return [dict(zip(keys, v)) for v in product(*values)]

def all_kwargs(only: List[str] = []) -> List[Tuple[str, Dict[str, Any]]]:
    all_combinations = []
    for metric_name in METRICS_KWARGS.keys():
        for kwargs in kwargs_product(metric_name):
            all_combinations.append((metric_name, kwargs))
    if only:
        all_combinations = [item for item in all_combinations if item[0] in only]
    return all_combinations

def kwargs_str(kwargs: Dict[str, Any]) -> str:
    return ",".join([f"{k}={v}" for k,v in kwargs.items()])

def _bleu_direct(ref: List[str], pred: List[str], max_order: int = 4, smooth: bool = False) -> List[Any]:
    metric = _sacrebleu.BLEU(max_ngram_order=max_order, smooth_method='exp' if smooth else 'none', tokenize='none')
    return [metric.sentence_score(p, [r]).score / 100.0 if (p and r) else 0.0 for p, r in zip(pred, ref)]

def _chrf_direct(ref: List[str], pred: List[str], char_order: int = 6, word_order: int = 0, beta: int = 2, lowercase: bool = False, whitespace: bool = False, eps_smoothing: bool = False) -> List[Any]:
    metric = _sacrebleu.CHRF(char_order=char_order, word_order=word_order, beta=beta, lowercase=lowercase, whitespace=whitespace, eps_smoothing=eps_smoothing)
    return [metric.sentence_score(p, [r]).score / 100.0 if (p and r) else 0.0 for p, r in zip(pred, ref)]

def _meteor_direct(ref: List[str], pred: List[str], alpha: float = 0.9, beta: int = 3, gamma: float = 0.5) -> List[Any]:
    return [_nltk_meteor([_word_tokenize(r)], _word_tokenize(p), alpha=alpha, beta=beta, gamma=gamma) if (p and r) else 0.0 for p, r in zip(pred, ref)]


def test_metrics() -> None:
    # "reference": "Остафій Дашкевич", "prediction": "Відомим першим кошовим отаманом вважається Остафій Дашкевич."
    refs = ["The cat is on the mat.", "There is a cat playing on the mat.", "Остафій Дашкевич", "Остафій Дашкевич", "Остафій Дашкевич Остафій Дашкевич Остафій Дашкевич"]
    preds = ["The cat is on mat.", "A cat plays on the mat.", "Відомим першим кошовим отаманом вважається Остафій Дашкевич.", "Остафій Дашкевич", "Остафій Дашкевич Остафій Дашкевич Остафій Дашкевич"]
    srcs = ["A cat is sitting on the mat.", "There is a cat on the mat.", "Остафій Дашкевич", "Остафій Дашкевич", "Остафій Дашкевич Остафій Дашкевич Остафій Дашкевич"]

    print("=== Comparing evaluate vs direct implementations ===")
    _DIRECT = {"bleu": _bleu_direct, "chrf": _chrf_direct, "meteor": _meteor_direct}
    for metric_name, kwargs in all_kwargs(only=["bleu", "chrf", "meteor"]):
        evaluate_scores = METRIC_FNS[metric_name](refs, preds, **kwargs)
        direct_scores   = _DIRECT[metric_name](refs, preds, **kwargs)
        max_diff = max(abs(a - b) for a, b in zip(evaluate_scores, direct_scores))
        match = "OK" if max_diff < 1e-6 else f"MISMATCH (max_diff={max_diff:.6f})"
        print(f"  {metric_name} ({kwargs_str(kwargs)}): {match}")
        if max_diff >= 1e-6:
            print(f"    evaluate: {evaluate_scores}")
            print(f"    direct:   {direct_scores}")

    print("\nTesting metrics with example predictions:")
    for metric_name, kwargs in all_kwargs(only=["bleu", "chrf", "rouge", "meteor", "bleurt", "bertscore", "comet"]):
        metric_fn = METRIC_FNS[metric_name]
        if metric_name == "comet":
            print(f"{metric_name} ({kwargs_str(kwargs)}): {metric_fn(refs, preds, srcs, **kwargs)}")
        else:
            print(f"{metric_name} ({kwargs_str(kwargs)}): {metric_fn(refs, preds, **kwargs)}")
    # print("BLEU:", bleu(refs, preds))
    # print("CHRF:", chrf(refs, preds))
    # print("ROUGE1:", rouge(refs, preds, "rouge1"))
    # print("ROUGE2:", rouge(refs, preds, "rouge2"))
    # print("ROUGE4:", rouge(refs, preds, "rouge4"))
    # print("ROUGEL:", rouge(refs, preds, "rougeL"))
    # print("METEOR:", meteor(refs, preds))
    # print("BLEURT:", bleurt(refs, preds))
    # print("BLEURT:", bleurt(refs, preds, checkpoint="bleurt-large-512"))
    # print("BERTScore (en):", bertscore(refs, preds, lang="en", device="cuda"))
    # print("COMET:", comet(refs, preds, srcs, device="cuda"))
    # print(CACHED_METRICS.keys())

if __name__ == "__main__":
    test_metrics()
