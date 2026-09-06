from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


DAMAGE_LEVELS = tuple(range(6))
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True)
class MetricSpec:
    """One metric configuration; a paper run must expose every spec in logs."""

    name: str
    kwargs: Dict[str, Any]
    family: str

    @property
    def key(self) -> str:
        args = ",".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"{self.name}_{args}" if args else self.name


def paper_metric_specs(include_bleurt: bool = True) -> List[MetricSpec]:
    """Return the paper's 7 x 4 design, adapted to Vietnamese where needed.

    Tokenization is deliberately not part of this list.  The underthesea
    run is an additional sensitivity branch, so the primary 28-configuration
    result remains comparable with the paper.
    """

    specs: List[MetricSpec] = []
    for order in (1, 2, 3, 4):
        specs.append(MetricSpec("bleu", {"max_order": order, "smooth": True}, "BLEU"))
    for char_order, word_order in ((4, 0), (4, 2), (6, 0), (6, 2)):
        specs.append(MetricSpec("chrf", {"char_order": char_order, "word_order": word_order}, "chrF"))
    for rouge_type in ("rouge1", "rouge2", "rouge4", "rougeL"):
        specs.append(MetricSpec("rouge", {"rouge_type": rouge_type}, "ROUGE"))
    for alpha, gamma in ((0.2, 0.0), (0.2, 0.5), (0.9, 0.0), (0.9, 0.5)):
        specs.append(MetricSpec("meteor", {"alpha": alpha, "beta": 3, "gamma": gamma}, "METEOR"))
    # The paper has two language settings x two depth settings.  Keep the
    # English setting as a diagnostic baseline and replace Czech with vi.
    for lang in ("en", "vi"):
        for num_layers in (None, 5):
            specs.append(MetricSpec("bertscore", {"lang": lang, "num_layers": num_layers}, "BERTScore"))
    for model_name in (
        "Unbabel/wmt22-comet-da",
        "Unbabel/eamt22-cometinho-da",
        "Unbabel/wmt20-comet-da",
        "Unbabel/wmt20-comet-qe-da",
    ):
        specs.append(MetricSpec("comet", {"model_name": model_name}, "COMET"))
    if include_bleurt:
        for checkpoint in ("bleurt-tiny-128", "bleurt-base-512", "bleurt-large-512", "BLEURT-20-D12"):
            specs.append(MetricSpec("bleurt", {"checkpoint": checkpoint}, "BLEURT"))
    return specs


def paper_metric_count(include_bleurt: bool = True) -> int:
    return len(paper_metric_specs(include_bleurt=include_bleurt))


def metric_summary(include_bleurt: bool = True) -> Dict[str, Any]:
    specs = paper_metric_specs(include_bleurt=include_bleurt)
    return {
        "metric_count": len(specs),
        "expected_paper_count": 28 if include_bleurt else 24,
        "families": sorted({spec.family for spec in specs}),
        "keys": [spec.key for spec in specs],
    }
