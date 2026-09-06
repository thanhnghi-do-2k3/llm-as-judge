from __future__ import annotations
from typing import Any, Dict, List, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class Example:
    id: str | None
    input: str
    context: str | None
    reference: str
    prediction: str
    model_name: str | None
    original_human_score: Any
    processed_human_score: float | None
    extra: Dict[str, Any] | None


class ProcessedDataset(list[Example]):
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
    
    def head(self, n: int = 10) -> ProcessedDataset:
        """Return the first n examples."""
        return ProcessedDataset(self[:n])
    
    def tail(self, n: int = 10) -> ProcessedDataset:
        """Return the last n examples."""
        return ProcessedDataset(self[-n:])
    
    def sample(self, n: int, generator: Any = None) -> ProcessedDataset:
        """
        Return a random sample of n examples.
        If generator is provided, it will be used for reproducibility.
        """
        import random
        if generator is not None:
            random.seed(generator)
        return ProcessedDataset(random.sample(self, n))
    
    def filter(self, fn: Callable[[Example], bool]) -> ProcessedDataset:
        """Return a new dataset with examples that satisfy the given function."""
        return ProcessedDataset([ex for ex in self if fn(ex)])

    def save(self, path: Path) -> None:
        """Save the dataset to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for example in self:
                json.dump(example.__dict__, f, ensure_ascii=False)
                f.write('\n')

    def column(self, name: str) -> List[Any]:
        """Return a list of values for the given column name."""
        return [getattr(example, name) for example in self]
    
    def indices(self, indices: List[int]) -> ProcessedDataset:
        """Return a new dataset with examples at the given indices."""
        return ProcessedDataset([self[i] for i in indices])

    @staticmethod
    def load(path: Path) -> ProcessedDataset:
        """Load the dataset from a JSONL file."""
        dataset = ProcessedDataset()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                dataset.append(Example(**example))
        return dataset
