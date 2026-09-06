from pathlib import Path
from typing import Set, Tuple, Iterable, List
from human_feedback_datasets.processed_dataset import ProcessedDataset, Example
from human_feedback_datasets.datasets_loading import load_mocha, load_rose, load_qa, load_wmt_jsonl
import argparse

def group(data: ProcessedDataset) -> Iterable[List[Example]]:
    grouped: List[Example] = []
    current_index = None
    
    for item in data:
        index = item.id.split("_")[0]
        if index != current_index:
            if grouped:
                yield grouped
            grouped = [item]
            current_index = index
        else:
            grouped.append(item)
    if grouped:
        yield grouped

if __name__=="__main__":
    same_counts = {}

    d_folder = Path("/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/")
    for dataset_file in d_folder.rglob("*.jsonl"):
        dataset = ProcessedDataset.load(dataset_file)
        same_counts = {}
        print(f"Dataset: {dataset_file.name}, length: {len(dataset)}")
        for grouped_items in group(dataset):
            predictions = [item.prediction for item in grouped_items]
            full_count = len(predictions)
            unique_count = len(set(predictions))
            diff = full_count - unique_count
            if diff not in same_counts:
                same_counts[diff] = 0
            same_counts[diff] += 1

        total = sum(same_counts.values())
        print(f"  Same predictions counts: { {k: round(v/total, 2) for k,v in sorted(same_counts.items(), key=lambda item: item[0])} }")