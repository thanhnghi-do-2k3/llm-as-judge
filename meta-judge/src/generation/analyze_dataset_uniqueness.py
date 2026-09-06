from human_feedback_datasets.processed_dataset import ProcessedDataset, Example
from pathlib import Path
from typing import Set, Tuple, List, Any, Dict

def get_unique_data(data: ProcessedDataset) -> ProcessedDataset:
    seen_inputs: Set[Tuple[str, str, str]] = set()
    unique_data = []
    for item in data:
        if (item.reference, item.input, item.context) not in seen_inputs:
            seen_inputs.add((item.reference, item.input, item.context))
            unique_data.append(item)
    return ProcessedDataset(unique_data)


if __name__ == "__main__":
    dataset_folder = Path('/mnt/proj1/fta-25-74/dev/master_thesis/datasets')
    for dataset_path in dataset_folder.rglob("*.jsonl"):
        dataset = ProcessedDataset.load(dataset_path)
        unique_dataset = get_unique_data(dataset)
        print(f"{dataset_path.stem}: Original size={len(dataset)}, Unique size={len(unique_dataset)}")