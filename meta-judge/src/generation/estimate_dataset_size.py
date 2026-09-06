from human_feedback_datasets.processed_dataset import ProcessedDataset
from pathlib import Path

def get_unique_data(data: ProcessedDataset) -> ProcessedDataset:
    seen_inputs = set()
    unique_data = []
    for item in data:
        if item.input not in seen_inputs:
            seen_inputs.add(item.input)
            unique_data.append(item)
    return ProcessedDataset(unique_data)

if __name__ == "__main__":
    dataest_folder = Path("/mnt/proj1/fta-25-74/dev/master_thesis/datasets")
    for path in dataest_folder.glob("*.jsonl"):
        dataset = ProcessedDataset.load(path)
        dataset = get_unique_data(dataset)
        print(f"Dataset: {path.name}, number of samples: {len(dataset)}")