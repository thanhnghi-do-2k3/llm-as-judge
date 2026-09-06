from pathlib import Path
from typing import Dict, Set, Tuple
from human_feedback_datasets.processed_dataset import ProcessedDataset, Example
from human_feedback_datasets.datasets_loading import load_mocha, load_rose, load_qa, load_wmt_jsonl
import argparse

def get_unique_data(data: ProcessedDataset) -> ProcessedDataset:
    seen_inputs: Set[str] = set()
    unique_data = []
    for item in data:
        if item.reference not in seen_inputs:
            seen_inputs.add(item.reference)
            unique_data.append(item)
    return ProcessedDataset(unique_data)

def get_unique_data2(data: ProcessedDataset) -> ProcessedDataset:
    seen_inputs: Set[Tuple[str, str, str | None]] = set()
    unique_data = []
    indices = []
    for item in data:
        if (item.reference, item.input, item.context) not in seen_inputs:
            seen_inputs.add((item.reference, item.input, item.context))
            unique_data.append(item)
    return ProcessedDataset(unique_data)

def get_source_index(example: Example) -> int:
    print(example.extra.keys())
    return int(example.extra['source_index'])
def get_source_indices(dataset: ProcessedDataset) -> Set[int]:
    return set(get_source_index(ex) for ex in dataset)

def get_level(example: Example) -> int:
    return int(example.id.split('_level')[1])
def get_index(example: Example) -> int:
    return int(example.id.split('_level')[0])

def missing_indices(curr_indices: Set[int], expected_indices: Set[int]) -> Dict[str, Set[int]]:
    missing = expected_indices - curr_indices
    extra = curr_indices - expected_indices
    return {
        'missing': missing,
        'extra': extra,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset integrity via lengths or indices.")
    parser.add_argument(
        "--mode", 
        choices=["lens", "indices"], 
        default="lens", 
        help="Toggle between checking dataset lengths ('lens') or source indices presence ('indices')."
    )
    args = parser.parse_args()
    
    wmt_datasets = load_wmt_jsonl(Path('/mnt/proj1/fta-25-74/dev/master_thesis/wmt24_esa.jsonl'))
    # dataset_folder = Path('/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets')
    # dataset_folder = Path('/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets_reasoning')
    dataset_folder = Path('/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets_reasoning_sample')

    # --- MODE: LENS ---
    if args.mode == "lens":
        print("--- Mode: Checking Lengths ---")
        
        # Only compute lengths if mode is lens
        REFERENCE_DATA = {
            "qa_uk_orig": len(get_unique_data2(load_qa('text-UA', split='dev', lang='orig'))),
            "qa_uk_en": len(get_unique_data2(load_qa('text-UA', split='dev', lang='en'))),
            "qa_cz_orig": len(get_unique_data2(load_qa('text-CZ', split='dev', lang='orig'))),
            "qa_cz_en": len(get_unique_data2(load_qa('text-CZ', split='dev', lang='en'))),
            "qa_sk_orig": len(get_unique_data2(load_qa('text-SK', split='dev', lang='orig'))),
            "qa_sk_en": len(get_unique_data2(load_qa('text-SK', split='dev', lang='en'))),

            # "mocha_minimal_pairs": len(get_unique_data2(load_mocha('minimal_pairs'))),
            # "mocha_train": len(get_unique_data2(load_mocha('train'))),
            # "mocha_test": len(get_unique_data2(load_mocha('test'))),
            "mocha_validation": len(get_unique_data2(load_mocha('validation'))),

            # "rose_cnndm_validation": len(get_unique_data2(load_rose('cnndm_validation'))),
            "rose_cnndm_test": len(get_unique_data2(load_rose('cnndm_test'))),
            # "rose_cnndm_protocol": len(get_unique_data2(load_rose('cnndm_protocol'))),
            # "rose_cnndm_protocol_gpt3": len(get_unique_data2(load_rose('cnndm_protocol_gpt3'))),
            # "rose_xsum": len(get_unique_data2(load_rose('xsum'))),

            "wmt_24_cs_uk": len(get_unique_data2(wmt_datasets['cs-uk'])),
            "wmt_24_en_cs": len(get_unique_data2(wmt_datasets['en-cs'])),
            "wmt_24_en_is": len(get_unique_data2(wmt_datasets['en-is'])),

            
        }

        for dataset_name in REFERENCE_DATA.keys():
            print(f"Dataset: {dataset_name}")
            for file in sorted(dataset_folder.rglob(f'*{dataset_name}*.jsonl')):
                dataset = ProcessedDataset.load(file)
                
                max_index = get_index(dataset[-1])
                min_level = min([get_level(ex) for ex in dataset])
                max_level = max([get_level(ex) for ex in dataset])
                indices = set(get_index(ex) for ex in dataset)
                
                expected_len = REFERENCE_DATA[dataset_name]
                levels_len = max_level - min_level + 1
                
                # Check logic
                actual_len = (max_index + 1) * levels_len
                target_len = expected_len * levels_len
                is_ok = actual_len == target_len
                
                print(f"  File: {file.name}: {len(dataset)} examples, source indices {len(indices)}, "
                      f"max index {max_index}, (min, max) levels: {(min_level, max_level)} -> "
                      f"{actual_len} len, expected len {target_len}; "
                      f"{'OK' if is_ok else 'MISMATCH'}")

    # --- MODE: INDICES ---
    elif args.mode == "indices":
        print("--- Mode: Checking Indices ---")
        
        # Only compute indices if mode is indices
        REFERENCE_DATA = {
            "qa_uk_orig": get_unique_data2(load_qa('text-UA', split='dev', lang='orig')),
            "qa_uk_en": get_unique_data2(load_qa('text-UA', split='dev', lang='en')),
            "qa_cz_orig": get_unique_data2(load_qa('text-CZ', split='dev', lang='orig')),
            "qa_cz_en": get_unique_data2(load_qa('text-CZ', split='dev', lang='en')),
            "qa_sk_orig": get_unique_data2(load_qa('text-SK', split='dev', lang='orig')),
            "qa_sk_en": get_unique_data2(load_qa('text-SK', split='dev', lang='en')),
            
            # "mocha_minimal_pairs": get_unique_data2(load_mocha('minimal_pairs')),
            # "mocha_train": get_unique_data2(load_mocha('train')),
            "mocha_test": get_unique_data2(load_mocha('test')),
            # "mocha_validation": get_unique_data2(load_mocha('validation')),

            # "rose_cnndm_validation": get_unique_data2(load_rose('cnndm_validation')),
            "rose_cnndm_test": get_unique_data2(load_rose('cnndm_test')),
            # "rose_cnndm_protocol": get_unique_data2(load_rose('cnndm_protocol')),
            # "rose_cnndm_protocol_gpt3": get_unique_data2(load_rose('cnndm_protocol_gpt3')),
            # "rose_xsum": get_unique_data2(load_rose('xsum')),

            "wmt_24_cs_uk": get_unique_data2(wmt_datasets['cs-uk']),
            "wmt_24_en_cs": get_unique_data2(wmt_datasets['en-cs']),
            "wmt_24_en_is": get_unique_data2(wmt_datasets['en-is']),
        }
        def get_triplet(example: Example) -> Tuple[str, str, str | None]:
            return (example.reference, example.input, example.context)
    
        for dataset_name in REFERENCE_DATA.keys():
            print(f"Dataset: {dataset_name}")
            for file in sorted(dataset_folder.rglob(f'*{dataset_name}*.jsonl')):
                dataset = ProcessedDataset.load(file)
                expected_dataset = REFERENCE_DATA[dataset_name]
                assert len(expected_dataset) * 6 >= len(dataset), f"Reference dataset (len {len(expected_dataset)} * {6}) must be larger than current dataset ({len(dataset)}) for index checking."
                not_found_indices = []
                for i in range(len(expected_dataset)):
                    expected_triplet = get_triplet(expected_dataset[i])
                    
                    found = False
                    for j in range(len(dataset)):
                        current_triplet = get_triplet(dataset[j])

                        if expected_triplet == current_triplet:
                            found = True
                            break
                    if not found:
                        not_found_indices.append(i)



                    # if expected_triplet == current_triplet:
                    #     continue
                    # dataset_index += 1

                
                status = "OK" if len(not_found_indices) == 0 else "MISMATCH"
                print(f"  File: {file.name}: found {len(expected_dataset) - len(not_found_indices)}; not found {len(not_found_indices)}; examples; {status}")
                # status = "OK" if dataset_index == len(dataset) // 6 else "MISMATCH"
                # print(f"  File: {file.name}: found {dataset_index} / {len(dataset) // 6} examples; {status}")