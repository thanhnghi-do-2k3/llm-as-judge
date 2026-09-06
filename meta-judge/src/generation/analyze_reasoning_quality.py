from pathlib import Path
import argparse
from typing import List, Optional, Dict, Any

table_format = "\R{.730} & \R{.661} & \R{.492} & \R{.751} & \R{.735} & \R{.533} & \R{.249} & \R{.365} & \R{.312} & \R{.265} & \R{.825} & \R{.148} & \R{.233}"
table_format = "\R{{{num}}}"
table_divider = " & "

def process_parts(parts: List[str]) -> Dict[str, str]:
    dat_max_index = {"qa": 4, "rose": 4, "mocha": 3, "wmt": 5}
    model_len = {"llama4scout": 1, "llama70b": 1, "qwen3": 2}

    dataset_part = parts[1]
    dataset = '_'.join(parts[1:dat_max_index[dataset_part]])
    model_part = parts[dat_max_index[dataset_part]]
    model = '_'.join(parts[dat_max_index[dataset_part]:dat_max_index[dataset_part]+model_len[model_part]])
    shot = '_'.join(parts[dat_max_index[dataset_part]+model_len[model_part]:dat_max_index[dataset_part]+model_len[model_part]+2])
    
    return {
        "dataset": dataset,
        "model": model,
        "shot": shot
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resolve target file path based on source file and target base directory.")
    parser.add_argument("file", type=Path, help="Path to the source file")
    parser.add_argument("--corr_type", type=str, default="s", help="Type of correlation to process (s for spearman, p for pearson)")
    args = parser.parse_args()

    file: Path = Path(args.file)
    lines = file.read_text().splitlines()
    lines = lines[2:]

    dataset_order = [
        'qa_cz_en', 'qa_sk_en', 'qa_uk_en', 
        'qa_cz_orig', 'qa_sk_orig', 'qa_uk_orig',
        'mocha_validation',
        'rose_cnndm_test',
        'wmt_21_en_ha', 'wmt_21_xh_zu',
        'wmt_24_cs_uk', 'wmt_24_en_cs', 'wmt_24_en_is' 
        ]
    model_order = ["llama4scout", "llama70b", "qwen3_30b"]
    shot_order = ["few_shot", "zero_shot"]

    results = {m: {s: {d: None for d in dataset_order} for s in shot_order} for m in model_order}
    corr_index = 2 if args.corr_type == 's' else 3

    for line in lines:
        parts = line.split("|")
        input_file = parts[0].strip()
        result = parts[corr_index].strip()
        file_info = process_parts(input_file.split("_"))
        
        results[file_info["model"]][file_info["shot"]][file_info["dataset"]] = result

        # print(file_info, result)
    print(results)
    for m in model_order:
        for s in shot_order:
            row = [table_format.format(num=f"{float(results[m][s][d]):.3f}".replace("0.", ".")) for d in dataset_order]
            print(f"{m} {s}: {table_divider.join(row)}")
        

