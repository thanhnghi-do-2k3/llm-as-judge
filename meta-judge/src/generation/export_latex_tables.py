import json
import argparse
from pathlib import Path
from collections import defaultdict

# ================= CONFIGURATION =================
RESULTS_DIR = Path("./corr_results")
CORRELATIONS_DIR = Path("../../correlations")

# ================= MAPPING LOGIC =================

def get_mapping_info(src_path: Path):
    """
    Parses source filename and returns:
    1. target_path: Path to the ground truth file
    2. model_name: A clean string representing the model/shot (for the column header)
    """
    filename = src_path.name

    # STRICT FILTER: Only process hf_corrs files
    if not filename.endswith("hf_corrs.json"):
        return None, None

    # Helper to strip suffix
    base_name = filename.replace("_hf_corrs.json", "")
    
    target_path = None
    dataset_prefix_len = 0 # To help extract model name

    # 1. HANDLE CUS_QA 
    if filename.startswith("dataset_qa_"):
        parts = filename.split('_')
        # Structure: dataset_qa_{LANG}_{TYPE}_{MODEL_PARTS...}
        lang_code = parts[2]
        qa_type = parts[3]

        lang_map = {"cz": "CZ", "sk": "SK", "uk": "UA"}
        tgt_lang = lang_map.get(lang_code, lang_code.upper())

        target_name = f"qa_text-{tgt_lang}_dev_{qa_type}_binary_hf_corrs.json"
        target_path = CORRELATIONS_DIR / target_name
        
        # identifying prefix length: dataset + qa + lang + type = 4 parts
        dataset_prefix = "_".join(parts[:4]) + "_"

    # 2. HANDLE MOCHA
    elif filename.startswith("dataset_mocha_"):
        if "_validation_" in filename:
            target_path = CORRELATIONS_DIR / "mocha_validation_hf_corrs.json"
            dataset_prefix = "dataset_mocha_validation_"
        else:
            return None, None

    # 3. HANDLE ROSE
    elif "rose_cnndm" in filename:
        # Handles both "dataset_rose..." and "dataset_specific_rose..."
        if "_test_" in filename:
            target_path = CORRELATIONS_DIR / "rose_cnndm_test_hf_corrs.json"
            if filename.startswith("dataset_specific_"):
                dataset_prefix = "dataset_specific_rose_cnndm_test_"
            else:
                dataset_prefix = "dataset_rose_cnndm_test_"
        else:
            return None, None

    # 4. HANDLE WMT
    elif filename.startswith("dataset_wmt_"):
        parts = filename.split('_')
        # Structure: dataset_wmt_{YY}_{L1}_{L2}_{MODEL...}
        year_short = parts[2]
        l1 = parts[3]
        l2 = parts[4]

        year_long = f"20{year_short}"
        pair = f"{l1}-{l2}"
        
        folder_name = f"wmt_{year_long}"
        target_name = f"wmt_{year_long}_{pair}_hf_corrs.json"
        target_path = CORRELATIONS_DIR / folder_name / target_name

        # dataset + wmt + yy + l1 + l2 = 5 parts
        dataset_prefix = "_".join(parts[:5]) + "_"

    else:
        return None, None

    # Extract Model Name by removing the dataset prefix and the suffix
    # e.g. "dataset_qa_cz_en_llama70b_few_shot" -> "llama70b_few_shot"
    model_name = base_name.replace(dataset_prefix, "")
    
    return target_path, model_name

# ================= COMPARISON UTILS =================

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def print_table(target_path, gt_data, model_results, metric):
    """
    target_path: Path object of GT file
    gt_data: Dict of GT data
    model_results: List of tuples (model_header_name, model_data_dict)
    metric: 'pearson', 'spearman', etc.
    """
    
    # 1. Find intersection of keys across GT and ALL models
    common_keys = set(gt_data.keys())
    for _, m_data in model_results:
        common_keys.intersection_update(set(m_data.keys()))
    
    if not common_keys:
        return # Skip empty intersections

    # Sort keys for consistent display
    sorted_keys = sorted(common_keys)

    # 2. Prepare Headers
    # Sort models alphabetically by name for consistent column ordering
    model_results.sort(key=lambda x: x[0])
    
    # Column widths
    col_width_metric = 50
    col_width_val = 12
    
    # Headers: Metric | GT | Model A | Model B ...
    headers = ["Metric", "GT (Ref)"] + [name for name, _ in model_results]
    
    # Formatting strings
    # Dynamic header format
    header_fmt = f"{{:<{col_width_metric}}} | {{:<{col_width_val}}}" + \
                 f" | {{:<{col_width_val}}}" * len(model_results)
    
    # Dynamic row format (float precision)
    row_fmt = f"{{:<{col_width_metric}}} | {{:<{col_width_val}.4f}}" + \
              f" | {{:<{col_width_val}.4f}}" * len(model_results)

    # 3. Print Output
    print(f"\n{'='*120}")
    print(f"TARGET: {target_path.name}")
    print(f"METRIC: {metric.upper()}")
    print(f"{'='*120}")
    
    # Print Headers (truncate model names if too long, strictly for display)
    display_headers = [h[:col_width_val-1] for h in headers]
    print(header_fmt.format(*display_headers))
    print("-" * 120)

    for key in sorted_keys:
        # Get GT Value
        gt_val = gt_data[key].get(metric)
        
        # Get Model Values
        row_vals = [gt_val]
        
        valid_row = True
        for _, m_data in model_results:
            m_val = m_data[key].get(metric)
            if m_val is None: 
                valid_row = False; break
            # Invert model value as requested
            row_vals.append(-m_val)
        
        if valid_row and gt_val is not None:
            # Print row
            print(row_fmt.format(key[:col_width_metric], *row_vals))


# ================= MAIN EXECUTION =================

def main():
    parser = argparse.ArgumentParser(description="Compare correlation results.")
    parser.add_argument("--metric", type=str, default="spearman", 
                        choices=["pearson", "spearman", "kendall"])
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR} not found.")
        return

    # Dictionary to group results: 
    # { target_path_object: [ (model_name_string, source_json_data), ... ] }
    groups = defaultdict(list)

    # 1. Collect and Group Data
    for src_file in RESULTS_DIR.rglob('*.json'):
        tgt_path, model_name = get_mapping_info(src_file)
        
        if tgt_path and tgt_path.exists():
            src_data = load_json(src_file)
            if src_data:
                groups[tgt_path].append((model_name, src_data))

    # 2. Process Groups
    # Sort groups by target filename for clean output order
    sorted_targets = sorted(groups.keys(), key=lambda p: p.name)

    for tgt_path in sorted_targets:
        gt_data = load_json(tgt_path)
        if not gt_data:
            continue
            
        model_results = groups[tgt_path]
        print_table(tgt_path, gt_data, model_results, args.metric)

if __name__ == "__main__":
    main()