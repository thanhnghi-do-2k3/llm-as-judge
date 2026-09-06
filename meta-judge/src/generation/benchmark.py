from transformers import pipeline
from pathlib import Path
import argparse
import torch, time, itertools, datetime
from typing import Any, List, Iterable, Dict, TypeVar

# --- IMPORTS FROM YOUR PROJECT ---
from human_feedback_datasets.processed_dataset import ProcessedDataset
from human_feedback_datasets.prompts import (
    SUMMARIZATION_PROMPT_V2, CUS_QA_PROMPT, CUS_QA_PROMPT_GEMINI_1, 
    CUS_QA_PROMPT_GEMINI_2, CUS_QA_PROMPT_GEMINI_3, MOCHA_PROMPT, 
    MOCHA_PROMPT_2, ROSE_PROMPT_LONG, ROSE_PROMPT_OPTIMIZED
)
from generate import get_unique_data, gen_data, batch, MODEL_CONFIGS, DATASET_CONFIGS, DEFAULT_GENERATION_PARAMS


# ==========================================
# MAIN ESTIMATION SCRIPT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate generation time for ALL datasets.")
    
    parser.add_argument("--model_config", type=str, required=True, help=f"Key from MODEL_CONFIGS: {list(MODEL_CONFIGS.keys())}")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used for estimation.")
    parser.add_argument("--estimation_samples", type=int, default=64, help="Samples per dataset to run for speed calculation.")
    
    parser.add_argument("--damage_level_min", type=int, default=0)
    parser.add_argument("--damage_level_max", type=int, default=5)
    parser.add_argument("--override_max_tokens", type=int, default=None, help="Override max_new_tokens for this estimation.")

    args = parser.parse_args()

    # 1. SETUP MODEL
    if args.model_config not in MODEL_CONFIGS:
        raise ValueError(f"Model config '{args.model_config}' not found.")

    model_conf = MODEL_CONFIGS[args.model_config]
    gen_params = model_conf["params"].copy()
    if args.override_max_tokens:
        gen_params["max_new_tokens"] = args.override_max_tokens

    print(f"\n{'='*60}")
    print(f"LOADING MODEL: {model_conf['model_id']}")
    print(f"{'='*60}")
    
    text_generator = pipeline(
        "text-generation", 
        model=model_conf['model_id'], 
        tokenizer=model_conf['model_id'], 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    if text_generator.tokenizer.pad_token_id is not None:
        gen_params["pad_token_id"] = text_generator.tokenizer.pad_token_id
    else:
        gen_params["pad_token_id"] = text_generator.tokenizer.eos_token_id
    
    print("Model loaded. Starting dataset loop...\n")

    # Storage for summary
    summary_data = []
    damage_levels = range(args.damage_level_min, args.damage_level_max+1)

    # 2. DATASET LOOP
    for ds_key, dataset_conf in DATASET_CONFIGS.items():
        print(f"--- Benchmarking Dataset: {ds_key.upper()} ---")
        
        if not dataset_conf['path'].exists():
            print(f"Skipping {ds_key}: File not found at {dataset_conf['path']}")
            continue

        try:
            # A. Load Data
            dataset = ProcessedDataset.load(dataset_conf['path'])
            dataset = get_unique_data(dataset)
            
            # B. Calculate Job Size
            input_count = len(dataset)
            prompt_list = dataset_conf["prompts"]
            total_gens = input_count * len(damage_levels) * len(prompt_list)
            
            # C. Run Estimation
            # We use the first prompt to test speed
            test_prompt = prompt_list[0]
            data_iterator = gen_data(dataset_conf, dataset, damage_levels, test_prompt)
            est_iterator = itertools.islice(data_iterator, args.estimation_samples)
            
            start_time = time.time()
            processed_count = 0
            
            for batch_items in batch(est_iterator, args.batch_size):
                batch_msgs = [b['messages'] for b in batch_items]
                _ = text_generator(batch_msgs, **gen_params, num_return_sequences=1)
                processed_count += len(batch_msgs)
                print(f"  > Processed {processed_count}/{args.estimation_samples} samples...", end='\r')
            
            end_time = time.time()
            print() # Newline after counter
            
            if processed_count == 0:
                print("  ! Warning: No samples processed (dataset empty?).")
                continue

            # D. Calculations
            duration = end_time - start_time
            avg_time = duration / processed_count
            est_seconds = avg_time * total_gens
            est_str = str(datetime.timedelta(seconds=int(est_seconds)))
            
            print(f"  > Speed: {avg_time:.3f}s/sample")
            print(f"  > Est. Time: {est_str}")
            
            summary_data.append({
                "dataset": ds_key,
                "inputs": input_count,
                "prompts": len(prompt_list),
                "total_gens": total_gens,
                "avg_speed": avg_time,
                "est_time": est_str,
                "est_seconds": est_seconds
            })

        except Exception as e:
            print(f"  ! Error benchmarking {ds_key}: {e}")

    # 3. FINAL SUMMARY
    print(f"\n\n{'='*75}")
    print(f"{'DATASET':<10} | {'INPUTS':<8} | {'PROMPTS':<7} | {'TOTAL GENS':<10} | {'SPEED (s)':<9} | {'EST TIME':<10}")
    print(f"{'-'*75}")
    
    total_pipeline_seconds = 0
    
    for row in summary_data:
        total_pipeline_seconds += row["est_seconds"]
        print(f"{row['dataset'].upper():<10} | {row['inputs']:<8} | {row['prompts']:<7} | {row['total_gens']:<10} | {row['avg_speed']:<9.3f} | {row['est_time']:<10}")
    
    total_str = str(datetime.timedelta(seconds=int(total_pipeline_seconds)))
    
    print(f"{'-'*75}")
    print(f"TOTAL PIPELINE ESTIMATE: {total_str}")
    print(f"{'='*75}")