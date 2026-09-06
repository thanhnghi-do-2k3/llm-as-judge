import argparse
import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings
from scipy.stats import pearsonr, spearmanr, kendalltau, ConstantInputWarning

# --- CORRELATION UTILS ---
_CORR = {
    "pearson": lambda x, y: pearsonr(x, y, alternative='greater'),
    "spearman": lambda x, y: spearmanr(x, y, alternative='greater'),
    "kendall": lambda x, y: kendalltau(x, y, alternative='greater'),
}

def safe_corr(name: str, xs: List[float], ys: List[float], with_pvalue: bool = False) -> Union[float | Dict[str, float]]:
    try:
        if len(xs) < 2 or len(ys) < 2:
            return {"correlation": float("nan"), "pvalue": float("nan")} if with_pvalue else float("nan")
        res = _CORR[name](np.array(xs), np.array(ys))
        if with_pvalue:
            return {"correlation": res[0], "pvalue": res[1]}
        else:
            return res[0]
    except (ConstantInputWarning, ValueError):
        return {"correlation": float("nan"), "pvalue": float("nan")} if with_pvalue else float("nan")

warnings.filterwarnings("ignore", category=ConstantInputWarning)

def load_json(path: Path) -> Dict[str, Dict[str, float]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _normalize_keys(data: Dict) -> Dict:
    """Strip ,device=<value> from metric keys so device=None and device=cuda match."""
    normalized = {}
    for k, v in data.items():
        nk = re.sub(r',device=[^,}]+', '', k)
        # Last writer wins if there's a collision; prefer cuda over None
        if nk not in normalized or 'device=None' in k:
            normalized[nk] = v
    return normalized

def compute_pair_meta_correlation(data_a: Dict, data_b: Dict) -> Dict[str, Any]:
    common = sorted(list(set(data_a.keys()).intersection(data_b.keys())))
    expected_metrics = 7 * 4
    if len(common) != expected_metrics:
        warnings.warn(
            "Expected 28 common metrics for the full paper analysis, "
            f"got {len(common)}. Continuing with the available overlap. "
            f"len(a)={len(data_a)}, len(b)={len(data_b)}",
            RuntimeWarning,
        )

    vecs = {c: ([], []) for c in ['pearson', 'spearman', 'kendall']}
    for m in common:
        for c in vecs:
            if c in data_a[m] and c in data_b[m]:
                if np.isnan(data_a[m][c]) or np.isnan(data_b[m][c]):
                    continue
                vecs[c][0].append(data_a[m][c])
                vecs[c][1].append(-data_b[m][c])

    res = {"common_metrics": len(common)}
    for c in vecs:
        corr_res = safe_corr(c, vecs[c][0], vecs[c][1], with_pvalue=True)

        res[f"meta_{c}"] = corr_res["correlation"]
        res[f"meta_{c}_pvalue"] = corr_res["pvalue"]
    return res

# --- INTELLIGENT MAPPING LOGIC ---
def resolve_target_file(source_file: Path, target_base_dir: Path) -> Optional[Path]:
    name = source_file.name

    # --- RULE 1: Exact Name Match (Recursive) ---
    found_exact = list(target_base_dir.rglob(name))
    if found_exact: return found_exact[0]

    # --- RULE 2: QA Mapping ---
    qa_match = re.search(r"qa_([a-z]{2})_([a-z]+)_", name)
    if qa_match:
        src, tgt = qa_match.groups()
        code_map = {"uk": "UA", "cz": "CZ", "sk": "SK"}
        iso_src = code_map.get(src, src.upper()) 
        tgt_pattern = tgt 
        search_glob = f"*text-{iso_src}*{tgt_pattern}*binary_hf_corrs.json"
        candidates = list(target_base_dir.rglob(search_glob))
        if candidates: return candidates[0]

    # --- RULE 3: ROSE / MOCHA ---
    if "rose_" in name or "mocha_" in name:
        if "rose_cnndm" in name:
            part = "rose_cnndm"
            if "test" in name: part += "_test"
            elif "validation" in name: part += "_validation"
            elif "protocol" in name: part += "_protocol"
            candidates = list(target_base_dir.rglob(f"{part}*hf_corrs.json"))
            if candidates: return candidates[0]
        elif "rose_xsum" in name:
            candidates = list(target_base_dir.rglob(f"rose_xsum*hf_corrs.json"))
            if candidates: return candidates[0]
        elif "mocha" in name:
            if "train" in name: candidates = list(target_base_dir.rglob("mocha_train*hf_corrs.json"))
            elif "validation" in name: candidates = list(target_base_dir.rglob("mocha_validation*hf_corrs.json"))
            else: candidates = []
            if candidates: return candidates[0]

    # --- RULE 4: WMT (Year + Lang) ---
    if "wmt_" in name:
        wmt_match = re.search(r"wmt_(\d{2,4})_([a-z]{2})[_-]([a-z]{2})", name)
        if wmt_match:
            year, src, tgt = wmt_match.groups()
            full_year = f"20{year}" if len(year) == 2 else year
            candidates = list(target_base_dir.rglob(f"wmt*{full_year}*{src}*{tgt}*hf_corrs.json"))
            if candidates: return candidates[0]

    # --- RULE 5: Fallback ---
    if name.startswith("dataset_"):
        stripped_name = name.replace("dataset_", "")
        stem_part = stripped_name.split('_metric_corrs')[0].split('_hf_corrs')[0]
        candidates = list(target_base_dir.rglob(f"{stem_part}*hf_corrs.json"))
        if candidates: return candidates[0]

    print(f"[WARN] Could not resolve target file for: {source_file.name}")
    return None

def _extract_model_prefix(name: str) -> str:
    """Extract the model name prefix from a filename by splitting at the first dataset keyword."""
    if name.startswith("dataset_"):
        # Format: dataset_{dataset_identifier}_{model_name}_{shot/nofmt/...}
        rest = name[len("dataset_"):]
        ds_m = re.match(r'(?:wmt_\d+_[a-z]+_[a-z]+|mocha_\w+|rose_\w+|qa_\w+)_', rest)
        model_part = rest[ds_m.end():] if ds_m else rest
        stop_m = re.search(r'_(nofmt|zero_shot|few_shot|final_model)', model_part)
        return model_part[:stop_m.start()] if stop_m else model_part.split('_hf_corrs')[0]
    m = re.search(r'_(mocha|rose_cnndm|rose_xsum|rose_|qa_|wmt_)', name)
    return name[:m.start()] if m else name.split('_hf_corrs')[0]


def _classify_file(name: str, subdir: str = '') -> dict:
    """Extract model group, model name, dataset, shot type and variant from a filename."""
    shot = 'few' if 'few_shot' in name else ('zero' if 'zero_shot' in name else '?')

    raw = _extract_model_prefix(name)

    if name.startswith('gemma3-1b-base'):
        model_group, model_name = 'Base', raw
    elif name.startswith('gemma-3-1b-it_'):
        model_group, model_name = 'Finetuned', raw
    elif 'Llama' in name:
        if 'models--unsloth' in name:
            model_group, model_name = 'Base', raw
        else:
            model_group, model_name = 'Finetuned', raw
    else:
        model_group, model_name = 'Unknown', raw

    m = re.search(r'qa_([a-z]{2})_(en|orig)', name)
    if m:
        base_ds = f"qa_{m.group(1)}_{m.group(2)}"
        if subdir == 'cus_qa':
            dataset = f"{base_ds} [binary]"
        elif subdir == 'cus_qa_nonbin':
            dataset = f"{base_ds} [nonbin]"
        else:
            dataset = base_ds
    elif 'mocha_validation' in name:
        dataset = 'mocha_valid'
    elif 'mocha_train' in name:
        dataset = 'mocha_train'
    elif 'rose_cnndm_validation' in name:
        dataset = 'rose_cnndm_valid'
    elif 'rose_cnndm_test' in name:
        dataset = 'rose_cnndm_test'
    elif 'rose_xsum' in name:
        dataset = 'rose_xsum'
    else:
        m2 = re.search(r'wmt_(\d{2,4})_([a-z]{2})[_-]([a-z]{2})', name)
        dataset = f"wmt{m2.group(1)}_{m2.group(2)}-{m2.group(3)}" if m2 else 'unknown'

    ckpt = re.search(r'checkpoint-(\d+)', name)
    if ckpt:
        variant = f"ckpt-{ckpt.group(1)}"
    elif 'final_model' in name:
        variant = 'final'
    elif 'nofmt' in name:
        variant = 'nofmt'
    else:
        variant = ''

    return {'model_group': model_group, 'model_name': model_name,
            'dataset': dataset, 'shot': shot, 'variant': variant}


def _run_matches(name: str, run_filter: list[str] | None) -> bool:
    """Return True if the filename matches any of the requested run variants."""
    if not run_filter:
        return True
    prefix = _extract_model_prefix(name)
    return any(
        (not re.search(r"_s\d+", prefix)) if f == "nos" else (f in prefix)
        for f in run_filter
    )


def _name_filter_matches(name: str, include: list[str] | None, exclude: list[str] | None) -> bool:
    """Return True if the model prefix passes include/exclude substring filters."""
    prefix = _extract_model_prefix(name)
    if include and not all(s in prefix for s in include):
        return False
    if exclude and any(s in prefix for s in exclude):
        return False
    return True


def run_analysis(path1: Path, path2: Path, output_path: Path = None, markdown: bool = False,
                 run_filter: list[str] = None, include: list[str] = None, exclude: list[str] = None):
    print(f"--- Meta-Correlation Analysis ---")
    print(f"Input: {path1}\nTarget: {path2}\n")
    
    files_a = sorted(list(path1.rglob('*.json')))
    filtered_files: List[Path] = []
    
    # --- FILTERING LOGIC ---
    for f in files_a:
        name = f.name
        
        # 1. Skip non-hf_corrs
        if "_hf_corrs" not in name: continue

        # 2. Skip files that look like Ground Truths
        is_ground_truth_pattern = name.startswith(("qa_text-", "rose_", "wmt_", "mocha_"))

        if is_ground_truth_pattern: continue
        if "mean_corrs" in name: continue
        if not _run_matches(name, run_filter):
            continue
        if not _name_filter_matches(name, include, exclude):
            continue

        filtered_files.append(f)

    # Collect all results before printing
    rows = []
    for file_a in filtered_files:
        file_b = resolve_target_file(file_a, path2)
        subdir = file_a.parent.name
        info = _classify_file(file_a.name, subdir)
        rel = file_a.relative_to(path1)

        if not file_b:
            rows.append({**info, 'file_a': file_a, 'file_b': None, 'rel': rel, 'error': 'No match', 'res': None})
            continue

        try:
            data_a = _normalize_keys(load_json(file_a))
            data_b = _normalize_keys(load_json(file_b))
            res = compute_pair_meta_correlation(data_a, data_b)
            rows.append({**info, 'file_a': file_a, 'file_b': file_b, 'rel': rel, 'error': None, 'res': res})
        except Exception as e:
            rows.append({**info, 'file_a': file_a, 'file_b': file_b, 'rel': rel, 'error': str(e), 'res': None})

        if output_path and rows[-1].get('res'):
            out_f = output_path / rel
            out_f.parent.mkdir(parents=True, exist_ok=True)
            rows[-1]['res']['_target_file'] = str(file_b)
            with open(out_f, 'w') as f: json.dump(rows[-1]['res'], f, indent=2)

    # Sort by dataset → model_group (Base first) → model_name → shot → variant
    _group_order = {'Base': 0, 'Finetuned': 1, 'Unknown': 2}
    rows.sort(key=lambda x: (
        x['dataset'], _group_order.get(x['model_group'], 99),
        x['model_name'], x['shot'], x['variant'],
    ))

    # --- Shared formatters ---
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
        return f"{v:.4f}"

    def fmt_cell(val, pval):
        if val is None or (isinstance(val, float) and np.isnan(val)): return "N/A"
        return f"{fmt(val)} ({fmt(pval)})"

    if markdown:
        # --- Markdown output ---
        print("# Meta-Correlation Analysis\n")
        current_dataset = None
        for row in rows:
            if row['dataset'] != current_dataset:
                if current_dataset is not None:
                    print()
                current_dataset = row['dataset']
                tgt_name = row['file_b'].name if row['file_b'] else 'N/A'
                print(f"## {current_dataset}")
                print(f"*Target: `{tgt_name}`*\n")
                print("| Group | Model | Shot | Variant | Meta-Spearman | Meta-Kendall |")
                print("|-------|-------|------|---------|---------------|--------------|")

            if row['error'] and not row['res']:
                print(f"| {row['model_group']} | {row['model_name']} | {row['shot']} | {row['variant']} | *Error* | *{row['error'][:40]}* |")
            else:
                res = row['res']
                cell_s = fmt_cell(res.get('meta_spearman'), res.get('meta_spearman_pvalue'))
                cell_k = fmt_cell(res.get('meta_kendall'),  res.get('meta_kendall_pvalue'))
                print(f"| {row['model_group']} | {row['model_name']} | {row['shot']} | {row['variant']} | {cell_s} | {cell_k} |")
        print()

    else:
        # --- Terminal pretty-print: outer = dataset, inner = model_group / model_name ---
        C_MD, C_SH, C_VR, C_VA = 50, 5, 12, 18
        row_w = 2 + C_MD + 1 + C_SH + 1 + C_VR + 1 + C_VA + 1 + C_VA
        thin, thick = "─" * row_w, "━" * row_w

        def print_col_header():
            print(f"  {'Model':<{C_MD}} {'Shot':<{C_SH}} {'Variant':<{C_VR}} {'Meta-Spearman':<{C_VA}} Meta-Kendall")
            print(f"  {thin}")

        current_dataset = current_group = None

        for row in rows:
            if row['dataset'] != current_dataset:
                if current_dataset is not None:
                    print()
                current_dataset = row['dataset']
                current_group = None
                tgt_name = row['file_b'].name if row['file_b'] else '[no target]'
                print(thick)
                print(f"  DATASET: {current_dataset}  →  {tgt_name}")
                print(thick)
                print_col_header()

            if row['model_group'] != current_group:
                current_group = row['model_group']
                print(f"\n  [{current_group}]")

            if row['error'] and not row['res']:
                print(f"  {row['model_name']:<{C_MD}} {row['shot']:<{C_SH}} {row['variant']:<{C_VR}} Error: {row['error'][:50]}")
            else:
                res = row['res']
                cell_s = fmt_cell(res.get('meta_spearman'), res.get('meta_spearman_pvalue'))
                cell_k = fmt_cell(res.get('meta_kendall'),  res.get('meta_kendall_pvalue'))
                print(f"  {row['model_name']:<{C_MD}} {row['shot']:<{C_SH}} {row['variant']:<{C_VR}} {cell_s:<{C_VA}} {cell_k}")

        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path1", type=Path, help="Input folder (Experiments)")
    parser.add_argument("path2", type=Path, help="Target folder (Ground Truths)")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--markdown", "-md", action="store_true", help="Output in Markdown format")
    parser.add_argument("--run", type=str, nargs="+", default=None,
                        help="Filter by run variant substrings (e.g. --run s1000 s2000 nos). "
                             "Use 'nos' to select models without any _s<N> parameter suffix. "
                             "Multiple values are OR-combined.")
    parser.add_argument("--include", "-inc", type=str, nargs="+", default=None,
                        help="Only show models whose name contains ALL of these substrings "
                             "(e.g. --include lp0.5 kl0.05).")
    parser.add_argument("--exclude", "-exc", type=str, nargs="+", default=None,
                        help="Skip models whose name contains ANY of these substrings "
                             "(e.g. --exclude split parall).")
    args = parser.parse_args()

    if args.path1.exists() and args.path2.exists():
        run_analysis(args.path1, args.path2, args.output, markdown=args.markdown,
                     run_filter=args.run, include=args.include, exclude=args.exclude)
    else:
        print("Error: One or both paths do not exist.")
