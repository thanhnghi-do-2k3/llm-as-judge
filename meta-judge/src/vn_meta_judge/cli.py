from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import DEFAULT_GEMINI_MODEL, metric_summary
from .data.loaders import BenchmarkRow, export_benchmark_artifacts, load_jsonl, save_jsonl
from .evaluation.correlations import compute_synthetic_correlations, write_correlation_report
from .generation.deepseek import DEFAULT_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL, generate_damage as generate_deepseek_damage
from .generation.gemini import generate_damage, load_local_env
from .generation.audit import audit_damage_file
from .generation.rule_based import generate_rule_baseline
from .evaluation.score_export import export_machine_scores, export_translations_csv
from .metrics.runner import run_metric_scores


def _rows_from_jsonl(path: Path):
    return [BenchmarkRow(**row) for row in load_jsonl(path)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vietnamese Meta-Judge experiment pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export-benchmark", help="Convert the group's Google Sheet XLSX export to canonical JSONL")
    export.add_argument("--scores-xlsx", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--expected-rows", type=int, default=300)

    machine_scores = sub.add_parser("export-machine-scores", help="Export human scores as model-readable JSON/JSONL")
    machine_scores.add_argument("--scores-xlsx", type=Path, required=True)
    machine_scores.add_argument("--output-dir", type=Path, required=True)
    machine_scores.add_argument("--expected-rows", type=int, default=300)

    translations = sub.add_parser("export-translations", help="Export reusable source/reference/A/B translations as CSV")
    translations.add_argument("--scores-xlsx", type=Path, required=True)
    translations.add_argument("--output", type=Path, required=True)
    translations.add_argument("--expected-rows", type=int, default=300)

    rule = sub.add_parser("rule-baseline", help="Create deterministic B1 damage data without API calls")
    rule.add_argument("--input", type=Path, required=True, help="references_300.jsonl")
    rule.add_argument("--output", type=Path, required=True)
    rule.add_argument("--seed", type=int, default=42)

    gemini = sub.add_parser("gemini", help="Generate one zero/few-shot Gemini branch, resumably")
    gemini.add_argument("--input", type=Path, required=True, help="references_300.jsonl")
    gemini.add_argument("--output", type=Path, required=True)
    gemini.add_argument("--model", default=None, help=f"Default: GEMINI_MODEL from .env, or {DEFAULT_GEMINI_MODEL}")
    gemini.add_argument("--prompt-type", choices=("zero_shot", "few_shot"), required=True)
    gemini.add_argument("--limit", type=int, default=None, help="Use 20/50 for pilot; omit for the full 300 rows")
    gemini.add_argument("--max-output-tokens", type=int, default=400)
    gemini.add_argument("--workers", type=int, default=1, help="Bounded concurrent API requests; lower if rate-limited")

    deepseek = sub.add_parser(
        "deepseek",
        help="Generate an isolated optional DeepSeek ablation branch; never mixed with the Gemini primary run",
    )
    deepseek.add_argument("--input", type=Path, required=True, help="references_300.jsonl")
    deepseek.add_argument("--output", type=Path, required=True)
    deepseek.add_argument("--model", default=None, help=f"Default: DEEPSEEK_MODEL from .env, or {DEFAULT_DEEPSEEK_MODEL}")
    deepseek.add_argument("--base-url", default=None, help=f"Default: DEEPSEEK_BASE_URL from .env, or {DEFAULT_DEEPSEEK_BASE_URL}")
    deepseek.add_argument("--prompt-type", choices=("zero_shot", "few_shot"), required=True)
    deepseek.add_argument("--limit", type=int, default=None, help="Use 20/50 for ablation pilot; omit for the full 300 rows")
    deepseek.add_argument("--max-output-tokens", type=int, default=400)

    audit = sub.add_parser("audit-damage", help="Check generated output schema and optionally score manual level audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--manual-audit", type=Path, default=None, help="Optional JSONL with requested_level and observed_level")

    metrics = sub.add_parser("metrics", help="Run author's metric implementations through the Vietnamese adapter")
    metrics.add_argument("--input", type=Path, required=True)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.add_argument("--tokenization", choices=("syllable", "underthesea"), default="syllable")
    metrics.add_argument("--without-bleurt", action="store_true")
    metrics.add_argument("--families", nargs="*", default=None, help="Optional: BLEU chrF ROUGE METEOR BERTScore COMET BLEURT")
    metrics.add_argument("--batch-size", type=int, default=32)

    corr = sub.add_parser("correlate", help="Compute r_hum, r_syn, and MC using Eq. 1-3")
    corr.add_argument("--human-jsonl", type=Path, required=True)
    corr.add_argument("--human-scores", type=Path, required=True)
    corr.add_argument("--synthetic-jsonl", type=Path, required=True)
    corr.add_argument("--synthetic-scores", type=Path, required=True)
    corr.add_argument("--output", type=Path, required=True)

    syn_corr = sub.add_parser("synthetic-correlate", help="Compute r_syn only for a synthetic branch before human scores are complete")
    syn_corr.add_argument("--synthetic-jsonl", type=Path, required=True)
    syn_corr.add_argument("--synthetic-scores", type=Path, required=True)
    syn_corr.add_argument("--output", type=Path, required=True)

    manifest = sub.add_parser("metric-manifest", help="Print the expected paper-aligned metric design")
    manifest.add_argument("--without-bleurt", action="store_true")
    return parser


def main() -> None:
    load_local_env()
    args = build_parser().parse_args()
    if args.command == "export-benchmark":
        summary = export_benchmark_artifacts(args.scores_xlsx, args.output_dir, expected_rows=args.expected_rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "export-machine-scores":
        result = export_machine_scores(args.scores_xlsx, args.output_dir, expected_rows=args.expected_rows)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "export-translations":
        result = export_translations_csv(args.scores_xlsx, args.output, expected_rows=args.expected_rows)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "rule-baseline":
        rows = _rows_from_jsonl(args.input)
        outputs = generate_rule_baseline(rows, seed=args.seed)
        save_jsonl(outputs, args.output)
        print(json.dumps({"output": str(args.output), "rows": len(outputs), "levels": 6}, ensure_ascii=False, indent=2))
    elif args.command == "gemini":
        model = args.model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        max_tokens = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", args.max_output_tokens))
        workers = int(os.environ.get("GEMINI_WORKERS", args.workers))
        print(json.dumps(generate_damage(args.input, args.output, model, args.prompt_type, args.limit, max_tokens, workers=workers), ensure_ascii=False, indent=2))
    elif args.command == "deepseek":
        model = args.model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
        base_url = args.base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL
        max_tokens = int(os.environ.get("DEEPSEEK_MAX_OUTPUT_TOKENS", args.max_output_tokens))
        print(json.dumps(
            generate_deepseek_damage(
                args.input,
                args.output,
                model,
                args.prompt_type,
                args.limit,
                max_tokens,
                base_url=base_url,
            ),
            ensure_ascii=False,
            indent=2,
        ))
    elif args.command == "audit-damage":
        result = audit_damage_file(args.input, manual_audit_path=args.manual_audit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "metrics":
        print(json.dumps(run_metric_scores(args.input, args.output, args.tokenization, not args.without_bleurt, args.families, args.batch_size), ensure_ascii=False, indent=2))
    elif args.command == "correlate":
        report = write_correlation_report(args.human_jsonl, args.human_scores, args.synthetic_jsonl, args.synthetic_scores, args.output)
        print(json.dumps(report["meta"], ensure_ascii=False, indent=2))
    elif args.command == "synthetic-correlate":
        synthetic = compute_synthetic_correlations(args.synthetic_jsonl, args.synthetic_scores)
        report = {
            "synthetic": synthetic,
            "equation": "r_syn = rho(metric(D_syn), -damage_level)",
            "note": "This is not MC; r_hum is required before meta-correlation.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "metrics": len(synthetic)}, ensure_ascii=False, indent=2))
    elif args.command == "metric-manifest":
        print(json.dumps(metric_summary(include_bleurt=not args.without_bleurt), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
