"""Portable experiment orchestration for the translation evaluation notebook.

Notebook cells express the research stages. This module owns validation,
checkpoints, bounded scheduling and artifact export.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from queue import Queue
from typing import Callable
from zipfile import ZipFile

from .config import paper_metric_specs
from .data.loaders import BenchmarkRow, load_benchmark_xlsx, load_jsonl, save_jsonl
from .generation.api_runner import generate_damage_with_backend
from .generation.audit import audit_damage_file
from .generation.gemini import call_gemini
from .generation.rule_based import generate_rule_baseline


LIGHT = ["BLEU", "chrF", "ROUGE", "METEOR"]
HEAVY = ["BERTScore", "COMET", "BLEURT"]


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pending.replace(path)


def key_list(values):
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(v, str) for v in values
    ):
        raise ValueError("GEMINI_API_KEYS phải là mảng các chuỗi.")
    return list(dict.fromkeys(v.strip() for v in values if v.strip()))


class _GeminiQuotaGroup:
    """One quota bucket shared by one or more keys from the same profile."""

    def __init__(self, keys, min_interval_seconds, caller, sleep_fn, monotonic_fn):
        self.keys = key_list(keys)
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.caller = caller
        self.sleep = sleep_fn
        self.monotonic = monotonic_fn
        self._key_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._key_index = 0
        self._next_request_at = 0.0

    def call(self, messages, model, tokens):
        with self._rate_lock:
            wait_seconds = self._next_request_at - self.monotonic()
            if wait_seconds > 0:
                self.sleep(wait_seconds)
            self._next_request_at = self.monotonic() + self.min_interval_seconds
        with self._key_lock:
            key = self.keys[self._key_index % len(self.keys)]
            self._key_index += 1
        caller = self.caller or call_gemini
        return caller(messages, model, key, tokens)


class GeminiRequestPool:
    """Run standard and high-quota Gemini keys through one shared scheduler.

    Standard keys retain the legacy shared concurrency and pacing rule. Each
    high-quota key receives an independent quota group with configurable lanes.
    Keys from the same Google project must not be treated as independent quota.
    """

    def __init__(
        self,
        standard_keys=(),
        high_quota_keys=(),
        *,
        standard_workers=1,
        standard_min_interval_seconds=4.2,
        high_quota_workers_per_key=4,
        high_quota_min_interval_seconds=0.25,
        caller=None,
        sleep_fn=time.sleep,
        monotonic_fn=time.monotonic,
    ):
        self.standard_keys = key_list(standard_keys)
        self.high_quota_keys = key_list(high_quota_keys)
        overlap = sorted(set(self.standard_keys) & set(self.high_quota_keys))
        if overlap:
            raise ValueError(
                "Một Gemini key không được nằm đồng thời trong STANDARD và "
                "HIGH_QUOTA."
            )
        self.keys = self.standard_keys + self.high_quota_keys
        self._lanes = Queue()

        if self.standard_keys:
            standard_group = _GeminiQuotaGroup(
                self.standard_keys,
                standard_min_interval_seconds,
                caller,
                sleep_fn,
                monotonic_fn,
            )
            for _ in range(max(1, int(standard_workers))):
                self._lanes.put(standard_group)

        for key in self.high_quota_keys:
            high_quota_group = _GeminiQuotaGroup(
                [key],
                high_quota_min_interval_seconds,
                caller,
                sleep_fn,
                monotonic_fn,
            )
            for _ in range(max(1, int(high_quota_workers_per_key))):
                self._lanes.put(high_quota_group)

        self.worker_count = self._lanes.qsize()

    def __call__(self, messages, model, tokens):
        if not self.keys:
            raise RuntimeError("Chưa cấu hình Gemini API key.")
        quota_group = self._lanes.get()
        try:
            return quota_group.call(messages, model, tokens)
        except Exception as exc:
            message = str(exc)
            for secret in self.keys:
                message = message.replace(secret, "[redacted]")
            raise RuntimeError(message) from None
        finally:
            self._lanes.put(quota_group)


def parallel_jobs(jobs: dict[str, Callable], workers: int):
    """Join all independent jobs before allowing the next stage to proceed."""

    def invoke(name, fn):
        try:
            return fn()
        except Exception as exc:
            return {"status": "failed", "job": name, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {name: pool.submit(invoke, name, fn) for name, fn in jobs.items()}
        return {name: future.result() for name, future in futures.items()}


def read_translations(path: Path, expected_rows: int, source_file=None):
    """Validate and order the existing source/reference/A/B table by STT."""
    import pandas as pd

    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, sheet_name="Cham_diem")
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(
            "Kết quả bước sinh bản dịch phải là XLSX (sheet Cham_diem) hoặc CSV."
        )
    if source_file is not None:
        source = pd.read_csv(source_file)
        needed = [c for c in source.columns if c == "STT" or c not in frame.columns]
        frame = frame.merge(
            source[needed], on="STT", how="outer", validate="one_to_one"
        )
    for old, new in {
        "Cau_nguon_ZH": "Nguon_ZH",
        "Cau_tham_chieu_VI": "Tham_chieu_VI",
    }.items():
        if new not in frame and old in frame:
            frame = frame.rename(columns={old: new})
    text_columns = ["Nguon_ZH", "Tham_chieu_VI", "Ban_dich_He_A", "Ban_dich_He_B"]
    required = ["STT", *text_columns]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    blank = (
        frame[required]
        .fillna("")
        .astype(str)
        .apply(lambda c: c.str.strip().eq(""))
        .all(axis=1)
    )
    frame = frame.loc[~blank].copy()
    numbers = pd.to_numeric(frame.STT, errors="coerce")
    if numbers.isna().any() or not numbers.mod(1).eq(0).all():
        raise ValueError("STT phải là số nguyên và không được trống.")
    frame["STT"] = numbers.astype(int)
    if frame.STT.duplicated().any() or set(frame.STT) != set(
        range(1, expected_rows + 1)
    ):
        raise ValueError(f"STT phải đủ 1–{expected_rows}, không trùng.")
    for column in text_columns:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Cột {column} còn ô trống.")
    if "ID_cau_VLSP" not in frame:
        frame["ID_cau_VLSP"] = frame.STT.map(lambda x: f"vi-zh-2022-test-{x:04d}")
    else:
        frame["ID_cau_VLSP"] = frame.ID_cau_VLSP.fillna("").astype(str).str.strip()
    if frame.ID_cau_VLSP.eq("").any() or frame.ID_cau_VLSP.duplicated().any():
        raise ValueError("ID_cau_VLSP trống hoặc bị trùng.")
    return frame.sort_values("STT").reset_index(drop=True)


class NotebookExperiment:
    def __init__(
        self,
        root,
        code_dir,
        run_name,
        *,
        model,
        limit,
        seed=42,
        api_workers=4,
        api_min_interval_seconds=0.0,
        cpu_workers=2,
        worker_python=None,
        heavy_worker_pythonpaths=None,
        heavy=False,
        underthesea=True,
        max_tokens=400,
        api_keys=(),
        high_quota_api_keys=(),
        high_quota_workers_per_key=4,
        high_quota_min_interval_seconds=0.25,
    ):
        if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
            raise ValueError("RUN_NAME phải là một tên thư mục.")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError("LIMIT phải là số nguyên dương hoặc None.")
        self.root = Path(root).resolve()
        self.code = Path(code_dir).resolve()
        self.output = self.root / "output" / run_name
        self.cache = self.root / "cache"
        self.logs = self.output / "logs"
        self.data = self.output / "data"
        self.generation_dir = self.output / "generation"
        self.metrics_dir = self.output / "metrics"
        self.analysis_dir = self.output / "analysis"
        self.model, self.limit, self.seed = model, limit, seed
        self.standard_api_workers = max(1, int(api_workers))
        self.api_min_interval_seconds = max(0.0, float(api_min_interval_seconds))
        self.cpu_workers = max(1, int(cpu_workers))
        self.worker_python = Path(worker_python or sys.executable).resolve()
        if not self.worker_python.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy Python cho metric subprocess: {self.worker_python}"
            )
        self.heavy_worker_pythonpaths = {
            str(family): Path(directory).resolve()
            for family, directory in (heavy_worker_pythonpaths or {}).items()
        }
        missing_worker_paths = [
            str(path)
            for path in self.heavy_worker_pythonpaths.values()
            if not path.is_dir()
        ]
        if missing_worker_paths:
            raise FileNotFoundError(
                "Không tìm thấy package path cho metric nặng: "
                + ", ".join(missing_worker_paths)
            )
        self.heavy, self.underthesea = heavy, underthesea
        self.max_tokens = max_tokens
        self.request_pool = GeminiRequestPool(
            standard_keys=api_keys,
            high_quota_keys=high_quota_api_keys,
            standard_workers=self.standard_api_workers,
            standard_min_interval_seconds=self.api_min_interval_seconds,
            high_quota_workers_per_key=high_quota_workers_per_key,
            high_quota_min_interval_seconds=high_quota_min_interval_seconds,
        )
        self.keys = self.request_pool.keys
        self.api_workers = max(1, self.request_pool.worker_count)
        self.frame = None
        self.human_rows = []
        self.datasets = {}
        self.reports = []
        self.state = {}
        for directory in [
            self.cache,
            self.logs,
            self.data,
            self.generation_dir,
            self.metrics_dir,
            self.analysis_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_translations(
        self,
        input_path,
        *,
        source_file=None,
        scores_file=None,
        expected_rows=300,
        translation_generation="reused_translations",
    ):
        """Debug entry point: no sampling, translation, API calls or downloads."""
        self.frame = None
        self.human_rows = []
        self.datasets = {}
        if hasattr(self, "references"):
            del self.references
        input_path = Path(input_path).resolve()
        frame = read_translations(input_path, expected_rows, source_file=source_file)
        scoring = (
            Path(scores_file).resolve()
            if scores_file
            else (input_path if input_path.suffix.lower() == ".xlsx" else None)
        )
        human = []
        if scoring:
            scored_frame = read_translations(scoring, expected_rows)
            for column in [
                "Nguon_ZH",
                "Tham_chieu_VI",
                "Ban_dich_He_A",
                "Ban_dich_He_B",
            ]:
                if not frame[column].equals(scored_frame[column]):
                    raise ValueError(
                        f"Kết quả bước sinh bản dịch không khớp workbook chấm điểm: {column}."
                    )
            human = load_benchmark_xlsx(scoring, expected_rows)
            ids = dict(zip(frame.STT, frame.ID_cau_VLSP))
            for row in human:
                row.id = ids[row.metadata["stt"]]
        self.frame, self.human_rows = frame, human
        self.sources = [input_path] + (
            [scoring] if scoring and scoring != input_path else []
        )
        if source_file:
            self.sources.append(Path(source_file).resolve())
        self.expected_rows = expected_rows
        self.translation_generation = str(translation_generation)
        return frame

    def prepare(self):
        if self.frame is None:
            raise RuntimeError("Chạy cell đọc kết quả bước sinh bản dịch trước.")
        from .generation.prompts import render_messages

        contract = {
            "version": 1,
            "model": self.model,
            "limit": self.limit,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "source_hashes": [
                hashlib.sha256(p.read_bytes()).hexdigest() for p in self.sources
            ],
            "prompts": [
                render_messages("source", "reference", 0, p)
                for p in ["zero_shot", "few_shot"]
            ],
        }
        contract_path = self.output / "experiment.json"
        old = read_json(contract_path)
        if old and old != contract:
            raise ValueError(
                "Input/model/prompt/LIMIT đã đổi. Chọn RUN_NAME mới để giữ checkpoint nhất quán."
            )
        write_json(contract_path, contract)
        self.frame.to_csv(
            self.data / "translations.csv", index=False, encoding="utf-8-sig"
        )
        references = [
            BenchmarkRow(
                id=row.ID_cau_VLSP,
                source_zh=row.Nguon_ZH,
                reference_vi=row.Tham_chieu_VI,
                metadata={"stt": int(row.STT)},
            )
            for row in self.frame.itertuples()
        ]
        save_jsonl(references, self.data / "references_all.jsonl")
        selected = references[: self.limit] if self.limit is not None else references
        self.references = self.data / "references.jsonl"
        save_jsonl(selected, self.references)
        save_jsonl(self.human_rows, self.data / "human_all.jsonl")
        clean = [
            r
            for r in self.human_rows
            if not r.quality_flags and r.human_score is not None
        ]
        self.human_file = self.data / "human_clean.jsonl"
        save_jsonl(clean, self.human_file)
        if clean:
            self.datasets["human"] = self.human_file
        self.summary = {
            "references": len(references),
            "selected_references": len(selected),
            "human_all": len(self.human_rows),
            "human_clean": len(clean),
            "human_excluded": len(self.human_rows) - len(clean),
            "translation_generation": getattr(
                self, "translation_generation", "reused_translations"
            ),
            "human_evaluation": "available" if clean else "missing_scores",
        }
        write_json(self.data / "input_audit.json", self.summary)
        return self.summary

    def _audit_branch(self, path, name):
        audit = audit_damage_file(path)
        rows = load_jsonl(path)
        expected = {r["id"]: r for r in load_jsonl(self.references)}
        audit["matches_input"] = {r["id"] for r in rows} == set(expected) and all(
            r["reference_vi"] == expected[r["id"]]["reference_vi"]
            and r["source_zh"] == expected[r["id"]]["source_zh"]
            for r in rows
        )
        audit["status"] = (
            "ready"
            if audit["pass_basic_schema"] and audit["matches_input"]
            else "failed"
        )
        write_json(self.generation_dir / f"audit_{name}.json", audit)
        return audit

    def gemini_caller(self, messages, model, tokens):
        """Shared request entry point for translation and damage generation."""
        return self.request_pool(messages, model, tokens)

    def _gemini_caller(self, messages, model, tokens):
        return self.gemini_caller(messages, model, tokens)

    def generate(self, regenerate_damage=True, damage_files=None, include_rule_based=True):
        if not hasattr(self, "references"):
            raise RuntimeError("Chạy bước chuẩn bị dữ liệu trước.")
        reuse_files = {
            str(name): Path(path).resolve()
            for name, path in (damage_files or {}).items()
        }

        def rule():
            rows = [BenchmarkRow(**r) for r in load_jsonl(self.references)]
            path = self.generation_dir / "rule_based.jsonl"
            save_jsonl(generate_rule_baseline(rows, seed=self.seed), path)
            return self._audit_branch(path, "rule_based")

        def prompt_branch(prompt):
            generated_path = self.generation_dir / f"{prompt}.jsonl"
            path = (
                generated_path
                if regenerate_damage
                else reuse_files.get(prompt, generated_path)
            )
            if regenerate_damage and self.keys:
                generate_damage_with_backend(
                    self.references,
                    path,
                    self.model,
                    prompt,
                    self._gemini_caller,
                    max_output_tokens=self.max_tokens,
                    backend_name="gemini",
                    # Two prompt branches run together and share one request
                    # pool, so each branch opens at most half the pool lanes.
                    workers=max(1, math.ceil(self.api_workers / 2)),
                )
            elif not path.exists():
                return {"status": "skipped", "reason": "empty_key_array"}
            return self._audit_branch(path, prompt)

        jobs = {"rule_based": rule} if include_rule_based else {}
        jobs.update(
            {p: lambda p=p: prompt_branch(p) for p in ["zero_shot", "few_shot"]}
        )
        self.state["generation"] = parallel_jobs(jobs, workers=3)
        for name, result in self.state["generation"].items():
            if result.get("status") == "ready":
                if name == "rule_based" or regenerate_damage:
                    self.datasets[name] = self.generation_dir / f"{name}.jsonl"
                else:
                    self.datasets[name] = reuse_files.get(
                        name, self.generation_dir / f"{name}.jsonl"
                    )
            else:
                self.datasets.pop(name, None)
        write_json(
            self.generation_dir / "generation_summary.json", self.state["generation"]
        )
        return self.state["generation"]

    def _process(self, name, arguments, *, extra_pythonpath=None, env_overrides=None):
        log_path = self.logs / f"{name}.log"
        env = os.environ.copy()
        python_paths = [str(self.code / "src")]
        if extra_pythonpath:
            python_paths.insert(0, str(extra_pythonpath))
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env.update(env_overrides or {})
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [str(self.worker_python), *map(str, arguments)],
                cwd=self.code,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return {
            "status": "ready" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "log": str(log_path.relative_to(self.output)),
        }

    def prepare_metric_cache(self):
        result = self._process(
            "metric_cache", ["-m", "vn_meta_judge.notebook_workflow", "prefetch"]
        )
        self.state["metric_cache"] = result
        return result

    def score(self, *, heavy=False):
        """CPU datasets run concurrently; each GPU metric gets its own process."""
        jobs, artifacts = {}, []
        specs = paper_metric_specs()
        groups = (
            [("light", None, LIGHT)]
            if not heavy
            else [
                (f"heavy_{i:02d}", spec.key, [spec.family])
                for i, spec in enumerate(specs)
                if spec.family in HEAVY
            ]
        )
        for name, input_path in self.datasets.items():
            for tokenization in (
                ["syllable", "underthesea"] if self.underthesea else ["syllable"]
            ):
                if heavy and tokenization != "syllable":
                    continue
                for group, key, families in groups:
                    selected = (
                        ["BLEU", "chrF"] if tokenization == "underthesea" else families
                    )
                    output = (
                        self.metrics_dir
                        / "scores"
                        / name
                        / tokenization
                        / f"{group}.json"
                    )
                    job_name = f"metric_{name}_{tokenization}_{group}"
                    args = [
                        "-m",
                        "vn_meta_judge.notebook_workflow",
                        "metric",
                        input_path,
                        output,
                        tokenization,
                        ",".join(selected),
                        key or "",
                    ]
                    extra_pythonpath = (
                        self.heavy_worker_pythonpaths.get(families[0])
                        if heavy
                        else None
                    )
                    env_overrides = (
                        {"USE_TF": "0"} if heavy and families[0] == "COMET" else None
                    )
                    jobs[job_name] = (
                        lambda n=job_name, a=args, p=extra_pythonpath, e=env_overrides: (
                            self._process(n, a, extra_pythonpath=p, env_overrides=e)
                        )
                    )
                    artifacts.append((job_name, output))
        result = parallel_jobs(jobs, workers=1 if heavy else self.cpu_workers)
        for name, output in artifacts:
            payload = read_json(output, {})
            result[name]["completed_metrics"] = len(payload.get("scores", {}))
            result[name]["errors"] = payload.get("errors", {})
            if payload.get("errors") or not payload.get("scores"):
                result[name]["status"] = (
                    "partial" if payload.get("scores") else "failed"
                )
        self.state["heavy_metrics" if heavy else "light_metrics"] = result
        write_json(
            self.metrics_dir / ("heavy_status.json" if heavy else "light_status.json"),
            result,
        )
        return result

    def _score_fast(self, *, smoke=False, gpu_batch_size=16):
        """Score all datasets together so each heavy model is loaded once.

        The original scheduler loads one model for every dataset. This runner
        concatenates datasets with explicit offsets, scores once, then restores
        every score vector to its source order.
        """
        if not self.datasets:
            raise RuntimeError("Chưa có dataset để chấm metric.")

        from .metrics.shard import (
            finalize_shard_scores,
            prepare_multi_dataset_inputs,
        )

        mode = "smoke" if smoke else "fast"
        root = self.metrics_dir / mode
        data_dir = root / "data"
        checkpoint_root = root / "checkpoints"
        result_dir = root / "results"
        manifest_path = data_dir / "manifest.json"
        combined_path = data_dir / "combined.jsonl"
        manifest = prepare_multi_dataset_inputs(
            self.datasets,
            combined_path,
            manifest_path,
            synthetic_sentence_limit=1 if smoke else None,
            human_row_limit=6 if smoke else None,
        )

        all_specs = paper_metric_specs()
        if smoke:
            preferred = {
                "BLEU": "bleu_max_order=1,smooth=True",
                "chrF": "chrf_char_order=4,word_order=0",
                "ROUGE": "rouge_rouge_type=rouge1",
                "METEOR": "meteor_alpha=0.9,beta=3,gamma=0.5",
                "BERTScore": "bertscore_lang=vi,num_layers=5",
                "COMET": "comet_model_name=Unbabel/wmt20-comet-da",
                "BLEURT": "bleurt_checkpoint=bleurt-tiny-128",
            }
            chosen = {
                family: next(
                    spec for spec in all_specs
                    if spec.family == family and spec.key == key
                )
                for family, key in preferred.items()
            }
        else:
            chosen = {}

        task_results = {}
        summaries = {}
        for tokenization in (
            ["syllable", "underthesea"] if self.underthesea else ["syllable"]
        ):
            task_defs = []
            if smoke:
                families = (
                    ["BLEU", "chrF"]
                    if tokenization == "underthesea"
                    else LIGHT + (HEAVY if self.heavy else [])
                )
                task_defs = [(family, chosen[family].key) for family in families]
            else:
                light_families = (
                    ["BLEU", "chrF"]
                    if tokenization == "underthesea"
                    else LIGHT
                )
                task_defs = [(family, None) for family in light_families]
                if tokenization == "syllable" and self.heavy:
                    task_defs.extend(
                        (spec.family, spec.key)
                        for spec in all_specs
                        if spec.family in HEAVY
                    )

            light_jobs = {}
            heavy_jobs = []
            checkpoints = []
            for family, selected_key in task_defs:
                identity = selected_key or f"{family}-all"
                digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
                checkpoint = (
                    checkpoint_root / tokenization / f"{family.lower()}-{digest}.json"
                )
                checkpoints.append(checkpoint)
                label = f"{mode}_{tokenization}_{family.lower()}_{digest}"
                args = [
                    "-m",
                    "vn_meta_judge.metrics.shard",
                    "worker",
                    "--input",
                    combined_path,
                    "--output",
                    checkpoint,
                    "--tokenization",
                    tokenization,
                    "--family",
                    family,
                    "--batch-size",
                    gpu_batch_size if family in HEAVY else 64,
                ]
                if selected_key:
                    args.extend(["--selected-key", selected_key])
                extra_path = (
                    self.heavy_worker_pythonpaths.get(family)
                    if family in HEAVY
                    else None
                )
                env_overrides = (
                    {"USE_TF": "0"} if family == "COMET" else None
                )
                job = lambda n=label, a=args, p=extra_path, e=env_overrides: (
                    self._process(n, a, extra_pythonpath=p, env_overrides=e)
                )
                if family in HEAVY:
                    heavy_jobs.append((label, job))
                else:
                    light_jobs[label] = job

            current = parallel_jobs(light_jobs, self.cpu_workers)
            for label, job in heavy_jobs:
                current[label] = job()
                if current[label]["status"] != "ready":
                    break
            task_results.update(current)
            failed = [
                name for name, item in current.items()
                if item["status"] != "ready"
            ]
            if failed:
                summaries[tokenization] = {
                    "status": "failed",
                    "failed_tasks": failed,
                    "combined_rows": manifest["combined_rows"],
                }
                continue

            if smoke:
                expected = [
                    chosen[family].key
                    for family in (
                        ["BLEU", "chrF"]
                        if tokenization == "underthesea"
                        else LIGHT + (HEAVY if self.heavy else [])
                    )
                ]
            else:
                expected = [
                    spec.key
                    for spec in all_specs
                    if tokenization == "syllable" or spec.family in {"BLEU", "chrF"}
                    if self.heavy or spec.family not in HEAVY
                ]
            summary = finalize_shard_scores(
                manifest_path,
                checkpoints,
                result_dir,
                tokenization=tokenization,
                expected_metric_keys=expected,
            )
            summary["status"] = (
                "ready"
                if not summary["missing_metrics"] and not summary["errors"]
                else "failed"
            )
            summaries[tokenization] = summary

        status = (
            "ready"
            if summaries
            and all(item.get("status") == "ready" for item in summaries.values())
            else "failed"
        )
        report = {
            "status": status,
            "mode": mode,
            "combined_rows": manifest["combined_rows"],
            "tasks": task_results,
            "summaries": summaries,
            "result_dir": str(result_dir),
        }
        self.state["metric_smoke" if smoke else "fast_metrics"] = report
        write_json(root / "status.json", report)
        return report

    def metric_smoke_test(self, *, gpu_batch_size=4):
        """Run one representative config per family on a tiny input."""
        return self._score_fast(smoke=True, gpu_batch_size=gpu_batch_size)

    def score_fast(self, *, gpu_batch_size=16):
        """Run the full metric grid with one model load per configuration."""
        return self._score_fast(smoke=False, gpu_batch_size=gpu_batch_size)

    def correlate(self):
        from .evaluation.correlations import (
            compute_human_correlations,
            compute_synthetic_correlations,
            compute_meta_correlation,
        )

        self.reports = []
        rows = []
        for tokenization in (
            ["syllable", "underthesea"] if self.underthesea else ["syllable"]
        ):
            merged = {}
            for name, dataset in self.datasets.items():
                scores = {}
                fast_path = (
                    self.metrics_dir
                    / "fast"
                    / "results"
                    / f"scores_{name}_{tokenization}.json"
                )
                paths = (
                    [fast_path]
                    if fast_path.is_file()
                    else sorted(
                        (self.metrics_dir / "scores" / name / tokenization).glob("*.json")
                    )
                )
                if name != "human" and not paths:
                    # A cached branch may only contain syllable scores (the
                    # rule-based baseline); do not create a misleading empty
                    # underthesea correlation row for it.
                    continue
                for path in paths:
                    payload = read_json(path, {})
                    for key, values in payload.get("scores", {}).items():
                        if len(values) == len(load_jsonl(dataset)) and all(
                            isinstance(v, (int, float)) and math.isfinite(v)
                            for v in values
                        ):
                            scores[key] = values
                merged[name] = self.metrics_dir / f"scores_{name}_{tokenization}.json"
                write_json(merged[name], {"scores": scores})
            human = (
                compute_human_correlations(self.human_file, merged["human"])
                if "human" in merged
                else {}
            )
            for name in sorted(set(merged) - {"human", "demo"}):
                synthetic = compute_synthetic_correlations(
                    self.datasets[name], merged[name]
                )
                common = [
                    key
                    for key in sorted(set(human) & set(synthetic))
                    if all(
                        math.isfinite(human[key][c])
                        and math.isfinite(synthetic[key][c])
                        for c in ["pearson", "spearman", "kendall"]
                    )
                ]
                meta = (
                    compute_meta_correlation(
                        {k: human[k] for k in common},
                        {k: synthetic[k] for k in common},
                    )
                    if len(common) >= 2
                    else None
                )
                report = {
                    "branch": name,
                    "tokenization": tokenization,
                    "human": human,
                    "synthetic": synthetic,
                    "meta": meta,
                    "excluded_nonfinite_metrics": sorted(
                        (set(human) & set(synthetic)) - set(common)
                    ),
                    "status": "ready"
                    if meta and math.isfinite(meta["meta_spearman"])
                    else "partial",
                }
                path = self.metrics_dir / f"correlation_{name}_{tokenization}.json"
                write_json(path, report)
                self.reports.append(path)
                rows.append(
                    {
                        "branch": name,
                        "tokenization": tokenization,
                        "metrics": len(common),
                        "MC_Spearman": meta["meta_spearman"] if meta else None,
                        "MC_Kendall": meta["meta_kendall"] if meta else None,
                        "status": report["status"],
                    }
                )
        import pandas as pd

        table = pd.DataFrame(rows)
        table.to_csv(self.metrics_dir / "correlation_summary.csv", index=False)
        return table

    def _export_scores(self):
        records = []
        for row in self.human_rows:
            records.append(
                {
                    "id": row.id,
                    "system_id": row.system_id,
                    "source_zh": row.source_zh,
                    "reference_vi": row.reference_vi,
                    "prediction_vi": row.prediction_vi,
                    "final_score": row.human_score,
                    "primary_score": row.metadata.get("primary_score"),
                    "cross_score": row.metadata.get("cross_score"),
                    "quality_flags": row.quality_flags,
                    "do_not_use_for_final_correlation": bool(
                        row.quality_flags or row.human_score is None
                    ),
                }
            )
        save_jsonl(
            [json_safe(r) for r in records],
            self.analysis_dir / "human_scoring_machine.jsonl",
        )
        write_json(
            self.analysis_dir / "human_scoring_machine.json",
            {"scale": [0, 100], "rows": records},
        )
        return {"status": "ready" if records else "skipped", "rows": len(records)}

    def _legacy_b2(self):
        archive = self.code / "data" / "correlations.zip"
        if not archive.is_file():
            return {"status": "skipped", "reason": "missing_correlations_zip"}
        target = self.analysis_dir / "b2"
        extracted = target / "source"
        if not (extracted / "correlations").is_dir():
            with ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    if (
                        not (extracted / member.filename)
                        .resolve()
                        .is_relative_to(extracted.resolve())
                    ):
                        raise ValueError("Invalid B2 archive path")
                bundle.extractall(extracted)
        output = target / "results"
        result = self._process(
            "legacy_b2",
            [
                self.code / "src/metrics/meta_correlation.py",
                extracted / "correlations/generated_datasets",
                extracted / "correlations/original_datasets",
                "--output",
                output,
                "--markdown",
            ],
        )
        result["result_files"] = len(list(output.rglob("*.json")))
        result["scope"] = (
            "Recompute MC from published correlations; not regeneration of MOCHA outputs."
        )
        write_json(target / "summary.json", result)
        return result

    def analyze(self, manual_audit=None):
        jobs = {"human_score_export": self._export_scores, "legacy_b2": self._legacy_b2}
        self.state["analysis"] = parallel_jobs(jobs, self.cpu_workers)
        report = {
            "input": self.summary,
            "generation": self.state.get("generation"),
            "metrics": {k: v for k, v in self.state.items() if k.endswith("metrics")},
            "baselines": self.state["analysis"],
            "manual_audit": {"status": "not_provided"},
        }
        if manual_audit:
            rows = load_jsonl(Path(manual_audit))
            valid = [
                r
                for r in rows
                if r.get("requested_level") in range(6)
                and r.get("observed_level") in range(6)
            ]
            report["manual_audit"] = {
                "status": "ready" if len(valid) == len(rows) and rows else "partial",
                "rows": len(valid),
                "invalid_rows": len(rows) - len(valid),
                "accuracy": sum(
                    r["requested_level"] == r["observed_level"] for r in valid
                )
                / len(valid)
                if valid
                else None,
            }
        write_json(self.analysis_dir / "error_analysis.json", report)
        return self.state["analysis"]

    def demo(self, sentence):
        if sentence.strip() in set(self.frame.Tham_chieu_VI):
            raise ValueError("Câu demo phải nằm ngoài bộ dữ liệu bước sinh bản dịch.")
        path = self.analysis_dir / "demo.jsonl"
        rows = generate_rule_baseline(
            [BenchmarkRow("demo-001", "", sentence)], seed=self.seed
        )
        save_jsonl(rows, path)
        scores = self.analysis_dir / "demo_scores.json"
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            scores.exists()
            and read_json(scores, {}).get("config", {}).get("input_sha256")
            != fingerprint
        ):
            scores.replace(
                self.analysis_dir
                / ("demo_scores_" + str(scores.stat().st_mtime_ns) + ".json")
            )
        result = self._process(
            "demo_metrics",
            [
                "-m",
                "vn_meta_judge.notebook_workflow",
                "metric",
                path,
                scores,
                "syllable",
                "BLEU,chrF",
                "",
            ],
        )
        import pandas as pd

        values = read_json(scores, {}).get("scores", {})
        table = pd.DataFrame(
            {
                "level": [r["damage_level"] for r in rows],
                "text": [r["prediction_vi"] for r in rows],
                **values,
            }
        )
        table.to_csv(self.analysis_dir / "demo.csv", index=False)
        self.state["demo"] = {
            **result,
            "metric_count": len(values),
            "backend": "rule_based",
        }
        if not values:
            self.state["demo"]["status"] = "failed"
        return table

    def metric_diagnostics(self):
        """Mean scores by damage level and direction violations for each metric."""
        import pandas as pd

        records = []
        for name, path in self.datasets.items():
            if name == "human":
                continue
            levels = pd.Series([r["damage_level"] for r in load_jsonl(path)])
            for tokenization in ["syllable", "underthesea"]:
                score_path = self.metrics_dir / f"scores_{name}_{tokenization}.json"
                if not score_path.is_file():
                    score_path = (
                        self.metrics_dir
                        / "fast"
                        / "results"
                        / f"scores_{name}_{tokenization}.json"
                    )
                scores = read_json(score_path, {}).get("scores", {})
                for metric, values in scores.items():
                    means = pd.Series(values).groupby(levels).mean().reindex(range(6))
                    records.append(
                        {
                            "branch": name,
                            "tokenization": tokenization,
                            "metric": metric,
                            **{f"level_{i}": means[i] for i in range(6)},
                            "increase_steps": int(means.diff().gt(0).sum()),
                            "drop_0_to_5": means[0] - means[5],
                        }
                    )
        table = pd.DataFrame(records)
        table.to_csv(self.analysis_dir / "metric_diagnostics.csv", index=False)
        return table

    def manual_audit_sample(self, count=30):
        import random

        rows = []
        for name in ["zero_shot", "few_shot"]:
            path = self.datasets.get(name)
            if path:
                rows.extend(load_jsonl(path))
        output = self.analysis_dir / "manual_audit_template.jsonl"
        if not rows:
            return None
        rng = random.Random(self.seed)
        rng.shuffle(rows)
        buckets = [
            [r for r in rows if r["damage_level"] == level] for level in range(6)
        ]
        selected = []
        while len(selected) < min(count, len(rows)):
            for bucket in buckets:
                if bucket and len(selected) < count:
                    row = bucket.pop()
                    selected.append(
                        {
                            "id": row["id"],
                            "prompt_type": row["prompt_type"],
                            "reference_vi": row["reference_vi"],
                            "prediction_vi": row["prediction_vi"],
                            "requested_level": row["damage_level"],
                            "observed_level": None,
                            "note": "",
                        }
                    )
        # Preserve any labels already entered in an existing template.
        if not output.exists():
            save_jsonl(selected, output)
        return output

    def finalize(self):
        stages = [
            value
            for group in self.state.values()
            for value in (
                group.values()
                if isinstance(group, dict) and "status" not in group
                else [group]
            )
            if isinstance(value, dict) and "status" in value
        ]
        cache_mode = (
            self.state.get("fast_metrics", {}).get("mode") == "source-cache"
        )
        issues = [
            item
            for item in stages
            if item["status"] not in {"ready", "skipped", "info"}
        ]
        if not self.heavy and not cache_mode:
            issues.append({"status": "skipped", "reason": "heavy_metrics_disabled"})
        if not self.human_rows:
            issues.append({"status": "skipped", "reason": "human_scores_missing"})
        if self.summary.get("human_excluded"):
            issues.append(
                {
                    "status": "info" if cache_mode else "partial",
                    "reason": "human_rows_excluded",
                    "rows": self.summary["human_excluded"],
                }
            )
        for path in self.reports:
            if read_json(path, {}).get("status") != "ready":
                issues.append(
                    {
                        "status": "partial",
                        "reason": "correlation_incomplete",
                        "file": path.name,
                    }
                )
        required = [
            self.data / "input_audit.json",
            self.generation_dir / "generation_summary.json",
            self.metrics_dir / "correlation_summary.csv",
            self.analysis_dir / "error_analysis.json",
        ]
        if self.state.get("demo", {}).get("status") != "skipped":
            required.append(self.analysis_dir / "demo.csv")
        for required in required:
            if not required.is_file():
                issues.append(
                    {
                        "status": "partial",
                        "reason": "missing_stage",
                        "file": required.name,
                    }
                )
        error_report = read_json(self.analysis_dir / "error_analysis.json", {})
        if error_report.get("manual_audit", {}).get("status") not in {
            "ready",
            "not_provided",
        }:
            issues.append({"status": "partial", "reason": "manual_audit_pending"})
        blocking_issues = [
            item for item in issues if item["status"] not in {"info", "skipped"}
        ]
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": self.summary,
            "status": "complete" if not blocking_issues else "partial",
            "issues": issues,
            "stages": self.state,
            "artifacts": [
                str(p.relative_to(self.output))
                for p in sorted(self.output.rglob("*"))
                if p.is_file() and p.suffix in {".json", ".jsonl", ".csv", ".png"}
            ],
            "correlations": [str(p.relative_to(self.output)) for p in self.reports],
        }
        write_json(self.output / "manifest.json", report)
        return report


def metric_worker(input_path, output, tokenization, families, key):
    from .metrics.runner import run_metric_scores

    # Heavy checkpoints are prepared in this process and released at process exit.
    if key:
        spec = next(s for s in paper_metric_specs() if s.key == key)
        if spec.family == "COMET":
            from comet import download_model

            download_model(spec.kwargs["model_name"])
        elif spec.family == "BLEURT":
            import evaluate
            from metrics.hf_metrics import _find_bleurt_local

            try:
                _find_bleurt_local(spec.kwargs["checkpoint"])
            except FileNotFoundError:
                evaluate.load(
                    "bleurt",
                    config_name=spec.kwargs["checkpoint"],
                )
    return run_metric_scores(
        Path(input_path),
        Path(output),
        tokenization,
        selected_families=families.split(","),
        selected_keys=[key] if key else None,
        batch_size=4 if key else 32,
    )


def prefetch_light(
    retry_delays=(15.0, 30.0, 60.0),
    sleep_fn=time.sleep,
):
    nltk_dir = Path(os.environ.get("NLTK_DATA", str(Path.cwd() / "cache" / "nltk")))
    nltk_dir.mkdir(parents=True, exist_ok=True)
    import evaluate
    import nltk

    def fetch(label, operation):
        attempts = len(retry_delays) + 1
        for attempt in range(1, attempts + 1):
            print(f"PREFETCH {label} | lần {attempt}/{attempts}", flush=True)
            try:
                result = operation()
                if result is False:
                    raise RuntimeError("download returned False")
                print(f"PREFETCH {label} | OK", flush=True)
                return result
            except Exception as exc:
                if attempt == attempts:
                    raise RuntimeError(
                        f"PREFETCH {label} thất bại: {type(exc).__name__}: {exc}"
                    ) from exc
                delay = float(retry_delays[attempt - 1])
                print(
                    f"PREFETCH {label} | lỗi {type(exc).__name__}: {exc} | "
                    f"thử lại sau {delay:.0f}s",
                    flush=True,
                )
                sleep_fn(delay)

    # chrF and METEOR use direct local adapters. Only BLEU and ROUGE need the
    # evaluate scripts, which halves Hub metadata requests during parallel setup.
    for name in ["bleu", "rouge"]:
        fetch(f"evaluate/{name}", lambda name=name: evaluate.load(name))

    for name in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
        fetch(
            f"nltk/{name}",
            lambda name=name: nltk.download(
                name,
                download_dir=str(nltk_dir),
                quiet=True,
                raise_on_error=True,
            ),
        )


if __name__ == "__main__":
    if sys.argv[1] == "prefetch":
        prefetch_light()
    elif sys.argv[1] == "metric":
        print(json.dumps(metric_worker(*sys.argv[2:])))
