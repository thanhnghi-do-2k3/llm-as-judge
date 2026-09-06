# Text Damager — Master's Thesis Code

Code accompanying my master's thesis (Lukas Eigler).

The project investigates a **"text damager"**: given an input text and a target score/level, the damager produces a version of the text degraded — semantically and/or syntactically — to roughly that level. The damaged outputs are used to validate and compare reference-free evaluation metrics, with a focus on Czech.

For the coursework's Vietnamese experiment, see [`README_VIETNAMESE_PIPELINE.md`](README_VIETNAMESE_PIPELINE.md). It documents the new separated pipeline for Person 2 (generation) and Person 3 (metrics), the Google Sheet export contract, commands, missing user inputs, and the reuse boundary with the original code.

The repo contains the active pipeline used in the thesis:

1. Loading and pre-processing of human-feedback datasets — ROSE, MOCHA, CUS-QA, WMT (`human_feedback_datasets/`).
2. Generating damaged texts with LLMs on the cluster (`generation/generate.py`).
3. Computing standard MT/summarisation metrics on the (reference, damaged) pairs (`metrics/`).
4. Correlating those metric scores with human judgements (`metrics/correlation.py`, `metrics/meta_correlation.py`).
5. SFT and GRPO fine-tuning of small LLMs as reference-free metric models (`finetuning/`).
6. Large-scale evaluation and analysis runs on the Karolina cluster (`generation/`).

## Repository layout

```
data/
├── correlations.zip            Pre-computed metric–human correlation results (JSON)
│                               Contains: generated_datasets/ (LLM + finetuned model runs)
│                                     and original_datasets/ (human-annotated benchmarks)
├── damaged_datasets/           LLM-generated damaged texts (one zip per dataset)
│   ├── cus_qa.zip, mocha.zip, rose.zip, wmt.zip, …
└── finetuned_model_outputs/    Fine-tuned model outputs, s2000_lp0.5_kl0.05 runs
    ├── cus_qa.zip, mocha.zip, rose.zip, wmt.zip
src/
├── vn_meta_judge/                Vietnamese coursework pipeline:
│                                 data → preprocessing → generation → metrics → evaluation
├── human_feedback_datasets/    ROSE / MOCHA / CUS-QA / WMT loaders + prompts
├── metrics/                    HF metrics, correlation, meta-correlation
├── finetuning/                 SFT + GRPO trainers, Slurm submit scripts,
│                               training-curve plots (`models_training_plots/`)
└── generation/                 Damaged-dataset generation pipeline,
                                evaluation and analysis scripts, Slurm scripts
```

The coursework pipeline has one declared primary generator (Gemini) and an
optional, explicitly isolated DeepSeek ablation. Both use the same resumable
API runner, but their output files and reports must never be concatenated.

## Datasets

The code consumes (but does not redistribute) the following public datasets:

- **ROSE** — *Revisiting the Gold Standard: Grounding Summarization Evaluation with Robust Human Evaluation* — <https://huggingface.co/datasets/Salesforce/rose>
- **MOCHA** (`anthonychen/mocha`) — 40 K human judgement scores on model outputs from 6 reading-comprehension QA datasets — <https://huggingface.co/datasets/anthonychen/mocha>
- **WMT 2021–2025** human evaluation data — <https://github.com/google-research/mt-metrics-eval>
- **CUS-QA** — Czech / Slovak / Ukrainian QA dataset — <https://huggingface.co/datasets/ufal/cus-qa>

Generated damaged texts are archived per-dataset under `data/damaged_datasets/`. Fine-tuned model outputs (best run: `s2000_lp0.5_kl0.05`) are in `data/finetuned_model_outputs/`. Pre-computed correlation results are in `data/correlations.zip`. Model checkpoints, training logs, and full evaluation outputs are not included (~1 TB total).

## Installation

The pipeline is built on PyTorch (CUDA 12.6) and a number of HuggingFace and metric libraries. The recommended setup uses [`uv`](https://docs.astral.sh/uv/):

```shell
uv venv --python 3.11
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu126
uv pip install \
    matplotlib scikit-learn seaborn pandas tqdm \
    sacrebleu nltk rouge_score bert_score unbabel-comet evaluate \
    transformers datasets accelerate \
    vllm unsloth prometheus-eval \
    "setuptools<82"
uv pip install git+https://github.com/google-research/bleurt.git
uv pip install tf-keras
```

For type-checking:

```shell
pip install scipy-stubs tqdm-stubs pandas-stubs
```

`mt-metrics-eval` (used for WMT correlations) should be cloned separately from <https://github.com/google-research/mt-metrics-eval> and installed in the same environment.

## Reproducing thesis results

The thesis has two main experimental pipelines:

### Part 1 — LLM as metric validator

Use an LLM to generate a synthetic damaged dataset, then measure how well standard metrics correlate with the pseudo-labels (damage level) it produces.

1. **Generate damaged texts** — `src/generation/generate.py`, submitted via `submit_generate.sh`.
2. **Score with metrics** — `src/metrics/eval_dataset.py` (computes BLEU, chrF, ROUGE, BERTScore, COMET, etc. on each generated example).
3. **Correlate metrics vs pseudo-labels** — `src/metrics/correlation.py` on the generated datasets.

### Part 2 — Meta-correlation

Extend Part 1 to measure how well metric rankings on synthetic data predict metric rankings on human-annotated data.

1. **Steps 1–3 from Part 1** (metrics vs pseudo-label correlations on generated datasets).
2. **Correlate metrics vs human judgements** — `src/metrics/correlation.py` on the human-annotated benchmarks (ROSE, MOCHA, CUS-QA, WMT).
3. **Meta-correlate** — `src/metrics/meta_correlation.py` correlates the two sets of per-metric correlations, producing a single score that reflects how faithfully the synthetic benchmark ranks metrics relative to human judgement.

Pre-computed results for both parts are archived in `data/correlations.zip`.

## Local/API adaptation for this coursework repo

The original generator defaults to local HuggingFace models and the Karolina
cluster dataset layout. In this checkout, `src/generation/generate.py` also
supports API backends so the Vietnamese experiment can run without 8 x A100
GPUs.

Set the dataset root either through `META_JUDGE_DATASETS_DIR` or with
`--datasets_dir`. The default is `./datasets` at the repository root.

Gemini example:

```shell
export GEMINI_API_KEY=...
PYTHONPATH=src python src/generation/generate.py \
  --backend gemini \
  --api_model gemini-2.5-flash-lite \
  --dataset_config wmt_24_en_cs \
  --datasets_dir ./datasets \
  --prompt_type zero_shot \
  --data_count 20 \
  --output_file outputs/damaged.json \
  --save_as_dataset
```

The legacy author's generator also has an OpenAI-compatible backend, for
example DeepSeek. This is separate from the coursework CLI below; it is not an
automatic fallback and its outputs must not be merged with a Gemini run:

```shell
export DEEPSEEK_API_KEY=...
PYTHONPATH=src python src/generation/generate.py \
  --backend openai \
  --api_model deepseek-chat \
  --dataset_config wmt_24_en_cs \
  --datasets_dir ./datasets \
  --prompt_type zero_shot \
  --data_count 20 \
  --output_file outputs/damaged.json \
  --save_as_dataset
```

`src/metrics/meta_correlation.py` now warns instead of crashing when fewer than
the original 28 metric configurations overlap. That keeps ablation runs usable,
but full reproduction should still aim for the 28 configurations from the
paper.

## Hardware

Experiments were run on the [IT4Innovations Karolina](https://www.it4i.cz/karolina) GPU cluster (8 × A100 80 GB nodes). Some scripts assume Slurm + that environment; adapt for other clusters as needed.

> **Note on hardcoded paths**: scripts under `src/generation/` and `src/finetuning/` contain absolute paths specific to the Karolina cluster (`/mnt/proj1/fta-25-74/dev/master_thesis/...`). Update these paths to match your local environment before running.
