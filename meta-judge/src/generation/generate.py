from pathlib import Path
from tqdm import tqdm
import argparse
import json, itertools, os, time, urllib.error, urllib.request
from human_feedback_datasets.processed_dataset import ProcessedDataset, Example
from human_feedback_datasets.prompts import (
    # SUMMARIZATION_PROMPT_V2, CUS_QA_PROMPT, CUS_QA_PROMPT_GEMINI_1, 
    # CUS_QA_PROMPT_GEMINI_2, CUS_QA_PROMPT_GEMINI_3, MOCHA_PROMPT, 
    # MOCHA_PROMPT_2, ROSE_PROMPT_LONG, ROSE_PROMPT_OPTIMIZED
    CUS_QA_ZERO_SHOT_PROMPT, CUS_QA_FEW_SHOT_PROMPT,
    CUS_QA_ZERO_SHOT_REASONING_PROMPT, CUS_QA_FEW_SHOT_REASONING_PROMPT,

    MOCHA_ZERO_SHOT_PROMPT, MOCHA_FEW_SHOT_PROMPT,
    MOCHA_ZERO_SHOT_REASONING_PROMPT, MOCHA_FEW_SHOT_REASONING_PROMPT,

    ROSE_ZERO_SHOT_PROMPT, ROSE_FEW_SHOT_PROMPT,
    ROSE_ZERO_SHOT_REASONING_PROMPT, ROSE_FEW_SHOT_REASONING_PROMPT,
    
    WMT_ZERO_SHOT_PROMPT, WMT_FEW_SHOT_PROMPT,
    WMT_ZERO_SHOT_REASONING_PROMPT, WMT_FEW_SHOT_REASONING_PROMPT
)
from human_feedback_datasets.alternative_prompts import (
    CUS_QA_ZERO_SHOT_PROMPT_TESTING_1, CUS_QA_FEW_SHOT_PROMPT_TESTING_1,
    CUS_QA_ZERO_SHOT_PROMPT_TESTING_2, CUS_QA_FEW_SHOT_PROMPT_TESTING_2,
    CUS_QA_ZERO_SHOT_PROMPT_TESTING_3, CUS_QA_FEW_SHOT_PROMPT_TESTING_3,
    CUS_QA_ZERO_SHOT_PROMPT_TESTING_4, CUS_QA_FEW_SHOT_PROMPT_TESTING_4,
    CUS_QA_ZERO_SHOT_PROMPT_TESTING_5, CUS_QA_FEW_SHOT_PROMPT_TESTING_5
)
from typing import Any, List, Iterable, Dict, TypeVar, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = Path(os.environ.get("META_JUDGE_DATASETS_DIR", PROJECT_ROOT / "datasets")).expanduser()

# --- DEFAULT GLOBAL PARAMS ---
DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 400,
    "do_sample": False,
}

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OPENAI_COMPATIBLE_MODEL = "deepseek-chat"

# ADD MODEL LOAD CONFIG

# --- 1. MODEL CONFIGS ---
MODEL_CONFIGS = {
    "qwen3": {
        "model_id": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "params": DEFAULT_GENERATION_PARAMS.copy()
    },
    "qwen3_14b": {
        "model_id": "Qwen/Qwen3-14B",
        "params": DEFAULT_GENERATION_PARAMS.copy()
    },
    "qwen3_30b": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "params": DEFAULT_GENERATION_PARAMS.copy()
    },
    "qwen3_30b_thinking": {
        "model_id": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "params": DEFAULT_GENERATION_PARAMS.copy()
    },
    "llama8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "params": DEFAULT_GENERATION_PARAMS.copy()
    },
    "llama70b": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "params": DEFAULT_GENERATION_PARAMS.copy()
    },
    "llama4scout": {
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "params": DEFAULT_GENERATION_PARAMS.copy()
    },
}

# --- 2. DATASET CONFIGS ---
DATASET_CONFIGS = {
    "rose_cnndm_validation": {
        "path": DATASETS_DIR / 'rose_cnndm_validation.jsonl',
        "prompts": [ROSE_ZERO_SHOT_PROMPT, ROSE_FEW_SHOT_PROMPT],
        "reasoning_prompts": [ROSE_ZERO_SHOT_REASONING_PROMPT, ROSE_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "source_text: {source}\nreference_summary: {reference}\ndamage_level: {damage_level}"
    },
    "rose_cnndm_test": {
        "path": DATASETS_DIR / 'rose_cnndm_test.jsonl',
        "prompts": [ROSE_ZERO_SHOT_PROMPT, ROSE_FEW_SHOT_PROMPT],
        "reasoning_prompts": [ROSE_ZERO_SHOT_REASONING_PROMPT, ROSE_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "source_text: {source}\nreference_summary: {reference}\ndamage_level: {damage_level}"
    },
    "rose_cnndm_protocol": {
        "path": DATASETS_DIR / 'rose_cnndm_protocol.jsonl',
        "prompts": [ROSE_ZERO_SHOT_PROMPT, ROSE_FEW_SHOT_PROMPT],
        "reasoning_prompts": [ROSE_ZERO_SHOT_REASONING_PROMPT, ROSE_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "source_text: {source}\nreference_summary: {reference}\ndamage_level: {damage_level}"
    },
    "rose_xsum": {
        "path": DATASETS_DIR / 'rose_xsum.jsonl',
        "prompts": [ROSE_ZERO_SHOT_PROMPT, ROSE_FEW_SHOT_PROMPT],
        "reasoning_prompts": [ROSE_ZERO_SHOT_REASONING_PROMPT, ROSE_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "source_text: {source}\nreference_summary: {reference}\ndamage_level: {damage_level}"
    },
    "rose_cnndm_protocol_gpt3": {
        "path": DATASETS_DIR / 'rose_cnndm_protocol_gpt3.jsonl',
        "prompts": [ROSE_ZERO_SHOT_PROMPT, ROSE_FEW_SHOT_PROMPT],
        "reasoning_prompts": [ROSE_ZERO_SHOT_REASONING_PROMPT, ROSE_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "source_text: {source}\nreference_summary: {reference}\ndamage_level: {damage_level}"
    },
    
    "mocha_train": {
        "path": DATASETS_DIR / 'mocha_train.jsonl',
        "prompts": [MOCHA_ZERO_SHOT_PROMPT, MOCHA_FEW_SHOT_PROMPT],
        "reasoning_prompts": [MOCHA_ZERO_SHOT_REASONING_PROMPT, MOCHA_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "passage: {context}\nquestion: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
    },
    "mocha_test": {
        "path": DATASETS_DIR / 'mocha_test.jsonl',
        "prompts": [MOCHA_ZERO_SHOT_PROMPT, MOCHA_FEW_SHOT_PROMPT],
        "reasoning_prompts": [MOCHA_ZERO_SHOT_REASONING_PROMPT, MOCHA_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "passage: {context}\nquestion: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
    },
    "mocha_validation": {
        "path": DATASETS_DIR / 'mocha_validation.jsonl',
        # "path": Path('/home/eiglerl/dev/school/master_thesis/datasets/mocha_validation.jsonl'),
        "prompts": [MOCHA_ZERO_SHOT_PROMPT, MOCHA_FEW_SHOT_PROMPT],
        "reasoning_prompts": [MOCHA_ZERO_SHOT_REASONING_PROMPT, MOCHA_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "passage: {context}\nquestion: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
    },
    "mocha_minimal_pairs": {
        "path": DATASETS_DIR / 'mocha_minimal_pairs.jsonl',
        "prompts": [MOCHA_ZERO_SHOT_PROMPT, MOCHA_FEW_SHOT_PROMPT],
        "reasoning_prompts": [MOCHA_ZERO_SHOT_REASONING_PROMPT, MOCHA_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "passage: {context}\nquestion: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
    },

    "qa_cz_en": {
        "path": DATASETS_DIR / 'qa_text-CZ_dev_en_binary.jsonl',
        "prompts": [CUS_QA_ZERO_SHOT_PROMPT, CUS_QA_FEW_SHOT_PROMPT],
        "reasoning_prompts": [CUS_QA_ZERO_SHOT_REASONING_PROMPT, CUS_QA_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "question: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
    },
    "qa_cz_orig": {
        "path": DATASETS_DIR / 'qa_text-CZ_dev_orig_binary.jsonl',
        "prompts": [CUS_QA_ZERO_SHOT_PROMPT, CUS_QA_FEW_SHOT_PROMPT],
        "reasoning_prompts": [CUS_QA_ZERO_SHOT_REASONING_PROMPT, CUS_QA_FEW_SHOT_REASONING_PROMPT],
        "testing_prompts": [    
            CUS_QA_ZERO_SHOT_PROMPT_TESTING_1, CUS_QA_FEW_SHOT_PROMPT_TESTING_1,
            CUS_QA_ZERO_SHOT_PROMPT_TESTING_2, CUS_QA_FEW_SHOT_PROMPT_TESTING_2,
            CUS_QA_ZERO_SHOT_PROMPT_TESTING_3, CUS_QA_FEW_SHOT_PROMPT_TESTING_3,
            CUS_QA_ZERO_SHOT_PROMPT_TESTING_4, CUS_QA_FEW_SHOT_PROMPT_TESTING_4,
            CUS_QA_ZERO_SHOT_PROMPT_TESTING_5, CUS_QA_FEW_SHOT_PROMPT_TESTING_5
                            ],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "question: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
    },
    "qa_sk_en": {
        "path": DATASETS_DIR / 'qa_text-SK_dev_en_binary.jsonl',
        "prompts": [CUS_QA_ZERO_SHOT_PROMPT, CUS_QA_FEW_SHOT_PROMPT],
        "reasoning_prompts": [CUS_QA_ZERO_SHOT_REASONING_PROMPT, CUS_QA_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "question: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
    },
    "qa_sk_orig": {
        "path": DATASETS_DIR / 'qa_text-SK_dev_orig_binary.jsonl',
        "prompts": [CUS_QA_ZERO_SHOT_PROMPT, CUS_QA_FEW_SHOT_PROMPT],
        "reasoning_prompts": [CUS_QA_ZERO_SHOT_REASONING_PROMPT, CUS_QA_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "question: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
    },
    "qa_uk_en": {
        "path": DATASETS_DIR / 'qa_text-UA_dev_en_binary.jsonl',
        "prompts": [CUS_QA_ZERO_SHOT_PROMPT, CUS_QA_FEW_SHOT_PROMPT],
        "reasoning_prompts": [CUS_QA_ZERO_SHOT_REASONING_PROMPT, CUS_QA_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "question: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
    },
    "qa_uk_orig": {
        "path": DATASETS_DIR / 'qa_text-UA_dev_orig_binary.jsonl',
        "prompts": [CUS_QA_ZERO_SHOT_PROMPT, CUS_QA_FEW_SHOT_PROMPT],
        "reasoning_prompts": [CUS_QA_ZERO_SHOT_REASONING_PROMPT, CUS_QA_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "question: {source}\ninput_answer: {reference}\ndamage_level: {damage_level}"
    },


    "wmt_24_cs_uk": {
        "path": DATASETS_DIR / 'wmt_2024' / 'wmt_2024_cs-uk.jsonl',
        "prompts": [WMT_ZERO_SHOT_PROMPT, WMT_FEW_SHOT_PROMPT],
        "reasoning_prompts": [WMT_ZERO_SHOT_REASONING_PROMPT, WMT_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "source_sentence: {source}\nreference_translation: {reference}\ndamage_level: {damage_level}"
    },
    "wmt_24_en_is": {
        "path": DATASETS_DIR / 'wmt_2024' / 'wmt_2024_en-is.jsonl',
        "prompts": [WMT_ZERO_SHOT_PROMPT, WMT_FEW_SHOT_PROMPT],
        "reasoning_prompts": [WMT_ZERO_SHOT_REASONING_PROMPT, WMT_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "source_sentence: {source}\nreference_translation: {reference}\ndamage_level: {damage_level}"
    },
    "wmt_24_en_cs": {
        "path": DATASETS_DIR / 'wmt_2024' / 'wmt_2024_en-cs.jsonl',
        "prompts": [WMT_ZERO_SHOT_PROMPT, WMT_FEW_SHOT_PROMPT],
        "reasoning_prompts": [WMT_ZERO_SHOT_REASONING_PROMPT, WMT_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "source_sentence: {source}\nreference_translation: {reference}\ndamage_level: {damage_level}"
    },
    "wmt_21_en_ha": {
        "path": DATASETS_DIR / 'wmt_2021' / 'wmt_2021_en-ha.jsonl',
        "prompts": [WMT_ZERO_SHOT_PROMPT, WMT_FEW_SHOT_PROMPT],
        "reasoning_prompts": [WMT_ZERO_SHOT_REASONING_PROMPT, WMT_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "source_sentence: {source}\nreference_translation: {reference}\ndamage_level: {damage_level}"
    },
    "wmt_21_xh_zu": {
        "path": DATASETS_DIR / 'wmt_2021' / 'wmt_2021_xh-zu.jsonl',
        "prompts": [WMT_ZERO_SHOT_PROMPT, WMT_FEW_SHOT_PROMPT],
        "reasoning_prompts": [WMT_ZERO_SHOT_REASONING_PROMPT, WMT_FEW_SHOT_REASONING_PROMPT],
        "prompt_names": ["zero_shot", "few_shot"],
        "user_template": "source_sentence: {source}\nreference_translation: {reference}\ndamage_level: {damage_level}"
    },


}

def get_unique_data(data: ProcessedDataset) -> ProcessedDataset:
    seen_inputs: Set[Tuple[str, str, str]] = set()
    unique_data = []
    for item in data:
        if (item.reference, item.input, item.context) not in seen_inputs:
            seen_inputs.add((item.reference, item.input, item.context))
            unique_data.append(item)
    return ProcessedDataset(unique_data)

def gen_data(dataset_config: Dict[str, str], data: ProcessedDataset, damage_levels: range, prompt: str) -> Iterable[Dict[str, Any]]:
    template = dataset_config["user_template"]
    for idx, item in enumerate(data):
        for level in damage_levels:
            messages = [
                {"role": "system", "content": prompt}, 
                {"role": "user", "content": template.format(
                    context=item.context if item.context else "",
                    source=item.input,
                    reference=item.reference,
                    damage_level=level
                )}
            ]
            yield {
                "messages": messages,
                "metadata": {
                    "index": idx,
                    "damage_level": level,
                    "input_text": item.input,
                    "reference": item.reference,
                    "context": item.context if item.context else ""
                }
            }

T = TypeVar('T')
def batch(iterable: Iterable[T], n: int) -> Iterable[List[T]]:
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch

def save_checkpoint(path: Path, config_header: Dict, outputs: List[Dict]) -> None:
    output_json = {
        "config": config_header,
        "outputs": outputs
    }
    temp_path = path.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
    temp_path.replace(path)
    print(f"  [Checkpoint] Saved {len(outputs)} samples to {path}")

def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)

def _request_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], retries: int = 3) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"API request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"API request failed: {exc}") from exc
        time.sleep(2 ** attempt)
    raise RuntimeError("API request failed after retries")

def call_gemini(messages: List[Dict[str, str]], model: str, max_output_tokens: int, temperature: float = 0) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use --backend gemini.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    system_text = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    user_text = "\n\n".join(message["content"] for message in messages if message["role"] != "system")
    payload = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    data = _request_json(url, payload, {"Content-Type": "application/json"})
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response shape: {data}") from exc

def call_openai_compatible(messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float = 0) -> str:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or DEEPSEEK_API_KEY to use --backend openai.")

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = _request_json(
        url,
        payload,
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenAI-compatible response shape: {data}") from exc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using Modular Configs.")
    
    # Selection Arguments
    parser.add_argument("--model_config", type=str, help=f"Key from MODEL_CONFIGS: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--dataset_config", type=str, required=True, help=f"Key from DATASET_CONFIGS: {list(DATASET_CONFIGS.keys())}")
    parser.add_argument("--backend", choices=["local", "gemini", "openai"], default="local", help="Generation backend. Use gemini/openai for API generation without loading a local HF model.")
    parser.add_argument("--api_model", type=str, default=None, help="API model name for --backend gemini/openai.")
    parser.add_argument("--datasets_dir", type=Path, default=None, help="Override dataset directory for built-in dataset configs.")
    parser.add_argument("--prompt_type", type=str, default=None, help="Specific prompt to use (e.g., 'zero_shot', 'few_shot'). If omitted, runs ALL prompts defined for the dataset.")
    parser.add_argument("--reasoning", action="store_true", default=False, help="If set, uses reasoning prompts instead of standard prompts.")
    parser.add_argument("--save_as_dataset", action="store_true", help="If set, saves as ProcessedDataset (.jsonl). Otherwise saves as standard JSON.")

    parser.add_argument("--test_set", type=int, default=None, help="Select which testing prompt pair to run (1-5). Runs both Zero and Few shot for that set.")
    # Standard Arguments
    parser.add_argument("--local_model_path", type=str, default=None, help="Path to a local model directory. Overrides the model_id from the config.")
    parser.add_argument("--output_file", type=Path, required=True, help="Base output path.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_count", type=int, default=-1)
    parser.add_argument("--damage_level_min", type=int, default=0)
    parser.add_argument("--damage_level_max", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--override_max_tokens", type=int, default=None, help="Override max_new_tokens for this run.")
    parser.add_argument("--sample", action="store_true", help="Override 'do_sample' for this run.")

    args = parser.parse_args()

    # 1. SETUP CONFIGS
    if args.datasets_dir:
        for config in DATASET_CONFIGS.values():
            config["path"] = args.datasets_dir.expanduser() / config["path"].relative_to(DATASETS_DIR)

    if args.backend == "gemini":
        model_conf = {
            "model_id": args.api_model or DEFAULT_GEMINI_MODEL,
            "params": DEFAULT_GENERATION_PARAMS.copy()
        }
    elif args.backend == "openai":
        model_conf = {
            "model_id": args.api_model or DEFAULT_OPENAI_COMPATIBLE_MODEL,
            "params": DEFAULT_GENERATION_PARAMS.copy()
        }
    elif args.local_model_path:
        print(f"Using local model path: {args.local_model_path}")
        model_conf = {
            "model_id": args.local_model_path,
            "params": DEFAULT_GENERATION_PARAMS.copy()
        }
    elif args.model_config not in MODEL_CONFIGS:
        raise ValueError(f"Model config '{args.model_config}' not found.")
    else:
        model_conf = MODEL_CONFIGS[args.model_config]
        
    if args.dataset_config not in DATASET_CONFIGS:
        raise ValueError(f"Dataset config '{args.dataset_config}' not found.")

    
    dataset_conf = DATASET_CONFIGS[args.dataset_config]
    
    gen_params = model_conf["params"].copy()
    if args.override_max_tokens:
        gen_params["max_new_tokens"] = args.override_max_tokens
    if args.sample:
        gen_params["do_sample"] = True

    print(f"=== Configuration: Model[{model_conf['model_id']}] | Dataset[{args.dataset_config}] ===")

    # 2. PROMPT SELECTION LOGIC
    if args.test_set:
        start_index = (args.test_set - 1) * 2  # Each test set has 2 prompts (zero and few shot)
        end_index = start_index + 2
        if "testing_prompts" not in dataset_conf:
            raise ValueError(f"Dataset '{args.dataset_config}' does not have testing prompts defined.")
        if end_index > len(dataset_conf["testing_prompts"]):
            raise ValueError(f"Test set {args.test_set} is out of range for dataset '{args.dataset_config}'. Available test sets: 1 to {len(dataset_conf['testing_prompts'])//2}")
        raw_prompts = dataset_conf["testing_prompts"][start_index:end_index]
    elif args.reasoning:
        if "reasoning_prompts" not in dataset_conf:
            raise ValueError(f"Dataset '{args.dataset_config}' does not have reasoning prompts defined.")
        raw_prompts = dataset_conf["reasoning_prompts"]
    else:
        raw_prompts = dataset_conf["prompts"]
    # Ensure names exist
    raw_names = dataset_conf.get("prompt_names", [f"prompt_{i}" for i in range(len(raw_prompts))])

    # Filter prompts based on argument
    if args.prompt_type:
        if args.prompt_type not in raw_names:
            raise ValueError(f"Prompt '{args.prompt_type}' not found in dataset '{args.dataset_config}'. Available prompts: {raw_names}")
        
        # Find index and select only that one
        target_index = raw_names.index(args.prompt_type)
        prompts_to_run = [raw_prompts[target_index]]
        names_to_run = [raw_names[target_index]]
        print(f"Selected specific prompt: {args.prompt_type}")
    else:
        # Run all
        prompts_to_run = raw_prompts
        names_to_run = raw_names
        print(f"No specific prompt selected. Running all {len(names_to_run)} prompts for this dataset.")

    # 3. LOAD DATASET
    if not dataset_conf['path'].exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_conf['path']}")
    dataset = ProcessedDataset.load(dataset_conf['path'])
    dataset = get_unique_data(dataset)
    if args.data_count > -1:
            dataset = dataset.head(args.data_count)
    print(f"Loaded {len(dataset)} samples from {dataset_conf['path']}.")

    # 4. LOAD MODEL
    model_path_or_id = model_conf['model_id']
    text_generator = None
    if args.backend == "local":
        from transformers import pipeline
        import torch

        print(f"Loading model: {model_path_or_id}")
        model_loading_args = {
            "attn_implementation": "sdpa"
        }
        text_generator = pipeline(
            "text-generation",
            model=model_path_or_id,
            tokenizer=model_path_or_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            model_kwargs=model_loading_args,
        )
        if text_generator.tokenizer.pad_token_id is None:
            text_generator.tokenizer.pad_token_id = text_generator.tokenizer.eos_token_id
        print("  Model loaded.")

        gen_params["pad_token_id"] = text_generator.tokenizer.pad_token_id
    else:
        print(f"Using {args.backend} API model: {model_path_or_id}")

    # 5. GENERATION LOOP
    damage_levels = range(args.damage_level_min, args.damage_level_max+1)

    for i, system_prompt in enumerate(prompts_to_run):
        prompt_name = names_to_run[i]
        start_time = time.time()
        
        # Construct filename: output_path_{dataset}_{model}_{prompt_name}.json(l)
        name_parts = [s for s in model_conf["model_id"].split('/') if s]
        if any(part == "snapshots" for part in name_parts):
            snapshot_index = name_parts.index("snapshots")
            safe_name = name_parts[snapshot_index - 1]
        else:
            safe_name = name_parts[-1]
        run_identifier = f"{args.dataset_config}_{safe_name}_{prompt_name}"

        if args.save_as_dataset:
            # Force .jsonl for Dataset mode
            filename_suffix = ".jsonl"
            current_output_path = args.output_file.with_name(f"{args.output_file.stem}_{run_identifier}{filename_suffix}")
        else:
            # JSON Mode
            filename_suffix = args.output_file.suffix if args.output_file.suffix == ".json" else ".json"
            current_output_path = args.output_file.with_name(f"{args.output_file.stem}_{run_identifier}{filename_suffix}")
            
        print(f"\n--- Starting Run: {prompt_name} -> {current_output_path.name} ---")

        config_header = {
            "run_id": run_identifier,
            "model_id": model_path_or_id,
            "generation_params": gen_params,
            "prompt_type": prompt_name,
            "prompt_text": system_prompt,
            "damage_levels": list(damage_levels)
        }

        # Resume Logic
        accumulated_outputs = []
        accumulated_dataset = ProcessedDataset()
        skip_count = 0
        if current_output_path.exists():
            try:
                if args.save_as_dataset:
                    # Dataset Mode Resume
                    accumulated_dataset = ProcessedDataset.load(current_output_path)
                    skip_count = len(accumulated_dataset)
                else:
                    # JSON Mode Resume
                    with open(current_output_path, 'r', encoding='utf-8') as f:
                        data_json = json.load(f)
                        accumulated_outputs = data_json.get("outputs", [])
                        skip_count = len(accumulated_outputs)
                
                print(f"Resuming {current_output_path.name} after {skip_count} samples.")
            except Exception as e:
                print(f"Warning: Corrupt or unreadable file {current_output_path.name} ({e}). Starting fresh.")

        # Data Iterator
        data_iterator = gen_data(dataset_conf, dataset, damage_levels, system_prompt)
        if skip_count > 0:
            data_iterator = itertools.islice(data_iterator, skip_count, None)
        print(f"Generating from sample {skip_count} to {len(dataset)*len(damage_levels)}.")

        last_checkpoint_count = len(accumulated_dataset) if args.save_as_dataset else len(accumulated_outputs)
        batch_counter = 0
        for batch_items in tqdm(batch(data_iterator, args.batch_size), desc="Generating batches", total=((len(dataset)*len(damage_levels)-skip_count)//args.batch_size)+1):
            batch_messages = [b['messages'] for b in batch_items]

            if args.backend == "local":
                batch_results = text_generator(batch_messages, **gen_params, num_return_sequences=1)
                generated_texts = [result[0]['generated_text'][-1]['content'] for result in batch_results]
            elif args.backend == "gemini":
                generated_texts = [
                    call_gemini(messages, model_path_or_id, gen_params["max_new_tokens"], 0.7 if gen_params.get("do_sample") else 0)
                    for messages in batch_messages
                ]
            else:
                generated_texts = [
                    call_openai_compatible(messages, model_path_or_id, gen_params["max_new_tokens"], 0.7 if gen_params.get("do_sample") else 0)
                    for messages in batch_messages
                ]

            for item_input, result in zip(batch_items, generated_texts):
                generated_text = result
                # if '</think>' in generated_text:
                #     generated_text = generated_text.split('</think>')[-1].strip()

                if args.save_as_dataset:
                    # Create Example object for ProcessedDataset
                    new_example = Example(
                        id=f"{item_input['metadata']['index']}_level{item_input['metadata']['damage_level']}",
                        input=item_input['metadata']['input_text'],
                        context=item_input['metadata']['context'],
                        reference=item_input['metadata']['reference'],
                        prediction=generated_text,
                        model_name=model_conf['model_id'],
                        original_human_score=int(item_input['metadata']['damage_level']),
                        processed_human_score=int(item_input['metadata']['damage_level']),
                        extra={
                            "damage_level": item_input['metadata']['damage_level'],
                            "prompt_type": prompt_name,
                            "full_prompt": item_input['messages'],
                            "source_index": item_input['metadata']['index']
                        }
                    )
                    accumulated_dataset.append(new_example)
                else:
                    # Create dictionary for JSON
                    accumulated_outputs.append({
                        **item_input['metadata'],
                        "full_prompt": item_input['messages'],
                        "generated_text": generated_text
                    })

            current_count = len(accumulated_dataset) if args.save_as_dataset else len(accumulated_outputs)
            if current_count - last_checkpoint_count >= args.save_every:
                if args.save_as_dataset:
                    accumulated_dataset.save(current_output_path)
                    # print(f"  [Checkpoint] Saved {len(accumulated_dataset)} samples (Dataset format).")
                    tqdm.write(f"  [Checkpoint] Saved {len(accumulated_dataset)} samples (Dataset format).")
                else:
                    save_checkpoint(current_output_path, config_header, accumulated_outputs)
                last_checkpoint_count = current_count

        if args.save_as_dataset:
            accumulated_dataset.save(current_output_path)
            # print(f"  [Final] Saved {len(accumulated_dataset)} samples (Dataset format).")
            tqdm.write(f"  [Final] Saved {len(accumulated_dataset)} samples (Dataset format).")
        else:
            save_checkpoint(current_output_path, config_header, accumulated_outputs)
        print(f"Finished run for prompt: {prompt_name}. Saved into: {current_output_path.name}")
        elapsed = time.time() - start_time
        print(f"Finished {prompt_name}: {elapsed:.2f}s")
