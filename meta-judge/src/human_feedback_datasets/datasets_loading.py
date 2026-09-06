from itertools import product
from human_feedback_datasets.processed_dataset import ProcessedDataset, Example
from datasets import load_dataset # type: ignore
from typing import List, Any, Dict, Literal, Optional, Tuple, cast
import gzip
import json
from pathlib import Path
from tqdm import tqdm

PREPARED_DATASETS = ['rose', 'mocha', 'qa', 'wmt25', 'wmt25_parallel']

HF_MAPPING = {
    'rose': 'Salesforce/rose',
    'mocha': 'anthonychen/mocha',
    'qa': 'ufal/cus-qa',
    'wmt25': 'DuarteMRAlves/wmt25'
}

_rose_splits = ['cnndm_test', 'cnndm_validation', 'cnndm_protocol', 'cnndm_protocol_gpt3', 'xsum']
def load_rose(split: str = 'test') -> ProcessedDataset:
    assert split in _rose_splits, f"Invalid ROSE split: {split}. Available splits: {_rose_splits}"
    raw = load_dataset("Salesforce/rose", split)['data']
    data = ProcessedDataset()
    for item in raw:
        for model in item['system_outputs'].keys():
            annotations = item['annotations'][model]
            hf = annotations['acu']
            data.append(
                Example(
                    id=item.get('count_id', None),
                    input=item['source'],
                    context=None,
                    reference=item['reference'],
                    prediction=item['system_outputs'][model],
                    model_name=model,
                    original_human_score=annotations,
                    processed_human_score=hf,
                    extra=None
                )
            )
    return data

_mocha_splits = ['train', 'validation', 'test', 'minimal_pairs']
def load_mocha(split: str = 'test') -> ProcessedDataset:
    assert split in _mocha_splits, f"Invalid Mocha split: {split}. Available splits: {_mocha_splits}"
    raw = load_dataset("anthonychen/mocha")[split]
    data = ProcessedDataset()
    for item in raw:
        data.append(
            Example(
                id=item.get('id', None),
                input=item['question'],
                context=item['context'],
                reference=item['reference'],
                prediction=item['candidate'],
                model_name=item.get('model_name', None),
                original_human_score=item['score'],
                processed_human_score=item['score'],
                extra=item['metadata']
            )
        )
    return data

_qa_configs = ['text-CZ', 'text-SK', 'text-UA']
_qa_splits = ['dev', 'test']
_qa_lang = ['orig', 'en']
def load_qa(config: str, split: str = 'test', lang: Literal['orig', 'en'] = 'orig', binary: bool = False) -> ProcessedDataset:
    assert config in _qa_configs, f"Invalid QA config: {config}. Available configs: {_qa_configs}"
    assert split in _qa_splits, f"Invalid QA split: {split}. Available splits: {_qa_splits}"
    assert lang in _qa_lang, f"Invalid QA lang: {lang}. Available langs: {_qa_lang}"

    raw = load_dataset("ufal/cus-qa", config)[split]
    quest_str = f'question_{lang}'
    answer_str = f'answer_{lang}'
    generated_str = f'generated_{lang}'
    hf_str = f'human_eval_{lang}'
    data = ProcessedDataset()
    for item in raw:
        raw_generated: Dict[str, Any] = json.loads(item[generated_str])
        raw_hf: Dict[str, Any] = json.loads(item[hf_str])
        for model in raw_generated.keys():
            binary_score = 1 if raw_hf[model]["answer_score"] == 4 else 0
            data.append(
                Example(
                    id=item['id'],
                    input=item[quest_str],
                    context=None,
                    reference=item[answer_str],
                    prediction=raw_generated[model],
                    model_name=model,
                    original_human_score=raw_hf[model],
                    processed_human_score=raw_hf[model]['answer_score'] if not binary else binary_score,
                    extra={'category': item['category'], 'wikititle': item['wikititle']}
                )
            )
    return data

_wmt25_configs = ['cs-de_DE', 'cs-uk_UA', 'en-ar_EG', 'en-bho_IN', 'en-cs_CZ', 'en-et_EE', 'en-is_IS', 'en-it_IT', 'en-ja_JP', 'en-mas_KE', 'en-ru_RU', 'en-sr_Cyrl_RS', 'en-uk_UA', 'en-zh_CN']
_wmt25_default_path = Path('data/wmt25-genmt-humeval.jsonl')

def load_wmt25(configs: List[str] = _wmt25_configs, data_path: Path = _wmt25_default_path, split_on_newline: bool = False) -> Dict[str, ProcessedDataset]:
    invalid = [c for c in configs if c not in _wmt25_configs]
    assert not invalid, f"Invalid WMT25 config(s): {invalid}. Available: {_wmt25_configs}"

    data: Dict[str, ProcessedDataset] = {config: ProcessedDataset() for config in configs}

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            lp = record['doc_id'].split("_#_")[0]

            # Match against requested configs (lp already has the full form, e.g. 'en-is_IS')
            if lp not in data:
                continue

            source = record["src_text"]
            reference = record["tgt_text"].get("refA")
            doc_id = record["doc_id"]

            for system_name, annotations in record["scores"].items():
                translated_text = record["tgt_text"].get(system_name)
                if translated_text is None:
                    continue
                scores = [annotation["score"] for annotation in annotations]
                human_score = sum(scores) / len(scores)

                if not split_on_newline:
                    data[lp].append(
                        Example(
                            id=f"{doc_id}_{system_name}",
                            input=source,
                            context=None,
                            reference=reference,
                            prediction=translated_text,
                            model_name=system_name,
                            original_human_score=scores,
                            processed_human_score=human_score,
                            extra={
                                'lang_pair': lp,
                                'doc_id': doc_id,
                                'system': system_name,
                                'annotators': [a.get('annotator') for a in annotations],
                                'error_spans': [a.get('errors') for a in annotations],
                            }
                        )
                    )
                else:
                    src_segments = [s for s in source.split("\n") if s.strip()]
                    ref_segments = [s for s in reference.split("\n") if s.strip()] if reference else []
                    pred_segments = [s for s in translated_text.split("\n") if s.strip()]

                    # Fall back to unsplit if counts differ or any side has only one segment
                    use_split = (
                        len(src_segments) == len(pred_segments)
                        and (reference is None or len(src_segments) == len(ref_segments))
                        and len(src_segments) > 1
                    )

                    if not use_split:
                        data[lp].append(
                            Example(
                                id=f"{doc_id}_{system_name}",
                                input=source,
                                context=None,
                                reference=reference,
                                prediction=translated_text,
                                model_name=system_name,
                                original_human_score=scores,
                                processed_human_score=human_score,
                                extra={
                                    'lang_pair': lp,
                                    'doc_id': doc_id,
                                    'system': system_name,
                                    'annotators': [a.get('annotator') for a in annotations],
                                    'error_spans': [a.get('errors') for a in annotations],
                                }
                            )
                        )
                    else:
                        for seg_idx, (src_seg, pred_seg) in enumerate(zip(src_segments, pred_segments)):
                            ref_seg = ref_segments[seg_idx] if reference is not None else None
                            data[lp].append(
                                Example(
                                    id=f"{doc_id}_{system_name}_{seg_idx}",
                                    input=src_seg,
                                    context=None,
                                    reference=ref_seg,
                                    prediction=pred_seg,
                                    model_name=system_name,
                                    original_human_score=scores,
                                    processed_human_score=human_score,
                                    extra={
                                        'lang_pair': lp,
                                        'doc_id': doc_id,
                                        'system': system_name,
                                        'seg_idx': seg_idx,
                                        'annotators': [a.get('annotator') for a in annotations],
                                        'error_spans': [a.get('errors') for a in annotations],
                                    }
                                )
                            )

    return data


_wmt25_parallel_configs: Dict[str, Path] = {
    'eng-ces': Path('data/wmt25_data_eng-ces'),
    'ces-ukr': Path('data/wmt25_data_ces-ukr'),
    'eng-isl': Path('data/wmt25_data_eng-isl'),
}

def load_wmt25_parallel(
    lang_pair: str,
    max_examples: Optional[int] = None,
    base_path: Optional[Path] = None,
) -> ProcessedDataset:
    """Load WMT25 parallel bitext (src/tgt pairs) as Examples with input=src, reference=tgt.

    Intended for GRPO: generate candidates from `input`, score them against `reference`.
    No human scores are available — prediction/processed_human_score are left as None.
    """
    assert lang_pair in _wmt25_parallel_configs, (
        f"Invalid WMT25 parallel lang_pair: {lang_pair}. "
        f"Available: {list(_wmt25_parallel_configs)}"
    )
    dir_path = base_path if base_path is not None else _wmt25_parallel_configs[lang_pair]
    src_lang, tgt_lang = lang_pair.split('-')
    src_file = dir_path / f"train.{src_lang}"
    tgt_file = dir_path / f"train.{tgt_lang}"
    meta_file = dir_path / "train.meta.jsonl.gz"

    data = ProcessedDataset()
    with open(src_file, 'r', encoding='utf-8') as sf, \
         open(tgt_file, 'r', encoding='utf-8') as tf, \
         gzip.open(meta_file, 'rt', encoding='utf-8') as mf:
        for i, (src, tgt, meta) in tqdm(enumerate(zip(sf, tf, mf)), desc=f"Loading {lang_pair}"):
            if max_examples is not None and i >= max_examples:
                break
            data.append(
                Example(
                    id=f"{lang_pair}_{i}",
                    input=src.rstrip('\n'),
                    context=None,
                    reference=tgt.rstrip('\n'),
                    prediction=None,  # type: ignore[arg-type]
                    model_name=None,
                    original_human_score=None,
                    processed_human_score=None,
                    extra={
                        'lang_pair': lang_pair,
                        'corpus': meta.strip(),
                        'line_idx': i,
                    },
                )
            )
    return data


def load_wmt_hf(lps: List[str] = ['de-en', 'en-de', 'cs-en', 'en-cs', 'de-cs', 'gu-en', 'kk-en', 'km-en'], min_year: int = 2017, max_year: int = 2022) -> Dict[str, ProcessedDataset]:
    # ['lp', 'src', 'mt', 'ref', 'score', 'raw', 'annotators', 'domain', 'year']
    assert 2017 <= min_year <= max_year <= 2022, "Years must be between 2017 and 2022"

    raw = load_dataset('RicardoRei/wmt-da-human-evaluation')['train']
    d: Dict[str, ProcessedDataset] = {}

    for i, item in enumerate(raw):
        if item['year'] < min_year or item['year'] > max_year or (lps and item['lp'] not in lps): continue
        if item['ref'] is None or item['src'] is None or item['mt'] is None:
            print(f"Skipping incomplete example {i} in WMT HF dataset: ref={item['ref']}, src={item['src']}, mt={item['mt']}")
            continue
        lp = item['lp']
        if lp not in d:
            d[lp] = ProcessedDataset()
        d[lp].append(
            Example(
                id=str(i),
                input=item['src'],
                context=None,
                reference=item['ref'],
                prediction=item['mt'],
                model_name=None,
                original_human_score=item['raw'],
                processed_human_score=item['raw'],
                extra={"domain": item['domain'], 'annotators': item['annotators'], 'lp': item['lp'], 'year': item['year']}
            )
        )
    return d

def load_wmt_jsonl(path: Path = Path('wmt24_esa.jsonl'), lps: List[str] = ['cs', 'en-kk']) -> Dict[str, ProcessedDataset]:
    # ['langs', 'line_id', 'src', 'tgt', 'doc_id', 'domain', 'esa_spans', 'esa_score', 'system', 'annotator', 'speech_info']
    raw = [json.loads(line) for line in open(path)]
    keys = ['langs', 'line_id', 'doc_id']
    grouped_items: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for item in raw:
        key = tuple(item[k] for k in keys)
        if key not in grouped_items:
            grouped_items[key] = []
        grouped_items[key].append(item)
    print(f"Loaded {len(raw)} examples from {path}, unique keys: {len(grouped_items)}")

    no_refs_count = 0
    skip_keys: List[Tuple[str, str, str]] = []
    for key, items in grouped_items.items():
        if len(items) > 1:
            refs = [item for item in items if 'ref' in item['system']]
            if len(refs) == 0:
                no_refs_count += 1
                print(f"Warning: Key {key} has no references, preds {len(items)}, pred systems {[item['system'] for item in items]}.")
                skip_keys.append(key)
            assert len(items) - len(refs) > 0, f"Warning: Key {key} has no predictions."
        else:
            no_refs_count += 1
            print(f"Warning: Key {key} has only one item, no references.")
            skip_keys.append(key)
    print(f"Total keys with no references: {no_refs_count} out of {len(grouped_items)}")


    def same_vals(list: List[Dict[str, Any]], key: str) -> bool:
        vals = set(item[key] for item in list)
        return len(vals) == 1

    for key, items in grouped_items.items():
        for ck in ['src', 'langs', 'line_id', 'doc_id']:
            assert same_vals(items, ck), f"Key {key} has differing values for {ck}: {[item[ck] for item in items]}"

    data: Dict[str, ProcessedDataset] = {}
    for key, items in grouped_items.items():
        if key in skip_keys:
            print(f"Skipping key {key} with no references.")
            continue
        refs = [item for item in items if 'ref' in item['system']]
        best_ref = max(refs, key=lambda x: x['esa_score'])
        preds = [item for item in items if 'ref' not in item['system']]
        langs = key[0]

        for pred in preds:
            if langs not in data:
                data[langs] = ProcessedDataset()
            data[langs].append(
                Example(
                    id=f"{key[1]}_{key[2]}_{pred['system']}",
                    input=best_ref['src'],
                    context=None,
                    reference=best_ref['tgt'],
                    prediction=pred['tgt'],
                    model_name=pred['system'],
                    original_human_score=float(pred['esa_score']),
                    processed_human_score=float(pred['esa_score']),
                    extra={'domain': pred['domain'], 'annotator': pred['annotator'], 'langs': pred['langs'], 'esa_spans': pred['esa_spans'], 'speech_info': pred['speech_info'], 'line_id': pred['line_id'], 'doc_id': pred['doc_id'], 'ref': best_ref}
                )
            )

    return data


def dict_rec(d: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_rec(v)
        elif isinstance(v, list):
            d[k] = list_rec(v)
        elif isinstance(v, str):
            try:
                temp = json.loads(v)
                d[k] = dict_rec(temp)
            except:
                pass
    return d

def list_rec(l: List[Any]) -> List[Any]:
    for i, v in enumerate(l):
        if isinstance(v, dict):
            l[i] = dict_rec(v)
        elif isinstance(v, list):
            l[i] = list_rec(v)
        elif isinstance(v, str):
            try:
                temp = json.loads(v)
                if isinstance(temp, dict):
                    l[i] = dict_rec(temp)
                elif isinstance(temp, list):
                    l[i] = list_rec(temp)
            except:
                pass
    return l

def save_into_file(data: Any, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict_rec(data), open(filename, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # for qa_config, qa_split, qa_lang in product(_qa_configs, ['test'], _qa_lang):
    #     print(f"Loading QA config: {qa_config}, split: {qa_split}, lang: {qa_lang}")
    #     qa_dataset = load_qa(qa_config, qa_split, lang=cast(Literal['orig', 'en'], qa_lang), binary=True)
    #     qa_dataset.save(Path(f'datasets/qa_{qa_config}_{qa_split}_{qa_lang}_binary.jsonl'))

    configs = ['en-is_IS', 'en-cs_CZ', 'cs-uk_UA']
    split = True
    wmt_datasets = load_wmt25(configs, split_on_newline=split)
    suffix = '_split' if split else ''
    for config, dataset in wmt_datasets.items():
        dataset.save(Path(f'datasets/wmt_2025/wmt25_{config}{suffix}.jsonl'))

    # for lp in _wmt25_parallel_configs:
    #     ds = load_wmt25_parallel(lp, max_examples=500_000)
    #     ds.save(Path(f'datasets/wmt_2025/wmt25_parallel_{lp}.jsonl'))
