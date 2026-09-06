from pathlib import Path
import json
from collections import defaultdict
from pprint import pprint

folder1 = Path('/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets_reasoning')
folder2 = Path('/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets_reasoning_sample')
non2splits = defaultdict(float)
whatnon2splits = defaultdict(list)
reasoning_after_text = defaultdict(float)


def round_dict(d):
    return {k: round(v * 100, 2) for k, v in d.items()}

def process_folder(folder: Path):
    for file in folder.rglob('*.jsonl'):
        with open(file, 'r') as f:
            data = [json.loads(line) for line in f]
        
        for ex in data:
            pred: str = ex['prediction']
            splits = pred.split('### TEXT')
            if len(splits) != 2:
                non2splits[file.stem] += 1
                whatnon2splits[file.stem].append(ex['id'])
            else:
                s = splits[1]
                if '### REASONING' in s:
                    reasoning_after_text[file.stem] += 1
        if file.stem in non2splits:
            non2splits[file.stem] /= len(data)
        if file.stem in reasoning_after_text:
            reasoning_after_text[file.stem] /= len(data)


    print("Files with non-2 splits:")
    pprint(round_dict(non2splits))
    print("Files with reasoning after text:")
    pprint(round_dict(reasoning_after_text))
# print("Files with non-2 splits details:", dict(whatnon2splits))

process_folder(folder1)
# process_folder(folder2)