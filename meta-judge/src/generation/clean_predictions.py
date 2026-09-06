from pathlib import Path
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input folder."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='',
        help="Path to the output folder."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    
    if args.output_path != '':
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        for file_path in input_path.glob("*.jsonl"):
            print(f"Processing file: {input_path / file_path.name}")
            indices = []
            with file_path.open("r", encoding="utf-8") as infile:
                lines = infile.readlines()

            with (output_path / file_path.name).open("w", encoding="utf-8") as outfile:
                for i, line in enumerate(lines):
                    data = json.loads(line)
                    
                    if 'Note:' in data['prediction']:
                        data['prediction'] = data['prediction'].split('Note:')[0].strip()
                    json.dump(data, outfile)
                    outfile.write("\n")
            print(f"Finished processing {output_path / file_path.name}.")
    
    else:
        total_note, total = 0, 0
        for file_path in input_path.rglob("*.jsonl"):
            longer = 0
            indices = []
            with file_path.open("r", encoding="utf-8") as infile:
                lines = infile.readlines()
            for i, line in enumerate(lines):
                data = json.loads(line)
                pred_len = len(data['prediction'].split('\n\n'))
                if pred_len > 1:
                    longer += 1
                    indices.append(i)
            print(f"File: {file_path.name}, Entries with 'Note:': {longer} out of {len(lines)}, Indices[:3]: {indices[:3]}")
            total_note += longer
            total += len(lines)
        print(f"Overall: Entries with 'Note:': {total_note} out of {total} ({(total_note/total)*100:.2f}%)")
