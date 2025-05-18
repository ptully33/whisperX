import argparse
import json
from pathlib import Path
from typing import List, Dict


def build_samples(json_path: Path) -> List[Dict[str, str]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data.get('segments', [])
    samples = []
    prev_text = None
    prev_speaker = None
    for seg in segments:
        speaker = seg.get('speaker')
        text = seg.get('text', '').strip()
        if speaker and speaker.lower().startswith('peter'):
            if prev_text and prev_speaker and prev_speaker != speaker:
                samples.append({
                    'instruction': f'{prev_speaker}: {prev_text}',
                    'output': f'{speaker}: {text}'
                })
        prev_text = text
        prev_speaker = speaker
    return samples


def process_folder(in_dir: Path, out_file: Path, limit: int = 50000):
    all_samples = []
    for file in sorted(in_dir.glob('*_timestamps.json')):
        all_samples.extend(build_samples(file))
        if len(all_samples) >= limit:
            break
    with open(out_file, 'w', encoding='utf-8') as f:
        for s in all_samples[:limit]:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')


def cli(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description='Prepare LoRA training data')
    parser.add_argument('input', type=Path, help='Folder with timestamp jsons')
    parser.add_argument('output', type=Path, help='Output train.jsonl path')
    parser.add_argument('--limit', type=int, default=50000, help='Max samples')
    args = parser.parse_args(argv)

    process_folder(args.input, args.output, args.limit)


if __name__ == '__main__':
    cli()
