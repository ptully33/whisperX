import argparse
import json
import uuid
from pathlib import Path
from typing import List, Dict

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt', quiet=True)


def chunk_transcript(text: str, max_tokens: int = 500) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    tokens = 0
    for sent in sentences:
        words = word_tokenize(sent)
        if tokens + len(words) > max_tokens and current:
            chunks.append(' '.join(current))
            current = []
            tokens = 0
        current.append(sent)
        tokens += len(words)
    if current:
        chunks.append(' '.join(current))
    return chunks


def make_header(text: str, source: str) -> Dict[str, str]:
    words = word_tokenize(text)
    title = ' '.join(words[:8])
    summary = ' '.join(words[:50])
    return {
        'id': str(uuid.uuid4()),
        'title': title,
        'summary': summary,
        'source': source,
    }


def process_transcript(path: Path, out_dir: Path, manifest: List[Dict]):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = chunk_transcript(text)
    for chunk in chunks:
        header = make_header(chunk, path.name)
        chunk_file = out_dir / f"{header['id']}.md"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write('---\n')
            for k, v in header.items():
                f.write(f"{k}: {v}\n")
            f.write('---\n\n')
            f.write(chunk)
        manifest.append({**header, 'file': chunk_file.name})


def build_rag_chunks(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for file in sorted(in_dir.glob("*_full_transcript.txt")):
        process_transcript(file, out_dir, manifest)
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def cli(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare RAG chunks")
    parser.add_argument("input", type=Path, help="Folder with transcripts")
    parser.add_argument("output", type=Path, help="Output folder")
    args = parser.parse_args(argv)

    build_rag_chunks(args.input, args.output)


if __name__ == "__main__":
    cli()
