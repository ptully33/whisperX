import argparse
import json
import os
from pathlib import Path
from typing import List

import nltk
import whisperx

nltk.download('punkt', quiet=True)

CONFIDENCE_THRESHOLD = 0.85


def flag_low_confidence(segment: dict) -> bool:
    """Return True if any word score is below threshold."""
    for w in segment.get("words", []):
        if w.get("score", 1.0) < CONFIDENCE_THRESHOLD:
            return True
    return False


def process_file(path: Path, out_dir: Path, model, device: str = "cuda") -> None:
    """Transcribe a single file and write outputs."""
    audio = whisperx.load_audio(str(path))
    result = model.transcribe(audio, batch_size=16)
    align_model, metadata = whisperx.load_align_model(result["language"], device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)

    base = out_dir / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # write full transcript
    with open(base.with_suffix("_full_transcript.txt"), "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            f.write(seg["text"].strip() + "\n")

    # write flagged sentences
    with open(base.with_suffix("_flagged_sentences.txt"), "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            if flag_low_confidence(seg):
                f.write(seg["text"].strip() + "\n")

    # save word-level timestamps JSON
    with open(base.with_suffix("_timestamps.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # also save srt subtitles
    writer = whisperx.utils.get_writer("srt", str(out_dir))
    writer(result, str(path), {})



def transcribe_folder(in_dir: Path, out_dir: Path, model_name: str = "large-v2", device: str = "cuda"):
    model = whisperx.load_model(model_name, device, compute_type="float16")
    for file in sorted(in_dir.iterdir()):
        if file.suffix.lower() not in {".mp4", ".wav"}:
            continue
        print(f"Transcribing {file.name}...")
        process_file(file, out_dir, model, device)
    del model


def cli(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Batch transcribe with WhisperX")
    parser.add_argument("input", type=Path, help="Folder of audio/video files")
    parser.add_argument("output", type=Path, help="Output folder")
    parser.add_argument("--model", default="large-v2", help="Whisper model name")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args(argv)

    transcribe_folder(args.input, args.output, args.model, args.device)


if __name__ == "__main__":
    cli()
