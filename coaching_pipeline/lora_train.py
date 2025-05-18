import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer


def train(train_path: Path, config: Dict, out_dir: Path):
    dataset = load_dataset('json', data_files=str(train_path))['train']
    model_name = config.get('base_model', 'microsoft/phi-2')
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=config.get('r', 8),
        lora_alpha=config.get('alpha', 16),
        lora_dropout=config.get('dropout', 0.1),
        target_modules=config.get('target_modules', ['q_proj', 'v_proj'])
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field='text',
        max_seq_length=tokenizer.model_max_length,
        args=config.get('trainer_args', {})
    )
    trainer.train()
    trainer.model.save_pretrained(out_dir)


def cli(argv=None):
    parser = argparse.ArgumentParser(description='Train LoRA model')
    parser.add_argument('train_file', type=Path, help='train.jsonl file')
    parser.add_argument('config', type=Path, help='json config')
    parser.add_argument('output', type=Path, help='output directory')
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = json.load(f)
    train(args.train_file, cfg, args.output)


if __name__ == '__main__':
    cli()
