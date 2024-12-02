import shutil
from pathlib import Path

import torch
from torch import nn


def save_model(model: nn.Module, save_dir: Path, safetensors: bool = True):
    if safetensors:
        # Some `generation_config` set the temperature > 0, but also set `do_sample` to False.
        # This is contradictory according to HuggingFace's standards.
        # As a workaround, we override `do_sample` and set it to `True`.
        model.generation_config.do_sample = True
        model.save_pretrained(save_dir)

    else:
        torch.save(model.state_dict(), save_dir / "model.pt")


def copy_hf_config_files(model_path, save_dir):
    for cfg_file in [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "added_tokens.json",
        # qwen
        "tokenization_qwen.py",
        "qwen.tiktoken",
        # telechat
        "tokenizer.json",
        # qwen2
        "vocab.json",
        "merges.txt",
        # baichuan
        "tokenization_baichuan.py",
        # chatglm3
        "tokenization_chatglm.py",
        # deepseek-v2
        "tokenization_deepseek_fast.py",
    ]:
        src_file_path = Path(model_path) / cfg_file
        if src_file_path.exists():
            shutil.copy(src_file_path, save_dir / cfg_file)
