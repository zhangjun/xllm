import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer


def get_local_dir(
    tokenizer,
    seed: int,
    n_samples: int,
    seq_len: int,
    data_files: str,
    preprocess_script: Optional[str] = None,
):
    if preprocess_script is None:
        traindata = load_dataset("json", data_files=data_files, split="train")
        rng = random.Random(seed)

        trainloader = []
        for _ in range(n_samples):
            while True:
                i = rng.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] > seq_len:
                    break

            i = rng.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
            inp = trainenc.input_ids[:, i : i + seq_len]
            trainloader.append(inp)
    else:
        spec = spec_from_file_location("data_process_module", preprocess_script)
        data_process_module = module_from_spec(spec)
        spec.loader.exec_module(data_process_module)

        trainloader = data_process_module.load_dataset_script(
            data_files, tokenizer, n_samples, seq_len
        )

    return trainloader


def get_wikitext2(tokenizer, seed: int, n_samples: int, seq_len: int, download_mode: str):
    traindata = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="train", download_mode=download_mode
    )

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    rng = random.Random(seed)
    trainloader = (
        rng.randint(0, trainenc.input_ids.shape[1] - seq_len - 1) for _ in range(n_samples)
    )
    trainloader = [trainenc.input_ids[:, i : i + seq_len] for i in trainloader]

    return trainloader


def get_ptb(tokenizer, seed: int, n_samples: int, seq_len: int, download_mode: str):
    traindata = load_dataset(
        "ptb_text_only", "penn_treebank", split="train", download_mode=download_mode
    )

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")

    rng = random.Random(seed)
    trainloader = (
        rng.randint(0, trainenc.input_ids.shape[1] - seq_len - 1) for _ in range(n_samples)
    )
    trainloader = [trainenc.input_ids[:, i : i + seq_len] for i in trainloader]

    return trainloader


def get_c4(tokenizer, seed: int, n_samples: int, seq_len: int, download_mode: str):
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        download_mode=download_mode,
    )

    rng = random.Random(seed)

    trainloader = []
    for _ in range(n_samples):
        while True:
            i = rng.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seq_len:
                break

        i = rng.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        inp = trainenc.input_ids[:, i : i + seq_len]
        trainloader.append(inp)

    return trainloader


class CalibDataLoader:
    def __init__(self) -> None:
        self.dataset_loader = {
            "wikitext": get_wikitext2,
            "ptb": get_ptb,
            "c4": get_c4,
        }

    def register_dataset(self, name, func):
        self.dataset_loader[name] = func

    def get_dataset(
        self,
        dataset_name_or_path: str,
        tokenizer: AutoTokenizer,
        seed: int,
        n_samples: int,
        seq_len: int,
        download_mode: str = "reuse_dataset_if_exists",
        preprocess_script: Optional[str] = None,
    ):
        if dataset_name_or_path in self.dataset_loader:
            return self.dataset_loader[dataset_name_or_path](
                tokenizer, seed, n_samples, seq_len, download_mode
            )
        elif Path(dataset_name_or_path).exists():
            return get_local_dir(
                tokenizer, seed, n_samples, seq_len, dataset_name_or_path, preprocess_script
            )
        else:
            raise ValueError(f"Unknown dataset {dataset_name_or_path}")
