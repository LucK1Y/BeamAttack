import argparse
import gc
import os
import pathlib
import random

import numpy as np
import torch
from datasets import Dataset

from utils.data_mappings import dataset_mapping, dataset_mapping_pairs
from victims.bert import readfromfile_generator
from transformers import AutoTokenizer

def get_available_device() -> torch.device:
    # Load the models
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def set_random_seeds():
    global dataset_shuffle_seed
    # determinism
    random.seed(10)
    torch.manual_seed(10)
    np.random.seed(0)
    dataset_shuffle_seed = 42


def trim(text, tokenizer):
    MAX_LEN = 512
    offsets = tokenizer(
        text, truncation=True, max_length=MAX_LEN + 10, return_offsets_mapping=True
    )["offset_mapping"]
    limit = len(text)
    if len(offsets) > MAX_LEN:
        limit = offsets[512][1]
    return text[:limit]


def roberta_readfromfile_generator(subset, dir, with_pairs=False, trim_text=False):
    pretrained_model = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    for line in open(dir / (subset + ".tsv")):
        parts = line.split("\t")
        label = int(parts[0])
        if not with_pairs:
            text = (
                parts[2]
                .strip()
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\\", "\\")
            )
            if trim_text:
                text = trim(text, tokenizer)
            yield {"fake": label, "text": text}
        else:
            text1 = (
                parts[2]
                .strip()
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\\", "\\")
            )
            text2 = (
                parts[3]
                .strip()
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\\", "\\")
            )
            if trim_text:
                text1 = trim(text1, tokenizer)
                text2 = trim(text2, tokenizer)
            yield {"fake": label, "text1": text1, "text2": text2}


def get_incredible_dataset(
    task: str,
    data_path: pathlib.Path,
    victim_model_generator="bert-style",
    subset="attack",
    first_n_samples: int = None,
    randomised=False,
):
    if victim_model_generator == "bert-style":
        generator = readfromfile_generator
    elif victim_model_generator == "surprise":
        generator = roberta_readfromfile_generator

    with_pairs = task == "FC" or task == "C19"
    test_dataset = Dataset.from_generator(
        generator,
        gen_kwargs={
            "subset": subset,
            "dir": data_path,
            "trim_text": True,
            "with_pairs": with_pairs,
        },
    )

    if not with_pairs:
        dataset = test_dataset.map(dataset_mapping)
        dataset = dataset.remove_columns(["text"])
    else:
        dataset = test_dataset.map(dataset_mapping_pairs)
        dataset = dataset.remove_columns(["text1", "text2"])

    dataset = dataset.remove_columns(["fake"])

    # Filter data
    if first_n_samples:
        if randomised:
            dataset = dataset.shuffle(seed=dataset_shuffle_seed).select(
                range(first_n_samples)
            )
        else:
            dataset = dataset.select(range(first_n_samples))

    return dataset, with_pairs


def free_up_model_space():
    torch.cuda.empty_cache()
    gc.collect()
    if "TOKENIZERS_PARALLELISM" in os.environ:
        del os.environ["TOKENIZERS_PARALLELISM"]

    # We need to create a random exception.
    # See here: https://stackoverflow.com/questions/57858433/how-to-clear-gpu-memory-after-pytorch-model-training-without-restarting-kernel
    try:
        1 / 0
    except ZeroDivisionError:
        pass
    torch.cuda.empty_cache()
    gc.collect()

