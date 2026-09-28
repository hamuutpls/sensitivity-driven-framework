"""Calibration data loading: WikiText-2, C4 or pile-10k, cut into fixed-length token windows."""

from __future__ import annotations

import random

import torch

from sdf.config import CalibrationConfig

_DATASETS = {
    "wikitext2": dict(path="wikitext", name="wikitext-2-raw-v1", split="train"),
    "c4": dict(
        path="allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    ),
    "pile10k": dict(path="NeelNanda/pile-10k", split="train"),
}


def load_texts(dataset: str) -> list[str]:
    """Load raw text documents for a calibration dataset."""
    if dataset not in _DATASETS:
        raise ValueError(f"unknown calibration dataset {dataset!r}; choose from {sorted(_DATASETS)}")
    from datasets import load_dataset

    ds = load_dataset(**_DATASETS[dataset])
    return [t for t in ds["text"] if t.strip()]


def make_batches(texts: list[str], tokenizer, cfg: CalibrationConfig) -> list[torch.Tensor]:
    """Tokenise `texts` and sample `n_batches` batches of random `seq_len` windows.

    WikiText-2 style: the documents are joined into one token stream and windows are sampled from it.
    """
    rng = random.Random(cfg.seed)
    n_windows = cfg.n_batches * cfg.batch_size
    needed = n_windows * cfg.seq_len * 4  # stop tokenising once the stream is 4x the tokens we sample

    stream: list[int] = []
    docs = list(texts)
    rng.shuffle(docs)
    for doc in docs:
        stream.extend(tokenizer(doc, add_special_tokens=False)["input_ids"])
        if len(stream) >= needed:
            break
    if len(stream) <= cfg.seq_len:
        raise ValueError(f"calibration text too short: {len(stream)} tokens for seq_len={cfg.seq_len}")

    ids = torch.tensor(stream, dtype=torch.long)
    starts = [rng.randint(0, len(ids) - cfg.seq_len - 1) for _ in range(n_windows)]
    windows = torch.stack([ids[s : s + cfg.seq_len] for s in starts])
    return list(windows.split(cfg.batch_size))


def load_calibration_batches(tokenizer, cfg: CalibrationConfig) -> list[torch.Tensor]:
    return make_batches(load_texts(cfg.dataset), tokenizer, cfg)
