"""Datasets: calibration windows (WikiText-2 / C4 / pile-10k) and the WikiText-2 test split for evaluation."""

from __future__ import annotations

import random
from typing import Any, Callable, Mapping

import torch

from sdf.utils.logging import get_logger

log = get_logger(__name__)

TextLoader = Callable[[str, str], list[str]]


def make_text_loader(sources: Mapping[str, Mapping[str, Any]]) -> TextLoader:
    """A loader for the Hub datasets described in the config (`data.sources`).

    Each source gives `load_dataset` kwargs (path, name, data_files...) plus `splits`, mapping our split names
    ("train" = calibration pool, "test" = evaluation) to the dataset's own split names.
    """

    def load_texts(dataset: str, split: str = "train") -> list[str]:
        if dataset not in sources:
            raise ValueError(f"unknown dataset {dataset!r}; configured: {sorted(sources)}")
        spec = dict(sources[dataset])
        splits = spec.pop("splits", {})
        if split not in splits:
            raise ValueError(f"dataset {dataset!r} has no {split!r} split configured (data.sources.{dataset}.splits)")
        from datasets import load_dataset

        log.info("loading %s/%s from %s", dataset, split, spec.get("path"))
        ds = load_dataset(**spec, split=splits[split])
        return list(ds["text"])

    return load_texts


def calibration_batches(
    texts: list[str], tokenizer, n_samples: int, seq_len: int, batch_size: int, seed: int
) -> list[torch.Tensor]:
    """`n_samples` random windows of `seq_len` tokens, grouped into batches of `batch_size`.

    Documents are shuffled with `seed`, joined into one token stream (tokenising stops once the stream holds
    4x the tokens we sample), and window starts are drawn from it with the same seed.
    """
    rng = random.Random(seed)
    docs = [t for t in texts if t.strip()]
    rng.shuffle(docs)
    needed = n_samples * seq_len * 4
    stream: list[int] = []
    for doc in docs:
        stream.extend(tokenizer(doc, add_special_tokens=False)["input_ids"])
        if len(stream) >= needed:
            break
    if len(stream) <= seq_len:
        raise ValueError(f"calibration text too short: {len(stream)} tokens for seq_len={seq_len}")
    ids = torch.tensor(stream, dtype=torch.long)
    starts = [rng.randint(0, len(ids) - seq_len - 1) for _ in range(n_samples)]
    windows = torch.stack([ids[s : s + seq_len] for s in starts])
    return list(windows.split(batch_size))


def eval_windows(
    texts: list[str], tokenizer, seq_len: int, max_windows: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the test text into non-overlapping `seq_len` windows and return (validation, held_out).

    Standard WikiText-2 perplexity protocol: documents joined with blank lines, tokenised once. The first half
    of the windows is the validation half (used by the search); the second half is held out (reported only).
    The split is contiguous and deterministic, so both halves are identical across every row of a comparison.
    """
    ids = tokenizer("\n\n".join(texts), return_tensors="pt")["input_ids"][0]
    n = ids.numel() // seq_len
    if max_windows is not None:
        n = min(n, max_windows)
    if n < 2:
        raise ValueError(f"evaluation text gives {n} windows of {seq_len} tokens; need at least 2")
    windows = ids[: n * seq_len].view(n, seq_len)
    half = n // 2
    log.info("eval windows: %d x %d tokens (validation %d, held-out %d)", n, seq_len, half, n - half)
    return windows[:half], windows[half:]
