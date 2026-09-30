"""The single configuration object. Stage logic reads every setting from here or from the candidate dict."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from sdf.requirements import DeploymentRequirement


@dataclass
class RunConfig:
    output_root: str = "thesis_compression/results"  # local folder, relative to the working directory
    run_id: str | None = None  # None -> timestamp + config hash
    cache_dir: str | None = None  # None -> <output_root>/../cache
    seed: int = 0
    deterministic: bool = True  # torch.use_deterministic_algorithms (warn-only)
    log_level: str = "INFO"


@dataclass
class ModelConfig:
    name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype: str = "float16"  # dtype of the FP16 baseline and of evaluation
    device: str = "auto"  # auto | cpu | cuda | cuda:N


def _default_sources() -> dict[str, dict[str, Any]]:
    return {
        "wikitext2": {"path": "Salesforce/wikitext", "name": "wikitext-2-raw-v1",
                      "splits": {"train": "train", "test": "test"}},
        "c4": {"path": "allenai/c4", "data_files": {"validation": "en/c4-validation.00000-of-00008.json.gz"},
               "splits": {"train": "validation"}},
        "pile10k": {"path": "NeelNanda/pile-10k", "splits": {"train": "train"}},
    }


@dataclass
class DataConfig:
    # Hugging Face Hub datasets: name -> load_dataset kwargs + "splits" ("train" = calibration pool, "test" = eval).
    sources: dict[str, dict[str, Any]] = field(default_factory=_default_sources)


@dataclass
class CalibrationConfig:
    # dataset and sample count are search-space parameters (calib_dataset, calib_samples)
    seq_len: int = 512
    batch_size: int = 1


@dataclass
class EvalConfig:
    dataset: str = "wikitext2"  # wikitext-2 raw *test* split
    seq_len: int = 512
    max_windows: int | None = None  # cap on non-overlapping windows (None = whole test split)
    # First half of the windows = validation (the search optimises it); second half = held-out (reported only).
    latency_prompt_len: int = 128
    latency_decode_tokens: int = 64
    latency_warmup: int = 3
    latency_repeats: int = 10


@dataclass
class Stage0Config:
    # rank | minmax. Rank is robust to outlier layers; see sdf.stage0.sensitivity.normalize.
    normalization: str = "rank"
    profile_dtype: str = "float32"  # fp16 gradients overflow; profile in fp32 (or bfloat16 on GPU)
    # How layer sensitivity is measured: grad_x_weight | layer_removal | layer_quant (see stage0/sensitivity.py)
    score: str = "layer_removal"
    protected_bits: int = 8
    compressed_bits: int = 4
    # Same-size plan without pruning: robust layers drop to this many bits instead, so the size match comes
    # from bits alone and the effect of the sensitivity guidance is not mixed with the effect of pruning.
    no_prune_compressed_bits: int = 3
    # "Original method" for Stage 0 = the uniform allocation a standard method uses without guidance.
    uniform_bits: int = 4
    uniform_prune_ratio: float = 0.0
    # Per quantisation group GPTQ stores a scale and a zero point; bits each, for the memory prediction.
    group_overhead_bits: int = 32
    baseline_bits: int = 16  # bits/weight of the FP16 model and of unquantised tensors (embeddings, norms, lm_head)
    # KV cache plan (see stage0/kv_cache.py). Per layer: key bits, value bits, share of past tokens kept.
    kv_cache: bool = True
    kv_bits_options: list[int] = field(default_factory=lambda: [2, 4, 8])  # tested per layer, ascending
    kv_uniform_bits: int = 4  # "original method": every key and value at this many bits, nothing evicted
    kv_avg_bits: float | None = None  # plan's average bit budget; None = kv_uniform_bits (same size as original)
    kv_group_size: int = 64  # numbers sharing one scale (tokens for keys, channels for values)
    kv_calib_samples: int = 16  # passages for the KV measurement (each layer x key/value x bits is one pass)
    kv_keep_ratios: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.75])
    kv_attention_coverage: float = 0.95  # keep the fewest tokens that still receive this share of attention
    kv_context_len: int = 2048  # tokens per sequence for the memory prediction
    kv_batch_size: int = 1  # sequences held at once for the memory prediction
    kv_module_names: list[str] = field(default_factory=lambda: ["k_proj", "v_proj"])  # key / value Linear names


@dataclass
class FrameworkConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    stage0: Stage0Config = field(default_factory=Stage0Config)
    requirement: DeploymentRequirement = field(default_factory=DeploymentRequirement)
    # Default hyperparameters for single (non-search) runs; keys are SEARCH_SPACE names.
    hyperparams: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FrameworkConfig":
        return _build(cls, d)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "FrameworkConfig":
        """Apply dotted-key overrides, e.g. {"eval.seq_len": 256, "hyperparams.sensitive_threshold": 0.7}."""
        d = copy.deepcopy(self.to_dict())
        for key, value in overrides.items():
            node = d
            *parents, leaf = key.split(".")
            for part in parents:
                if part not in node:
                    raise KeyError(f"unknown config key {key!r}")
                node = node[part]
            if leaf not in node and parents != ["hyperparams"]:
                raise KeyError(f"unknown config key {key!r}")
            node[leaf] = value
        return FrameworkConfig.from_dict(d)


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> FrameworkConfig:
    cfg = FrameworkConfig()
    if path is not None:
        import yaml

        cfg = FrameworkConfig.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return cfg.with_overrides(overrides or {})


def config_hash(*parts: Any, length: int = 12) -> str:
    """Stable hash of JSON-serialisable parts (dataclasses allowed); used for run ids and cache keys."""
    payload = json.dumps([asdict(p) if is_dataclass(p) else p for p in parts], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def _build(cls, d: Mapping[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in d.items():
        ftype = _field_type(cls, name)
        kwargs[name] = _build(ftype, value) if is_dataclass(ftype) and isinstance(value, Mapping) else value
    return cls(**kwargs)


def _field_type(cls, name: str):
    import typing

    return typing.get_type_hints(cls)[name]

