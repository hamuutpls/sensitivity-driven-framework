import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sdf.config import FrameworkConfig


class CharTokenizer:
    """Tiny offline tokenizer: one token per character."""

    vocab_size = 64

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) % self.vocab_size for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


def fake_texts(dataset, split):
    words = "the quick brown fox jumps over the lazy dog while sensitive layers stay protected".split()
    return [" ".join(words[(i + j) % len(words)] for j in range(60)) for i in range(40)]


@pytest.fixture
def tiny_llama():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    return LlamaForCausalLM(cfg)


@pytest.fixture
def tokenizer():
    return CharTokenizer()


@pytest.fixture
def small_cfg(tmp_path):
    return FrameworkConfig().with_overrides({
        "run.output_root": str(tmp_path / "results"),
        "run.run_id": "test",
        "model.dtype": "float32",
        "model.device": "cpu",
        "calibration.seq_len": 16,
        "calibration.batch_size": 2,
        "eval.seq_len": 16,
        "eval.max_windows": 6,
        "eval.latency_prompt_len": 8,
        "eval.latency_decode_tokens": 4,
        "eval.latency_warmup": 1,
        "eval.latency_repeats": 3,
        "stage0.kv_group_size": 8,
        "stage0.kv_calib_samples": 4,
        "stage0.guard_top_k": 1,  # the toy model has 4 layers; guard one so the others can still be pruned
    })
