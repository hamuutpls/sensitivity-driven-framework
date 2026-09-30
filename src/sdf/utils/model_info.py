"""The original (uncompressed) model's parameters, recorded in every stage report.

Read from the Hugging Face model config, so it works for any model: fields a model's config lacks are left
out rather than guessed.
"""

from __future__ import annotations

from typing import Any

# key -> (label, plain-language meaning). Order is the order in the report.
MODEL_FACTS: dict[str, tuple[str, str]] = {
    "name": ("Model", "The published model every result in this report starts from."),
    "architecture": ("Architecture", "The model family's design; models of the same family are built the same way."),
    "num_parameters": ("Parameters", "How many numbers the model has learned. More numbers usually means a more "
                                     "capable but bigger and slower model."),
    "num_layers": ("Layers", "How many similar building blocks the text passes through, one after another. "
                             "Stage 0 measures each one separately."),
    "hidden_size": ("Hidden size", "How many numbers describe each word inside the model, as it passes from "
                                   "layer to layer."),
    "intermediate_size": ("Feed-forward size", "How wide the \"thinking\" part inside each layer is, in numbers "
                                               "per word."),
    "num_attention_heads": ("Attention heads", "How many separate ways each layer looks back at earlier words "
                                               "at the same time."),
    "num_key_value_heads": ("Key/value heads", "How many sets of notes each layer stores about earlier words "
                                               "(the KV cache). Fewer than the attention heads means the notes "
                                               "are shared, which saves memory."),
    "head_dim": ("Numbers per head", "How many numbers each attention head uses per word."),
    "vocab_size": ("Vocabulary", "How many different word pieces (tokens) the model knows."),
    "max_context": ("Maximum context", "The most tokens (word pieces) the model can read and write at once."),
    "published_dtype": ("Published number format", "How the numbers are stored in the published model files."),
    "bits_per_parameter": ("Bits per number", "Storage per number in the uncompressed model; compression "
                                              "lowers this."),
    "fp16_size_gb": ("Size at 16 bits", "Memory the numbers take in the uncompressed 16-bit format "
                                        "(parameters x 2 bytes)."),
    "tied_embeddings": ("Input and output word tables shared", "Whether the table turning words into numbers is "
                                                              "reused to turn numbers back into words, which "
                                                              "saves memory."),
}


def describe_model(config: Any, name: str | None = None, num_parameters: int | None = None,
                   bits: int = 16) -> dict[str, Any]:
    """Facts about the original model from its HF config (`model.config` or `AutoConfig`).

    `num_parameters` (unique parameters, tied weights once) cannot be read from the config, so pass it when
    known; without it the parameter count and 16-bit size are left out.
    """
    def get(*names: str) -> Any:
        for n in names:
            v = getattr(config, n, None)
            if v is not None:
                return v
        return None

    heads = get("num_attention_heads", "n_head")
    hidden = get("hidden_size", "n_embd", "d_model")
    dtype = get("torch_dtype", "dtype")
    archs = get("architectures")
    info: dict[str, Any] = {
        "name": name or get("name_or_path", "_name_or_path") or None,
        "architecture": archs[0] if archs else get("model_type"),
        "num_parameters": num_parameters,
        "num_layers": get("num_hidden_layers", "n_layer", "num_layers"),
        "hidden_size": hidden,
        "intermediate_size": get("intermediate_size", "ffn_dim", "n_inner"),
        "num_attention_heads": heads,
        "num_key_value_heads": get("num_key_value_heads", "num_kv_heads") or heads,
        "head_dim": get("head_dim") or (hidden // heads if hidden and heads else None),
        "vocab_size": get("vocab_size"),
        "max_context": get("max_position_embeddings", "n_positions", "max_sequence_length"),
        "published_dtype": str(dtype).replace("torch.", "") if dtype is not None else None,
        "bits_per_parameter": bits,
        "fp16_size_gb": num_parameters * bits / 8 / 1e9 if num_parameters else None,
        "tied_embeddings": get("tie_word_embeddings"),
    }
    return {k: v for k, v in info.items() if v not in (None, "")}


def count_parameters(model: Any) -> int:
    """Unique parameters (tied weights counted once)."""
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
