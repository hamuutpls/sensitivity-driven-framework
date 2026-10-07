import json

import numpy as np
import pytest
import torch
from torch import nn

from sdf.eval import gptq_export as g
from sdf.stage0.planner import uniform_plan


def unpack(words: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of g.pack, written from the GPTQ 3-bit layout (bit positions listed out) rather than from pack."""
    w = words.view(np.uint32).astype(np.uint64)
    if bits != 3:
        per = 32 // bits
        out = np.stack([(w >> np.uint64(bits * k)) & np.uint64(2 ** bits - 1) for k in range(per)], axis=1)
        return out.reshape(-1, w.shape[1])
    blocks = w.reshape(-1, 3, w.shape[1])
    rows = []
    for a, b, c in blocks:
        v = [(a >> np.uint64(3 * k)) & np.uint64(7) for k in range(10)]
        v.append(((a >> np.uint64(30)) & np.uint64(3)) | ((b & np.uint64(1)) << np.uint64(2)))
        v += [(b >> np.uint64(3 * k + 1)) & np.uint64(7) for k in range(10)]
        v.append(((b >> np.uint64(31)) & np.uint64(1)) | ((c & np.uint64(3)) << np.uint64(1)))
        v += [(c >> np.uint64(3 * k + 2)) & np.uint64(7) for k in range(10)]
        rows.append(np.stack(v))
    return np.concatenate(rows)


@pytest.mark.parametrize("bits", g.BITS)
def test_pack_round_trip(bits):
    q = np.random.default_rng(0).integers(0, 2 ** bits, size=(64, 5))
    packed = g.pack(q, bits)
    assert packed.dtype == np.int32 and packed.shape == (64 * bits // 32, 5)
    assert (unpack(packed, bits) == q).all()


@pytest.mark.parametrize("bits", g.BITS)
def test_quantize_linear_matches_its_grid(bits):
    w = torch.randn(64, 256, generator=torch.Generator().manual_seed(bits))
    t = g.quantize_linear(w, bits, 128)
    assert t["qweight"].shape == (256 * bits // 32, 64) and t["qzeros"].shape == (2, 64 * bits // 32)
    q = torch.from_numpy(unpack(t["qweight"].numpy(), bits).astype(np.int64)).T.reshape(64, 2, 128)
    zero = torch.from_numpy(unpack(t["qzeros"].numpy().T.copy(), bits).astype(np.int64)).T + 1  # (groups, out)
    assert (zero == 2 ** (bits - 1)).all()  # symmetric: the only GPTQ layout vLLM loads
    deq = (q - zero.T[..., None]) * t["scales"].float().T[..., None]
    step = t["scales"].float().T[..., None]
    assert ((deq.reshape(64, 256) - w).abs() <= step.expand(-1, -1, 128).reshape(64, 256) * 0.51 + 1e-3).all()
    assert (t["g_idx"] == torch.arange(256) // 128).all()


def test_export_writes_per_layer_bits(tmp_path):
    from safetensors.torch import load_file

    class Tok:
        def save_pretrained(self, d):
            pass

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(128, 64, bias=False)

    class M(nn.Module):
        config = type("C", (), {"to_dict": lambda self: {"model_type": "llama"}})()

        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block() for _ in range(3)])
            self.lm_head = nn.Linear(128, 64, bias=False)

    plan = uniform_plan([0.0] * 3, 4, 0.0)
    from dataclasses import replace
    plan = replace(plan, layers=tuple(replace(lp, bit_width=8) if lp.layer == 1 else lp for lp in plan.layers))
    size = g.export(M(), Tok(), plan, 128, tmp_path)
    t = load_file(tmp_path / "model.safetensors")
    assert t["model.layers.0.proj.qweight"].shape == (128 * 4 // 32, 64)
    assert t["model.layers.1.proj.qweight"].shape == (128 * 8 // 32, 64)
    assert t["lm_head.weight"].dtype == torch.float16 and "model.layers.0.proj.weight" not in t and size > 0
    qc = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    assert qc["bits"] == 4 and qc["sym"] is True and qc["dynamic"] == {r"+:.*layers\.1\..*": {"bits": 8}}
