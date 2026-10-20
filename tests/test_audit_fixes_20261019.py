"""Fixes from the 2026-10-19 ponytail audit (numbers are the audit's findings). Every test here fails on main 0c3c70d."""

import pytest

from sdf.config import FrameworkConfig


# 17 -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key, value", [("eval.latency_mode", "cuda-graph"), ("eval.latency_repeats", 0),
                                        ("eval.latency_decode_tokens", 0)])
def test_timing_settings_are_checked_when_the_config_is_built(key, value):
    with pytest.raises(ValueError, match=key):
        FrameworkConfig().with_overrides({key: value})
    FrameworkConfig().with_overrides({"eval.latency_mode": "eager", "eval.latency_repeats": 1,
                                      "eval.latency_decode_tokens": 1})
