import random

import pytest

from sdf.config import FrameworkConfig, config_hash, load_config
from sdf.requirements import DeploymentRequirement
from sdf.search_space import PER_CHANNEL, SEARCH_SPACE, Param, SearchSpace


def test_search_space_has_spec_parameters():
    assert SEARCH_SPACE.names == ["sensitive_threshold", "prune_ratio_aggressive", "calib_dataset", "calib_samples",
                                  "gptq_groupsize"]
    assert [p.name for p in SEARCH_SPACE.for_stage(0)][:2] == ["sensitive_threshold", "prune_ratio_aggressive"]


def test_candidate_validation():
    c = SEARCH_SPACE.make({"gptq_groupsize": PER_CHANNEL, "sensitive_threshold": 0.7})
    assert c.gptq_groupsize == PER_CHANNEL and c["sensitive_threshold"] == 0.7
    for bad in ({"sensitive_threshold": 0.95}, {"prune_ratio_aggressive": 0.7}, {"gptq_groupsize": 16},
                {"calib_dataset": "imagenet"}, {"not_a_param": 1}):
        with pytest.raises(ValueError):
            SEARCH_SPACE.make(bad)


def test_search_space_sampling_is_valid_and_seeded():
    a = SEARCH_SPACE.sample(random.Random(3))
    assert a == SEARCH_SPACE.sample(random.Random(3))
    SEARCH_SPACE.validate(dict(a))


def test_adding_a_parameter_is_one_line():
    space = SearchSpace(list(SEARCH_SPACE.params) + [Param("new_knob", stage=1, default=2, choices=(1, 2))])
    assert space.default()["new_knob"] == 2


def test_config_overrides_and_yaml(tmp_path):
    cfg = FrameworkConfig().with_overrides({"eval.seq_len": 256, "hyperparams.sensitive_threshold": 0.6})
    assert cfg.eval.seq_len == 256 and cfg.hyperparams["sensitive_threshold"] == 0.6
    with pytest.raises(KeyError):
        FrameworkConfig().with_overrides({"eval.nope": 1})

    loaded = load_config("configs/tinyllama.yaml")
    assert loaded.stage0.protected_bits == 8
    SEARCH_SPACE.make(loaded.hyperparams)  # the YAML defaults are inside the search space
    assert config_hash(loaded.to_dict()) == config_hash(load_config("configs/tinyllama.yaml").to_dict())


def test_requirement_check():
    req = DeploymentRequirement(target_memory_gb=1.0, target_ppl=10.0, target_latency_ms=5.0)
    ok = req.check({"model_size_gb": 0.8, "ppl_val": 9.0, "decode_ms_per_token_mean": 4.0})
    assert ok.met is True and not ok.shortfall
    bad = req.check({"predicted_weight_memory_gb": 1.5, "ppl_val": 9.0})
    assert bad.met is False and bad.shortfall == {"target_memory_gb": pytest.approx(0.5)}
    assert bad.unmeasured == ["target_latency_ms"]
    partial = req.check({"model_size_gb": 0.5})
    assert partial.met is None

