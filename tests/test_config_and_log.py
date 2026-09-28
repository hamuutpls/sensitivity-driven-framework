import pytest

from sdf.config import DEFAULT_SEARCH_SPACE, PER_CHANNEL, CandidateConfig
from sdf.trial_log import Objectives, TrialLog, TrialRecord


def test_default_space_accepts_valid_candidate():
    DEFAULT_SEARCH_SPACE.validate(
        CandidateConfig(sensitivity_threshold=0.5, pruning_ratio=0.3, group_size=PER_CHANNEL,
                        migration_strength=0.8, cache_bits=4)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sensitivity_threshold=0.95, pruning_ratio=0.3),
        dict(sensitivity_threshold=0.5, pruning_ratio=0.7),
        dict(sensitivity_threshold=0.5, pruning_ratio=0.3, group_size=16),
        dict(sensitivity_threshold=0.5, pruning_ratio=0.3, migration_strength=0.2),
        dict(sensitivity_threshold=0.5, pruning_ratio=0.3, cache_bits=3),
    ],
)
def test_default_space_rejects_out_of_range(kwargs):
    with pytest.raises(ValueError):
        DEFAULT_SEARCH_SPACE.validate(CandidateConfig(**kwargs))


def test_trial_log_roundtrip(tmp_path):
    log = TrialLog(tmp_path / "trials.jsonl")
    log.clear()
    cfg = CandidateConfig(sensitivity_threshold=0.4, pruning_ratio=0.2, group_size=64)
    log.append(TrialRecord(trial_id=0, searcher="mobo", config=cfg,
                           objectives=Objectives(memory_gb=1.1, latency_ms_per_token=20.0, accuracy=0.61),
                           cost={"wall_clock_s": 12.5}))
    log.append(TrialRecord(trial_id=1, searcher="nsga3", config=cfg, fidelity="rung0"))

    records = log.records()
    assert len(log) == 2
    assert records[0].config == cfg
    assert records[0].objectives.accuracy == 0.61
    assert records[1].objectives is None
    assert [r.trial_id for r in log.records(searcher="nsga3")] == [1]
