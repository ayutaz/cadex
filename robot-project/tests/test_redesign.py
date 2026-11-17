import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from redesign import parse_scale, plan_steps, resolve_sets  # noqa: E402


def test_scale_multiplies_the_base_parameters():
    base = {"thigh_length": 45.0, "shin_length": 50.0, "body_length": 110.0}
    sets = resolve_sets(base, scales=[parse_scale("thigh_length,shin_length=1.2")], sets={})
    assert sets == {"thigh_length": pytest.approx(54.0), "shin_length": pytest.approx(60.0)}


def test_explicit_set_wins_over_scale():
    base = {"thigh_length": 45.0}
    sets = resolve_sets(base, scales=[parse_scale("thigh_length=1.2")], sets={"thigh_length": 50.0})
    assert sets == {"thigh_length": 50.0}


def test_unknown_parameter_is_refused():
    with pytest.raises(KeyError):
        resolve_sets({"thigh_length": 45.0}, scales=[parse_scale("leg=1.2")], sets={})


def test_plan_lists_every_stage_and_training_only_when_asked():
    steps = plan_steps("leg120", train="none", policy_base=None)
    assert [s[0] for s in steps] == ["build", "evaluate_base", "evaluate_variant", "compare"]
    steps = plan_steps("leg120", train="local", policy_base=None)
    assert [s[0] for s in steps] == ["build", "train_variant", "evaluate_base", "evaluate_variant", "compare"]
