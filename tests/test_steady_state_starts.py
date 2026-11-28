"""find_steady_states reaches states on the scale of the model's parameters, not only of its
initial values (a model whose initial values are all 0), and states of either sign."""
import json

import pytest

from dynalyst import config


def _states(**kw):
    from dynalyst.analysis.steady_states import find_steady_states
    fs = getattr(find_steady_states, "func", find_steady_states)
    return json.loads(fs(**kw))["steady_states"]


def test_both_stable_states_of_a_model_starting_at_zero_are_found():
    path = config.PROJECT_ROOT / "data" / "sbml_models" / "Full_model_for_SN_plus_TB.xml"
    if not path.exists():
        pytest.skip("Full_model_for_SN_plus_TB.xml is not shipped here")
    states = _states(model_name_or_path=str(path), fixed_params={"ATOT": 200})
    stable = sorted(s["state"]["APP"] for s in states if s["stability"].startswith("stable"))
    assert stable == [pytest.approx(4.443, abs=0.01), pytest.approx(64.241, abs=0.01)]
    assert any(abs(s["state"]["APP"] - 5.78) < 0.01 for s in states)        # the saddle between


def test_negative_states_are_found_where_a_variable_may_be_negative():
    xs = sorted(s["state"]["x"] for s in _states(species=["x"], odes={"x": "p + x - x**3"},
                                                 parameters={"p": 0.0}))
    assert xs == [pytest.approx(-1.0), pytest.approx(0.0, abs=1e-9), pytest.approx(1.0)]
