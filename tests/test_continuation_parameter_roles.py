"""Eligibility depends on the equations, not a preference for conserved totals."""
import json

import pytest

from dynalyst.analysis.ode_model import continuation_parameter_error, resolve_model
from dynalyst.orchestration.inputs import parameter_problem
from dynalyst.orchestration.tools import inspect_model

MODEL = "BIOMD0000000150.xml"


def test_binding_rate_is_a_valid_continuation_parameter():
    model = resolve_model(MODEL)
    assert model["param_kinds"]["Binding_kf"] == "local parameter"
    assert continuation_parameter_error(model, "Binding_kf") is None
    assert parameter_problem({"model_name_or_path": MODEL, "param_name": "Binding_kf"}) is None
    info = json.loads(inspect_model(MODEL))
    assert "Binding_kf" in info["continuation_parameters"]
    unused = next(p for p in model["param_names"] if "basal" in p.lower())
    assert unused not in info["continuation_parameters"]


def test_unused_observable_constant_and_state_are_explained_before_auto():
    model = resolve_model(MODEL)
    # Read the actual spelling from the model, without confusing an observable with a rate.
    unused = next(p for p in model["param_names"] if "basal" in p.lower())
    problem = continuation_parameter_error(model, unused)
    assert "does not occur" in problem["error"]
    assert "Binding_kf" in problem["available_parameters"]
    assert "does not occur" in parameter_problem({"model_name_or_path": MODEL, "param_name": unused})
    assert "not a parameter" in continuation_parameter_error(model, "CyclinA")["error"]


@pytest.mark.parametrize("name", ["k", "total"])
def test_rate_and_explicit_total_are_equally_eligible(name):
    model = resolve_model(species=["x"], odes={"x": "k*(total-x)-x"},
                          parameters={"k": 1, "total": 2})
    assert continuation_parameter_error(model, name) is None


def test_invalid_parameter_fails_at_tool_entry_without_a_numerical_run(monkeypatch):
    from dynalyst.analysis import bifurcation_auto

    def must_not_run(*args, **kwargs):
        pytest.fail("An unused observable constant must be rejected before continuation")

    monkeypatch.setattr(bifurcation_auto, "_run_single_bifurcation_analysis", must_not_run)
    model = resolve_model(MODEL)
    unused = next(p for p in model["param_names"] if "basal" in p.lower())
    result = json.loads(bifurcation_auto.run_bifurcation_analysis(
        model_name_or_path=MODEL, param_name=unused))
    assert "does not occur" in result["error"]
    assert "Binding_kf" in result["available_parameters"]
