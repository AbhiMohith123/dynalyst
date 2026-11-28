"""The tools added for the agent: model classifier, annotations, custom plot,
parameter estimation -- and the jaxkineticmodel simulation fix for SBML with
assignment rules."""
import json

import numpy as np
import pytest

from dynalyst import config


def test_classifier_dimensions():
    from dynalyst.analysis.model_classifier import classify_model
    # assignment rules: the research reader substitutes them now (it used to reject them and
    # the regime parser took over); both give the same equations, see test_tools_smoke
    r = json.loads(classify_model("pitchfork_model.xml"))
    assert (r["dimension_class"], r["reader"]) == ("2D", "research SBML/ODE reader")
    r = json.loads(classify_model("Markevich2004"))             # M + Mp + Mpp conserved
    assert r["n_state_variables"] == 3 and r["n_conservation_laws"] == 1
    assert r["dimension_class"] == "2D"
    r = json.loads(classify_model(species=["x"], odes={"x": "mu + x**2"}))
    assert r["dimension_class"] == "1D" and "No Hopf" in r["implications"]
    r = json.loads(classify_model("Garde2020"))                 # SBML function definitions, expanded
    assert r["reader"] == "research SBML/ODE reader" and r["independent_dimension"] == 6


def test_annotations():
    from dynalyst.analysis.annotations import get_annotations
    a = json.loads(get_annotations("Markevich2004"))
    refs = [u for t in a["model"]["cross_references"] for u in t["resources"]]
    assert any("BIOMD0000000027" in u for u in refs) and any("pubmed" in u for u in refs)
    assert a["model"]["history"]["creators"]
    assert json.loads(get_annotations("pitchfork_model.xml"))["has_any_annotation"] is False


def _reference_app(t_end):
    import sympy as sp
    from scipy.integrate import solve_ivp
    from dynalyst.regime.parser import parse_sbml
    parsed = parse_sbml(str(config.MODELS_DIR / "pitchfork_model.xml"))
    subs = {sp.Symbol(k): v for k, v in parsed["param_values"].items()}
    f = sp.lambdify([sp.Symbol(s) for s in parsed["species"]],
                    [parsed["ode_rhs"][s].subs(subs) for s in parsed["species"]])
    sol = solve_ivp(lambda t, y: f(*y), (0, t_end), [0.01, 0.01], method="LSODA",
                    rtol=1e-10, atol=1e-12)
    return float(sol.y[parsed["species"].index("APP"), -1])


def test_simulation_of_assignment_rule_sbml_is_not_flat():
    """jaxkineticmodel treats assignment-rule fluxes as constant 0 (a flat
    line); the converted copy must reproduce an independent SciPy solve."""
    from dynalyst.analysis.jaxkm_tools import simulate_model
    r = json.loads(simulate_model("pitchfork_model.xml", t_end=2000, n_points=5))
    assert r["final_concentrations"]["APP"] == pytest.approx(_reference_app(2000), rel=1e-6)
    assert r["final_concentrations"]["APP"] > 0.03  # started at 0.01


def test_custom_plot_overlays_conditions():
    from dynalyst.analysis.custom_plot import custom_plot
    r = json.loads(custom_plot("pitchfork_model.xml", ["APP"], t_end=2000, conditions=[
        {"label": "KTOT=100"}, {"label": "KTOT=200", "parameter_overrides": {"KTOT": 200}}]))
    a, b = (c["final"]["APP"] for c in r["conditions"])
    assert a != pytest.approx(b) and (config.PROJECT_ROOT / r["plot_path"]).exists()
    assert "Unknown species" in json.loads(custom_plot("pitchfork_model.xml", ["nope"]))["error"]


@pytest.fixture
def synthetic_timecourse(tmp_path):
    import jax.numpy as jnp
    import pandas as pd
    from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
    from dynalyst.analysis.jaxkm_compat import jaxkm_ready_sbml
    m = SBMLModel(str(jaxkm_ready_sbml(config.MODELS_DIR / "pitchfork_model.xml")))
    true = dict(m.parameters)
    true["k4"] *= 2
    true["k8"] *= 0.5
    ts = np.linspace(0, 400, 41)
    ys = np.asarray(m.get_kinetic_model()(jnp.asarray(ts), m.y0, true))
    path = tmp_path / "synthetic.csv"
    pd.DataFrame({"time": ts, **{s: ys[:, i] for i, s in enumerate(m.species_names)}}).to_csv(
        path, index=False)
    return str(path), true


def test_inspect_dataset(synthetic_timecourse):
    from dynalyst.analysis.parameter_estimation import inspect_dataset
    path, _ = synthetic_timecourse
    r = json.loads(inspect_dataset(path, "pitchfork_model.xml"))
    assert sorted(r["matched_species"]) == ["A", "APP"] and r["n_time_points"] == 41
    assert "k4" in r["fittable_parameters"] and "r1" not in r["fittable_parameters"]


def test_parameter_estimation_recovers_true_values(synthetic_timecourse):
    from dynalyst.analysis.parameter_estimation import estimate_parameters
    path, true = synthetic_timecourse
    r = json.loads(estimate_parameters("pitchfork_model.xml", path,
                                       parameters_to_fit=["k4", "k8"], n_initializations=3))
    fitted = {p: v["fitted"] for p, v in r["fitted_parameters"].items()}
    assert fitted["k4"] == pytest.approx(true["k4"], rel=0.01)
    assert fitted["k8"] == pytest.approx(true["k8"], rel=0.01)
    assert r["final_loss"] < r["initial_loss"]
    assert "r1" not in r["fixed_parameters"]  # assignment-rule fluxes are not parameters
    assert (config.PROJECT_ROOT / r["plot_path"]).exists()


def test_a_time_course_scan_runs_in_one_call_and_survives_a_bad_value():
    from dynalyst.analysis.custom_plot import custom_plot
    conditions = [{"label": f"MAPKK={v}", "parameter_overrides": {"MAPKK": v}} for v in (40, 50, 70)]
    conditions.append({"label": "bad", "parameter_overrides": {"no_such_parameter": 1}})
    r = json.loads(custom_plot("Markevich2004.xml", ["Mpp"], t_end=5000, conditions=conditions))
    assert "error" in r["conditions"][-1] and "final" in r["conditions"][0]
    assert (config.PROJECT_ROOT / r["plot_path"]).exists()


def test_a_scan_of_one_parameter_gives_final_values_from_its_one_start():
    from dynalyst.analysis.custom_plot import custom_plot
    r = json.loads(custom_plot("Markevich2004.xml", ["Mpp"], t_end=5000, conditions=[
        {"label": f"MAPKK={v}", "parameter_overrides": {"MAPKK": v}} for v in (40, 50, 70)]))
    scan = r["final_vs_parameter"]
    assert scan["parameter"] == "MAPKK" and scan["values"] == [40, 50, 70] and "one branch" in r["note"]
    low_40, low_50, high_70 = scan["final"]["Mpp"]
    assert low_40 < low_50 < 100 < 400 < high_70       # the upper fold is at 57.38
    assert all(scan["settled"]["Mpp"]) and not scan["failed"]
