"""Model reduction (conservation laws, timescale analysis, QSSA), LHS
parameter estimation with identifiability, and follow-up suggestions."""
import json
from types import SimpleNamespace

import pytest

MM = dict(species=["S", "E", "C", "P"],
          odes={"S": "-k1*E*S + km1*C", "E": "-k1*E*S + (km1+k2)*C",
                "C": "k1*E*S - (km1+k2)*C", "P": "k2*C"},
          parameters={"k1": 10.0, "km1": 1.0, "k2": 1.0},
          initial_conditions={"S": 10.0, "E": 0.1, "C": 0.0, "P": 0.0})


def test_qssa_reproduces_michaelis_menten():
    """E0 << S0: the enzyme is fast; QSSA must give dS/dt = -k2*E0*S/(Km + S)."""
    import sympy as sp
    from dynalyst.analysis.model_reduction import reduce_model
    r = json.loads(reduce_model(**MM, export=False))
    assert r["conservation_laws_eliminated"] and r["timescale_analysis"]["qssa_recommended"]
    assert r["qssa"]["closed_form"] and r["reduced_state_variables"] == ["S"]
    S, k1, k2, km1 = sp.symbols("S k1 k2 km1")
    got = sp.sympify(r["reduced_odes"]["dS/dt"], locals={"S": S})
    textbook = -0.1 * k2 * S / (S + (k2 + km1) / k1)
    assert sp.simplify(got - textbook) == 0 or abs(float((got - textbook).subs(
        {S: 3.0, k1: 10.0, k2: 1.0, km1: 1.0}))) < 1e-12
    # A QSSA starts off the full model's fast transient: report that discrepancy,
    # then check all reconstructed states after the identified initial layer.
    assert not r["validation"]["accurate"]
    assert r["validation"]["post_initial_layer"]["accurate"]
    # The stricter check now includes enzyme/complex reconstruction, whose error
    # is larger than the retained substrate's but below the requested 5% limit.
    assert r["validation"]["post_initial_layer"]["max_relative_error"] < 0.05


def test_no_qssa_without_timescale_gap():
    from dynalyst.analysis.model_reduction import reduce_model
    r = json.loads(reduce_model("pitchfork_model.xml", export=False))
    assert not r["timescale_analysis"]["qssa_recommended"] and "qssa" not in r


def test_full_mass_action_model_fast_species_found_from_the_model():
    """KTOT, PTOT >> ATOT: the free substrate AP (not a complex) is fastest.
    The numerical QSSA is validated and honestly reported."""
    from dynalyst.analysis.model_reduction import reduce_model
    r = json.loads(reduce_model("Full_model_for_SN", export=False))
    ts = r["timescale_analysis"]
    assert ts["qssa_recommended"] and "AP" in ts["fast_species"]
    assert len(r["qssa"]["slow_species"]) == 2
    assert "max_relative_error" in r["validation"]


def test_multi_law_conservation_no_longer_crashes_research_tools():
    """steady_states.conservation_reduce used to fail with >= 2 conservation laws."""
    from dynalyst.analysis.steady_states import conservation_reduce
    from dynalyst.analysis.ode_model import model_from_odes
    m = conservation_reduce(model_from_odes(MM["species"], MM["odes"], MM["parameters"],
                                            MM["initial_conditions"]))
    assert len(m["state_names"]) == 2 and len(m["eliminated"]) == 2


def test_classifier_reports_timescales():
    from dynalyst.analysis.model_classifier import classify_model
    r = json.loads(classify_model(**MM))
    assert r["timescales"]["qssa_recommended"] is True
    assert "quasi-steady-state" in r["reduction_advice"]


@pytest.fixture(scope="module")
def steady_state_data(tmp_path_factory):
    import jax.numpy as jnp
    import numpy as np
    import pandas as pd
    from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
    from dynalyst import config
    from dynalyst.analysis.jaxkm_compat import jaxkm_ready_sbml
    m = SBMLModel(str(jaxkm_ready_sbml(config.MODELS_DIR / "pitchfork_model.xml")))
    k = m.get_kinetic_model()
    ts = np.linspace(0, 400, 41)
    y_ss = np.asarray(k(jnp.asarray(ts), m.y0, m.parameters))[-1]
    flat = np.asarray(k(jnp.asarray(ts), jnp.asarray(y_ss), m.parameters))
    path = tmp_path_factory.mktemp("data") / "steady.csv"
    pd.DataFrame({"time": ts, **{s: flat[:, i] for i, s in enumerate(m.species_names)}}).to_csv(
        path, index=False)
    return str(path)


def test_lhs_multistart_detects_non_identifiability(steady_state_data):
    """Steady-state data only pin down ratios: four rate constants cannot all
    be identified, and the pairs that trade off must be reported."""
    from dynalyst.analysis.parameter_estimation import estimate_parameters
    r = json.loads(estimate_parameters("pitchfork_model.xml", steady_state_data,
                                       parameters_to_fit=["k2", "k4", "k6", "k8"],
                                       n_initializations=6))
    assert r["lhs_design"]["sampler"].startswith("scipy.stats.qmc.LatinHypercube (log")
    assert len(r["lhs_design"]["start_points"]) == 6
    ms = r["multistart"]
    assert all(v["status"].startswith("not identifiable") for v in ms["parameters"].values())
    pairs = {frozenset(c["parameters"]) for c in ms["correlated_pairs"]}
    assert frozenset({"k2", "k8"}) in pairs and frozenset({"k4", "k6"}) in pairs


def test_lhs_bounds_and_jaxkm_linear_sampler(steady_state_data):
    from dynalyst.analysis.parameter_estimation import estimate_parameters
    r = json.loads(estimate_parameters("pitchfork_model.xml", steady_state_data,
                                       parameters_to_fit=["k4"], n_initializations=2, n_iter=20,
                                       parameter_bounds={"k4": [0.1, 0.3]}))
    assert r["lhs_design"]["bounds"]["k4"] == [0.1, 0.3]
    assert all(0.1 <= p["k4"] <= 0.3 for p in r["lhs_design"]["start_points"])
    r = json.loads(estimate_parameters("pitchfork_model.xml", steady_state_data,
                                       parameters_to_fit=["k4"], n_initializations=2, n_iter=20,
                                       lhs_space="linear"))
    assert r["lhs_design"]["sampler"].startswith("jaxkineticmodel Trainer.latinhypercube_sampling")


def _record(*steps):
    return SimpleNamespace(steps=[{"kind": "delegation", **s} for s in steps])


def test_follow_ups_from_results():
    from dynalyst.orchestration.next_steps import format_follow_ups, suggest
    rec = _record({"sub_agent": "steady_state", "tool": "find_steady_states", "status": "ok",
                   "arguments": {"model_name_or_path": "Markevich2004"},
                   "output": json.dumps({"steady_states": [{"stability": "stable_node"},
                                                           {"stability": "saddle (unstable)"},
                                                           {"stability": "stable_node"}]})})
    items = suggest(rec)
    assert "bistability" in items[0]["question"] and items[0]["query"].startswith(
        "Run a bifurcation analysis of Markevich2004")
    assert "**What next?**" in format_follow_ups(items)

    rec = _record({"sub_agent": "parameter_estimation", "tool": "estimate_parameters",
                   "status": "ok",
                   "arguments": {"model_name_or_path": "m.xml", "data_path": "d.csv"},
                   "output": json.dumps({"fitted_parameters": {"k2": {"fitted": 1}, "k8": {"fitted": 1}},
                                         "multistart": {"parameters": {
                                             "k2": {"status": "not identifiable from these data"},
                                             "k8": {"status": "not identifiable from these data"}},
                                             "correlated_pairs": [{"parameters": ["k2", "k8"]}]}})})
    q = suggest(rec)[0]
    assert "Fix k2" in q["question"] and "['k8']" in q["query"]


def test_follow_ups_offer_qssa_for_high_dimensional_models():
    from dynalyst.orchestration.next_steps import suggest
    rec = _record({"sub_agent": "simulation", "tool": "simulate_model", "status": "ok",
                   "arguments": {"model_name_or_path": "Full_model_for_SN"}, "output": "{}"})
    items = suggest(rec)
    # the timescale offer comes right after the run's own first suggestion (next_steps.suggest)
    offer = next(i for i, x in enumerate(items) if "QSSA" in x["question"])
    assert offer <= 1 and items[offer]["query"].startswith("Reduce Full_model_for_SN")
