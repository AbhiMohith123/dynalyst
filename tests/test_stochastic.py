"""Stochastic simulation (analysis/stochastic.py): reaction channels from the
ODEs, Gillespie SSA, chemical Langevin equation, and their agent wiring.

The reference case is the guide's reduced Michaelis-Menten model
(INSTRUCTIONS_STOCHSIM.pdf / SSA_MM_RED.m): V = [1 -1 0; 0 1 -1],
w1 = vs*Omega, w2 = vm*Omega*S/(Km*Omega + S), w3 = vd*P.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from dynalyst.analysis import stochastic as st


def sp_symbol(name):
    import sympy as sp
    return sp.Symbol(name)

MM = dict(species=["S", "P"], odes={"S": "vs - vm*S/(Km+S)", "P": "vm*S/(Km+S) - vd*P"},
          parameters={"vs": 0.1, "vm": 1, "Km": 2, "vd": 0.1})


def _run(**kw):
    # These legacy tests compare numerical methods and deterministic attractors.
    kw.setdefault("methods", ["ode", "ssa", "cle"])
    out = json.loads(st.compare_simulations(**kw))
    assert "error" not in out, out.get("error")
    return out


def test_mm_channels_match_the_guides_table():
    out = json.loads(st.propensity_table(**MM))
    assert out["stoichiometric_matrix"]["V"] == [[1, -1, 0], [0, 1, -1]]
    props = [r["propensity"] for r in out["reactions"]]
    assert props[0] == "w1 = Omega*vs"
    assert props[1].replace(" ", "") in ("w2=Omega*S*vm/(Km*Omega+S)", "w2=Omega*S*vm/(Omega*Km+S)")
    assert props[2] == "w3 = P*vd"


def test_the_xpp_file_gives_the_same_table():
    out = json.loads(st.propensity_table(model_name_or_path="mm_reduced.ode"))
    assert out["stoichiometric_matrix"]["V"] == [[1, -1, 0], [0, 1, -1]]


def test_simplified_odes_are_merged_back_into_the_same_reactions():
    # the XPP parser writes each RHS as one fraction; same-stoichiometry terms are summed
    out = json.loads(st.propensity_table(
        species=["S", "P"],
        odes={"S": "(Km*vs + S*vs - S*vm)/(Km + S)", "P": "(S*vm - Km*P*vd - P*S*vd)/(Km + S)"},
        parameters=MM["parameters"]))
    assert out["n_reactions"] == 3
    assert out["stoichiometric_matrix"]["V"] == [[1, -1, 0], [0, 1, -1]]


def test_a_reversible_reaction_is_two_channels():
    out = json.loads(st.propensity_table(
        species=["A", "B"], odes={"A": "-kf*A + kr*B", "B": "kf*A - kr*B"},
        parameters={"kf": 1, "kr": 0.5}))
    assert sorted(map(tuple, np.array(out["stoichiometric_matrix"]["V"]).T.tolist())) == \
        [(-1, 1), (1, -1)]


def test_ssa_and_cle_approach_the_ode_at_large_omega():
    out = _run(**MM, t_end=100, omega=[1000], n_runs=8, seed=3)
    for sp_name in ("S", "P"):
        s = out["per_omega"][0]["species"][sp_name]
        assert s["ssa_vs_ode_mean_rel_diff"] < 0.1
        assert abs(s["cle_mean"] - s["ode_final"]) / s["ode_final"] < 0.1


def test_noise_falls_as_omega_grows():
    out = _run(**MM, t_end=100, omega=[10, 1000], n_runs=8, seed=1)
    cv = [p["species"]["P"]["ssa_cv"] for p in out["per_omega"]]
    assert cv[0] > 3 * cv[1]


def test_ferrell_3d_oscillates_in_the_ode_and_in_the_ssa():
    out = _run(model_name_or_path="ferrell_cell_cycle_3d.ode", t_end=25, omega=[1000],
               n_runs=3, methods=["ode", "ssa"], seed=2)
    s = out["per_omega"][0]["species"]["CDK1"]
    assert 3.0 < s["ode_oscillation"]["mean_period"] < 6.0
    assert isinstance(s["ssa_run1_oscillation"], dict)


def test_ferrell_1d_cannot_oscillate():
    out = _run(model_name_or_path="ferrell_cell_cycle_1d.ode", t_end=25, omega=[100],
               n_runs=3, seed=0)
    assert "ode_oscillation" not in out["per_omega"][0]["species"]["CDK1"]


def test_non_integer_stoichiometry_skips_the_ssa_but_runs_the_cle():
    out = _run(species=["A", "B"], odes={"A": "-k*A", "B": "0.7*k*A"},
               parameters={"k": 1}, initial_conditions=None, t_end=5, omega=[100], n_runs=3)
    assert "ssa" not in out["methods"]
    assert any("SSA skipped" in w for w in out["warnings"])
    assert "cle" in out["per_omega"][0]


def test_an_ssa_over_the_time_budget_is_skipped_with_the_options(monkeypatch):
    from dynalyst.analysis import stochastic_ssa
    monkeypatch.setattr(st, "SSA_MAX_SECONDS", 1e-6)
    monkeypatch.setattr(stochastic_ssa, "jax_available", lambda: False)
    out = _run(**MM, t_end=100, omega=[1000], n_runs=2)
    assert "skipped" in out["per_omega"][0]["ssa"]
    assert "ssa_method='exact'" in out["per_omega"][0]["ssa"]["skipped"]
    assert "cle" in out["per_omega"][0]


def test_matlab_script_mirrors_the_guide(tmp_path):
    out = json.loads(st.propensity_table(**MM, export_matlab=True))
    text = Path(out["matlab_script"]).read_text()
    assert "V = [1 -1 0; 0 1 -1];" in text
    assert "omega.*vs" in text and "ode45" in text and "randn" in text


def test_unknown_parameter_is_refused():
    out = json.loads(st.stochastic_simulation(model_name_or_path="mm_reduced.ode",
                                              parameters={"nope": 1}, t_end=1))
    assert "error" in out


def test_stochastic_agent_owns_the_stochastic_tools():
    from dynalyst.orchestration.subagents import owner_of
    assert owner_of("stochastic_simulation") == "stochastic"
    assert owner_of("propensity_table") == "stochastic"


def _fake_sbml(rate, reversible):
    """A model dict as model_from_sbml returns it: simplified rhs plus the reactions."""
    import sympy as sp
    A, B, Vf, Vr, Ka, Kb = sp.symbols("A B Vf Vr Ka Kb")
    rate = rate(A, B, Vf, Vr, Ka, Kb)
    return {"label": "fake", "source": "sbml", "state_names": ["A", "B"], "state_symbols": [A, B],
            "param_names": ["Vf", "Vr", "Ka", "Kb"], "param_symbols": [Vf, Vr, Ka, Kb],
            "param_defaults": {"Vf": 1.0, "Vr": 0.2, "Ka": 1.0, "Kb": 1.0},
            "rhs": [sp.simplify(-rate), sp.simplify(rate)], "y0": {"A": 1.0, "B": 0.0},
            "display": {"reactions": [{"id": "R1", "rate": rate, "reactants": [("A", 1)],
                                       "products": [("B", 1)], "reversible": reversible}],
                        "rate_rules": {}, "assignment_rules": {}, "constants": {}}}


def test_an_sbml_reversible_rate_law_fires_both_ways():
    # net flux A -> B never changes sign here, yet the reverse events still happen
    m = _fake_sbml(lambda A, B, Vf, Vr, Ka, Kb: (Vf * A / Ka - Vr * B / Kb) / (1 + A / Ka + B / Kb),
                   True)
    v = m["param_defaults"]
    ch = st.reaction_channels(m, v, st.sample_points(m, v, 10))
    assert sorted(tuple(c["stoich"]) for c in ch) == [(-1.0, 1.0), (1.0, -1.0)]
    assert {c["label"] for c in ch} == {"R1", "R1 (reverse)"}


def test_an_sbml_irreversible_reaction_is_one_channel():
    m = _fake_sbml(lambda A, B, Vf, Vr, Ka, Kb: Vf * A / (Ka + A), False)
    v = m["param_defaults"]
    ch = st.reaction_channels(m, v, st.sample_points(m, v, 10))
    assert [tuple(c["stoich"]) for c in ch] == [(-1.0, 1.0)]


# --- stationary distribution and the stability verdict ---------------------------------

def _verdict(out, method="SSA", k=0):
    return out["per_omega"][k]["distribution"]["by_method"][method]


def test_a_monostable_model_is_one_peak_and_monostable():
    out = _run(**MM, t_end=200, omega=[1000], n_runs=6, seed=1)
    v = _verdict(out)
    assert v["n_peaks"] == 1 and v["verdict"] == "one stable state found"
    assert "does not prove" in " ".join(v["explanation"])
    assert out["distribution_plot_path"]


def test_a_toggle_switch_is_bistable_and_switches():
    out = _run(species=["u", "v"], odes={"u": "a/(1+v**2) - u", "v": "a/(1+u**2) - v"},
               parameters={"a": 3}, initial_conditions={"u": 1, "v": 1.1}, t_end=200,
               n_runs=16, omega=[30], seed=0)
    v = _verdict(out)
    assert v["n_peaks"] == 2 and v["verdict"] == "bistable"
    stable = [a for a in out["per_omega"][0]["distribution"]["ode_attractors"]
              if a.get("stability") == "stable"]
    assert len(stable) == 2


def test_an_oscillator_is_called_oscillatory_not_bistable():
    out = _run(model_name_or_path="ferrell_cell_cycle_3d.ode", t_end=60, omega=[200],
               n_runs=6, methods=["ode", "ssa"], seed=0)
    assert _verdict(out)["verdict"] == "oscillatory in the ODE"


def test_the_pitchfork_normal_form_names_the_missing_negative_branch():
    out = _run(species=["p"], odes={"p": "alpha*p - p**3"}, parameters={"alpha": 1},
               initial_conditions={"p": 0.5}, t_end=50, n_runs=8, omega=[100])
    assert "negative" in " ".join(_verdict(out)["explanation"])


def test_the_saddle_node_normal_form_has_two_stable_states_and_an_absorbing_zero():
    out = _run(species=["p"], odes={"p": "alpha*p**2/(1+p**2) - p"}, parameters={"alpha": 2.2},
               initial_conditions={"p": 0.9}, t_end=50, n_runs=12, omega=[30], seed=3)
    att = out["per_omega"][0]["distribution"]["ode_attractors"]
    assert sorted(round(a["readout"], 2) for a in att if a.get("stability") == "stable") == [0.0, 1.56]


def test_the_mapk_sn_model_has_two_stable_states_at_atot_430():
    # the guide's MAPKDUALPDP_..._GILLESPIE.m parameter set
    out = _run(model_name_or_path="Full_model_for_SN.txt", parameters={"ATOT": 430},
               t_end=100, n_runs=2, record_dt=2, methods=["ode", "ssa"])
    assert out["omega"] == [1.0]                      # written in molecule numbers
    d = out["per_omega"][0]["distribution"]
    assert d["readouts"][0]["species"] == "APP"        # the file's autovar
    stable = sorted(a["readout"] for a in d["ode_attractors"] if a.get("stability") == "stable")
    assert len(stable) == 2 and stable[0] < 30 < 200 < stable[1]
    assert out["stability_verdicts"]


def test_the_matlab_export_pools_replicates_into_a_histogram():
    out = _run(**MM, t_end=50, omega=[100], n_runs=3, export_matlab=True)
    text = Path(out["matlab_script"]).read_text()
    assert "nRep = 3" in text and "histogram(pooled" in text and "names{" in text


@pytest.mark.skipif(not __import__("dynalyst.analysis.stochastic_ssa",
                                   fromlist=["x"]).jax_available(), reason="jax not installed")
def test_the_jax_engine_matches_numpy():
    from dynalyst.analysis import stochastic_ssa as eng
    model, values, _, channels = st._prepare(None, MM["species"], MM["odes"], MM["parameters"],
                                             None, 100)
    V = np.array([c["stoich"] for c in channels]).T
    grid = np.linspace(0, 100, 201)
    a_fn = st._propensity_fn(model, channels, values, 100.0)
    x0 = np.zeros(2)
    a, _ = eng.exact_numpy(a_fn, V, x0, grid, np.random.default_rng(0), 20, 10**7)
    b, info = eng.exact_jax(model, channels, values, 100.0, V, x0, grid, 0, 20, 10**7)
    assert info["engine"] == "exact (JAX)"
    for j in range(2):
        ma, mb = np.mean(a[:, 100:, j]), np.mean(b[:, 100:, j])
        assert abs(ma - mb) / ma < 0.1


def test_markevich_ordered_mm_is_bistable_at_its_default_parameters():
    """BIOMD0000000027 (Markevich 2004, ordered Michaelis-Menten) with MAPKK = 50:
    stable states at Mpp ~ 49 and ~ 482 (M ~ 438 and ~ 12). The runs all start at
    M = 500 and stay low, so the second state must be found from the whole conserved
    space M + Mp + Mpp = 500, not only from where the runs went."""
    r1 = "k1cat*MAPKK*M/Km1/(1 + M/Km1 + Mp/Km2)"
    r2 = "k2cat*MAPKK*Mp/Km2/(1 + M/Km1 + Mp/Km2)"
    r3 = "k3cat*MKP3*Mpp/Km3/(1 + Mpp/Km3 + Mp/Km4 + M/Km5)"
    r4 = "k4cat*MKP3*Mp/Km4/(1 + Mpp/Km3 + Mp/Km4 + M/Km5)"
    out = _run(species=["M", "Mp", "Mpp"],
               odes={"M": f"-({r1}) + ({r4})", "Mp": f"({r1}) - ({r2}) + ({r3}) - ({r4})",
                     "Mpp": f"({r2}) - ({r3})"},
               parameters=dict(k1cat=0.01, Km1=50, k2cat=15, Km2=500, k3cat=0.084, Km3=22,
                               k4cat=0.06, Km4=18, Km5=78, MAPKK=50, MKP3=100),
               initial_conditions={"M": 500, "Mp": 0, "Mpp": 0}, t_end=2000, n_runs=4,
               methods=["ode", "ssa"], start_in_each_state=True)
    d = out["per_omega"][0]["distribution"]
    stable = [a for a in d["ode_attractors"] if a.get("stability") == "stable"]
    assert len(stable) == 2
    assert d["readouts"][0]["species"] in ("M", "Mpp")          # not Mp
    assert "bistable" in _verdict(out)["verdict"]
    # every run stayed low, so runs are also started in each stable state: both hold
    each = out["per_omega"][0]["distribution"]["started_in_each_state"]
    assert each["n_peaks"] == 2 and each["runs_that_left_their_start_state"] == 0
    assert each["verdict"].startswith("bistable")


def test_xpp_and_ode_files_are_listed_as_models():
    """The orchestrator once refused mm_reduced.ode because the model list only held SBML."""
    pytest.importorskip("jax")
    from dynalyst.analysis.jaxkm_tools import list_available_models
    listed = json.loads(list_available_models())["xpp_ode_models_by_directory"]
    names = [f for files in listed.values() for f in files]
    assert "mm_reduced.ode" in names and "ferrell_cell_cycle_3d.ode" in names
    assert any(n.endswith("Full_model_for_SN.txt") for n in names)


@pytest.mark.skipif(not __import__("dynalyst.analysis.stochastic_ssa",
                                   fromlist=["x"]).jax_available(), reason="jax not installed")
def test_the_jax_engine_keeps_to_its_time_limit():
    from dynalyst.analysis import stochastic_ssa as eng
    model, values, _, channels = st._prepare(None, MM["species"], MM["odes"], MM["parameters"],
                                             None, 100)
    V = np.array([c["stoich"] for c in channels]).T
    grid = np.linspace(0, 100, 201)
    x0 = np.zeros(2)
    # a chunk of a few events and no time at all: the runs stop after the first chunk
    old = eng.JAX_CHUNK_EVENTS
    eng.JAX_CHUNK_EVENTS = 50
    try:
        out, info = eng.exact_jax(model, channels, values, 1000.0, V, x0, grid, 0, 4, 10**7,
                                  max_seconds=0.0)
    finally:
        eng.JAX_CHUNK_EVENTS = old
    assert info["time_limited"] == 0.0 and info["truncated"].all()
    assert np.all(info["events"] == 50)
    # unlimited, the same engine reaches t_end
    out, info = eng.exact_jax(model, channels, values, 100.0, V, x0, grid, 0, 4, 10**7)
    assert info["time_limited"] is None and not info["truncated"].any()


def test_reviewer_warns_when_the_ssa_misses_the_ode_at_large_omega():
    from dynalyst.orchestration.reviewer import Reviewer
    data = {"reactions": [{"no": 1}], "warnings": [],
            "sampled_time_series": [{"omega": 1000, "time": 0, "S_ode": 0.0}],
            "per_omega": [{"omega": 1000, "ssa": {"runs": 5},
                           "species": {"S": {"ode_final": 1.0, "ssa_vs_ode_mean_rel_diff": 0.5}}}]}
    checks = Reviewer()._review_stochastic_simulation({}, data, json.dumps(data))
    assert any(c.name == "deterministic_limit" and not c.passed for c in checks)






def test_the_wait_question_counts_system_sizes_not_calls():
    from dynalyst.orchestration.agent import _omega_progress
    from dynalyst.orchestration.approval import stochastic_wait_message
    run = lambda status: {"output": json.dumps({
        "omega_plan": {"series": [1.0, 10.0, 100.0]},
        "per_omega": [{"omega": 100.0, "method_results": {"ode": {"status": "completed"},
                                                          "ssa": {"status": status}}}]})}
    stopped_twice = _omega_progress([run("incomplete"), run("incomplete")])      # one size, two calls
    assert stopped_twice == {"finished": [], "stopped": [100.0], "not_run": [1.0, 10.0]}
    assert "stopped before t_end 100; not run 1, 10" in stochastic_wait_message(36, stopped_twice)
    assert _omega_progress([run("incomplete"), run("completed")])["finished"] == [100.0]
    assert _omega_progress([{"output": "not json"}]) == {"finished": [], "stopped": [], "not_run": []}


def test_omega_is_1_unless_given_and_a_given_omega_is_kept():
    out = json.loads(st.stochastic_simulation(**MM, t_end=2.0, n_runs=2, distribution=False))
    assert out["omega"] == [1.0] and "omega = 1" in out["omega_note"] and "per_omega" in out
    given = json.loads(st.stochastic_simulation(**MM, t_end=2.0, n_runs=2, omega=[37.0], distribution=False))
    assert given["omega"] == [37.0] and "omega_note" not in given and "omega_plan" not in given


def test_a_reference_ode_that_does_not_finish_falls_back_and_says_so(monkeypatch):
    from dynalyst.analysis import deterministic
    def slow(*a, **k):
        raise TimeoutError("the ODE passed its 30 s limit")
    monkeypatch.setattr(deterministic, "integrate", slow)
    out = json.loads(st.stochastic_simulation(**MM, t_end=2.0, n_runs=2, omega=10.0, distribution=False))
    assert out["ode_reference"]["status"] == "timed out" and "no period was compared" in out["period_note"]
    assert any("start values" in w for w in out["warnings"])


def test_the_integrator_stops_at_its_time_limit():
    from dynalyst.analysis.deterministic import integrate
    from dynalyst.analysis.model_classifier import load_any_model
    model, _ = load_any_model(None, MM["species"], MM["odes"], MM["parameters"], {"S": 1.0, "P": 0.0})
    with pytest.raises(TimeoutError):
        integrate(model, model["param_defaults"], np.linspace(0, 100, 11), max_seconds=-1.0)


def test_a_run_too_slow_for_its_limit_stops_after_the_pilot_and_concludes_nothing(monkeypatch):
    from dynalyst.analysis import stochastic_ssa as eng
    monkeypatch.setattr(eng, "SSA_PILOT_SECONDS", 0.5)
    monkeypatch.setattr(st, "SSA_STOP_FACTOR", 0.0)
    kw = dict(**MM, omega=1000.0, t_end=2000.0, n_runs=2, burn_in=0.0, methods=["ode", "ssa"], distribution=True)
    out = json.loads(st.compare_simulations(**kw))
    ssa = out["per_omega"][0]["ssa"]
    assert ssa["stopped_early"]["projected_seconds"] >= 0 and ssa["truncated_runs"] == 2
    assert "insufficient evidence" in ssa["stopped_early"]["note"] and ssa["run_status"]["truncated"] == 2
    by_method = out["per_omega"][0]["distribution"]["by_method"]       # too few samples: no verdict at all
    assert not by_method or by_method["SSA"]["verdict"].startswith("insufficient data")
    assert "projected_full_run" in ssa
    exact = json.loads(st.compare_simulations(**{**kw, "t_end": 2.0}, ssa_method="exact"))     # asked for: it runs on
    assert "stopped_early" not in exact["per_omega"][0]["ssa"]


def test_a_pace_that_can_finish_does_not_stop():
    from dynalyst.analysis import stochastic_ssa as eng
    assert eng.projected_seconds([3, 3], [100, 100], np.linspace(0, 100, 11), 300.0) == pytest.approx(1500.0)
    assert eng.projected_seconds([11, 11], [100, 100], np.linspace(0, 100, 11), 300.0) == pytest.approx(300.0)


def test_approximate_channels_are_named():
    out = json.loads(st.propensity_table(**MM))
    assert out["approximate_propensities"]["channels"] and "S*vm" in out["approximate_propensities"]["channels"][0]
    dimer = json.loads(st.propensity_table(species=DIMER["species"], odes=DIMER["odes"], parameters=DIMER["parameters"]))
    assert "approximate_propensities" not in dimer


def test_ssa_reports_periods_without_being_asked_and_the_floor_is_explained():
    osc = dict(species=["x", "y"], odes={"x": "a - x + x**2*y", "y": "b - x**2*y"}, parameters={"a": 0.1, "b": 0.5},
               initial_conditions={"x": 0.0, "y": 0.0})
    out = json.loads(st.stochastic_simulation(**osc, t_end=60.0, n_runs=2, omega=100.0, distribution=False))
    x = out["per_omega"][0]["species"]["x"]
    assert x["ode_oscillation"]["mean_period"] == pytest.approx(12.48, abs=0.6)
    assert "ssa_oscillation" in x and "ssa_oscillation" in out["period_note"]
    assert not any(k == "ode" for k in out["executed_methods"])


def _engines(system, om=1.0, runs=40, t_end=5.0):
    from dynalyst.analysis import stochastic_ssa as eng
    model, values, _, channels = st._prepare(None, system["species"], system["odes"], system["parameters"],
                                             system["initial_conditions"], t_end)
    V = np.array([c["stoich"] for c in channels]).T
    x0 = np.round(np.array([model["y0"][n] for n in model["state_names"]]) * om)
    grid = np.linspace(0.0, t_end, 21)
    a = st._propensity_fn(model, channels, values, om)
    yield "numpy", *eng.exact_numpy(a, V, x0, grid, np.random.default_rng(0), runs, 100000)
    yield "tau", *eng.tau_leap(a, V, x0, grid, np.random.default_rng(0), runs)
    if eng.jax_available():
        yield "jax", *eng.exact_jax(model, channels, values, om, V, x0, grid, 0, runs, 100000)


DIMER = dict(species=["A", "B"], odes={"A": "-2*k*A**2", "B": "k*A**2"}, parameters={"k": 1.0},
             initial_conditions={"A": 3.0, "B": 0.0})


def test_a_repeated_reactant_counts_distinct_molecules_and_never_goes_negative():
    import sympy as sp
    model, values, _, channels = st._prepare(None, DIMER["species"], DIMER["odes"], DIMER["parameters"],
                                             DIMER["initial_conditions"], 5.0)
    a = st._propensity_fn(model, channels, values, 1.0)
    assert [float(np.ravel(a(np.array([[x], [0.0]])))[0]) for x in (3.0, 1.0, 0.0)] == [6.0, 0.0, 0.0]
    for name, out, d in _engines(DIMER):
        assert np.nanmin(out) >= 0, name
        assert set(np.unique(out[:, -1, 0][np.isfinite(out[:, -1, 0])])) <= {1.0, 3.0}, name   # odd count stays odd
    # the deterministic limit stays k*A**2
    w = st.molecular_propensity(model, channels[0]["rate"], channels[0]["stoich"])
    om, A = st._omega_symbol(), model["state_symbols"][0]
    assert sp.limit(w.subs(A, sp.Symbol("c") * om) / om, om, sp.oo) == sp.Symbol("k") * sp.Symbol("c") ** 2


def test_custom_rate_laws_keep_their_form_but_cannot_take_a_count_below_zero():
    model, values, _, channels = st._prepare(None, ["X"], {"X": "v - r*X**2/(K**2 + X**2)"}, {"v": 0.5, "r": 2.0, "K": 1.0},
                                             {"X": 2.0}, 5.0)
    for ch in channels:      # a Hill law is not an elementary reaction
        assert st._repeated_reactants(model, ch["rate"], ch["stoich"]) is None
    X = model["state_symbols"][0]
    assert st._repeated_reactants(model, sp_symbol("k") * X ** 2, np.array([-1.0])) is None     # power law, one molecule used
    assert st._repeated_reactants(model, sp_symbol("k") * X ** 2, np.array([-2.0])) == {X: 2}
    removal = dict(species=["X"], odes={"X": "v - k"}, parameters={"v": 0.5, "k": 1.0}, initial_conditions={"X": 2.0})
    for name, out, d in _engines(removal, t_end=20.0):
        assert np.nanmin(out) >= 0, name
        assert d["blocked_events"] > 0, name


def test_both_exact_engines_agree_and_report_alike():
    runs = {name: (out, d) for name, out, d in _engines(DIMER, runs=300) if name != "tau"}
    if "jax" in runs:
        means = [float(np.nanmean(o[:, -1, 0])) for o, _ in runs.values()]
        assert abs(means[0] - means[1]) < 0.25
        assert set(runs["numpy"][1]) & set(runs["jax"][1]) >= {"negative_propensity_steps", "blocked_events", "time_limited"}


def test_a_search_is_reported_and_one_state_is_not_called_monostable():
    from dynalyst.analysis.stochastic_distribution import verdict
    state = [{"kind": "steady state", "stability": "stable", "readout": 1.0}]
    label, lines = verdict(1, state, 0, 5, None, "X", search={"starts_tried": 7, "starts_available": 40,
                                                              "time_limit_reached": True, "time_limit_seconds": 90})
    assert label == "one stable state found" and "7 of 40" in lines[0] and "90 s limit" in lines[0]




def test_a_period_from_noisy_runs_is_marked_unreliable():
    t = np.linspace(0, 100, 1001)
    clean = np.array([np.sin(2 * np.pi * t / 10 + k) for k in range(4)])
    good = st._period_across_runs(clean, t)
    assert good["reliable"] and good["median_period"] == pytest.approx(10.0, abs=0.2) and good["runs_with_regular_peaks"] == 4
    few = st._period_across_runs(np.array([np.sin(2 * np.pi * t / 40)]), t)      # 2 or 3 peaks: no trusted period
    assert not few["reliable"]
    assert st._period_across_runs(np.full((2, 1001), np.nan), t) == {
        "runs_finished": 0, "runs_with_regular_peaks": 0, "reliable": False, "reason": "no run finished"}


def test_a_retry_the_projection_rules_out_is_refused():
    from dynalyst.orchestration.guard import ExecutionGuard
    guard, tool = ExecutionGuard(), "stochastic_simulation"
    run = {"status": "ok", "output": json.dumps({"per_omega": [{"omega": 100.0, "method_results": {"ssa": {"details": {
        "projected_full_run": {"minutes": 523.0, "retry": "a retry cannot finish"}}}}}]})}
    a = {"model_name_or_path": "m", "omega": 100, "n_runs": 10, "t_end": 500}
    guard.record(tool, a, run)
    same = guard.check("stochastic", tool, {**a, "seed": 42})
    assert not same.allowed and "523" in same.reason
    assert guard.check("stochastic", tool, {**a, "n_runs": 2}).allowed          # asks for less
    assert guard.check("stochastic", tool, {**a, "ssa_method": "tau_leap"}).allowed
    assert guard.check("stochastic", tool, {**a, "omega": 1}).allowed






def test_each_omega_gets_one_run_and_one_retry():
    from dynalyst.orchestration.guard import ExecutionGuard
    guard, tool = ExecutionGuard(), "stochastic_simulation"
    ran = lambda *sizes: {"status": "ok", "output": json.dumps({"per_omega": [{"omega": o} for o in sizes]})}
    guard.record(tool, {"model_name_or_path": "m", "omega": 100, "n_runs": 8}, ran(100.0))
    assert guard.check("stochastic", tool, {"model_name_or_path": "m", "omega": 100, "n_runs": 2}).allowed
    guard.record(tool, {"model_name_or_path": "m", "omega": 100, "n_runs": 2}, ran(100.0))
    third = guard.check("stochastic", tool, {"model_name_or_path": "m.xml", "omega": [10, 100], "n_runs": 1})
    assert not third.allowed and "100.0" in third.reason
    assert guard.check("stochastic", tool, {"model_name_or_path": "m", "omega": [1, 10]}).allowed
    assert guard.check("stochastic", tool, {"model_name_or_path": "other", "omega": 100}).allowed


def test_a_stopped_run_projects_its_full_time_from_its_own_event_rate():
    from dynalyst.analysis.stochastic import _projected_run
    grid = np.linspace(0.0, 100.0, 11)
    out = np.full((2, 11, 1), np.nan)
    out[:, :3] = 1.0                           # both runs reached t = 20 of 100
    full = _projected_run(out, grid, [1000, 1000], 300.0, 300.0)
    assert full["approximate"] and full["seconds"] == 1500 and full["events_per_run"] == 5000
    assert "cannot finish" in full["retry"]
    assert "may finish" in _projected_run(out, grid, [1000, 1000], 30.0, 300.0)["retry"]
    out[:, :] = np.nan
    out[:, 0] = 1.0                            # no point past the start: never a division by zero
    assert np.isfinite(_projected_run(out, grid, [10, 10], 1.0, 300.0)["seconds"])


def test_an_ssa_without_the_odes_points_to_compare_simulations_for_periods():
    out = json.loads(st.stochastic_simulation(**MM, t_end=2.0, n_runs=2, omega=10.0, distribution=False))
    assert "ssa_oscillation" in out["period_note"] and out["ode_reference"]["status"] == "succeeded"
    both = json.loads(st.compare_simulations(**MM, t_end=2.0, n_runs=2, omega=10.0, distribution=False,
                                             methods=["ode", "ssa"]))
    assert "period_note" not in both


def test_the_exact_engine_says_it_stopped_at_its_time_limit():
    from dynalyst.analysis.stochastic_ssa import exact_numpy
    run = lambda limit: exact_numpy(lambda x: x * 1.0, np.array([[-1.0]]), np.array([20.0]),
                                    np.linspace(0, 1, 3), np.random.default_rng(0), 2, 10_000,
                                    max_seconds=limit)[1]
    stopped = run(-1.0)
    assert stopped["time_limited"] == -1.0 and stopped["truncated"].all()
    assert run(60.0)["time_limited"] is None


def test_tau_leaping_stops_at_its_time_limit():
    from dynalyst.analysis.stochastic_ssa import tau_leap
    out, d = tau_leap(lambda x: x * 1.0, np.array([[-1.0]]), np.array([1000.0]), np.linspace(0, 1, 3),
                      np.random.default_rng(0), 2, max_seconds=-1.0)
    assert d["time_limited"] == -1.0 and d["truncated"].all() and np.isnan(out[:, 1:]).all()
    _, d = tau_leap(lambda x: x * 1.0, np.array([[-1.0]]), np.array([1000.0]), np.linspace(0, 1, 3),
                    np.random.default_rng(0), 2, max_seconds=60.0)
    assert d["time_limited"] is None and not d["truncated"].any()


def _stochastic_agent(monkeypatch, *llm_replies):
    from dynalyst import config
    from dynalyst.orchestration import tools as registry
    from dynalyst.orchestration.agent import Agent
    from dynalyst.orchestration.state import SessionState
    from tests.fakes import call, say, scripted, tools, via_teams
    monkeypatch.setattr(config, "STOCHASTIC_ASK_SECONDS", 0.0)
    real = registry.execute
    monkeypatch.setattr(registry, "execute", lambda name, args: (
        json.dumps({"status": "completed", "omega": args.get("omega"),
                    "omega_plan": {"calls_needed": 4, "series": [1.0, 10.0, 100.0, 1000.0]},
                    "per_omega": [{"omega": 1.0, "method_results": {"ssa": {"status": "completed"}}}]})
        if name == "stochastic_simulation" else real(name, args)))
    run = call("stochastic_simulation", model_name_or_path="Markevich2004.xml", t_end=1, omega=[1])
    sub = scripted(tools(run), say("Ran one."))
    orch = scripted(tools(call("delegate_to_stochastic", task="SSA at several sizes")), *llm_replies)
    return Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}), verbose=False, save_records=False,
                 state=SessionState(persist=False), approval="ask")


def test_a_long_stochastic_series_asks_the_person_chatting(monkeypatch):
    agent = _stochastic_agent(monkeypatch, *[])
    rec = agent.run("Compare the system sizes")
    assert rec.awaiting_approval and "more time" in rec.final_answer and "**no**" in rec.final_answer
    assert agent.state["pending_approval"]["kind"] == "stochastic_wait"


def test_runs_that_move_between_ode_states_are_switching_even_with_one_peak():
    from dynalyst.analysis import stochastic_distribution as dist
    states = [{"kind": "steady state", "stability": "stable", "readout": x} for x in (0.4, 4.8)]
    moved = {"switches_total": 3, "runs_visiting_more_than_one_state": 2, "mean_residence_time": 500.0}
    label, lines = dist.verdict(1, states, 0, 10, moved, "Mpp")
    assert label == "bistable with switching" and "2 of 10 runs" in lines[0]
    assert dist.verdict(1, states, 0, 10, None, "Mpp")[0].startswith("bistable (deterministic only")


def test_an_omega_budget_counts_per_start_state():
    from dynalyst.orchestration import guard
    low, high = ({"model_name_or_path": "m", "omega": 1.0, "initial_conditions": {"M": n}} for n in (500, 0))
    assert guard._omega_key(low, 1.0) != guard._omega_key(high, 1.0)
    assert guard._omega_key(low, 1.0) == guard._omega_key(dict(low), 1.0)


def test_a_stopped_run_still_gives_a_period_up_to_where_it_stopped():
    grid = np.linspace(0, 100, 1001)
    run = np.sin(2 * np.pi * grid / 5) + 2
    run[800:] = np.nan
    found = st._period_across_runs(np.vstack([run, run, run]), grid)
    assert found["reliable"] and found["runs_finished"] == 0 and found["runs_used"] == 3
    assert abs(found["median_period"] - 5) < 0.1


def test_omega_and_t_end_are_the_users_and_runs_count_per_parameter_value():
    from dynalyst.orchestration import guard
    from dynalyst.orchestration.agent import _user_values_only
    text = "Run SSA of the Markevich model with omega 5 at MKP3 = 70 and 150 to t = 400"
    args = {"model_name_or_path": "m", "omega": 5, "t_end": 400, "parameters": {"MKP3": 70}}
    assert _user_values_only(args, text) == args                              # the user's values stay
    picked = _user_values_only({**args, "omega": [10], "t_end": 15}, text)
    assert "omega" not in picked and "t_end" not in picked                    # the model's choices go
    assert "t_end" not in _user_values_only({"t_end": "long"}, text)
    # a number the user assigned to a parameter is not a setting
    t2 = "Gillespie run of Markevich2004 with MKP3 = 100"
    assert _user_values_only({"t_end": 100, "omega": 100}, t2) == {}
    t3 = "Simulate for 500 time units at omega = 1, 10 and 100 with MKP3 = 100"
    assert _user_values_only({"t_end": 500, "omega": [1, 10, 100]}, t3) == {"t_end": 500, "omega": [1, 10, 100]}
    assert _user_values_only({"t_end": 2000}, "run it to 2000") == {"t_end": 2000}
    at = lambda v: {"model_name_or_path": "m", "omega": 5, "parameters": {"MKP3": v}}
    assert guard._omega_key(at(70), 5.0) != guard._omega_key(at(150), 5.0)


def test_a_retyped_steady_state_runs_as_find_steady_states_computed_it():
    from dynalyst.orchestration.agent import _exact_states
    found = {"tool": "find_steady_states", "status": "ok", "arguments": {"model_name_or_path": "BIOMD0000000027"},
             "output": json.dumps({"steady_states": [
                 {"state": {"M": 437.7299837211522, "Mp": 12.852481482899327, "Mpp": 49.41753479594844}},
                 {"state": {"M": 19.545, "Mp": 8.3944, "Mpp": 472.0606}}]})}
    call = lambda ic: {"model_name_or_path": "BIOMD0000000027.xml", "initial_conditions": ic}
    out = _exact_states(call({"M": 437.73, "Mp": 12.852, "Mpp": 49.418}), [found])
    assert out["initial_conditions"]["M"] == 437.7299837211522
    assert _exact_states(call({"M": 19.55, "Mp": 8.39, "Mpp": 472.06}), [found])["initial_conditions"]["Mp"] == 8.3944
    mine = call({"M": 400.0, "Mp": 50.0, "Mpp": 50.0})                     # the user's own start
    assert _exact_states(mine, [found]) == mine
    other = {**call({"M": 437.73, "Mp": 12.852, "Mpp": 49.418}), "model_name_or_path": "other.xml"}
    assert _exact_states(other, [found]) == other
