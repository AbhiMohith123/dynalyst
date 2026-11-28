"""Method isolation, canonical inputs and honest comparison reporting."""
import json
from pathlib import Path

import numpy as np
import pytest

from dynalyst import config
from dynalyst.analysis import stochastic as st
from tests.fakes import via_teams

MODEL = dict(species=["X"], odes={"X": "birth-death*X"},
             parameters={"birth": 2, "death": 1}, initial_conditions={"X": 1})
RUN = dict(t_end=3, n_runs=4, n_points=41, omega=[10], seed=12)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PLOTS_DIR", tmp_path)
    return tmp_path


def forbidden(*args, **kwargs):
    raise AssertionError("An unrequested numerical method ran.")


def test_agents_have_disjoint_simulation_ownership():
    from dynalyst.orchestration.subagents import SUBAGENTS, owner_of
    from dynalyst.orchestration.teams import team_of
    assert owner_of("simulate_model") == "simulation"
    for tool in ("stochastic_simulation", "propensity_table", "compare_simulations"):
        assert owner_of(tool) == "stochastic"
    assert not set(SUBAGENTS["simulation"].objective_tools) & set(SUBAGENTS["stochastic"].objective_tools)
    assert team_of("stochastic") == "dynamics_team"


@pytest.mark.parametrize("methods,method_key", [(None, "ssa"), (["cle"], "cle")])
def test_stochastic_only_never_runs_ode_or_attractors(isolated, monkeypatch, methods, method_key):
    from dynalyst.analysis import deterministic, stochastic_distribution
    monkeypatch.setattr(st, "_ode", forbidden)
    monkeypatch.setattr(deterministic, "integrate", forbidden)
    monkeypatch.setattr(stochastic_distribution, "ode_attractors", forbidden)
    if method_key == "ssa":
        monkeypatch.setattr(st, "cle_runs", forbidden)
    else:
        monkeypatch.setattr(st, "_run_ssa", forbidden)
    out = json.loads(st.stochastic_simulation(**MODEL, **RUN, methods=methods))
    assert "error" not in out, out
    assert out["requested_methods"] == out["executed_methods"] == [method_key]
    assert out["status"] == "completed"
    p = out["per_omega"][0]
    assert "ode_final" not in p["species"]["X"]
    assert not p["distribution"]["deterministic_stability_assessed"]
    assert p["distribution"]["by_method"][method_key.upper()]["verdict"].startswith("empirical")
    assert p["method_results"][method_key]["backend"] in ("NumPy", "JAX")


def test_propensity_table_never_simulates(monkeypatch):
    monkeypatch.setattr(st, "_ode", forbidden)
    out = json.loads(st.propensity_table(species=MODEL["species"], odes=MODEL["odes"],
                                          parameters=MODEL["parameters"]))
    assert out["n_reactions"] == 2
    assert out["stochastic_model"]["channel_sources"] == ["inferred from ODE terms"]


def test_comparison_is_explicit_and_preserves_inputs(isolated, monkeypatch):
    original = st._ode
    calls = []
    def reference(model, values, t_end, grid):
        calls.append((model, values))
        return original(model, values, t_end, grid)
    monkeypatch.setattr(st, "_ode", reference)
    monkeypatch.setattr(st, "cle_runs", forbidden)
    out = json.loads(st.compare_simulations(**MODEL, **RUN, distribution=False))
    assert "error" not in out, out
    assert len(calls) == 1
    assert out["executed_methods"] == ["ode", "ssa"]
    assert out["model_inputs"]["initial_conditions"] == {"X": 1}
    assert out["model_inputs"]["parameters"] == MODEL["parameters"]
    assert out["per_omega"][0]["method_results"]["ode"]["backend"] == "SciPy LSODA"
    expected = 2 - np.exp(-RUN["t_end"])
    assert out["per_omega"][0]["species"]["X"]["ode_final"] == pytest.approx(expected, rel=1e-6)


def test_failed_ode_reference_does_not_prevent_ssa(isolated, monkeypatch):
    def broken(*a):
        raise ValueError("reference failed")
    monkeypatch.setattr(st, "_ode", broken)
    monkeypatch.setattr(st, "_reference_ode", lambda *a: (None, {"status": "failed", "reason": "reference failed"}))
    out = json.loads(st.compare_simulations(**MODEL, **RUN))
    assert out["status"] == "partial"
    assert out["executed_methods"] == ["ssa"]
    assert out["per_omega"][0]["method_results"]["ode"]["reason"] == "reference failed"
    assert not out["per_omega"][0]["distribution"]["deterministic_stability_assessed"]


@pytest.mark.parametrize("methods", [["ode"], ["ode", "ssa"], [], ["bogus"]])
def test_stochastic_tool_refuses_cross_method_arguments(methods):
    assert "error" in json.loads(st.stochastic_simulation(**MODEL, methods=methods))


@pytest.mark.parametrize("methods", [["ssa"], ["ode"], []])
def test_comparison_requires_both_kinds(methods):
    assert "error" in json.loads(st.compare_simulations(**MODEL, methods=methods))


def test_no_approximate_substitution_for_invalid_ssa():
    out = json.loads(st.stochastic_simulation(species=["A", "B"],
        odes={"A": "-A", "B": ".7*A"}, initial_conditions={"A": 1}, **RUN))
    assert "integer stoichiometry" in out["error"]


def test_ssa_export_contains_only_requested_methods(isolated):
    out = json.loads(st.stochastic_simulation(**MODEL, **RUN, export_matlab=True, distribution=False))
    assert "error" not in out, out
    script = Path(out["matlab_script"]).read_text()
    assert "Gillespie direct" in script
    assert "ode45" not in script and "randn" not in script
    assert out["matlab_methods"] == ["ssa"]


def test_shared_model_relative_overrides(tmp_path):
    from dynalyst.analysis.simulation_model import load_model, provenance
    from dynalyst.analysis.jaxkm_tools import _get_trajectory
    path = tmp_path / "decay.ode"
    path.write_text("par k=2\nx'=-k*x\ninit x=3\ndone")
    m, values, _ = load_model(path, parameters={"k": "/2"}, initial_conditions={"x": "*2"})
    df, names, _, src = _get_trajectory(str(path), 1, 21, {"x": "*2"}, {"k": "/2"})
    assert names == ["x"] and m["y0"] == {"x": 6}
    assert src["model_inputs"]["model_signature"] == provenance(m, values)["model_signature"]
    assert df.x.iloc[-1] == pytest.approx(6*np.exp(-1), rel=1e-6)


@pytest.mark.parametrize("omega", [[0], [float("nan")], [float("inf")]])
def test_invalid_count_scale_is_rejected(omega):
    out = json.loads(st.stochastic_simulation(**MODEL, omega=omega, t_end=1))
    assert "error" in out


def test_failed_ssa_keeps_requested_deterministic_result(isolated, monkeypatch):
    def failed(*a):
        raise ValueError("invalid propensity")
    monkeypatch.setattr(st, "_run_ssa", failed)
    out = json.loads(st.compare_simulations(**MODEL, **RUN, distribution=False))
    assert out["status"] == "partial" and out["executed_methods"] == ["ode"]
    assert out["per_omega"][0]["method_results"]["ssa"]["status"] == "failed"


def test_failed_cle_keeps_other_requested_results(isolated, monkeypatch):
    def failed(*a):
        raise ValueError("non-finite CLE rate")
    monkeypatch.setattr(st, "cle_runs", failed)
    out = json.loads(st.compare_simulations(**MODEL, **RUN, methods=["ode", "ssa", "cle"],
                                            distribution=False))
    assert out["status"] == "partial" and out["executed_methods"] == ["ode", "ssa"]
    assert out["per_omega"][0]["method_results"]["cle"]["status"] == "failed"


def test_cle_nonfinite_rates_are_errors():
    with pytest.raises(ValueError, match="Non-finite CLE"):
        st.cle_runs(lambda x: np.full((1, 2), np.nan), np.array([[-1]]), [1],
                    np.linspace(0, 1, 5), np.random.default_rng(0), 2, .1)


def test_tau_leaping_is_identified_and_not_exported_as_exact_ssa(isolated):
    out = json.loads(st.stochastic_simulation(**MODEL, **RUN, ssa_method="tau_leap",
                                              export_matlab=True))
    assert "error" not in out, out
    info = out["per_omega"][0]["method_results"]["ssa"]
    assert info["method"] == "tau-leaping" and info["approximate"]
    assert "TAU-LEAPING" in out["per_omega"][0]["distribution"]["by_method"]
    assert "matlab_script" not in out
    assert any("tau-leaping export" in w for w in out["warnings"])


def test_cle_followup_preserves_method():
    from dynalyst.orchestration.next_steps import _suggest_for
    step = {"tool": "stochastic_simulation", "arguments": {"model_name_or_path": "model.xml"},
            "output": json.dumps({"model": "model.xml", "omega": [10],
                                  "methods": ["cle"], "requested_methods": ["cle"]})}
    offers = _suggest_for(step, set())
    query = next(q for label, q in offers if "system sizes" in label)
    assert "CLE" in query and "Gillespie" not in query


def test_sbml_volume_is_preserved_in_reaction_channels(tmp_path, isolated):
    import libsbml
    from tests.test_audit_fixes import sbml, param
    doc, m = sbml(volume=2, value=4)
    param(m, "k", 1)
    reaction = m.createReaction()
    reaction.setId("decay"); reaction.setReversible(False)
    sr = reaction.createReactant()
    sr.setSpecies("X"); sr.setStoichiometry(1); sr.setConstant(True)
    reaction.createKineticLaw().setMath(libsbml.parseL3Formula("k*X"))
    path = tmp_path / "volume.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    out = json.loads(st.compare_simulations(model_name_or_path=str(path), **RUN, distribution=False))
    assert "error" not in out, out
    assert out["stochastic_model"]["channel_sources"] == ["SBML reaction"]
    assert out["model_inputs"]["compartments"] == {"cell": 2}
    assert out["per_omega"][0]["species"]["X"]["ode_final"] == pytest.approx(4*np.exp(-1.5), rel=1e-6)


def test_dynamic_parameter_is_deterministic_state_not_molecule_count(tmp_path, isolated):
    import libsbml
    from tests.test_audit_fixes import sbml, param, rate
    from dynalyst.analysis.jaxkm_tools import simulate_model
    doc, m = sbml()
    param(m, "p", 1, False)
    rate(m, "X", "-p*X"); rate(m, "p", "1")
    path = tmp_path / "dynamic.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    out = json.loads(st.stochastic_simulation(model_name_or_path=str(path), **RUN))
    assert "molecule-count" in out["error"]
    deterministic = json.loads(simulate_model(str(path), t_end=1, n_points=21))
    assert "error" not in deterministic, deterministic
    assert deterministic["final_concentrations"]["X"] == pytest.approx(np.exp(-1.5), rel=1e-6)


@pytest.mark.parametrize("tool,query", [
    ("stochastic_simulation", "Run Gillespie SSA"),
    ("compare_simulations", "Compare deterministic ODE and Gillespie SSA"),
])
def test_scripted_agent_routes_the_new_stochastic_owner(isolated, tool, query):
    from dynalyst.orchestration.agent import Agent
    from tests.fakes import scripted, tools, call, say
    task = query + " for mm_reduced.ode with omega=10, n_runs=3 and t_end=2."
    run = call(tool, model_name_or_path="mm_reduced.ode", omega=[10], n_runs=3, t_end=2, distribution=False)
    sub = scripted(tools(run))
    supervisor = scripted(tools(call("delegate_to_stochastic", task=task)),
                          say("The requested simulations completed."))
    rec = Agent(save_records=False,
                llms=via_teams({"subagent": sub, "orchestrator": supervisor})).run(task)
    steps = [s for s in rec.steps if s["kind"] == "delegation"]
    assert [s["tool"] for s in steps] == [tool]
    assert steps[0]["status"] == "ok", steps[0]
    result = json.loads(steps[0]["output"])
    assert result["executed_methods"] == (["ssa"] if tool == "stochastic_simulation" else ["ode", "ssa"])
