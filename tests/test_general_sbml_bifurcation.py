"""General SBML models (not the A/APP family) through both bifurcation
tools and the agent. Elowitz_Nature2000 (the repressilator) exercises what
used to break: compartment symbols in kinetic laws, 8 state variables (AUTO
writes only 6 to its branch table), initial assignments, an observable-only
assignment rule, fractional Hill exponents, and no obvious bifurcation
parameter."""
import contextlib
import io
import json

import pytest

from dynalyst.orchestration.agent import Agent
from tests.fakes import call, say, scripted, tools, via_teams

ELOWITZ = "Elowitz_Nature2000.xml"


def test_parser_substitutes_compartments_and_keeps_only_dynamic_parameters():
    from dynalyst import config
    from dynalyst.regime.parser import parse_sbml
    p = parse_sbml(str(config.LOCAL_SBML_MODELS_DIR / ELOWITZ))
    free = set().union(*(e.free_symbols for e in p["ode_rhs"].values()))
    assert "cell" not in {str(x) for x in free}
    assert "n_Hill" in p["params"] and not any(x.startswith("init_") for x in p["params"])
    assert "background" not in p["params"]  # used only by the fluorescence observable
    assert p["initial_guess"]["X_protein"] == pytest.approx(30.8087735629587)  # initialAssignment


def test_observable_assignment_rule_is_ignored_by_generic_reader():
    from dynalyst.analysis.generic_sbml import load_generic_sbml
    m = load_generic_sbml(ELOWITZ)
    assert m["ignored_observables"] == ["fluorescence"] and len(m["dynamic_species"]) == 8


def test_display_path_outside_project(tmp_path):
    from dynalyst import config
    assert config.display_path(tmp_path / "x.png") == (tmp_path / "x.png").resolve().as_posix()
    assert config.display_path(config.PROJECT_ROOT / "outputs" / "a.png") == "outputs/a.png"


@pytest.mark.auto
def test_repressilator_hopf_with_full_8d_state():
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    out = json.loads(run_bifurcation_analysis(ELOWITZ, param_name="n_Hill", param_min=1, param_max=20,
                                              show_auto_log=False, _cycles=False))
    details = out["bifurcation_points"]
    assert [d["type"] for d in details] == ["HB", "HB"]
    assert sorted(1.4 < d["n_Hill"] < 1.6 or 3.0 < d["n_Hill"] < 3.3 for d in details) == [True, True]
    assert all(len(d["full_state"]) == 8 for d in details)


@pytest.mark.auto
def test_agent_runs_general_sbml_bifurcation_and_reviewer_verifies_hopf():
    sub = scripted(tools(call("run_bifurcation_analysis", model_name_or_path=ELOWITZ)))
    orch = scripted(tools(call("delegate_to_bifurcation", task="Bifurcation analysis of Elowitz")),
                    say("A Hopf bifurcation was found."))
    with contextlib.redirect_stdout(io.StringIO()):
        rec = Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}),
                    save_records=False).run("Run bifurcation analysis of Elowitz_Nature2000.xml")
    step = [s for s in rec.steps if s["kind"] == "delegation"][0]
    assert step["status"] == "ok", step["reviewer"]
    assert any(c["check"] == "verify_HB" and c["passed"] for c in step["reviewer"]["checks"])


# Garde2020 writes every rate law as an SBML <functionDefinition> call, which
# used to reach the Fortran printer as an unknown function
# (PrintMethodNotImplementedError) and was refused by the generic reader.

def test_function_definitions_are_expanded_in_both_readers():
    import sympy as sp
    from dynalyst import config
    from dynalyst.analysis.generic_sbml import load_generic_sbml
    from dynalyst.regime.parser import parse_sbml
    p = parse_sbml(str(config.LOCAL_SBML_MODELS_DIR / "Garde2020.xml"))
    from sympy.core.function import AppliedUndef
    assert not any(e.atoms(AppliedUndef) for e in p["ode_rhs"].values())  # no unknown f(...)
    Gp, Ap, Ge, k1, k2, k4 = sp.symbols("Gp Ap Ge k1 k2 k4")
    assert sp.simplify(p["ode_rhs"]["Gp"] - Gp * (Ge * k1 - k4 - k2 * Ap)) == 0
    assert len(load_generic_sbml("Garde2020")["dynamic_species"]) == 6


@pytest.mark.auto
def test_garde_hopf_in_k1_and_transcritical_in_ge():
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    details = json.loads(run_bifurcation_analysis("Garde2020.xml", param_name="k1", show_auto_log=False,
                                                  _cycles=False))["bifurcation_points"]
    hb = [d for d in details if d["type"] == "HB"]
    assert len(hb) == 1 and 0.3 < hb[0]["k1"] < 0.34
    # the transcritical point Ge*k1 = k4 (Ge = 30, k4 = 2) lies in k1 as well
    assert [d["k1"] for d in details if d["type"] == "BP"] == [pytest.approx(2.0 / 30, rel=1e-3)]
    assert {d["type"] for d in details} == {"HB", "BP"}
    d = json.loads(run_bifurcation_analysis("Garde2020.xml", param_name="Ge", show_auto_log=False))
    bp = [p for p in d["bifurcation_points"] if p["type"] == "BP"]
    assert bp and bp[0]["Ge"] == pytest.approx(2.0 / 0.34, rel=1e-4)  # Ge*k1 = k4


@pytest.mark.auto
def test_names_that_clash_in_fortran_are_renamed():
    """Fortran is case-insensitive and reserves F, T, U, PAR...; model names
    are mapped to U(i)/PAR(i) before any Fortran is written."""
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    d = json.loads(run_bifurcation_analysis(
        species=["x", "F"], odes={"x": "T - Km*x + x**2*F/(KM + x)", "F": "Km - x*F"},
        parameters={"T": 1.0, "Km": 2.0, "KM": 0.5}, param_name="T", param_min=0.1,
        param_max=10, show_auto_log=False))
    assert d.get("error") is None and d["bifurcation_points"]


def test_leftover_symbol_is_reported_before_fortran():
    import sympy as sp
    from dynalyst.fortran import FortranError, auto_symbols, fortran_assign
    x, k, cell = sp.symbols("x k cell")
    sub = auto_symbols(["x"], ["k"])
    assert "U(1)" in fortran_assign((k * x).xreplace(sub), "F(1)", "dx/dt")
    with pytest.raises(FortranError, match="cell"):
        fortran_assign((cell * k * x).xreplace(sub), "F(1)", "dx/dt")
