"""General rules behind the agent's robustness: typed equations in any common
form, exact conservation laws, start states found independently of units,
model provenance, approval, argument coercion, the final-answer number check,
and the supervisor's handling of plans and teams. Placeholder models only."""
import json

import numpy as np
import pytest
import sympy as sp

from dynalyst import config
from dynalyst.analysis.ode_model import normalise_odes, resolve_model


def _write_sbml(path, model_id, model_name, species=("x",), rate="k*x", k=1.0):
    """A one-reaction SBML file: nothing -> species[0] at `rate`."""
    import libsbml
    doc = libsbml.SBMLDocument(3, 1)
    m = doc.createModel()
    m.setId(model_id)
    m.setName(model_name)
    c = m.createCompartment()
    c.setId("cell"), c.setSize(1.0), c.setConstant(True)
    for s in species:
        sp_ = m.createSpecies()
        sp_.setId(s), sp_.setCompartment("cell"), sp_.setInitialConcentration(0.5)
        sp_.setHasOnlySubstanceUnits(False), sp_.setBoundaryCondition(False), sp_.setConstant(False)
    p = m.createParameter()
    p.setId("k"), p.setValue(k), p.setConstant(True)
    r = m.createReaction()
    r.setId("r1"), r.setReversible(False), r.setFast(False)
    prod = r.createProduct()
    prod.setSpecies(species[0]), prod.setStoichiometry(1.0), prod.setConstant(True)
    r.createKineticLaw().setMath(libsbml.parseL3Formula(rate))
    libsbml.writeSBMLToFile(doc, str(path))
    return path


# -- typed equations -----------------------------------------------------------------

def test_derivative_keys_name_the_states():
    m = resolve_model(odes={"dx/dt": "a - x*y", "y'": "x - b*y", "d[z]/dt": "-z"})
    assert m["state_names"] == ["x", "y", "z"] and m["param_names"] == ["a", "b"]


def test_a_declared_state_named_like_a_derivative_is_kept():
    assert normalise_odes(["dx"], {"dx": "1"}) == (["dx"], {"dx": "1"})


def test_two_right_hand_sides_for_one_state_are_an_error():
    with pytest.raises(ValueError, match="Two right-hand sides"):
        normalise_odes(None, {"x": "1", "dx/dt": "2"})


# -- equations typed in a request ------------------------------------------------------

@pytest.mark.parametrize("text, odes, params, start", [
    ("Diagram in a: dX/dt = a + X**2/(1 + X**2) - b*X, with a = 0.05, b = 0.45 and X(0) = 0",
     {"X": "a + X**2/(1 + X**2) - b*X"}, {"a": 0.05, "b": 0.45}, {"X": 0.0}),
    ("x' = y\ny' = -x + mu*(1 - x^2)*y\nmu = 2; x0 = 1",
     {"x": "y", "y": "-x + mu*(1 - x**2)*y"}, {"mu": 2.0}, {"x": 1.0, "y": 0.0}),
    ("Is it bistable? d[A]/dt = k*A^2 - A; k = 3", {"A": "k*A**2 - A"}, {"k": 3.0}, {"A": 0.0}),
])
def test_typed_equations_are_read_by_code(text, odes, params, start):
    from dynalyst.analysis.typed_equations import parse_typed_system
    r = parse_typed_system(text)
    assert r["odes"] == odes and r["parameters"] == params and r["initial_conditions"] == start
    assert r["unvalued"] == []


def test_typed_equations_report_missing_values_and_ignore_plain_requests():
    from dynalyst.analysis.typed_equations import parse_typed_system
    assert parse_typed_system("What is the steady state of dX/dt = a - X?")["unvalued"] == ["a"]
    assert parse_typed_system("Run a bifurcation analysis of model.xml in k1 from 1 to 10") is None
    assert parse_typed_system("dX/dt = a - X, a = 1, a = 2") is None     # two values for a


def test_a_typed_system_is_saved_once_by_content(tmp_path, monkeypatch):
    from dynalyst.analysis.typed_equations import parse_typed_system, save_typed_system
    monkeypatch.setattr(config, "LOCAL_SBML_MODELS_DIR", tmp_path)
    system = parse_typed_system("dX/dt = k1 - k2*X with k1 = 10, k2 = 0.1, X(0) = 0")
    first, again = save_typed_system(system), save_typed_system(system)
    assert first["file"] == again["file"] and first["file"].startswith("typed_")
    m = resolve_model(first["path"])
    assert m["state_names"] == ["X"] and m["param_defaults"] == {"k1": 10.0, "k2": 0.1}
    other = save_typed_system(parse_typed_system("dX/dt = k1 - k2*X with k1 = 5, k2 = 0.1"))
    assert other["file"] != first["file"]
    # powers and rational terms are written as floats in the file (X**2.0): still the same
    hill = save_typed_system(parse_typed_system(
        "dX/dt = a + X**2/(1 + X**2) - b*X, with a = 0.05, b = 0.45 and X(0) = 0"))
    assert "error" not in hill


def test_an_outward_spiral_is_not_called_a_plain_saddle():
    from dynalyst.analysis.steady_states import classify_stability, spirals_outward
    ev = np.array([0.23 + 0.6j, 0.23 - 0.6j, -0.85 + 0j])
    assert classify_stability(ev).startswith("unstable_focus_saddle") and spirals_outward(ev)
    assert classify_stability(np.array([0.5, -1.0])).startswith("saddle")
    assert not spirals_outward(np.array([-0.2 + 1j, -0.2 - 1j]))


def test_typed_files_are_marked_so_negative_values_are_allowed(tmp_path, monkeypatch):
    from dynalyst.analysis.typed_equations import (is_typed_file, parse_typed_system,
                                                   save_typed_system)
    monkeypatch.setattr(config, "LOCAL_SBML_MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "SBML_SEARCH_DIRS", [tmp_path])
    saved = save_typed_system(parse_typed_system("x' = y, y' = -x with x(0) = 1"))
    assert is_typed_file(saved["file"])
    _write_sbml(tmp_path / "plain.xml", "plain", "plain")
    assert not is_typed_file("plain.xml")


@pytest.mark.auto
def test_a_range_without_the_model_value_is_reached_from_it(tmp_path, monkeypatch):
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    from dynalyst.analysis.typed_equations import parse_typed_system, save_typed_system
    monkeypatch.setattr(config, "LOCAL_SBML_MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "SBML_SEARCH_DIRS", [tmp_path])
    f = save_typed_system(parse_typed_system(
        "dX/dt = a + X**2/(1 + X**2) - b*X, with a = 0.05, b = 0.45 and X(0) = 0"))["file"]
    d = json.loads(run_bifurcation_analysis(model_name_or_path=f, param_name="b",
                                            param_min=0.46, param_max=0.7))
    # the middle (unstable) branch leaves the range at b = 0.46: the range grows to show the
    # fold that ends it, so both folds of the S-curve are in the one diagram
    assert "range_widened" in d and d["swept_range"][0] < 0.4352 and d["swept_range"][1] == pytest.approx(0.7, abs=0.05)
    assert sorted(p["b"] for p in d["bifurcation_points"] if p["type"] == "LP") ==         [pytest.approx(0.4352, abs=1e-3), pytest.approx(0.5528, abs=1e-3)]


# -- exact conservation laws -----------------------------------------------------------

def test_conservation_laws_are_exact():
    from dynalyst.analysis.steady_states import conservation_reduce, exact_conservation_laws, law_text
    m = resolve_model(odes={"a": "-k1*a + k2*b", "b": "3*k1*a - 3*k2*b", "c": "s - d*c"},
                      parameters={"k1": 0.1, "k2": 0.7, "s": 1, "d": 0.3},
                      initial_conditions={"a": 0.1, "b": 0.2, "c": 0})
    laws = exact_conservation_laws(m)
    assert [law_text(c, m["state_names"]) for c in laws] == ["3*a + b"]
    red = conservation_reduce(m)
    (name, expr), = red["eliminated"]
    # 3a + b = 3*0.1 + 0.2 = 1/2 exactly, not 0.5000000000000001; the last state of
    # the law is the one eliminated
    assert name == "b" and sp.simplify(expr - (sp.Rational(1, 2) - 3 * sp.Symbol("a"))) == 0
    assert "conservation_note" not in red


def test_a_law_hidden_by_a_merged_fraction_is_confirmed_exactly():
    from dynalyst.analysis.steady_states import exact_conservation_laws, law_is_exact
    m = resolve_model(odes={"x": "-V*x/(K + x) + k*y", "y": "V/(1 + K/x) - k*y"},
                      parameters={"V": 1, "K": 0.5, "k": 0.2}, initial_conditions={"x": 1, "y": 0})
    laws = exact_conservation_laws(m)
    assert len(laws) == 1 and law_is_exact(laws[0]) and list(laws[0]) == [1, 1]


def test_exact_integers_beyond_fortran_integers_are_written_as_doubles():
    """Exact arithmetic can leave integers like 10**15 in a Jacobian; as Fortran
    integer literals they overflow and the equations file does not compile."""
    from dynalyst.fortran import fortran_assign
    u = sp.Symbol("U(1)")
    line = fortran_assign(10 ** 15 * u + 2 * u ** 2, "F(1)", "test")
    assert "1000000000000000.0d0*U(1)" in line and "U(1)**2" in line


# -- start states and steady states -------------------------------------------------------

def test_start_state_is_judged_relative_to_the_rates():
    """In very small units every rate is tiny: x = 0 is not an equilibrium."""
    from dynalyst.analysis.bifurcation_auto import _find_start_state
    m = resolve_model(odes={"x": "k1*S - k2*x"}, parameters={"k1": 1.0, "S": 1e-9, "k2": 1.0},
                      initial_conditions={"x": 0})
    y, _ = _find_start_state(m, "S", 1e-9, {})
    assert y[0] == pytest.approx(1e-9, rel=1e-8)


def test_no_equilibrium_is_reported_without_hanging():
    from dynalyst.analysis.bifurcation_auto import _find_start_state
    m = resolve_model(odes={"x": "-x**2 - p"}, parameters={"p": 1.0}, initial_conditions={"x": 0})
    with pytest.raises(ValueError, match="Could not find an equilibrium"):
        _find_start_state(m, "p", 1.0, {})


def test_negative_solutions_of_an_sbml_model_are_not_counted(tmp_path, monkeypatch):
    from dynalyst.analysis.steady_states import find_steady_states
    path = _write_sbml(tmp_path / "placeholder_square.xml", "placeholder_square", "placeholder",
                       rate="k - x*x")
    r = json.loads(find_steady_states(str(path)))
    assert r["n_distinct_steady_states"] == 1
    assert r["steady_states"][0]["state"]["x"] == pytest.approx(1.0)
    assert [s["state"]["x"] for s in r["non_physical_solutions"]] == [pytest.approx(-1.0)]
    assert np.array(r["steady_states"][0]["jacobian"]).shape == (1, 1)


# -- model resolution and provenance ---------------------------------------------------------

def test_a_model_is_found_by_its_declared_name(tmp_path, monkeypatch):
    from dynalyst.analysis.generic_sbml import resolve_sbml_path
    _write_sbml(tmp_path / "file_one.xml", "m1", "Placeholder Switch - two states")
    monkeypatch.setattr(config, "SBML_SEARCH_DIRS", [tmp_path])
    assert resolve_sbml_path("placeholder switch two states").name == "file_one.xml"


def test_two_files_declaring_one_name_are_ambiguous(tmp_path, monkeypatch):
    from dynalyst.analysis.generic_sbml import resolve_sbml_path
    _write_sbml(tmp_path / "a1.xml", "m1", "Placeholder Net", k=1.0)
    _write_sbml(tmp_path / "a2.xml", "m2", "Placeholder Net", k=2.0)
    monkeypatch.setattr(config, "SBML_SEARCH_DIRS", [tmp_path])
    with pytest.raises(ValueError, match="2 different models"):
        resolve_sbml_path("Placeholder Net")


# -- approval, arguments ------------------------------------------------------------------------

def test_a_one_item_list_for_a_string_argument_becomes_the_string():
    from dynalyst.orchestration.tools import coerce_args
    out = coerce_args("find_steady_states", {"model_name_or_path": ["m.xml"], "species": ["a"]})
    assert out == {"model_name_or_path": "m.xml", "species": ["a"]}


# -- the final answer --------------------------------------------------------------------------

@pytest.mark.parametrize("answer", ["x is 3.22 × 10^{-19}", r"x = $3.2 \times 10^{-19}$",
                                    "x = 3.2·10⁻¹⁹", "x ≈ 3.2e-19", "x = 3.2×10−19"])
def test_numbers_are_compared_by_value_in_any_notation(answer):
    from dynalyst.orchestration.final_check import unsupported_values
    assert unsupported_values(answer, ['{"x": 3.2156e-19}']) == []


def test_a_wrong_number_in_scientific_notation_is_still_caught():
    from dynalyst.orchestration.final_check import unsupported_values
    assert unsupported_values("x = 7.1 × 10^{-19}", ['{"x": 3.2156e-19}']) == ["7.1e-19"]


def test_the_answers_own_next_steps_give_way_to_the_follow_ups():
    from dynalyst.orchestration.next_steps import strip_own_next_steps
    assert strip_own_next_steps("Result: x.\n\n**Next steps:**\n1. a\n2. b") == "Result: x."
    kept = "Result.\n\n### What next?\n- a\n\n## Details\nmore"
    assert strip_own_next_steps(kept) == kept
    assert strip_own_next_steps("The next step is AUTO.") == "The next step is AUTO."


# -- the supervisor ------------------------------------------------------------------------------

class _Chat:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def bind_tools(self, tools, **kw):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls += 1
        return self.fn(messages)


def _call(name, **args):
    import itertools
    _call.n = getattr(_call, "n", itertools.count())
    return {"name": name, "args": args, "id": f"r{next(_call.n)}", "type": "tool_call"}


# -- CRNT in a worker process with a time limit -------------------------------------------------------

_CYCLE = dict(odes={"a": "-k1*a + k2*b", "b": "k1*a - k2*b"}, parameters={"k1": 1.0, "k2": 2.0},
              initial_conditions={"a": 1.0, "b": 0.0})


def test_crnt_in_the_worker_gives_the_same_answer(monkeypatch):
    from dynalyst.analysis.crnt_analysis import crnt_bistability
    monkeypatch.setattr(config, "CRNT_MAX_SECONDS", 600)
    d = json.loads(crnt_bistability(**_CYCLE))
    assert "timed_out" not in d and d["bistable"] is False      # deficiency zero


def test_a_crnt_run_over_the_time_limit_is_stopped_and_says_so(monkeypatch):
    from dynalyst.analysis.crnt_analysis import crnt_bistability
    monkeypatch.setattr(config, "CRNT_MAX_SECONDS", 0.5)   # less than the worker's start-up
    d = json.loads(crnt_bistability(**_CYCLE))
    assert d["timed_out"] is True and d["bistable"] is None
    assert d["conclusion"].startswith("Not decided")


# -- stochastic system size -----------------------------------------------------------------------



# -- a model named by ID, physical conservation laws -------------------------------------------

def test_conservation_laws_are_total_amounts_with_non_negative_coefficients():
    from dynalyst.analysis.crnt_network import conservation_laws
    # E + S <-> ES -> E + P, P -> S: total enzyme E + ES and total substrate S + ES + P
    net = {"species": ["E", "S", "ES", "P"], "reactions": [
        {"reactants": {"E": 1, "S": 1}, "products": {"ES": 1}},
        {"reactants": {"ES": 1}, "products": {"E": 1, "S": 1}},
        {"reactants": {"ES": 1}, "products": {"E": 1, "P": 1}},
        {"reactants": {"P": 1}, "products": {"S": 1}}]}
    laws = conservation_laws(net)
    assert len(laws) == 2 and all(c > 0 for law in laws for c in law.values())
    assert {frozenset(law) for law in laws} == {frozenset({"E", "ES"}), frozenset({"S", "ES", "P"})}


def test_crnt_general_approach_runs_on_current_scipy():
    import importlib.util
    if importlib.util.find_spec("crnt4sbml") is None:
        pytest.skip("crnt4sbml is not installed")
    from dynalyst.analysis.crnt_network import import_crnt4sbml
    import_crnt4sbml()
    from crnt4sbml import general_approach
    res = general_approach.scipy.optimize.dual_annealing(
        lambda x: float((x[0] - 0.3) ** 2), bounds=[(0, 1)], seed=0, maxiter=20,
        local_search_options={"method": "Nelder-Mead"})
    assert abs(res.x[0] - 0.3) < 1e-3
    assert general_approach.scipy.linalg is __import__("scipy.linalg").linalg


def test_crnt_offers_each_further_analysis_separately():
    from types import SimpleNamespace
    from dynalyst.orchestration.next_steps import suggest
    out = json.dumps({"model": "net_a", "mass_action": True, "bistable": None,
                      "multistationary": True, "multistationary_sets": [
                          {"model_file": "outputs/crnt/net_a_crnt_set_1.xml",
                           "conservation_totals": {"T1": 5.0}}]})
    rec = SimpleNamespace(steps=[{"kind": "delegation", "tool": "crnt_bistability",
                                  "arguments": {"model_name_or_path": "net_a.xml"},
                                  "output": out, "status": "ok"}])
    queries = [f["query"] for f in suggest(rec)]
    assert any(q.startswith("Run a bifurcation analysis of net_a") for q in queries)
    assert queries[0] == "Run a bifurcation analysis of outputs/crnt/net_a_crnt_set_1.xml and check for bistability"
    assert any("LHS" in q for q in queries) and any("reaction network" in q for q in queries)
    assert any("Jacobian" in q for q in queries)
    assert not any("quasi-steady-state" in q for q in queries)


def test_a_name_like_x0_used_in_the_equations_is_a_parameter():
    from dynalyst.analysis.typed_equations import parse_typed_system
    r = parse_typed_system("dA/dt = k*B0 - d*A, dB/dt = k*A0 - d*B, k = 1, d = 0.1, A0 = 1, B0 = 2")
    assert r["parameters"]["A0"] == 1 and r["parameters"]["B0"] == 2 and not r["unvalued"]
    r = parse_typed_system("dX/dt = a - b*X with a = 0.05, b = 0.45 and X0 = 2")
    assert r["initial_conditions"]["X"] == 2 and "X0" not in r["parameters"]


def test_output_paths_with_spaces_are_read_whole():
    from dynalyst.orchestration.reviewer import _TEXT_ARTIFACTS
    text = "Regime: x\nSubplot figure: /a/b (4)/c d.png\nCounts chart: /a/e.png  \nother"
    assert _TEXT_ARTIFACTS.findall(text) == ["/a/b (4)/c d.png", "/a/e.png"]


def test_no_absence_is_concluded_from_methods_that_did_not_apply():
    from dynalyst.orchestration.final_check import undecided_absence_claims
    open_ = {"kind": "delegation", "status": "ok",
             "output": json.dumps({"bistable": None, "conclusion": "CRNT does not apply"})}
    failed = {"kind": "delegation", "status": "unresolved", "output": json.dumps({"error": "x"})}
    decided = {"kind": "delegation", "status": "ok",
               "output": json.dumps({"bifurcation_points": [], "classification": {}})}
    claim = "The model is not bistable and does not oscillate."
    assert undecided_absence_claims(claim, [open_, failed]) == ["bistability", "oscillation"]
    assert undecided_absence_claims(claim, [open_, decided]) == []
    assert undecided_absence_claims("Bistability is neither shown nor ruled out.", [open_]) == []



def test_crnt_alone_cannot_make_the_model_bistable():
    from dynalyst.orchestration.final_check import stability_claim_without_check
    crnt = {"kind": "delegation", "tool": "crnt_bistability", "status": "ok",
            "output": json.dumps({"bistable": None, "multistationary": True})}
    diagram = {"kind": "delegation", "tool": "run_bifurcation_analysis", "status": "ok",
               "output": json.dumps({"classification": {"category": "bistable"}})}
    assert stability_claim_without_check("The model is bistable.", [crnt])
    assert not stability_claim_without_check("The network can have several steady states.", [crnt])
    assert not stability_claim_without_check("The model is bistable.", [crnt, diagram])



def test_a_negated_or_open_sentence_is_not_a_bistability_claim():
    from dynalyst.orchestration.final_check import stability_claim_without_check
    crnt = {"kind": "delegation", "tool": "crnt_bistability", "status": "ok",
            "output": json.dumps({"bistable": None, "multistationary": True})}
    proof = {"kind": "delegation", "tool": "crnt_bistability", "status": "ok",
             "output": json.dumps({"bistable": False, "proof": True})}
    assert not stability_claim_without_check("It cannot exhibit bistability.", [crnt])
    assert not stability_claim_without_check("A diagram shows whether it is bistable.", [crnt])
    assert not stability_claim_without_check("The model is not bistable.", [proof])

