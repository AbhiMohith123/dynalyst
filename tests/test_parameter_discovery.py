"""Continuation parameters are DISCOVERED from the model, never guessed.

The BIOMD0000000015 failure: no global parameters, 111 reaction-local constants
and two clamped (boundary) species R5P / Pi -- the loader exposed none of them,
so every tool rejected R5P ("Available parameters: []"), the LHS tools crashed
with IndexError, and the sub-agent filled in ATOT from the A/APP examples.
These tests rebuild that shape in a small SBML file."""
import json

import libsbml
import pytest

from dynalyst.analysis.ode_model import canonical_param, resolve_model


def _sbml(path, with_params=True):
    """S -> X -> 0 with a clamped source S, rate constants local to each reaction."""
    doc = libsbml.SBMLDocument(3, 1)
    m = doc.createModel()
    m.setId("toy")
    c = m.createCompartment(); c.setId("cell"); c.setSize(1); c.setConstant(True)
    for sid, val, boundary in (("S", 2.0, True), ("X", 1.0, False)):
        s = m.createSpecies(); s.setId(sid); s.setCompartment("cell")
        s.setInitialConcentration(val); s.setBoundaryCondition(boundary)
        s.setConstant(False); s.setHasOnlySubstanceUnits(False)
    for rid, reac, prod, law, local in (
            ("make", "S", "X", "k * S" if with_params else "0.5 * S", {"k": 0.5}),
            ("decay", "X", None, "k * X / (Km + X)" if with_params else "X", {"k": 1.0, "Km": 0.3})):
        r = m.createReaction(); r.setId(rid); r.setReversible(False); r.setFast(False)
        sr = r.createReactant(); sr.setSpecies(reac); sr.setStoichiometry(1); sr.setConstant(True)
        if prod:
            sp_ = r.createProduct(); sp_.setSpecies(prod); sp_.setStoichiometry(1); sp_.setConstant(True)
        kl = r.createKineticLaw(); kl.setMath(libsbml.parseL3Formula(law))
        if with_params:
            for name, val in local.items():
                lp = kl.createLocalParameter(); lp.setId(name); lp.setValue(val)
    libsbml.writeSBMLToFile(doc, str(path))
    return str(path)


def test_fixed_species_and_local_constants_are_parameters(tmp_path):
    m = resolve_model(_sbml(tmp_path / "toy.xml"))
    assert m["param_kinds"] == {"S": "fixed species", "make_k": "local parameter",
                                "decay_k": "local parameter", "decay_Km": "local parameter"}
    assert canonical_param(m, "S") == "S"
    assert canonical_param(m, "Km") == "decay_Km"         # a bare local name that is unique
    assert canonical_param(m, "k") is None                 # ambiguous: two reactions have k
    assert canonical_param(m, "X") is None                 # a changing species is not one
    assert canonical_param(m, "ATOT") is None


def test_legacy_regime_parser_uses_the_canonical_local_parameter_inventory(tmp_path):
    from dynalyst.analysis.ode_model import resolve_model
    from dynalyst.regime.parser import parse_sbml

    path = _sbml(tmp_path / "toy.xml")
    canonical = resolve_model(path)
    parsed = parse_sbml(path)

    assert parsed["species"] == canonical["state_names"]
    assert parsed["params"] == canonical["param_names"]
    assert parsed["param_values"] == canonical["param_defaults"]
    assert parsed["params"] == ["S", "make_k", "decay_k", "decay_Km"]


def test_all_sbml_resolvers_accept_the_downloaded_biomodels_stem(tmp_path, monkeypatch):
    from dynalyst import config
    from dynalyst.analysis import biomodels
    from dynalyst.analysis.generic_sbml import resolve_sbml_path
    from dynalyst.analysis.ode_model import resolve_model

    source = _sbml(tmp_path / "remote.xml")
    payload = open(source, "rb").read()

    class Response:
        content = payload
        text = payload.decode()

    def fake_get(url, params=None, timeout=None, as_json=True):
        if as_json:
            return {"main": [{"name": "remote.xml"}]}
        return Response()

    monkeypatch.setattr(biomodels, "_get", fake_get)
    monkeypatch.setattr(config, "LOCAL_SBML_MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "SBML_SEARCH_DIRS", [tmp_path])

    result = json.loads(biomodels.fetch_biomodels_model("BIOMD0000000014"))
    assert result["use_as"] == "BIOMD0000000014"
    assert resolve_sbml_path("BIOMD0000000014").name == "BIOMD0000000014.xml"
    assert resolve_model("BIOMD0000000014")["param_names"] == [
        "S", "make_k", "decay_k", "decay_Km"
    ]


def test_the_guard_rejects_a_guessed_name_and_lists_the_real_ones(tmp_path):
    from dynalyst.orchestration.inputs import parameter_problem
    path = _sbml(tmp_path / "toy.xml")
    msg = parameter_problem({"model_name_or_path": path, "param_name": "ATOT"})
    assert "not a parameter" in msg and "decay_Km" in msg and "fixed species" in msg
    assert parameter_problem({"model_name_or_path": path, "param_name": "S"}) is None
    assert parameter_problem({"model": path, "bif_param": "Km"}) is None
    assert parameter_problem({"model": "pdp_mm_bistable.txt", "bif_param": "ATOT"}) is None
    assert "not a parameter" in parameter_problem({"model": "pdp_mm_bistable.txt", "bif_param": "S"})


def test_a_model_without_any_parameter_gets_a_clear_message_not_an_index_error(tmp_path):
    from dynalyst.regime import model_source
    path = _sbml(tmp_path / "bare.xml", with_params=False)
    m = resolve_model(path)
    assert m["param_names"] == ["S"]                 # only the clamped source is left
    lp = _sbml(tmp_path / "none.xml", with_params=False)
    doc = libsbml.readSBMLFromFile(lp)
    doc.getModel().getSpecies("S").setBoundaryCondition(False)   # nothing tunable at all
    doc.getModel().getSpecies("S").setConstant(True)
    doc.getModel().getReaction("make").getKineticLaw().setMath(libsbml.parseL3Formula("0.5"))
    doc.getModel().getReaction("make").removeReactant("S")
    libsbml.writeSBMLToFile(doc, lp)
    with pytest.raises(ValueError):
        model_source.resolve_target(lp)


def test_lhs_target_takes_the_users_name_and_never_guesses_from_names(tmp_path):
    from dynalyst.regime import model_source
    path = _sbml(tmp_path / "toy.xml")
    assert model_source.resolve_target(path, "Km")[0] == "decay_Km"
    assert model_source.resolve_target(path, "S")[0] == "S"
    with pytest.raises(ValueError, match="not a parameter"):
        model_source.resolve_target(path, "ATOT")


def test_a_runaway_is_not_reported_as_a_fold():
    import sympy as sp
    from dynalyst.analysis.parameter_screen import screen_parameters
    x, a, b = sp.symbols("x a b")
    # dx/dt = a - b x/(1+x): x* = a/(b-a) grows without bound as a -> b, no fold
    model = {"state_names": ["x"], "state_symbols": [x], "param_names": ["a", "b"],
             "param_symbols": [a, b], "param_defaults": {"a": 0.5, "b": 1.0},
             "rhs": [a - b * x / (1 + x)], "y0": {"x": 1.0}}
    rec = {r["parameter"]: r for r in screen_parameters(model, {"x": 1.0})}
    ev = rec["a"]["events"][0]
    assert ev.get("runaway") and abs(ev["near"] - 1.0) < 0.05


def test_the_examples_no_longer_prime_a_names():
    from dynalyst.orchestration.understanding import RANGE_EXAMPLES, RANGE_GLOSSARY
    text = RANGE_GLOSSARY + RANGE_EXAMPLES
    for name in ("km7", "k2cat", "KTOT", "Km5"):
        assert name not in text
    from dynalyst.orchestration.subagents import SUBAGENTS     # the rule is in every sub-agent's
    assert "never guess one" in SUBAGENTS["bifurcation"].prompt  # prompt now (RULES 1)



def test_reaction_rates_are_divided_by_the_compartment_volume(tmp_path):
    # BIOMD0000000016 (volume 1e-15): rates are amounts / time, so a concentration changes
    # by rate / volume. Without the division every ODE was 1e-15 too small, all
    # eigenvalues ~0, and AUTO stopped ('MX') at every parameter value.
    import sympy as sp
    small = tmp_path / "small.xml"
    _sbml(small)
    doc = libsbml.readSBMLFromFile(str(small))
    doc.getModel().getCompartment("cell").setSize(1e-15)
    for r in doc.getModel().getListOfReactions():          # rates written per amount, as SBML does
        r.getKineticLaw().setMath(libsbml.parseL3Formula(
            "cell * (" + libsbml.formulaToL3String(r.getKineticLaw().getMath()) + ")"))
    libsbml.writeSBMLToFile(doc, str(small))
    unit = resolve_model(_sbml(tmp_path / "unit.xml"))
    tiny = resolve_model(str(small))
    point = {sp.Symbol("X"): 0.7, **{sp.Symbol(k): v for k, v in tiny["param_defaults"].items()}}
    assert float(tiny["rhs"][0].subs(point)) == pytest.approx(float(unit["rhs"][0].subs(point)))


def test_a_model_starting_at_zero_with_hill_terms_is_screened_and_not_blocked():
    # Leloup-Goldbeter (BIOMD0000000021): every species starts at 0 and the Jacobian has
    # c**n / c terms, 0/0 at the start. That crashed the screen, and the sanity check then
    # blocked the run on its OWN inability to evaluate. Neither may happen.
    from dynalyst.analysis.model_check import sanity
    from dynalyst.analysis.parameter_screen import settled_state
    m = resolve_model(species=["c", "m"], odes={"m": "v*K**n/(c**n + K**n) - m", "c": "m - c"},
                      parameters={"v": 2.0, "K": 1.0, "n": 4.0}, initial_conditions={"c": 0.0, "m": 0.0})
    state, note = settled_state(m)                      # no LinAlgError
    assert all(abs(v) < 1e6 for v in state.values())
    assert sanity(m)["ok"]
    broken = {**m, "param_defaults": {k: float("nan") for k in m["param_defaults"]}}
    assert sanity(broken)["ok"] or sanity(broken)["problems"]   # never raises


def test_initial_assignments_are_applied_and_unsupported_features_are_reported(tmp_path):
    path = tmp_path / "ia.xml"
    _sbml(path)
    doc = libsbml.readSBMLFromFile(str(path))
    m = doc.getModel()
    ia = m.createInitialAssignment()                       # X(0) = 3 * S
    ia.setSymbol("X"); ia.setMath(libsbml.parseL3Formula("3 * S"))
    ev = m.createEvent(); ev.setUseValuesFromTriggerTime(True)       # an event: not applied
    ev.createTrigger().setMath(libsbml.parseL3Formula("S > 100")); ev.getTrigger().setInitialValue(False)
    ev.getTrigger().setPersistent(True)
    ea = ev.createEventAssignment(); ea.setVariable("S"); ea.setMath(libsbml.parseL3Formula("0"))
    libsbml.writeSBMLToFile(doc, str(path))
    model = resolve_model(str(path))
    assert model["y0"]["X"] == pytest.approx(6.0)          # 3 * S(=2), not the file's 1.0
    assert any("event" in c for c in model["caveats"])


def test_an_mx_retry_is_suggested_once_and_capped():
    from dynalyst import config
    from dynalyst.orchestration.reviewer import Reviewer
    out = {"swept_parameter": "p", "bifurcation_points": [], "range_coverage": {
        "abnormal_terminations": 2}, "auto_constants_used": {"nmx": 10 ** 9, "dsmin": 1e-30}}
    first = Reviewer()._review_run_bifurcation_analysis({"model_name_or_path": "m"}, out, "")
    mx = next(c for c in first if c.name == "continuation_complete")
    assert mx.suggested_args["nmx"] <= config.AUTO_MAX_NMX
    assert mx.suggested_args["dsmin"] >= config.AUTO_MIN_DSMIN
    again = Reviewer()._review_run_bifurcation_analysis(
        {"model_name_or_path": "m", "n_pts": 40000}, out, "")
    mx = next(c for c in again if c.name == "continuation_complete")
    assert not mx.suggested_args and "reported as it is" in mx.detail
