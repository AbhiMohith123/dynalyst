"""Simulation applies SBML events and delays with their standard meaning (hybrid_sim),
checked against solutions known in closed form."""
import json
import math

import libsbml
import numpy as np
import pytest


def _model(tmp_path, name, rate_rules, params, species=(("x", 1.0),), events=()):
    """An SBML file: species with rate rules, parameters, and events
    (trigger, [(variable, value formula)], delay formula or None, extra attributes)."""
    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    c = m.createCompartment(); c.setId("cell"); c.setSize(1); c.setConstant(True)
    for sid, x0 in species:
        s = m.createSpecies(); s.setId(sid); s.setCompartment("cell"); s.setInitialConcentration(x0)
        s.setHasOnlySubstanceUnits(False); s.setBoundaryCondition(False); s.setConstant(False)
    for pid, val, const in params:
        p = m.createParameter(); p.setId(pid); p.setValue(val); p.setConstant(const)
    for var, formula in rate_rules:
        r = m.createRateRule(); r.setVariable(var); r.setMath(libsbml.parseL3Formula(formula))
    for trig, assigns, delay, attrs in events:
        e = m.createEvent(); e.setUseValuesFromTriggerTime(attrs.get("use_trigger_values", True))
        t = e.createTrigger(); t.setMath(libsbml.parseL3Formula(trig))
        t.setInitialValue(attrs.get("initial_value", True)); t.setPersistent(attrs.get("persistent", True))
        if delay:
            d = e.createDelay(); d.setMath(libsbml.parseL3Formula(delay))
        for var, f in assigns:
            a = e.createEventAssignment(); a.setVariable(var); a.setMath(libsbml.parseL3Formula(f))
    path = tmp_path / f"{name}.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    return str(path)


def _run(path, t_end, n=201):
    from dynalyst.analysis.jaxkm_tools import _get_trajectory
    df, names, _, src = _get_trajectory(path, t_end, n)
    return df, src


def test_a_dose_at_a_set_time_resets_the_state(tmp_path):
    path = _model(tmp_path, "dose", [("x", "-k*x")], [("k", 0.5, True)],
                  events=[("time >= 5", [("x", "x + 1")], None, {})])
    df, src = _run(path, 10)
    t, x = df["time"].to_numpy(), df["x"].to_numpy()
    before = math.exp(-0.5 * 5)
    expected = np.where(t < 5, np.exp(-0.5 * t), (before + 1) * np.exp(-0.5 * (t - 5)))
    assert np.max(np.abs(x - expected)) < 1e-5
    assert [e["event"] for e in src["events_fired"]] and abs(src["events_fired"][0]["time"] - 5) < 1e-6


def test_a_state_trigger_fires_each_time_it_becomes_true(tmp_path):
    # decays at rate 1 and is reset to 1 whenever it falls below 0.5: a sawtooth of period ln 2
    path = _model(tmp_path, "saw", [("x", "-x")], [],
                  events=[("x < 0.5", [("x", "1")], None, {})])
    df, src = _run(path, 5)
    times = [e["time"] for e in src["events_fired"]]
    assert len(times) == int(5 / math.log(2))
    assert np.allclose(np.diff(times), math.log(2), atol=1e-6)


def test_a_delayed_event_uses_the_values_at_trigger_time(tmp_path):
    path = _model(tmp_path, "late", [("x", "-k*x"), ("y", "0")], [("k", 1.0, True)],
                  species=(("x", 1.0), ("y", 0.0)),
                  events=[("time >= 1", [("y", "x")], "2", {"use_trigger_values": True})])
    df, src = _run(path, 4)
    assert abs(src["events_fired"][0]["time"] - 3) < 1e-6
    assert abs(df["y"].iloc[-1] - math.exp(-1)) < 1e-5        # x at the trigger, t = 1


def test_a_trigger_true_at_the_start_fires_only_if_it_was_false_before(tmp_path):
    for initial, n in ((True, 0), (False, 1)):
        path = _model(tmp_path, f"start_{initial}", [("x", "0")], [],
                      events=[("time >= 0", [("x", "x + 1")], None, {"initial_value": initial})])
        df, src = _run(path, 1)
        assert len(src["events_fired"]) == n and abs(df["x"].iloc[-1] - (1 + n)) < 1e-9


def test_an_event_can_change_a_parameter(tmp_path):
    path = _model(tmp_path, "switch", [("x", "k")], [("k", 1.0, False)],
                  events=[("time >= 2", [("k", "0")], None, {})])
    df, _ = _run(path, 4)
    assert abs(df["x"].iloc[-1] - 3.0) < 1e-6                 # 1 + 2 * 1, then flat


def test_a_constant_delay_is_simulated_by_the_method_of_steps(tmp_path):
    # x' = -k x(t - 1), history x = 1: x = 1 - k t on [0, 1],
    # then x = (1 - k) - k (t - 1) + k^2 (t - 1)^2 / 2 on [1, 2]
    path = _model(tmp_path, "dde", [("x", "-k * delay(x, tau)")], [("k", 0.5, True), ("tau", 1.0, True)])
    df, src = _run(path, 2, 401)
    t, x = df["time"].to_numpy(), df["x"].to_numpy()
    k = 0.5
    exact = np.where(t <= 1, 1 - k * t, (1 - k) - k * (t - 1) + k ** 2 * (t - 1) ** 2 / 2)
    assert np.max(np.abs(x - exact)) < 1e-6
    assert "delay" in src["delay_differential_equation"]


def test_simulate_model_reports_events_and_delays(tmp_path):
    from dynalyst.analysis.jaxkm_tools import simulate_model
    path = _model(tmp_path, "dose2", [("x", "-k*x")], [("k", 0.5, True)],
                  events=[("time >= 5", [("x", "x + 1")], None, {})])
    out = json.loads(simulate_model(path, t_end=10))
    assert "error" not in out and out["events_fired"][0]["time"] == pytest.approx(5, abs=1e-6)


def _rxn_model(tmp_path, name, build):
    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    build(m)
    path = tmp_path / f"{name}.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    return str(path)


def _species(m, sid, x0, comp="cell", amount=False):
    s = m.createSpecies(); s.setId(sid); s.setCompartment(comp); s.setInitialConcentration(x0)
    s.setHasOnlySubstanceUnits(amount); s.setBoundaryCondition(False); s.setConstant(False)


def _param(m, pid, v, const=True):
    p = m.createParameter(); p.setId(pid); p.setValue(v); p.setConstant(const)


def _reaction(m, rid, reactants, products, law):
    r = m.createReaction(); r.setId(rid); r.setReversible(False)
    for s in reactants:
        x = r.createReactant(); x.setSpecies(s); x.setStoichiometry(1); x.setConstant(True)
    for s in products:
        x = r.createProduct(); x.setSpecies(s); x.setStoichiometry(1); x.setConstant(True)
    r.createKineticLaw().setMath(libsbml.parseL3Formula(law))


def test_an_algebraic_rule_that_fixes_one_variable_is_solved_for_it(tmp_path):
    import sympy as sp
    from dynalyst.analysis.generic_sbml import build_symbolic_model

    def build(m):
        c = m.createCompartment(); c.setId("cell"); c.setSize(1); c.setConstant(True)
        _species(m, "A", 0.0); _species(m, "B", 1.0)
        _param(m, "T", 1.0); _param(m, "k1", 2.0); _param(m, "k2", 1.0)
        _reaction(m, "make", [], ["A"], "k1 * B")          # B = T - A, set by the rule
        _reaction(m, "lose", ["A"], [], "k2 * A")
        r = m.createAlgebraicRule(); r.setMath(libsbml.parseL3Formula("A + B - T"))
    model = build_symbolic_model(_rxn_model(tmp_path, "alg", build))
    A, T, k1, k2 = sp.symbols("A T k1 k2")
    assert model["state_names"] == ["A"]
    assert sp.simplify(model["rhs"][0] - (k1 * (T - A) - k2 * A)) == 0


def test_an_algebraic_rule_that_fixes_nothing_unique_is_refused(tmp_path):
    from dynalyst.analysis.generic_sbml import build_symbolic_model

    def build(m):
        c = m.createCompartment(); c.setId("cell"); c.setSize(1); c.setConstant(True)
        _species(m, "A", 1.0); _param(m, "u", 1.0, const=False); _param(m, "w", 1.0, const=False)
        _reaction(m, "lose", ["A"], [], "u * A")
        r = m.createAlgebraicRule(); r.setMath(libsbml.parseL3Formula("u + w - 2"))
    with pytest.raises(ValueError, match="differential-algebraic"):
        build_symbolic_model(_rxn_model(tmp_path, "alg2", build))


def test_a_growing_compartment_dilutes_what_is_in_it(tmp_path):
    import sympy as sp
    from dynalyst.analysis.generic_sbml import build_symbolic_model

    def build(m):
        c = m.createCompartment(); c.setId("V"); c.setSize(2.0); c.setConstant(False)
        _species(m, "S", 1.0, comp="V")
        _param(m, "r", 0.1); _param(m, "p", 0.5)
        _reaction(m, "made", [], ["S"], "p")               # amount per time
        rr = m.createRateRule(); rr.setVariable("V"); rr.setMath(libsbml.parseL3Formula("r * V"))
    path = _rxn_model(tmp_path, "grow", build)
    model = build_symbolic_model(path)
    S, V, r, p = sp.symbols("S V r p")
    rhs = dict(zip(model["state_names"], model["rhs"]))
    assert sp.simplify(rhs["S"] - (p / V - S * r)) == 0 and sp.simplify(rhs["V"] - r * V) == 0
    assert model["y0"]["V"] == 2.0
    # the amount S*V grows by p per unit time
    df, _ = _run(path, 10)
    amount = df["S"].to_numpy() * df["V"].to_numpy()
    assert np.allclose(amount, 2.0 + 0.5 * df["time"].to_numpy(), rtol=1e-6)


def test_a_fired_event_says_what_it_changed(tmp_path):
    path = _model(tmp_path, "dose3", [("x", "-k*x")], [("k", 0.5, False)],
                  events=[("time >= 2", [("x", "x + 1"), ("k", "0")], None, {})])
    _, src = _run(path, 4)
    changed = src["events_fired"][0]["changed"]
    assert changed["k"] == "0.5 -> 0" and changed["x"].endswith("-> " + f"{math.exp(-1) + 1:.6g}")
