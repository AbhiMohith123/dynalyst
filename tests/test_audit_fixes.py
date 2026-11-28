"""Regression cases for first-party audit findings N001--N025."""
import json
import math
import os
from types import SimpleNamespace
from unittest.mock import Mock

import libsbml
import numpy as np
import pytest
import sympy as sp

from dynalyst import config
from dynalyst.analysis.generic_sbml import build_symbolic_model, DELAY, SBML_TIME
from dynalyst.analysis.hybrid_sim import simulate


def sbml(volume=1, amount=False, only_amount=False, value=1):
    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel(); m.setId("audit_fixture")
    c = m.createCompartment(); c.setId("cell"); c.setSize(volume)
    c.setConstant(True); c.setSpatialDimensions(3)
    s = m.createSpecies(); s.setId("X"); s.setCompartment("cell")
    s.setConstant(False); s.setBoundaryCondition(False); s.setHasOnlySubstanceUnits(only_amount)
    (s.setInitialAmount if amount else s.setInitialConcentration)(value)
    return doc, m


def rate(m, name, formula):
    r = m.createRateRule(); r.setVariable(name); r.setMath(libsbml.parseL3Formula(formula))


def param(m, name, value, constant=True):
    p = m.createParameter(); p.setId(name); p.setValue(value); p.setConstant(constant)


def initial(m, name, formula):
    a = m.createInitialAssignment(); a.setSymbol(name); a.setMath(libsbml.parseL3Formula(formula))


def load(doc, tmp_path, name="model.xml", **kwargs):
    path = tmp_path / name
    libsbml.writeSBMLToFile(doc, str(path))
    return build_symbolic_model(path, **kwargs)


@pytest.mark.parametrize("amount,only,value,expected", [(True,False,20,10), (False,True,10,20)])
def test_n001_initial_units(tmp_path, amount, only, value, expected):
    doc, m = sbml(2, amount, only, value); rate(m, "X", "-X")
    assert load(doc, tmp_path)["y0"]["X"] == expected
    initial(m, "X", "7")
    assert load(doc, tmp_path)["y0"]["X"] == 7  # assignment already in symbol units


@pytest.mark.parametrize("reverse", [False, True])
def test_n002_initial_dependency_order_and_volume(tmp_path, reverse):
    doc, m = sbml(1, amount=True, value=20); rate(m, "X", "a-X")
    param(m, "a", 0); param(m, "b", 1)
    pairs = [("a", "b+1"), ("b", "4"), ("cell", "2")]
    for name, formula in reversed(pairs) if reverse else pairs:
        initial(m, name, formula)
    parsed = load(doc, tmp_path)
    assert parsed["param_defaults"]["a"] == 5
    assert parsed["y0"]["X"] == 10


def test_n002_initial_cycle_is_rejected(tmp_path):
    doc, m = sbml(); rate(m, "X", "a-X"); param(m, "a", 0); param(m, "b", 1)
    initial(m, "a", "b+1"); initial(m, "b", "a+1")
    with pytest.raises(ValueError, match="Cyclic"):
        load(doc, tmp_path)


def test_n003_rate_parameter_is_a_state(tmp_path):
    doc, m = sbml(); param(m, "p", 1, False)
    rate(m, "X", "-p*X"); rate(m, "p", "1")
    parsed = load(doc, tmp_path, keep_time=True)
    assert parsed["state_names"] == ["X", "p"]
    assert "p" not in parsed["param_names"]
    ts = np.linspace(0, 2, 21)
    ys, _ = simulate(parsed, [1, 1], {}, 2, ts)
    np.testing.assert_allclose(ys[:, 0], np.exp(-ts-ts**2/2), atol=1e-7)
    np.testing.assert_allclose(ys[:, 1], 1+ts, atol=1e-7)
    from dynalyst.analysis.jaxkm_tools import _get_trajectory
    table, names, _, _ = _get_trajectory(str(tmp_path/"model.xml"), 2, 21)
    assert "p" in names
    np.testing.assert_allclose(table["X"], np.exp(-ts-ts**2/2), atol=1e-7)
    from dynalyst.analysis.generic_sbml import simulate_generic_sbml
    _, legacy, names, _ = simulate_generic_sbml(tmp_path/"model.xml", 2, 21)
    np.testing.assert_allclose(legacy[:, names.index("p")], 1+ts, atol=1e-7)


def test_n004_event_only_rule_chain(tmp_path):
    doc, m = sbml(); rate(m, "X", "1")
    for name, formula in [("obs", "X+1"), ("obs2", "obs+1")]:
        param(m, name, 0, False)
        r = m.createAssignmentRule(); r.setVariable(name); r.setMath(libsbml.parseL3Formula(formula))
    e = m.createEvent(); e.setId("reset"); e.setUseValuesFromTriggerTime(True)
    t = e.createTrigger(); t.setInitialValue(False); t.setPersistent(True)
    t.setMath(libsbml.parseL3Formula("obs2 > 3"))
    a = e.createEventAssignment(); a.setVariable("X"); a.setMath(libsbml.parseL3Formula("0"))
    parsed = load(doc, tmp_path, keep_time=True)
    assert parsed["events"][0]["trigger"].free_symbols == {sp.Symbol("X")}


def hybrid(names, rhs, y0, events=()):
    return {"state_names": names, "state_symbols": list(map(sp.Symbol, names)),
            "rhs": list(map(sp.sympify, rhs)), "y0": dict(zip(names, y0)),
            "param_names": [], "param_symbols": [], "events": events, "time_symbol": SBML_TIME}


def event(name, trigger, assignments, delay=None, persistent=True):
    return {"id": name, "trigger": trigger, "initial_value": False,
            "persistent": persistent, "delay": None if delay is None else sp.Float(delay),
            "use_trigger_values": False, "has_priority": False,
            "assignments": [(n,"state",sp.sympify(v)) for n,v in assignments]}


@pytest.mark.parametrize("horizon", [2, 100])
def test_n005_zero_delay_matches_ode(horizon):
    parsed = hybrid(["X"], [-DELAY(sp.Symbol("X"), 0)], [1])
    out, _ = simulate(parsed, [1], {}, horizon, np.array([0,1,2]))
    np.testing.assert_allclose(out[:, 0], np.exp(-np.arange(3)), atol=1e-7)


@pytest.mark.parametrize("delay", [None, 0, 1])
def test_n006_event_cascade(delay):
    ev = [event("set_x", SBML_TIME>=1, [("X",1)], delay),
          event("respond", sp.Symbol("X")>=1, [("Y",2)])]
    out, _ = simulate(hybrid(["X","Y"], [0,0], [0,0], ev), [0,0], {}, 3, [0,3])
    np.testing.assert_array_equal(out[-1], [1,2])


@pytest.mark.parametrize("at", [0, 1])
def test_n007_jump_sample_is_post_event(at):
    ev = [event("jump", SBML_TIME>=at, [("X",5)])]
    out, _ = simulate(hybrid(["X"], [0], [0], ev), [0], {}, 1, [at])
    assert out[0, 0] == 5


def test_n008_nonpersistent_episode_cancelled():
    tr = sp.Or(sp.And(SBML_TIME>=1, SBML_TIME<2), SBML_TIME>=3)
    ev = [event("pulse", tr, [("X",sp.Symbol("X")+1)], delay=4, persistent=False)]
    out, info = simulate(hybrid(["X"], [0], [0], ev), [0], {}, 8, [0,6,8])
    np.testing.assert_array_equal(out[:, 0], [0,0,1])
    assert [x[1] for x in info["events_fired"]] == pytest.approx([7])


@pytest.mark.parametrize("eigs", [[-1,0], [-1,1j,-1j], [0]])
def test_n009_nonhyperbolic_is_not_stable(eigs):
    from dynalyst.analysis.steady_states import classify_stability
    assert classify_stability(np.array(eigs)).startswith("nonhyperbolic")
    assert classify_stability(np.array([-1,0]), 1).startswith("stable")
    assert classify_stability(np.array([1,0])).startswith("unstable")


def test_n010_ambiguous_local_save_is_atomic():
    from dynalyst.analysis.save_model import _set_parameter
    doc, m = sbml()
    for name, value in [("r1",1), ("r2",2)]:
        r = m.createReaction(); r.setId(name); r.setReversible(False)
        kl = r.createKineticLaw(); kl.setMath(libsbml.parseL3Formula("k*X"))
        lp = kl.createLocalParameter(); lp.setId("k"); lp.setValue(value)
    with pytest.raises(ValueError, match="Ambiguous"):
        _set_parameter(m, "k", 99)
    assert [r.getKineticLaw().getLocalParameter("k").getValue() for r in m.getListOfReactions()] == [1,2]
    _set_parameter(m, "r2_k", 99)
    assert m.getReaction("r2").getKineticLaw().getLocalParameter("k").getValue() == 99


def test_n011_negative_fold_bracket(monkeypatch):
    from dynalyst.analysis.ode_model import model_from_odes
    from dynalyst.analysis import parameter_screen as ps
    monkeypatch.setattr(ps, "FOLLOW_EVALUATIONS", 1000)
    monkeypatch.setattr(ps, "FOLLOW_SECONDS", 1)
    m = model_from_odes(["x"], {"x":"p+2-x**2"}, {"p":-1}, {"x":1})
    found = ps.screen_parameters(m, {"x":1}, candidates=["p"], steps_per_decade=4)
    assert found[0]["events"][0]["near"] == pytest.approx(-2, abs=.01)


def test_n012_invalid_download_preserves_existing(tmp_path, monkeypatch):
    from dynalyst.analysis import biomodels
    monkeypatch.setattr(config, "LOCAL_SBML_MODELS_DIR", tmp_path)
    path = tmp_path / "BIOMD0000000001.xml"; path.write_bytes(b"original model")
    monkeypatch.setattr(biomodels, "_get", lambda *a, **k: SimpleNamespace(content=b"not XML",text="not XML"))
    result = json.loads(biomodels.fetch_biomodels_model("BIOMD0000000001", filename="alternative.xml"))
    assert "error" in result and path.read_bytes() == b"original model"


def test_n014_cleanup_preserves_lhs_results(tmp_path, monkeypatch):
    from dynalyst import housekeeping
    for attr, name in [("WORKDIR","work"), ("AUTO_WORK_DIR","auto"), ("OUTPUT_DIR","out")]:
        monkeypatch.setattr(config, attr, tmp_path/name)
    run = config.WORKDIR / "lhs_run"; run.mkdir(parents=True)
    summary = run / "summary.csv"; summary.write_text("sample_id,k\n0,1\n")
    os.utime(summary, (0,0)); os.utime(run, (0,0))
    housekeeping.prune_scratch(days=7)
    assert summary.exists()


@pytest.mark.parametrize("number,value", [("2 x 10^+3",2000), ("2 × 10^(-3)",.002),
                                           ("+.5e+2",50), ("-2e-3",-.002)])
def test_n015_publication_numbers(number, value):
    from dynalyst.analysis.paper_map import _find_values
    assert _find_values("k1", [(1, f"k1 = {number} per second.")])[0]["value"] == value


def test_n016_same_basename_distinct_sources(tmp_path):
    from dynalyst.rag.model_index import model_chunks, _save_chunks
    doc, m = sbml(); rate(m, "X", "-X")
    all_records = []
    for name in ["a","b"]:
        folder = tmp_path/name; folder.mkdir(); load(doc, folder)
        records = model_chunks(str(folder/"model.xml"), online=False)
        all_records.append(records)
        _save_chunks("model.xml", records, tmp_path/"chunks.jsonl")
    assert all_records[0][0]["id"] != all_records[1][0]["id"]
    saved = [json.loads(s) for s in (tmp_path/"chunks.jsonl").read_text().splitlines()]
    assert len(saved) == sum(map(len, all_records))


def test_n017_failed_index_retried(tmp_path, monkeypatch):
    from dynalyst.rag import model_index as mi
    monkeypatch.setattr(config, "PAPERS_DIR", tmp_path)
    monkeypatch.setattr(config, "BM25_INDEX_PATH", tmp_path/"absent.pkl")
    monkeypatch.setattr(mi, "PAPER_CHUNKS_FILE", tmp_path/"pending.jsonl")
    (tmp_path/"newpaper.pdf").write_bytes(b"fixture")
    mi.PAPER_CHUNKS_FILE.write_text(json.dumps({"metadata":{"source_name":"newpaper"}})+"\n")
    index = Mock(return_value='{"knowledge_base":"saved, not yet searchable"}')
    monkeypatch.setattr(mi, "index_pdf", index)
    assert mi.sync_papers() == []
    index.return_value = '{"knowledge_base":"updated"}'
    assert mi.sync_papers() == ["newpaper.pdf"] and index.call_count == 2


@pytest.mark.parametrize("volume", [2, 3, .5])
def test_n018_export_preserves_dynamics(tmp_path, monkeypatch, volume):
    from dynalyst.analysis.jaxkm_ode_to_sbml import odes_to_sbml
    monkeypatch.setattr(config, "LOCAL_SBML_MODELS_DIR", tmp_path)
    result = json.loads(odes_to_sbml(["X"], {"X":"-k*X"}, {"k":.4}, {"X":10},
                                     model_name="volume", compartment_size=volume))
    assert "error" not in result, result
    m = build_symbolic_model(result["path"])
    value = m["rhs"][0].subs({sp.Symbol("X"):10, **{sp.Symbol(k):v for k,v in m["param_defaults"].items()}})
    assert float(value) == pytest.approx(-4) and m["y0"]["X"] == 10


def test_n019_clear_discards_agent(tmp_path, monkeypatch):
    from dynalyst.agents import unified_agent
    from dynalyst import chat_ui, notebook
    monkeypatch.setattr(unified_agent, "_AGENTS", {(None,True,"ask"): SimpleNamespace(pending={"model":"old"})})
    monkeypatch.setattr(chat_ui, "display", lambda *a: None)
    monkeypatch.setattr(notebook, "LAST_RUN", object())
    ui = chat_ui.launch_chat_ui()
    try:
        ui.children[1].children[1].click()
        assert unified_agent._AGENTS == {} and notebook.LAST_RUN is None
        assert not ui.children[2].children
    finally:
        ui.close()


@pytest.mark.parametrize("bad", ["NaN","inf","-inf"])
def test_n020_nonfinite_time_rejected(tmp_path, bad):
    from dynalyst.analysis.parameter_estimation import inspect_dataset
    p = tmp_path/"data.csv"; p.write_text(f"time,X\n0,1\n{bad},2\n2,3\n")
    assert "finite" in json.loads(inspect_dataset(str(p)))["error"]


@pytest.mark.parametrize("engine", ["exact_numpy", "tau_leap"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_n021_nonfinite_propensity_fails(engine, bad):
    from dynalyst.analysis import stochastic_ssa as ssa
    fn = getattr(ssa, engine)
    with pytest.raises(ValueError, match="Non-finite"):
        fn(lambda x: np.full((1,x.shape[1]),bad), np.array([[-1.]]), np.array([1.]),
           np.linspace(0,1,3), np.random.default_rng(0), 1,
           **({"max_events":100} if engine == "exact_numpy" else {}))


def test_n023_xpp_semantics_are_explicit():
    from dynalyst.analysis.xpp_to_sbml import parse_xpp
    with pytest.raises(ValueError, match="line 3"):
        parse_xpp("x'=1\ninit x=0\nglobal 1 x-1 {x=0}\ndone")
    assert parse_xpp("par k=3\nx'=-x\nx(0)=k\ndone")[3] == {"x":3}


def test_n024_fold_sensitivity_undefined():
    from dynalyst.analysis.jacobian_network import _sensitivities
    out = _sensitivities(lambda *x:[[1]], [0], np.array([[0.]]), [sp.Symbol("p")], ["x"])
    assert out["sensitivities"] is None and "sensitivities_dxdp_top" not in out
    # A supposed constraint must not hide an inconsistent steady-state derivative.
    out = _sensitivities(lambda *x:[[1]], [0], np.array([[0.]]), [sp.Symbol("p")], ["x"], [[1]])
    assert out["sensitivities"] is None


@pytest.mark.parametrize("phrase,expected", [("subcritical Hopf bifurcation","hopf_subcritical"),
                                            ("generalized Hopf bifurcation","generalized_hopf")])
def test_n025_specific_hopf_category(phrase, expected):
    from dynalyst.regime.generic_models import resolve_generic_category
    assert resolve_generic_category("Please demonstrate a " + phrase) == expected


def test_f01_typed_equation_prose_and_nonscalar():
    from dynalyst.analysis.typed_equations import parse_typed_system
    parsed = parse_typed_system("dx/dt = -k*x, parameters k=0.4, x(0)=1.")
    assert parsed["odes"] == {"x":"-k*x"} and parsed["parameters"] == {"k":.4}
    assert parse_typed_system("dx/dt = (1,2)") is None


def test_n021_jax_nonfinite_propensity_fails():
    from dynalyst.analysis.ode_model import model_from_odes
    from dynalyst.analysis.stochastic_ssa import exact_jax
    model = model_from_odes(["x"], {"x":"-x"}, {}, {"x":1})
    with pytest.raises(ValueError, match="Non-finite"):
        exact_jax(model, [{"rate":sp.nan}], {}, 1, np.array([[-1.]]), np.array([1.]),
                  np.linspace(0,1,3), 0, 1, 100)


def test_n002_initial_assignment_builtin_constant(tmp_path):
    doc, m = sbml(); param(m, "p", 0); rate(m, "X", "-p*X")
    initial(m, "p", "avogadro/1e23")
    assert load(doc, tmp_path)["param_defaults"]["p"] == pytest.approx(6.02214076)


@pytest.mark.parametrize("rhs,expected", [("-X/3",-1), ("-1e-15*X",-3e-15)])
def test_n018_export_fraction_and_tiny_rate(tmp_path, monkeypatch, rhs, expected):
    from dynalyst.analysis.jaxkm_ode_to_sbml import odes_to_sbml
    monkeypatch.setattr(config, "LOCAL_SBML_MODELS_DIR", tmp_path)
    result = json.loads(odes_to_sbml(["X"], {"X":rhs}, {}, {"X":3},
                                     model_name="fraction", compartment_size=3))
    assert "error" not in result, result
    model = build_symbolic_model(result["path"])
    actual = float(model["rhs"][0].subs(sp.Symbol("X"), 3))
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0)


def test_n009_two_degenerate_roots_do_not_prove_an_equilibrium_line():
    from dynalyst.analysis.steady_states import _summarise_continuum
    x, y = sp.symbols("x y")
    rhs = [x**3*(x-1)**3, -y]
    fn = sp.lambdify([x,y], rhs, "numpy")
    jf = sp.lambdify([x,y], sp.Matrix(rhs).jacobian([x,y]), "numpy")
    f = lambda v: np.asarray(fn(*v), dtype=float)
    jac = lambda v: np.asarray(jf(*v), dtype=float)
    points = [{"state":{"x":i,"y":0}} for i in [0,1]]
    summary, _ = _summarise_continuum(points, ["x","y"], f, jac,
                                     lambda v: np.all(np.abs(f(v)) < 1e-12), rhs)
    assert not summary["equilibrium_line_verified"]
    assert summary["stable_points_found"] == 0 and summary["inconclusive_points_found"] == 2


@pytest.mark.parametrize("declaration", ["par bogus", "init bogus"])
def test_n023_malformed_declarations_are_not_ignored(declaration):
    from dynalyst.analysis.xpp_to_sbml import parse_xpp
    with pytest.raises(ValueError, match="line 2"):
        parse_xpp("x'=1\n" + declaration + "\ndone")
    assert parse_xpp("par k=1,\nx'=-k*x\ninit x=1,\ndone")[3] == {"x":1}


def test_n018_export_preserves_tiny_initial_value(tmp_path, monkeypatch):
    from dynalyst.analysis.jaxkm_ode_to_sbml import odes_to_sbml
    monkeypatch.setattr(config, "LOCAL_SBML_MODELS_DIR", tmp_path)
    result = json.loads(odes_to_sbml(["X"], {"X":"-X"}, {}, {"X":1e-15},
                                     model_name="tiny_initial", compartment_size=2))
    assert "error" not in result, result
    assert math.isclose(build_symbolic_model(result["path"])["y0"]["X"],
                        1e-15, rel_tol=1e-12, abs_tol=0)


def test_n002_initial_assignment_reads_solved_algebraic_rule(tmp_path):
    doc, m = sbml(); param(m, "a", 0); param(m, "z", 0, False)
    rate(m, "X", "a-X")
    rule = m.createAlgebraicRule(); rule.setMath(libsbml.parseL3Formula("z-X-1"))
    initial(m, "a", "z+1")
    assert load(doc, tmp_path)["param_defaults"]["a"] == 3
