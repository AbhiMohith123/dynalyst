"""The generic range-resolution layer, the sample taxonomy, goal-driven LHS and
the tools built on them. Everything here runs without AUTO (the AUTO step is
stubbed by the `stubbed_auto` fixture in conftest.py)."""
import csv

import numpy as np
import pytest

from dynalyst.regime import batch_lhs_runner, lhs_goals, range_spec, taxonomy
from dynalyst.regime.range_spec import RangeSpec, resolve_bounds

BASE = {"a": 2.0, "b": 10.0, "c": 0.5, "d": 4.0, "z": 0.0}


# ---------------------------------------------------------------------------
# range resolution
# ---------------------------------------------------------------------------

def test_default_is_a_log_fold_around_the_baseline():
    r = resolve_bounds(BASE, RangeSpec(exclude={"d"}))
    assert "d" not in r.bounds
    assert r.bounds["a"] == (1.0, 4.0, True)                 # fold 2: baseline/2 .. baseline*2
    assert r.bounds["z"][0] == r.bounds["z"][1] == 0.0       # non-positive baseline is held
    assert r.strategy == ("fold", 2.0)


def test_precedence_fixed_vary_only_exact_window_global():
    spec = RangeSpec(rel=0.1, vary_only=["a", "b", "c"], fixed={"a": 3.0},
                     manual={"b": (8, 12)}, windows={"c": {"fold": 4}})
    r = resolve_bounds(BASE, spec)
    assert r.bounds["a"] == (3.0, 3.0, False)                # fixed wins over everything
    assert r.bounds["b"][:2] == (8.0, 12.0)                  # exact bounds beat the global window
    assert r.bounds["c"][:2] == (0.125, 2.0)                 # per-parameter window beats the global
    assert r.bounds["d"][0] == r.bounds["d"][1] == 4.0       # not in vary_only: held at baseline


def test_global_window_order_rel_then_offset_then_fold():
    assert resolve_bounds(BASE, RangeSpec(rel=0.1, offset=5, fold=3)).strategy[0] == "rel"
    assert resolve_bounds(BASE, RangeSpec(offset=5, fold=3)).strategy[0] == "offset"
    assert resolve_bounds(BASE, RangeSpec(fold=3)).strategy[0] == "fold"
    assert resolve_bounds(BASE, RangeSpec(offset=5)).bounds["a"][:2] == (2.0, 7.0)


def test_changed_baselines_recentre_windows_and_held_values():
    r = resolve_bounds(BASE, RangeSpec(rel=0.1, baseline={"a": 20.0}, vary_only=["a"]))
    assert r.bounds["a"][:2] == pytest.approx((18.0, 22.0))
    assert r.bounds["b"][:2] == (10.0, 10.0) and r.baseline["a"] == 20.0
    r = resolve_bounds(BASE, RangeSpec(baseline={"b": 7.0}, vary_only=["a"]))
    assert r.bounds["b"][:2] == (7.0, 7.0)                   # held at the CHANGED baseline
    r = resolve_bounds(BASE, RangeSpec(fixed={"b": None}, baseline={"b": 7.0}))
    assert r.bounds["b"][:2] == (7.0, 7.0)


def test_percentages_use_the_allowed_range_and_never_leave_it():
    r = resolve_bounds(BASE, RangeSpec(rel=0.01, allowed={"b": (0.0, 1000.0), "a": (1.99, 50.0)}))
    assert r.bounds["b"][:2] == pytest.approx((0.0, 20.0))   # 1% of 1000 is 10: [0, 20], floor-clipped
    assert r.bounds["a"][0] == pytest.approx(1.99) and r.bounds["a"][1] == pytest.approx(2.0 + 0.4801)
    assert r.bounds["c"][:2] == pytest.approx((0.495, 0.505))  # no allowed range: 1% of the baseline


def test_contradictions_are_reported_not_hidden():
    spec = RangeSpec(rel=0.1, vary_only=["a"], fixed={"a": 1.0}, manual={"b": (0, 1000)},
                     windows={"d": {"rel": 0.5}, "c": {"rel": 0.2}}, allowed={"b": (1, 100)},
                     exclude={"c"})
    r = resolve_bounds(BASE, spec)
    text = " | ".join(r.conflicts)
    assert "both fixed and given a range" in text          # a
    assert "not in vary_only" in text                      # b and d carry explicit ranges
    assert "outside its allowed range" in text             # b is clipped
    assert "continuation parameter" in text                # c is excluded
    assert r.bounds["b"][:2] == (1.0, 100.0)


def test_unknown_names_raise_and_the_spec_is_not_mutated():
    with pytest.raises(ValueError, match="not parameters here"):
        resolve_bounds(BASE, RangeSpec(vary_only=["nope"]))
    with pytest.raises(ValueError, match="fixed"):
        resolve_bounds(BASE, RangeSpec(fixed={"nope": 1}))
    spec = RangeSpec(windows={"a": {"abs": (1, 3)}})
    resolve_bounds(BASE, spec)
    assert spec.manual == {}                                # the abs window did not leak into it


def test_tool_json_arguments():
    ex = range_spec.extras_from_tool_args('{"a": 3}', '["b", "c"]',
                                          '{"d": {"rel": 0.1}, "z": {"abs": [0, 1]}}')
    assert ex.baseline == {"a": 3.0} and ex.fixed == {"b": None, "c": None}
    assert ex.windows["d"] == {"rel": 0.1}
    assert range_spec.extras_from_tool_args(None, '{"b": 2}', None).fixed == {"b": 2.0}
    with pytest.raises(ValueError, match="valid JSON"):
        range_spec.extras_from_tool_args("{bad", None, None)
    with pytest.raises(ValueError, match="unknown keys"):
        range_spec.extras_from_tool_args(None, None, '{"a": {"wobble": 5}}')
    # aliases are understood: "percent" is a +-% window
    assert range_spec.extras_from_tool_args(None, None, '{"a": {"percent": 5}}').windows == \
        {"a": {"rel": 0.05}}


def test_the_same_engine_samples_whatever_way_the_range_was_asked_for():
    from dynalyst.regime.lhs_sampling import lhs_sample_params
    asks = [RangeSpec(rel=0.1), RangeSpec(offset=1.0), RangeSpec(fold=1.5),
            RangeSpec(manual={"a": (1, 3), "b": (5, 15), "c": (0.2, 0.9), "d": (3, 5)})]
    for spec in asks:
        b = resolve_bounds(BASE, spec).bounds
        for s in lhs_sample_params(b, 8, seed=0):
            assert all(b[p][0] <= v <= b[p][1] for p, v in s.items())


def test_the_default_is_a_fold_around_the_baseline_not_an_absolute_offset():
    from dynalyst.regime import model_source
    from dynalyst.regime.lhs_sampling import resolve_model_bounds
    parsed = model_source.load_model("pdp_mm_bistable.txt")["parsed"]
    res = resolve_model_bounds(parsed, exclude_params=["ATOT"])
    assert "ATOT" not in res.bounds and res.strategy == ("fold", 2.0)
    lo, hi, log = res.bounds["km5"]
    assert (lo, hi) == pytest.approx((2.55, 10.2)) and log      # baseline 5.1
    res = resolve_model_bounds(parsed, additive_offset=200.0, exclude_params=["ATOT"])
    assert res.bounds["km5"][:2] == pytest.approx((5.1, 205.1))  # still available on request


# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------

class StableBranch(dict):
    def stability(self):
        return [-len(self["x"])]


def _entry(*types, error=None, coexist=False):
    special = [(i, t, i + 1) for i, t in enumerate(types)]
    branches = ([StableBranch(x=np.array([0., 1.]), y=np.array([v, v])) for v in (0., 2.)]
                if coexist else [])
    return {"error": error, "special": None if error else special, "all_branches": branches,
            "signature": None, "params": {}}


def test_taxonomy_primary_category_and_features():
    def a(*t, **k):
        return taxonomy.analyse(_entry(*t, **k), "x", "y")
    assert a("LP", "LP")["category"] == "multiple_folds"
    assert a("LP", "LP", "LP", "LP")["category"] == "multiple_folds"
    assert a("LP", "LP", coexist=True)["category"] == "bistable"
    assert a("LP")["features"] == ["single_fold"]
    assert a("BP")["features"] == ["branch_point"]
    assert a("HB")["category"] == "hopf"
    both = a("LP", "LP", "BP", "HB", coexist=True)
    assert both["category"] == "bistable" and both["features"] == ["bistable", "branch_point", "hopf"]
    assert a("LP", "BP")["features"] == ["branch_point", "single_fold"]      # BP outranks a lone fold
    assert a(error="AUTORuntimeError")["category"] == "failed"
    assert a()["category"] == "no_bifurcation"                                # no branch to rate


def test_taxonomy_splits_monotonic_samples_by_response_ratio():
    def entry_for(x, y):
        return {"error": None, "special": [], "signature": "none", "params": {},
                "all_branches": [{"x": x, "y": y(x)}]}
    steep = taxonomy.analyse(entry_for(np.linspace(0.1, 10, 400),
                                       lambda x: 1 / (1 + np.exp(-(x - 5) * 4))), "x", "y")
    # a response spread over six decades: its 10%-90% ratio is far above 81
    shallow = taxonomy.analyse(entry_for(np.geomspace(1e-3, 1e3, 400), np.log10), "x", "y")
    assert steep["category"] == "ultrasensitive" and shallow["category"] == "graded"


def test_categorize_results_primary_partitions_and_multi_label_repeats():
    from dynalyst.regime.categorize_lhs import categorize_results
    rs = [dict(_entry("LP", "LP", "BP", coexist=True), sample_id=0, params={}),
          dict(_entry("BP"), sample_id=1, params={}),
          dict(_entry(error="boom"), sample_id=2, params={})]
    prim = categorize_results(rs, "x", "y")
    assert {k: len(v) for k, v in prim.items()} == {"bistable": 1, "branch_point": 1}
    multi = categorize_results(rs, "x", "y", multi=True)
    assert {k: len(v) for k, v in multi.items()} == {"bistable": 1, "branch_point": 2}


# ---------------------------------------------------------------------------
# goals
# ---------------------------------------------------------------------------

def test_goals_parse_and_set_the_default_window_only_when_nothing_else_is_given():
    assert [g.name for g in lhs_goals.parse_goals("Bistability")] == ["bistability"]
    assert [g.name for g in lhs_goals.parse_goals("pitchfork, transcritical")] == ["branch_point"]
    assert [g.name for g in lhs_goals.parse_goals("bistable and ultra-sensitive")] == \
        ["bistability", "ultrasensitivity"]
    assert lhs_goals.parse_goals(None) == []
    with pytest.raises(ValueError, match="unknown goal"):
        lhs_goals.parse_goals("chaos")
    _, extras, control = lhs_goals.prepare("bistability", None)
    assert control and extras.default_fold == 1.5
    _, extras2, _ = lhs_goals.prepare("bistability, ultrasensitivity", None)
    assert extras2.default_fold == 1.5                    # the smaller: start near the baseline
    # the goal's fold is only a default: an explicit percentage still wins
    r = resolve_bounds(BASE, RangeSpec(rel=0.05, default_fold=extras.default_fold))
    assert r.strategy == ("rel", 0.05)
    r = resolve_bounds(BASE, RangeSpec(default_fold=extras.default_fold))
    assert r.strategy == ("fold", 1.5) and r.bounds["a"][:2] == pytest.approx((2 / 1.5, 3.0))


def _batch(entries, control=None, strategy=("fold", 2.0)):
    out = batch_lhs_runner.BatchResults(entries)
    out.control = control
    out.resolution = range_spec.Resolution({}, {}, {}, [], strategy)
    return out


def _sample(i, *types, error=None, a=1.0):
    # Goal-report examples include an explicit stable coexistence witness.
    e = dict(_entry(*types, error=error, coexist=types.count("LP") >= 2),
             sample_id=i, params={"a": a + i, "b": 5.0})
    e["signature"] = batch_lhs_runner.classify_signature(e["special"] or [])
    return e


def test_goal_report_found_reports_where():
    rs = _batch([_sample(0, "LP", "LP"), _sample(1), _sample(2, "LP", "LP")])
    text = lhs_goals.report(lhs_goals.parse_goals("bistability"), rs, "x", "y")
    assert "FOUND bistability" in text and "2/3 samples" in text and "a in [1, 3]" in text
    assert "NOT FOUND" not in text


def test_goal_report_not_found_explains_and_suggests_when_the_baseline_has_it():
    control = _sample(0, "LP", "LP")
    control["sample_id"] = "control"
    rs = _batch([_sample(i, error="AUTORuntimeError: boom") if i < 3 else _sample(i)
                 for i in range(5)], control=control)
    text = lhs_goals.report(lhs_goals.parse_goals("bistability"), rs, "x", "y")
    assert "NOT FOUND" in text and "What the samples did show instead" in text
    assert "3/5 samples failed" in text and "AUTORuntimeError: boom" in text
    assert "baseline has it" in text                      # control shows bistable, samples lost it
    assert text.index("Narrow the search") < text.index("Widen the search")     # narrow first
    assert "spread_factor=1.5" in text and "spread_factor=3" in text
    assert "not verified" in text


def test_goal_report_baseline_without_the_feature_leads_with_widening():
    control = _sample(0)
    control["sample_id"] = "control"
    rs = _batch([_sample(i) for i in range(4)], control=control, strategy=("rel", 0.01))
    text = lhs_goals.report(lhs_goals.parse_goals("branch point"), rs, "x", "y")
    assert "does not show it" in text and "exact relations" in text
    assert text.index("Widen the search") < text.index("Narrow the search")
    assert "rel_window=0.02" in text and "rel_window=0.005" in text


def test_goal_report_explicit_ranges_are_not_second_guessed():
    rs = _batch([_sample(i) for i in range(3)], strategy=("explicit", None))
    text = lhs_goals.report(lhs_goals.parse_goals("ultrasensitivity"), rs, "x", "y")
    assert "honoured as given" in text


def test_goal_report_is_empty_without_a_goal():
    assert lhs_goals.report([], _batch([_sample(0)]), "x", "y") == ""


# ---------------------------------------------------------------------------
# the tools, end to end with AUTO stubbed
# ---------------------------------------------------------------------------

def test_one_overlay_always_and_the_split_only_on_request(stubbed_auto, monkeypatch):
    from dynalyst.regime import categorize_lhs
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan
    monkeypatch.setattr(categorize_lhs, "plot_categorized", lambda *a, **k: a[3])
    monkeypatch.setattr(categorize_lhs, "plot_category_counts", lambda *a, **k: a[2])
    monkeypatch.setattr(categorize_lhs, "write_category_excels",
                        lambda buckets, out_dir: {c: f"{c}.xlsx" for c in buckets})
    monkeypatch.setattr("dynalyst.regime.interactive_plotting.plot_categorized_interactive",
                        lambda *a, **k: None)
    monkeypatch.setattr("dynalyst.regime.interactive_plotting.plot_category_counts_interactive",
                        lambda *a, **k: None)
    stubbed_auto["specials"][:] = [[(1, "LP", 1), (2, "LP", 2)], [], [(1, "BP", 1)]]
    out = lhs_bifurcation_scan.func(model="Markevich2004", n_samples=6, bif_param="k2cat", seed=1)
    assert "Combined plot:" in out and "Per-category" not in out
    assert ("Categories (primary, per sample): "
            "{'branch_point': 2, 'multiple_folds': 2, 'no_bifurcation': 2}") in out
    out = lhs_bifurcation_scan.func(model="Markevich2004", n_samples=6, bif_param="k2cat", seed=1,
                                    separate=True)
    assert "Combined plot:" in out and "Per-category subplots:" in out
    assert "multiple_folds.xlsx" in out and "branch_point.xlsx" in out


def test_goal_adds_a_control_run_the_goal_report_and_a_near_baseline_default(stubbed_auto):
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan
    stubbed_auto["specials"][:] = [[(1, "LP", 1)]]          # single folds only: no bistability
    out = lhs_bifurcation_scan.func(model="pdp_mm_bistable.txt", bif_param="ATOT", n_samples=4,
                                    goal="bistability", seed=3)
    assert any(c.endswith("control") for c in stubbed_auto["calls"])
    assert "Goal check:" in out and "NOT FOUND bistability" in out
    assert "Baseline control:" in out and "Suggestions:" in out
    assert "a fold of 1.5 around" in out                     # the goal's default window, not +200


def test_explicit_ranges_beat_the_goal_default(stubbed_auto):
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan
    out = lhs_bifurcation_scan.func(
        model="pdp_mm_bistable.txt", bif_param="ATOT", n_samples=4, goal="bistability", seed=3,
        rel_window=0.02, fixed_params='{"KTOT": 480}', baseline_values='{"k4": 14}')
    assert "within +-2%" in out and "a fold of" not in out
    assert "KTOT: held at 480" in out and "(changed baseline)" in out


def test_no_goal_means_no_control_run_and_no_goal_section(stubbed_auto):
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan
    out = lhs_bifurcation_scan.func(model="Markevich2004", n_samples=3, bif_param="k2cat")
    assert "Goal check" not in out
    assert not any(c.endswith("control") for c in stubbed_auto["calls"])


def test_a_scan_where_every_sample_fails_is_reported_not_crashed(monkeypatch, tmp_path):
    # the user's original 20/20-failed case, through the categorized tool with real plotting
    import os
    from dynalyst.regime.categorize_tool import categorized_lhs_scan

    def all_fail(samples, bif_param, param_range, out_dir, y_species, parsed_base, unavailable=None, **_):
        return batch_lhs_runner.BatchResults(
            {"sample_id": i, "params": s, "all_branches": None, "special": None, "signature": None,
             "error": "AUTORuntimeError: Error running AUTO"} for i, s in enumerate(samples))

    monkeypatch.setattr(batch_lhs_runner, "_run_samples", all_fail)
    out = categorized_lhs_scan.func(model="Markevich2004", n_samples=4, bif_param="k2cat", seed=0)
    assert "Category counts: {}" in out
    figure = out.split("Subplot figure: ")[1].split("\n")[0]
    assert os.path.exists(figure)


def test_the_summary_csv_carries_category_features_and_the_control_row(stubbed_auto, tmp_path):
    stubbed_auto["specials"][:] = [[(1, "LP", 1), (2, "LP", 2), (3, "BP", 3)]]
    rs = batch_lhs_runner.run_lhs_batch_for_model("Markevich2004", 3, bif_param="k2cat",
                                                  control=True, out_dir=str(tmp_path))[0]
    for r in list(rs) + [rs.control]:
        taxonomy.analyse(r, "k2cat", "Mp")
    path = batch_lhs_runner.write_summary_csv(rs, str(tmp_path / "s.csv"))
    rows = list(csv.DictReader(open(path)))
    assert rows[0]["sample_id"] == "control" and len(rows) == 4
    assert rows[1]["category"] == "branch_point" and rows[1]["features"] == "branch_point, multiple_folds"
    assert rows[1]["LP_count"] == "2" and rows[1]["BP_count"] == "1"
    assert rows[1]["max_coexisting_stable"] == ""


# ---------------------------------------------------------------------------
# refusals with a reason, and the Hopf goal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model, name, value, basis", [
    ("brannmark", "insulin", 10.0, "long-time"),                      # two steps: 1.2 then +8.8
    ("Beer_MolBioSystems2014", "lag", 1.0, "long-time"),              # a step at tau
    ("Fiedler_BMC2016", "k1max", None, "long-time"),                  # -> k10
    ("Boehm_JProteomeRes2014", "BaF3_Epo", 1.25e-7, "time 0"),        # decays to 0: dose at t=0
])
def test_time_dependent_inputs_are_frozen_into_parameters(model, name, value, basis):
    import sympy as sp
    from dynalyst.analysis.ode_model import model_from_sbml, time_inputs_note
    m = model_from_sbml(model)
    (t,) = m["time_inputs"]
    assert t["input"] == name and basis in t["basis"]
    if value is not None:
        assert t["value"] == pytest.approx(value)
    else:
        assert t["value"] == pytest.approx(m["param_defaults"]["k10"]) and t["long_time_expression"] == "k10"
    assert name in m["param_names"] and m["param_defaults"][name] == t["value"]
    free = set().union(*(sp.sympify(e).free_symbols for e in m["rhs"]))
    assert sp.Symbol(name) in free and not any(s.name == "__sbml_time__" for s in free)
    assert name in time_inputs_note(m)


def test_forcing_without_a_long_time_value_is_refused_with_the_reason():
    import sympy as sp
    from dynalyst.analysis.generic_sbml import SBML_TIME, TimeDependentModel, freeze_time_input
    with pytest.raises(TimeDependentModel, match="no long-time value"):
        freeze_time_input("u", sp.sin(SBML_TIME), {})
    assert freeze_time_input("u", 2 + sp.exp(-SBML_TIME), {})[0] == pytest.approx(2.0)


def test_sbml_comparisons_and_logic_are_read():
    import libsbml
    from dynalyst.analysis.generic_sbml import eval_ast_sympy
    import sympy as sp
    x, y = sp.symbols("x y")
    ast = libsbml.parseL3Formula("piecewise(1, (x < y) && !(x >= 3), 0)")
    e = eval_ast_sympy(ast, {"x": x, "y": y})
    values = [float(e.subs({x: a, y: b})) for a, b in ((1, 2), (4, 5), (2, 1))]
    assert values == [1.0, 0.0, 0.0]


def test_time_note_goes_on_analysis_results_not_on_simulations():
    import json
    from dynalyst.orchestration import tools as registry
    assert "simulate_model" not in registry.FROZEN_INPUT_TOOLS
    t = registry.all_tools()
    d = json.loads(t["find_steady_states"].invoke({"model_name_or_path": "Fiedler_BMC2016"}))
    assert "error" not in d and "k1max" in d["time_dependent_inputs"]
    d = json.loads(t["find_steady_states"].invoke({"model_name_or_path": "Markevich2004"}))
    assert "time_dependent_inputs" not in d


def test_lhs_scan_of_a_time_dependent_model_uses_the_frozen_input(stubbed_auto):
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan
    out = lhs_bifurcation_scan.func(model="brannmark.xml", bif_param="insulin", n_samples=3, seed=0)
    assert stubbed_auto["bif"] == "insulin"
    assert "This model is time-dependent" in out and "frozen at 10" in out


def test_nothing_to_sample_is_refused_with_the_reason():
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan
    with pytest.raises(ValueError, match="only parameter its equations use is alpha"):
        lhs_bifurcation_scan.func(model="saddlenode.xml", n_samples=3)
    with pytest.raises(ValueError, match="every parameter other than"):
        lhs_bifurcation_scan.func(model="p_dynamics.xml", n_samples=3, fixed_params='["k2"]')


def test_a_name_shared_by_different_models_is_refused_everywhere_and_paths_work():
    from dynalyst.analysis.generic_sbml import resolve_sbml_path
    from dynalyst.analysis.jaxkm_tools import _find_sbml
    for resolve in (resolve_sbml_path, _find_sbml):
        with pytest.raises(ValueError, match="name of 2 different models"):
            resolve("pitchfork.xml")
    a = resolve_sbml_path("models/pitchfork.xml")
    b = resolve_sbml_path("data/sbml_models/pitchfork.xml")
    assert a.parent.name == "models" and b.parent.name == "sbml_models"
    assert resolve_sbml_path("Markevich2004").name == "Markevich2004.xml"      # unique names unchanged


def test_hopf_goal_parses_and_is_impossible_in_one_dimension():
    assert [g.name for g in lhs_goals.parse_goals("oscillations")] == ["hopf"]
    assert [g.name for g in lhs_goals.parse_goals("Hopf, bistability")] == ["hopf", "bistability"]
    rs = _batch([_sample(i) for i in range(3)])
    rs.n_states = 1
    text = lhs_goals.report(lhs_goals.parse_goals("hopf"), rs, "x", "y")
    assert "IMPOSSIBLE" in text and "at least 2" in text and "Widen" not in text
    rs = _batch([_sample(0), _sample(1, "HB"), _sample(2)])
    rs.n_states = 3
    text = lhs_goals.report(lhs_goals.parse_goals("hopf"), rs, "x", "y")
    assert "FOUND a Hopf point" in text and "1/3 samples" in text


def test_coexistence_counts_only_inside_the_range_asked_for():
    import numpy as np
    from dynalyst.regime.taxonomy import classify_diagram, max_coexisting_stable
    x = np.linspace(0.0, 1.0, 50)
    curves = [(x, 1 + x, np.ones(50, dtype=int)), (x[:15], 5 + x[:15], np.ones(15, dtype=int))]   # second branch stops at 0.29
    assert max_coexisting_stable(curves)[0] == 2
    assert max_coexisting_stable(curves, within=(0.5, 1.0))[0] == 1
    assert max_coexisting_stable(curves, within=(2.0, 3.0)) == (0, None)
    assert classify_diagram([], curves, "k", "X", within=(0.5, 1.0))["category"] != "bistable"
    assert classify_diagram([], curves, "k", "X", within=(0.0, 0.2))["category"] == "bistable"
