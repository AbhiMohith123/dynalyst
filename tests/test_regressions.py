"""One small synthetic case per bug fixed: each failed before its fix."""
import json

import numpy as np
import pandas as pd
import pytest

from dynalyst.analysis.typed_equations import parse_typed_system, save_typed_system


def _typed(text: str) -> str:
    saved = save_typed_system(parse_typed_system(text))
    assert not saved.get("error"), saved
    return saved["file"]


# --- a file model takes the parameter values and starting state it is given -------------
def test_overrides_reach_a_model_read_from_a_file():
    from dynalyst.analysis.ode_model import resolve_model
    f = _typed("dx/dt = a - b*x, a = 1, b = 2, x(0) = 0.5")
    m = resolve_model(f, parameters={"a": 3, "b": "*2"}, initial_conditions={"x": 7})
    assert m["param_defaults"]["a"] == 3 and m["param_defaults"]["b"] == 4
    assert m["y0"]["x"] == 7
    with pytest.raises(ValueError):
        resolve_model(f, parameters={"nope": 1})


# --- a run with NaN values is a failed run, not a settled one -----------------------------
def test_non_finite_values_are_a_failed_run():
    from dynalyst.analysis.jaxkm_tools import time_course_behaviour
    t = np.linspace(0, 10, 50)
    df = pd.DataFrame({"time": t, "a": np.exp(-t), "b": np.where(t > 5, np.nan, 1.0)})
    assert time_course_behaviour(df, ["a", "b"])["verdict"] == "integration failed"


# --- the period of a bursting oscillation is the period of its spikes, not its wiggles ---
def test_period_counts_the_main_spikes_only():
    from dynalyst.analysis.jaxkm_tools import time_course_behaviour
    t = np.linspace(0, 200, 8001)
    spikes = np.exp(-((t % 12.0) - 1.0) ** 2 * 4)          # one spike every 12
    wiggle = 0.05 * np.sin(2 * np.pi * t / 0.8)              # small fast ripple
    df = pd.DataFrame({"time": t, "c": 1 + spikes + wiggle})
    b = time_course_behaviour(df, ["c"])
    assert b["verdict"] == "oscillating"
    assert b["oscillating_species"]["c"]["period"] == pytest.approx(12.0, rel=0.02)


# --- a line of steady states is reported as one, not as many isolated states -------------
def test_a_continuum_of_steady_states_is_flagged():
    from dynalyst.analysis.steady_states import find_steady_states
    res = json.loads(find_steady_states(_typed(
        "dS/dt = -b*S*I, dI/dt = b*S*I - g*I, b = 0.3, g = 0.1, S(0) = 0.99, I(0) = 0.01")))
    assert res["steady_states"] and all(s.get("isolated") is False for s in res["steady_states"])
    assert "continuum" in res["continuum_note"]


def test_isolated_steady_states_are_not_flagged():
    from dynalyst.analysis.steady_states import find_steady_states
    res = json.loads(find_steady_states(_typed("dm/dt = 5 - m, dp/dt = 2*m - 0.5*p, m(0) = 0, p(0) = 0")))
    assert all(s.get("isolated") is not False for s in res["steady_states"])
    assert "continuum_note" not in res


# --- dependent, numerically found conservation laws still reduce -------------------------
def test_dependent_numeric_conservation_laws_reduce():
    import sympy as sp
    from dynalyst.analysis.steady_states import _numeric_elimination
    laws = sp.Matrix([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0]])          # the third is the sum of the first two
    C, pivots, inv = _numeric_elimination(laws)
    assert C.rows == 2 and len(pivots) == 2
    A = np.array(C.tolist(), dtype=float)
    assert np.allclose(np.array(inv.tolist(), dtype=float) @ A[:, pivots], np.eye(2))


# --- a Jacobian that is not finite somewhere is skipped by the parameter screen -----------
def test_signature_of_a_non_finite_jacobian_is_none():
    from dynalyst.analysis.parameter_screen import _signature
    assert _signature(np.array([[np.nan, 0.0], [0.0, -1.0]])) is None
    assert _signature(np.array([[-1.0, 0.0], [0.0, -2.0]])) == (0, None)


# --- the timescale trajectory stops by the clock without solve_ivp events ----------------
def test_timescale_trajectory_is_a_plain_time_course():
    from dynalyst.analysis import model_reduction as mr
    from dynalyst.analysis.ode_model import resolve_model
    m = resolve_model(_typed("dx/dt = -x + y, dy/dt = -10*y, x(0) = 1, y(0) = 1"))
    sol = mr._trajectory(m, m["param_defaults"], t_end=50.0)
    assert sol.success and sol.y.shape[0] == 2 and sol.t[-1] == pytest.approx(50.0)
    assert abs(sol.y[0, -1]) < 1e-6


# --- the final check reads identifiers and thousands separators correctly ----------------
def test_final_check_identifiers_and_separators():
    from dynalyst.orchestration.final_check import unsupported_values
    ev = ['{"uri": "http://identifiers.org/pubmed/11744196", "ec": "ec-code/2.7.1.1", '
          '"ratio": 2986.4743}']
    assert unsupported_values("PubMed 11744196, EC 2.7.1.1, ratio 2,986.47", ev) == []
    assert unsupported_values("a made-up 123.4", ev) == ["123.4"]


# --- follow-ups do not repeat what this run already did ----------------------------------
def test_follow_ups_skip_steps_already_run():
    from dynalyst.orchestration import next_steps
    rec = type("R", (), {})()
    rec.steps = [
        {"kind": "delegation", "tool": "fetch_biomodels_model", "arguments": {"model_id": "M1"},
         "status": "ok", "output": json.dumps({"use_as": "M1"})},
        {"kind": "delegation", "tool": "classify_model", "status": "ok",
         "arguments": {"model_name_or_path": "data/sbml_models/M1.xml"},
         "output": json.dumps({"model": "M1", "independent_dimension": 3})}]
    queries = [x["query"] for x in next_steps.suggest(rec)]
    assert not any(q.startswith(("Classify ", "Download BioModels model ")) for q in queries)
    assert len({next_steps._key(q) for q in queries}) == len(queries)


# --- the CRNT and limit-cycle questions read the user's replies --------------------------


# --- conservation laws: only directions conserved at every state, judged by relative size --
def test_conservation_laws_with_very_different_rate_scales():
    from dynalyst.analysis.ode_model import resolve_model
    from dynalyst.analysis.steady_states import exact_conservation_laws
    # x <-> y conserves x + y; z is made and removed (no law); rates span 1e-7 .. 1e3
    m = resolve_model(_typed("dx/dt = -k1*x + k2*y, dy/dt = k1*x - k2*y, dz/dt = k3 - k4*z, "
                             "k1 = 1e-7, k2 = 1e-6, k3 = 1000, k4 = 10, x(0) = 1, y(0) = 1, z(0) = 0"))
    laws = exact_conservation_laws(m)
    assert len(laws) == 1
    c = [float(v) for v in laws[0]]
    assert c[2] == 0 and c[0] == pytest.approx(c[1])


# --- AUTO reads the species-name line with a 2048-character buffer ------------------------
def test_constants_file_lines_fit_auto(tmp_path):
    from dynalyst.analysis.bifurcation_auto import _write_constants_file
    from dynalyst.analysis.ode_model import resolve_model
    n = 120
    eqs = ", ".join(f"dSpecies_with_a_long_name_{i:03d}/dt = k - Species_with_a_long_name_{i:03d}"
                    for i in range(n))
    m = resolve_model(_typed(eqs + ", k = 1"))
    out = tmp_path / "c.model"
    _write_constants_file(m, "k", 0.5, 2.0, out, 50)
    # the unames line is the one AUTO copies into fort.2 (parnames stays in its Python side)
    unames = next(line for line in out.read_text().splitlines() if line.startswith("unames"))
    assert len(unames) < 2048


# --- a forced model is simulated with its inputs as functions of time ---------------------
def test_a_forced_model_is_simulated_with_time(tmp_path):
    import libsbml
    from dynalyst.analysis.jaxkm_tools import simulate_model
    doc = libsbml.SBMLDocument(3, 1)
    m = doc.createModel()
    m.setId("forced")
    c = m.createCompartment(); c.setId("cell"); c.setSize(1); c.setConstant(True)
    s = m.createSpecies(); s.setId("x"); s.setCompartment("cell"); s.setInitialConcentration(0)
    s.setHasOnlySubstanceUnits(False); s.setBoundaryCondition(False); s.setConstant(False)
    r = m.createRateRule(); r.setVariable("x")
    r.setMath(libsbml.parseL3Formula("2 + sin(time) - x"))   # forcing with no long-time value
    path = tmp_path / "forced.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    res = json.loads(simulate_model(str(path), t_end=100, n_points=2000))
    assert not res.get("error"), res.get("error")
    b = res["behaviour"]
    assert b["verdict"] == "oscillating"
    assert b["oscillating_species"]["x"]["period"] == pytest.approx(2 * np.pi, rel=0.02)


# --- a diverging AUTO run is stopped before it fills the disk ------------------------------
def test_auto_watchdog_stops_a_run_whose_log_keeps_growing(tmp_path, monkeypatch):
    import sys
    from dynalyst.analysis import bifurcation_auto as B
    monkeypatch.setattr(B, "AUTO_LOG_MAX_BYTES", 2**20)      # 1 MB for the test
    writer = ("import time\n"
              "f = open('fort.9', 'w')\n"
              "while True:\n"
              "    f.write('NOTE:Pivot 1 = 0.0 in GE; ' * 5000); f.flush(); time.sleep(0.05)\n")
    cmd = [sys.executable, "-c", writer]
    stopped = B._watch_auto(cmd, tmp_path, None, tmp_path / "o.txt", tmp_path / "e.txt", 120)
    assert stopped == "diverged"
    assert not (tmp_path / "fort.9").exists()                # the runaway log is deleted


def test_auto_watchdog_stops_a_run_past_its_time_limit(tmp_path):
    import sys
    from dynalyst.analysis import bifurcation_auto as B
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    assert B._watch_auto(cmd, tmp_path, None, tmp_path / "o.txt", tmp_path / "e.txt", 3) == "time"


def test_a_journal_link_is_not_template_text():
    from dynalyst.orchestration.final_check import placeholders
    assert placeholders("Markevich (2004). [Journal of Cell Biology](https://doi.org/10.1083/x)") == []
    assert placeholders("[Biophysical Journal](https://doi.org/10.1529/x)") == []
    assert placeholders("1. [Paper 1 Title] in [Journal Name]") == ["[Journal Name]", "[Paper 1 Title]"]


def test_filler_arguments_mean_not_given():
    from dynalyst.orchestration.tools import coerce_args
    args = coerce_args("lhs_bifurcation_scan", {"model": "m", "bif_param": "", "param_min": 0,
                                                 "param_max": 0, "manual_bounds": "", "seed": 0})
    assert args == {"model": "m", "seed": 0}
    kept = coerce_args("run_bifurcation_analysis", {"model_name_or_path": "m", "param_name": "b",
                                                     "param_min": 0.1, "param_max": 2})
    assert kept["param_min"] == 0.1 and kept["param_max"] == 2
    from dynalyst.regime.lhs_tool import _param_range, _width
    assert _param_range(0, 0) is None and _param_range(0.1, 2) == (0.1, 2)
    assert _width(0) is None and _width(0.2) == 0.2




def test_a_derived_width_is_supported_and_an_invented_value_is_not():
    from dynalyst.orchestration.final_check import unsupported_values
    ev = ['{"lo": 39.24612, "hi": 57.38031}']
    assert unsupported_values("bistable for 39.246-57.380 (width 18.134)", ev) == []
    assert unsupported_values("bistable for 39.246-57.380 (width 31.7)", ev) == ["31.7"]


def test_only_a_cited_quote_is_checked_as_a_paper_title():
    from dynalyst.orchestration.final_check import unsupported_titles
    ev = ["Bistability in the JNK cascade. Bagowski CP, Ferrell JE. 2001"]
    cited = "- “[Bistability in the JNK cascade](https://doi.org/10.1016/S0960-9822(01)00330-X),” 2001"
    assert unsupported_titles(cited, ev) == []
    assert unsupported_titles("The Hopf point is “subcritical: the cycles born there are unstable”.", ev) == []
    assert unsupported_titles("- Smith (2019), “An invented title on kinase switches”.", ev) != []



def test_only_the_wrong_paragraph_is_rewritten_and_it_must_keep_its_numbers():
    from dynalyst.orchestration.final_check import (blocks_with, evidence_numbers, keeps_content,
                                                    split_blocks)
    answer = ("Folds at 39.246 and 57.380.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
              "A third fold at 123.4.\n\n![plot](out/p.png)")
    blocks = split_blocks(answer)
    assert blocks_with(blocks, ["123.4"]) == [2]
    ev = evidence_numbers(['{"lp": [39.24612, 57.38031]}'])
    assert keeps_content(blocks[0], "Folds at 39.246 and 57.380.", {"123.4"}, ev)
    assert not keeps_content(blocks[0], "Two folds were found.", {"123.4"}, ev)    # lost numbers
    assert not keeps_content(blocks[3], "See the diagram.", set(), ev)              # lost the plot


def test_the_environment_check_names_what_is_missing(tmp_path, monkeypatch):
    from dynalyst import env_check
    monkeypatch.setattr(env_check, "SEPARATE", {})
    req = tmp_path / "requirements.txt"
    req.write_text("numpy\n# a comment\nnbformat>=4.2   # why\nno-such-package-xyz==1.0\n"
                   "numpy<0.1\n", encoding="utf-8")
    assert env_check.requirements(req) == [("numpy", ""), ("nbformat", ">=4.2"),
                                           ("no-such-package-xyz", "==1.0"), ("numpy", "<0.1")]
    names = [(p["name"], p["have"] is None) for p in env_check.problems(req)]
    assert ("no-such-package-xyz", True) in names
    assert ("numpy", False) in names                    # installed, but not at a version < 0.1
    assert all(n != "nbformat" for n, _ in names)


def test_every_tool_reads_filler_arguments_as_not_given():
    """For every tool: an optional argument written as "", [], {} or a meaningless 0 is
    dropped (the tool's default applies); a real value, a seed of 0 and a flag are kept."""
    from dynalyst.orchestration.tools import (ZERO_IS_A_VALUE, _types, all_tools, coerce_args,
                                              schema_of)
    for name, tool in all_tools().items():
        schema = schema_of(tool)
        required = set(schema.get("required", []))
        for key, prop in schema.get("properties", {}).items():
            if key in required:
                continue
            types = _types(prop)
            if "string" in types and "array" not in types:
                assert key not in coerce_args(name, {key: ""}), (name, key)
            if "array" in types or "object" in types:
                filler = [] if "array" in types else {}
                assert key not in coerce_args(name, {key: filler}), (name, key)
            if ("number" in types or "integer" in types) and not key.endswith(("_min", "_max")):
                kept = key in coerce_args(name, {key: 0})
                assert kept == (key in ZERO_IS_A_VALUE), (name, key)
                assert key in coerce_args(name, {key: 3}), (name, key)
            if "boolean" in types:
                assert coerce_args(name, {key: False}).get(key) is False, (name, key)




def test_a_typed_system_ends_before_its_parameter_range():
    from dynalyst.analysis.typed_equations import parse_typed_system
    s = parse_typed_system("Bifurcation analysis of dx/dt = mu*x - x^3 in mu from -1 to 1, "
                           "with x(0) = 0.1. What type is it?")
    assert s["odes"] == {"x": "mu*x - x**3"} and s["unvalued"] == []
    assert s["parameters"] == {"mu": -1.0} and s["initial_conditions"] == {"x": 0.1}
    given = parse_typed_system("dx/dt = r*x - x^2, r = 0.5, x(0) = 1, in r from 0 to 2")
    assert given["parameters"] == {"r": 0.5}          # a value given stays the value


def test_a_bistable_model_whose_species_start_at_zero_shows_all_its_states():
    from dynalyst.analysis.steady_states import find_steady_states
    d = json.loads(find_steady_states("Markevich2004"))      # Mp = Mpp = 0 at the start
    stable = [s for s in d["steady_states"] if s["stability"].startswith("stable")]
    assert len(d["steady_states"]) == 3 and len(stable) == 2
    assert max(s["state"]["Mpp"] for s in stable) > 400        # the high branch


def test_sensitivities_hold_conserved_totals_fixed():
    from dynalyst.orchestration.tools import execute
    d = json.loads(execute("jacobian_network", {"model_name_or_path": "Markevich2004"}))
    high = max(d["steady_states"], key=lambda s: s["state"]["Mpp"])
    top = {(e["species"], e["parameter"]): e["dx_dp"]
           for e in high["parameter_jacobian"]["sensitivities_dxdp_top"]}
    # checked against a finite difference along the high branch
    assert top[("Mpp", "k1cat")] == pytest.approx(2044.84, rel=1e-3)
    assert top[("M", "k1cat")] == pytest.approx(-1859.36, rel=1e-3)


def test_overriding_a_fixed_species_runs_through_the_generic_reader():
    from dynalyst.analysis.jaxkm_tools import simulate_model
    lo = json.loads(simulate_model("Markevich2004", t_end=2000, parameters={"MAPKK": 30}))
    hi = json.loads(simulate_model("Markevich2004", t_end=2000, parameters={"MAPKK": 70}))
    assert lo["final_concentrations"]["Mpp"] < 50 < 400 < hi["final_concentrations"]["Mpp"]


# --- an XPP model file is found by every tool, by its name with or without the extension --
@pytest.mark.parametrize("name", ["ferrell_cell_cycle_2d.ode", "pdp_mm_bistable.txt", "pdp_mm_bistable"])
def test_xpp_files_are_found_by_name_in_every_tool(name):
    from dynalyst.orchestration.tools import execute
    for tool, args in [("get_model_odes", {}), ("simulate_model", {"t_end": 10}),
                       ("plot_rate_of_change", {"t_end": 5}), ("inspect_model", {})]:
        out = json.loads(execute(tool, {"model_name_or_path": name, **args}))
        assert not out.get("error"), (tool, out.get("error"))


def test_an_unknown_name_still_says_so():
    from dynalyst.analysis.ode_model import resolve_model_file
    with pytest.raises(FileNotFoundError):
        resolve_model_file("no_such_model_anywhere")


# --- a continuum of steady states is summarised, not listed point by point ----------------
def test_a_line_of_steady_states_is_summarised_with_where_its_stability_changes():
    from dynalyst.analysis.steady_states import CONTINUUM_POINTS_SHOWN, find_steady_states
    res = json.loads(find_steady_states(_typed(
        "dS/dt = -b*S*I/N, dI/dt = b*S*I/N - g*I, b = 0.3, g = 0.1, N = 1000, "
        "S(0) = 990, I(0) = 10")))
    c = res["continuum"]
    assert c["fixed_on_it"] == {"I": 0.0} and list(c["sampled_range"]) == ["S"]
    seg = c["stability_along_it"]
    assert [s["stability"] for s in seg] == ["stable", "unstable"]
    assert seg[0]["to"] == pytest.approx(0.1 * 1000 / 0.3, rel=1e-5)    # S* = g N / b
    assert len(res["steady_states"]) <= CONTINUUM_POINTS_SHOWN
    assert res["n_distinct_steady_states"] == 0


# --- solutions with a negative value are reported, not sent back as 'none found' ----------


# --- a share or a change in percent of two quoted values is supported by them -------------
def test_percentages_of_quoted_values_are_supported():
    from dynalyst.orchestration.final_check import unsupported_values
    evidence = ["counts: hopf 19, none 1", "20 samples", "fold at 50", "fold at 60"]
    assert unsupported_values("19 of 20 samples (95%) oscillate", evidence) == []
    assert unsupported_values("the fold moved from 50 to 60, up 20%", evidence) == []
    assert unsupported_values("a fold at 123.4", evidence) == ["123.4"]


# --- a model file named with its extension is that file ----------------------------------


# --- the candidate screen looks further when nothing is found near the values -------------
def test_a_crossing_four_decades_from_the_parameter_value_is_found():
    from dynalyst.analysis.candidates import propose
    # a fold at p = c = 1000, four decades above p's value 0.1
    res = propose(_typed("dx/dt = p - c + x^2, p = 0.1, c = 1000, x(0) = -30"))
    near = {c["parameter"]: c.get("crossing_near") for c in res["candidates"]}
    assert near.get("p") == pytest.approx(1000, rel=0.05)
    assert res["searched"].startswith("5 decade")


# --- the parameter named as the bifurcation parameter outranks one said to 'vary' --------


# --- a reply naming a parameter the approval did not number still runs it ----------------
def test_a_reply_can_name_a_parameter_beyond_the_numbered_candidates():
    from dynalyst.orchestration.approval import parse_name
    others = [{"parameter": "p1"}, {"parameter": "p2"}]
    assert parse_name("p2", others) == ("approve", 1)
    assert parse_name("run it in p1 please", others) == ("approve", 0)
    assert parse_name("p1 and p2", others) == ("other", None)


# --- a lone state with extra zero eigenvalues is a steady state, not a continuum ----------
def test_a_lone_degenerate_state_is_counted():
    from dynalyst.analysis.steady_states import find_steady_states
    # x = 0 has eigenvalue 0 from the x^2 term at the origin, with no line of states through it
    res = json.loads(find_steady_states(_typed("dx/dt = -x^3, dy/dt = -y, x(0) = 1, y(0) = 1")))
    assert res["n_distinct_steady_states"] == 1 and "continuum" not in res
    assert "degenerate" in res["steady_states"][0]["stability"]


# --- a model's declared time unit is shown, and a delay run too long for it is refused ---
def test_the_time_unit_is_shown_and_an_overlong_delay_run_is_refused():
    from dynalyst.orchestration.tools import execute
    info = json.loads(execute("inspect_model", {"model_name_or_path": "BIOMD0000000024"}))
    assert info["time_unit"].startswith("hour") and "delay_differential_equation" in info
    out = json.loads(execute("simulate_model", {"model_name_or_path": "BIOMD0000000024",
                                                "t_end": 720000}))
    assert "delay intervals" in out["error"] and "time unit" in out["error"]


def test_xpp_files_get_a_clear_answer_where_only_sbml_is_meant():
    from dynalyst.orchestration.tools import execute
    out = json.loads(execute("get_annotations", {"model_name_or_path": "pdp_mm_bistable.txt"}))
    assert "XPP file" in out["error"]


# --- relative values work in the steady-state tools, as in the simulation tools ----------
def test_relative_fixed_params_in_steady_states():
    from dynalyst.analysis.steady_states import find_steady_states
    f = _typed("dm/dt = a - m, a = 5, m(0) = 0")
    res = json.loads(find_steady_states(f, fixed_params={"a": "*2"}))
    assert res["steady_states"][0]["state"]["m"] == pytest.approx(10.0)


# --- a value or factor nobody gave is rejected; one the request states is not -----------


# --- a continuum's stability change is not called bistability ----------------------------
def test_a_continuum_is_said_not_to_be_bistability():
    from dynalyst.analysis.steady_states import find_steady_states
    res = json.loads(find_steady_states(_typed(
        "dS/dt = -b*S*I/N, dI/dt = b*S*I/N - g*I, b = 0.3, g = 0.1, N = 1000, "
        "S(0) = 990, I(0) = 10")))
    assert "NOT bistability" in res["interpretation"]


def test_the_reviewer_accepts_solutions_reported_as_non_physical():
    from dynalyst.orchestration.reviewer import Reviewer
    data = {"steady_states": [], "non_physical_solutions": [{"state": {"v": -0.8, "w": -0.13}}],
            "non_physical_note": "1 further solution ..."}
    v = Reviewer().review("find_steady_states", {"model_name_or_path": "m.xml"}, json.dumps(data))
    assert v.verdict != "replan"
    v = Reviewer().review("find_steady_states", {"model_name_or_path": "m.xml"},
                          json.dumps({"steady_states": []}))
    assert v.verdict == "replan"


def test_an_lhs_signature_counts_as_a_found_point():
    from dynalyst.orchestration.final_check import unfound_bifurcation_claim
    steps = [{"kind": "delegation", "tool": "lhs_bifurcation_scan",
              "output": "Bifurcation signatures found: {'2 LP (bistable)': 3, 'none': 7}"}]
    assert not unfound_bifurcation_claim("The scan found two limit points.", steps)



# --- the CRNT and limit-cycle questions read the user's replies --------------------------
@pytest.mark.parametrize("reply,choice", [("continue", "continue"), ("yes", "continue"),
                                          ("plot", "plot"), ("stop and plot", "plot"),
                                          ("no", "plot"), ("what about another model?", "other")])
def test_crnt_wait_replies(reply, choice):
    from dynalyst.orchestration.approval import parse_crnt_wait
    assert parse_crnt_wait(reply) == choice



def test_folds_the_answer_leaves_out_are_found():
    from dynalyst.orchestration.final_check import unreported_folds
    out = json.dumps({"swept_parameter": "k", "classification": {
        "category": "bistable", "evidence": {"counts": {"LP": 2, "HB": 2},
                                             "max_coexisting_stable": 2, "at": 0.2988}}})
    steps = [{"kind": "delegation", "tool": "run_bifurcation_analysis", "output": out}]
    assert unreported_folds("Two Hopf points, at k = 0.30 and 0.42.", steps) == [
        "2 fold(s) (LP) in k; 2 stable steady states coexist at k = 0.2988 (bistable)"]
    assert unreported_folds("Hopf points at 0.30 and 0.42; two folds bound a bistable window.",
                            steps) == []



def test_replies_to_the_answer_review_are_removed_but_no_result_is_lost():
    from dynalyst.orchestration.final_check import tidy_answer
    answer = ("The review is right — I presented the detected bifurcations as the model's complete "
              "set, but the continuation only followed branches from stress = 0. I need to qualify "
              "the conclusions properly. Let me write the corrected answer.\n\n"
              "**Answer:** two Hopf points, at stress = 0.1803 and 1.3991, followed from stress = 0.\n"
              "You are right that the fold sits at stress = 2.9376.")
    out = tidy_answer(answer)
    assert "review is right" not in out and "Let me write" not in out and "I need to" not in out
    assert "0.1803" in out and "1.3991" in out
    # the only place 2.9376 appears: the reply sentence stays, so the result is not lost
    assert "2.9376" in out


def test_a_decimal_point_does_not_end_a_review_reply():
    from dynalyst.orchestration.final_check import tidy_answer
    out = tidy_answer("The model has 2 folds. The review is right: the fold is at 2.26, not 2.3. "
                      "Both are stable.")
    assert "2.26" in out and ".26," not in out.replace("2.26", "")
    out = tidy_answer("The model has 2 folds. The review is right. Folds at 2.26 and 2.94.")
    assert "review is right" not in out and "2.26 and 2.94" in out


def test_three_coexisting_stable_states_are_called_tristable(monkeypatch):
    from dynalyst.analysis import bifurcation_auto
    from dynalyst.orchestration.final_check import unreported_folds
    from dynalyst.orchestration.next_steps import _suggest_for
    from dynalyst.regime import taxonomy
    monkeypatch.setattr(taxonomy, "classify_diagram", lambda *a, **k: {
        "category": "bistable", "features": ["bistable", "hopf"],
        "evidence": {"counts": {"LP": 4, "HB": 2}, "max_coexisting_stable": 3, "at": 2.95}})
    pts = [{"type": "LP", "stress": v} for v in (2.26, 2.94, 4.0, 4.46)]
    c = bifurcation_auto._classification(pts, [], "stress", "AMPK")
    assert c["multistability"] == "tristable" and c["reason"].startswith("tristable: 3 stable")
    assert "tristable" in c["category_note"]
    out = json.dumps({"swept_parameter": "stress", "classification": c})
    steps = [{"kind": "delegation", "tool": "run_bifurcation_analysis", "output": out}]
    assert unreported_folds("Two Hopf points.", steps)[0].endswith("(tristable)")
    step = {"tool": "run_bifurcation_analysis", "status": "ok", "sub_agent": "bifurcation",
            "arguments": {"model_name_or_path": "m"},
            "output": json.dumps({"swept_parameter": "stress", "bifurcation_points": pts})}
    texts = [q for q, _ in _suggest_for(step, set())]
    assert any(t.startswith("4 folds were found in stress") for t in texts)
