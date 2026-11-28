"""Textbook systems whose bifurcation points and steady states are known in closed form:
the tools must find them (AUTO-07p tests need a built AUTO, marked 'auto')."""
import json

import numpy as np
import pytest

from dynalyst.analysis.typed_equations import parse_typed_system, save_typed_system


def _typed(text: str) -> str:
    system = parse_typed_system(text)
    assert system is not None, text
    saved = save_typed_system(system)
    assert not saved.get("error"), saved
    return saved["file"]


def _points(res: dict, kind: str, param: str) -> list:
    return sorted(float(p[param]) for p in res.get("bifurcation_points") or [] if p.get("type") == kind)


def _continue(text, **kw) -> dict:
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    res = json.loads(run_bifurcation_analysis(_typed(text), **kw))
    assert not res.get("error"), res.get("error")
    return res


@pytest.mark.auto
def test_spruce_budworm_folds():
    # nontrivial equilibria r = x / ((1 + x^2)(1 - x/10)); its extrema are the two folds
    x = np.linspace(0.01, 9.99, 400001)
    r = x / ((1 + x ** 2) * (1 - x / 10))
    turns = np.flatnonzero(np.diff(np.sign(np.diff(r)))) + 1
    expected = sorted(r[turns])
    res = _continue("dx/dt = r*x*(1 - x/10) - x^2/(1 + x^2) with r = 0.3, x(0) = 0.1",
                    param_name="r", param_min=0.3, param_max=0.7, plot_state="x")
    assert _points(res, "LP", "r") == pytest.approx(expected, rel=1e-3)


@pytest.mark.auto
def test_toggle_switch_pitchfork_at_a_equal_2():
    # symmetric state u = v with u (1 + u^2) = a; it loses stability where a = 2 (u = 1)
    res = _continue("du/dt = a/(1 + v^2) - u, dv/dt = a/(1 + u^2) - v with a = 1, u(0) = 1, "
                    "v(0) = 0.5", param_name="a", param_min=0.5, param_max=5, plot_state="u")
    assert _points(res, "BP", "a") == pytest.approx([2.0], rel=1e-3)


@pytest.mark.auto
def test_sir_endemic_threshold():
    # R0 = beta / (gamma + mu) = 1 at beta = 0.52
    res = _continue("dS/dt = mu - beta*S*I - mu*S, dI/dt = beta*S*I - (gamma + mu)*I, mu = 0.02, "
                    "gamma = 0.5, beta = 0.3, S(0) = 0.99, I(0) = 0.01",
                    param_name="beta", param_min=0.1, param_max=2, plot_state="I")
    assert _points(res, "BP", "beta") == pytest.approx([0.52], rel=1e-3)


@pytest.mark.auto
def test_rosenzweig_macarthur_transcritical_and_hopf():
    # N* = 0.2 / (0.5 - 0.2) = 2/3: predator invades at K = N*, Hopf at K = 1 + 2 N*
    res = _continue("dN/dt = N*(1 - N/K) - N*P/(1 + N), dP/dt = 0.5*N*P/(1 + N) - 0.2*P, K = 2, "
                    "N(0) = 1, P(0) = 0.5", param_name="K", param_min=0.5, param_max=5,
                    plot_state="P")
    assert _points(res, "BP", "K") == pytest.approx([2 / 3], rel=1e-3)
    assert _points(res, "HB", "K") == pytest.approx([7 / 3], rel=1e-3)
    # at the Hopf point trace J = 0 and omega^2 = det J = 0.4 * 0.5 * P*/(1 + N*)^2 with
    # P* = 25/21: the cycles are born with period 2 pi / omega, not the branch's longest period
    omega = np.sqrt(0.4 * 0.5 * (25 / 21) / (5 / 3) ** 2)
    hb = [p for p in res["bifurcation_points"] if p["type"] == "HB"]
    assert hb[0]["period_at_onset"] == pytest.approx(2 * np.pi / omega, rel=1e-3)


@pytest.mark.auto
def test_allee_harvesting_fold_from_a_start_inside_the_window():
    # (N - 0.2)(1 - N) = h has a fold at N = 0.6, h = 0.16; at the model's own h = 0.3 only
    # N = 0 exists, so the continuation starts at h = 0 on the upper state N = 1
    res = _continue("dN/dt = N*(N - 0.2)*(1 - N) - h*N with h = 0.3, N(0) = 0.5",
                    param_name="h", param_min=0, param_max=0.4, plot_state="N",
                    fixed_params={"h": 0}, initial_conditions={"N": 1})
    assert res["starting_equilibrium"]["N"] == pytest.approx(1.0, abs=1e-6)
    assert _points(res, "LP", "h") == pytest.approx([0.16], rel=1e-3)


def test_steady_states_across_a_range_show_where_several_coexist():
    from dynalyst.analysis.steady_states import find_steady_states
    from dynalyst.orchestration.reviewer import Reviewer
    args = dict(species=["x"], odes={"x": "mu + x**2/(1 + x**2) - 0.4*x"}, parameters={"mu": 0.0},
                initial_conditions={"x": 1.0}, param_name="mu", param_min=0.0, param_max=1.0, n_values=2)
    res = json.loads(find_steady_states(**args))
    at_zero, at_one = res["states_across"]
    assert {round(s["state"]["x"], 6) for s in at_zero["steady_states"]} == {0.0, 0.5, 2.0}
    assert len(at_one["steady_states"]) == 1 and res["most_states"] == 3 and res["at_values"] == [0.0]
    checks = Reviewer()._review_find_steady_states(args, res, json.dumps(res))
    assert checks and all(c.passed for c in checks)
    assert "error" in json.loads(find_steady_states(**{**args, "param_max": -1.0}))
    assert "error" in json.loads(find_steady_states(**{**args, "param_name": "not_a_parameter"}))


def test_predator_prey_steady_states_and_stability():
    from dynalyst.analysis.steady_states import find_steady_states
    res = json.loads(find_steady_states(_typed(
        "dx/dt = x*(1 - x) - x*y/(1 + x), dy/dt = 0.3*x*y/(1 + x) - 0.1*y with x(0) = 0.5, "
        "y(0) = 0.2")))
    found = {(round(s["state"]["x"], 6), round(s["state"]["y"], 6)): s["stability"]
             for s in res["steady_states"]
             if s["state"]["x"] >= -1e-9 and s["state"]["y"] >= -1e-9}
    assert set(found) == {(0.0, 0.0), (1.0, 0.0), (0.5, 0.75)}
    assert found[(0.5, 0.75)].startswith("stable")
    assert not found[(0.0, 0.0)].startswith("stable")
    assert not found[(1.0, 0.0)].startswith("stable")


def test_gene_expression_steady_state():
    from dynalyst.analysis.steady_states import find_steady_states
    res = json.loads(find_steady_states(_typed(
        "dm/dt = 5 - m, dp/dt = 2*m - 0.5*p with m(0) = 0, p(0) = 0")))
    (s,) = res["steady_states"]
    assert (s["state"]["m"], s["state"]["p"]) == pytest.approx((5.0, 20.0))
    assert s["stability"].startswith("stable")


# --- the same system written in different ways is read the same way ---------------------
_FORMS = [
    "dx/dt = a - x^2 + y, dy/dt = -b*y, a = 1, b = 2, x(0) = 0.5, y(0) = 0.1",
    "dx/dt = a - x**2 + y; dy/dt = -b*y; a = 1; b = 2; x(0) = 0.5; y(0) = 0.1.",
    "dx/dt = a - x^2 + y\ndy/dt = -b*y\nwith a = 1 and b = 2, x(0) = 0.5 and y(0) = 0.1.",
    "d x / d t = a - x^2 + y, d y / d t = -b*y. Use a = 1, b = 2. Start at x0 = 0.5, y0 = 0.1.",
    "x' = a - x^2 + y, y' = -b*y with a = 1, b = 2, x(0) = 0.5, y(0) = 0.1",
]


@pytest.mark.parametrize("text", _FORMS)
def test_one_system_in_many_spellings(text):
    import sympy as sp
    s = parse_typed_system(text)
    assert s is not None, text
    assert s["parameters"] == {"a": 1.0, "b": 2.0}
    assert s["initial_conditions"] == {"x": 0.5, "y": 0.1}
    x, y, a, b = sp.symbols("x y a b")
    assert sp.simplify(sp.sympify(s["odes"]["x"]) - (a - x ** 2 + y)) == 0
    assert sp.simplify(sp.sympify(s["odes"]["y"]) - (-b * y)) == 0


@pytest.mark.auto
def test_an_isola_off_the_start_branch_is_found_by_the_log_spaced_search():
    # x = 0 for every p (the start branch) and, apart from it, the circle
    # (x - 2)^2 + ((p - 0.5)/0.2)^2 = 1: an isola over p in [0.3, 0.7] with folds at its ends.
    # Evenly spaced probes over [0, 10] (2.5, 5, ...) miss it; log-spaced ones (0.316) do not.
    res = _continue("dx/dt = -x*((x - 2)^2 + ((p - 0.5)/0.2)^2 - 1) with p = 5, x(0) = 0",
                    param_name="p", param_min=0, param_max=10, plot_state="x", cycles=False)
    assert _points(res, "LP", "p") == pytest.approx([0.3, 0.7], rel=1e-3)


def test_states_across_a_pole_of_the_rate_laws_are_not_continued_from():
    import sympy as sp
    from dynalyst.analysis.bifurcation_auto import _probe_values
    from dynalyst.analysis.steady_states import across_a_pole as _across_a_pole
    x, T, J = sp.symbols("x T J")
    # Goldbeter-Koshland-type term: x may not pass its total T (the denominator J + T - x)
    model = {"state_symbols": [x], "state_names": ["x"], "param_symbols": [T, J],
             "param_names": ["T", "J"], "y0": {"x": 0.1}, "rhs": [(T - x) / (J + T - x) - x]}
    pv = {"T": 1.0, "J": 0.01}
    assert not _across_a_pole(model, [0.5], pv) and not _across_a_pole(model, [-0.2], pv)
    assert _across_a_pole(model, [1.8], pv)
    assert _probe_values(0, 6, 0, True) == pytest.approx([0, 0.006, 0.03374, 0.1897, 1.067, 6],
                                                         rel=1e-3)
    assert _probe_values(0, 6, 0, False) == [1.5, 3.0, 4.5, 6.0]      # the LHS scans by default


def test_lhs_probes_evenly_unless_log_probes_is_asked_for(monkeypatch):
    from dynalyst.analysis import bifurcation_auto
    from dynalyst.regime import batch_lhs_runner
    seen = []

    def fake(**kw):
        seen.append(kw.get("log_probes"))
        return json.dumps({"error": "stop here"})
    monkeypatch.setattr(bifurcation_auto, "_run_single_bifurcation_analysis", fake)
    base = {"source_path": "m.xml", "param_values": {"s": 1.0}}
    for flag in (False, True):
        with pytest.raises(RuntimeError):
            batch_lhs_runner.compute_bifurcation({}, "s", (0.01, 50), "y", base, flag)
    assert seen == [False, True]
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan
    from dynalyst.regime.categorize_tool import categorized_lhs_scan
    for t in (lhs_bifurcation_scan, categorized_lhs_scan):
        assert "log_probes" in t.args and "log scaling" in t.description
