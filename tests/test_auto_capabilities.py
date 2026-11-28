"""AUTO-07p capabilities beyond one-parameter diagrams, checked against answers known by hand:
classic problems for run_auto_problem."""
import json

import numpy as np
import pytest

from dynalyst import config
from dynalyst.analysis.auto_problems import run_auto_problem
from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis

pytestmark = pytest.mark.auto


def _problem(**kw):
    d = json.loads(run_auto_problem(**kw))
    assert not d.get("error"), d
    return d


def test_map_period_doubling_of_the_logistic_map():
    """x(k+1) = r*x*(1-x): the fixed point 1-1/r period-doubles at r = 3."""
    d = _problem(kind="map", states=["x"], equations={"x": "r*x*(1-x)"}, free_parameters=["r"],
                 parameters={"r": 2.0}, initial_state={"x": 0.5}, param_min=1.5, param_max=3.4)
    pd = [p for p in d["special_points"] if p["type"] == "PD"]
    assert pd and pd[0]["r"] == pytest.approx(3.0, abs=1e-3) and pd[0]["x"] == pytest.approx(2 / 3, abs=1e-3)


def test_boundary_value_fold_of_bratu():
    """u'' + lam*exp(u) = 0, u(0) = u(1) = 0: a fold at lam = 3.5138 (Bratu's problem)."""
    d = _problem(kind="boundary_value", states=["u", "v"], equations={"u": "v", "v": "-lam*exp(u)"},
                 free_parameters=["lam"], parameters={"lam": 0.0}, initial_state={"u": 0.0, "v": 0.0},
                 boundary_conditions=["u_0", "u_1"], param_min=0.0, param_max=4.0)
    lp = [p for p in d["special_points"] if p["type"] == "LP"]
    assert lp and lp[0]["lam"] == pytest.approx(3.5138, abs=1e-2)


def test_optimization_on_a_sphere():
    """Extremum of a linear objective on a circle x^2 + a^2 = 1: objective a has extremes at +-1."""
    d = _problem(kind="optimization", states=["x"], equations={"x": "x**2 + a**2 - 1"},
                 free_parameters=["a"], parameters={"a": 0.0}, initial_state={"x": 1.0}, objective="a")
    lp = [p for p in d["special_points"] if p["type"] == "LP"]
    assert lp and abs(abs(lp[0]["a"]) - 1.0) < 1e-3


def test_rotation_from_data(tmp_path):
    """AUTO's demo pen: coupled pendula, a rotation read from numerical data."""
    period = 1.5738797205
    rows = np.loadtxt(config.PROJECT_ROOT / "vendor/auto-07p/demos/pen/pen.dat")
    rows[:, 0] *= period
    start = tmp_path / "pen.txt"
    np.savetxt(start, rows)
    d = _problem(kind="orbit", states=["p1", "p2", "s1", "s2"],
                 equations={"p1": "s1", "p2": "s2",
                            "s1": "-e*s1 - sin(p1) + ri + g*(p2 - p1)",
                            "s2": "-e*s2 - sin(p2) + ri + g*(p1 - p2)"},
                 free_parameters=["ri"], parameters={"ri": 0.4, "g": 0.175, "e": 0.1},
                 start_orbit_file=str(start), nmx=100)
    assert d["branches"] and d["branches"][0]["n_points"] > 5


def test_waves_start_at_a_hopf_point_of_the_uniform_states():
    """AUTO's demo wav: wave trains from the Hopf points of the spatially uniform states."""
    d = _problem(kind="wave", states=["a", "b"],
                 equations={"a": "-3*(210*(b/(3.4 + b))*(a/(1 + a + 0.023*a**2)) - (145 - a))",
                            "b": "-3*(210*(b/(3.4 + b))*(a/(1 + a + 0.023*a**2)) - 0.2*(p - b))"},
                 free_parameters=["p"], parameters={"p": 500.0},
                 initial_state={"a": 62.7662, "b": 88.8308},
                 diffusion={"a": 1.0, "b": 5.0}, wave_speed=0.05, param_min=400, param_max=800, nmx=100)
    assert any(p["type"] == "HB" for p in d["special_points"]), d


def test_a_problem_without_its_inputs_is_refused():
    d = json.loads(run_auto_problem("boundary_value", ["u"], {"u": "u"}, ["a"], {"a": 1.0}))
    assert "boundary_conditions" in d["error"]
    d = json.loads(run_auto_problem("wave", ["u"], {"u": "u"}, ["a"], {"a": 1.0}))
    assert "wave_speed" in d["error"]
    d = json.loads(run_auto_problem("nonsense", ["u"], {"u": "u"}, ["a"], {"a": 1.0}))
    assert "kind" in d["error"]


def test_a_fold_past_the_range_does_not_hide_the_one_inside():
    """Markevich2004 at MKP3 = 137: folds at MAPKK 53.77 and 78.61. Over 25 to 75 the branch from the
    start runs past 75 and turns there, so the fold inside the range lies beyond the range."""
    d = json.loads(run_bifurcation_analysis("Markevich2004", param_name="MAPKK", param_min=25, param_max=75,
                                            fixed_params={"MKP3": 137.0}, plot_state="Mp", show_auto_log=False, _cycles=False))
    assert [round(p["MAPKK"], 1) for p in d["bifurcation_points"] if p["type"] == "LP"] == [53.8]
    assert d["classification"]["category"] == "bistable"
