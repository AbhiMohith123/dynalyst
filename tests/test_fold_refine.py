"""A fold is accepted only where the fold equations (f = 0, zero eigenvalue) put it.

The stiff model below has the folds of dx/dt = p + x - x^3 (p = -/+ 2/(3*sqrt(3)) = -/+ 0.3849)
and a fast variable y with eigenvalue -1000. A point 1 % away from the fold has a smallest
eigenvalue about 2e-4 of the largest, which a ratio test of the eigenvalues accepts as a fold.
"""
import json
import math

import numpy as np
import pytest

from dynalyst.analysis.fold_refine import refine_fold

FOLD = 2 / (3 * math.sqrt(3))
STIFF = dict(species=["x", "y"], odes={"x": "p + x - x**3", "y": "-1000*(y - x)"},
             parameters={"p": 0.0}, initial_conditions={"x": -1.0, "y": -1.0})


def _f(y, p):
    return np.array([p + y[0] - y[0] ** 3, -1000 * (y[1] - y[0])])


def _j(y, p):
    return np.array([[1 - 3 * y[0] ** 2, 0.0], [1000.0, -1000.0]])


def _near_fold(shift):
    x = 1 / math.sqrt(3) + shift
    return [x, x], -(x - x ** 3)


def test_a_reported_fold_on_the_fold_is_accepted():
    y, p = _near_fold(0.0)
    r = refine_fold(_f, _j, y, p)
    assert r["converged"] and r["accepted"] and r["p"] == pytest.approx(-FOLD, abs=1e-9)


def test_a_point_a_step_past_the_fold_is_moved_to_the_fold():
    y, p = _near_fold(0.05)
    ev = np.abs(np.linalg.eigvals(_j(y, p)))
    assert ev.min() / (1 + ev.max()) < 1e-2          # the old ratio test would have accepted it
    r = refine_fold(_f, _j, y, p)
    assert r["converged"] and not r["accepted"]
    assert r["p"] == pytest.approx(-FOLD, abs=1e-9) and r["rel_shift"] > 1e-3
    assert "moved" in r["detail"]


def test_the_reviewer_rejects_a_fold_a_step_away_and_accepts_the_true_one():
    from dynalyst.orchestration.reviewer import Reviewer
    rv = Reviewer()
    model = rv._model(None, STIFF["species"], STIFF["odes"], STIFF["parameters"],
                      STIFF["initial_conditions"])
    y, p = _near_fold(0.05)
    bad = rv._verify_point(model, "p", p, {"x": y[0], "y": y[1]}, "LP")
    assert not bad.passed and "moved" in bad.detail
    y, p = _near_fold(0.0)
    good = rv._verify_point(model, "p", p, {"x": y[0], "y": y[1]}, "LP")
    assert good.passed and "confirmed" in good.detail


@pytest.mark.auto
def test_the_diagram_reports_the_folds_of_the_stiff_model_exactly():
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    d = json.loads(run_bifurcation_analysis(**STIFF, param_name="p", param_min=-1.0, param_max=1.0,
                                            plot_state="x", _cycles=False))
    assert not d.get("error"), d
    lps = sorted(pt["p"] for pt in d["bifurcation_points"] if pt["type"] == "LP")
    assert lps == pytest.approx([-FOLD, FOLD], abs=1e-6)
    assert all("fold_check" in pt for pt in d["bifurcation_points"] if pt["type"] == "LP")


def test_a_slow_complex_pair_is_not_taken_for_the_fold():
    """A weakly damped complex pair nearer to zero than the fold's real eigenvalue must not be
    tested in its place (it used to give 'confirmed' at the wrong point)."""
    import numpy as np
    from dynalyst.analysis.fold_refine import refine_fold
    for a, b in ((-1.0, 5.0), (-1e-3, 1e-2), (-1e-9, 1e-2)):
        f = lambda y, p, a=a, b=b: np.array([p + y[0] - y[0] ** 3, a * y[1] - b * y[2], b * y[1] + a * y[2]])
        j = lambda y, p, a=a, b=b: np.array([[1 - 3 * y[0] ** 2, 0, 0], [0, a, -b], [0, b, a]])
        x0 = 0.57735 * 1.002
        res = refine_fold(f, j, [x0, 0.0, 0.0], -0.383314)
        assert res["converged"] and not res["accepted"]
        assert abs(res["p"] + 2 / (3 * np.sqrt(3))) < 1e-8
