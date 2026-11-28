"""
test_auto_pipeline.py

Integration-level tests that actually run AUTO-07p continuation -- unlike
test_pipeline.py, these need a built AUTO-07p (scripts/build_auto.sh, or
AUTO_DIR pointing at an existing install). They are skipped otherwise.

Each test here costs at least one real AUTO continuation run, so this file
is intentionally separate from test_pipeline.py and slower (seconds to
tens of seconds per test rather than milliseconds). Run it on its own:

    pytest tests/test_auto_pipeline.py -v -s

The model is models/pdp_mm_bistable.txt read as a plain XPP file, with its
three parameter sets (tests/xpp_sets.py).
"""
import json

import pytest

from dynalyst.regime.batch_lhs_runner import compute_bifurcation, classify_signature
from tests import xpp_sets

pytestmark = pytest.mark.auto


def _compute(tmp_path, values, name, params=None):
    path = xpp_sets.write_xpp(tmp_path / f"{name}.ode", values)
    return compute_bifurcation(dict(params or values), 'ATOT', (0.1, 200), 'APP', xpp_sets.parsed(path))


# ---------------------------------------------------------------------------
# compute_bifurcation -- one real AUTO run per parameter set (not sampled),
# so these are the "does the pipeline even work" checks before trusting any
# LHS batch result.
# ---------------------------------------------------------------------------

def test_compute_bifurcation_pitchfork_runs_and_returns_branches(tmp_path):
    all_branches, special = _compute(tmp_path, xpp_sets.PITCHFORK, 'pf_e2e')
    assert len(all_branches) >= 1


def test_compute_bifurcation_markevich_finds_bistability(tmp_path):
    """The Markevich set is the model's original bistability example
    (Markevich et al.) -- its own paper's parameters are expected to show
    the classic S-shaped 2-LP fold structure. If this fails, something
    upstream (steady-state guess, codegen, branch switching) regressed for
    a case we independently know should be bistable."""
    all_branches, special = _compute(tmp_path, xpp_sets.MARKEVICH, 'markevich_e2e')
    sig = classify_signature(special)
    assert 'LP' in sig, (
        f"Expected the Markevich set's own parameters to show a fold "
        f"(LP) structure, got signature={sig!r}. This is the known-good "
        f"case, so if this fails the bug is upstream of any LHS sampling.")


def test_compute_bifurcation_transcritical_runs(tmp_path):
    all_branches, special = _compute(tmp_path, xpp_sets.TRANSCRITICAL, 'tc_e2e')
    assert len(all_branches) >= 1


# ---------------------------------------------------------------------------
# one diagram of an XPP file (not through the LLM agent: no API key needed)
# ---------------------------------------------------------------------------

def test_bifurcation_of_the_xpp_file(tmp_path, monkeypatch):
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    monkeypatch.chdir(tmp_path)
    path = xpp_sets.write_xpp(tmp_path / 'pf.ode', xpp_sets.PITCHFORK)
    result = json.loads(run_bifurcation_analysis(path, param_name='ATOT', plot_state='APP', _cycles=False))
    assert 'BP' in [p['type'] for p in result['bifurcation_points']]


# ---------------------------------------------------------------------------
# LHS around the pitchfork set
# ---------------------------------------------------------------------------

def test_pitchfork_lhs_batch_converges(tmp_path):
    """An LHS batch around the pitchfork set converges. Its samples show no branch point,
    and rightly so: see test_pitchfork_exists_only_while_the_set_stays_symmetric."""
    from dynalyst.regime.batch_lhs_runner import run_lhs_batch_for_model
    path = xpp_sets.write_xpp(tmp_path / 'pf.ode', xpp_sets.PITCHFORK)
    out = run_lhs_batch_for_model(path, n_samples=15, bif_param='ATOT', y_species='APP',
                                  param_range=(0.1, 200), spread_factor=1.2, seed=0)
    results = out[0] if isinstance(out, tuple) else out
    successes = [r for r in results if r['error'] is None]
    assert len(successes) > 0, (
        f"All {len(results)} pitchfork samples failed to converge at all "
        f"(spread_factor=1.2) -- see the r['error'] entries.")
    # The pitchfork set is symmetric (km1 = km5, km3 = km7, k2 = k6, k4 = k8), and a
    # pitchfork exists only under that symmetry. LHS varies each parameter on its own, which
    # breaks it: the pitchfork unfolds (imperfect bifurcation), the branch followed has no
    # branch point, and 'none' is the right answer for those samples.


def test_pitchfork_exists_only_while_the_set_stays_symmetric():
    import json
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    base = {k: v for k, v in xpp_sets.PITCHFORK.items() if k != "ATOT"}

    def points(fixed):
        d = json.loads(run_bifurcation_analysis("pitchfork_model.xml", param_name="ATOT",
                                                param_min=0.05, param_max=199,
                                                fixed_params=fixed, _cycles=False))
        assert not d.get("error"), d.get("error")
        return [p["type"] for p in d["bifurcation_points"]]

    assert points(base) == ["BP"]                                   # symmetric: pitchfork
    assert "BP" not in points({**base, "k2": 0.0105})               # symmetry broken: unfolds
    assert points({**base, "k2": 0.0105, "k6": 0.0105}) == ["BP"]   # kept: pitchfork again
