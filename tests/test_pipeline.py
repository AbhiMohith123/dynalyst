"""
test_pipeline.py

The parts of the LHS scans that need no compiled AUTO-07p: SBML parsing, LHS sampling,
signature classification and the branch record. tests/test_auto_pipeline.py runs AUTO.

Run with:  pytest tests/test_pipeline.py -v
"""
import os
import math
import numpy as np
import pytest
import sympy as sp

from dynalyst.regime import parser as sbml_parser
from dynalyst.regime import lhs_sampling
from dynalyst.regime.batch_lhs_runner import _Branch, classify_signature
from dynalyst import config

HERE = os.path.dirname(os.path.abspath(__file__))
PITCHFORK_XML = str(config.MODELS_DIR / 'pitchfork_model.xml')


# ---------------------------------------------------------------------------
# parser.py
# ---------------------------------------------------------------------------

def test_parse_sbml_pitchfork_species_and_params():
    parsed = sbml_parser.parse_sbml(PITCHFORK_XML)
    assert set(parsed['species']) == {'A', 'APP'}
    # r1..r4 are AssignmentRule-defined fluxes, not free parameters, so they
    # must NOT leak into the free-parameter list
    assert 'r1' not in parsed['params']
    for expected in ['ATOT', 'KTOT', 'PTOT', 'km1', 'km3', 'km5', 'km7',
                     'k2', 'k4', 'k6', 'k8']:
        assert expected in parsed['params'], f"missing param {expected}"


def test_parse_sbml_ode_rhs_matches_known_equations():
    parsed = sbml_parser.parse_sbml(PITCHFORK_XML)
    A, APP = sp.symbols('A APP')
    subs = {sp.Symbol(k): v for k, v in parsed['param_values'].items()}
    # dA/dt should equal r3 - r1 at a specific numeric point -- check
    # against the hand-written formulas (the model's rate laws) rather
    # than re-deriving symbolically, so this test
    # catches drift between the SBML and the parser independently of both.
    point = {A: 0.03, APP: 0.02}
    k2, k4, k6, k8 = (parsed['param_values'][p] for p in ['k2', 'k4', 'k6', 'k8'])
    km1, km3, km5, km7 = (parsed['param_values'][p] for p in ['km1', 'km3', 'km5', 'km7'])
    KTOT, PTOT, ATOT = (parsed['param_values'][p] for p in ['KTOT', 'PTOT', 'ATOT'])
    a, app = point[A], point[APP]
    r1 = (k2 * KTOT * a / km1) / (1 + a / km1 + (ATOT - a - app) / km3)
    r3 = (k8 * PTOT * (ATOT - a - app) / km7) / (1 + (ATOT - a - app) / km7 + app / km5)
    expected_dA = r3 - r1
    got_dA = float(parsed['ode_rhs']['A'].subs(subs).subs(point))
    assert math.isclose(got_dA, expected_dA, rel_tol=1e-9)


def test_parse_sbml_missing_file_raises():
    with pytest.raises(Exception):
        sbml_parser.parse_sbml(os.path.join(HERE, 'does_not_exist.xml'))


# ---------------------------------------------------------------------------
# lhs_sampling.py
# ---------------------------------------------------------------------------

def test_model_bounds_spread_factor_excludes_bif_param():
    parsed = sbml_parser.parse_sbml(PITCHFORK_XML)
    bounds = lhs_sampling.resolve_model_bounds(parsed, spread_factor=2.0, exclude_params=['ATOT']).bounds
    assert 'ATOT' not in bounds
    lo, hi, _ = bounds['KTOT']
    assert lo == pytest.approx(parsed['param_values']['KTOT'] / 2.0)
    assert hi == pytest.approx(parsed['param_values']['KTOT'] * 2.0)


def test_lhs_sample_params_respects_bounds_and_latin_property():
    bounds = {'a': (1.0, 2.0, False), 'b': (10.0, 1000.0, True)}
    samples = lhs_sampling.lhs_sample_params(bounds, n_samples=50, seed=0)
    assert len(samples) == 50
    for s in samples:
        assert 1.0 <= s['a'] <= 2.0
        assert 10.0 <= s['b'] <= 1000.0
    # Latin hypercube: with 50 samples split into 50 strata on [1,2], every
    # 1/50-wide bin should contain roughly one sample -- check no more than
    # a small cluster collapses into one bin (a crude but real check that
    # this isn't just np.random.uniform under the hood)
    a_vals = sorted(s['a'] for s in samples)
    bin_counts = np.histogram(a_vals, bins=50, range=(1.0, 2.0))[0]
    assert bin_counts.max() <= 3


def test_lhs_sample_params_fixed_param_stays_fixed():
    bounds = {'ATOT': (0.1, 0.1, False)}
    samples = lhs_sampling.lhs_sample_params(bounds, n_samples=10, seed=1)
    assert all(s['ATOT'] == 0.1 for s in samples)


# ---------------------------------------------------------------------------
# batch_lhs_runner.classify_signature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("special,expected", [
    ([], 'none'),
    ([(1, 'LP', 1), (2, 'LP', 2)], '2 LP'),
    ([(1, 'LP', 1)], '1 LP'),
    ([(1, 'BP', 1)], '1 BP'),
    ([(1, 'HB', 1)], '1 HB'),
    ([(1, 'BP', 1), (2, 'HB', 2)], '1 BP + 1 HB'),
])
def test_classify_signature(special, expected):
    assert classify_signature(special) == expected


def test_branch_stability_is_in_autos_format():
    assert _Branch({}, [True, True, False, True]).stability() == [-2, 3, -4]
    assert _Branch({}, [False]).stability() == [1]
