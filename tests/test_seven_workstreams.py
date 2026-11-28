import json
import numpy as np
import pytest


def test_f02_damping_is_not_sustained_oscillation():
    import pandas as pd
    from dynalyst.analysis.jaxkm_tools import time_course_behaviour
    t = np.linspace(0, 60, 3001)
    result = time_course_behaviour(pd.DataFrame({'time':t, 'X':2+np.exp(-.02*t)*np.sin(t)}), ['X'])
    assert result['oscillating_species']['X']['amplitude_trend'] == 'decreasing'
    assert result['limit_cycle_established'] is False
    assert not result['note'].startswith('Sustained')


def test_f02_folds_require_stability_witness():
    from dynalyst.regime.taxonomy import classify_diagram
    assert 'bistable' not in classify_diagram(['LP','LP'])['features']
    curves = [(np.array([0,1]), np.array([0,0]), np.ones(2)),
              (np.array([0,1]), np.array([2,2]), np.ones(2))]
    assert 'bistable' in classify_diagram(['LP','LP'], curves)['features']


def test_f03_total_changes_reduced_dynamics_and_reconstruction():
    from dynalyst.analysis.ode_model import model_from_odes, eligible_parameters
    from dynalyst.analysis.conservation_parameters import expose_totals
    from dynalyst.analysis.model_prep import _admissible
    from dynalyst.analysis.steady_states import conservation_reduce, reconstruct_eliminated, _make_numeric_funcs
    m = expose_totals(model_from_odes(['A','B'], {'A':'-k*A+B','B':'k*A-B'}, {'k':2}, {'A':1,'B':2}))
    assert 'total_A' in eligible_parameters(m)
    red = conservation_reduce(m)
    f, _ = _make_numeric_funcs(red, {**red['param_defaults'], 'total_A':6})
    assert f([2])[0] == pytest.approx(0)
    assert reconstruct_eliminated(red, {'A':2}, {'total_A':6})['B'] == pytest.approx(4)
    adjusted = _admissible(m, {**m['param_defaults'], 'total_A':6}, {'A':1}, {'A','B'})
    assert adjusted['y0']['B'] == pytest.approx(5)
    with pytest.raises(ValueError, match='compatible'):
        _admissible(m, {**m['param_defaults'], 'total_A':.5}, {'A':1}, {'A','B'})


def test_f03_substrate_total_of_dual_phosphorylation_is_exposed():
    """The null-space basis gives A - K + AP - P + APP; the pool is the substrate total (EMBED,
    Harika & Sriram 2026), and the reduced model must use its parameter."""
    from dynalyst.analysis.ode_model import model_from_odes
    from dynalyst.analysis.conservation_parameters import expose_totals
    from dynalyst.analysis.steady_states import conservation_reduce, _make_numeric_funcs
    v = {'v1': 'k1*A*K - km1*AK', 'v2': 'k2*AK', 'v3': 'k3*AP*K - km3*APK', 'v4': 'k4*APK',
         'v5': 'k5*APP*P - km5*APPC', 'v6': 'k6*APPC', 'v7': 'k7*AP*P - km7*APC', 'v8': 'k8*APC'}
    odes = {'A': '-(v1) + (v8)', 'K': '-(v1) + (v2) - (v3) + (v4)', 'AK': '(v1) - (v2)',
            'APC': '(v7) - (v8)', 'AP': '(v2) - (v3) + (v6) - (v7)', 'APK': '(v3) - (v4)',
            'P': '-(v5) + (v6) - (v7) + (v8)', 'APPC': '(v5) - (v6)', 'APP': '(v4) - (v5)'}
    for k, e in v.items():
        odes = {s: r.replace(k, e) for s, r in odes.items()}
    pars = {'k1': .02, 'km1': 1, 'k2': .01, 'k3': .0024, 'km3': 1, 'k4': .21,
            'k5': .198, 'km5': 1, 'k6': .01, 'k7': .0024, 'km7': 1, 'k8': .2}
    y0 = {s: 0.0 for s in odes} | {'A': .1, 'K': 100.0, 'P': 100.0}
    m = expose_totals(model_from_odes(list(odes), odes, pars, y0))
    pools = {n: set(d['coefficients']) for n, d in m['conserved_totals'].items()}
    assert {'A', 'AK', 'AP', 'APK', 'APP', 'APPC', 'APC'} in pools.values()
    assert {'K', 'AK', 'APK'} in pools.values() and {'P', 'APC', 'APPC'} in pools.values()
    total = next(n for n, s in pools.items() if 'A' in s)
    assert m['param_defaults'][total] == pytest.approx(.1)
    red = conservation_reduce(m)
    assert len(red['state_names']) == 6
    assert any(sp_sym.name == total for _, e in red['eliminated'] for sp_sym in e.free_symbols)


def test_f03_pools_hidden_by_subtraction_and_float_rounding_are_exposed():
    """Huang-Ferrell's KKK/KK/KKPase totals need a law subtracted from a mixed one; Bachmann's
    STAT5 total (STAT5 + 11/16 npSTAT5 + pSTAT5) is found numerically in float-rounded rates."""
    from dynalyst.analysis.ode_model import resolve_model
    from dynalyst.analysis.conservation_parameters import expose_totals
    huang = expose_totals(resolve_model('BIOMD0000000009', None, None, None, None))['conserved_totals']
    assert {'total_KKK', 'total_KK', 'total_K', 'total_E1', 'total_E2'} <= set(huang) and len(huang) == 7
    assert all(float(c) > 0 for d in huang.values() for c in d['coefficients'].values())
    stat5 = expose_totals(resolve_model('Bachmann_MSB2011', None, None, None, None))['conserved_totals']['total_STAT5']
    assert stat5['coefficients'] == {'STAT5': '16', 'npSTAT5': '11', 'pSTAT5': '16'}


def test_f03_nominal_parameter_coincidence_is_not_a_pool():
    from dynalyst.analysis.ode_model import model_from_odes
    from dynalyst.analysis.conservation_parameters import expose_totals
    m = model_from_odes(['A','B'], {'A':'-k*A','B':'q*A'}, {'k':1,'q':1}, {'A':1,'B':0})
    assert not expose_totals(m)['conserved_totals']


def test_f03_changing_pool_preserves_unrelated_initial_conditions():
    from dynalyst.analysis.ode_model import model_from_odes
    from dynalyst.analysis.conservation_parameters import expose_totals
    from dynalyst.analysis.model_prep import _admissible
    m = expose_totals(model_from_odes(['A','B','X'],
        {'A':'-A+B','B':'A-B','X':'-X'}, {}, {'A':1,'B':1,'X':7}))
    adjusted = _admissible(m, {'total_A':4}, None, {'A','B','X'})
    assert adjusted['y0']['X'] == 7
    assert adjusted['y0']['A']+adjusted['y0']['B'] == pytest.approx(4)


def test_f03_markevich_observable_and_model35_total_are_available():
    from dynalyst.regime.model_source import prepare
    from dynalyst.orchestration.tools import inspect_model
    target = prepare('Markevich2004', 'MAPKK', 'Mpp')
    assert 'Mpp' in target['parsed']['species']
    m = json.loads(inspect_model('BIOMD0000000035.xml'))
    assert 'total_DA' in m['continuation_parameters']
    assert m['conserved_totals']['total_DA']['identity_verified']


def test_f06_failed_start_preserves_other_start_and_diagnostics(tmp_path):
    import jax.numpy as jnp
    import pandas as pd
    from types import SimpleNamespace
    from dynalyst.analysis.fitting_runtime import train_diagnosed
    def loss(params, ts, ys):
        return jnp.where(params['k'] < .2, jnp.nan, (params['k']-.4)**2)
    trainer = SimpleNamespace(dataset=[[1.]], ts=[0.], optim_space='linear', n_iter=2,
        loss_threshold=1e-8, loss_func=loss, optimizer=SimpleNamespace(init=lambda p: None),
        parameter_sets=pd.DataFrame([{'k':.1},{'k':.4}]))
    path = tmp_path/'steps.jsonl'
    best, runs = train_diagnosed(trainer, path)
    assert runs[0]['failed'] and 'Initial loss is not finite' in runs[0]['error']
    assert not runs[1]['failed'] and runs[1]['converged'] and best[1]['k'] == .4
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert len([event for event in events if event['event']=='finished']) == 2


@pytest.mark.parametrize('sentence', ['Bistability is not established.', 'The model is not bistable.',
    'Whether it is bistable remains unknown.', 'No evidence of bistability was found.'])
def test_f02_inconclusive_statement_is_not_positive_claim(sentence):
    from dynalyst.orchestration.final_check import stability_claim_without_check
    step = {'kind':'delegation', 'status':'ok', 'tool':'crnt_bistability', 'output':'{"bistable": null}'}
    assert not stability_claim_without_check(sentence, [step])



def test_f02_only_successful_nonzero_point_evidence_supports_claims():
    from dynalyst.orchestration.final_check import reported_point_types, qualify_empty_continuation
    step = {'kind':'delegation', 'status':'ok', 'tool':'run_bifurcation_analysis'}
    empty = json.dumps({'bifurcation_points':[], 'swept_parameter':'k', 'swept_range':[0,2],
                        'classification':{'evidence':{'counts':{'LP':0,'HB':0}}}})
    assert reported_point_types({**step, 'output':empty}) == set()
    assert reported_point_types({**step, 'output':'0 LP + 0 HB'}) == set()
    assert reported_point_types({**step, 'output':'Special point details: [{"type":"HB"}]'}) == {'HB'}
    assert reported_point_types({**step, 'status':'reject', 'output':'1 HB'}) == set()
    assert reported_point_types({**step, 'output':'{"error":"failed with HB"}'}) == set()
    answer = qualify_empty_continuation('No bifurcations were found.', [{**step,'output':empty}])
    assert 'k in [0, 2]' in answer and 'not a global absence result' in answer

