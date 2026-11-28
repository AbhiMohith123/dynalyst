import json
from pathlib import Path
import numpy as np
import pytest


def test_f04_design_and_artifacts_preserve_precision(stubbed_auto):
    from dynalyst.regime.batch_lhs_runner import run_and_report
    report, results, directory = run_and_report(5, model='Markevich2004', bif_param='MAPKK',
        y_species='Mpp', vary_only=['k1cat'], seed=17, param_range=(0,100))
    path = Path(directory)
    design = json.loads((path/'design.json').read_text())
    saved = json.loads((path/'results.json').read_text())
    assert len(design['samples']) == len(results) == 5
    assert design['samples'] == [r['params'] for r in results]
    assert [r['params'] for r in saved['samples']] == design['samples']
    assert all(Path(p).is_file() for p in saved['artifacts'].values())
    assert saved['bif_param'] == 'MAPKK' and saved['y_species'] == 'Mpp'
    assert '.17g' not in report


def test_f04_failed_batch_keeps_design_and_every_failure(tmp_path, monkeypatch):
    from dynalyst.regime import batch_lhs_runner as runner
    def fail(*args, **kwargs):
        raise RuntimeError('test numerical failure')
    monkeypatch.setattr(runner, 'compute_bifurcation', fail)
    _, results, directory = runner.run_and_report(3, model='Markevich2004', bif_param='MAPKK',
        y_species='Mpp', vary_only=['k1cat'], seed=17, param_range=(1,100))
    saved = json.loads((Path(directory)/'results.json').read_text())
    assert saved['status'] == 'failed' and saved['n_failed'] == 3
    assert len(json.loads((Path(directory)/'design.json').read_text())['samples']) == 3
    assert all('test numerical failure' in r['error'] for r in results)
    assert len(list(Path(directory).glob('sample_*/result.json'))) == 3


def test_f04_auto_unavailable_control_stops_compilation_and_saves_batch(tmp_path, monkeypatch):
    from dynalyst import config
    from dynalyst.auto_env import AutoUnavailable
    from dynalyst.regime import batch_lhs_runner as runner
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise AutoUnavailable('test compiler unavailable')
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(runner, 'compute_bifurcation', fail)
    with pytest.raises(AutoUnavailable, match='failures were saved'):
        runner.run_and_report(3, model='Markevich2004', bif_param='MAPKK',
            y_species='Mpp', vary_only=['k1cat'], seed=17, param_range=(1,100), goal='bistability')
    assert len(calls) == 1
    path = next(tmp_path.glob('lhs/*/results.json'))
    saved = json.loads(path.read_text())
    assert saved['status'] == 'failed' and saved['n_failed'] == 3
    assert all(Path(p).exists() for p in saved['artifacts'].values())


def test_f07_full_state_grid_and_export(tmp_path, monkeypatch):
    from dynalyst import config
    from dynalyst.analysis.model_reduction import reduce_model
    # Export writes a new named fixture using the application's normal exporter.
    result = json.loads(reduce_model(species=['A','B'], odes={'A':'-k*A+B','B':'k*A-B'},
        parameters={'k':2}, initial_conditions={'A':1,'B':0}, method='conservation',
        model_name='seven_group_conservation', validation_t_end=5, validation_points=501))
    val = result['validation']
    assert val['accurate'] and val['n_time_points'] == 501
    assert set(val['comparison_species']) == {'A','B'} and val['reconstructed_species']
    assert result['sbml_export']['validation']['accurate']
    assert result['sbml_export']['validated_against_full']
    path = Path(val['full_grid_path'])
    if not path.is_absolute():
        path = config.PROJECT_ROOT/path
    curves = np.load(path)
    assert curves['full'].shape == curves['reconstructed'].shape == (2,501)
    assert np.max(np.abs(curves['full']-curves['reconstructed'])) < 1e-7


def test_f07_reconstruction_error_cannot_hide_behind_retained_states():
    import sympy as sp
    from dynalyst.analysis.ode_model import model_from_odes
    from dynalyst.analysis.reduction_validation import validate
    full = model_from_odes(['A','B'], {'A':'-A','B':'A'}, {}, {'A':1,'B':0})
    reduced = model_from_odes(['A'], {'A':'-A'}, {}, {'A':1})
    result = validate(full,reduced,{}, {'B':2-sp.Symbol('A')}, t_end=3)
    assert not result['accurate']
    assert result['per_species_max_error']['A'] < 1e-6
    assert result['per_species_max_error']['B'] > 1


def test_f07_reloaded_export_detects_altered_saved_parameter(tmp_path, monkeypatch):
    import libsbml
    from dynalyst import config
    from dynalyst.analysis.ode_model import model_from_odes
    from dynalyst.analysis.jaxkm_ode_to_sbml import odes_to_sbml
    from dynalyst.analysis.reduction_validation import validate_export
    monkeypatch.setattr(config, 'LOCAL_SBML_MODELS_DIR', tmp_path)
    result = json.loads(odes_to_sbml(['X'], {'X':'-k*X'}, {'k':1}, {'X':1},
                                   model_name='seven_export_modified'))
    path = result['path']
    doc = libsbml.readSBML(path)
    doc.getModel().getParameter('k').setValue(2)
    libsbml.writeSBMLToFile(doc, path)
    model = model_from_odes(['X'], {'X':'-k*X'}, {'k':1}, {'X':1})
    times = np.linspace(0, 3, 101)
    checked = validate_export(path, model, {'k':1}, times, np.exp(-times)[None,:])
    assert not checked['accurate'] and checked['equation_max_scaled_error'] > .1


def test_f07_huang_original_conservation_request():
    from dynalyst.analysis.model_reduction import reduce_model
    result = json.loads(reduce_model('BIOMD0000000009.xml', method='conservation',
        model_name='seven_group_huang_conservation', validation_t_end=100, validation_points=501))
    root = Path(__file__).resolve().parents[1]/'outputs/seven_workstreams_20261028'
    root.mkdir(parents=True, exist_ok=True)
    (root/'huang_reduction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert result['validation']['accurate'], result['validation']
    assert len(result['validation']['comparison_species']) == result['original_dimension']
    assert result['sbml_export'].get('validated_against_full'), result['sbml_export']
    root = Path(__file__).resolve().parents[1]/'outputs/seven_workstreams_20261028'
    root.mkdir(parents=True, exist_ok=True)
    (root/'huang_reduction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
