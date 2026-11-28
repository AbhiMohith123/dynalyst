"""Full-grid, full-state validation of a reduction and its exported equations."""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


def _run(model, params, times, rhs=None):
    from dynalyst.analysis.model_reduction import _numeric, _capped, VALIDATION_EVALUATIONS
    fn = rhs or _numeric(model, params)[0]
    initial = np.array([model['y0'][n] for n in model['state_names']], dtype=float)
    if not len(initial):
        return np.empty((0, len(times)))
    result = solve_ivp(_capped(fn, VALIDATION_EVALUATIONS), (times[0], times[-1]), initial,
                       t_eval=times, method='LSODA', rtol=1e-10, atol=1e-13)
    if not result.success or result.y.shape != (len(initial), len(times)) or not np.isfinite(result.y).all():
        raise ValueError('Integration did not return a finite value for every requested time point.')
    return result.y


def _errors(reference, candidate, names, tolerance, absolute_tolerance):
    absolute = np.abs(reference-candidate)
    scale = np.maximum(np.max(np.abs(reference), axis=1, keepdims=True), absolute_tolerance)
    relative = absolute/scale
    return {'max_relative_error': float(relative.max()),
            'max_absolute_error': float(absolute.max()),
            'final_relative_error': float(relative[:,-1].max()),
            'per_species_max_error': dict(zip(names, map(float, relative.max(axis=1)))),
            'accurate': bool(np.all(absolute <= absolute_tolerance + tolerance*scale))}


def validate(full, reduced, params, reconstruction, *, tolerance=.05, absolute_tolerance=1e-9,
             t_end=100, n_points=501, initial_layer=0, numeric_rhs=None, numeric_reconstruct=None):
    names = full['state_names']
    times = np.linspace(0, float(t_end), int(n_points))
    out = {'accurate':False, 't_end':float(t_end), 'n_time_points':int(n_points),
           'tolerance':float(tolerance), 'absolute_tolerance':float(absolute_tolerance),
           'comparison_species':names, 'reconstructed_species':[n for n in names if n not in reduced['state_names']],
           't_skip_initial_layer':float(initial_layer),
           'initial_condition_policy':'same retained initial values; all original states checked from time zero',
           'reconstruction':{str(n):str(e) for n,e in reconstruction.items()}}
    try:
        if len(times) < 2 or not np.isfinite(times).all() or times[-1] <= 0:
            raise ValueError('Validation requires a positive finite horizon and at least two points.')
        a = _run(full, params, times)
        retained = _run(reduced, params, times, numeric_rhs)
        symbols = dict(zip(reduced['state_names'], reduced['state_symbols']))
        exprs = {str(n):sp.sympify(e) for n,e in reconstruction.items()}
        # Compose chained QSSA and conservation substitutions before evaluating.
        for _ in range(len(exprs)+1):
            replacements = {sp.Symbol(n):e for n,e in exprs.items()}
            updated = {n:e.xreplace({k:v for k,v in replacements.items() if str(k) != n}) for n,e in exprs.items()}
            if updated == exprs:
                break
            exprs = updated
        psubs = {sp.Symbol(n):v for n,v in params.items()}
        b = []
        numeric = ([numeric_reconstruct(retained[:,i]) for i in range(len(times))]
                   if numeric_reconstruct else None)
        for name in names:
            if name in reduced['state_names']:
                values = retained[reduced['state_names'].index(name)]
            elif numeric is not None and name in numeric[0]:
                values = [row[name] for row in numeric]
            else:
                expr = exprs[name].subs(psubs)
                if numeric is not None:
                    values = [float(expr.subs({sp.Symbol(n):v for n,v in row.items()})) for row in numeric]
                else:
                    fn = sp.lambdify(list(symbols.values()), expr, 'numpy')
                    values = np.broadcast_to(np.asarray(fn(*retained), dtype=float), (len(times),))
            b.append(values)
        b = np.asarray(b, dtype=float)
        if not np.isfinite(b).all():
            raise ValueError('Reconstructed states contain non-finite values.')
        out.update(_errors(a,b,names,tolerance,absolute_tolerance))
        after = times >= initial_layer
        if after.any():
            post = _errors(a[:,after], b[:,after], names,tolerance,absolute_tolerance)
            out['post_initial_layer'] = {**post, 'first_checked_time':float(times[after][0]),
                                         'n_time_points':int(after.sum())}
        out['_curves'] = (times,a,b)
        out['_retained'] = retained
    except Exception as exc:
        out['error'] = f'{type(exc).__name__}: {exc}'
    return out


def validate_export(path, reduced, params, times, reference, tolerance=1e-7):
    from dynalyst.analysis.ode_model import model_from_sbml
    from dynalyst.analysis.model_reduction import _numeric
    try:
        exported = model_from_sbml(path)
        if set(exported['state_names']) != set(reduced['state_names']):
            raise ValueError('Exported state names differ from the reduced model.')
        # Verify saved parameters/initial values, not values silently supplied by caller.
        actual = _run(exported, exported['param_defaults'], times)
        order = [exported['state_names'].index(n) for n in reduced['state_names']]
        actual = actual[order]
        result = _errors(reference,actual,reduced['state_names'],tolerance,1e-10)
        source_rhs = _numeric(reduced,params)[0]
        exported_rhs = _numeric(exported,exported['param_defaults'])[0]
        residuals = []
        for y in reference.T:
            exported_y = np.array([y[reduced['state_names'].index(n)] for n in exported['state_names']])
            a,b = source_rhs(y), exported_rhs(exported_y)[order]
            residuals.append(float(np.max(np.abs(a-b)/(1+np.abs(a)))))
        result.update(n_time_points=len(times), equation_max_scaled_error=max(residuals),
                      initial_values_match=bool(np.allclose(actual[:,0],reference[:,0],rtol=1e-10,atol=1e-12)))
        result['accurate'] = bool(result['accurate'] and result['initial_values_match'] and max(residuals) <= tolerance)
        return result
    except Exception as exc:
        return {'accurate':False, 'error':f'{type(exc).__name__}: {exc}'}
