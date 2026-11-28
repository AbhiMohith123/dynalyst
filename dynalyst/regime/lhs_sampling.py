"""
Latin Hypercube sampling of parameter sets, in two steps:

  1. range resolution (range_spec.resolve_bounds): whatever the request said
     -- percentages, folds, absolute offsets, exact bounds, changed
     baselines, fixed / vary-only parameters -- becomes final per-parameter
     bounds {param: (lo, hi, is_log)};
  2. lhs_sample_params: the one LHS engine, which only ever sees those bounds.

resolve_model_bounds anchors the ranges on a model's own parameter values.
"""
import numpy as np
from scipy.stats import qmc

from dynalyst.regime.range_spec import resolve_bounds, spec_from_args


# ---------------------------------------------------------------------------
# any model
# ---------------------------------------------------------------------------

def resolve_model_bounds(parsed, spread_factor=None, additive_offset=None, manual_bounds=None,
                         vary_only=None, exclude_params=None, rel_window=None,
                         allowed_bounds=None, extras=None):
    """Resolution for ANY model, anchored on its own parameter values (the ones
    its equations use: parsed['params'] of regime.parser / model_source)."""
    base = {p: parsed['param_values'][p] for p in parsed['params']}
    spec = spec_from_args(
        spread_factor=spread_factor, additive_offset=additive_offset, rel_window=rel_window,
        manual_bounds=manual_bounds, allowed_bounds=allowed_bounds, vary_only=vary_only,
        exclude_params=exclude_params, extras=extras)
    return resolve_bounds(base, spec)


# ---------------------------------------------------------------------------
# the LHS engine
# ---------------------------------------------------------------------------

def lhs_sample_params(bounds, n_samples, seed=None):
    """bounds: {param: (lo, hi, is_log)}. Returns a list of n_samples dicts,
    each {param: value}, spread via Latin Hypercube sampling. A parameter with
    lo == hi keeps that value in every sample."""
    params = list(bounds.keys())
    d = len(params)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_cube = sampler.random(n=n_samples)  # shape (n_samples, d), each col in [0,1)

    samples = []
    for row in unit_cube:
        sample = {}
        for j, p in enumerate(params):
            lo, hi, is_log = bounds[p]
            u = row[j]
            if lo == hi:
                sample[p] = lo
            elif is_log:
                sample[p] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
            else:
                sample[p] = float(lo + u * (hi - lo))
        samples.append(sample)
    return samples
