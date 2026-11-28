"""Cheap sanity check of a loaded model, before any AUTO run (no model calls, no AUTO).

Catches the failures that look like numerical trouble but are in the equations:
  * degenerate scale  -- every Jacobian eigenvalue ~0 at the settled state (e.g. rates
                         never divided by a 1e-15 compartment volume): AUTO then stops at
                         every parameter value, whatever the settings;
  * no steady state   -- the settled state is not a root of the equations;
  * non-finite        -- the right-hand side is NaN / inf there.
"""
import numpy as np

from dynalyst import config


def sanity(model: dict) -> dict:
    """{'ok': bool, 'problems': [...], 'notes': [...], 'max_eig', 'residual'} (never raises)."""
    out = {"ok": True, "problems": [], "notes": [], "max_eig": None, "residual": None}
    names = model.get("state_names") or []
    if not names:
        return {**out, "ok": False, "problems": ["the model has no state variables"]}
    try:
        from dynalyst.analysis.parameter_screen import settled_state
        from dynalyst.analysis.steady_states import _lambdify_model
        rhs_fn, jac_fn = _lambdify_model(model)
        start, how = settled_state(model)
        y = np.array([start[s] for s in names], dtype=float)
        pv = [model["param_defaults"][p] for p in model["param_names"]]
        with np.errstate(all="ignore"):
            f = np.asarray(rhs_fn(*y, *pv), dtype=float)
            ev = np.linalg.eigvals(np.array(jac_fn(*y, *pv), dtype=float))
    except Exception as e:
        # our own inability to evaluate is not evidence against the model: say so, do not block
        return {**out, "notes": [f"sanity check could not be evaluated ({type(e).__name__}: "
                                 f"{str(e)[:100]}); not blocking"]}
    out["notes"].append(how)
    out["notes"] += [f"caveat: {c}" for c in model.get("caveats") or []]
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(ev)):
        out["problems"].append("the equations are not finite at the settled state")
    else:
        out["residual"] = float(np.max(np.abs(f)) / (1.0 + np.max(np.abs(y))))
        out["max_eig"] = float(np.max(np.abs(ev)))
        if out["max_eig"] < config.SANITY_MIN_EIG:
            out["problems"].append(
                f"every Jacobian eigenvalue is ~0 (largest {out['max_eig']:.1e}): the equations "
                "have no usable scale, so continuation cannot work. Usual cause: reaction rates "
                "not divided by the compartment volume, or a unit mix-up in the file")
        # only a run that settled must sit on a root; an oscillating run, one still moving at
        # the end or one stopped for time gives a start for Newton, not a steady state
        if out["residual"] > config.SANITY_MAX_RESIDUAL and how.startswith("settles"):
            out["problems"].append(f"the settled state is not a steady state (residual "
                                   f"{out['residual']:.1e})")
    out["ok"] = not out["problems"]
    return out
