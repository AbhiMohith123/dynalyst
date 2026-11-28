"""Deterministic integration of the canonical model. No stochastic engines."""
import numpy as np
import sympy as sp


def integrate(model, values, grid, max_seconds=None):
    """max_seconds: raises TimeoutError past this wall time (a stiff model can take very long)."""
    import time as clock
    from scipy.integrate import solve_ivp
    from dynalyst.analysis import hybrid_sim
    grid = np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid))
            or grid[0] != 0 or np.any(np.diff(grid) <= 0)):
        raise ValueError("Simulation times must start at 0 and increase finitely.")
    start = [float(model["y0"][n]) for n in model["state_names"]]
    details = {}
    rhs = None
    method = "deterministic ODE"
    if model.get("events") or hybrid_sim.has_delay(model):
        trajectory, details = hybrid_sim.simulate(model, start, values, grid[-1], grid)
        backend = "SciPy LSODA with event/delay handling"
        method = "deterministic DDE" if hybrid_sim.has_delay(model) else "deterministic hybrid ODE"
    else:
        params = [values[n] for n in model["param_names"]]
        t = model.get("time_symbol", sp.Symbol("__simulation_time"))
        fn = sp.lambdify([t, *model["state_symbols"], *model["param_symbols"]],
                         model["rhs"], "numpy")
        rhs = lambda time, y: np.asarray(
            [np.broadcast_to(v, ()) for v in fn(time, *y, *params)], dtype=float)
        began = clock.time()

        def timed(time, y):
            if max_seconds is not None and clock.time() - began > max_seconds:
                raise TimeoutError(f"the ODE passed its {max_seconds:g} s limit")
            return rhs(time, y)
        sol = solve_ivp(timed, (0, grid[-1]), start, t_eval=grid, method="LSODA",
                        rtol=1e-8, atol=1e-10,
                        max_step=grid[-1] / 200 if any(e.has(t) for e in model["rhs"]) else np.inf)
        if not sol.success:
            raise ValueError(f"Integration of {model['label']} failed: {sol.message}")
        trajectory, backend = sol.y.T, "SciPy LSODA"
    if not np.all(np.isfinite(trajectory)):
        raise ValueError(f"Integration of {model['label']} gave non-finite values."
                         + _before_failure(model, trajectory, grid, rhs, start))
    negative = _below_zero(model, grid, trajectory)
    if negative:
        details["notes"] = details.get("notes", []) + [negative]
    return trajectory, {"method": method, "backend": backend, **details}


_BELOW_ZERO = ("a species amount below zero is outside what the model describes, and a rate law "
               "with a non-integer power or a root of it is undefined there. The equations as "
               "given let {name} fall below zero; for a modified model, check the change.")


def _species_only(model) -> bool:
    """SBML species cannot be negative; a typed equation file may have variables that can."""
    from dynalyst.analysis.typed_equations import is_typed_file
    return (str(model.get("source", "")).startswith("sbml")
            and not is_typed_file(model.get("source_path") or ""))


def _below_zero(model, times, ys):
    """The first species of an SBML model to go below zero in this run, said as a note."""
    if not _species_only(model) or not len(ys):
        return None
    scale = max(1e-12, float(np.max(np.abs(ys))))
    hits = np.argwhere(ys < -1e-9 * scale)
    if not len(hits):
        return None
    i, j = hits[hits[:, 0].argmin()]
    name = model["state_names"][j]
    return (f"{name} went below zero (first at t = {times[i]:.4g}, lowest {ys[i:, j].min():.3g}): "
            + _BELOW_ZERO.format(name=name))


def _before_failure(model, trajectory, grid, rhs, start) -> str:
    """Where a run that gave non-finite values left the range the model describes: the time
    of the first non-finite value, and the species that went below zero before it, or was
    still falling as it reached zero (the step that took it below zero is the one that failed)."""
    from scipy.integrate import solve_ivp
    bad = int(np.argmax(~np.all(np.isfinite(trajectory), axis=1)))
    times, ys = grid[:bad], trajectory[:bad]
    if rhs is not None and grid[bad] > 0:
        # the solver's own steps up to the failure: a dip below zero between output times shows
        try:
            sol = solve_ivp(rhs, (0, grid[bad]), start, method="LSODA", rtol=1e-8, atol=1e-10)
            times, ys = sol.t, sol.y.T
        except Exception:
            pass
    keep = np.all(np.isfinite(ys), axis=1)
    times, ys = times[keep], ys[keep]
    msg = f" The first non-finite value is at t = {grid[bad]:.4g}."
    note = _below_zero(model, times, ys)
    if note:
        return msg + f" Before it, {note}"
    if rhs is None or not len(ys) or not _species_only(model):
        return msg
    try:
        slope = np.asarray(rhs(times[-1], ys[-1]), dtype=float)
    except Exception:
        return msg
    top = np.maximum(np.max(np.abs(ys), axis=0), 1e-300)
    # at its rate there, which species reaches zero before the output time that failed
    to_zero = {j: ys[-1, j] / -slope[j] for j in range(ys.shape[1]) if slope[j] < 0}
    near = [j for j, dt in to_zero.items() if dt <= grid[bad] - times[-1]]
    if not near:
        return msg
    j = min(near, key=to_zero.get)
    name = model["state_names"][j]
    return (msg + f" Just before it (t = {times[-1]:.4g}), {name} had fallen to {ys[-1, j]:.3g} "
            f"(largest in the run {top[j]:.3g}) and was still falling (d{name}/dt = "
            f"{slope[j]:.3g}): the run failed as {name} reached zero. "
            + "A" + _BELOW_ZERO.format(name=name)[1:])
