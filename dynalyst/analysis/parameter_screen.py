"""Which parameter is worth continuing? A cheap eigenvalue screen run before
AUTO when the user names no bifurcation parameter.

XPPAUT only pre-selects the first declared parameter; the user integrates to
a steady state and picks the parameter. This does both automatically and
from the model alone (no parameter names are assumed):

1. settle: integrate the model from its initial state and take where it
   ends up (the time-average of the last stretch, which is the equilibrium
   an oscillation circles around), so the analysis starts on the state the
   system actually lives on, not on whatever root is nearest the file's
   initial values;
2. for every parameter the equations depend on, follow that equilibrium
   over a decade either side of the parameter's value (natural continuation
   with Newton), and watch the Jacobian eigenvalues:
     - a real eigenvalue changing sign        -> fold or branch point
     - a complex pair's real part changing sign -> Hopf
     - the equilibrium disappearing             -> fold
3. rank parameters by how close (in decades) to their current value the
   first such event occurs. Parameters with no event keep the model's order.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from dynalyst import config
from dynalyst.analysis.steady_states import _lambdify_model, conservation_reduce


SETTLE_EVALUATIONS = 200_000   # right-hand-side evaluations per settling integration try
SETTLE_SECONDS = 120.0     # wall-clock safety net only: the evaluation count keeps results the same
FOLLOW_EVALUATIONS = 20_000   # ... and of one integration that looks for a lost equilibrium
FOLLOW_SECONDS = 60.0      # wall-clock safety net only


class _OutOfTime(Exception):
    pass


def settled_state(model: dict, t_end: float | None = None) -> tuple[dict, str]:
    """(state, note): where the model goes from its initial condition."""
    rhs_fn, jac_fn = _lambdify_model(model)
    names = model["state_names"]
    pv = [model["param_defaults"][p] for p in model["param_names"]]
    y0 = np.array([model["y0"][s] for s in names], dtype=float)
    if t_end is None:
        # a few hundred of the slowest relaxation times at the start
        # The Jacobian can be 0/0 exactly at the starting state (Hill terms with every species
        # starting at 0): look at a state nudged off it, and fall back to a fixed horizon.
        rates = np.array([])
        for probe in (y0, y0 + 1e-3 * (1.0 + np.abs(y0)), y0 + 1e-1 * (1.0 + np.abs(y0))):
            with np.errstate(all="ignore"):
                J = np.array(jac_fn(*probe, *pv), dtype=float)
            if np.all(np.isfinite(J)):
                ev = np.linalg.eigvals(J)
                rates = np.abs(ev.real[np.isfinite(ev.real) & (np.abs(ev.real) > 1e-12)])
                break
        t_end = float(np.clip(300.0 / rates.min(), 100.0, 1e6)) if rates.size else 1e3
    # A slow model is still moving at the first horizon: integrate on (10x, up to a horizon
    # of 1e7 and a time budget) while it does, and call it oscillating only when the swing
    # does not die down -- a slow approach to a steady state is not a cycle.
    import time as _time
    began = _time.monotonic()
    deadline = [began + SETTLE_SECONDS]

    calls, total = [0], [0]

    def f(t, y):
        # one integration can run without end on a stiff, fast-spiking model over a long
        # horizon: stopped after a fixed amount of work (the same on a busy or idle machine),
        # with the clock as a safety net
        calls[0] += 1
        total[0] += 1
        if calls[0] > SETTLE_EVALUATIONS or _time.monotonic() > deadline[0]:
            raise _OutOfTime
        return np.asarray(rhs_fn(*y, *pv), dtype=float)
    t0, y_from, swings, shorter = 0.0, y0, [], 0
    while len(swings) < 4:
        ts = np.linspace(t0 + 0.8 * (t_end - t0), t_end, 400)
        try:
            sol = solve_ivp(f, (t0, t_end), y_from, method="LSODA", rtol=1e-8, atol=1e-10,
                            t_eval=ts)
        except _OutOfTime:
            if swings or shorter >= 2:
                break
            # the first horizon is too long for this model: a 100x shorter one, fresh budget
            t_end, shorter = t0 + (t_end - t0) / 100.0, shorter + 1
            deadline[0] = _time.monotonic() + SETTLE_SECONDS
            calls[0] = 0
            continue
        if not sol.success or sol.y.shape[1] != ts.size or not np.all(np.isfinite(sol.y)):
            if swings:
                break
            return dict(model["y0"]), ("integration failed (" + (sol.message if not sol.success
                                       else "non-finite values") + "); using the initial state")
        tail = sol.y
        mean = tail.mean(axis=1)
        swings.append(float(np.max(np.ptp(tail, axis=1) / (np.abs(mean) + 1e-12))))
        if swings[-1] <= 1e-3 or (len(swings) > 1 and swings[-1] > 0.5 * swings[-2]):
            break           # settled, or a swing that keeps its size (a sustained cycle)
        if t_end * 10 > 1e7 or total[0] > SETTLE_EVALUATIONS:
            break           # no longer horizon: report it as still changing
        t0, y_from, t_end = t_end, sol.y[:, -1], t_end * 10
    if not swings:
        return dict(model["y0"]), ("integration too slow on this model (stopped after "
                                   f"{SETTLE_EVALUATIONS:,} evaluations per try); using the "
                                   "initial state")
    settled = swings[-1] <= 1e-3
    sustained = not settled and len(swings) > 1 and swings[-1] > 0.5 * swings[-2]
    note = ("settles to a steady state" if settled
            else "oscillates (sustained cycles); starting from the equilibrium they circle"
            if sustained else "still changing at the end of the run; starting from where it got to")
    return {s: float(v) for s, v in zip(names, mean)}, note


def _signature(J):
    """(number of real positive eigenvalues, largest real part of a complex pair), or None
    when the Jacobian is not finite there (e.g. a rate law dividing by a species at 0):
    no eigenvalues to compare, so no event is read from that point."""
    if not np.all(np.isfinite(J)):
        return None
    ev = np.linalg.eigvals(J)
    scale = max(1.0, float(np.max(np.abs(ev)))) if ev.size else 1.0
    cplx = np.abs(ev.imag) > 1e-8 * scale
    n_pos_real = int(np.sum((ev.real > 0) & ~cplx))
    hopf = float(np.max(ev.real[cplx])) if np.any(cplx) else None
    return n_pos_real, hopf


def screen_parameters(model: dict, start: dict, candidates=None, decades: float = 1.0,
                      steps_per_decade: int = 20) -> list[dict]:
    """One record per candidate parameter, most promising first."""
    red = conservation_reduce({**model, "y0": {**model["y0"], **start}})
    rhs_fn, jac_fn = _lambdify_model(red)
    names, pnames = red["state_names"], red["param_names"]
    used = {str(s) for e in red["rhs"] for s in sp.sympify(e).free_symbols}
    cands = [p for p in (candidates or model["param_names"]) if p in used and p in pnames]
    base = dict(red["param_defaults"])
    y_start = np.array([red["y0"][s] for s in names], dtype=float)

    def solve(pv, guess):
        f = lambda y: np.asarray(rhs_fn(*y, *pv), dtype=float)
        with np.errstate(all="ignore"):
            y, _, ier, _ = fsolve(f, guess, full_output=True, xtol=1e-11)
            ok = ier == 1 and np.all(np.isfinite(y)) and np.max(np.abs(f(y))) < 1e-6 * (1 + np.max(np.abs(y)))
        return (y if ok else None)

    pv0 = [base[p] for p in pnames]
    y_eq = solve(pv0, y_start)
    if y_eq is None:
        return []

    # parameters that are exponents (Hill coefficients, orders): searched only a factor of
    # EXPONENT_FACTOR either way -- a power of hundreds turns the rate into a step the
    # solver cannot follow, and that failure would be read as a fold
    exponents = {str(s) for e in red["rhs"] for pw in sp.sympify(e).atoms(sp.Pow)
                 for s in pw.exp.free_symbols}

    records = []
    for p in cands:
        v0 = float(base[p])
        k = pnames.index(p)
        span = min(decades, float(np.log10(EXPONENT_FACTOR))) if p in exponents else decades
        n = max(4, int(span * steps_per_decade))
        if v0 != 0:
            grids = [v0 * 10 ** (np.arange(1, n + 1) * s * span / n) for s in (+1, -1)]
        else:  # no scale to go by: a unit interval either side
            grids = [np.linspace(0, 1, n + 1)[1:], -np.linspace(0, 1, n + 1)[1:]]
        with np.errstate(all="ignore"):
            sig0 = _signature(np.array(jac_fn(*y_eq, *pv0), dtype=float)) or "undefined"

        def follow(v_from, v_to, y):
            """The equilibrium at v_to, reached from (v_from, y). A failed Newton
            solve is not yet a fold: stiff models lose it on a big step. Retry in
            small parameter sub-steps, then by integrating to where the system
            settles; only an equilibrium no route reaches has disappeared."""
            pv = list(pv0); pv[k] = float(v_to)

            def same_branch(a, b):
                # a fold's partner is another branch: the system JUMPS there (a big
                # change for a small parameter step). Following the same branch, each
                # small step moves the state a little.
                scale = np.abs(a) + 1e-6 * (np.max(np.abs(a)) + 1e-12)
                return bool(np.max(np.abs(b - a) / scale) < config.SCREEN_SAME_BRANCH_REL)

            # a direct solve that converges is taken (a one-point jump to another
            # equilibrium is caught by the persistence check where events are found)
            y_new = solve(pv, y)
            if y_new is not None:
                return y_new

            for n_sub in (8, 32):
                y_sub, ok = y.copy(), True
                for v in np.linspace(v_from, v_to, n_sub + 1)[1:]:
                    p_sub = list(pv0); p_sub[k] = float(v)
                    y_next = solve(p_sub, y_sub)
                    if y_next is None or not same_branch(y_sub, y_next):
                        ok = False
                        break
                    y_sub = y_next
                if ok:
                    return y_sub
            import time as _time
            deadline = _time.monotonic() + FOLLOW_SECONDS
            n_calls = [0]

            def g(t, z):     # one integration per point, stopped after a fixed amount of work
                n_calls[0] += 1
                if n_calls[0] > FOLLOW_EVALUATIONS or _time.monotonic() > deadline:
                    raise _OutOfTime
                return np.asarray(rhs_fn(*z, *pv), dtype=float)
            try:
                sol = solve_ivp(g, (0.0, 1e5), y, method="LSODA", rtol=1e-8, atol=1e-10)
                if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                    y_ivp = solve(pv, sol.y[:, -1])
                    if y_ivp is not None and same_branch(y, y_ivp):
                        return y_ivp
            except Exception:     # incl. out of time: no route reached it
                pass
            return None

        def step(v, y, v_prev=None):
            pv = list(pv0); pv[k] = float(v)
            y_new = follow(v0 if v_prev is None else v_prev, v, y)
            if y_new is None:
                return None, None
            with np.errstate(all="ignore"):
                sig = _signature(np.array(jac_fn(*y_new, *pv), dtype=float))
            return y_new, ("undefined" if sig is None else sig)

        def event(a, b):
            if a == "undefined" or b == "undefined":
                return None      # no eigenvalues at one of the points: nothing to compare
            if b is None:
                return "fold (equilibrium disappears)"
            if a[0] != b[0]:
                return "fold or branch point (real eigenvalue crosses 0)"
            if a[1] is not None and b[1] is not None and a[1] * b[1] < 0:
                return "Hopf (complex pair crosses 0)"
            return None

        events = []
        for grid in grids:
            y, prev_sig, prev_v = y_eq.copy(), sig0, v0
            for i, v in enumerate(grid):
                y_new, sig = step(v, y, prev_v)
                kind = event(prev_sig, sig)
                if kind and sig is not None and i + 1 < len(grid):
                    # a crossing persists past it; a signature that returns at the next
                    # point was a one-point excursion to another equilibrium, not a crossing
                    _, sig_next = step(grid[i + 1], y, prev_v)
                    if sig_next is not None and event(prev_sig, sig_next) is None:
                        continue
                if kind:
                    lo, hi, y_lo = prev_v, float(v), y
                    for _ in range(14):  # bisect the bracket for the crossing
                        mid = (float(np.copysign(np.sqrt(abs(lo)) * np.sqrt(abs(hi)), lo))
                               if lo * hi > 0 else lo / 2 + hi / 2)
                        y_mid, s_mid = step(mid, y_lo, lo)
                        if event(prev_sig, s_mid):
                            hi = mid
                        else:
                            lo, y_lo = mid, y_mid
                    ev = {"type": kind, "near": hi}
                    if kind.startswith("fold (equilibrium disappears)"):
                        # a real fold meets a partner branch at a finite state; a state
                        # growing or vanishing without bound instead means no steady
                        # state exists past here (a balance the model needs is broken)
                        with np.errstate(all="ignore"):
                            ratio = np.abs(y_lo) / (np.abs(y_eq) + 1e-300)
                            spread = float(np.nanmax(np.abs(np.log10(ratio[ratio > 0])))) \
                                if np.any(ratio > 0) else 0.0
                        if spread > config.SCREEN_RUNAWAY_DECADES:
                            ev["runaway"] = True
                    events.append(ev)
                    break
                # a point without eigenvalues keeps the last signature to compare with
                y, prev_v = y_new, float(v)
                prev_sig = prev_sig if sig == "undefined" else sig
        for e in events:
            e["decades_from_value"] = (abs(float(np.log10(abs(e["near"] / v0))))
                                       if v0 and e["near"] else None)
            # an event within ~1% of the model's own value: a model sitting exactly on a
            # bifurcation is rare; this is usually parameters the model ties together
            # (e.g. a rate split into two that must add up), which no single change keeps
            if e["decades_from_value"] is not None and e["decades_from_value"] < AT_VALUE_DECADES:
                e["at_model_value"] = True
        events.sort(key=lambda e: (_suspect(e),
                                   e["decades_from_value"] if e["decades_from_value"] is not None else 9))
        records.append({"parameter": p, "value": v0, "events": events})

    order = {p: i for i, p in enumerate(cands)}
    records.sort(key=lambda r: (not r["events"],
                                bool(r["events"] and _suspect(r["events"][0])),
                                # a model value of 0 has no distance in decades: ranked after the
                                # parameters whose crossing is measured from their own value
                                (9 if r["events"][0]["decades_from_value"] is None
                                 else r["events"][0]["decades_from_value"]) if r["events"] else 0,
                                order[r["parameter"]]))
    return records


AT_VALUE_DECADES = config.SCREEN_AT_VALUE_DECADES
EXPONENT_FACTOR = 4.0       # an exponent parameter is searched from value/4 to value*4


def _suspect(e) -> bool:
    """An event that is probably not a bifurcation (ranked after the real ones)."""
    return bool(e.get("at_model_value") or e.get("runaway"))
