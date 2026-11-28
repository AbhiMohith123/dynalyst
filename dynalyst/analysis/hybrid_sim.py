"""Simulation of an SBML model with what an ODE solver alone does not handle: events
(discrete changes when a condition becomes true) and delays (rates that depend on an
earlier state, delay(x, tau) = x(t - tau)).

Delays: the method of steps. The run is integrated in pieces no longer than the shortest
delay, so x(t - tau) always falls in a piece already computed (exact for constant delays;
before time 0 every quantity has its starting value). A delay that depends on the state or
on time is refused.

Events (SBML semantics): an event fires when its trigger changes from false to true
(initialValue: the trigger's value just before time 0, so a trigger already true at the
start fires only when initialValue is false). The moment is found by bisection on the
solver's continuous output. Its assignments are applied at once, or after its delay, with
the values at trigger time (useValuesFromTriggerTime) or at execution time; a delayed
event that is not persistent is dropped when its trigger has become false by then. Events
at the same moment fire in document order (priorities are not used); an assignment that
makes another trigger true fires that event too.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from dynalyst.analysis.generic_sbml import DELAY, SBML_TIME

MAX_CASCADE = 100          # events firing one another at one moment
PIECES = 200               # a run without delays is integrated in this many pieces at least
MAX_DELAY_INTERVALS = 5000  # a delay equation is integrated one delay at a time: at most this many
SAMPLES = 40               # trigger checks per piece (plus the solver's own steps)


def has_delay(model: dict) -> bool:
    return any(isinstance(e, sp.Basic) and e.has(DELAY) for e in model["rhs"])


def simulate(model: dict, y0, param_values: dict, t_end: float, ts) -> tuple[np.ndarray, dict]:
    """(values at the times ts, one row per time, info). info: events fired (id, time,
    [(variable, before, after)]), notes on anything not applied."""
    names, states = model["state_names"], list(model["state_symbols"])
    pnames, psyms = list(model["param_names"]), list(model["param_symbols"])
    t_sym = model.get("time_symbol") or SBML_TIME
    pv = np.array([float(param_values[p]) for p in pnames], dtype=float)
    p_index = {p: i for i, p in enumerate(pnames)}
    s_index = {s: i for i, s in enumerate(names)}
    notes, fired = [], []

    # --- delays: each delay(expr, tau) becomes an argument computed from the history
    delays = sorted({d for e in model["rhs"] if isinstance(e, sp.Basic) for d in e.atoms(DELAY)},
                    key=sp.sstr)
    dsyms = [sp.Dummy(f"delayed{k}") for k in range(len(delays))]
    rhs = [e.xreplace(dict(zip(delays, dsyms))) if isinstance(e, sp.Basic) else e
           for e in model["rhs"]]
    rhs_fn = sp.lambdify([t_sym, *states, *psyms, *dsyms], rhs, modules="numpy")
    delay_expr, delay_tau = [], []
    for d in delays:
        what, tau = d.args
        if tau.free_symbols - set(psyms):
            raise NotImplementedError(
                f"delay({sp.sstr(what)}, {sp.sstr(tau)}): the delay depends on the state or on "
                "time; only constant delays are simulated")
        delay_expr.append(sp.lambdify([t_sym, *states, *psyms], what, modules="numpy"))
        delay_tau.append(sp.lambdify([*psyms], tau, modules="numpy"))

    pieces = []        # (start, end, dense solution) in time order: the run and its history
    jumps = {}         # right-continuous values, including jumps at t_end
    y_start = np.asarray(y0, dtype=float)

    def state_at(t):
        if t in jumps:
            return jumps[t]
        if t <= 0.0 or not pieces:
            return y_start
        for a, b, fn in reversed(pieces):
            if a <= t <= b:
                return fn(t)
        return pieces[-1][2](pieces[-1][1])

    def delayed_values(t, y):
        out = []
        for f_expr, f_tau in zip(delay_expr, delay_tau):
            tau = float(f_tau(*pv))
            if not np.isfinite(tau) or tau < 0:
                raise ValueError(f"Delay must be finite and nonnegative, got {tau}.")
            past = t - tau
            out.append(float(f_expr(past, *(y if tau == 0 else state_at(past)), *pv)))
        return out

    def f(t, y):
        vals = rhs_fn(t, *y, *pv, *delayed_values(t, y))
        return np.asarray([np.broadcast_to(v, ()) for v in vals], dtype=float)

    # --- events
    events = model.get("events", [])
    triggers, ev_delay, ev_assign = [], [], []
    for ev in events:
        triggers.append(sp.lambdify([t_sym, *states, *psyms], ev["trigger"], modules="numpy"))
        ev_delay.append(sp.lambdify([t_sym, *states, *psyms], ev["delay"], modules="numpy")
                        if ev["delay"] is not None else None)
        doable = []
        for var, kind, expr in ev["assignments"]:
            if kind not in ("state", "param"):
                notes.append(f"event {ev['id']}: the assignment to {var} ({kind}) is not applied")
                continue
            doable.append((var, kind, sp.lambdify([t_sym, *states, *psyms], expr, modules="numpy")))
        ev_assign.append(doable)
        if ev.get("has_priority"):
            notes.append(f"event {ev['id']} has a priority: simultaneous events fire in document "
                         "order instead")

    def true_now(k, t, y):
        try:
            return bool(triggers[k](t, *y, *pv))
        except Exception:
            return False

    def compute(k, t, y):
        """The values event k assigns, evaluated at (t, y) with the current parameters."""
        return [(var, kind, float(fn(t, *y, *pv))) for var, kind, fn in ev_assign[k]]

    def apply(values, y):
        """y with the assignments made (parameters changed in place); also what changed,
        [(variable, before, after)]."""
        y, changes = np.array(y, dtype=float), []
        for var, kind, value in values:
            if kind == "state":
                changes.append((var, float(y[s_index[var]]), value))
                y[s_index[var]] = value
            else:
                changes.append((var, float(pv[p_index[var]]), value))
                pv[p_index[var]] = value
        return y, changes

    scheduled = []     # (time, k, values or None)

    def fire(k, t, y):
        """Event k triggered at t: executed now, or scheduled after its delay."""
        if ev_delay[k] is not None:
            d = float(ev_delay[k](t, *y, *pv))
            values = compute(k, t, y) if events[k]["use_trigger_values"] else None
            scheduled.append((t + max(d, 0.0), k, values))
            scheduled.sort(key=lambda s: s[0])
            return y
        y, changes = apply(compute(k, t, y), y)
        fired.append((events[k]["id"], float(t), changes))
        return y

    def settle_events(t, y, before):
        """Fire every event whose trigger went false -> true at t, and the ones those make
        true (each at most MAX_CASCADE times at one moment). before: trigger values just
        before t; returns (y, trigger values now)."""
        for _ in range(MAX_CASCADE):
            now = [true_now(k, t, y) for k in range(len(events))]
            cancelled = {k for k in range(len(events))
                         if before[k] and not now[k] and not events[k]["persistent"]}
            scheduled[:] = [item for item in scheduled if item[1] not in cancelled]
            rising = [k for k in range(len(events)) if now[k] and not before[k]]
            if not rising:
                return y, now
            for k in rising:
                y = fire(k, t, y)
            before = now
        notes.append(f"events kept triggering one another at t = {t:g}; stopped after "
                     f"{MAX_CASCADE} rounds")
        return y, [true_now(k, t, y) for k in range(len(events))]

    t, y = 0.0, y_start.copy()
    before = [ev["initial_value"] for ev in events]
    if events:
        y, before = settle_events(0.0, y, before)
        jumps[0.0] = y.copy()

    min_tau = min([float(ft(*pv)) for ft in delay_tau if float(ft(*pv)) > 0] or [np.inf])
    piece = min(min_tau, t_end / PIECES) if (events or delays) else t_end
    if delays and min_tau == np.inf:
        piece = t_end / PIECES if events else t_end
    if delays and np.isfinite(min_tau) and t_end / min_tau > MAX_DELAY_INTERVALS:
        raise ValueError(
            f"t_end = {t_end:g} is {t_end / min_tau:.0f} delay intervals (the shortest delay is "
            f"{min_tau:g}); a delay equation is integrated one delay at a time, at most "
            f"{MAX_DELAY_INTERVALS} of them. Check that t_end is in the model's own time unit "
            f"(inspect_model's time_unit), or simulate a shorter time.")
    span = max(t_end, 1e-300)

    while t < t_end - 1e-12 * span:
        t_next = min(t_end, t + piece)
        if scheduled and scheduled[0][0] < t_next:
            t_next = max(scheduled[0][0], t)
        if t_next > t:
            sol = solve_ivp(f, (t, t_next), y, method="LSODA", dense_output=True,
                            rtol=1e-8, atol=1e-10)
            if not sol.success:
                raise ValueError(f"integration failed at t = {t:g}: {sol.message}")
            # Locate both edges: falling edges cancel nonpersistent instances.
            hit = None
            if events:
                grid = np.unique(np.concatenate([sol.t, np.linspace(t, t_next, SAMPLES)]))
                prev = before
                for a, b in zip(grid[:-1], grid[1:]):
                    now = [true_now(k, b, sol.sol(b)) for k in range(len(events))]
                    if now != prev:
                        lo, hi = a, b
                        for _ in range(60):
                            mid = 0.5 * (lo + hi)
                            mid_now = [true_now(k, mid, sol.sol(mid)) for k in range(len(events))]
                            if mid_now != prev:
                                hi = mid
                            else:
                                lo = mid
                        hit = hi
                        break
                    prev = now
            end = hit if hit is not None else t_next
            pieces.append((t, end, sol.sol))
            t, y = end, sol.sol(end)
            if hit is not None:
                y, before = settle_events(t, y, prev)
                jumps[t] = y.copy()
            if events:
                before = [true_now(k, t, y) for k in range(len(events))]
        # delayed events due now
        while scheduled and scheduled[0][0] <= t + 1e-12 * span:
            _, k, values = scheduled.pop(0)
            if not events[k]["persistent"] and not true_now(k, t, y):
                notes.append(f"event {events[k]['id']} dropped at t = {t:g}: its trigger was no "
                             "longer true when its delay ended (persistent = false)")
                continue
            before = [true_now(j, t, y) for j in range(len(events))]
            y, changes = apply(values if values is not None else compute(k, t, y), y)
            fired.append((events[k]["id"], float(t), changes))
            y, before = settle_events(t, y, before)
            jumps[t] = y.copy()

    out = np.empty((len(ts), len(names)))
    for i, tq in enumerate(ts):
        # at a jump the value after the event is shown (the last piece starting there)
        row = jumps.get(float(tq))
        if row is not None:
            out[i] = row
            continue
        for a, b, fn in reversed(pieces):
            if a <= tq <= b:
                row = fn(tq)
                break
        out[i] = row if row is not None else (pieces[-1][2](pieces[-1][1]) if pieces else y)
    info = {"events_fired": fired}
    if delays:
        info["delays"] = [f"{sp.sstr(d.args[0])}(t - {sp.sstr(d.args[1])})" for d in delays]
    if notes:
        info["notes"] = sorted(set(notes), key=notes.index)
    return out, info
