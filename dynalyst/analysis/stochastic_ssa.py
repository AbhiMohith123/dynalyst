"""Gillespie SSA engines, all runs at once.

  exact_numpy  Gillespie's direct method, the runs advanced in lock-step with
               numpy (one event per run per step). Exact.
  exact_jax    the same algorithm compiled with JAX (lax.while_loop + vmap).
               Exact and much faster for long runs (millions of events);
               used when jax is installed (it is, for jaxkineticmodel).
  tau_leap     explicit tau-leaping with Cao-Gillespie-Petzold (2006) step
               selection, exact SSA steps where few events would fire, and
               step rejection when a count would go negative. Approximate:
               used only when an exact run would take too long, and the
               result says so.

Every engine returns the state on a fixed time grid (molecules), the event /
step count per run, which runs ended in an absorbing state (no reaction can
fire any more), which hit the event limit, and how often a negative
propensity had to be set to 0.
"""
import time

import numpy as np

from dynalyst import config


def _record(out, k, run, hi, x):
    if hi > k[run]:
        out[run, k[run]:hi] = x
        k[run] = hi


SSA_PILOT_SECONDS = float(getattr(config, "SSA_PILOT_SECONDS", 10))


def projected_seconds(k, events, grid, seconds) -> float:
    """Approximate wall time of the full runs, from their own events so far: k is each run's next
    grid index, so the events over the model time reached give the events still to come."""
    reached = np.asarray(grid)[np.maximum(np.asarray(k) - 1, 1)]
    events = np.asarray(events, dtype=float)
    return float(seconds * (events * grid[-1] / reached).sum() / max(events.sum(), 1.0))


def _short_of_molecules(X, V):
    """(channels, runs) True where firing would take a species below zero. A rate law need not
    vanish when its reactant runs out (a constant removal, a custom law): molecules cannot go negative."""
    rows, cols = np.nonzero(V < 0)
    short = np.zeros((V.shape[1], X.shape[1]), dtype=bool)
    np.logical_or.at(short, cols, X[rows] + V[rows, cols][:, None] < 0)
    return short


def exact_numpy(a_fn, V, x0, grid, rng, n_runs, max_events, max_seconds=None, stop_over=None):
    """max_seconds: runs still going after this much wall time stop there (truncated,
    NaN after): an event-count estimate can be far off, e.g. from a busier state.
    stop_over: after SSA_PILOT_SECONDS, stop if these runs project to this many seconds or more."""
    t_wall = time.time()
    n, m = V.shape
    n_t = len(grid)
    t_end = grid[-1]
    X = np.repeat(np.asarray(x0, dtype=float)[:, None], n_runs, axis=1)
    t = np.zeros(n_runs)
    k = np.zeros(n_runs, dtype=int)
    ev = np.zeros(n_runs, dtype=np.int64)
    out = np.full((n_runs, n_t, n), np.nan)
    done = np.zeros(n_runs, dtype=bool)
    absorbed = np.zeros(n_runs, dtype=bool)
    trunc = np.zeros(n_runs, dtype=bool)
    clipped = blocked = 0
    time_limited = stopped_early = None
    while True:
        act = np.flatnonzero(~done)
        if act.size == 0:
            break
        if max_seconds is not None and time.time() - t_wall > max_seconds:
            trunc[act] = True
            time_limited = float(max_seconds)
            break
        if stop_over is not None and time.time() - t_wall >= SSA_PILOT_SECONDS:
            spent = time.time() - t_wall
            full = projected_seconds(k, ev, grid, spent)
            if full >= stop_over:
                trunc[act] = True
                stopped_early = {"after_seconds": round(spent, 1), "projected_seconds": round(full)}
                break
            stop_over = None
        Xa = X[:, act]
        a = a_fn(Xa)
        if not np.all(np.isfinite(a)):
            raise ValueError("Non-finite SSA propensity; simulation failed (not an absorbing state).")
        bad = ~(a >= 0)
        if bad.any():
            clipped += int(np.any(a < 0, axis=0).sum())
            a = np.where(bad, 0.0, a)
        short = _short_of_molecules(Xa, V)
        blocked += int(np.count_nonzero(short & (a > 0)))
        a = np.where(short, 0.0, a)
        a0 = a.sum(axis=0)
        if not np.all(np.isfinite(a0)):
            raise ValueError("Non-finite total SSA propensity.")
        with np.errstate(divide="ignore"):
            tn = t[act] + rng.exponential(size=act.size) / a0
        hi = np.minimum(np.searchsorted(grid, tn, side="left"), n_t)
        for c in np.flatnonzero(hi > k[act]):
            _record(out, k, act[c], hi[c], Xa[:, c])
        fin = (hi >= n_t) | (tn > t_end)
        for c in np.flatnonzero(fin):
            r = act[c]
            _record(out, k, r, n_t, Xa[:, c])
            done[r] = True
            absorbed[r] = a0[c] <= 0
        go = np.flatnonzero(~fin)
        if go.size:
            cum = np.cumsum(a[:, go], axis=0)
            u = rng.random(go.size) * a0[go]
            j = np.minimum((cum < u[None, :]).sum(axis=0), m - 1)
            idx = act[go]
            X[:, idx] += V[:, j]
            t[idx] = tn[go]
            ev[idx] += 1
            over = idx[ev[idx] >= max_events]
            if over.size:
                trunc[over] = True
                done[over] = True
    return out, {"events": ev, "absorbed": absorbed, "truncated": trunc,
                 "negative_propensity_steps": clipped, "blocked_events": blocked, "engine": "exact (numpy)",
                 "time_limited": time_limited, "stopped_early": stopped_early}


def tau_leap(a_fn, V, x0, grid, rng, n_runs, eps=0.03, max_steps=50_000_000, max_seconds=None, stop_over=None):
    """Explicit tau-leaping (Cao, Gillespie & Petzold 2006, J Chem Phys 124:044109),
    vectorised over runs. max_seconds: runs still going after this much wall time stop there
    (truncated, NaN after)."""
    t_wall, time_limited, stopped_early = time.time(), None, None
    n, m = V.shape
    n_t = len(grid)
    t_end = grid[-1]
    gap = float(grid[1] - grid[0]) if n_t > 1 else t_end
    X = np.repeat(np.asarray(x0, dtype=float)[:, None], n_runs, axis=1)
    t = np.zeros(n_runs)
    k = np.zeros(n_runs, dtype=int)
    ev = np.zeros(n_runs, dtype=np.int64)
    scale = np.ones(n_runs)
    out = np.full((n_runs, n_t, n), np.nan)
    done = np.zeros(n_runs, dtype=bool)
    absorbed = np.zeros(n_runs, dtype=bool)
    clipped = rejected = leaps = exact_steps = blocked = 0
    V2 = V ** 2
    g = 2.0                      # highest reaction order assumed 2 (conservative)
    steps = 0
    while steps < max_steps:
        steps += 1
        act = np.flatnonzero(~done)
        if act.size == 0:
            break
        if max_seconds is not None and time.time() - t_wall > max_seconds:
            time_limited = float(max_seconds)
            break
        if stop_over is not None and time.time() - t_wall >= SSA_PILOT_SECONDS:
            spent = time.time() - t_wall
            full = projected_seconds(k, ev, grid, spent)
            if full >= stop_over:
                stopped_early = {"after_seconds": round(spent, 1), "projected_seconds": round(full)}
                break
            stop_over = None
        Xa = X[:, act]
        a = a_fn(Xa)
        if not np.all(np.isfinite(a)):
            raise ValueError("Non-finite tau-leap propensity; simulation failed (not absorption).")
        bad = ~(a >= 0)
        if bad.any():
            clipped += int(np.any(a < 0, axis=0).sum())
            a = np.where(bad, 0.0, a)
        short = _short_of_molecules(Xa, V)
        blocked += int(np.count_nonzero(short & (a > 0)))
        a = np.where(short, 0.0, a)
        a0 = a.sum(axis=0)
        if not np.all(np.isfinite(a0)):
            raise ValueError("Non-finite total tau-leap propensity.")
        mu = np.abs(V @ a)
        s2 = V2 @ a
        bound = np.maximum(eps * Xa / g, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(mu > 0, bound / mu, np.inf).min(axis=0)
            t2 = np.where(s2 > 0, bound ** 2 / s2, np.inf).min(axis=0)
        tau = np.minimum(np.minimum(t1, t2) * scale[act], gap)
        tau = np.minimum(tau, t_end - t[act])
        exact = (tau * a0 < 10) | (a0 <= 0)
        # -- exact SSA steps where few reactions would fire --------------------------
        ex = np.flatnonzero(exact)
        if ex.size:
            with np.errstate(divide="ignore"):
                tn = t[act[ex]] + rng.exponential(size=ex.size) / a0[ex]
            hi = np.minimum(np.searchsorted(grid, tn, side="left"), n_t)
            fin = (hi >= n_t) | (tn > t_end)
            for c, r in enumerate(act[ex]):
                _record(out, k, r, n_t if fin[c] else hi[c], Xa[:, ex[c]])
                if fin[c]:
                    done[r] = True
                    absorbed[r] = a0[ex[c]] <= 0
            go = np.flatnonzero(~fin)
            if go.size:
                cum = np.cumsum(a[:, ex[go]], axis=0)
                u = rng.random(go.size) * a0[ex[go]]
                j = np.minimum((cum < u[None, :]).sum(axis=0), m - 1)
                idx = act[ex[go]]
                X[:, idx] += V[:, j]
                t[idx] = tn[go]
                ev[idx] += 1
                exact_steps += go.size
        # -- leaps ------------------------------------------------------------------
        lp = np.flatnonzero(~exact)
        if lp.size:
            kj = rng.poisson(a[:, lp] * tau[lp][None, :])
            Xn = Xa[:, lp] + V @ kj
            ok = (Xn >= 0).all(axis=0)
            rej = act[lp[~ok]]
            scale[rej] *= 0.5
            rejected += rej.size
            acc = np.flatnonzero(ok)
            if acc.size:
                runs = act[lp[acc]]
                tn = t[runs] + tau[lp[acc]]
                hi = np.minimum(np.searchsorted(grid, tn, side="left"), n_t)
                for c, r in enumerate(runs):
                    _record(out, k, r, hi[c], Xa[:, lp[acc[c]]])
                X[:, runs] = Xn[:, acc]
                t[runs] = tn
                ev[runs] += kj[:, acc].sum(axis=0)
                scale[runs] = np.minimum(1.0, scale[runs] * 1.5)
                leaps += acc.size
                fin = runs[tn >= t_end - 1e-12]
                for r in fin:
                    _record(out, k, r, n_t, X[:, r])
                    done[r] = True
    trunc = ~done
    return out, {"events": ev, "absorbed": absorbed, "truncated": trunc,
                 "negative_propensity_steps": clipped, "blocked_events": blocked,
                 "engine": "tau-leaping (approximate)",
                 "leaps": leaps, "exact_steps": exact_steps, "rejected_leaps": rejected,
                 "epsilon": eps, "time_limited": time_limited, "stopped_early": stopped_early}


def jax_available() -> bool:
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


JAX_CHUNK_EVENTS = 200_000     # events per run between two wall-time checks


def exact_jax(model, channels, values, omega, V, x0, grid, seed, n_runs, max_events,
              max_seconds=None, stop_over=None):
    """Gillespie's direct method compiled with JAX; one compiled loop per run,
    vmapped over runs. Same algorithm and outputs as exact_numpy. The loop runs in
    chunks of JAX_CHUNK_EVENTS events, the state carried from one to the next, so
    max_seconds can stop it between chunks: runs still going then are truncated."""
    import time
    import jax
    import jax.numpy as jnp
    import sympy as sp
    from dynalyst.analysis.stochastic import _omega_symbol, _param_subs, molecular_propensity

    jax.config.update("jax_enable_x64", True)
    n, m = V.shape
    Xs = sp.symbols(f"X0:{n}")
    subs = _param_subs(model, values)
    exprs = [molecular_propensity(model, ch["rate"], ch.get("stoich")).subs(subs).subs(_omega_symbol(), omega)
             .subs(dict(zip(model["state_symbols"], Xs))) for ch in channels]
    try:
        f = sp.lambdify([Xs], exprs, "jax")
    except Exception:
        f = sp.lambdify([Xs], exprs, [{"Abs": jnp.abs}, jnp])

    def props(x):
        vals = f(x)
        return jnp.stack([jnp.asarray(v, dtype=jnp.float64) * jnp.ones(()) for v in vals])

    Vj = jnp.asarray(V, dtype=jnp.float64)
    G = jnp.asarray(grid, dtype=jnp.float64)
    n_t = int(len(grid))
    t_end = float(grid[-1])
    x_init = jnp.asarray(x0, dtype=jnp.float64)
    max_ev = int(max_events)

    def chunk(c, limit):
        def cond(c):
            t, x, k, out, key, ev, a0, neg, blocked = c
            return (k < n_t) & (ev < limit) & jnp.isfinite(a0)

        def body(c):
            t, x, k, out, key, ev, _, neg, blocked = c
            key, k1, k2 = jax.random.split(key, 3)
            a = props(x)
            valid = jnp.all(jnp.isfinite(a))
            neg = neg + jnp.any(a < 0)
            a = jnp.where(jnp.isfinite(a) & (a > 0), a, 0.0)
            short = jnp.any((x[:, None] + Vj < 0) & (Vj < 0), axis=0)
            blocked = blocked + jnp.sum(short & (a > 0))
            a = jnp.where(short, 0.0, a)
            a0 = jnp.where(valid, jnp.sum(a), jnp.nan)
            tau = jnp.where(a0 > 0, jax.random.exponential(k1, dtype=jnp.float64) / a0, jnp.inf)
            tn = t + tau

            def rcond(s):
                kk, o = s
                return (kk < n_t) & (G[jnp.minimum(kk, n_t - 1)] < tn)

            def rbody(s):
                kk, o = s
                return kk + 1, o.at[kk].set(x)

            k, out = jax.lax.while_loop(rcond, rbody, (k, out))
            cum = jnp.cumsum(a)
            j = jnp.minimum(jnp.searchsorted(cum, jax.random.uniform(k2, dtype=jnp.float64) * a0,
                                             side="right"), m - 1)
            moving = (k < n_t) & (tn <= t_end)
            x = jnp.where(moving, x + Vj[:, j], x)
            return (tn, x, k, out, key, ev + 1, a0, neg, blocked)

        return jax.lax.while_loop(cond, body, c)

    def start(key):
        out0 = jnp.full((n_t, n), jnp.nan, dtype=jnp.float64)
        return (jnp.float64(0.0), x_init, jnp.int64(0), out0, key, jnp.int64(0), jnp.float64(1.0),
                jnp.int64(0), jnp.int64(0))

    keys = jax.random.split(jax.random.PRNGKey(int(seed)), int(n_runs))
    state = jax.vmap(start)(keys)
    step = jax.jit(jax.vmap(chunk, in_axes=(0, None)))
    t_wall, limit, time_limited, stopped_early = time.time(), 0, None, None
    while True:
        limit = min(max_ev, limit + JAX_CHUNK_EVENTS)
        state = step(state, jnp.int64(limit))
        t, x, k, out, key, ev, a0, neg, blocked = state
        if not np.all(np.isfinite(np.asarray(a0))):
            raise ValueError("Non-finite JAX SSA propensity; simulation failed (not absorption).")
        k_np, ev_np = np.asarray(k), np.asarray(ev)
        running = (k_np < n_t) & (ev_np >= limit)
        if not running.any() or limit >= max_ev:
            break
        if max_seconds is not None and time.time() - t_wall > max_seconds:
            time_limited = float(max_seconds)
            break
        spent = time.time() - t_wall        # chunks: the pilot can run past SSA_PILOT_SECONDS
        if stop_over is not None and spent >= SSA_PILOT_SECONDS:
            full = projected_seconds(k_np, ev_np, grid, spent)
            if full >= stop_over:
                stopped_early = {"after_seconds": round(spent, 1), "projected_seconds": round(full)}
                break
            stop_over = None
    t, x, k, out, key, ev, a0, neg, blocked = state
    return np.asarray(out), {"events": np.asarray(ev), "absorbed": np.asarray(a0 <= 0),
                             "truncated": np.asarray(k) < n_t,
                             "negative_propensity_steps": int(np.sum(neg)), "blocked_events": int(np.sum(blocked)),
                             "engine": "exact (JAX)", "time_limited": time_limited,
                             "stopped_early": stopped_early}


def numpy_seconds_per_step(a_fn, x0, n_runs, rng) -> float:
    """Measured cost of one lock-step iteration (propensities + selection)."""
    import time
    X = np.repeat(np.asarray(x0, dtype=float)[:, None] + 1.0, n_runs, axis=1)
    t0 = time.perf_counter()
    for _ in range(30):
        a = a_fn(X)
        a = np.where(a >= 0, a, 0.0)
        np.cumsum(a, axis=0)
        rng.random(n_runs)
    return (time.perf_counter() - t0) / 30 * 1.6     # + bookkeeping
