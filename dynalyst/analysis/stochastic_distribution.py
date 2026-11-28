"""What the stochastic runs say about the number of stable states.

From long SSA (and CLE) runs, after a burn-in, every sampled value of a
read-out species is pooled over time and runs
into a distribution; its peaks are counted:

  1 peak   unimodal       one state
  2 peaks  bimodal        two states (bistability, or noise-induced)
  3+       multimodal     several states (multistability)

Independently, the deterministic attractors are found by integrating the ODE
from many states the stochastic runs visited (and the initial state) and
checking each end point's Jacobian eigenvalues (stable / marginal) or whether
it keeps oscillating (limit cycle). The verdict combines both:

  ODE stable states  SSA peaks   verdict
  1                  1           one stable state found (a sample: not proof of monostability)
  >= 2               >= 2        bistable / multistable, noise visits the states
  >= 2               1           deterministically bistable, but at this system
                                 size noise keeps the runs in (or merges) one state
  1                  >= 2        noise-induced bimodality (not in the ODE)
  limit cycle        any         oscillatory: the distribution's two humps are
                                 the cycle's turning points, not two states
  absorbing state                every run that reaches it stays (e.g. p = 0 in
                                 the normal forms): long-time distribution is a
                                 point mass there
Switching: each run's samples are assigned to the peak whose core they are in;
a switch is a move from one core to another. Runs that never switch while
different runs sit in different peaks mean the switching time is longer than
t_end (run longer to see hopping).
"""
import re
import time

import numpy as np
import sympy as sp

ATTRACTOR_SEARCH_SECONDS = 90     # Newton starts are tried until this much time has passed


# --- the read-out species ----------------------------------------------------------

def xpp_autovar(model) -> str | None:
    """The XPP file's @ autovar= (the species its bifurcation diagram plots)."""
    path = model.get("source_path")
    if not path or model.get("source") != "xpp":
        return None
    try:
        m = re.search(r"autovar\s*=\s*([A-Za-z_]\w*)", open(path, errors="replace").read())
    except OSError:
        return None
    return m.group(1) if m and m.group(1) in model["state_names"] else None


def choose_readout(model, samples, names):
    """autovar, else the species whose pooled samples vary most (relative)."""
    av = xpp_autovar(model)
    if av:
        return av, "the XPP file's autovar"
    best, score = names[0], -1.0
    for j, n in enumerate(names):
        v = samples[..., j].ravel()
        v = v[np.isfinite(v)]
        if v.size and np.mean(np.abs(v)) > 0:
            s = float(np.std(v) / (np.mean(np.abs(v)) + 1e-12))
            if s > score:
                best, score = n, s
    return best, "largest relative spread"


# --- modes ------------------------------------------------------------------------------

def modes(values, discrete: bool, max_points=400_000, seed=0):
    """Peaks of the pooled distribution. values in molecules (SSA) or the same
    units (CLE). Returns {n_modes, modes:[{at, weight, lo, hi}], grid, density,
    valleys, bandwidth}."""
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 20:
        return None
    if v.size > max_points:
        v = np.random.default_rng(seed).choice(v, max_points, replace=False)
    lo, hi = float(v.min()), float(v.max())
    spread = hi - lo
    if spread <= 1e-12 * max(1.0, abs(hi)) or (discrete and spread < 1):
        return {"n_modes": 1, "modes": [{"at": lo, "weight": 1.0, "lo": lo, "hi": hi}],
                "grid": np.array([lo - 0.5, lo, lo + 0.5]), "density": np.array([0, 1.0, 0]),
                "valleys": [], "bandwidth": 0.0, "point_mass": True}
    sd = float(np.std(v))
    scott = 1.06 * sd * v.size ** (-0.2)
    bw = max(scott, 0.75 if discrete else 0.0, 0.02 * spread)   # never resolve single integers
    sub = v if v.size <= 20_000 else np.random.default_rng(seed).choice(v, 20_000, replace=False)
    kde = gaussian_kde(sub, bw_method=bw / max(np.std(sub), 1e-12))
    grid = np.linspace(lo - 3 * bw, hi + 3 * bw, 600)
    dens = kde(grid)
    pk, _ = find_peaks(np.concatenate([[0], dens, [0]]), prominence=0.05 * dens.max())
    pk = pk - 1
    pk = pk[(pk >= 0) & (pk < len(grid))]
    if pk.size == 0:
        pk = np.array([int(np.argmax(dens))])
    # merge neighbouring peaks without a real dip between them (a shoulder, e.g. a slow
    # drift still in the pooled samples): the valley must be < 60 % of the lower peak
    merged = [pk[0]]
    for p in pk[1:]:
        a = merged[-1]
        dip = dens[a:p + 1].min()
        if dip > 0.6 * min(dens[a], dens[p]):
            merged[-1] = a if dens[a] >= dens[p] else p
        else:
            merged.append(p)
    pk = np.array(merged)
    # valleys between neighbouring peaks, then drop peaks holding < 2 % of the samples
    valleys = [int(a + np.argmin(dens[a:b + 1])) for a, b in zip(pk[:-1], pk[1:])]
    edges = [-np.inf] + [grid[i] for i in valleys] + [np.inf]
    weights = np.array([np.mean((v > edges[i]) & (v <= edges[i + 1])) for i in range(len(pk))])
    keep = weights >= 0.02
    if keep.sum() < len(pk):
        pk = pk[keep]
        valleys = [int(a + np.argmin(dens[a:b + 1])) for a, b in zip(pk[:-1], pk[1:])]
        edges = [-np.inf] + [grid[i] for i in valleys] + [np.inf]
        weights = np.array([np.mean((v > edges[i]) & (v <= edges[i + 1])) for i in range(len(pk))])
    found = [{"at": float(grid[p]), "weight": float(w),
              "lo": float(edges[i]) if np.isfinite(edges[i]) else lo,
              "hi": float(edges[i + 1]) if np.isfinite(edges[i + 1]) else hi}
             for i, (p, w) in enumerate(zip(pk, weights))]
    return {"n_modes": len(found), "modes": found, "grid": grid, "density": dens,
            "valleys": [float(grid[i]) for i in valleys], "bandwidth": float(bw),
            "point_mass": False}


def switching(series_per_run, mode_info, gap):
    """Switches between peak cores per run. series_per_run: (runs, time)."""
    ms = mode_info["modes"]
    if len(ms) < 2:
        return None
    centers = np.array([m["at"] for m in ms])
    vall = np.array(mode_info["valleys"])
    # core of a peak: closer to it than half-way to the neighbouring valley
    lo_core = np.array([centers[i] - 0.5 * (centers[i] - (vall[i - 1] if i > 0 else -np.inf))
                        for i in range(len(ms))])
    hi_core = np.array([centers[i] + 0.5 * ((vall[i] if i < len(vall) else np.inf) - centers[i])
                        for i in range(len(ms))])
    per_run, where_end, visited = [], [], []
    for s in series_per_run:
        s = s[np.isfinite(s)]
        state, count, seen = None, 0, set()
        for x in s:
            hit = np.flatnonzero((x >= lo_core) & (x <= hi_core))
            if hit.size:
                if state is not None and hit[0] != state:
                    count += 1
                state = int(hit[0])
                seen.add(state)
        per_run.append(count)
        where_end.append(state)
        visited.append(len(seen))
    total_time = gap * sum(len(s) for s in series_per_run)
    n_sw = int(sum(per_run))
    return {"switches_total": n_sw, "switches_per_run": per_run,
            "runs_visiting_more_than_one_state": int(sum(1 for x in visited if x > 1)),
            "runs_ending_in_each_state": {f"state {i + 1}": int(sum(1 for e in where_end if e == i))
                                          for i in range(len(ms))},
            "mean_residence_time": (float(total_time / (n_sw + len(series_per_run)))
                                    if n_sw else None)}


# --- deterministic attractors ------------------------------------------------------------

def ode_attractors(model, values, starts, t_settle, readout_j, scale, physical=None,
                   cycle_starts=8, report=None):
    """Steady states (Newton from many starts, physical ones only, stability from
    the Jacobian) and limit cycles (ODE integration from a few starts).

    starts: concentration vectors (visited states, the ODE path, random states
    in the visited box). physical(y) -> bool rejects states with negative
    species or negative propensities (a conserved total, e.g. free kinase,
    below zero)."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import fsolve, least_squares
    from dynalyst.analysis.stochastic import _param_subs
    states = model["state_symbols"]
    subs = _param_subs(model, values)
    rhs = [sp.sympify(e).subs(subs) for e in model["rhs"]]
    f_raw = sp.lambdify(states, rhs, "numpy")
    J_raw = sp.lambdify(states, sp.Matrix(rhs).jacobian(states), "numpy")
    f = lambda y: np.array([float(np.broadcast_to(v, ())) for v in f_raw(*y)])
    J = lambda y: np.array(J_raw(*y), dtype=float)
    n = len(states)
    try:
        from dynalyst.analysis.model_classifier import conservation_laws
        laws = conservation_laws({**model, "param_defaults": dict(values)})
    except Exception:
        laws = []
    C = np.array(laws) if len(laws) else np.zeros((0, n))
    y_ref = np.asarray(starts[0], dtype=float)
    target = C @ y_ref if len(C) else np.array([])
    starts = list(starts) + _conserved_starts(C, target, starts, n)

    found, nonphysical = [], []
    big = 1.0 + float(np.max(np.abs(starts))) if len(starts) else 1.0
    # the exact Jacobian: finite differences cost n evaluations per step and stall on stiff models
    jac = (lambda y: np.vstack([J(y), C])) if len(C) else None
    t_start, tried, stopped = time.time(), 0, False
    for y_guess in starts:
        if time.time() - t_start > ATTRACTOR_SEARCH_SECONDS:
            stopped = True               # enough starts tried: the search is a sample anyway
            break
        tried += 1
        y_guess = np.asarray(y_guess, dtype=float)
        if not np.all(np.isfinite(y_guess)):
            continue
        try:
            with np.errstate(all="ignore"):
                if len(C):
                    sol = least_squares(lambda y: np.concatenate([f(y), C @ y - target]), y_guess,
                                        jac=jac, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                        max_nfev=400).x
                else:
                    sol, _, ier, _ = fsolve(f, y_guess, fprime=J, full_output=True, xtol=1e-12)
                res = float(np.max(np.abs(f(sol))))
        except Exception:
            continue
        if not np.all(np.isfinite(sol)) or res > 1e-7 * (1 + float(np.max(np.abs(sol)))) * big:
            continue
        if not _converged(sol, f, J, C):
            continue    # a point the solver stalled on (a slow, nearly flat direction), not a root
        if any(np.max(np.abs(sol - q)) < 1e-5 * (1 + np.max(np.abs(q))) for q in found + nonphysical):
            continue
        ok = bool(np.all(sol >= -1e-7 * big)) and (physical is None or physical(sol))
        (found if ok else nonphysical).append(sol)

    def classify(y):
        eig = np.linalg.eigvals(J(y))
        re = np.sort(eig.real)[::-1]
        tol = 1e-7 * max(1.0, float(np.max(np.abs(eig))))
        if len(C):                       # drop one ~zero eigenvalue per conservation law
            idx = np.argsort(np.abs(eig.real))[len(C):]
            re = np.sort(eig.real[idx])[::-1]
        lead = float(re[0]) if re.size else 0.0
        stab = "stable" if lead < -tol else ("marginal (at a bifurcation point)"
                                             if abs(lead) <= tol else "unstable")
        return stab, lead

    out = []
    for y in found:
        stab, lead = classify(y)
        out.append({"kind": "steady state", "stability": stab, "leading_eigenvalue": lead,
                    "state": [float(x) for x in y], "readout": float(y[readout_j])})
    neg_stable = []
    for y in nonphysical:
        try:
            stab, lead = classify(y)
        except Exception:
            continue
        if stab == "stable":
            neg_stable.append({"readout": float(y[readout_j]),
                               "negative": bool(np.any(y < -1e-7 * big))})

    # limit cycles: integrate from a few starts and look for regular peaks
    fun = lambda t, y: f(y)
    for y0 in list(starts)[:cycle_starts]:
        try:
            s1 = solve_ivp(fun, (0, t_settle), y0, method="LSODA", rtol=1e-7, atol=1e-10)
            if not s1.success:
                continue
            tail = solve_ivp(fun, (0, 0.5 * t_settle), s1.y[:, -1], method="LSODA", rtol=1e-7,
                             atol=1e-10, t_eval=np.linspace(0, 0.5 * t_settle, 800))
        except Exception:
            continue
        r = tail.y[readout_j]
        if not np.all(np.isfinite(r)) or np.ptp(r) <= 1e-3 * max(scale, 1e-12):
            continue
        from scipy.signal import find_peaks
        pk, _ = find_peaks(r, prominence=0.3 * np.ptp(r))
        if len(pk) >= 3 and np.std(np.diff(pk)) / np.mean(np.diff(pk)) < 0.3:
            cyc = {"kind": "limit cycle", "readout_min": float(r.min()),
                   "readout_max": float(r.max()),
                   "period": float(np.mean(np.diff(tail.t[pk])))}
            if not any(a["kind"] == "limit cycle" and
                       abs(a["readout_max"] - cyc["readout_max"]) < 0.05 * np.ptp(r) for a in out):
                out.append(cyc)
    out.sort(key=lambda a: a.get("readout", a.get("readout_max", 0.0)))
    if report is not None:
        report.update(starts_tried=tried, starts_available=len(starts), time_limit_reached=stopped,
                      time_limit_seconds=ATTRACTOR_SEARCH_SECONDS)
    return out, neg_stable


NEWTON_STEP_TOLERANCE = 1e-6   # relative size of the Newton step that a steady state allows


def _converged(y, f, J, C) -> bool:
    """Is y a steady state, judged scale-free: the Newton step from y (with the conserved
    totals held) is negligible next to y. A small residual alone is not enough: on a slow
    direction (eigenvalue ~1e-4) a residual of 1e-5 is still a step of 0.1 from the root."""
    try:
        with np.errstate(all="ignore"):
            A = np.vstack([J(y), C]) if len(C) else J(y)
            b = np.concatenate([f(y), np.zeros(len(C))]) if len(C) else f(y)
            step = np.linalg.lstsq(A, b, rcond=None)[0]
    except (np.linalg.LinAlgError, ValueError):
        return True     # cannot judge: the residual test above decides
    if not np.all(np.isfinite(step)):
        return True
    return float(np.max(np.abs(step))) <= NEWTON_STEP_TOLERANCE * (1.0 + float(np.max(np.abs(y))))


def _conserved_starts(C, target, starts, n, n_random=60, seed=11):
    """Starts spread over the WHOLE conserved space, not only where the runs went:
    with X + Xp + Xpp = T and every run near the state where most of T is X, a box
    around the visited states never reaches a stable state where most of it is Xpp. For each law with
    non-negative coefficients the total is shared out among its species: once
    almost all in each species (corners), then at random (Dirichlet)."""
    if not len(C):
        return []
    rows = []
    for c, t in zip(C, target):
        if np.all(c <= 1e-12):
            c, t = -c, -t
        if np.all(c >= -1e-12) and t > 0:
            rows.append((c, t))
    if not rows:
        return []
    covered = np.zeros(n, dtype=bool)
    for c, _ in rows:
        covered |= c > 1e-12
    vis = np.nanmax(np.abs(np.array(starts, dtype=float)), axis=0) if len(starts) else np.ones(n)
    vis = np.where(np.isfinite(vis) & (vis > 0), vis, 1.0)
    rng = np.random.default_rng(seed)

    def make(pick):
        y = np.zeros(n)
        for c, t in rows:
            idx = np.flatnonzero(c > 1e-12)
            y[idx] += pick(len(idx)) * t / c[idx]
        free = ~covered
        y[free] = rng.uniform(0, 1.3, int(free.sum())) * vis[free]
        return y

    out = []
    width = max(len(np.flatnonzero(c > 1e-12)) for c, _ in rows)
    for k in range(width):                       # corners: ~90 % in one species of each law
        def corner(m, k=k):
            w = np.full(m, 0.1 / max(m - 1, 1))
            w[k % m] = 0.9 if m > 1 else 1.0
            return w
        out.append(make(corner))
    for _ in range(n_random):
        out.append(make(lambda m: rng.dirichlet(np.ones(m))))
    return out


def readout_from_states(attractors, names, current):
    """With two or more stable steady states, the species that separates them most
    (relative to its size) -- one that is low in one state and high in the other,
    rather than one that hardly differs."""
    st = [a["state"] for a in attractors
          if a["kind"] == "steady state" and a["stability"] != "unstable"]
    if len(st) < 2:
        return current
    st = np.array(st)
    score = np.ptp(st, axis=0) / (np.max(np.abs(st), axis=0) + 1e-12)
    return names[int(np.argmax(score))]


# --- verdict -----------------------------------------------------------------------------

def verdict(n_modes, attractors, absorbed_runs, n_runs, switch_info, readout, neg_stable=None,
            ends=None, search=None):
    stable = [a for a in attractors if a["kind"] == "steady state" and a["stability"] != "unstable"]
    cycles = [a for a in attractors if a["kind"] == "limit cycle"]
    n_ss = len(stable)
    lines = []
    if cycles:
        label = "oscillatory in the ODE"
        lines.append(f"The ODE settles on a limit cycle ({readout} between "
                     f"{cycles[0]['readout_min']:.4g} and {cycles[0]['readout_max']:.4g}, period "
                     f"{cycles[0]['period']:.3g}). The stochastic distribution ({n_modes} peak(s)) "
                     "reflects where the cycle spends its time, not separate stable states. This label "
                     "does not test the stochastic runs: ssa_oscillation does.")
        if n_ss:
            lines.append(f"There is also {n_ss} stable steady state(s): the cycle and the "
                         "state(s) coexist.")
    elif n_ss >= 2 and n_modes >= 2:
        label = "bistable" if max(n_ss, n_modes) == 2 else "multistable"
        lines.append(f"{n_ss} stable ODE states and {n_modes} peaks in the stochastic "
                     f"distribution of {readout}: {label}.")
    elif n_ss >= 2 and switch_info and switch_info["switches_total"]:
        label = "bistable with switching"
        lines.append(f"The ODE has {n_ss} stable states. The distribution of {readout} has one peak, but "
                     f"{switch_info['runs_visiting_more_than_one_state']} of {n_runs} runs moved between "
                     "the states within t_end.")
    elif n_ss >= 2:
        label = "bistable (deterministic only at this system size)"
        moved = {k: v for k, v in (ends or {}).items() if v} if ends else {}
        if len(moved) > 1:
            lines.append(f"The ODE has {n_ss} stable states and the stochastic distribution of "
                         f"{readout} has one peak so far, but runs ended near different stable "
                         f"states ({', '.join(f'{v} near {k}' for k, v in moved.items())}): the "
                         "runs are switching, slowly. The second peak needs a longer t_end "
                         "(or a burn_in that drops the start).")
        else:
            lines.append(f"The ODE has {n_ss} stable states, but the stochastic distribution of "
                         f"{readout} has one peak: within t_end no run left the first state (or "
                         "the states are too close to separate). Try longer runs, other initial "
                         "states, or another omega.")
    elif n_modes >= 2:
        label = "noise-induced bimodality" if n_modes == 2 else "noise-induced multimodality"
        lines.append(f"Only one stable ODE state, but {n_modes} peaks in the stochastic "
                     f"distribution of {readout}: the peaks come from noise (common near a fold, "
                     "or at low copy numbers), not from deterministic bistability.")
    else:
        label = "one stable state found"
        tried = (f"{search['starts_tried']} of {search['starts_available']} starting points" +
                 (f", stopped at its {search['time_limit_seconds']:.0f} s limit" if search.get("time_limit_reached") else "")
                 if search else "an unknown number of starting points")
        lines.append(f"One stable ODE state found and one peak in the stochastic distribution of {readout}. "
                     f"The search is a sample ({tried}): this does not prove the model is monostable.")
    if any(a["kind"] == "steady state" and a["stability"].startswith("marginal") for a in attractors):
        lines.append("A steady state has a zero eigenvalue: the parameters sit at a bifurcation "
                     "point (e.g. exactly at the fold), where noise lets runs leave it slowly.")
    if neg_stable and any(x["negative"] for x in neg_stable):
        vals = ", ".join(f"{x['readout']:.4g}" for x in neg_stable if x["negative"])
        lines.append(f"The ODE also has stable state(s) with negative values ({readout} = {vals}), "
                     "e.g. a pitchfork's lower branch in a normal form. Molecule numbers cannot be "
                     "negative, so the SSA / CLE only see the non-negative branch(es).")
    if absorbed_runs:
        lines.append(f"{absorbed_runs}/{n_runs} runs reached an absorbing state (no reaction can "
                     "fire, e.g. zero molecules of a self-producing species) and stayed there: "
                     "given enough time every run ends there, so the long-time distribution is a "
                     "point mass at it.")
    if switch_info:
        if switch_info["switches_total"]:
            lines.append(f"Runs switch between the states ({switch_info['switches_total']} switches, "
                         f"mean residence time ~{switch_info['mean_residence_time']:.3g}).")
        else:
            lines.append("No run switched between peaks within t_end: different runs sit in "
                         "different states. Switching is slower than t_end; run longer to see it.")
    return label, lines


# --- plot ----------------------------------------------------------------------------------

def plot(entries, grid, label, path):
    """One row per (omega, method): A) read-out traces, B) pooled histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = len(entries)
    fig, axes = plt.subplots(rows, 2, squeeze=False, figsize=(11, 3.8 * rows),
                             gridspec_kw={"width_ratios": [3, 1.2]})
    for r, e in enumerate(entries):
        ax, hx = axes[r]
        col = ("tab:red" if e["method"] == "SSA" else "tab:purple"
               if e["method"].startswith("SSA") else "tab:orange")
        for run in e["series"][:50]:
            ax.plot(grid, run, color=col, lw=0.8, alpha=0.35)
        if e.get("burn_in"):
            ax.axvline(e["burn_in"], color="grey", ls=":", lw=1)
        v = e["pooled"]
        v = v[np.isfinite(v)]
        bins = (np.arange(np.floor(v.min()) - 0.5, np.ceil(v.max()) + 1.5)
                if e["discrete"] and v.max() - v.min() < 150 else 60)
        hx.hist(v, bins=bins, orientation="horizontal", color="0.7", density=True)
        md = e["modes"]
        if md and not md.get("point_mass"):
            hx.plot(md["density"], md["grid"], color="k", lw=1.2)
        for a in e["attractors"]:
            if a["kind"] == "steady state" and a["stability"] != "unstable":
                for axx in (ax, hx):
                    axx.axhline(a["readout"] * e["omega"], color="b", ls="--", lw=1.1)
            elif a["kind"] == "limit cycle":
                for axx in (ax, hx):
                    axx.axhspan(a["readout_min"] * e["omega"], a["readout_max"] * e["omega"],
                                color="b", alpha=0.07)
        ax.set_ylabel(f"{e['readout']} (molecules)")
        ax.set_xlabel("time")
        what = {"SSA": "SSA (Gillespie) from the model's initial state",
                "CLE": "CLE (Langevin) from the model's initial state"}.get(
            e["method"], "SSA basin test: runs split, half started in each stable ODE state"
            if e["method"].startswith("SSA, started") else e["method"])
        ax.set_title(f"{what}, Ω = {e['omega']:g}\n{e['label']}", fontsize=9.5)
        hx.set_xlabel("density")
        hx.invert_xaxis()
        hx.set_ylim(ax.get_ylim())
        hx.set_title(f"{md['n_modes'] if md else '?'} peak(s)", fontsize=10)
    reference_note = (" (blue dashed: stable ODE states, shaded: ODE limit cycle)"
                      if any(e["attractors"] for e in entries) else " (empirical samples)")
    fig.suptitle(f"{label}: stochastic distribution{reference_note}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
