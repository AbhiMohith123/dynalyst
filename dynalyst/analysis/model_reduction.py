"""
Timescale analysis and model reduction: when is a quasi-steady-state
approximation (QSSA) justified, which variables are fast, and does the
reduced model reproduce the full one?

1. Conservation laws are eliminated first (exact, always valid).
2. Timescale analysis at a reference state (the steady state reached from
   the model's initial conditions): the Jacobian's eigenvalues give the
   relaxation rates |Re(lambda)|; the largest ratio between consecutive
   rates is the spectral gap. A gap >= gap_threshold (default 10) separates
   fast from slow modes; each fast mode is assigned the species that
   participates in it most (participation factors |w_ik * v_ik|). The fast
   species are found from the model, not assumed -- e.g. with enzymes in
   large excess the free substrate, not the enzyme complex, can be fastest.
3. QSSA: the fast species' equations are set to zero and solved for them,
   one variable at a time (linear, or quadratic with the physical root
   chosen at the reference state). If no closed form is found in time, a
   numerical slow manifold is used instead (reduced model usable for the
   validation but not exported).
4. Validation uses the original retained initial values and a shared time grid.
   All original species, including reconstructed states, are checked. Initial
   transient errors remain in the overall accuracy result.
5. The reduced model is exported as SBML (data/sbml_models/<name>.xml) so
   every other tool can use it.
"""
import json
import time

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from dynalyst import config


# --- expression hygiene -------------------------------------------------------------

def _clean(expr):
    """Integer-valued float exponents -> integers (SBML round trips write
    AK**2.0, which sympy's polynomial tools reject); float noise below 1e-12
    -> 0; other floats rounded to 12 significant digits."""
    expr = expr.replace(lambda e: e.is_Pow and e.exp.is_Float and float(e.exp).is_integer(),
                        lambda e: sp.Pow(e.base, int(e.exp)))
    return expr.xreplace({f: (sp.Integer(0) if abs(float(f)) < 1e-12 else
                              sp.Float(float(f"{float(f):.12g}")))
                          for f in expr.atoms(sp.Float)})


def _tidy(expr):
    # only tidied (tiny floats to 0, long floats shortened): together/cancel are cosmetic and
    # can run without end on the large expressions a QSSA substitution gives
    return _clean(expr)


# --- timescales --------------------------------------------------------------------

def _numeric(model, params):
    from dynalyst.analysis.steady_states import _lambdify_model
    rhs_fn, jac_fn = _lambdify_model(model)
    vec = [params[p] for p in model["param_names"]]
    return (lambda y: np.array(rhs_fn(*y, *vec), dtype=float),
            lambda y: np.array(jac_fn(*y, *vec), dtype=float))


_TRAJECTORY_STEPS = 20000    # solver steps of the run the timescales are sampled along
_TRAJECTORY_SECONDS = 120.0  # wall-clock safety net only (a step count is what limits a run)


def _trajectory(model, params, t_end=1e5):
    """The time course from the model's start, for sampling its timescales. A large model,
    or one that oscillates and never settles, is stopped after _TRAJECTORY_STEPS: the
    samples come from the part that was run."""
    import time as _time
    from types import SimpleNamespace
    from scipy.integrate import LSODA
    f, _ = _numeric(model, params)
    y0 = np.array([model["y0"][s] for s in model["state_names"]], dtype=float)
    # stepped by hand and stopped by the clock between steps: a wall-clock 'event' in
    # solve_ivp breaks its event root-finding (the clock does not go back when the solver
    # re-evaluates an earlier time) and fails at random under load
    solver = LSODA(lambda t, y: f(y), 0.0, y0, t_end, rtol=1e-10, atol=1e-13)
    ts, ys = [0.0], [y0]
    began = _time.monotonic()
    # stopped after a fixed number of solver steps, so the same model gives the same samples
    # on a busy or an idle machine; the clock is only a safety net
    while (solver.status == "running" and len(ts) <= _TRAJECTORY_STEPS
           and _time.monotonic() - began < _TRAJECTORY_SECONDS):
        message = solver.step()
        if solver.status == "failed":
            break
        ts.append(solver.t)
        ys.append(np.array(solver.y, dtype=float))
    return SimpleNamespace(t=np.array(ts), y=np.array(ys).T,
                           success=solver.status != "failed", message=message if ts[1:] else "")


def reference_state(model, params, t_end=1e5):
    f, _ = _numeric(model, params)
    y = _trajectory(model, params, t_end).y[:, -1]
    return y, float(np.max(np.abs(f(y))))


def _split_at(jac_y, names, n_structural, gap_threshold):
    """Rates, gap and fast species at one state; the n_structural smallest
    |eigenvalues| (conservation laws) are ignored."""
    lam, V = np.linalg.eig(jac_y)
    try:
        W = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        # repeated eigenvalues without a full set of eigenvectors (a defective Jacobian):
        # the pseudo-inverse still gives each species' share in each mode
        W = np.linalg.pinv(V)
    order = np.argsort(-np.abs(lam.real))
    if n_structural:
        order = order[:-n_structural]
    rates = np.abs(lam.real[order])
    part = np.abs(W.T * V)[:, order]
    part = part / np.maximum(part.sum(axis=0, keepdims=True), 1e-300)
    if len(rates) < 2:
        return rates, 1.0, 0, [], part, lam
    ratios = rates[:-1] / np.maximum(rates[1:], 1e-300)
    k = int(np.argmax(ratios))
    gap = float(ratios[k])
    n_fast = k + 1 if gap >= gap_threshold else 0
    fast, taken = [], set()
    for mode in range(n_fast):  # each fast mode -> its most involved species not yet taken
        for i in np.argsort(-part[:, mode]):
            if names[i] not in taken:
                fast.append(names[i]); taken.add(names[i]); break
    return rates, gap, n_fast, fast, part, lam


def timescale_analysis(model, params=None, gap_threshold=10.0, n_samples=25,
                       n_conservation_laws=0, min_fraction=0.6) -> dict:
    """Timescale separation along the trajectory from the model's initial
    conditions (sampled log-uniformly in time, plus the end state). QSSA is
    recommended when the same fast/slow split holds, with a gap >=
    gap_threshold, at >= min_fraction of the samples. Works in the model's
    own variables; conservation laws show up as structural zero eigenvalues
    and are skipped (pass their number)."""
    params = {**model["param_defaults"], **(params or {})}
    names = model["state_names"]
    n_dyn = len(names) - n_conservation_laws
    if n_dyn < 2:
        return {"qssa_recommended": False, "fast_species": [], "slow_species": list(names),
                "reason": f"{n_dyn}D after conservation laws: nothing to separate"}
    sol = _trajectory(model, params)
    f, jac = _numeric(model, params)
    ts = sol.t[sol.t > 0]
    picks = sorted({int(np.argmin(np.abs(ts - t))) for t in
                    np.geomspace(ts[0], ts[-1], n_samples)}) if len(ts) else []
    samples, skipped = [], 0
    for i in picks:
        y = sol.y[:, np.searchsorted(sol.t, ts[i])]
        with np.errstate(all="ignore"):
            J = jac(y)
        if not np.all(np.isfinite(J)):
            skipped += 1      # e.g. a rate law dividing by a species that is 0 there
            continue
        rates, gap, n_fast, fast, _, lam = _split_at(J, names, n_conservation_laws,
                                                     gap_threshold)
        samples.append({"t": float(ts[i]), "gap": gap, "fast": fast,
                        "rates": [float(r) for r in rates], "state": y})
    if not samples:
        return {"qssa_recommended": False, "fast_species": [], "slow_species": list(names),
                "reason": "the Jacobian is not finite at any sampled point of the trajectory "
                          "(e.g. a rate law divides by a species that is 0 there), so the "
                          "timescales cannot be read"}
    split_counts = {}
    for smp in samples:
        if smp["fast"]:
            key = tuple(sorted(smp["fast"]))
            split_counts[key] = split_counts.get(key, 0) + 1
    consensus = max(split_counts, key=split_counts.get) if split_counts else ()
    agree = [smp for smp in samples if tuple(sorted(smp["fast"])) == consensus and consensus]
    fraction = len(agree) / len(samples) if samples else 0.0
    ref = sorted(agree, key=lambda smp: smp["gap"])[len(agree) // 2] if agree else samples[-1]
    fast = [s for s in names if s in consensus]
    slow = [s for s in names if s not in consensus]
    recommended = bool(consensus) and fraction >= min_fraction and len(slow) > n_conservation_laws
    gaps = [smp["gap"] for smp in samples]
    return {
        "samples_along_trajectory": len(samples),
        "fraction_with_this_split": fraction,
        "spectral_gap": {"min": float(min(gaps)), "median": float(np.median(gaps)),
                         "max": float(max(gaps))} if gaps else None,
        "gap_threshold": gap_threshold,
        "relaxation_rates_at_reference": ref["rates"],
        "reference_time": ref["t"],
        "reference_state": dict(zip(names, map(float, ref["state"]))),
        "end_state_is_steady": bool(np.max(np.abs(f(sol.y[:, -1]))) <
                                    1e-6 * (1 + float(np.max(np.abs(sol.y[:, -1]))))),
        "fast_species": fast, "slow_species": slow,
        "qssa_recommended": recommended,
        "reason": (f"{fast} relax >= {gap_threshold:g}x faster than the rest at "
                   f"{fraction:.0%} of the trajectory (median gap "
                   f"{np.median([smp['gap'] for smp in agree]):.3g}x): QSSA on them is justified"
                   if recommended else
                   f"no consistent timescale gap >= {gap_threshold:g}x along the trajectory "
                   f"(best split {list(consensus) or 'none'} holds at {fraction:.0%} of it): "
                   f"QSSA not justified"),
    }


def _eliminate_conservation(model, laws, avoid=()):
    """Eliminate conservation laws, choosing which variable each removes:
    never one in `avoid` (the fast species QSSA needs to keep) if possible,
    preferring variables no right-hand side depends on (pure products), then
    the best-conditioned choice."""
    if not laws:
        return model
    C = np.array(laws)
    k, n = C.shape
    names, syms = model["state_names"], model["state_symbols"]
    used = set().union(*(e.free_symbols for e in model["rhs"]))
    order = sorted(range(n), key=lambda j: (names[j] in avoid, syms[j] in used,
                                             -float(np.max(np.abs(C[:, j])))))
    pivots = []
    for j in order:
        if np.linalg.matrix_rank(C[:, pivots + [j]]) == len(pivots) + 1:
            pivots.append(j)
        if len(pivots) == k:
            break
    y0 = np.array([model["y0"][s] for s in names], dtype=float)
    totals = C @ y0
    rest = [j for j in range(n) if j not in pivots]
    inv = np.linalg.inv(C[:, pivots])
    subs = {}
    for a, i in enumerate(pivots):
        subs[syms[i]] = _clean(sp.expand(sum(sp.Float(inv[a, b]) * (sp.Float(totals[b]) - sum(
            sp.Float(C[b, j]) * syms[j] for j in rest)) for b in range(k))))
    return {**model, "state_names": [names[j] for j in rest],
            "state_symbols": [syms[j] for j in rest],
            # substituted, not expanded: expanding is cosmetic and can run without end on the
            # large rational rate laws of big models
            "rhs": [_clean(model["rhs"][j].xreplace(subs)) for j in rest],
            "y0": {names[j]: float(y0[j]) for j in rest},
            "eliminated": [(names[i], subs[syms[i]]) for i in pivots]}


# --- QSSA ----------------------------------------------------------------------------

# Work limits of the closed-form QSSA (a fixed amount of algebra, so the same model always
# gets the same answer): an expression past QSSA_MAX_OPS operations is never handed to
# together/solve, which can run for hours on one swollen expression; at most
# QSSA_MAX_ORDERS elimination orders are tried (k fast species have k! of them). Past a
# limit the slow manifold is found numerically instead.
QSSA_MAX_OPS = 1500
QSSA_MAX_ORDERS = 24


def _qssa_symbolic(model, fast, params, ref, time_limit=20.0, info=None):
    """{fast_symbol: expression in slow symbols}, or None if no closed form.

    Eliminates the fast variables one at a time: for each variable, the
    remaining equation of lowest degree (1 or 2) in it is solved and
    substituted into the rest. Quadratics keep the physical root (the one
    matching the reference state). Elimination orders are tried until one
    succeeds, within the work limits above (time_limit: a safety net only).
    info (a dict) receives 'stopped' when a work limit ended the search."""
    import itertools
    info = {} if info is None else info
    syms = dict(zip(model["state_names"], model["state_symbols"]))
    pvals = {sp.Symbol(p): v for p, v in params.items()}
    ref_subs = {syms[s]: ref[s] for s in model["state_names"]}

    def too_big(e) -> bool:
        return sp.count_ops(e) > QSSA_MAX_OPS

    raw = [model["rhs"][model["state_names"].index(s)] for s in fast]
    if any(too_big(e) for e in raw):
        info["stopped"] = (f"a fast species' equation has more than {QSSA_MAX_OPS} operations, "
                           "past what is solved in closed form")
        return None
    eqs0 = [sp.numer(sp.together(e)) for e in raw]
    deadline = time.time() + time_limit

    def pick_root(roots, v):
        def err(r):
            val = complex(r.subs(ref_subs).subs(pvals).evalf())
            return abs(val.imag) * 1e6 + abs(val.real - ref[str(v)])
        return min(roots, key=err)

    for n_order, order in enumerate(itertools.permutations([syms[s] for s in fast])):
        if n_order >= QSSA_MAX_ORDERS:
            info["stopped"] = (f"no closed form in the first {QSSA_MAX_ORDERS} elimination "
                               "orders of the fast species")
            return None
        eqs, solved, ok = list(eqs0), {}, True
        for v in order:
            if time.time() > deadline:
                info["stopped"] = "the closed-form search ran past its time limit"
                return None
            best = None
            for ei, eq in enumerate(eqs):
                if not eq.has(v) or too_big(eq):
                    continue
                try:
                    poly = sp.Poly(eq, v)
                except sp.PolynomialError:
                    continue
                d = poly.degree()
                if d in (1, 2) and (best is None or d < best[0]):
                    best = (d, ei, poly)
            if best is None:
                ok = False
                break
            # the roots of a linear or quadratic equation from its coefficients, directly
            # (sp.solve simplifies them first, which can run for hours on large ones)
            coeffs = best[2].all_coeffs()
            if best[0] == 1:
                roots = [-coeffs[1] / coeffs[0]]
            else:
                qa, qb, qc = coeffs
                disc = sp.sqrt(qb ** 2 - 4 * qa * qc)
                roots = [(-qb + disc) / (2 * qa), (-qb - disc) / (2 * qa)]
            if not roots:
                ok = False
                break
            try:
                root = roots[0] if len(roots) == 1 else pick_root(roots, v)
            except (TypeError, ValueError):
                ok = False
                break
            if too_big(root):           # the expressions are swelling: try another order
                ok = False
                break
            solved = {k: e.subs(v, root) for k, e in solved.items()}
            solved[v] = root
            substituted = [e.subs(v, root) for j, e in enumerate(eqs) if j != best[1]]
            if any(too_big(e) for e in substituted):
                ok = False
                break
            eqs = [sp.numer(sp.together(e)) for e in substituted]
        if ok:
            return solved
    return None


# Work limits of the validation runs (right-hand-side evaluations, so the same model always
# gets the same answer). A reduced model without a closed form solves for its fast species at
# every evaluation, which is far dearer: it gets fewer.
VALIDATION_EVALUATIONS = 500_000
NUMERIC_QSSA_EVALUATIONS = 20_000


class _OutOfWork(Exception):
    pass


def _capped(fn, limit):
    calls = [0]

    def wrapped(t, y):
        calls[0] += 1
        if calls[0] > limit:
            raise _OutOfWork
        return fn(y)
    return wrapped


def _validate(full, reduced_rhs, fast, slow, params, ts_info, tolerance, t_end=None,
              numeric=False):
    """Integrate full and reduced models from the same slow-manifold point, each within a
    fixed amount of work (numeric: the reduced model solves for its fast species at each
    evaluation)."""
    f_full, _ = _numeric(full, params)
    names = full["state_names"]
    rates = ts_info["relaxation_rates_at_reference"]
    n_fast = len(fast)
    t_skip = 5.0 / rates[n_fast - 1]            # a few of the slowest fast timescales
    t_end = t_end or min(10.0 / max(rates[-1], 1e-12), 1e5)
    y0 = np.array([full["y0"][s] for s in names], dtype=float)
    limit = NUMERIC_QSSA_EVALUATIONS if numeric else VALIDATION_EVALUATIONS
    t_eval = np.linspace(0, t_end, 200)
    s_idx = [names.index(s) for s in slow]
    try:
        layer = solve_ivp(_capped(f_full, VALIDATION_EVALUATIONS), (0, t_skip), y0, method="LSODA",
                          rtol=1e-10, atol=1e-13)
        start = layer.y[:, -1]
        full_run = solve_ivp(_capped(f_full, VALIDATION_EVALUATIONS), (0, t_end), start,
                             method="LSODA", t_eval=t_eval, rtol=1e-10, atol=1e-13)
        red_run = solve_ivp(_capped(reduced_rhs, limit), (0, t_end), start[s_idx], method="LSODA",
                            t_eval=t_eval, rtol=1e-10, atol=1e-13)
    except _OutOfWork:
        return {"t_skip_initial_layer": t_skip, "t_end": t_end, "accurate": False,
                "tolerance": tolerance,
                "error": ("the validation run did not finish within its fixed amount of work"
                          + (" (the reduced model has no closed form, so each step solves for "
                             "the fast species)" if numeric else "")
                          + ": the reduction is NOT validated and should not replace the full "
                            "model")}
    if not (full_run.success and red_run.success):
        return {"error": "integration failed during validation"}
    a, b = full_run.y[s_idx], red_run.y
    if not np.all(np.isfinite(b)):
        # e.g. a QSSA root with a square root whose argument turns negative along the run
        return {"t_skip_initial_layer": t_skip, "t_end": t_end, "accurate": False,
                "tolerance": tolerance,
                "error": "the reduced model produced non-finite values (NaN or infinity) along "
                         "the run (its QSSA solution leaves its domain, e.g. the square root of "
                         "a negative number), so it cannot replace the full model"}
    scale = np.max(np.abs(a), axis=1, keepdims=True) + 1e-300
    rel = np.abs(a - b) / scale
    max_err, final_err = float(np.max(rel)), float(np.max(rel[:, -1]))
    return {"t_skip_initial_layer": t_skip, "t_end": t_end,
            "max_relative_error": max_err, "final_relative_error": final_err,
            "per_species_max_error": {s: float(e) for s, e in zip(slow, np.max(rel, axis=1))},
            "accurate": max_err <= tolerance, "tolerance": tolerance,
            "_curves": (t_eval, a, b)}


ONE_DIMENSION_NOTE = ("A one-dimensional autonomous model cannot oscillate: it settles or diverges. "
                      "An oscillation of the full model is lost in this reduced model.")


def reduce_model(model_name_or_path: str | None = None, species: list | None = None,
                 odes: dict | None = None, parameters: dict | None = None,
                 initial_conditions: dict | None = None, method: str = "auto",
                 fast_species: list | None = None, gap_threshold: float = 10.0,
                 tolerance: float = 0.05, parameter_overrides: dict | None = None,
                 model_name: str | None = None, export: bool = True,
                 validation_t_end: float | None = None, validation_points: int = 501) -> str:
    """Tool: conservation-law elimination + (if justified) QSSA, validated."""
    from dynalyst.analysis.model_classifier import load_any_model
    from dynalyst.analysis.ode_model import ode_strings
    if method not in ("auto", "conservation", "qssa"):
        return json.dumps({"error": "method must be 'auto', 'conservation' or 'qssa'."})
    try:
        full0, reader = load_any_model(model_name_or_path, species, odes, parameters,
                                       initial_conditions)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    params = {**full0["param_defaults"], **(parameter_overrides or {})}
    full0 = {**full0, "param_defaults": params, "rhs": [_clean(e) for e in full0["rhs"]]}
    from dynalyst.analysis.model_classifier import conservation_laws
    laws = conservation_laws(full0)
    # timescales in the model's own variables first, so the conservation
    # elimination below can keep the fast species as states
    ts_info = timescale_analysis(full0, params, gap_threshold=gap_threshold,
                                 n_conservation_laws=len(laws))
    ts_info.pop("reference_state_array", None)
    do_qssa = method == "qssa" or (method == "auto" and (ts_info.get("qssa_recommended")
                                                           or fast_species))
    fast = list(fast_species or ts_info.get("fast_species") or []) if method != "conservation" else []
    full = _eliminate_conservation(full0, laws, avoid=set(fast))
    eliminated = [name for name, _ in full.get("eliminated", [])]
    out = {"model": full0["label"], "reader": reader,
           "original_dimension": len(full0["state_names"]),
           "conservation_laws_eliminated": eliminated,
           "dimension_after_conservation": len(full["state_names"]),
           "timescale_analysis": ts_info}
    if method == "conservation":
        do_qssa = False
    if not config.QSSA_ENABLED and method != "conservation":
        # this setup has the QSSA turned off: conservation laws only, and it says so
        do_qssa, fast = False, []
        out["qssa_skipped"] = ("QSSA reduction is turned off in this setup (DYNALYST_QSSA=0): "
                               "only conservation laws were eliminated. The timescale analysis "
                               "is still reported.")
    if do_qssa and not fast:
        do_qssa = False
        out["qssa_skipped"] = "no fast species identified (pass fast_species to force one)"
    unknown = [s for s in fast if s not in full["state_names"]]
    if unknown and fast_species:     # the caller named them: say so, do not guess
        return json.dumps({"error": f"fast_species {unknown} are not state variables after "
                                    f"conservation elimination: {full['state_names']}"})
    if unknown:
        # found by the timescale analysis, but a conservation law made of fast species only
        # had to eliminate one of them: it is already algebraic (given by the law), so the
        # QSSA has nothing left to do for it
        fast = [s for s in fast if s not in unknown]
        out["fast_species_eliminated_by_conservation"] = unknown
        if not fast:
            do_qssa = False
            out["qssa_skipped"] = (f"the fast species {unknown} are already given by conservation "
                                   "laws (eliminated above); no other fast species remains")

    reduced = full
    reconstruction = dict(full.get('eliminated', []))
    numeric_rhs = numeric_reconstruct = None
    initial_layer = 0.0
    if do_qssa:
        slow = [s for s in full["state_names"] if s not in fast]
        if not slow:
            return json.dumps({"error": "every species is fast; nothing would remain."})
        ref = ts_info.get("reference_state") or dict(zip(full["state_names"],
                                                        reference_state(full, params)[0]))
        search = {}
        sol = _qssa_symbolic(full, fast, params, ref, info=search)
        syms = dict(zip(full["state_names"], full["state_symbols"]))
        f_full, _ = _numeric(full, params)
        s_idx = [full["state_names"].index(s) for s in slow]
        f_idx = [full["state_names"].index(s) for s in fast]
        if sol is not None:
            reconstruction.update({str(k):v for k,v in sol.items()})
            rhs = [_tidy(full["rhs"][i].subs(sol)) for i in s_idx]
            reduced = {**full, "state_names": slow, "state_symbols": [syms[s] for s in slow],
                       "rhs": rhs, "y0": {s: full["y0"][s] for s in slow}, "label": full["label"]}
            f_red, _ = _numeric(reduced, params)
            reduced_rhs = f_red
            out["qssa"] = {"fast_species": fast, "slow_species": slow, "closed_form": True,
                           "fast_species_expressions": {str(k): sp.sstr(_clean(v)) for k, v in sol.items()}}
        else:  # numerical slow manifold
            state = {"fast": np.array([ref[s] for s in fast])}

            def reconstruct_numeric(ys):
                def g(yf):
                    y = np.empty(len(full["state_names"])); y[s_idx] = ys; y[f_idx] = yf
                    return f_full(y)[f_idx]
                yf, info, ier, message = fsolve(g, state["fast"], xtol=1e-12, full_output=True)
                if not np.isfinite(yf).all() or np.max(np.abs(g(yf))) > 1e-7*(1+np.max(np.abs(yf))):
                    raise ValueError('Numerical QSSA root did not satisfy the fast equations: ' + message)
                state["fast"] = yf
                y = np.empty(len(full["state_names"])); y[s_idx] = ys; y[f_idx] = yf
                return dict(zip(full['state_names'], map(float,y)))
            def reduced_rhs(ys):
                y = reconstruct_numeric(ys)
                return f_full(np.array([y[n] for n in full['state_names']]))[s_idx]
            numeric_rhs, numeric_reconstruct = reduced_rhs, reconstruct_numeric
            reduced = {**full, 'state_names':slow, 'state_symbols':[syms[s] for s in slow],
                       'rhs':[sp.Integer(0)]*len(slow), 'y0':{s:full['y0'][s] for s in slow}}
            out["qssa"] = {"fast_species": fast, "slow_species": slow, "closed_form": False,
                           "note": "no closed-form quasi-steady state found"
                                   + (f" ({search['stopped']})" if search.get("stopped") else "")
                                   + "; the reduced model was evaluated numerically (the fast "
                                     "species solved at each step) for validation, not exported"}
        rates = ts_info.get('relaxation_rates_at_reference') or []
        if len(rates) >= len(fast) and rates[len(fast)-1] > 0:
            initial_layer = 5.0/rates[len(fast)-1]

    from dynalyst.analysis.reduction_validation import validate, validate_export
    rates = ts_info.get('relaxation_rates_at_reference') or []
    horizon = (validation_t_end if validation_t_end is not None else
               min(10/max(rates[-1],1e-12),1e5) if rates else 100.0)
    val = validate(full0, reduced, params, reconstruction, tolerance=tolerance,
                   t_end=horizon, n_points=validation_points, initial_layer=initial_layer,
                   numeric_rhs=numeric_rhs, numeric_reconstruct=numeric_reconstruct)
    curves, retained = val.pop('_curves', None), val.pop('_retained', None)
    out['validation'] = val
    if curves is not None:
        import uuid
        evidence_dir = config.OUTPUT_DIR / 'reduction' / (full0['label'] + '_' + uuid.uuid4().hex[:10])
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = evidence_dir / 'full_grid.npz'
        np.savez_compressed(evidence_path, time=curves[0], full=curves[1], reconstructed=curves[2],
                            species=np.array(full0['state_names']), retained=retained)
        val['full_grid_path'] = config.display_path(evidence_path)
        out['plot_path'] = _plot_validation(full0['label'], full0['state_names'], *curves)
    if numeric_rhs is not None:
        out['reduced_dimension'] = len(reduced['state_names'])
        if out['reduced_dimension'] == 1:
            out['dimension_note'] = ONE_DIMENSION_NOTE
        reduced = None

    if reduced is not None:
        out["reduced_dimension"] = len(reduced["state_names"])
        if out["reduced_dimension"] == 1:
            out["dimension_note"] = ONE_DIMENSION_NOTE
        out["reduced_state_variables"] = reduced["state_names"]
        out["reduced_odes"] = ode_strings(reduced)
        changed = reduced is not full0 and len(reduced["state_names"]) < len(full0["state_names"])
        if export and changed and curves is not None:
            name = model_name or f"{full0['label']}_reduced"
            res = _export(reduced, params, name)
            out["sbml_export"] = res
            if res.get('path') and not res.get('error'):
                res['validation'] = validate_export(res['path'], reduced, params, curves[0], retained)
                res['validated_against_full'] = bool(val['accurate'] and res['validation']['accurate'])
                res['usage_note'] = ('Validated on the reported full time grid.' if res['validated_against_full']
                                     else 'Exploratory reduction: full-grid agreement is not established; '
                                          'do not replace the full model with this export.')
        elif export and changed:
            out['sbml_export'] = {'error':'Export withheld because the full-grid validation did not complete.'}
    out["summary"] = _summary(out)
    return json.dumps(out, indent=2, default=float)


def _export(model, params, name) -> dict:
    from dynalyst.analysis.jaxkm_ode_to_sbml import odes_to_sbml
    base, n = name, 2
    while (config.LOCAL_SBML_MODELS_DIR / f"{name}.xml").exists():   # never replace a file
        name, n = f"{base}_{n}", n + 1
    odes = {s: sp.sstr(e) for s, e in zip(model["state_names"], model["rhs"])}
    used = set().union(*(e.free_symbols for e in model["rhs"])) - set(model["state_symbols"])
    res = json.loads(odes_to_sbml(species=model["state_names"], odes=odes,
                                  parameters={str(p): params[str(p)] for p in used},
                                  initial_conditions=model["y0"], model_name=name))
    return {"path": res.get("path"), "model_name": name, "error": res.get("error")}


def _plot_validation(label, slow, t, full, red):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(slow), 1, figsize=(7, 2.4 * len(slow) + 0.6), sharex=True,
                             squeeze=False)
    for ax, s, a, b in zip(axes[:, 0], slow, full, red):
        ax.plot(t, a, color="black", label="full model")
        ax.plot(t, b, "--", color="tab:red", label="reduced/reconstructed")
        ax.set_ylabel(s); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize="small"); axes[-1, 0].set_xlabel("Time")
    axes[0, 0].set_title(f"{label} -- full vs reduced model (all states)")
    fig.tight_layout()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PLOTS_DIR / f"{label}_qssa_validation.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return config.display_path(path)


def _summary(out) -> str:
    parts = [f"{out['original_dimension']}D model"]
    if out["conservation_laws_eliminated"]:
        parts.append(f"{len(out['conservation_laws_eliminated'])} conservation law(s) eliminated "
                     f"-> {out['dimension_after_conservation']}D")
    q = out.get("qssa")
    if q:
        v = out.get("validation", {})
        parts.append(f"QSSA on {q['fast_species']} -> {len(q['slow_species'])}D"
                     + (f"; max error vs full model {v['max_relative_error']:.2%} "
                        f"({'accurate' if v.get('accurate') else 'NOT accurate'} at "
                        f"{v.get('tolerance', 0):.0%} tolerance)" if "max_relative_error" in v else ""))
    else:
        parts.append(out["timescale_analysis"].get("reason", ""))
    return "; ".join(p for p in parts if p)


TOOL_DEFINITIONS = [{
    "name": "reduce_model",
    "description": (
        "Reduce a model's dimension: eliminates conservation laws, analyses timescales "
        "(Jacobian eigenvalues at steady state, spectral gap, which species are fast) and, when "
        "the timescale separation justifies it (or when forced), applies the quasi-steady-state "
        "approximation (QSSA) to the fast species. Validates the reduced model against the full "
        "one and exports it as SBML for the other tools."),
    "input_schema": {"type": "object", "properties": {
        "model_name_or_path": {"type": "string"},
        "species": {"type": "array", "items": {"type": "string"}},
        "odes": {"type": "object"}, "parameters": {"type": "object"},
        "initial_conditions": {"type": "object"},
        "method": {"type": "string", "enum": ["auto", "conservation", "qssa"],
                   "description": "'auto' (default): QSSA only if the timescale gap justifies it; "
                                  "'conservation': only eliminate conservation laws; 'qssa': force QSSA."},
        "fast_species": {"type": "array", "items": {"type": "string"},
                         "description": "Override which species are treated as fast."},
        "gap_threshold": {"type": "number", "description": "Rate ratio that counts as a timescale gap. Default 10."},
        "tolerance": {"type": "number", "description": "Max relative error for 'accurate'. Default 0.05."},
        "parameter_overrides": {"type": "object", "description": "Parameter values to reduce at."},
        "model_name": {"type": "string", "description": "Name for the exported SBML (default <model>_reduced)."},
        "export": {"type": "boolean", "description": "Export the reduced model as SBML. Default true."},
        "validation_t_end": {"type":"number", "description":"Full-grid validation horizon; starts at time zero."},
        "validation_points": {"type":"integer", "description":"Number of validation samples; default 501, every original state reconstructed."},
    }, "required": []},
}]

TOOL_FUNCTIONS = {"reduce_model": lambda **kw: reduce_model(**kw)}
