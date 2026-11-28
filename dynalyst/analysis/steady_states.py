"""
Steady-state finding + stability classification for ANY-dimensional model
(SBML-derived or directly-specified ODEs), built on ode_model.py.

Workflow:
  1. Lambdify the symbolic RHS and its Jacobian (sympy -> numpy, once).
  2. Multi-start scipy.optimize.fsolve/root from several initial guesses
     (the model's own default y0, plus randomized perturbations and a few
     long-time-simulation endpoints) to find distinct real steady states --
     necessary because bistable/multistable models (the whole reason this
     project cares about bifurcations) have MULTIPLE coexisting steady
     states, not one.
  3. Deduplicate solutions that converged to the same point.
  4. At each steady state, evaluate the Jacobian's eigenvalues and classify:
       - all Re(eigenvalue) < 0            -> stable
       - all Re(eigenvalue) > 0            -> unstable
       - mixed signs                       -> saddle (unstable), or unstable_focus_saddle
                                              when a complex pair has positive real part
       - any nonzero Im(eigenvalue) present -> ...-focus / oscillatory approach
     This generalizes the old 1D "lambda = df/dy" sign check to n dimensions.
"""
import json

import time

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, least_squares

from dynalyst.analysis.ode_model import jacobian, ode_strings, resolve_model


def _detect_conservation_laws(jac_fn, n: int, param_vec: list, rng: np.random.Generator,
                               n_samples: int = 6, tol: float = 1e-8) -> np.ndarray:
    """
    Detects linear conserved quantities c.y (e.g. total protein amount in a
    phosphorylation cycle) WITHOUT needing the reaction stoichiometric matrix,
    so it works uniformly for reaction-based, rateRule-based, or
    directly-specified ODE systems.

    Rationale: if c.y is conserved, d(c.y)/dt = c^T @ rhs(y) = 0 for EVERY y,
    which (differentiating that identity) means c^T @ Jacobian(y) = 0 for
    EVERY y too. So: take the left null space of the Jacobian at several
    random points and intersect them (via a stacked SVD) -- directions that
    survive across many random, unrelated points are structural conservation
    laws, not coincidental rank deficiency at one particular point.

    Returns an array of shape (k, n): rows are (roughly) orthonormal
    conserved directions (k=0 if none detected).
    """
    jacobians = []
    for _ in range(n_samples):
        # Positive states only: a conservation law holds at every state, and
        # negative concentrations make fractional powers (Hill terms X**n) NaN.
        for _attempt in range(5):
            y = rng.uniform(0.1, 5, n)
            with np.errstate(all="ignore"):
                J = np.array(jac_fn(*y, *param_vec), dtype=float)
            if np.all(np.isfinite(J)):
                jacobians.append(J)
                break
        else:
            return np.zeros((0, n))
    # A law holds at EVERY state: c^T J(y) = 0 for all the sampled y at once, i.e. c is in
    # the left null space of [J(y1) J(y2) ...] -- the intersection of the per-point null
    # spaces (their union would let coincidental near-null directions in). Zero is judged
    # relative to the largest singular value: rate constants may be tiny in some models.
    A = np.hstack(jacobians)
    _, s, vt = np.linalg.svd(A.T, full_matrices=True)
    s_full = np.concatenate([s, np.zeros(n - s.size)]) if s.size < n else s[:n]
    tol = max(tol, 1e-12) * (float(s.max()) if s.size else 1.0)
    null = vt[s_full <= tol]
    if null.shape[0] == 0 or null.shape[0] > n - 1:
        return np.zeros((0, n))
    return null


def _lambdify_model(model: dict):
    all_symbols = model["state_symbols"] + model["param_symbols"]
    rhs_fn = sp.lambdify(all_symbols, model["rhs"], modules="numpy")
    jac_expr = jacobian(model)
    jac_fn = sp.lambdify(all_symbols, jac_expr, modules="numpy")
    return rhs_fn, jac_fn


def _make_numeric_funcs(model: dict, param_values: dict):
    rhs_fn, jac_fn = _lambdify_model(model)
    param_vec = [param_values[p] for p in model["param_names"]]

    def f(y):
        return np.array(rhs_fn(*y, *param_vec), dtype=float)

    def jac(y):
        return np.array(jac_fn(*y, *param_vec), dtype=float)

    return f, jac


def across_a_pole(model, y, pv) -> bool:
    """Whether a state lies across a pole of the equations from the model's own starting state:
    some denominator of a rate law has the opposite sign there (or is 0). No trajectory from the
    starting state reaches it (an active form above its total in a Goldbeter-Koshland term),
    so it is a root of the algebra, not a state of the system. Negative states with no
    denominator changing sign are kept."""
    syms = model["state_symbols"] + model["param_symbols"]
    states = set(model["state_symbols"])
    dens = {a.base for ex in model["rhs"] for a in sp.sympify(ex).atoms(sp.Pow)
            if a.exp.is_negative and a.base.free_symbols & states}
    if not dens:
        return False
    pvec = [pv[p] for p in model["param_names"]]
    y0 = [float((model.get("y0") or {}).get(s, 0.0)) for s in model["state_names"]]
    for d in dens:
        fd = sp.lambdify(syms, d, "numpy")
        try:
            at_start, at_y = float(fd(*y0, *pvec)), float(fd(*y, *pvec))
        except (ZeroDivisionError, TypeError, ValueError):
            continue
        if at_start != 0 and np.isfinite(at_start) and not at_y * at_start > 0:
            return True
    return False


def exact_number(v) -> sp.Rational:
    """A model number as the exact rational its decimal form states (0.1 -> 1/10),
    not the binary approximation a float holds."""
    return sp.Rational(repr(float(v)))


def _exact_expr(e):
    return e.xreplace({x: exact_number(x) for x in e.atoms(sp.Float)})


def normalised_law(v: list) -> list:
    """Integer coefficients when the law is rational, first non-zero positive."""
    v = [sp.sympify(x) for x in v]
    first = next((x for x in v if x != 0), sp.Integer(1))
    if v and all(isinstance(x, sp.Rational) for x in v):
        den = sp.ilcm(*[x.q for x in v]) if len(v) > 1 else v[0].q
        ints = [x * den for x in v]
        nonzero = [abs(int(x)) for x in ints if x != 0]
        g = sp.igcd(*nonzero) if len(nonzero) > 1 else (nonzero[0] if nonzero else 1)
        ints = [sp.Integer(int(x) // g) for x in ints]
        # small integers read best; otherwise exact fractions with the first coefficient 1
        v = ints if max(nonzero or [0]) // g <= 10 ** 6 else [x / first for x in v]
        first = next((x for x in v if x != 0), sp.Integer(1))
    return [-x for x in v] if (first.is_number and first < 0) else v


def _vanishes(expr, n_points: int = 4) -> bool:
    """Is `expr` identically 0? Evaluated in exact rational arithmetic at random
    rational values of all its symbols: a non-zero rational function is 0 at such
    points only with negligible probability (Schwartz-Zippel), and nothing is
    rounded. Fast where symbolic simplification of large fractions is not."""
    syms = sorted(expr.free_symbols, key=str)
    rng = np.random.default_rng(12345)
    for _ in range(n_points):
        point = {s: sp.Rational(int(rng.integers(1, 10 ** 6)), int(rng.integers(1, 10 ** 3)))
                 for s in syms}
        try:
            val = expr.xreplace(point)
            # rationals evaluate exactly; anything else (exp, a power with a non-integer
            # exponent) to 60 digits: a sum that cancels exactly is 0 to every digit, one
            # that does not is far from it. (simplify can run without end on such sums.)
            if val.is_Rational:
                zero = val == 0
            else:
                terms = val.args if val.is_Add else (val,)
                size = max([abs(sp.N(t, 60)) for t in terms] + [sp.Integer(1)])
                zero = abs(sp.N(val, 60)) <= sp.Float("1e-40", 60) * size
        except Exception:
            return False
        if not zero:
            return False
    return True


_LAWS_CACHE = {}


def exact_conservation_laws(model: dict) -> list[list]:
    """Linear conserved quantities c.y of the model, with EXACT coefficients.

    Each right-hand side, at the model's parameter values (taken as the exact
    decimals they are written as), is a sum of numeric coefficients times
    state-dependent terms; c is conserved when c^T M = 0 for the matrix M of
    those coefficients (one column per distinct term), which sympy solves in
    exact arithmetic. A law the numerical Jacobian test still finds beyond
    those (equal terms written in different forms) is turned into rationals and
    kept only if c.rhs simplifies to exactly 0; failing that it is kept with
    float coefficients, and law_is_exact() says so."""
    states = list(model["state_symbols"])
    n = len(states)
    if n == 0:
        return []
    key = (tuple(model["state_names"]), tuple(str(e) for e in model["rhs"]),
           tuple(sorted((k, float(v)) for k, v in model["param_defaults"].items())))
    if key in _LAWS_CACHE:
        return [list(c) for c in _LAWS_CACHE[key]]
    subs = {sym: exact_number(model["param_defaults"][name])
            for sym, name in zip(model["param_symbols"], model["param_names"])
            if name in model["param_defaults"]}
    # floats to exact rationals FIRST: a rational put next to a float would turn back into one
    symbolic = [_exact_expr(sp.sympify(e)) for e in model["rhs"]]
    rhs = [e.xreplace(subs) for e in symbolic]
    columns = {}
    for i, e in enumerate(rhs):
        try:
            e = sp.expand(e, power_base=False, power_exp=False, log=False, multinomial=False)
        except Exception:
            pass
        for t in sp.Add.make_args(e):
            if t == 0:
                continue
            coeff, term = t.as_coeff_Mul()
            columns.setdefault(term, [sp.Integer(0)] * n)[i] += coeff
    if not columns:      # every right-hand side is 0: every state is conserved
        laws = [[sp.Integer(int(i == j)) for j in range(n)] for i in range(n)]
    else:
        M = sp.Matrix([[col[i] for col in columns.values()] for i in range(n)])
        laws = [normalised_law(list(v)) for v in M.T.nullspace()]

        # the numerical test: any law the term matrix missed?
        try:
            rhs_fn, jac_fn = _lambdify_model(model)
            param_vec = [model["param_defaults"][p] for p in model["param_names"]]
            numeric = _detect_conservation_laws(jac_fn, n, param_vec, np.random.default_rng(0))
        except Exception:
            numeric = np.zeros((0, n))
        if len(numeric) > len(laws):
            found = np.array([[float(x) for x in c] for c in laws]) if laws else np.zeros((0, n))
            for c in numeric:
                if len(laws) >= len(numeric):
                    break
                trial = np.vstack([found, c])
                if np.linalg.matrix_rank(trial, tol=1e-8) == found.shape[0]:
                    continue        # already in the span of the exact laws
                # mixed with a known law a direction has no simple coefficients (CaPr + Pr in
                # BIOMD39's total Ca): remove each known law at its last state first
                for law in found:
                    piv = int(np.flatnonzero(np.abs(law) > 1e-12)[-1])
                    c = c - c[piv] / law[piv] * law
                c = c / c[np.argmax(np.abs(c))]
                c[np.abs(c) < 1e-6] = 0.0   # SVD noise (1e-12 on 20 species) blocks the exact check
                guess = [sp.nsimplify(float(x), rational=True, tolerance=1e-8) for x in c]
                exact = any(_vanishes(sum(g * e for g, e in zip(guess, eqs)))
                            for eqs in (symbolic, rhs))   # for any parameter values, or these
                # a model written with float rounding (volume ratios as 0.6875000000000001) leaves
                # a 1e-17 residual: simple fractions that hold numerically are the law
                if (not exact and max(sp.Rational(g).q for g in guess) <= 10 ** 4
                        and _conserved_numerically([float(g) for g in guess], rhs_fn, param_vec, model)):
                    exact = True
                if not exact and not _conserved_numerically(c, rhs_fn, param_vec, model):
                    continue    # a near-null direction of the sampled Jacobians, not a law
                law = normalised_law(guess) if exact else [sp.Float(float(x), 17) for x in c]
                if np.linalg.matrix_rank(np.vstack([found, [float(x) for x in law]]), tol=1e-8) == found.shape[0]:
                    continue        # cleaned, it is a law already found (two noisy copies of one)
                laws.append(law)
                found = np.vstack([found, [float(x) for x in law]])
    laws = _as_pools(laws)
    _LAWS_CACHE[key] = [list(c) for c in laws]
    return laws


def _as_pools(laws: list) -> list:
    """The exact laws rewritten as total amounts (non-negative coefficients) where the span
    allows, smallest first, then the original laws the pools do not span. The null-space
    basis mixes pools: in the dual phosphorylation cycle A - K + AP - P + APP is the
    substrate total minus the kinase and phosphatase totals. Same span, so elimination is
    unchanged; each pool can then be a parameter."""
    exact = [c for c in laws if law_is_exact(c)]
    n = len(laws[0]) if laws else 0
    if not exact or len(exact) == n:
        return laws
    from dynalyst.analysis.crnt_network import _non_negative_laws
    # x is a combination of the laws exactly when x^T K = 0, K the null space of the laws
    K = sp.Matrix.hstack(*sp.Matrix(exact).nullspace())
    chosen = [normalised_law(p) for p in _non_negative_laws(K, len(exact))]
    for c in exact:
        if len(chosen) == len(exact):
            break
        if sp.Matrix(chosen + [list(c)]).rank() > len(chosen):
            chosen.append(c)
    return chosen + [c for c in laws if not law_is_exact(c)]


def _conserved_numerically(c, rhs_fn, param_vec, model, n_states: int = 12) -> bool:
    """Is c.y constant along the dynamics: c.f(y) = 0 (relative to the size of the terms)
    at states spread around the model's own, not only where the Jacobians were sampled?"""
    y0 = np.array([float(model["y0"][s]) for s in model["state_names"]], dtype=float)
    scale = np.abs(y0) + 1.0
    rng = np.random.default_rng(1)
    c = np.asarray(c, dtype=float)
    for _ in range(n_states):
        y = np.abs(y0) * rng.uniform(0.2, 2.0, y0.size) + rng.uniform(0.0, 1.0, y0.size) * scale
        with np.errstate(all="ignore"):
            f = np.asarray(rhs_fn(*y, *param_vec), dtype=float)
        if not np.all(np.isfinite(f)):
            continue
        if abs(float(c @ f)) > 1e-8 * (float(np.abs(c) @ np.abs(f)) + 1e-300):
            return False
    return True


def law_holds(law, rhs) -> bool:
    """c.rhs = 0 for every parameter value: exactly, or up to the float rounding of the model's
    own numbers (relative 1e-12 at random parameter and state values). False for a law true
    only at the nominal parameter values."""
    terms = [sp.sympify(c) * _exact_expr(sp.sympify(e)) for c, e in zip(law, rhs) if c != 0]
    res = sp.expand(sum(terms, sp.Integer(0)))
    if res == 0:
        return True
    syms = sorted(set().union(*(t.free_symbols for t in terms)), key=str)
    rng = np.random.default_rng(7)
    for _ in range(4):
        point = {s: sp.Float(rng.uniform(0.5, 2.0), 30) for s in syms}
        size = sum(abs(complex(t.xreplace(point))) for t in terms)
        if abs(complex(res.xreplace(point))) > 1e-12 * (size + 1e-300):
            return False
    return True


def law_is_exact(law) -> bool:
    return not any(isinstance(sp.sympify(x), sp.Float) for x in law)


def law_text(law, names) -> str:
    """'DA + DAp', '2*x + y/3': the conserved combination, coefficients as they are."""
    expr = sum((sp.sympify(c) * sp.Symbol(n) for c, n in zip(law, names) if c != 0), sp.Integer(0))
    return str(expr) + ("" if law_is_exact(law) else "  (coefficients found numerically)")


def classify_stability(eigenvalues: np.ndarray, conserved_count: int = 0) -> str:
    """
    Classifies stability from the Jacobian eigenvalues at a steady state.

    Zero eigenvalues are handled specially: models with a conservation law
    (e.g. a protein's total over its forms, X+Xp+Xpp = const, extremely common in kinetic
    models) structurally have exactly one zero eigenvalue at EVERY steady
    state, for EVERY parameter value -- that is a fixed consequence of the
    conserved quantity, not evidence of a bifurcation. A genuine bifurcation
    is a zero eigenvalue that appears only at a SPECIFIC parameter value
    (which is what run_bifurcation_analysis actually
    tests for, by tracking eigenvalues AS a parameter is swept -- a single
    steady-state snapshot can't distinguish the two cases on its own).
    Only `conserved_count` independently verified zero modes are excluded.
    Other neutral modes make linear stability inconclusive unless a positive
    real eigenvalue already establishes instability.
    """
    # A pointwise zero eigenvalue is not evidence of a conserved quantity.
    # Only callers with independently established laws may remove those modes.
    tol = 1e-7
    eigenvalues = np.asarray(eigenvalues, dtype=complex)
    if not np.all(np.isfinite(eigenvalues)):
        return "inconclusive (non-finite eigenvalues)"
    if conserved_count:
        order = np.argsort(np.abs(eigenvalues))
        removed = order[:conserved_count]
        if len(removed) != conserved_count or np.any(np.abs(eigenvalues[removed]) > tol):
            return "inconclusive (conservation modes could not be resolved)"
        eigenvalues = eigenvalues[order[conserved_count:]]
    re = eigenvalues.real
    im = eigenvalues.imag
    zero_mask = np.abs(re) <= tol
    n_zero = int(np.sum(zero_mask))
    nonzero_re = re[~zero_mask]
    oscillatory = np.any(np.abs(im[~zero_mask]) > 1e-6)

    suffix = f" (within {conserved_count} verified conserved total(s))" if conserved_count else ""

    if nonzero_re.size == 0:
        return "nonhyperbolic (linearization does not determine stability)" + suffix
    if n_zero and not np.any(nonzero_re > tol):
        return "nonhyperbolic (linearization does not determine stability)" + suffix
    if np.all(nonzero_re < -tol):
        return ("stable_focus" if oscillatory else "stable_node") + suffix
    if np.all(nonzero_re > tol):
        return ("unstable_focus" if oscillatory else "unstable_node") + suffix
    spiral_out = np.any((np.abs(im) > 1e-6) & (re > tol))
    if spiral_out:
        return ("unstable_focus_saddle (complex eigenvalues with positive real part: trajectories "
                "spiral away from it)" + suffix)
    return "saddle (unstable)" + suffix


CONTINUUM_POINTS_SHOWN = 6     # sample points of a continuum listed besides its summary
CONTINUUM_BISECTIONS = 50


def _is_stable(J) -> bool:
    return classify_stability(np.linalg.eigvals(J)).startswith("stable")


def _summarise_continuum(points, names, f, jac, is_root, symbolic_rhs=None):
    """(summary, the sample points to list) for steady states that lie on a continuum.
    The summary gives the variables that are the same at every point found (the continuum
    is where they take these values), the range the others were sampled over and, when one
    variable varies, where the stability off the continuum changes along it, located by
    bisection between neighbouring samples of different stability (as long as the points
    between them are still steady states)."""
    X = np.array([[e["state"][n] for n in names] for e in points], dtype=float)
    scale = max(1.0, float(np.max(np.abs(X))))
    spread = X.max(axis=0) - X.min(axis=0)
    fixed = {n: float(np.median(X[:, i])) for i, n in enumerate(names) if spread[i] <= 1e-6 * scale}
    varying = [n for n in names if n not in fixed]
    fixed_subs = {sp.Symbol(name): value for name, value in fixed.items()}
    line_verified = (len(varying) == 1 and symbolic_rhs is not None
                     and all(sp.simplify(sp.sympify(e).subs(fixed_subs)) == 0 for e in symbolic_rhs))
    def stability_at(y):
        J = np.asarray(jac(y), dtype=float)
        if line_verified:
            i = names.index(varying[0])
            # For a verified line of equilibria its tangent is neutral. Report
            # stability transverse to that line, without calling it a total.
            if is_root(y) and np.all(np.abs(J[:, i]) <= 1e-10 * (1 + np.max(np.abs(J)))):
                J = np.delete(np.delete(J, i, axis=0), i, axis=1)
        label = classify_stability(np.linalg.eigvals(J))
        return ("stable" if label.startswith("stable") else
                "unstable" if label.startswith(("unstable", "saddle")) else "inconclusive")
    labels = [stability_at(y) for y in X]
    summary = {
        "points_found": len(points),
        "fixed_on_it": fixed,
        "sampled_range": {n: [float(X[:, names.index(n)].min()), float(X[:, names.index(n)].max())]
                          for n in varying},
        "stable_points_found": labels.count("stable"),
        "unstable_points_found": labels.count("unstable"),
        "inconclusive_points_found": labels.count("inconclusive"),
        "equilibrium_line_verified": line_verified,
        "stability_basis": "transverse to a verified equilibrium line; otherwise full linearization",
    }
    keep = set()
    if len(varying) == 1:
        i = names.index(varying[0])
        order = list(np.argsort(X[:, i]))
        keep |= {order[0], order[-1]}
        segments, start = [], float(X[order[0], i])
        for a, b in zip(order, order[1:]):
            if labels[a] == labels[b]:
                continue
            keep |= {a, b}
            lo, hi, exact = float(X[a, i]), float(X[b, i]), True
            for _ in range(CONTINUUM_BISECTIONS):
                y = X[a].copy()
                y[i] = 0.5 * (lo + hi)
                if not is_root(y):
                    exact = False
                    break
                if stability_at(y) == labels[a]:
                    lo = y[i]
                else:
                    hi = y[i]
            change = 0.5 * (lo + hi)
            segments.append({"from": start, "to": change,
                             "stability": labels[a]})
            if not exact:
                segments[-1]["note"] = (f"the change lies between {X[a, i]:.6g} and {X[b, i]:.6g}; "
                                        "the points between are not all steady states")
            start = change
        segments.append({"from": start, "to": float(X[order[-1], i]),
                         "stability": labels[order[-1]]})
        summary["stability_along_it"] = segments
        where = ", ".join(f"{k} = {v:.6g}" for k, v in fixed.items()) or "the points found"
        summary["description"] = (
            (f"a verified line of steady states ({where}, any {varying[0]}): " if line_verified else
             f"a candidate equilibrium set ({where}); a continuous line is not established: ")
            + "; ".join(f"{seg['stability']} for {varying[0]} from {seg['from']:.6g} to {seg['to']:.6g}"
                        for seg in segments)
            + " (stability off the line; the ends are the range sampled, not the line's ends)")
    for k in np.linspace(0, len(points) - 1, CONTINUUM_POINTS_SHOWN).round().astype(int):
        if len(keep) >= CONTINUUM_POINTS_SHOWN:
            break
        keep.add(int(k))
    shown = [points[k] for k in sorted(keep)]
    return summary, shown


def spirals_outward(eigenvalues) -> bool:
    """A complex pair with positive real part: nearby trajectories spiral away."""
    ev = np.asarray(eigenvalues)
    return bool(np.any((np.abs(ev.imag) > 1e-6) & (ev.real > 1e-7)))


def conservation_reduce(model: dict, keep=()) -> dict:
    """
    For models with a structural conservation law (e.g. total protein amount
    X+Xp+Xpp = const), eliminates ONE state variable per conserved quantity
    via the algebraic relation implied by that law, substituting it out of
    every remaining RHS expression.

    This matters for anything beyond simple steady-state root-finding:
    continuation methods (AUTO-07p) generically assume the state Jacobian is
    full rank except at isolated bifurcation points. A conservation law makes
    it rank-deficient by k EVERYWHERE, which reads to the continuation engine
    as a permanent, unmoving fold -- causing spurious non-convergence, not a
    real bifurcation. Eliminating the redundant variable(s) first (standard
    practice for moiety-conserved kinetic models) restores a generically
    full-rank system, which is what continuation actually needs.

    Returns the model unchanged if no conservation law is detected. Otherwise
    returns a NEW dict with:
      - state_names / state_symbols / rhs / y0 reduced by one entry per law
      - "eliminated": [(name, expr_in_terms_of_remaining_states), ...] so the
        caller can reconstruct the dropped variable's value at any point.
    """
    laws = exact_conservation_laws(model)
    if 'conserved_totals' in model:
        # Continuation varies parameters: a nominal-parameter coincidence must not
        # remove a genuine dynamical direction away from that nominal point.
        laws = [c for c in laws if law_is_exact(c) and law_holds(c, model['rhs'])]
    if not laws:
        return model

    state_symbols = list(model["state_symbols"])
    state_names = list(model["state_names"])
    rhs = list(model["rhs"])
    y0 = dict(model["y0"])
    exact = all(law_is_exact(c) for c in laws)
    # the totals from the initial state, as the exact decimals it is written as
    y0_vec = sp.Matrix([exact_number(y0[s]) for s in state_names])

    # Eliminate one state per law, jointly: solve C_E y_E = T - C_R y_R for the
    # pivot states E, in exact arithmetic. Pivots are taken from the LAST states
    # (row reduction of the column-reversed matrix): the first states, which other
    # tools use as defaults (e.g. the plotted species), are kept.
    C = sp.Matrix(laws)
    n = C.cols
    if exact:
        # dependent laws (one a combination of others) eliminate nothing new: keep an
        # independent set spanning the same laws, one state per law
        R, independent = C.rref()
        if len(independent) < C.rows:
            C = R[:len(independent), :]
        order = sorted(range(n - 1, -1, -1), key=lambda j: state_names[j] in keep)
        reversed_C = C[:, order]
        _, rev = reversed_C.rref()
        pivots = sorted(order[int(p)] for p in rev[:C.rows])
        CE_inv = C[:, pivots].inv()
    else:
        # laws found numerically: exact row reduction of float noise picks inconsistent
        # pivots, so independence and pivots are judged by numerical rank instead
        C, pivots, CE_inv = _numeric_elimination(C)
    k = C.rows
    totals = C * y0_vec
    for b in range(C.rows):
        coeff = {n: str(C[b, j]) for j, n in enumerate(state_names) if C[b, j] != 0}
        for total, definition in (model.get("conserved_totals") or {}).items():
            if coeff == definition["coefficients"]:
                totals[b] = sp.Symbol(total)
                break
    rest = [j for j in range(len(state_names)) if j not in pivots]
    rest_syms = [state_symbols[j] for j in rest]
    subs, eliminated = {}, []
    for a, i in enumerate(pivots):
        expr = sp.expand(sum(CE_inv[a, b] * (totals[b] - sum(
            C[b, j] * s for j, s in zip(rest, rest_syms))) for b in range(k)))
        subs[state_symbols[i]] = expr
        eliminated.append((state_names[i], expr))
    # exact after substitution; sympy's simplify adds nothing to correctness and takes
    # minutes on rational (Michaelis-Menten) rate laws
    new_rhs = [rhs[j].subs(subs) for j in rest]
    out = {
        **model,
        "state_names": [state_names[j] for j in rest],
        "state_symbols": rest_syms,
        "rhs": new_rhs,
        "y0": {state_names[j]: float(y0[state_names[j]]) for j in rest},
        "eliminated": eliminated,
    }
    if not exact:
        out["conservation_note"] = (
            "some conservation-law coefficients were found numerically and could not be "
            "confirmed exactly; the eliminated species carry that numerical error")
    return out


def _numeric_elimination(C: sp.Matrix):
    """(independent laws, pivot columns, inverse of the pivot block) for numerically found
    conservation laws. Laws are kept while they raise the numerical rank; pivots are taken
    from the LAST states (as in the exact case), each one kept when it raises the rank of
    the pivot block, so the block is well conditioned and invertible."""
    A = np.array(C.tolist(), dtype=float)
    rank = lambda M: np.linalg.matrix_rank(M, tol=1e-9 * max(1.0, float(np.max(np.abs(M)))))
    rows = []
    for i in range(A.shape[0]):
        if rank(A[rows + [i], :]) > len(rows):
            rows.append(i)
    A = A[rows, :]
    k, n = A.shape
    pivots = []
    for j in range(n - 1, -1, -1):
        if len(pivots) == k:
            break
        if rank(A[:, pivots + [j]]) > len(pivots):
            pivots.append(j)
    pivots = sorted(pivots)
    return (sp.Matrix(A), pivots, sp.Matrix(np.linalg.inv(A[:, pivots])))


def reconstruct_eliminated(model_with_elimination: dict, reduced_state: dict, parameters=None) -> dict:
    """Given a dict of the reduced model's state values, adds back any
    variables eliminated by conservation_reduce, evaluated numerically."""
    full = dict(reduced_state)
    for name, expr in model_with_elimination.get("eliminated", []):
        subs = {sp.Symbol(k): v for k, v in {**model_with_elimination["param_defaults"],
                                              **(parameters or {}), **full}.items()}
        full[name] = float(expr.subs(subs))
    return full


def _term_scale_fn(model: dict):
    """For each right-hand side, the sum of the absolute values of its rate terms
    (products distributed over sums): the size of the fluxes an equilibrium
    balances, in the model's own units."""
    terms = []
    for expr in model["rhs"]:
        e = sp.sympify(expr)
        try:
            e = sp.expand(e, power_base=False, power_exp=False, log=False, multinomial=False)
        except Exception:
            pass
        terms.append(sp.Add(*[sp.Abs(t) for t in sp.Add.make_args(e)]))
    return sp.lambdify(model["state_symbols"] + model["param_symbols"], terms, modules="numpy")


def flux_scale(scale) -> np.ndarray:
    """What each equation's residual is measured against: its own rate terms, but at least
    1/1000 of the largest flux in the system. An equation whose terms all vanish at the
    state (every term proportional to a species that is 0 there) has no scale of its own:
    its residual is rounding noise, not imbalance."""
    scale = np.abs(np.asarray(scale, dtype=float))
    top = float(np.max(scale)) if scale.size and np.all(np.isfinite(scale)) else 0.0
    return scale + 1e-3 * top


def _free_amounts(model: dict, param_values: dict) -> list:
    """[(a, c)]: the free amounts the rate laws are written with, c - a.y >= 0 with c > 0 and
    a >= 0 (a total minus its bound forms, e.g. KTOT - AK), read off as linear factors of
    each state's rate at 0 that multiply a state (a binding rate AP*(KTOT - AK)); a bare
    I - w (FitzHugh-Nagumo's voltage) is a balance of two terms, not a free amount."""
    syms = model["state_symbols"]
    pv = {s: param_values[str(s)] for s in model["param_symbols"] if str(s) in param_values}
    found = []
    for s, rhs in zip(syms, model["rhs"]):
        e = rhs.subs(s, 0)
        if sp.count_ops(e) > 400:     # shortcut: factoring a huge rate law is too slow; skipped
            continue
        try:
            e = sp.factor(sp.nsimplify(e, rational=True))
        except Exception:
            continue
        factors = [f.base if f.is_Pow else f for f in sp.Mul.make_args(sp.fraction(e)[0])]
        for f in factors:
            if not f.free_symbols & set(syms) or not any(
                    g is not f and g.free_symbols & set(syms) for g in factors):
                continue
            try:
                poly = sp.Poly(f, *syms)
            except sp.PolynomialError:
                continue
            if poly.total_degree() != 1:
                continue
            try:
                a = np.array([-float(poly.coeff_monomial(x).subs(pv)) for x in syms])
                c = float(poly.coeff_monomial(1).subs(pv))
            except (TypeError, ValueError):
                continue
            for sign in (1.0, -1.0):        # the factor may come out as -(KTOT - AK)
                if sign * c > 0 and np.all(sign * a >= 0) and np.any(sign * a > 0):
                    found.append((sign * a, sign * c))
    return found


def kept_nonnegative(model: dict, param_values: dict) -> set:
    """The states the model itself keeps >= 0: their rate at 0 is never negative when the
    other states are >= 0 (a concentration). A state whose rate at 0 can be negative (a
    voltage, a recovery variable) may have either sign, so a negative value is physical."""
    n = len(model["state_names"])
    rhs_fn = sp.lambdify(model["state_symbols"] + model["param_symbols"], model["rhs"], modules="numpy")
    pvec = [param_values[p] for p in model["param_names"]]
    scale = max([1.0] + [abs(float(v)) for v in model["y0"].values()])
    # log-spread samples: a concentration's rate at 0 is its production, >= 0 at any size
    rng = np.random.default_rng(0)
    pts = scale * 10.0 ** rng.uniform(-3, 3, (64, n))
    # a model written with its totals substituted (free kinase = KTOT - AK - APK) only ever
    # reaches states whose free amounts are >= 0: test those, not AK = 500 with KTOT = 100
    for a, c in _free_amounts(model, param_values):
        load = pts @ a
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(load > 0, c / load, np.inf)
        pts = pts * np.minimum(1.0, t)[:, None]
    signs = rng.choice([-1.0, 1.0], (64, n))
    kept = set(model["state_names"])
    while True:     # a state of either sign can push another one below 0: test again with it
        mask = np.array([nm not in kept for nm in model["state_names"]])
        sample = np.where(mask, pts * signs, pts)
        now = set()
        for i, name in enumerate(model["state_names"]):
            y = sample.copy()
            y[:, i] = 0.0
            with np.errstate(all="ignore"):
                vals = np.array([float(np.asarray(rhs_fn(*row, *pvec), dtype=float)[i]) for row in y])
            vals = vals[np.isfinite(vals)]
            # tolerance from the typical size, not the largest (a Z**4 term reaches 1e12)
            if name in kept and (not vals.size or np.all(
                    vals >= -1e-9 * max(1.0, float(np.median(np.abs(vals)))))):
                now.add(name)
        if now == kept:
            return kept
        kept = now


def one_way_rate_check(model: dict, param_values: dict, kept: set, reached: list):
    """check(y) -> the rate-term factors of the `kept` states' equations that are negative at
    y. A factor counts when it is >= 0 everywhere in the box the model reaches (its initial
    state and simulation ends), e.g. (1 - M) with M a fraction: a state that makes it negative
    runs a one-way rate backwards, so it is not one the system can be in."""
    syms, names = model["state_symbols"], model["state_names"]
    free = {s for s, nm in zip(syms, names) if nm not in kept}
    reached = np.abs(np.array(reached, dtype=float))
    hi = np.max(reached, axis=0)
    hi = np.where(hi > 0, hi, max(float(np.max(hi)), 1e-9))
    pts = np.vstack([reached, np.random.default_rng(0).uniform(0, 1, (64, len(names))) * hi])
    pvec = [param_values[p] for p in model["param_names"]]
    terms = []
    for nm, rhs in zip(names, model["rhs"]):
        if nm not in kept:
            continue
        # factor by factor: (1 - M)/(K1 - M + 1) stays > 0 when both factors turn negative
        for t in sp.Add.make_args(sp.sympify(rhs)):
            for fac in sp.Mul.make_args(t.as_coeff_Mul()[1]):
                if fac.free_symbols & free or not fac.free_symbols & set(syms) or fac.is_Symbol:
                    continue
                if any(str(fac) == s for s, _, _ in terms):
                    continue
                fn = sp.lambdify(syms + model["param_symbols"], fac, modules="numpy")
                with np.errstate(all="ignore"):
                    vals = np.array([float(fn(*y, *pvec)) for y in pts])
                if np.all(np.isfinite(vals)) and np.all(vals >= 0):
                    terms.append((str(fac), fn, max(float(np.max(np.abs(vals))), 1e-300)))

    def check(y):
        with np.errstate(all="ignore"):
            return [s for s, fn, top in terms if float(fn(*y, *pvec)) < -1e-9 * top]
    return check


def _dedupe(points: list[np.ndarray], tol: float = 1e-5) -> list[np.ndarray]:
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - u) < tol * (1 + np.linalg.norm(u)) for u in unique):
            unique.append(p)
    return unique


def _settle_time(f, y) -> float:
    """A few hundred of the slowest relaxation times at y (finite-difference Jacobian),
    within [100, 1e5]: long enough to settle, short enough for an oscillating model."""
    y = np.asarray(y, dtype=float)
    try:
        h = 1e-6 * np.maximum(1.0, np.abs(y))
        cols = []
        for j in range(y.size):
            e = np.zeros(y.size)
            e[j] = h[j]
            cols.append((np.asarray(f(y + e), dtype=float) - np.asarray(f(y - e), dtype=float))
                        / (2 * h[j]))
        ev = np.linalg.eigvals(np.column_stack(cols))
        rates = np.abs(ev.real[np.isfinite(ev.real) & (np.abs(ev.real) > 1e-12)])
        return float(np.clip(300.0 / rates.min(), 100.0, 1e5)) if rates.size else 1e4
    except Exception:
        return 1e4


_WARMUP_STEPS = 3000      # solver steps per warm-up simulation (a starting guess only)
_WARMUP_SECONDS = 30.0   # wall-clock safety net only: a step count keeps results the same
STEADY_SEARCH_SECONDS = 90.0   # Newton starts are tried until this much time has passed


def _candidate_starts(model: dict, f, n_random: int, rng: np.random.Generator,
                      wide: bool = False) -> list[np.ndarray]:
    """Starting guesses for the root finder. wide=True (find_steady_states) adds starts on the
    scale of the model's parameters as well: a model whose initial values are all 0 has its
    states on the scale of its totals and constants, which the starts around y0 never reach."""
    n = len(model["state_names"])
    y0_default = np.array([model["y0"][s] for s in model["state_names"]], dtype=float)
    starts = [y0_default]

    # A couple of long-time forward simulations from default and from zero,
    # since simulating to a steady state is a very reliable initial guess
    # for fsolve even when the system is stiff or the default y0 is far off.
    # a run that heads off to infinity (no equilibrium that way) is stopped, not chased
    limit = 1e12 * max(1.0, float(np.max(np.abs(y0_default))) if n else 1.0)
    # each run only supplies a starting guess (the root finder gives the steady states), so
    # it gets a few seconds: an oscillating model never settles and would run for minutes.
    # Stepped by hand and stopped by the clock between steps (a wall-clock event in
    # solve_ivp breaks its event root-finding under load)
    import time as _time
    from scipy.integrate import LSODA
    for start in (y0_default, np.zeros(n)):
        try:
            solver = LSODA(lambda t, y: f(y), 0.0, start, _settle_time(f, start),
                           rtol=1e-6, atol=1e-9)
            began, y = _time.monotonic(), start
            steps = 0
            while (solver.status == "running" and steps < _WARMUP_STEPS
                   and _time.monotonic() - began < _WARMUP_SECONDS):
                solver.step()
                steps += 1
                y = np.asarray(solver.y, dtype=float)
                if not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) > limit:
                    y = None     # headed off to infinity: no equilibrium that way
                    break
            if y is not None and solver.status != "failed":
                starts.append(y)
        except Exception:
            pass

    # Randomized perturbations around the default state (log-scale where
    # positive, since concentrations span orders of magnitude) to surface
    # OTHER steady states in multistable systems.
    scale = np.maximum(np.abs(y0_default), 1.0)
    # a species that starts at 0 can still be large at a steady state (all of a conserved
    # total phosphorylated, e.g.): starts spread over the model's overall scale too
    overall = max(1.0, float(np.max(np.abs(y0_default)))) if n else 1.0
    for _ in range(n_random):
        starts.append(y0_default + rng.normal(0, 1.0, n) * scale)
        starts.append(rng.uniform(0, 2.0, n) * scale)
        starts.append(rng.uniform(0, 1.0, n) * overall)

    if wide and n:
        # after the starts above (their order, and what they find, is unchanged): the size of
        # the largest non-zero parameter, its sign ignored, spread evenly and over every decade
        # from 0.01, non-negative (negative states still come from the starts around y0)
        mags = []
        for v in (model.get("param_defaults") or {}).values():
            try:
                v = abs(float(v))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                mags.append(v)
        big = max([overall] + mags)
        lo, hi = np.log10(min(0.01, big)), np.log10(big)
        for _ in range(n_random):
            starts.append(rng.uniform(0, 1.0, n) * big)
            starts.append(10.0 ** rng.uniform(lo, hi, n))

    return starts


def _states_across(param_name, param_min, param_max, n_values, fixed_params, model_args) -> str:
    """The steady states at values across one parameter's range: find_steady_states at each."""
    try:
        low, high, count = float(param_min), float(param_max), int(n_values)
    except (TypeError, ValueError):
        return json.dumps({"error": "param_name needs numbers param_min and param_max."})
    if not (np.isfinite(low) and np.isfinite(high) and low < high) or count < 2:
        return json.dumps({"error": "param_min must be below param_max, both finite, with n_values of 2 or more."})
    rows, began, one = [], time.time(), {}
    for value in np.linspace(low, high, count):
        if rows and time.time() - began > 3 * STEADY_SEARCH_SECONDS:
            break
        one = json.loads(find_steady_states(**model_args, fixed_params={**(fixed_params or {}),
                                                                        param_name: float(value)}))
        if "error" in one:
            return json.dumps(one)
        rows.append({"value": float(value), "steady_states": [
            {k: s[k] for k in ("state", "stability", "isolated") if k in s}
            for s in one["steady_states"] if "state" in s]})
    most = max(len(r["steady_states"]) for r in rows)
    return json.dumps({
        "model": one["model"], "scan_parameter": param_name, "parameters_used": one["parameters_used"],
        "most_states": most, "at_values": [r["value"] for r in rows if len(r["steady_states"]) == most],
        "states_across": rows,
        **({"search_note": f"the search stopped at {len(rows)} of {count} values (time limit)."}
           if len(rows) < count else {}),
        "note": ("Only at these values, from a sampled search: a state between them, or one the search "
                 "missed, is not listed. run_bifurcation_analysis follows a state a diagram did not "
                 "show, with its value in fixed_params and the state in initial_conditions.")})


def find_steady_states(model_name_or_path: str | None = None,
                        species: list[str] | None = None,
                        odes: dict | None = None,
                        parameters: dict | None = None,
                        initial_conditions: dict | None = None,
                        fixed_params: dict | None = None,
                        n_random_starts: int = 40,
                        seed: int = 0,
                        param_name: str | None = None,
                        param_min: float | None = None,
                        param_max: float | None = None,
                        n_values: int = 11) -> str:
    """
    Tool: find all distinct steady states of a model (SBML, any dimension, or
    directly-specified ODEs) and classify each one's stability from the
    Jacobian eigenvalues.

    fixed_params: override any parameter's default value for this analysis
      (does not modify the model itself).
    param_name, param_min, param_max: search at n_values values across that range instead.
    """
    if param_name is not None:
        return _states_across(param_name, param_min, param_max, n_values, fixed_params, {
            "model_name_or_path": model_name_or_path, "species": species, "odes": odes,
            "parameters": parameters, "initial_conditions": initial_conditions,
            "n_random_starts": n_random_starts, "seed": seed})
    from dynalyst.analysis.generic_sbml import DELAY, DelayModel
    delay_terms = []
    try:
        model = resolve_model(model_name_or_path, species, odes, parameters, initial_conditions)
    except DelayModel:
        # a delay differential equation: at a steady state x(t - tau) = x, so its steady states
        # are those of the equations with each delay(x, tau) read as x. Their stability needs
        # the delay terms (a characteristic equation with exp(-lambda tau)): not given here.
        model = resolve_model(model_name_or_path, species, odes, parameters, initial_conditions,
                              allow_delay=True)
        delay_terms = sorted({str(d) for e in model["rhs"] for d in sp.sympify(e).atoms(DELAY)})
        model = {**model, "rhs": [sp.sympify(e).replace(lambda x: getattr(x, "func", None) == DELAY,
                                                         lambda x: x.args[0])
                                  for e in model["rhs"]]}
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    param_values = dict(model["param_defaults"])
    if fixed_params:
        unknown = [k for k in fixed_params if k not in param_values]
        if unknown:
            return json.dumps({
                "error": f"Unknown parameter(s) {unknown}. Valid parameters: {list(param_values.keys())}"
            })
        from dynalyst.analysis.ode_model import relative_value      # '*2' / '/2' allowed
        param_values.update({k: relative_value(v, param_values[k]) for k, v in fixed_params.items()})

    try:
        rhs_fn, jac_fn = _lambdify_model(model)
        param_vec = [param_values[p] for p in model["param_names"]]
        f, jac = _make_numeric_funcs(model, param_values)
    except Exception as e:
        return json.dumps({"error": f"Could not evaluate model numerically: {e}"})

    n = len(model["state_names"])
    y0_default = np.array([model["y0"][s] for s in model["state_names"]], dtype=float)
    rng = np.random.default_rng(seed)

    # Detect structural conservation laws (e.g. total protein amount fixed
    # across X/Xp/Xpp) -- without them, the raw RHS=0 system is
    # rank-deficient and has a whole MANIFOLD of "steady states", not
    # isolated points, and random starts just land on arbitrary points of
    # that manifold rather than the one manifold slice set by this model's
    # actual initial conditions.
    conserved_directions = [np.array([float(x) for x in c])
                            for c in exact_conservation_laws({**model, "param_defaults": param_values})]
    conserved_directions = np.array(conserved_directions) if conserved_directions else np.zeros((0, n))
    conserved_targets = conserved_directions @ y0_default if len(conserved_directions) else np.array([])

    def f_constrained(y):
        base = f(y)
        if len(conserved_directions):
            constraint_residual = conserved_directions @ y - conserved_targets
            return np.concatenate([base, constraint_residual])
        return base

    starts = _candidate_starts({**model, "param_defaults": param_values}, f, n_random_starts, rng,
                               wide=True)

    scale_fn = _term_scale_fn(model)
    found = []
    # the exact Jacobian (finite differences cost n evaluations per step and stall on
    # stiff models) and a time budget for the whole search: it is a sample of starts anyway
    jac_constrained = (lambda y: np.vstack([jac(y), conserved_directions]))
    t_search, tried = time.time(), 0
    for y_guess in starts:
        if time.time() - t_search > STEADY_SEARCH_SECONDS:
            break
        tried += 1
        try:
            if len(conserved_directions):
                sol_obj = least_squares(f_constrained, y_guess, jac=jac_constrained,
                                        xtol=1e-13, ftol=1e-13, max_nfev=400)
                sol, success = sol_obj.x, sol_obj.success
            else:
                sol, info, ier, _ = fsolve(f, y_guess, fprime=jac, full_output=True, xtol=1e-11)
                success = (ier == 1)
        except Exception:
            continue
        if not success:
            continue
        # rates balanced relative to the rate terms they balance (independent of units),
        # and the conserved totals held relative to their size
        with np.errstate(all="ignore"):
            r = np.abs(f(sol))
            scale = np.abs(np.array(scale_fn(*sol, *param_vec), dtype=float))
        if not (np.all(np.isfinite(r)) and np.all(r <= 1e-6 * flux_scale(scale))):
            continue
        if len(conserved_directions) and np.any(
                np.abs(conserved_directions @ sol - conserved_targets)
                > 1e-8 * (np.abs(conserved_directions) @ np.abs(sol) + 1e-300)):
            continue
        found.append(sol)

    unique_states = _dedupe(found)

    # A species the model keeps >= 0 (an amount or concentration) cannot be negative, and no
    # one-way rate can run backwards: such a solution is not a state the system can be in.
    # (Typed ODEs may have variables of either sign.)
    from dynalyst.analysis.typed_equations import is_typed_file
    nonneg = (str(model.get("source", "")).startswith("sbml")
              and not is_typed_file(model.get("source_path") or ""))
    kept, backwards = set(), (lambda y: [])
    if nonneg:
        kept = kept_nonnegative(model, param_values)
        n_sims = len(starts) - 3 * n_random_starts      # y0 and the warm-up simulation ends
        backwards = one_way_rate_check(model, param_values, kept, starts[:n_sims])
    results, non_physical = [], []
    for y_ss in unique_states:
        try:
            J = np.array(jac(y_ss), dtype=float)
            eig = np.linalg.eigvals(J)
        except Exception as e:
            results.append({"error": f"Jacobian evaluation failed: {e}"})
            continue
        entry = {
            "state": {name: float(v) for name, v in zip(model["state_names"], y_ss)},
            "eigenvalues": [{"real": float(e.real), "imag": float(e.imag)} for e in eig],
            "stability": classify_stability(eig, len(conserved_directions)),
            "jacobian": J.tolist(),
        }
        # more zero eigenvalues than conservation laws: the state is not isolated, it lies
        # on a curve (or surface) of steady states, so the points found are a sample of it
        n_zero = int(np.sum(np.abs(eig) <= 1e-7 * max(1.0, float(np.max(np.abs(eig))))))
        if n_zero > len(conserved_directions):
            entry["isolated"] = False
            entry["stability"] += (" -- NOT isolated: part of a continuum of steady states; "
                                   "the stability refers to directions off the continuum")
        negative = [name for name, v in zip(model["state_names"], y_ss)
                    if name in kept and v < -1e-9 * max(1.0, float(np.max(np.abs(y_ss))))]
        reversed_rates = backwards(y_ss)
        pole = nonneg and across_a_pole(model, y_ss, param_values)
        if negative or reversed_rates or pole:
            non_physical.append({**entry, "negative": negative,
                                 **({"negative_rates": reversed_rates} if reversed_rates else {}),
                                 **({"across_a_pole": True} if pole else {})})
        else:
            results.append(entry)

    continuum = None
    on_continuum = [e for e in results if e.get("isolated") is False]
    if len(on_continuum) == 1:
        # one such point only: extra zero eigenvalues alone (e.g. at a state where most species
        # are 0, whose Jacobian rows vanish) are no evidence of a line of steady states -- a
        # real one is met at many points by the multi-start search. Kept as a steady state.
        e = on_continuum[0]
        e.pop("isolated")
        e["stability"] = (e["stability"].split(" -- NOT isolated")[0]
                          + " -- degenerate: more zero eigenvalues than conservation laws, so "
                          "these eigenvalues alone do not decide its stability")
        on_continuum = []
    if on_continuum:
        def is_root(y):
            with np.errstate(all="ignore"):
                r = np.abs(f(y))
                sc = np.abs(np.array(scale_fn(*y, *param_vec), dtype=float))
            return bool(np.all(np.isfinite(r)) and np.all(r <= 1e-6 * flux_scale(sc)))
        param_subs = {s: param_values[p] for s, p in zip(model["param_symbols"], model["param_names"])}
        symbolic_rhs = [sp.sympify(e).subs(param_subs) for e in model["rhs"]]
        continuum, shown = _summarise_continuum(on_continuum, model["state_names"], f, jac, is_root,
                                               symbolic_rhs)
        results = [e for e in results if e.get("isolated") is not False] + shown

    if delay_terms:     # the ODE Jacobian's eigenvalues do not decide a delay equation's stability
        for e in results + non_physical:
            for k in ("eigenvalues", "jacobian", "isolated"):
                e.pop(k, None)
            e["stability"] = "not determined (delay differential equation)"

    spiralling = [i for i, e in enumerate(results, start=1) if "eigenvalues" in e and spirals_outward(
        [complex(v["real"], v["imag"]) for v in e["eigenvalues"]])]
    oscillation_note = (
        f"Steady state(s) {spiralling} are unstable with complex eigenvalues of positive real part: "
        "nearby trajectories spiral away from them. This does not establish a limit cycle, "
        "even if trajectories remain bounded. Simulation can examine transients and candidate "
        "oscillations; periodic-orbit continuation and stability checks "
        "are needed to characterize a candidate cycle."
        if spiralling else None)

    conservation_note = None
    if len(conserved_directions):
        readable = [law_text(c, model["state_names"]) for c in exact_conservation_laws(model)]
        conservation_note = (
            f"Detected {len(conserved_directions)} conserved quantity/quantities "
            f"({'; '.join(readable)}) -- steady states were solved constrained to "
            f"this model's own initial-condition manifold (the value these "
            f"quantities have at the default y0), since without that constraint "
            f"the raw equations have a whole continuum of solutions rather than "
            f"isolated steady states."
        )

    return json.dumps({
        "model": model["label"],
        "source": model["source"],
        "parameters_used": param_values,
        "odes": ode_strings(model),
        "n_starts_tried": tried,
        **({"search_note": f"the search stopped after {tried} of {len(starts)} starts (time limit {STEADY_SEARCH_SECONDS:g} s): steady states the other starts would reach may be missing"} if tried < len(starts) else {}),
        "n_distinct_steady_states": len([e for e in results if e.get("isolated") is not False]),
        **({"delay_differential_equation": (
            f"The model is a delay differential equation ({', '.join(delay_terms)}): at a "
            "steady state the delayed value equals the current one, so these are its steady "
            "states. Their stability is NOT given: it depends on the delays (the roots of a "
            "characteristic equation with exp(-lambda*tau) terms), not on the ODE Jacobian. A "
            "simulation (simulate_model, which integrates the delays) shows whether the system "
            "settles there or oscillates around it.")} if delay_terms else {}),
        "conservation_laws_detected": conservation_note,
        **({"oscillation_note": oscillation_note} if oscillation_note else {}),
        **({"continuum": continuum,
            "continuum_note": (
            f"{continuum['points_found']} of the states found are not isolated: they lie on a continuum of steady states (the Jacobian has more zero eigenvalues than there are conservation laws). Any point of it is a steady state, so these points are a sample, not a count of distinct states ({sum(1 for e in results if e.get('isolated') is False)} of them are listed; 'continuum' summarises it); where the system ends on it depends on where it starts.")}
           if continuum else {}),
        "steady_states": results,
        "jacobian_order": ("rows: d(d<state>/dt), columns: d<state>, both in the order "
                           f"{model['state_names']}"),
        **({"non_physical_solutions": non_physical,
            "non_physical_note": (
                f"{len(non_physical)} further solution(s) of the equations have a negative "
                "value of a species the model keeps >= 0, make a one-way rate negative "
                "('negative_rates', e.g. a fraction above 1), or lie across a pole of a rate law "
                "from the model's starting state ('across_a_pole': a denominator changes sign, "
                "e.g. an active form above its total), so they are not counted above: "
                + "; ".join(
                    "(" + ", ".join(f"{k} = {v:.6g}" for k, v in e["state"].items()) + ") "
                    + str(e["stability"]).split(" (")[0] for e in non_physical) + ".")}
           if non_physical else {}),
        "interpretation": (
            ("The states listed lie on ONE continuum of steady states (see 'continuum'): "
             "different stability along it is NOT bistability, which needs two separate stable "
             "states; where the system ends on the continuum depends on where it starts. "
             if continuum else "")
            + ("One steady state at these parameter values: no bistability here (that needs two "
             "stable states at the same parameter values); it can still appear at other values. "
             if len([r for r in results if "error" not in r]) == 1 else "")
            + "Multiple distinct steady states with different stability = the "
            "hallmark of bistability/multistability. A 'saddle' steady state "
            "sitting between a stable and another stable state is the "
            "separatrix between their basins of attraction. Use "
            "run_bifurcation_analysis to see how these steady states (and "
            "their stability) change as a parameter varies -- that is where "
            "saddle-node/fold, transcritical/pitchfork (branch points), and "
            "Hopf bifurcations actually get identified."
        ),
    }, indent=2, default=float)


TOOL_DEFINITIONS = [
    {
        "name": "find_steady_states",
        "description": (
            "Find ALL distinct steady states of a model (any dimension -- SBML file "
            "or a directly-specified ODE system) via multi-start numerical "
            "root-finding, and classify the LOCAL STABILITY of each one from the "
            "eigenvalues of the Jacobian (stable node/focus, unstable node/focus, "
            "or saddle). Always run this BEFORE a bifurcation analysis, since AUTO-07p "
            "continuation needs a starting steady state, and bistable/multistable "
            "models have more than one to choose from. With param_name, param_min and "
            "param_max: the states at values across that range, to see where several coexist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name_or_path": {"type": "string", "description": "SBML model name or path."},
                "species": {"type": "array", "items": {"type": "string"}, "description": "For a directly-specified ODE system."},
                "odes": {"type": "object", "description": "{species_name: rhs_expression_string} for a directly-specified system."},
                "parameters": {"type": "object", "description": "{parameter_name: default_value} for a directly-specified system."},
                "initial_conditions": {"type": "object", "description": "{species_name: value} for a directly-specified system."},
                "fixed_params": {
                    "type": "object",
                    "description": "Optional override of any parameter's value for THIS analysis (e.g. to see steady states at a specific parameter setting).",
                },
                "n_random_starts": {"type": "integer", "description": "Number of randomized initial guesses to try, in addition to defaults/simulation endpoints. Default 40."},
                "param_name": {"type": "string", "description": "Optional. A parameter to search across, with param_min and param_max."},
                "param_min": {"type": "number", "description": "Lower end of the range of param_name."},
                "param_max": {"type": "number", "description": "Upper end of the range of param_name."},
                "n_values": {"type": "integer", "description": "How many values across the range. Default 11."},
            },
            "required": [],
        },
    },
]

TOOL_FUNCTIONS = {
    "find_steady_states": lambda **kwargs: find_steady_states(**kwargs),
}
