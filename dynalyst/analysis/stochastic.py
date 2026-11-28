"""Stochastic simulation and explicit comparisons using one canonical model.

stochastic_simulation runs only the requested stochastic methods (default SSA).
compare_simulations explicitly coordinates a deterministic reference and SSA/CLE.
JAX accelerates our Gillespie engine; JaxKineticModel does not implement SSA.
ODE-derived propensities are modelling assumptions, not a unique microscopic model.
"""

import json
from copy import deepcopy
import math
import re
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import sympy as sp

from dynalyst import config
from dynalyst.analysis.simulation_model import selected_methods as _selected_methods

MAX_EVENTS_JAX = int(getattr(config, "SSA_MAX_EVENTS_PER_RUN_JAX", 500_000_000))
# a run stopped early has NaN after it stops, and its status already says "incomplete"
warnings.filterwarnings("ignore", message="Mean of empty slice|Degrees of freedom", category=RuntimeWarning)
SSA_MAX_SECONDS = float(getattr(config, "SSA_MAX_SECONDS", 300))
SSA_MAX_SECONDS_FORCED = float(getattr(config, "SSA_MAX_SECONDS_FORCED", 3 * 3600))
# a computational choice for a run that can finish, not a biological population size
SSA_REFERENCE_ODE_SECONDS = float(getattr(config, "SSA_REFERENCE_ODE_SECONDS", 30))
SSA_STOP_FACTOR = float(getattr(config, "SSA_STOP_FACTOR", 3))
_TOL = 1e-9


class NotDecomposable(ValueError):
    """The ODEs cannot be written as reaction channels with non-negative rates."""


# --- model ----------------------------------------------------------------------

def _load(model_name_or_path=None, species=None, odes=None, parameters=None,
          initial_conditions=None):
    from dynalyst.analysis.simulation_model import load_model
    model, values, changed = load_model(model_name_or_path, species, odes, parameters,
                                         initial_conditions)
    if model.get("source") == "xpp":
        model = {**model, "canonical_rhs": model["rhs"]}
        model = _xpp_as_written(model, model_name_or_path)
    return model, values, changed


def _xpp_as_written(model, path_or_name):
    """The XPP parser simplifies each right-hand side into one fraction, which
    hides the separate reaction rates. Re-read the derivative lines as written
    (named algebraic expressions substituted, nothing simplified) and keep
    them only if they equal the parsed ODEs."""
    from dynalyst.analysis import xpp_to_sbml as xpp
    try:
        text = xpp._resolve_ode_path(path_or_name).read_text(errors="replace")
        raw, alg = {}, {}
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            low = line.lower()
            if (not line or line.startswith("@") or low == "done" or xpp._INIT_RE.match(line)
                    or low.startswith(("par ", "p "))):
                continue
            if low.startswith("aux "):
                line = line.split(None, 1)[1].strip()
            m = xpp._DERIV_RE.match(line)
            if m:
                raw[m.group(1)] = m.group(2)
                continue
            m = xpp._ASSIGN_RE.match(line)
            if m:
                alg[m.group(1)] = m.group(2)
        if set(raw) != set(model["state_names"]):
            return model
        known = set(raw) | set(model["param_defaults"]) | set(alg)
        alg_expr = {sp.Symbol(k): xpp._sympify_xpp(v, known) for k, v in alg.items()}
        rhs = []
        for name in model["state_names"]:
            e = xpp._sympify_xpp(raw[name], known)
            for _ in range(10):
                if not (e.free_symbols & set(alg_expr)):
                    break
                e = e.xreplace(alg_expr)
            rhs.append(e)
        # same equations? (numerically, at random positive points)
        syms = model["state_symbols"] + model["param_symbols"]
        f_new = sp.lambdify(syms, rhs, "numpy")
        f_old = sp.lambdify(syms, model["rhs"], "numpy")
        rng = np.random.default_rng(1)
        for _ in range(8):
            p = rng.uniform(0.1, 2.0, len(syms))
            a = np.array(f_new(*p), dtype=float)
            b = np.array(f_old(*p), dtype=float)
            if not np.allclose(a, b, rtol=1e-8, atol=1e-10):
                return model
        return {**model, "rhs": rhs}
    except Exception:
        return model


def _param_subs(model, values):
    return {sym: values[name] for name, sym in zip(model["param_names"], model["param_symbols"])}


# --- 1. reaction channels from the ODEs -----------------------------------------------

def _sign_on(expr, states, points, subs):
    """'pos' | 'neg' | 'zero' | 'mixed' of expr over the sample points."""
    f = sp.lambdify(states, expr.subs(subs), "numpy")
    with np.errstate(all="ignore"):
        vals = np.array([float(np.real(f(*p))) for p in points])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "mixed"
    scale = 1.0 + float(np.max(np.abs(vals)))
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= _TOL * scale and lo >= -_TOL * scale:
        return "zero"
    if lo >= -_TOL * scale:
        return "pos"
    if hi <= _TOL * scale:
        return "neg"
    return "mixed"


def _signed_terms(rhs, states, points, subs, depth=0):
    """[(coefficient, non-negative magnitude)] whose sum is rhs."""
    out = []
    for term in (rhs if isinstance(rhs, list) else sp.Add.make_args(rhs)):
        coeff, mag = term.as_coeff_Mul()
        coeff = float(coeff)
        sign = _sign_on(mag, states, points["path"], subs)
        if sign == "zero":          # zero along the path: judge it on the probe states
            sign = _sign_on(mag, states, points["probe"], subs)
        if sign == "zero":
            continue
        if sign == "pos":
            out.append((coeff, mag))
        elif sign == "neg":
            out.append((-coeff, sp.expand(-mag) if mag.is_Add else -mag))
        else:
            expanded = sp.expand(term)
            if depth == 0 and len(sp.Add.make_args(expanded)) > 1:
                out += _signed_terms(expanded, states, points, subs, depth + 1)
            else:
                raise NotDecomposable(
                    f"The term {sp.sstr(term)} changes sign inside the region the system "
                    "visits, so it cannot be the rate of a reaction (a propensity must be "
                    "non-negative). Write the model so every term is a non-negative rate "
                    "times a +/- sign.")
    return out


def sample_points(model, values, t_end, extra=None):
    """Static validation probes; never integrate an ODE to construct propensities."""
    y0 = np.asarray([model["y0"][n] for n in model["state_names"]], dtype=float)
    path = list(np.asarray(extra, dtype=float)) if extra is not None else [y0]
    path = [row for row in path if np.all(np.isfinite(row))]
    top = np.maximum(np.abs(y0), 1.0)
    rng = np.random.default_rng(0)
    probe = [np.zeros_like(y0), *[rng.uniform(0.0, 1.0, len(y0)) * top for _ in range(40)]]
    return {"path": path or probe, "probe": probe}


def _all_points(points):
    return list(points["path"]) + list(points["probe"])


def _channel(stoich, rate, label=None):
    stoich = np.asarray(stoich, dtype=float)
    integer = bool(np.all(np.abs(stoich - np.round(stoich)) < 1e-9))
    return {"stoich": np.round(stoich) if integer else stoich, "rate": rate,
            "integer": integer, "label": label}


def _sbml_channels(model, values, points):
    """One channel per SBML reaction (two for a reversible rate law that
    changes sign), rates taken from the kinetic laws. None when the model has
    rate rules or no reactions (then the ODE terms are used)."""
    disp = model.get("display") or {}
    reactions = disp.get("reactions") or []
    if model.get("source") != "sbml" or not reactions or disp.get("rate_rules"):
        return None
    states = model["state_symbols"]
    names = model["state_names"]
    known = set(model["param_names"])
    consts = {sp.Symbol(k): v["value"] for k, v in (disp.get("constants") or {}).items()
              if k not in known and v.get("value") is not None}
    rules = {sp.Symbol(k): e for k, e in (disp.get("assignment_rules") or {}).items()
             if k not in known}
    subs = _param_subs(model, values)
    channels = []
    for r in reactions:
        nu = np.zeros(len(names))
        for s_, n in r["reactants"]:
            if s_ in names:
                nu[names.index(s_)] -= float(n)
        for s_, n in r["products"]:
            if s_ in names:
                nu[names.index(s_)] += float(n)
        if not np.any(np.abs(nu) > 0):
            continue
        rate = sp.sympify(r["rate"])
        for _ in range(20):
            if not (rate.free_symbols & set(rules)):
                break
            rate = rate.xreplace(rules)
        rate = rate.xreplace(consts)
        # Kinetic laws are amounts/time; concentration-valued species need /volume.
        involved = [model.get("state_metadata", {}).get(n, {}) for n, s in zip(names, nu) if s]
        volumes = {float(model.get("compartments", {}).get(s.get("compartment"), 1.0))
                   if not s.get("amount_units") else 1.0 for s in involved}
        if len(volumes) > 1:
            raise NotDecomposable("Reaction crosses differently scaled species; per-compartment count scaling is required.")
        if volumes:
            rate /= sp.Rational(str(next(iter(volumes))))
        # a reversible rate law is forward minus backward: two channels, even where the
        # net flux keeps one sign (each direction fires on its own)
        terms = list(sp.Add.make_args(sp.expand(rate))) if r.get("reversible") else rate
        for c, mag in _signed_terms(terms, states, points, subs):
            label = r.get("name") or r["id"]
            channels.append(_channel(nu * np.sign(c),
                                     sp.Float(abs(c)) * mag if abs(abs(c) - 1) > 1e-12 else mag,
                                     label if c > 0 else f"{label} (reverse)"))
    merged = {}                     # pieces of one direction of one reaction -> one channel
    for ch in channels:
        merged.setdefault((ch["label"], tuple(ch["stoich"])), []).append(ch)
    return [group[0] if len(group) == 1 else
            _channel(group[0]["stoich"], _tidy_sum([g["rate"] for g in group]), group[0]["label"])
            for group in merged.values()]


def reaction_channels(model, values, points) -> list[dict]:
    """[{stoich: np.array, rate: sympy expr in concentrations, integer: bool}].

    SBML: one channel per reaction (kinetic law). Otherwise from the ODE
    terms; then channels with the same stoichiometry are summed into one --
    exact for both the SSA (sum of independent Poisson channels) and the CLE
    (same noise in distribution), and it undoes any algebraic splitting."""
    if model.get("source") == "sbml":
        try:
            sb = _sbml_channels(model, values, points)
            if sb:
                _check_reconstruction(model, sb, values, points)
                for channel in sb:
                    channel["source"] = "SBML reaction"
                return _ordered(sb)
        except (ValueError, KeyError, TypeError) as exc:
            raise NotDecomposable(f"SBML reaction channels could not be validated: {exc}") from exc
    states = model["state_symbols"]
    subs = _param_subs(model, values)
    found = {}                      # magnitude -> coefficient per species
    for i, terms in enumerate(_split_shared(model["rhs"])):
        for coeff, mag in _signed_terms(terms, states, points, subs):
            found.setdefault(mag, np.zeros(len(states)))[i] += coeff
    by_stoich = {}
    for mag, coeffs in found.items():
        nz = np.abs(coeffs) > _TOL
        if not nz.any():
            continue                # cancelled out
        base = float(np.min(np.abs(coeffs[nz])))
        stoich = coeffs / base
        rate = sp.Float(base) * mag if abs(base - 1) > 1e-12 else mag
        key = tuple(np.round(stoich, 9))
        by_stoich.setdefault(key, []).append(rate)
    channels = []
    for key, rates in by_stoich.items():
        rate = rates[0] if len(rates) == 1 else _tidy_sum(rates)
        channels.append(_channel(np.array(key), rate))
    for channel in channels:
        channel["source"] = "inferred from ODE terms"
    return _ordered(channels)


def _split_shared(rhs_list):
    """Each equation's terms. A term is expanded into its pieces only when one
    of the pieces also occurs in ANOTHER equation (a shared reaction rate that
    simplification folded into a fraction, e.g. (Km*vs + S*vs - S*vm)/(Km + S)).
    Otherwise it stays whole: a2*(1 - x)*h is one activation, not a production
    plus a separate removal (that would add noise that is not there)."""
    def mag(p):
        return p.as_coeff_Mul()[1]
    eqs = [list(sp.Add.make_args(sp.sympify(r))) for r in rhs_list]
    pieces = [[list(sp.Add.make_args(sp.expand(t))) for t in terms] for terms in eqs]
    where, whole = {}, {}
    for i, (terms, per_term) in enumerate(zip(eqs, pieces)):
        for t in terms:
            whole.setdefault(mag(t), set()).add(i)
        for ps in per_term:
            for p in ps:
                where.setdefault(mag(p), set()).add(i)
    out = []
    for i, (terms, per_term) in enumerate(zip(eqs, pieces)):
        row = []
        for t, ps in zip(terms, per_term):
            if whole[mag(t)] - {i}:
                row.append(t)            # the whole term is a rate shared with another equation
            elif len(ps) > 1 and any(where[mag(p)] - {i} for p in ps):
                row.extend(ps)
            else:
                row.append(t)
        out.append(row)
    return out


def _tidy_sum(rates):
    total = sp.Add(*rates)
    try:
        nice = sp.factor(sp.cancel(sp.together(total)))
        return nice if sp.count_ops(nice) <= sp.count_ops(total) else total
    except Exception:
        return total


def _ordered(channels):
    # sources first, then conversions, then sinks: reads like the guide's table
    channels = list(channels)
    channels.sort(key=lambda c: (int(np.any(c["stoich"] < 0)) + int(not np.any(c["stoich"] > 0)),
                                 int(np.argmax(np.abs(c["stoich"])))))
    return channels


def _check_reconstruction(model, channels, values, points):
    """V @ rates must give back the ODEs (catches any decomposition slip)."""
    states = model["state_symbols"]
    subs = _param_subs(model, values)
    rhs_f = sp.lambdify(states, [sp.sympify(e).subs(subs) for e in model["rhs"]], "numpy")
    rate_f = sp.lambdify(states, [c["rate"].subs(subs) for c in channels], "numpy")
    V = np.array([c["stoich"] for c in channels]).T
    worst = 0.0
    for p in _all_points(points)[:60]:
        a = np.array([float(np.broadcast_to(v, ())) for v in rate_f(*p)])
        f = np.array([float(np.broadcast_to(v, ())) for v in rhs_f(*p)])
        # measured against the size of the fluxes, not of the net rate: large opposite fluxes
        # (e.g. transport scaled by a volume ratio of 1e12) cancel to a small net rate, whose
        # rounding error is not a decomposition error
        flux = np.abs(V) @ np.abs(a)
        worst = max(worst, float(np.max(np.abs(V @ a - f) / (1 + flux + np.abs(f)))))
    if worst > 1e-6:
        raise NotDecomposable(f"Reaction channels do not reproduce the ODEs (error {worst:.2e}).")


def _omega_symbol():
    return sp.Symbol("Omega", positive=True)


def _repeated_reactants(model, rate, stoich):
    """{species: count} when the rate is a plain product of powers and each species raised to a
    power >= 2 is consumed by exactly that many molecules (k*A**2 with A falling by 2), so the
    channel is an elementary reaction. None for anything else: a Hill or Michaelis-Menten law, or
    a power law that is not that reaction, has no exact combinatorial form and keeps its own."""
    states = {s: i for i, s in enumerate(model["state_symbols"])}
    powers = {}
    for factor in sp.Mul.make_args(rate):
        base, exp = factor.as_base_exp()
        if base in states:
            if not (exp.is_Integer and exp > 0):
                return None
            powers[base] = int(exp)
        elif factor.free_symbols & set(states):
            return None
    repeated = {s: n for s, n in powers.items() if n >= 2}
    if not repeated or any(float(stoich[states[s]]) != -n for s, n in repeated.items()):
        return None
    return repeated


def _approximate_channels(model, channels) -> list:
    """The channels whose propensity is only Omega * rate(X/Omega): anything but an elementary
    reaction (a Hill or Michaelis-Menten law, a power law that is not that reaction)."""
    states = set(model["state_symbols"])
    out = []
    for k, ch in enumerate(channels, 1):
        factors = [f.as_base_exp() for f in sp.Mul.make_args(ch["rate"]) if f.free_symbols & states]
        simple = all(b in states and e.is_Integer and e > 0 for b, e in factors)
        repeated = any(e != 1 for _, e in factors)
        if not simple or (repeated and not _repeated_reactants(model, ch["rate"], ch["stoich"])):
            out.append(f"w{k}: {sp.sstr(ch['rate'])}")
    return out


APPROXIMATE_NOTE = ("These rate laws are not elementary reactions, so they have no exact combinatorial "
                    "propensity. They use Omega * rate(X/Omega), which is an approximation at low "
                    "molecule counts.")


def molecular_propensity(model, rate, stoich=None):
    """w(X) = Omega * f(X/Omega), as a sympy expression in the species names. An elementary
    reaction with a repeated reactant counts distinct molecules: k*A**2 with A falling by 2 gives
    k*A*(A - 1)/Omega, which has no event at one molecule and the same deterministic limit."""
    om = _omega_symbol()
    repeated = None if stoich is None else _repeated_reactants(model, rate, stoich)
    if repeated:
        others = {s: s / om for s in model["state_symbols"] if s not in repeated}
        return om * rate.xreplace(others).xreplace(
            {s ** n: sp.prod([s - j for j in range(n)]) / om ** n for s, n in repeated.items()})
    sub = {s: s / om for s in model["state_symbols"]}
    w = om * rate.subs(sub)
    options = [w]
    for make in (lambda e: sp.cancel(e), lambda e: sp.factor(sp.cancel(e)),
                 lambda e: sp.powsimp(sp.factor(sp.cancel(e)), force=True)):
        try:
            options.append(make(w))
        except Exception:
            pass
    options += [_positive_factors(e) for e in options]
    # the guide's form: no Omega inside a denominator sum, no leading minus
    return min(options, key=lambda e: ("/Omega" in sp.sstr(e), sp.sstr(e).startswith("-"),
                                       sp.count_ops(e), len(sp.sstr(e))))


def _positive_factors(e):
    """-c*(A - Omega) -> c*(Omega - A)."""
    try:
        if not (isinstance(e, sp.Mul) and e.could_extract_minus_sign()):
            return e
        for f in e.args:
            if isinstance(f, sp.Add):
                flipped = (-(e / f)) * (-f)
                if not sp.sstr(flipped).startswith("-"):
                    return flipped
    except Exception:
        pass
    return e


def _reaction_text(model, ch, values):
    names = model["state_names"]
    st = ch["stoich"]
    left = " + ".join(_coef(-st[i]) + names[i] for i in range(len(names)) if st[i] < 0)
    right = " + ".join(_coef(st[i]) + names[i] for i in range(len(names)) if st[i] > 0)
    used = {str(s) for s in ch["rate"].free_symbols}
    changed = {names[i] for i in range(len(names)) if abs(st[i]) > 0}
    mods = [n for n in names if n in used and n not in changed]
    text = f"{left or '∅'} -> {right or '∅'}"
    if ch.get("label"):
        text = f"{ch['label']}: {text}"
    return text + (f"  (rate depends on {', '.join(mods)})" if mods else "")


def _coef(x):
    x = float(x)
    if abs(x - 1) < 1e-12:
        return ""
    return (f"{int(round(x))} " if abs(x - round(x)) < 1e-12 else f"{x:g} ")


def propensity_rows(model, channels, values):
    params_in = lambda e: {str(s): values[str(s)] for s in e.free_symbols if str(s) in values}
    rows = []
    for k, ch in enumerate(channels, 1):
        rows.append({
            "no": k,
            "reaction": _reaction_text(model, ch, values),
            "stoichiometry": {n: (int(v) if ch["integer"] else float(v))
                              for n, v in zip(model["state_names"], ch["stoich"]) if abs(v) > 0},
            "rate_in_ode": sp.sstr(ch["rate"]),
            "propensity": f"w{k} = " + sp.sstr(molecular_propensity(model, ch["rate"], ch.get("stoich"))),
            "parameter_values": params_in(ch["rate"]),
        })
    return rows


def _markdown_table(rows):
    lines = ["| No. | Reaction | Propensity | Parameter values |",
             "|---|---|---|---|"]
    for r in rows:
        pv = ", ".join(f"{k} = {v:g}" for k, v in r["parameter_values"].items()) or "-"
        lines.append(f"| {r['no']} | {r['reaction']} | {r['propensity']} | {pv} |")
    return "\n".join(lines)


# --- numerics ----------------------------------------------------------------------

def _reference_ode(model, values, grid):
    """The deterministic trajectory behind an SSA call, for Omega and the period comparison only:
    not a requested method, not plotted. (trajectory or None, what happened)."""
    from dynalyst.analysis import hybrid_sim
    from dynalyst.analysis.deterministic import integrate
    if model.get("events") or hybrid_sim.has_delay(model):
        return None, {"status": "not run", "reason": "the model has events or delays"}
    began = time.time()
    try:
        ode = integrate(model, values, grid, max_seconds=SSA_REFERENCE_ODE_SECONDS)[0]
    except TimeoutError as e:
        return None, {"status": "timed out", "reason": str(e), "seconds": round(time.time() - began, 2)}
    except Exception as e:      # a reference must never cost the stochastic run
        return None, {"status": "failed", "reason": f"{type(e).__name__}: {e}"[:200]}
    if not np.all(np.isfinite(ode)):
        return None, {"status": "failed", "reason": "the trajectory is not finite"}
    return ode, {"status": "succeeded", "seconds": round(time.time() - began, 2)}


def _ode(model, values, t_end, grid):
    """Explicit deterministic reference, delegated to the deterministic service."""
    from dynalyst.analysis.deterministic import integrate
    return integrate(model, values, grid)[0]


def _propensity_fn(model, channels, values, omega):
    """a(X) for a state vector X (molecules); also works on (n, runs) arrays."""
    X = sp.symbols(f"X0:{len(model['state_names'])}")
    subs = _param_subs(model, values)
    om = _omega_symbol()
    exprs = []
    for ch in channels:
        w = molecular_propensity(model, ch["rate"], ch.get("stoich")).subs(subs).subs(om, omega)
        exprs.append(w.subs(dict(zip(model["state_symbols"], X))))
    try:
        f = sp.lambdify([X], exprs, "numpy", cse=True)
    except TypeError:
        f = sp.lambdify([X], exprs, "numpy")

    def a(x):
        with np.errstate(all="ignore"):
            out = f(x)
        shape = np.shape(x)[1:]          # () for one state, (runs,) for many
        if not shape:                    # one state (the SSA loop): the fast path
            return np.array(out, dtype=float)
        res = np.empty((len(out),) + shape)
        for i, v in enumerate(out):      # a constant channel broadcasts on assignment
            res[i] = v
        return res
    return a


def ssa_run(a_fn, V, x0, grid, rng, max_events):
    """Gillespie's direct method; the state is recorded on `grid` (piecewise
    constant between events). Returns (states, events, clipped, truncated)."""
    n_t = len(grid)
    out = np.empty((n_t, len(x0)))
    x = np.array(x0, dtype=float)
    t, k, events, clipped = 0.0, 0, 0, 0
    t_end = grid[-1]
    while True:
        a = a_fn(x)
        if not (a >= 0).all():              # negative or NaN rates
            clipped += int(np.any(a < 0))
            a = np.where(a >= 0, a, 0.0)
        cum = np.cumsum(a)
        a0 = float(cum[-1])
        t_next = t + rng.exponential(1.0 / a0) if a0 > 0 else math.inf
        while k < n_t and grid[k] < t_next:
            out[k] = x
            k += 1
        if k == n_t or t_next > t_end:
            out[k:] = x
            return out, events, clipped, False
        j = int(np.searchsorted(cum, rng.random() * a0, side="right"))
        x = x + V[:, min(j, V.shape[1] - 1)]
        t = t_next
        events += 1
        if events >= max_events:
            out[k:] = np.nan
            return out, events, clipped, True


def cle_runs(a_fn, V, x0, grid, rng, n_runs, dt):
    """Euler-Maruyama for the chemical Langevin equation, all runs at once.
    Propensities are clipped at 0 and states at 0 (counted)."""
    n_t = len(grid)
    gap = float(grid[1] - grid[0])
    sub = max(1, int(math.ceil(gap / dt)))
    h = gap / sub
    X = np.repeat(np.asarray(x0, dtype=float)[:, None], n_runs, axis=1)   # (n, runs)
    out = np.empty((n_runs, n_t, X.shape[0]))
    out[:, 0, :] = X.T
    clipped_a = clipped_x = 0
    for k in range(1, n_t):
        for _ in range(sub):
            a = a_fn(X)                                   # (m, runs)
            if not np.all(np.isfinite(a)):
                raise ValueError("Non-finite CLE propensity; simulation failed.")
            bad = ~(a >= 0)
            if bad.any():
                clipped_a += int(bad.sum())
                a = np.where(bad, 0.0, a)
            noise = np.sqrt(a) * rng.standard_normal(a.shape)
            X = X + h * (V @ a) + math.sqrt(h) * (V @ noise)
            neg = X < 0
            if neg.any():
                clipped_x += int(neg.sum())
                X = np.where(neg, 0.0, X)
        out[:, k, :] = X.T
    return out, {"dt": h, "clipped_propensities": clipped_a, "clipped_states": clipped_x}


def _peaks(series, t):
    """Mean period from peaks of at least 20 % of the series' range, or None."""
    from scipy.signal import find_peaks
    s = np.asarray(series, dtype=float)
    if not np.all(np.isfinite(s)) or np.ptp(s) <= 0:
        return None
    idx, _ = find_peaks(s, prominence=0.2 * np.ptp(s))
    if len(idx) < 3:
        return None
    return {"n_peaks": int(len(idx)), "mean_period": float(np.mean(np.diff(t[idx])))}


def _two_modes(values):
    """Sarle's coefficient > 0.555 AND a histogram with two separated peaks
    (the coefficient alone flags skewed low-copy distributions)."""
    from scipy.signal import find_peaks
    b = _bimodality(values)
    if b is None or b <= 0.555:
        return False, b
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    counts, _ = np.histogram(v, bins=min(15, max(5, v.size // 4)))
    padded = np.concatenate([[0], counts, [0]])
    idx, _ = find_peaks(padded, prominence=max(2, 0.25 * counts.max()))
    return len(idx) >= 2, b


def _regular_peaks(series, t):
    """_peaks, kept only when the intervals are regular (CV < 0.35)."""
    from scipy.signal import find_peaks
    s = np.asarray(series, dtype=float)
    ok = np.isfinite(s)            # a run that stopped early: the part before it stopped
    s, t = s[ok], np.asarray(t)[ok]
    if s.size == 0 or np.ptp(s) <= 0:
        return None
    idx, _ = find_peaks(s, prominence=0.4 * np.ptp(s), distance=3)
    if len(idx) < 3:
        return None
    gaps = np.diff(t[idx])
    if np.std(gaps) / np.mean(gaps) > 0.35:
        return None
    return {"n_peaks": int(len(idx)), "mean_period": float(np.mean(gaps)),
            "period_cv": float(np.std(gaps) / np.mean(gaps))}


def _bimodality(values):
    """Sarle's bimodality coefficient; > 0.555 suggests two modes."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 10 or np.std(v) == 0:
        return None
    from scipy.stats import kurtosis, skew
    g, k = float(skew(v)), float(kurtosis(v))
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


# --- plots, files ------------------------------------------------------------------

def _out_dir(label):
    d = Path(config.PLOTS_DIR) / "stochastic" / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plot(results, grid, names, show, label, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [names.index(s) for s in show]
    fig, axes = plt.subplots(len(results), len(cols), squeeze=False,
                             figsize=(4.2 * len(cols), 2.9 * len(results)))
    for r, res in enumerate(results):
        for c, j in enumerate(cols):
            ax = axes[r][c]
            if res.get("ssa") is not None:
                for run in res["ssa"][:5]:
                    ax.step(grid, run[:, j], where="post", color="tab:blue", alpha=0.25, lw=0.8)
                ax.plot(grid, np.nanmean(res["ssa"][:, :, j], axis=0), color="tab:blue",
                        lw=1.8, label=f"{res.get('ssa_label', 'SSA')} mean" if r == c == 0 else None)
            if res.get("cle") is not None:
                ax.plot(grid, res["cle"][0, :, j], color="tab:orange", lw=0.9, alpha=0.8,
                        label="CLE run" if r == c == 0 else None)
            if res.get("ode") is not None:
                ax.plot(grid, res["ode"][:, j], color="black", lw=1.6, ls="--",
                        label="ODE" if r == c == 0 else None)
            ax.set_title(f"{names[j]}, Ω = {res['omega']:g}", fontsize=9)
            ax.set_xlabel("time", fontsize=8)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=8)  # at the top it covers the title
    shown = list(dict.fromkeys(r.get("ssa_label", "SSA") if m == "ssa" else m.upper()
                 for m in ("ode", "ssa", "cle") for r in results if r.get(m) is not None))
    fig.suptitle(f"{label}: {' vs '.join(shown) or 'no completed simulation'} (model units)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.4 / fig.get_figheight(), 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _hist(results, names, show, label, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with_ssa = [r for r in results if r.get("ssa") is not None]
    if not with_ssa:
        return None
    cols = [names.index(s) for s in show]
    fig, axes = plt.subplots(len(with_ssa), len(cols), squeeze=False,
                             figsize=(3.6 * len(cols), 2.5 * len(with_ssa)))
    for r, res in enumerate(with_ssa):
        for c, j in enumerate(cols):
            ax = axes[r][c]
            final = res["ssa"][:, -1, j]
            ax.hist(final[np.isfinite(final)], bins=20, color="tab:blue", alpha=0.7)
            if res.get("ode") is not None:
                ax.axvline(res["ode"][-1, j], color="black", ls="--", lw=1.2)
            ax.set_title(f"{names[j]} at t_end, Ω = {res['omega']:g}", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"{label}: SSA final-state distribution (dashed: ODE)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _write_csv(results, grid, names, path):
    import pandas as pd
    frames = []
    for res in results:
        df = pd.DataFrame({"omega": res["omega"], "time": grid})
        for j, n in enumerate(names):
            if res.get("ode") is not None:
                df[f"{n}_ode"] = res["ode"][:, j]
            if res.get("ssa") is not None:
                df[f"{n}_ssa_mean"] = np.nanmean(res["ssa"][:, :, j], axis=0)
                df[f"{n}_ssa_std"] = np.nanstd(res["ssa"][:, :, j], axis=0)
                df[f"{n}_ssa_run1"] = res["ssa"][0, :, j]
            if res.get("cle") is not None:
                df[f"{n}_cle_mean"] = np.nanmean(res["cle"][:, :, j], axis=0)
                df[f"{n}_cle_run1"] = res["cle"][0, :, j]
        frames.append(df)
    pd.concat(frames).to_csv(path, index=False)


_MATLAB_RESERVED = {"i", "j", "e", "pi", "eps", "inf", "nan", "beta", "gamma", "t", "omega",
                    "V", "X", "a", "asum", "tau", "tim", "coun", "dt", "sol", "tfinal", "k",
                    "x", "Y", "L", "S", "f", "s", "names", "tode", "xode", "Ycle", "tcle", "prop", "x0", "rng",
                    "end", "function", "if", "else", "for", "while"}


def matlab_script(model, channels, values, omega, t_end, seed, dt, readout=None,
                  n_rep=20, tech=None, burn_in=0.0, methods=None) -> str:
    """A MATLAB script in the style of SSA_MM_RED.m: SSA, CLE and ode45."""
    names = model["state_names"]
    ren = {p: (f"{p}_p" if p in _MATLAB_RESERVED else p) for p in model["param_names"]}
    psym = {sym: sp.Symbol(ren[n]) for n, sym in zip(model["param_names"], model["param_symbols"])}
    om = _omega_symbol()
    Xm = {s: sp.Symbol(f"X({i + 1})") for i, s in enumerate(model["state_symbols"])}
    xm = {s: sp.Symbol(f"x({i + 1})") for i, s in enumerate(model["state_symbols"])}
    code = lambda e: sp.octave_code(e).replace("Omega", "omega")
    props = [code(molecular_propensity(model, ch["rate"], ch.get("stoich")).xreplace({**psym, **Xm}))
             for ch in channels]
    rhs = [code(sp.sympify(e).xreplace({**psym, **xm})) for e in model["rhs"]]
    V = np.array([c["stoich"] for c in channels]).T
    vrows = "; ".join(" ".join(f"{int(v)}" for v in row) for row in V)
    y0 = [model["y0"][n] for n in names]
    used = {str(s) for e in model["rhs"] for s in sp.sympify(e).free_symbols}
    lines = [
        f"% {model['label']}: deterministic (ode45), stochastic (Gillespie SSA) and CLE",
        "% Generated by dynalyst (stochastic.py) following Higham (2008) SIAM Review 50:347.",
        "% Species order: " + ", ".join(names),
        "% Concentrations are molecules/omega; smaller omega = fewer molecules = more noise.",
        f"rng({seed});", f"omega = {omega:g};", f"tfinal = {t_end:g};", "",
        "% parameters",
    ]
    lines += [f"{ren[p]} = {values[p]:.10g};" for p in model["param_names"] if p in used]
    lines += ["", "% stoichiometric matrix (rows: species, columns: reactions)",
              f"V = [{vrows}];",
              f"x0 = [{'; '.join(f'{v:.10g}' for v in y0)}];   % initial concentrations",
              "X = round(omega*x0);",
              "prop = @(X) [" + "; ...\n              ".join(props) + "];", "",
              "% --- SSA (Gillespie direct method), nRep replicates on a time grid ---",
              f"nRep = {int(n_rep)}; tech = {float(tech or t_end / 400):g}; "
              f"trans = {float(burn_in):g};   % record every tech, pool after trans",
              "tgrid = 0:tech:tfinal; nT = numel(tgrid);",
              "Sall = zeros(nT, numel(x0), nRep);",
              "for rep = 1:nRep",
              "    X = round(omega*x0); t = 0; kk = 1;",
              "    while kk <= nT",
              "        a = max(prop(X), 0);",
              "        asum = sum(a);",
              "        if asum == 0, tau = inf; else, tau = log(1/rand)/asum; end",
              "        while kk <= nT && tgrid(kk) < t + tau",
              "            Sall(kk,:,rep) = X'; kk = kk + 1;",
              "        end",
              "        if kk > nT, break; end",
              "        j = find(rand < cumsum(a/asum), 1);",
              "        X = X + V(:,j);",
              "        t = t + tau;",
              "    end",
              "end",
              "tim = tgrid; S = Sall(:,:,1);   % first replicate for the time-course comparison",
              "", 
              "% --- CLE (Euler-Maruyama) ---",
              f"dt = {dt:g}; L = round(tfinal/dt);",
              "Y = round(omega*x0); Ycle = zeros(L+1, numel(Y)); Ycle(1,:) = Y';",
              "for k = 1:L",
              "    a = max(prop(Y), 0);",
              "    Y = Y + dt*V*a + sqrt(dt)*V*(sqrt(a).*randn(size(a)));",
              "    Y = max(Y, 0);",
              "    Ycle(k+1,:) = Y';",
              "end",
              "tcle = (0:L)*dt;", "",
              "% --- deterministic ODE (concentrations) ---",
              "f = @(t,x) [" + "; ...\n             ".join(rhs) + "];",
              "[tode, xode] = ode45(f, [0 tfinal], x0);", "",
              "% --- compare (concentration units) ---",
              f"names = {{{', '.join(repr(n) for n in names)}}};",
              "figure;",
              f"for s = 1:{len(names)}",
              f"    subplot({len(names)},1,s); hold on",
              "    stairs(tim, S(:,s)/omega, 'b');",
              "    plot(tcle, Ycle(:,s)/omega, 'Color', [1 0.5 0]);",
              "    plot(tode, xode(:,s), 'k--', 'LineWidth', 1.5);",
              f"    names = {{{', '.join(repr(n) for n in names)}}};",
              "    ylabel(names{s}); legend('SSA','CLE','ODE');",
              "end",
              "xlabel('time');", "",
              "% --- stationary distribution: all replicates pooled after trans ---",
              "ro = @RO@;   % read-out species (@RONAME@)",
              "keep = tgrid >= trans;",
              "pooled = reshape(Sall(keep, ro, :), [], 1);",
              "figure;",
              "subplot(1,2,1); hold on",
              "for rep = 1:nRep, plot(tgrid, squeeze(Sall(:, ro, rep)), 'Color', [0.85 0 0]); end",
              "xlabel('time'); ylabel(['molecules of ' names{ro}]);",
              "subplot(1,2,2)",
              "histogram(pooled, 'Orientation', 'horizontal', 'FaceColor', [0.75 0.75 0.75]);",
              "set(gca, 'XDir', 'reverse'); xlabel('counts');",
              "subplot(1,2,1); yl = ylim; subplot(1,2,2); ylim(yl);",
              "% Empirical peaks alone do not establish stationarity or deterministic stability."]
    methods = set(methods or ["ode", "ssa", "cle"])
    kept, block = [], None
    for line in lines:
        if line.startswith("% --- SSA"):
            block = "ssa"
        elif line.startswith("% --- CLE"):
            block = "cle"
        elif line.startswith("% --- deterministic"):
            block = "ode"
        elif line.startswith("% --- compare"):
            block = None
        elif line.startswith("% --- stationary distribution"):
            block = "ssa"
            line = "% --- empirical distribution: samples pooled after trans ---"
        if block and block not in methods:
            continue
        if ("stairs(tim" in line and "ssa" not in methods
                or "plot(tcle" in line and "cle" not in methods
                or "plot(tode" in line and "ode" not in methods):
            continue
        if "legend('SSA','CLE','ODE')" in line:
            labels = ",".join(repr(m.upper()) for m in ("ssa", "cle", "ode") if m in methods)
            line = f"    ylabel(names{{s}}); legend({labels});"
        kept.append(line)
    lines = kept
    lines[0] = f"% {model['label']}: " + ", ".join(m.upper() for m in ("ode", "ssa", "cle") if m in methods)
    ro_name = readout if readout in names else names[0]
    lines = [ln.replace("@RO@", str(names.index(ro_name) + 1)).replace("@RONAME@", ro_name)
             for ln in lines]
    renamed = {p: n for p, n in ren.items() if p != n}
    if renamed:
        lines.insert(4, "% renamed to avoid MATLAB names: "
                     + ", ".join(f"{a} -> {b}" for a, b in renamed.items()))
    return "\n".join(lines) + "\n"


# --- tools -------------------------------------------------------------------------

def _prepare(model_name_or_path, species, odes, parameters, initial_conditions, t_end):
    model, values, changed = _load(model_name_or_path, species, odes, parameters,
                                   initial_conditions)
    from dynalyst.analysis.hybrid_sim import has_delay
    if model.get("events") or has_delay(model) or any(
            model.get("time_symbol") in e.free_symbols for e in model["rhs"]):
        raise NotDecomposable("This stochastic engine requires autonomous reaction rates without events or delays; use deterministic simulation for this model.")
    metadata = model.get("state_metadata", {})
    if any(s.get("kind") != "species" for s in metadata.values()):
        raise NotDecomposable("A rate-rule parameter or compartment is not a molecule-count state.")
    if len({s.get("amount_units", False) for s in metadata.values()}) > 1:
        raise NotDecomposable("Mixed amount/concentration species require explicit per-species count scaling.")
    if len({v for v in model.get("compartments", {}).values()}) > 1:
        raise NotDecomposable("Multiple compartment sizes require explicit per-compartment count scaling.")
    if any(float(v) < 0 for v in model["y0"].values()):
        raise NotDecomposable("Stochastic molecular initial values must be non-negative.")
    pts = sample_points(model, values, t_end)
    channels = reaction_channels(model, values, pts)
    if not channels:
        raise NotDecomposable("The model has no reactions (every right-hand side is zero).")
    _check_reconstruction(model, channels, values, pts)
    return model, values, changed, channels


def propensity_table(model_name_or_path: str | None = None, species: list | None = None,
                     odes: dict | list | None = None, parameters: dict | None = None,
                     omega: float | None = None, export_matlab: bool = False,
                     t_end: float = 100.0, state: dict | None = None) -> str:
    """The model as reaction channels: stoichiometry and propensities (the
    table that defines the SSA), with an optional MATLAB script, and the propensities'
    values at the initial state (or at `state`, in the model's concentration units) for
    system size omega (default 1)."""
    try:
        model, values, changed, channels = _prepare(model_name_or_path, species, odes,
                                                    parameters, None, t_end)
    except (NotDecomposable, ValueError, FileNotFoundError, RuntimeError) as e:
        return json.dumps({"error": str(e)})
    rows = propensity_rows(model, channels, values)
    out = {"model": model["label"], "species": model["state_names"],
           "n_reactions": len(rows), "reactions": rows,
           "stoichiometric_matrix": {"rows (species)": model["state_names"],
                                     "V": np.array([c["stoich"] for c in channels]).T.tolist()},
           "integer_stoichiometry": all(c["integer"] for c in channels),
           "table_markdown": _markdown_table(rows),
           "stochastic_model": _model_assumptions(model, channels),
           "how_derived": ("each ODE term that appears in several equations is one reaction; "
                           "propensity w = Omega * rate(X/Omega), X in molecules, Omega "
                           "molecules per concentration unit (Higham 2008)")}
    try:
        om = float(omega) if omega else 1.0
        conc = {n: float((state or {}).get(n, model["y0"][n])) for n in model["state_names"]}
        x = np.array([conc[n] * om for n in model["state_names"]], dtype=float)
        a = _propensity_fn(model, channels, values, om)(x)
        out["values_at_state"] = {
            "state": "the initial state" if not state else "the state given",
            "concentrations": conc, "omega": om, "molecules": dict(zip(model["state_names"], x.tolist())),
            "propensities": {f"w{k}": float(v) for k, v in enumerate(np.atleast_1d(a), 1)},
            "total_propensity": float(np.sum(a))}
    except Exception as e:          # the table itself is still valid
        out["values_at_state"] = {"error": f"propensities could not be evaluated: {e}"}
    if changed:
        out["parameters_changed"] = {k: {"model_value": a, "used": b} for k, (a, b) in changed.items()}
    approximate = _approximate_channels(model, channels)
    if approximate:
        out["approximate_propensities"] = {"channels": approximate, "note": APPROXIMATE_NOTE}
    if not out["integer_stoichiometry"]:
        out["warning"] = ("Some stoichiometry is not whole numbers: the SSA cannot run on this "
                          "model as written (the CLE can).")
    if export_matlab and out["integer_stoichiometry"]:
        d = _out_dir(model["label"])
        p = d / f"{_safe(model['label'])}_ssa_cle.m"
        p.write_text(matlab_script(model, channels, values, omega or 1000.0, t_end, 1000,
                                   t_end / 10000), encoding="utf-8")
        out["matlab_script"] = config.display_path(p)
    return json.dumps(out, indent=2, default=float)


def _safe(label):
    s = re.sub(r"\W", "_", str(label))
    return s if s and s[0].isalpha() else f"m_{s}"


def _model_assumptions(model, channels):
    sources = sorted({c.get("source", "inferred from ODE terms") for c in channels})
    return {
        "channel_sources": sources,
        "propensity_convention": "Omega * rate(X/Omega)",
        "count_scale": "Omega is molecules per model state unit, not a volume inferred from SBML units.",
        "assumptions": [
            "Autonomous, well-mixed reaction system with the reported jump vectors.",
            "ODE-derived channels are one possible stochastic model; matching drift does not identify noise.",
            "Density-scaled nonlinear rates do not automatically supply elementary-reaction combinatorial corrections.",
            "Exact SSA refers to sampling the specified jump process, not validation of its physical assumptions.",
        ],
        "validation": "Finite inputs, supported state units, channel reconstruction and sampled rates; not a global propensity proof.",
    }


def stochastic_simulation(model_name_or_path=None, **kwargs):
    """Stochastic methods only. Default: exact Gillespie SSA with NumPy/JAX."""
    try:
        kwargs["methods"] = _selected_methods(kwargs.get("methods"), comparison=False)
        if kwargs.get("start_in_each_state"):
            raise ValueError("Starting from deterministic attractors requires compare_simulations.")
        return _run_simulations(model_name_or_path=model_name_or_path, **kwargs)
    except (ValueError, TypeError, RuntimeError, FloatingPointError) as exc:
        return json.dumps({"error": str(exc), "requested_methods": kwargs.get("methods")})


def compare_simulations(model_name_or_path=None, **kwargs):
    """Coordinate deterministic and stochastic solvers on identical model inputs."""
    try:
        kwargs["methods"] = _selected_methods(kwargs.get("methods"), comparison=True)
        return _run_simulations(model_name_or_path=model_name_or_path, **kwargs)
    except (ValueError, TypeError, RuntimeError, FloatingPointError) as exc:
        return json.dumps({"error": str(exc), "requested_methods": kwargs.get("methods")})


def _default_burn_in(ode, grid):
    """10 % of t_end, or longer when the ODE is still relaxing from the start: the
    time after which every species stays within 5 % (of its range) of its final value,
    capped at half of t_end (an oscillation never settles and keeps the 10 %)."""
    t_end = float(grid[-1])
    burn = 0.1 * t_end
    try:
        rng_ = np.ptp(ode, axis=0)
        rng_ = np.where(rng_ > 0, rng_, 1.0)
        off = np.any(np.abs(ode - ode[-1]) > 0.05 * rng_, axis=1)
        if off.any():
            settle = float(grid[np.flatnonzero(off)[-1]])
            tail = ode[grid >= 0.75 * t_end]
            settled = np.all(np.ptp(tail, axis=0) <= 0.05 * rng_)
            if settled:
                burn = max(burn, min(settle, 0.5 * t_end))
    except Exception:
        pass
    return burn


def _fastest_rate(model, values, ode):
    """Largest |eigenvalue| of the Jacobian along the ODE path (1/time)."""
    try:
        subs = _param_subs(model, values)
        states = model["state_symbols"]
        J = sp.lambdify(states, sp.Matrix([sp.sympify(e).subs(subs) for e in model["rhs"]])
                        .jacobian(states), "numpy")
        lam = 0.0
        for row in ode[:: max(1, len(ode) // 12)]:
            ev = np.linalg.eigvals(np.array(J(*row), dtype=float))
            lam = max(lam, float(np.max(np.abs(ev))))
        return lam
    except Exception:
        return 0.0


def _run_ssa(method, model, channels, values, om, a_fn, V, x0, grid, rng, seed, n_runs,
             est_events, budget_s):
    """(trajectories in molecules or None, info). Chooses the engine:
    auto   exact numpy if it fits the time budget, else exact JAX if installed,
           else skipped with the estimate and the options;
    exact  as auto, but numpy runs even over the budget (up to SSA_MAX_SECONDS_FORCED);
    tau_leap  tau-leaping (approximate; only helps when the fast species have many
           molecules -- with low-copy fast species it falls back to exact steps)."""
    from dynalyst.analysis import stochastic_ssa as eng
    started = time.time()
    stop_over = None if method == "exact" else SSA_STOP_FACTOR * SSA_MAX_SECONDS
    per_step = eng.numpy_seconds_per_step(a_fn, x0, n_runs, rng)
    np_time = 1.3 * est_events * per_step          # all runs advance together
    info = {"expected_events_per_run": int(est_events),
            "estimated_numpy_exact_seconds": round(np_time, 1)}
    d = None
    if method == "tau_leap":
        out, d = eng.tau_leap(a_fn, V, x0, grid, rng, n_runs, max_steps=_steps_for(budget_s),
                              max_seconds=max(budget_s, 10.0), stop_over=stop_over)
    elif np_time <= budget_s:
        out, d = eng.exact_numpy(a_fn, V, x0, grid, rng, n_runs, MAX_EVENTS_JAX,
                                 max_seconds=max(budget_s, 10.0), stop_over=stop_over)
    else:
        if eng.jax_available() and est_events <= MAX_EVENTS_JAX:
            try:
                out, d = eng.exact_jax(model, channels, values, om, V, x0, grid, seed, n_runs,
                                       MAX_EVENTS_JAX,      # a safety limit: the estimate can be far off
                                       max_seconds=(SSA_MAX_SECONDS_FORCED if method == "exact"
                                                    else max(budget_s, 10.0)), stop_over=stop_over)
            except Exception as e:           # fall back rather than fail
                info["jax_error"] = f"{type(e).__name__}: {e}"[:300]
        if d is None and method == "exact" and np_time <= SSA_MAX_SECONDS_FORCED:
            out, d = eng.exact_numpy(a_fn, V, x0, grid, rng, n_runs, MAX_EVENTS_JAX,
                                     max_seconds=SSA_MAX_SECONDS_FORCED)
        if d is None:
            why = ("JAX failed (see jax_error)" if "jax_error" in info
                   else "JAX is not installed" if not eng.jax_available()
                   else "the run needs more events than the JAX limit")
            return None, {**info, "skipped": (
                f"an exact SSA needs about {est_events:.2g} events per run: about "
                f"{np_time / 60:.0f} min with numpy (budget {budget_s / 60:.0f} min) and {why}. "
                "Options: ssa_method='exact' to run it anyway (numpy, up to "
                f"{SSA_MAX_SECONDS_FORCED / 3600:.0f} h), fewer runs, or "
                "install jax (pip install jax) for the fast exact engine.")}
    info.update({"blocked_events": int(d.get("blocked_events") or 0),
                 "observed_over_estimated_events": round(float(np.mean(d["events"])) / max(est_events, 1.0), 2),
                 "run_status": {"completed": int(n_runs - np.sum(d["truncated"])),
                                "truncated": int(np.sum(d["truncated"])), "failed": 0}})
    info.update({"engine": d["engine"], "runs": int(n_runs),
                 "mean_events_per_run": int(np.mean(d["events"])),
                 "negative_propensity_steps": int(d["negative_propensity_steps"]),
                 "truncated_runs": int(np.sum(d["truncated"])),
                 "absorbed_runs": int(np.sum(d["absorbed"]))})
    for key in ("leaps", "exact_steps", "rejected_leaps", "epsilon"):
        if key in d:
            info[key] = d[key]
    if d.get("time_limited"):
        reached = int(n_runs - np.sum(d["truncated"]))
        info["time_limit"] = (
            f"the run stopped at its time limit ({d['time_limited']:.0f} s): {reached} of "
            f"{int(n_runs)} runs reached t_end; the others end where they stopped (their later "
            "time points are missing). Options: fewer runs (n_runs), or ssa_method='exact' to run "
            f"up to {SSA_MAX_SECONDS_FORCED / 3600:.0f} h.")
    if d.get("stopped_early"):
        info["stopped_early"] = {**d["stopped_early"], "note": (
            "Stopped once its own pace showed a full run was impractical. The partial runs are "
            "insufficient evidence: this says nothing against oscillation or any other behaviour.")}
    if info["truncated_runs"]:
        info["projected_full_run"] = _projected_run(out, grid, d["events"], time.time() - started,
                                                    SSA_MAX_SECONDS)
    if "leaps" in d and d["exact_steps"] > 10 * max(1, d["leaps"]):
        info["note"] = ("tau-leaping fell back to exact steps almost everywhere (the fast "
                        "reactions involve species with few molecules), so it was not faster.")
    return out, info


def _rounding(names, y0, om) -> dict:
    """How the start turns into whole molecules at one size, with each species' relative error."""
    out = {}
    for n, c in zip(names, y0):
        count = float(np.round(c * om))
        out[n] = {"concentration": float(c), "molecules": int(count),
                  "relative_error": round(abs(count / om - c) / abs(c), 4) if c else 0.0}
    return out


def _projected_run(out, grid, events, seconds, limit) -> dict:
    """Approximate wall time of a full run at this Omega, from this run's own event rate: the
    events so far over the model time each run reached give the events it still needs."""
    reached = grid[np.maximum(np.isfinite(out[:, :, 0]).sum(axis=1) - 1, 1)]
    events = np.asarray(events, dtype=float)
    needed = events * grid[-1] / reached
    full = float(seconds * needed.sum() / max(events.sum(), 1.0))
    return {"approximate": True, "seconds": round(full), "minutes": round(full / 60, 1),
            "remaining_minutes": round(max(full - seconds, 0.0) / 60, 1),
            "events_per_run": int(needed.mean()),
            "basis": "this run's event rate and the model time it reached, at the same pace to t_end",
            "retry": (f"With these settings a retry cannot finish inside one call's {limit:.0f} s limit: "
                      "it stops early again. Fewer runs need less."
                      if full > limit else "These settings may finish in one call.")}


def _steps_for(budget_s):
    return int(max(1e5, budget_s / 60e-6))


def _run_simulations(model_name_or_path: str | None = None, species: list | None = None,
                          odes: dict | list | None = None, t_end: float = 100.0,
                          omega: float | list | None = None, methods: list | None = None,
                          n_runs: int = 20, n_points: int = 400, dt: float | None = None,
                          seed: int = 0, parameters: dict | None = None,
                          initial_conditions: dict | None = None,
                          plot_species: list | None = None,
                          export_matlab: bool = False, record_dt: float | None = None,
                          burn_in: float | None = None, distribution: bool = True,
                          distribution_species: str | list | None = None,
                          ssa_method: str = "auto", start_in_each_state: bool = False) -> str:
    """ODE vs Gillespie SSA vs chemical Langevin equation for one model, at one
    or several system sizes Omega (default 1), with the stationary distribution of a
    read-out species and a verdict on how many stable states there are."""
    from dynalyst.analysis import stochastic_distribution as dist
    t0 = time.time()
    methods = [m.lower() for m in methods]
    bad = [m for m in methods if m not in ("ode", "ssa", "cle")]
    if bad:
        return json.dumps({"error": f"Unknown methods {bad}; use 'ode', 'ssa', 'cle'."})
    if ssa_method not in ("auto", "exact", "tau_leap"):
        return json.dumps({"error": "ssa_method must be 'auto', 'exact' or 'tau_leap'."})
    if record_dt:
        n_points = int(min(20001, math.floor(float(t_end) / float(record_dt)) + 1))
    if (not np.isfinite(t_end) or t_end <= 0 or n_points < 2 or n_runs < 1
            or int(n_runs) != n_runs or int(n_points) != n_points):
        return json.dumps({"error": "Need t_end > 0, n_points >= 2 and n_runs >= 1."})
    try:
        model, values, changed, channels = _prepare(model_name_or_path, species, odes,
                                                    parameters, initial_conditions, t_end)
    except (NotDecomposable, ValueError, FileNotFoundError, RuntimeError) as e:
        return json.dumps({"error": str(e)})
    names = model["state_names"]
    show = plot_species or names[:4]
    unknown = [s for s in show if s not in names]
    if isinstance(distribution_species, str):
        distribution_species = [distribution_species]
    unknown += [s for s in (distribution_species or []) if s not in names]
    if unknown:
        return json.dumps({"error": f"Unknown species {unknown}. Species: {names}"})

    integer = all(c["integer"] for c in channels)
    warnings = []
    unavailable = {}
    requested_methods = list(methods)
    if "ssa" in methods and not integer:
        methods = [m for m in methods if m != "ssa"]
        unavailable["ssa"] = "SSA requires integer stoichiometry; no other method was substituted."
        warnings.append("SSA skipped: " + unavailable["ssa"])
        if not methods:
            return json.dumps({"error": unavailable["ssa"], "requested_methods": requested_methods})
    grid = np.linspace(0.0, float(t_end), int(n_points))
    gap = float(grid[1] - grid[0])
    V = np.array([c["stoich"] for c in channels]).T
    y0 = np.array([model["y0"][n] for n in names], dtype=float)
    ode = None
    if "ode" in methods:         # a comparison plans from what the model reaches
        try:
            ode = _ode(model, values, t_end, grid)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            unavailable["ode"] = str(exc)
            warnings.append(f"Deterministic reference failed: {exc}; stochastic methods can still run.")
    reference, reference_info = ode, {"status": "succeeded", "source": "the requested ODE"} if ode is not None else None
    if ode is None and "ssa" in methods:
        reference, reference_info = _reference_ode(model, values, grid)
    if reference_info and reference is None:
        warnings.append(f"Deterministic reference {reference_info['status']}: the event estimate comes "
                        "from the start values only, and no period is compared.")
    probes = reference if reference is not None else np.asarray(sample_points(model, values, t_end)["path"])
    if reference_info and reference is not None:
        peak = int(np.argmax(np.nanmax(np.abs(reference), axis=0)))
        reference_info.update(peak_species=names[peak], peak_value=float(np.nanmax(np.abs(reference[:, peak]))),
                              used_for="the event estimate and the period comparison")
    if omega is None or omega == "auto" or omega == []:
        omegas = [1.0]
        omega_note = ("omega = 1, the default: no omega was given. The tool never chooses or "
                      "changes omega; a given omega is used as it is.")
    else:
        omegas = [float(o) for o in (omega if isinstance(omega, (list, tuple)) else [omega])]
        omega_note = None
    if any(not np.isfinite(o) or o <= 0 for o in omegas):
        return json.dumps({"error": "omega must be finite and positive (molecules per model state unit)."})
    lam = _fastest_rate(model, values, probes) if "cle" in methods else 0.0
    cle_dt = float(dt) if dt else min(gap, 0.05 / lam if lam > 0 else gap / 20)
    if not np.isfinite(cle_dt) or cle_dt <= 0:
        return json.dumps({"error": "dt must be finite and positive."})
    rng = np.random.default_rng(seed)
    budget = float(SSA_MAX_SECONDS)
    # one time budget for the whole call (every omega, SSA, CLE and extra runs), not per step
    left = lambda: max(0.0, budget - (time.time() - t0))
    burn = (float(burn_in) if burn_in is not None else
            _default_burn_in(ode, grid) if ode is not None else 0.1 * t_end)
    if not np.isfinite(burn) or not 0 <= burn < t_end:
        return json.dumps({"error": "burn_in must be finite and in [0, t_end)."})
    keep = grid >= burn

    results, summary, dist_entries = [], [], []
    for om in omegas:
        a_fn = _propensity_fn(model, channels, values, om)
        x0 = np.round(y0 * om)
        res = {"omega": om, "ode": ode if "ode" in methods else None}
        info = {"omega": om, "initial_molecules": dict(zip(names, x0.astype(int).tolist())),
                "method_results": {m: {"status": "unavailable", "reason": why}
                                   for m, why in unavailable.items()}}
        if ode is not None:
            info["method_results"]["ode"] = {"status": "completed",
                "method": "deterministic ODE", "backend": "SciPy LSODA"}
        lost = [n for n, a, x in zip(names, y0, x0) if a > 0 and x == 0]
        if lost and ("ssa" in methods or "cle" in methods):
            # the SSA counts whole molecules: say so whenever a species that is present
            # would start at 0, rather than silently dropping it
            info["species_rounded_to_zero"] = lost
            made = dict(zip(names, (V > 0).any(axis=1)))
            gone = [n for n in lost if not made[n]]
            amounts = ', '.join(f'{n} = {a:g}' for n, a in zip(names, y0) if n in lost)
            warnings.append(
                f"Ω={om:g}: {lost} start with less than half a molecule ({amounts} concentration "
                "units), so the stochastic runs start them at 0 molecules"
                + (f"; {gone} are made by no reaction, so they stay absent there. A larger omega "
                   "keeps them." if gone else
                   ": the reactions produce them, so they appear during the runs (the start "
                   "differs slightly from the ODE's)."))
        rates = np.array([a_fn(row * om).sum() for row in probes[:: max(1, len(probes) // 50)]])
        if not np.all(np.isfinite(rates)) or np.any(rates < 0):
            return json.dumps({"error": "Invalid propensity at the estimation states."})
        est = float(np.nanmean(rates) * t_end) if rates.size else 0.0
        info["expected_events_per_ssa_run"] = int(est)
        info["event_estimate_basis"] = "deterministic reference" if reference is not None else "initial rates only; event count may change"
        if "ssa" in methods:
            try:
                traj, sinfo = _run_ssa(ssa_method, model, channels, values, om, a_fn, V, x0, grid,
                                       rng, seed, int(n_runs), est, left())
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                traj, sinfo = None, {"failed": str(exc), "skipped": f"SSA failed: {exc}"}
            info["ssa"] = sinfo
            res["ssa_stopped_early"] = sinfo.get("stopped_early")
            engine = sinfo.get("engine", "")
            info["method_results"]["ssa"] = {
                "status": "failed" if sinfo.get("failed") else "unavailable" if traj is None else "incomplete" if sinfo.get("truncated_runs") else "completed",
                "method": "tau-leaping" if engine.startswith("tau") else "Gillespie SSA",
                "backend": "JAX" if "jax" in engine.lower() else "NumPy" if engine else None,
                "approximate": engine.startswith("tau"), "details": sinfo}
            if traj is None:
                warnings.append(f"Ω={om:g}: SSA skipped: {sinfo['skipped']}")
            else:
                res["ssa"] = traj / om
                res["ssa_label"] = "tau-leaping" if engine.startswith("tau") else "Gillespie SSA"
                if sinfo["engine"].startswith("tau"):
                    warnings.append(f"Ω={om:g}: tau-leaping is approximate (Cao et al. 2006)."
                                    + (" " + sinfo["note"] if sinfo.get("note") else ""))
                if sinfo["blocked_events"]:
                    warnings.append(f"Ω={om:g}: {sinfo['blocked_events']} events were blocked for lack of "
                                    "molecules: a rate law here does not vanish when its reactant runs out.")
                if sinfo["negative_propensity_steps"]:
                    warnings.append(f"Ω={om:g}: negative propensities were set to 0 "
                                    f"({sinfo['negative_propensity_steps']} times): a rate law is "
                                    "not a valid propensity everywhere the SSA went.")
                if sinfo["truncated_runs"]:
                    full = sinfo["projected_full_run"]
                    if sinfo.get("stopped_early"):
                        warnings.append(f"Ω={om:g}: stopped after {sinfo['stopped_early']['after_seconds']:g} s "
                                        "of a pilot: insufficient evidence, not a negative result.")
                    warnings.append(f"Ω={om:g}: {sinfo['truncated_runs']} SSA run(s) stopped "
                                    "early (event or time limit); NaN after that. Approximate time "
                                    f"for a full run: {full['minutes']:g} min. {full['retry']}")
        if "cle" in methods:
            n_steps = float(t_end) / cle_dt
            from dynalyst.analysis.stochastic_ssa import numpy_seconds_per_step
            cle_time = n_steps * 1.5 * numpy_seconds_per_step(a_fn, x0, int(n_runs), rng)
            if cle_time > left():
                info["cle"] = (f"skipped: {n_steps:.2g} Euler-Maruyama steps (dt {cle_dt:.3g}, "
                               f"set by the fastest timescale) would take about "
                               f"{cle_time / 60:.0f} min. Pass a larger dt or a shorter t_end.")
                warnings.append(f"Ω={om:g}: CLE skipped (too many steps).")
            else:
                try:
                    cle, cinfo = cle_runs(a_fn, V, x0, grid, rng, int(n_runs), cle_dt)
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    info["cle"] = {"failed": str(exc)}
                    warnings.append(f"Ω={om:g}: CLE failed: {exc}")
                else:
                    res["cle"] = cle / om
                    info["cle"] = {"runs": int(n_runs), **cinfo}
                    frac = cinfo["clipped_states"] / max(1, len(grid) * int(n_runs) * len(names)
                                                         * max(1, round(gap / cinfo["dt"])))
                    info["cle"]["clipped_state_fraction"] = frac
                    if frac > 1e-3:
                        warnings.append(f"Ω={om:g}: the CLE went below zero {cinfo['clipped_states']} "
                                        "time(s) and was clipped at 0: copy numbers get low there "
                                        "(e.g. troughs), where the CLE is not accurate; the SSA is "
                                        "exact.")
        if "cle" in methods:
            info["method_results"]["cle"] = {
                "status": "completed" if res.get("cle") is not None else
                          "failed" if isinstance(info.get("cle"), dict) and info["cle"].get("failed") else "unavailable",
                "method": "chemical Langevin equation (Euler-Maruyama)",
                "backend": "NumPy", "approximate": True, "details": info.get("cle")}
        info["species"] = _species_stats(res, reference, grid, names, om)
        if (res.get("ssa") is not None and _approximate_channels(model, channels)
                and np.nanmax(res["ssa"]) * om < config.SSA_FEW_MOLECULES):
            warnings.append(f"Ω={om:g}: at most {np.nanmax(res['ssa']) * om:.0f} molecules of any species. The "
                            "Michaelis-Menten or Hill rate laws assume many molecules, so these runs, any "
                            "switching included, are not reliable at this size.")
        if distribution and (res.get("ssa") is not None or res.get("cle") is not None):
            info["_channels"] = channels
            # an SSA-only call still has the internal ODE: the stable states and the verdict need no requested ODE
            d = _distribution(model, values, res, om, grid, keep, burn, names,
                              ode if ode is not None else reference, y0,
                              distribution_species, info, float(t_end), gap)
            info.pop("_channels", None)
            info["distribution"] = d["summary"]
            for e_ in d["entries"]:
                if e_["discrete"] and res.get("ssa") is not None:
                    e_["states"], e_["_a_fn"], e_["state_names"] = res["ssa"] * om, a_fn, names
            dist_entries += d["entries"]
            if (start_in_each_state and res.get("ssa") is not None
                    and "deterministic only" in str(((info.get("distribution") or {})
                                                     .get("by_method", {}).get("SSA") or {})
                                                    .get("verdict"))):
                extra = _each_state_runs(ssa_method, model, channels, values, om, a_fn, V,
                                         grid, rng, seed, int(n_runs), keep, burn, gap, info,
                                         dist_entries, left(), est)
                if extra:
                    info["distribution"]["started_in_each_state"] = extra
        results.append(res)
        summary.append(info)

    label = model["label"]
    d = _out_dir(_safe(label))
    plot = d / f"{_safe(label)}_stochastic.png"
    _plot(results, grid, names, show, label, plot)
    dist_plot = None
    if dist_entries:
        dist_plot = d / f"{_safe(label)}_distribution.png"
        dist.plot(dist_entries, grid, label, dist_plot)
    switch_plot, switch_info = None, None
    from dynalyst.analysis import stochastic_switching as swp
    for e_ in sorted((e for e in dist_entries if e.get("discrete")),
                     key=lambda e: -((e.get("modes") or {}).get("n_modes") or 0)):
        candidate = d / f"{_safe(label)}_switching_omega{e_['omega']:g}.png"
        try:
            got = swp.plot(e_, grid, label, candidate, a_fn=e_.get("_a_fn"), V=V,
                           ssa_run=ssa_run)
        except (ValueError, RuntimeError, FloatingPointError, IndexError) as exc:
            warnings.append(f"switching plot failed: {exc}")
            got = None
        if got:
            switch_plot = candidate
            switch_info = {"omega": e_["omega"], "method": e_["method"],
                           "readout": e_["readout"],
                           **{k: v for k, v in got.items() if k != "path"}}
            break
    csv = d / f"{_safe(label)}_stochastic.csv"
    _write_csv(results, grid, names, csv)
    rows = propensity_rows(model, channels, values)
    (d / "propensity_table.md").write_text(_markdown_table(rows), encoding="utf-8")
    verdicts = [{"omega": i["omega"], "method": m, "verdict": v["verdict"]}
                for i in summary for m, v in (i.get("distribution") or {}).get("by_method", {}).items()]
    verdicts += [{"omega": i["omega"], "method": "SSA, started in each stable state",
                  "verdict": i["distribution"]["started_in_each_state"]["verdict"]}
                 for i in summary
                 if ((i.get("distribution") or {}).get("started_in_each_state") or {}).get("verdict")]
    from dynalyst.analysis.simulation_model import provenance
    executed = [m for m in requested_methods if any(r.get(m) is not None for r in results)]
    complete = all(p["method_results"].get(m, {}).get("status") == "completed"
                   for p in summary for m in requested_methods)
    out = {
        "model": label, "species": names, "methods": methods, "omega": omegas,
        "requested_methods": requested_methods, "executed_methods": executed,
        "status": "completed" if complete else "partial" if executed else "failed",
        "model_inputs": provenance(model, values),
        "stochastic_model": _model_assumptions(model, channels),
        "t_end": t_end, "n_runs": n_runs, "seed": seed, "burn_in": burn, "ssa_method": ssa_method,
        "units": "model state units (molecule counts / Omega); distributions in molecule counts",
        "stability_verdicts": verdicts,
        "reactions": rows, "table_markdown": _markdown_table(rows),
        "per_omega": summary,
        "interpretation": _interpret(summary, names),
        "sampled_time_series": _sample(results, grid, names),
        "plot_path": config.display_path(plot),
        "distribution_plot_path": config.display_path(dist_plot) if dist_plot else None,
        "switching_plot_path": config.display_path(switch_plot) if switch_plot else None,
        "switching_events": switch_info,
        "csv": config.display_path(csv),
        "plot_note": ("Real plots were saved. Embed them in the answer as markdown: "
                      f"![{label} {' vs '.join(executed).upper()}]({config.display_path(plot)})"
                      + (f" and ![{label} stochastic distribution]({config.display_path(dist_plot)})"
                         if dist_plot else "")
                      + (f" and ![{label} switching between states]({config.display_path(switch_plot)})"
                         if switch_plot else "")),
        "method_note": ("Reaction channels from the SBML reactions or the ODE terms; propensity "
                        "w = Omega * rate(X/Omega); SSA = Gillespie direct method (or "
                        "tau-leaping when stated); CLE = Euler-Maruyama (Higham 2008). The "
                        "distribution pools every sample after burn_in over time and runs; its "
                        "peaks alone do not establish stationarity or deterministic stability. "
                        "An explicit ODE comparison can add deterministic attractor evidence."),
        "warnings": warnings,
        "runtime_s": round(time.time() - t0, 2),
    }
    if switch_info and switch_info.get("jumps"):
        _note_switching(out, switch_info)
    if omega_note:
        out["omega_note"] = omega_note
    if any(info.get("species_rounded_to_zero") for info in summary):
        out["start_rounding"] = {f"{i['omega']:g}": _rounding(names, y0, i["omega"]) for i in summary}
    if reference_info:
        out["ode_reference"] = reference_info
    approximate = _approximate_channels(model, channels)
    if approximate:
        out["approximate_propensities"] = {"channels": approximate, "note": APPROXIMATE_NOTE}
    if "ssa" in methods and ode is None:
        out["period_note"] = ((
            "The ODE ran internally for the event estimate and periods only: ode_oscillation and ssa_oscillation per "
            "species. ssa_oscillation covers all runs, a stopped run up to where it stopped; report its median_period only when "
            "reliable is true. compare_simulations also plots them together.") if reference is not None
            else ("No deterministic reference finished, so no period was compared; say so, do not infer one."))
    if changed:
        out["parameters_changed"] = {k: {"model_value": a, "used": b} for k, (a, b) in changed.items()}
    if initial_conditions:
        out["initial_conditions_used"] = initial_conditions
    if export_matlab and ssa_method == "tau_leap" and "ssa" in methods:
        out["warnings"].append("MATLAB tau-leaping export is not implemented; no different algorithm was exported.")
    elif export_matlab and integer:
        p = d / f"{_safe(label)}_ssa_cle.m"
        ro = None
        for i in summary:
            for r_ in ((i.get("distribution") or {}).get("readouts") or []):
                ro = ro or r_["species"]
        p.write_text(matlab_script(model, channels, values, omegas[0], t_end, seed or 1000,
                                   cle_dt, readout=ro, n_rep=n_runs, tech=gap, burn_in=burn, methods=methods),
                     encoding="utf-8")
        out["matlab_script"] = config.display_path(p)
        out["matlab_methods"] = methods
    return json.dumps(out, indent=2, default=float)


def _each_state_runs(ssa_method, model, channels, values, om, a_fn, V, grid, rng, seed, n_runs,
                     keep, burn, gap, info, dist_entries, budget, est):
    """The ODE has two or more stable states but every run stayed in one: start the
    runs split over all stable states and see whether each state holds under noise
    (bistable with slow switching) or the runs leave it (only one state survives noise)."""
    from dynalyst.analysis import stochastic_distribution as dist
    entry = next((e for e in dist_entries if e["method"] == "SSA" and e["omega"] == om), None)
    if entry is None:
        return None
    stable = [a for a in entry["attractors"]
              if a["kind"] == "steady state" and a["stability"] == "stable"]
    if len(stable) < 2:
        return None
    names = model["state_names"]
    j = names.index(entry["readout"])
    per = max(1, n_runs // len(stable))
    series, groups, full = [], [], []
    for k, a in enumerate(stable):
        x0 = np.round(np.array(a["state"]) * om)
        # events from THIS state's own rates: another state can be far busier
        est_k = max(est, float(np.sum(a_fn(x0))) * float(grid[-1]))
        traj, sinfo = _run_ssa(ssa_method, model, channels, values, om, a_fn, V, x0, grid,
                               rng, seed + 101 * (k + 1), per, est_k,
                               budget / (len(stable) - k))
        if traj is None:
            return {"skipped": sinfo.get("skipped")}
        series.append(traj[:, :, j])
        full.append(traj)
        groups.append({"started_at": round(a["readout"] * om, 4), "runs": per})
    series = np.vstack(series)
    pooled = series[:, keep].ravel()
    md = dist.modes(pooled, discrete=True)
    if md is None:
        return None
    sw = dist.switching(series[:, keep], md, gap)
    left = 0
    for g_idx, a in enumerate(stable):
        block = series[g_idx * per:(g_idx + 1) * per, -1]
        targets = [b["readout"] * om for b in stable]
        left += int(sum(int(np.argmin([abs(x - t) for t in targets])) != g_idx
                        for x in block[np.isfinite(block)]))
    if md["n_modes"] >= 2 and left == 0:
        label = "bistable: each stable state holds under noise"
        text = (f"Runs started in each of the {len(stable)} stable ODE states stayed there for "
                f"the whole t_end: every state survives the noise at omega={om:g}, so the "
                "system is bistable and switching between the states is slower than t_end. "
                "(A smaller omega -- fewer molecules -- makes switching faster.)")
    elif md["n_modes"] >= 2:
        label = "bistable with switching"
        text = (f"Started in each stable state, {left} run(s) ended in another state: the "
                "states are bistable and noise switches between them.")
    else:
        label = "only one state survives noise"
        text = (f"Started in each stable ODE state, the runs ended in one peak: at omega={om:g} "
                "noise removes the other state(s).")
    dist_entries.append({"method": "SSA, started in each stable state", "omega": om,
                         "readout": entry["readout"], "series": series, "pooled": pooled,
                         "modes": md, "attractors": entry["attractors"], "label": label,
                         "burn_in": burn, "discrete": True, "states": np.vstack(full),
                         "_a_fn": a_fn, "state_names": names})
    return {"groups": groups, "n_peaks": md["n_modes"],
            "peaks_molecules": [{"at": round(m["at"], 4), "weight": round(m["weight"], 4)}
                                for m in md["modes"]],
            "runs_that_left_their_start_state": left, "switching": sw,
            "verdict": label, "explanation": text}


def _distribution(model, values, res, om, grid, keep, burn, names, ode, y0, wanted, info,
                  t_end, gap):
    """Pooled distribution(s), deterministic attractors and the verdict for one omega."""
    from dynalyst.analysis import stochastic_distribution as dist
    stoch = res.get("ssa") if res.get("ssa") is not None else res.get("cle")
    samples = stoch[:, keep, :]
    if wanted:
        readouts = [(w, "requested") for w in wanted]
    else:
        readouts = [dist.choose_readout(model, samples, names)]
    auto_readout = not wanted and readouts[0][1] != "the XPP file's autovar"
    # ODE states from where the runs went (concentrations), the ODE path and the visited box
    rng = np.random.default_rng(7)
    flat = samples.reshape(-1, samples.shape[-1])
    flat = flat[np.all(np.isfinite(flat), axis=1)]
    reference = ode if ode is not None else np.asarray([y0])
    starts = [y0, reference[-1]] + list(reference[:: max(1, len(reference) // 10)])
    if flat.size:
        starts += list(flat[rng.choice(len(flat), min(25, len(flat)), replace=False)])
    hi = np.nanmax(np.vstack([np.abs(reference), np.abs(flat) if flat.size else np.abs(reference)]), axis=0)
    hi = np.where(hi > 0, hi, 1.0)
    starts += [rng.uniform(0, 1.3, len(names)) * hi for _ in range(60)]
    a_check = _propensity_fn(model, info["_channels"], values, 1.0)

    def physical(y):
        a = a_check(np.asarray(y, dtype=float))
        return bool(np.all(a >= -1e-7 * (1 + np.max(np.abs(a)))))
    summary = {"readouts": [], "by_method": {}}
    entries = []
    for rname, why in readouts:
        j = names.index(rname)
        if flat.size:
            starts += [flat[np.argmin(flat[:, j])], flat[np.argmax(flat[:, j])]]
        scale = float(np.nanmax(np.abs(reference[:, j])))
        search = {}
        attractors, neg_stable = (dist.ode_attractors(model, values, starts, t_end, j,
                                                     max(scale, 1e-12), physical, report=search)
                                  if ode is not None else ([], []))
        if search:
            summary["attractor_search"] = search
        summary["deterministic_stability_assessed"] = ode is not None
        if auto_readout:       # bistable: read out the species that tells the states apart
            better = dist.readout_from_states(attractors, names, rname)
            if better != rname:
                rname, j = better, names.index(better)
                why = "separates the stable ODE states most"
                for a in attractors:
                    if a["kind"] == "steady state":
                        a["readout"] = float(a["state"][j])
                scale = float(np.nanmax(np.abs(ode[:, j]))) if np.size(ode) else 1.0
        summary["readouts"].append({"species": rname, "chosen_by": why})  # final choice
        summary["ode_attractors"] = [
            {k: v for k, v in a.items() if k != "state"} for a in attractors]
        for method in ("ssa", "cle"):
            if res.get(method) is None:
                continue
            series = res[method][:, :, j] * om                 # molecules
            pooled = series[:, keep].ravel()
            md = dist.modes(pooled, discrete=(method == "ssa"))
            if md is None:
                continue
            sw = dist.switching(series[:, keep], md, gap)
            absorbed = (info.get("ssa", {}).get("absorbed_runs", 0)
                        if method == "ssa" and isinstance(info.get("ssa"), dict) else 0)
            st_r = [a["readout"] * om for a in attractors
                    if a["kind"] == "steady state" and a["stability"] == "stable"]
            ends = None
            if len(st_r) > 1:
                last = series[:, -1]
                last = last[np.isfinite(last)]
                near = [int(np.argmin([abs(x - r_) for r_ in st_r])) for x in last]
                ends = {f"{rname}≈{st_r[i]:.4g}": near.count(i) for i in range(len(st_r))}
                if sw is None:      # one pooled peak can hide runs that moved: judge them by the ODE states
                    at = sorted(st_r)
                    moved = dist.switching(series[:, keep], {"modes": [{"at": x} for x in at],
                                                             "valleys": [(a + b) / 2 for a, b in zip(at, at[1:])]}, gap)
                    sw = moved if moved["switches_total"] else None
            if ode is None:
                label = f"empirical distribution: {md['n_modes']} mode(s)"
                lines = ["These are finite-window samples; stationarity and deterministic stability were not assessed."]
            else:
                label, lines = dist.verdict(md["n_modes"], attractors, absorbed, series.shape[0],
                                            sw, rname, neg_stable, ends, search)
            if method == "ssa" and res.get("ssa_stopped_early"):
                label = "insufficient data: the runs stopped early"
                lines = [res["ssa_stopped_early"]["note"]]
            display_method = ("TAU-LEAPING" if method == "ssa" and res.get("ssa_label") == "tau-leaping"
                              else method.upper())
            summary["by_method"][display_method] = {
                "readout": rname, "n_peaks": md["n_modes"],
                "peaks_molecules": [{"at": round(m["at"], 4), "weight": round(m["weight"], 4)}
                                    for m in md["modes"]],
                "peaks_concentration": [round(m["at"] / om, 6) for m in md["modes"]],
                "switching": sw, "verdict": label, "explanation": lines}
            entries.append({"method": display_method, "omega": om, "readout": rname,
                            "series": series, "pooled": pooled, "modes": md,
                            "attractors": attractors, "label": label, "burn_in": burn,
                            "discrete": method == "ssa"})
    return {"summary": summary, "entries": entries}


INSUFFICIENT = "insufficient data: the first run stopped before t_end"
# when a period read from noisy runs is trusted: chosen on small oscillators, not derived
PERIOD_MIN_PEAKS = 5
PERIOD_MAX_CV = 0.2


def _period_across_runs(runs, grid) -> dict:
    """The period over every run (runs x time), a stopped run up to where it stopped, with whether
    it can be trusted: low copy numbers can give a period several times off, with few peaks or
    irregular intervals."""
    finished = [r for r in runs if not np.isnan(r).any()]
    usable = [r for r in runs if np.isfinite(r).sum() > 3]
    found = [p for p in (_regular_peaks(r, grid) for r in usable) if p]
    if not found:
        return {"runs_finished": len(finished), "runs_with_regular_peaks": 0, "reliable": False,
                "reason": "no run has regular peaks" if usable else "no run finished"}
    peaks = float(np.median([p["n_peaks"] for p in found]))
    cv = float(np.median([p["period_cv"] for p in found]))
    half = len(found) * 2 >= len(usable)
    reliable = bool(peaks >= PERIOD_MIN_PEAKS and cv <= PERIOD_MAX_CV and half)
    return {"median_period": float(np.median([p["mean_period"] for p in found])),
            "period_range": [float(min(p["mean_period"] for p in found)), float(max(p["mean_period"] for p in found))],
            "runs_finished": len(finished), "runs_used": len(usable), "runs_with_regular_peaks": len(found),
            "median_peaks": peaks, "median_period_cv": round(cv, 3), "reliable": reliable,
            **({} if reliable else {"reason": (
                "low copy numbers: too few peaks, irregular intervals, or few runs with regular peaks. "
                "A larger omega reduces the noise; do not report this as the period.")})}


def _species_stats(res, ode, grid, names, om):
    """Late-time statistics (last quarter of the window) per species."""
    late = grid >= grid[-1] * 0.75
    stats = {}
    for j, n in enumerate(names):
        s = {"ode_final": float(ode[-1, j])} if ode is not None else {}
        per = _peaks(ode[:, j], grid) if ode is not None else None
        if per:
            s["ode_oscillation"] = per
        if per:
            s["note"] = ("oscillates: the late-time mean / std below include the oscillation "
                         "itself, not only noise")
        if res.get("ssa") is not None:
            v = res["ssa"][:, late, j]
            mean, std = float(np.nanmean(v)), float(np.nanstd(v))
            s.update(ssa_mean=mean, ssa_std=std,
                     ssa_cv=(std / mean if mean > 0 else None),
                     ssa_fano_molecules=(float(np.nanvar(v * om) / (mean * om)) if mean > 0 else None))
            if ode is not None:
                s["ssa_vs_ode_mean_rel_diff"] = _rel(mean, float(np.mean(ode[late, j])))
            if s.get("ode_oscillation"):      # noise alone makes peaks: compare only then
                run = res["ssa"][0, :, j]
                s["ssa_run1_oscillation"] = (INSUFFICIENT if np.isnan(run).any() else
                                             _regular_peaks(run, grid) or "no regular oscillation")
                s["ssa_oscillation"] = _period_across_runs(res["ssa"][:, :, j], grid)
            two, b = _two_modes(res["ssa"][:, -1, j])
            if b is not None:
                s["ssa_final_bimodality_coefficient"] = b
                s["ssa_final_two_modes"] = two
        if res.get("cle") is not None:
            v = res["cle"][:, late, j]
            s.update(cle_mean=float(np.nanmean(v)), cle_std=float(np.nanstd(v)))
        stats[n] = s
    return stats


def _rel(a, b):
    return float(abs(a - b) / max(abs(b), 1e-12))


def _interpret(summary, names):
    lines = []
    for info in summary:
        om = info["omega"]
        for n in names:
            s = info["species"][n]
            if "ssa_cv" in s and s["ssa_cv"] is not None and not s.get("ode_oscillation"):
                comparison = (f", SSA mean {100 * s['ssa_vs_ode_mean_rel_diff']:.1f}% from the ODE"
                              if "ssa_vs_ode_mean_rel_diff" in s else "")
                lines.append(f"Ω={om:g}, {n}: SSA noise CV {s['ssa_cv']:.3f}{comparison}.")
            o, q, all_runs = s.get("ode_oscillation"), s.get("ssa_run1_oscillation"), s.get("ssa_oscillation") or {}
            if o:
                line = f"Ω={om:g}, {n}: ODE oscillates with period {o['mean_period']:.3g}"
                if all_runs.get("median_period"):
                    line += (f"; SSA period {all_runs['median_period']:.3g} over {all_runs['runs_with_regular_peaks']} "
                             f"of {all_runs['runs_used']} runs"
                             + ("" if all_runs["runs_finished"] == all_runs["runs_used"]
                                else ", some only up to where they stopped early")
                             + ("" if all_runs["reliable"] else ", NOT RELIABLE (low copy numbers)")
                             + " (stochastic runs drift "
                             "out of phase, so their average damps even though each run cycles)")
                elif q == INSUFFICIENT:
                    line += "; " + INSUFFICIENT + ", so nothing is concluded about SSA oscillation"
                elif q:
                    line += ("; no SSA run has regular peak spacing, so no SSA period is measured "
                             "(this does not say the runs stop cycling)")
                lines.append(line)

    for info in summary:
        for m, v in ((info.get("distribution") or {}).get("by_method") or {}).items():
            lines.append(f"Ω={info['omega']:g}, {m}: {v['verdict']} -- " + " ".join(v["explanation"]))
        e = (info.get("distribution") or {}).get("started_in_each_state") or {}
        if e.get("verdict"):
            lines.append(f"Ω={info['omega']:g}, SSA started in each stable state: {e['verdict']} -- "
                         + e["explanation"])
    if len(summary) > 1:
        lines.append("Noise falls roughly as 1/sqrt(Ω): compare the CV across Ω values.")
    return lines[:40]


def _sample(results, grid, names, n=20):
    step = max(1, len(grid) // n)
    out = []
    for res in results:
        for k in range(0, len(grid), step):
            row = {"omega": res["omega"], "time": float(grid[k])}
            for j, s in enumerate(names):
                if res.get("ode") is not None:
                    row[f"{s}_ode"] = float(res["ode"][k, j])
                if res.get("ssa") is not None:
                    row[f"{s}_ssa_mean"] = float(np.nanmean(res["ssa"][:, k, j]))
                if res.get("cle") is not None:
                    row[f"{s}_cle_mean"] = float(np.nanmean(res["cle"][:, k, j]))
            out.append(row)
    return out


# --- tool definitions ---------------------------------------------------------------

_MODEL_ARGS = {
    "model_name_or_path": {"type": "string", "description": (
        "SBML file / name, XPP .ode/.txt file, or BioModels ID. Leave out when typing ODEs.")},
    "species": {"type": "array", "items": {"type": "string"},
                "description": "Typed-ODE models only: species names in order."},
    "odes": {"type": "object", "description": "Typed-ODE models only: {species: right-hand side}."},
    "parameters": {"type": "object", "description": (
        "{name: value}: changed parameter values for this run (for typed ODEs: their values).")},
}

TOOL_DEFINITIONS = [
    {"name": "stochastic_simulation",
     "description": (
         "Stochastic simulation only: Gillespie SSA by default; CLE only when requested. "
         "Never runs an ODE or deterministic attractor search. Reports molecule counts, noise "
         "statistics and empirical distributions with modelling assumptions. Use "
         "compare_simulations for an explicit deterministic comparison."),
     "input_schema": {"type": "object", "properties": {
         **_MODEL_ARGS,
         "t_end": {"type": "number", "description": ("End time (model time units): only when the task "
                                                    "gives one, exactly. Left out, t_end = 100. Never "
                                                    "choose or change it.")},
         "omega": {"type": "array", "items": {"type": "number"},
                   "description": ("System size(s) Omega, molecules per model state unit: pass it "
                                   "only when the task names one (exactly that value), or several "
                                   "the task names to compare. Left out, omega = 1. Never choose one.")},
         "methods": {"type": "array", "items": {"type": "string", "enum": ["ssa", "cle"]},
                     "description": "Default ['ssa']. Use ['cle'] for CLE only; ['ssa','cle'] only when both are requested."},
         "n_runs": {"type": "integer", "description": "Stochastic runs per method and omega (default 20)."},
         "n_points": {"type": "integer", "description": "Output time points (default 400)."},
         "dt": {"type": "number", "description": "CLE time step (default t_end / (20 n_points))."},
         "seed": {"type": "integer"},
         "initial_conditions": {"type": "object", "description": "{species: initial value in the model's state units}"},
         "plot_species": {"type": "array", "items": {"type": "string"},
                          "description": "Species to plot (default the first four)."},
         "export_matlab": {"type": "boolean", "description": "Write MATLAB code for the selected methods. Tau-leaping export is unsupported."},
         "record_dt": {"type": "number", "description": "Sampling interval of the recorded time course (sets n_points)."},
         "burn_in": {"type": "number", "description": "Time discarded before pooling the distribution (default 10% of t_end)."},
         "distribution": {"type": "boolean", "description": "Empirical pooled distribution (default true); peaks do not establish stationarity or deterministic stability."},
         "distribution_species": {"type": "array", "items": {"type": "string"},
                                  "description": "Read-out species for the distribution (default: the XPP autovar, else the most variable species)."},
         "start_in_each_state": {"type": "boolean", "description": (
             "When the ODE has several stable states but every run stayed in one, also start "
             "runs in each stable state and show whether each holds under noise. Comparison tool only; default false.")},
         "ssa_method": {"type": "string", "enum": ["auto", "exact", "tau_leap"],
                        "description": ("auto: exact SSA (numpy when it fits the time budget, else JAX if installed, "
                                        "else skipped with an estimate); exact: as auto but numpy runs even when long; "
                                        "tau_leap: approximate tau-leaping.")}},
         "required": []}},
    {"name": "propensity_table",
     "description": (
         "The model written as reactions for stochastic simulation: each reaction with its "
         "stoichiometry, its propensity w = omega * rate(X/omega) and parameter values (the "
         "table that defines the Gillespie SSA), the stoichiometric matrix V, and optionally a "
         "MATLAB script, and the value of every propensity at the initial state (or at a given "
         "state) for the given omega. No simulation is run."),
     "input_schema": {"type": "object", "properties": {
         **_MODEL_ARGS,
         "omega": {"type": "number", "description": "System size: molecules per concentration unit. "
                   "Used for the propensity values (default 1) and the MATLAB script (default 1000)."},
         "state": {"type": "object", "description": "Optional {species: concentration} at which to "
                   "evaluate the propensities; default the model's initial state."},
         "export_matlab": {"type": "boolean"}},
         "required": []}},
]


_comparison_definition = deepcopy(TOOL_DEFINITIONS[0])
_comparison_definition["name"] = "compare_simulations"
_comparison_definition["description"] = (
    "Explicit comparison of deterministic ODE and stochastic trajectories using the same "
    "canonical model, parameters and initial conditions. Default ['ode','ssa']; add 'cle' "
    "only when requested. Coordinates separate solvers and reports each backend and status. "
    "Can compare empirical distributions with deterministic attractors.")
_comparison_definition["input_schema"]["properties"]["methods"] = {
    "type": "array", "items": {"type": "string", "enum": ["ode", "ssa", "cle"]},
    "description": "Default ['ode','ssa']; requires ode plus ssa and/or cle."}
TOOL_DEFINITIONS.insert(1, _comparison_definition)

TOOL_FUNCTIONS = {"stochastic_simulation": stochastic_simulation,
                  "compare_simulations": compare_simulations,
                  "propensity_table": propensity_table}


def _note_switching(out, info):
    """The switching plot found jumps: say so plainly (direction, how many runs, first-jump times),
    and drop any line elsewhere in the result that says no run left its first state."""
    import numpy as _np
    jumps = info["jumps"]                   # capped at 50; the counts below cover every jump
    first = {}
    for e in jumps:
        first.setdefault(e["run"], e["t_leave"])
    times = _np.array(info.get("first_jump_t_per_run") or list(first.values()), float)
    up = info.get("up", sum(e["to_peak"] > e["from_peak"] for e in jumps))
    n = info.get("jumps_in_all_runs", len(jumps))
    line = (f"Switching (omega = {info['omega']:g}, {info['readout']}): {n} jump(s) between stable states in "
            f"{len(times)} run(s): {up} to the higher state, {n - up} to the lower. "
            f"First jump per run: median t = {float(_np.median(times)):.4g}, range {float(times.min()):.4g} to "
            f"{float(times.max()):.4g} (time at which the run left its first state).")
    out["switching_summary"] = line
    bad = "no run left the first state"
    if isinstance(out.get("interpretation"), str):
        kept = [x for x in out["interpretation"].split("\n") if bad not in x]
        out["interpretation"] = "\n".join(kept + [line])
    elif isinstance(out.get("interpretation"), list):
        out["interpretation"] = [x for x in out["interpretation"] if bad not in str(x)] + [line]
    # runs did switch, so "deterministic only" (no switching seen) no longer holds at this omega
    seen = "bistable (switching seen; one peak in the distribution)"
    for v in out.get("stability_verdicts") or []:
        if bad in str(v.get("verdict", "")):
            v["verdict"] = line
        elif "deterministic only" in str(v.get("verdict", "")) and v.get("omega") == info["omega"]:
            v["verdict"] = seen
    for po in out.get("per_omega") or []:
        d = po.get("distribution") or {}
        for m, v in (d.get("by_method") or {}).items():
            if isinstance(v, dict) and bad in str(v.get("explanation", "")) + str(v.get("verdict", "")):
                v["explanation"] = [x for x in (v.get("explanation") or []) if bad not in str(x)] + [line]
                if "deterministic only" in str(v.get("verdict", "")):
                    v["verdict"] = seen
