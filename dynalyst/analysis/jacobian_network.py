"""The interaction network of a model, read from its Jacobian at each steady state.

For dx/dt = f(x, p) the Jacobian J = df/dx (v x u with v = u = number of species;
row i = the equation of species i, column j = the species it is differentiated
by) is evaluated at every steady state x*:

    J_ij = d f_i / d x_j        > 0   x_j ACTIVATES x_i   (edge j -> i, +)
                                < 0   x_j INHIBITS  x_i   (edge j -> i, -)
                                = 0   no direct effect
    J_ii                              self-regulation (usually < 0: removal)

From it:
  * the signed, directed, weighted network at that state, with the local response
    coefficients of modular response analysis r_ij = -J_ij / J_ii (how much x_i
    shifts per unit x_j, other species fixed) as edge weights;
  * the structural network: entries that are not identically zero in the symbolic
    Jacobian, i.e. every edge the equations allow at any state;
  * edges that differ between steady states (an edge can vanish or flip sign, e.g.
    between the two states of a switch);
  * feedback loops with their signs (Thomas: a positive loop is necessary for
    multistationarity, a negative loop of length >= 2 for sustained oscillation), self-loops
    (J_ii) included;
  * when the rates split into reactions, each entry (edge, self-regulation, loop step) as
    the sum of what each reaction adds to it: J = V^T dv/dx;
  * optionally the parameter Jacobian Jp = df/dp (v x number of parameters) and the
    steady-state sensitivities dx*/dp = -J^-1 Jp.

Steady states come from Newton from many starts (the visited states, the ODE path,
random states, and states spread over every conserved total), the same search the
stochastic distribution uses, or from a state the caller gives.

Exports per steady state: the Jacobian (CSV), the edge list, a network figure,
GraphML and SIF (Cytoscape), and one Excel workbook with everything.
"""
import json
import time
from pathlib import Path

import numpy as np
import sympy as sp

from dynalyst import config


def _out_dir(label):
    d = Path(config.PLOTS_DIR) / "jacobian_network" / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _short(e, n=120):
    s = sp.sstr(e)
    return s if len(s) <= n else s[: n - 3] + "..."


def _refine(model, values, y):
    """The steady state next to a given state: a state copied by a language model is
    rounded (437.73 for 437.7312...), so it is near a steady state, not on one. Newton from
    it, holding the conserved totals the given state has. None when the solver does not
    converge or lands farther than 1 % of the state's size away (another state)."""
    from scipy.optimize import least_squares
    from dynalyst.analysis.stochastic import _param_subs
    from dynalyst.analysis.model_classifier import conservation_laws
    X = model["state_symbols"]
    rhs = [sp.sympify(e).subs(_param_subs(model, values)) for e in model["rhs"]]
    f = sp.lambdify(X, rhs, "numpy")
    J = sp.lambdify(X, sp.Matrix(rhs).jacobian(X), "numpy")
    try:
        laws = conservation_laws({**model, "param_defaults": dict(values)})
    except Exception:
        laws = []
    C = np.array(laws, dtype=float) if len(laws) else np.zeros((0, len(X)))
    target = C @ y
    F = lambda z: np.concatenate([np.array(f(*z), dtype=float).ravel(), C @ z - target])
    jac = lambda z: np.vstack([np.array(J(*z), dtype=float), C])
    try:
        with np.errstate(all="ignore"):
            z = least_squares(F, y, jac=jac, xtol=1e-15, ftol=1e-15, gtol=1e-15).x
            residual = float(np.max(np.abs(F(z))))
    except Exception:
        return None
    size = 1.0 + float(np.max(np.abs(y)))
    if (not np.all(np.isfinite(z)) or residual > 1e-9 * size
            or float(np.max(np.abs(z - y))) > 0.01 * size):
        return None
    return z


def _steady_states(model, values, t_end, given):
    """[(state vector, stability, leading eigenvalue)] -- the given state, or all
    physical steady states found."""
    from dynalyst.analysis.stochastic import _ode
    from dynalyst.analysis.stochastic_distribution import ode_attractors
    names = model["state_names"]
    if given:
        unknown = [k for k in given if k not in names]
        if unknown:
            raise ValueError(f"Unknown species {unknown}. Species: {names}")
        y = np.array([float(given.get(n, model["y0"][n])) for n in names])
        from dynalyst.analysis.typed_equations import is_typed_file
        negative = [n for n, v in zip(names, y) if v < 0]
        if (negative and str(model.get("source", "")).startswith("sbml")
                and not is_typed_file(model.get("source_path") or "")):
            raise ValueError(
                f"The given state has negative values for {negative}: no SBML species can be "
                "negative, so this is not a state the system can be in (a solution of the "
                "equations listed under non_physical_solutions is not a steady state). Use a "
                "state from steady_states, or leave state out to use the model's own.")
        refined = _refine(model, values, y)
        if refined is not None:
            return [{"state": refined.tolist(), "stability": None, "given": True,
                     "refined_from": dict(zip(names, y.tolist()))}]
        return [{"state": y.tolist(), "stability": None, "given": True}]
    grid = np.linspace(0, t_end, 120)
    try:
        path = _ode(model, values, t_end, grid)
    except Exception:
        path = np.array([[model["y0"][n] for n in names]])
    y0 = np.array([model["y0"][n] for n in names], dtype=float)
    hi = np.nanmax(np.abs(np.vstack([path, y0[None, :]])), axis=0)
    hi = np.where(np.isfinite(hi) & (hi > 0), hi, 1.0)
    rng = np.random.default_rng(3)
    starts = [y0, path[-1]] + list(path[:: max(1, len(path) // 12)])
    starts += [rng.uniform(0, 1.5, len(names)) * hi for _ in range(80)]
    physical = None
    try:      # reject states with a negative rate term (e.g. a conserved total below zero)
        from dynalyst.analysis.stochastic import (_propensity_fn, reaction_channels,
                                                  sample_points)
        ch = reaction_channels(model, values, sample_points(model, values, t_end, path))
        a_fn = _propensity_fn(model, ch, values, 1.0)
        physical = lambda y: bool(np.all(a_fn(np.asarray(y, float))
                                         >= -1e-7 * (1 + np.max(np.abs(a_fn(np.asarray(y, float)))))))
    except Exception:
        pass
    found, _ = ode_attractors(model, values, starts, t_end, 0, float(np.max(hi)),
                              physical=physical, cycle_starts=0)
    return [a for a in found if a["kind"] == "steady state"]


def _loops(edges, names, max_len, diag=None, tol=0.0):
    import networkx as nx
    g = nx.DiGraph()
    g.add_nodes_from(names)
    for e in edges:
        g.add_edge(e["source"], e["target"], sign=1 if e["sign"] == "+" else -1)
    try:
        cycles = nx.simple_cycles(g, length_bound=max_len)
    except TypeError:                      # networkx < 3.1
        cycles = (c for c in nx.simple_cycles(g) if len(c) <= max_len)
    out = []
    for c in cycles:
        if len(out) >= 300:
            break
        if len(c) < 2:
            continue
        # networkx starts a cycle at any of its species (it varies between runs): start it at
        # the first in the model's order, so a loop has one name
        first = min(range(len(c)), key=lambda i: names.index(c[i]) if c[i] in names else len(names))
        c = c[first:] + c[:first]
        sign = 1
        for a, b in zip(c, c[1:] + c[:1]):
            sign *= g[a][b]["sign"]
        out.append({"loop": " -> ".join(c + [c[0]]), "length": len(c),
                    "sign": "positive" if sign > 0 else "negative"})
    # a species acting on its own rate (J_ii) is a loop of length 1: a positive one (e.g.
    # autocatalysis) counts for multistationarity like any positive loop
    for n, v in (diag or {}).items():
        if abs(v) > tol:
            out.append({"loop": f"{n} -> {n}", "length": 1,
                        "sign": "positive" if v > 0 else "negative"})
    out.sort(key=lambda x: (x["length"], x["loop"]))
    return out


def _reaction_split(model, values, t_end):
    """Per reaction: its stoichiometry and the derivative of its rate by each species, so
    that each Jacobian entry splits into the reactions that make it (J = V^T dv/dx). None
    when the model's rates cannot be split into reactions."""
    from dynalyst.analysis.stochastic import (_param_subs, _reaction_text, reaction_channels,
                                              sample_points)
    try:
        ch = reaction_channels(model, values, sample_points(model, values, t_end))
        if not ch:
            return None
        subs = _param_subs(model, values)
        X = model["state_symbols"]
        rates = sp.Matrix([sp.sympify(c["rate"]).subs(subs) for c in ch])
        dv = sp.lambdify(X, rates.jacobian(X), "numpy")
        V = np.array([c["stoich"] for c in ch], dtype=float)          # reactions x species
        labels = [c.get("label") or _reaction_text(model, c, values) for c in ch]
        return labels, V, dv
    except Exception:
        return None


def _parts_at(split, y, J):
    """parts[r, i, j]: what reaction r adds to J_ij at state y (None if they do not add
    up to J, e.g. a rate the split could not reproduce)."""
    if split is None:
        return None
    labels, V, dv = split
    try:
        D = np.array(dv(*y), dtype=float).reshape(len(labels), J.shape[0])
    except Exception:
        return None
    parts = V[:, :, None] * D[:, None, :]
    if not np.all(np.isfinite(parts)) or (
            np.max(np.abs(parts.sum(axis=0) - J)) > 1e-6 * (1 + float(np.max(np.abs(J))))):
        return None
    return parts


def _by_reaction(parts, labels, i, j, tol):
    """The reactions that make J_ij, largest first: [{reaction, part}]."""
    if parts is None:
        return None
    rs = [r for r in np.argsort(-np.abs(parts[:, i, j])) if abs(parts[r, i, j]) > tol]
    return [{"reaction": labels[r], "part": float(parts[r, i, j])} for r in rs[:6]]


def _plot(names, edges, self_reg, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import FancyArrowPatch
    g = nx.DiGraph()
    g.add_nodes_from(names)
    g.add_edges_from((e["source"], e["target"]) for e in edges)
    pos = (nx.circular_layout(g) if len(names) <= 8
           else nx.spring_layout(g, seed=1, k=1.5 / np.sqrt(len(names))))
    n = len(names)
    fig, ax = plt.subplots(figsize=(min(4 + 0.9 * n, 14), min(4 + 0.8 * n, 12)))
    w = np.array([abs(e["weight"]) for e in edges]) if edges else np.array([1.0])
    wmax = float(np.max(w)) if w.size and np.max(w) > 0 else 1.0
    for e in edges:
        a, b = pos[e["source"]], pos[e["target"]]
        both = any(x["source"] == e["target"] and x["target"] == e["source"] for x in edges)
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>" if e["sign"] == "+" else "|-|,widthA=0,widthB=0.35",
            mutation_scale=14, lw=0.8 + 3.2 * abs(e["weight"]) / wmax,
            color="tab:green" if e["sign"] == "+" else "tab:red",
            connectionstyle=f"arc3,rad={0.18 if both else 0.06}",
            shrinkA=18, shrinkB=18, alpha=0.85))
    for s in names:
        x, y = pos[s]
        reg = self_reg.get(s, 0.0)
        ax.scatter([x], [y], s=1100, color="white", zorder=3,
                   edgecolors="tab:red" if reg < 0 else "tab:green" if reg > 0 else "grey",
                   linewidths=2.2)
        ax.text(x, y, s, ha="center", va="center", fontsize=9, zorder=4)
    ax.set_title(title + "\ngreen arrow: activation (J_ij > 0)   red bar: inhibition (J_ij < 0)"
                 "   width: |r_ij|   node ring: sign of J_ii", fontsize=9)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.margins(0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def jacobian_network(model_name_or_path: str | None = None, species: list | None = None,
                     odes: dict | None = None, parameters: dict | None = None,
                     initial_conditions: dict | None = None, state: dict | None = None,
                     which: str = "all", t_end: float = 1000.0,
                     parameter_jacobian: bool = True, max_loop_length: int = 6) -> str:
    """Jacobian at each steady state and the signed interaction network it defines."""
    from dynalyst.analysis.stochastic import _load, _param_subs, _safe
    t0 = time.time()
    try:
        model, values, changed = _load(model_name_or_path, species, odes, parameters,
                                       initial_conditions)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        return json.dumps({"error": str(e)})
    names = model["state_names"]
    X = model["state_symbols"]
    subs = _param_subs(model, values)
    rhs = [sp.sympify(e) for e in model["rhs"]]
    J_sym = sp.Matrix(rhs).jacobian(X)
    J_num = sp.lambdify(X, J_sym.subs(subs), "numpy")
    structural = [[J_sym[i, j] != 0 for j in range(len(names))] for i in range(len(names))]
    P = [s for s, n in zip(model["param_symbols"], model["param_names"])
         if any(s in e.free_symbols for e in rhs)]
    Jp_num = (sp.lambdify(X, sp.Matrix(rhs).jacobian(P).subs(subs), "numpy")
              if parameter_jacobian and P else None)
    f_num = sp.lambdify(X, [e.subs(subs) for e in rhs], "numpy")
    from dynalyst.analysis.steady_states import exact_conservation_laws
    laws = exact_conservation_laws({**model, "param_defaults": values}) if Jp_num is not None else []

    try:
        states = _steady_states(model, values, float(t_end), state)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    try:
        from dynalyst.analysis.model_classifier import conservation_laws
        n_laws = len(conservation_laws({**model, "param_defaults": dict(values)}))
    except Exception:
        n_laws = 0
    none_stable_note = None
    if which == "stable":
        stable = [s for s in states if s.get("stability") in (None, "stable")]
        if states and not stable:     # the network exists all the same: shown, and said so
            none_stable_note = (f"None of the {len(states)} steady state(s) is stable; the "
                                "network is shown at the unstable one(s).")
        else:
            states = stable
    if not states:
        return json.dumps({"error": "No steady state found (or none stable). Give one with "
                                    "state={species: value}, or check the model and parameters."})

    label = _safe(model["label"])
    d = _out_dir(label)
    split = _reaction_split(model, values, float(t_end))
    results, all_edges = [], []
    for k, s in enumerate(states, 1):
        y = np.array(s["state"], dtype=float)
        J = np.array(J_num(*y), dtype=float)
        resid = float(np.max(np.abs(np.array(f_num(*y), dtype=float))))
        eig = np.linalg.eigvals(J)
        scale = max(1e-300, float(np.max(np.abs(J))))
        tol = 1e-9 * scale
        diag = {names[i]: float(J[i, i]) for i in range(len(names))}
        parts = _parts_at(split, y, J)
        labels = split[0] if parts is not None else None
        edges = []
        for i, tgt in enumerate(names):
            for j, src in enumerate(names):
                if i == j or abs(J[i, j]) <= tol:
                    continue
                r = -J[i, j] / J[i, i] if abs(J[i, i]) > tol else None
                edges.append({"source": src, "target": tgt, "J": float(J[i, j]),
                              "sign": "+" if J[i, j] > 0 else "-",
                              "local_response_r": None if r is None else float(r),
                              "weight": float(r if r is not None else J[i, j]),
                              "symbolic": _short(J_sym[i, j]),
                              **({"by_reaction": _by_reaction(parts, labels, i, j, tol)}
                                 if parts is not None else {})})
        edges.sort(key=lambda e: -abs(e["weight"]))
        loops = _loops(edges, names, int(max_loop_length), diag, tol)
        if parts is not None:
            # each step of a loop with the reactions that make it: the rate term a change
            # to the model has to touch to cut that loop
            at = {n: i for i, n in enumerate(names)}
            for lp in loops:
                nodes = lp["loop"].split(" -> ")
                lp["steps"] = [{"step": f"{a} -> {b}", "by_reaction": _by_reaction(
                    parts, labels, at[b], at[a], tol)} for a, b in zip(nodes, nodes[1:])]
        # each conserved total gives one zero eigenvalue: it does not decide stability
        free = eig[np.argsort(np.abs(eig))[n_laws:]]
        top = float(np.max(free.real)) if len(free) else 0.0
        stab = s.get("stability") or ("stable" if top < -1e-9 * scale else
                                      "unstable" if top > 1e-9 * scale else
                                      "marginal")
        tag = f"state{k}"
        np.savetxt(d / f"jacobian_{tag}.csv", J, delimiter=",",
                   header=",".join(names), comments="")
        png = d / f"network_{tag}.png"
        _plot(names, edges, diag, f"{model['label']}: steady state {k} ({stab})", png)
        _write_graph(names, edges, diag, d / f"network_{tag}")
        entry = {"index": k, "stability": stab, "given": bool(s.get("given")),
                 "state": dict(zip(names, [float(v) for v in y])),
                 **({"refined_from_given": s["refined_from"],
                     "refined_note": "the given state was rounded; this is the steady state "
                                     "next to it (same conserved totals)"}
                    if s.get("refined_from") else {}),
                 "residual_max_abs_dxdt": resid,
                 "eigenvalues": [{"re": float(e.real), "im": float(e.imag)}
                                 for e in sorted(eig, key=lambda z: -z.real)],
                 "n_edges": len(edges),
                 "edges": edges[:60],
                 "self_regulation": diag,
                 **({"self_regulation_by_reaction": {
                     n: _by_reaction(parts, labels, i, i, tol) for i, n in enumerate(names)}}
                    if parts is not None else {}),
                 "loops": {"n_positive": sum(l["sign"] == "positive" for l in loops),
                           "n_negative": sum(l["sign"] == "negative" for l in loops),
                           "list": loops[:40]},
                 "jacobian": J.tolist() if len(names) <= 25 else "see CSV",
                 "network_plot": config.display_path(png),
                 "jacobian_csv": config.display_path(d / f"jacobian_{tag}.csv")}
        if Jp_num is not None:
            entry["parameter_jacobian"] = _sensitivities(Jp_num, y, J, P, names, laws)
        if resid > 1e-6 * (1 + float(np.max(np.abs(y)))):
            entry["warning"] = (f"this state is not a steady state (max |dx/dt| = {resid:.3g}); "
                                "the network is the Jacobian at this point, not at equilibrium.")
        results.append(entry)
        all_edges += [{"state": k, **e} for e in edges]

    struct_edges = [{"source": names[j], "target": names[i], "symbolic": _short(J_sym[i, j])}
                    for i in range(len(names)) for j in range(len(names))
                    if i != j and structural[i][j]]
    diffs = _differences(results)
    xlsx = _excel(d / f"{label}_jacobian_network.xlsx", results, all_edges, struct_edges, names)
    out = {
        "model": model["label"], "species": names, "n_steady_states": len(results),
        "orientation": ("J[i][j] = d f_i / d x_j: row = species whose rate changes (target), "
                        "column = species that changes it (source); edge source -> target"),
        "steady_states": results,
        "structural_edges": struct_edges,
        "n_structural_edges": len(struct_edges),
        "differences_between_states": diffs,
        "interpretation": _interpret(results, diffs, names),
        "excel": config.display_path(xlsx) if xlsx else None,
        "output_dir": config.display_path(d),
        "plot_note": ("Real network plots were saved. Embed them as markdown: "
                      + " ".join(f"![steady state {r['index']}]({r['network_plot']})"
                                 for r in results)),
        "graph_files": "GraphML and SIF per state in output_dir (open in Cytoscape).",
        "runtime_s": round(time.time() - t0, 2),
    }
    if changed:
        out["parameters_changed"] = {k: {"model_value": a, "used": b} for k, (a, b) in changed.items()}
    if none_stable_note:
        out["note"] = none_stable_note
    return json.dumps(out, indent=2, default=float)


def _sensitivities(Jp_num, y, J, P, names, conservation_laws=()):
    Jp = np.array(Jp_num(*y), dtype=float).reshape(len(names), len(P))
    out = {"parameters": [str(p) for p in P],
           "Jp_nonzero": [{"species": names[i], "parameter": str(P[c]), "value": float(Jp[i, c])}
                          for i in range(len(names)) for c in range(len(P))
                          if abs(Jp[i, c]) > 1e-12 * (1 + np.max(np.abs(Jp)))][:80]}
    cond = np.linalg.cond(J)
    if not np.isfinite(cond) or cond > 1e12:
        # conserved totals make J singular: the steady state still moves with p, along the
        # surface where each total keeps its value: J dx = -Jp dp together with C dx = 0,
        # C contains structural conservation laws verified from the model RHS,
        # not a nullspace inferred from J at this particular state.
        C = np.asarray(conservation_laws, dtype=float).reshape(-1, len(names))
        A = np.vstack([J, C])
        if C.shape[0] == 0 or np.linalg.matrix_rank(A, tol=1e-10 * max(1.0, np.abs(A).max())) < len(names):
            out["sensitivities"] = None
            out["note"] = ("J is singular here and the conserved totals do not fix the state "
                           "(e.g. a bifurcation point), so dx*/dp is not defined at this state.")
            return out
        S = -np.linalg.lstsq(A, np.vstack([Jp, np.zeros((C.shape[0], Jp.shape[1]))]),
                             rcond=None)[0]
        residual = np.vstack([J @ S + Jp, C @ S])
        if not np.all(np.isfinite(S)) or np.max(np.abs(residual)) > 1e-8 * (1 + np.max(np.abs(Jp))):
            out["sensitivities"] = None
            out["note"] = "The constrained sensitivity equations are inconsistent; dx*/dp is undefined."
            return out
        out["sensitivities_note"] = (f"J is singular ({C.shape[0]} conserved total(s)): the "
                                     "sensitivities hold every conserved total at its value.")
    else:
        S = -np.linalg.solve(J, Jp)
    top = sorted(((abs(S[i, c]), names[i], str(P[c]), float(S[i, c]))
                  for i in range(len(names)) for c in range(len(P))), reverse=True)[:30]
    out["sensitivities_dxdp_top"] = [{"species": s, "parameter": p, "dx_dp": v}
                                     for _, s, p, v in top]
    return out


def _differences(results):
    if len(results) < 2:
        return []
    keys = {}
    for r in results:
        for e in r["edges"]:
            keys.setdefault((e["source"], e["target"]), {})[r["index"]] = e["sign"]
    out = []
    for (a, b), by in sorted(keys.items()):
        signs = {by.get(r["index"], "0") for r in results}
        if len(signs) > 1:
            out.append({"edge": f"{a} -> {b}",
                        "sign_per_state": {r["index"]: by.get(r["index"], "absent") for r in results}})
    return out


def _interpret(results, diffs, names):
    lines = []
    for r in results:
        lp = r["loops"]
        lines.append(f"State {r['index']} ({r['stability']}): {r['n_edges']} directed edges, "
                     f"{lp['n_positive']} positive and {lp['n_negative']} negative feedback "
                     "loop(s) (a species acting on its own rate counts as a loop of length 1).")
        if r["edges"]:
            e = r["edges"][0]
            lines.append(f"  strongest edge: {e['source']} {'activates' if e['sign'] == '+' else 'inhibits'} "
                         f"{e['target']} (J = {e['J']:.3g}"
                         + (f", r = {e['local_response_r']:.3g})" if e["local_response_r"] is not None else ")"))
    if any(r["loops"]["n_positive"] for r in results):
        lines.append("Positive feedback present: necessary (not sufficient) for multistability "
                     "(Thomas' rule).")
    if any(l["sign"] == "negative" and l["length"] >= 2 for r in results for l in r["loops"]["list"]):
        lines.append("Negative feedback of length >= 2 present: necessary (not sufficient) for "
                     "sustained oscillations.")
    if diffs:
        lines.append(f"{len(diffs)} edge(s) differ between the steady states (vanish or flip sign): "
                     + "; ".join(d["edge"] for d in diffs[:6]) + ".")
    elif len(results) > 1:
        lines.append("The steady states share the same signed network; only edge strengths differ.")
    return lines


def _write_graph(names, edges, diag, stem):
    import networkx as nx
    g = nx.DiGraph()
    for n in names:
        g.add_node(n, self_regulation=float(diag.get(n, 0.0)))
    for e in edges:
        g.add_edge(e["source"], e["target"], J=e["J"], sign=e["sign"],
                   weight=float(e["weight"]))
    nx.write_graphml(g, str(stem) + ".graphml")
    with open(str(stem) + ".sif", "w") as fh:
        for e in edges:
            fh.write(f"{e['source']}\t{'activates' if e['sign'] == '+' else 'inhibits'}\t{e['target']}\n")


def _excel(path, results, all_edges, struct_edges, names):
    try:
        import pandas as pd
        with pd.ExcelWriter(path) as xw:
            pd.DataFrame([{"state": r["index"], "stability": r["stability"], **r["state"]}
                          for r in results]).to_excel(xw, sheet_name="steady_states", index=False)
            pd.DataFrame(all_edges).drop(columns=["symbolic", "by_reaction"], errors="ignore").to_excel(xw, sheet_name="edges", index=False)
            pd.DataFrame(struct_edges).to_excel(xw, sheet_name="structural_edges", index=False)
            for r in results:
                if isinstance(r["jacobian"], list):
                    pd.DataFrame(r["jacobian"], index=names, columns=names).to_excel(xw, sheet_name=f"J_state{r['index']}")
                pd.DataFrame(r["loops"]["list"]).drop(columns=["steps"], errors="ignore").to_excel(xw, sheet_name=f"loops_state{r['index']}", index=False)
        return path
    except Exception:
        return None


TOOL_DEFINITIONS = [
    {"name": "jacobian_network",
     "description": (
         "The Jacobian J = df/dx at each steady state of a model and the signed, directed "
         "interaction network it defines (J_ij > 0: species j activates species i; < 0: "
         "inhibits), with local response coefficients r_ij = -J_ij/J_ii as edge weights, the "
         "structural network from the symbolic Jacobian, edges that differ between steady "
         "states, positive / negative feedback loops, and optionally the parameter Jacobian "
         "df/dp with steady-state sensitivities dx*/dp. Saves a network figure, the Jacobian "
         "CSV, GraphML / SIF for Cytoscape and an Excel workbook. Use for 'Jacobian', "
         "'interaction network', 'network from the Jacobian', 'who regulates whom', "
         "'feedback loops', 'sensitivity'."),
     "input_schema": {"type": "object", "properties": {
         "model_name_or_path": {"type": "string", "description": "SBML, XPP .ode/.txt or BioModels ID."},
         "species": {"type": "array", "items": {"type": "string"}, "description": "Typed ODEs only."},
         "odes": {"type": "object", "description": "Typed ODEs only: {species: rhs}."},
         "parameters": {"type": "object", "description": "{name: value} for this analysis."},
         "initial_conditions": {"type": "object"},
         "state": {"type": "object", "description": "Use this state {species: value} instead of searching."},
         "which": {"type": "string", "enum": ["all", "stable"], "description": "Steady states to analyse."},
         "t_end": {"type": "number", "description": "Time for the ODE path used to seed the search."},
         "parameter_jacobian": {"type": "boolean"},
         "max_loop_length": {"type": "integer"}},
         "required": []}},
]
TOOL_FUNCTIONS = {"jacobian_network": jacobian_network}
