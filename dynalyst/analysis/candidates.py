"""Ranked bifurcation candidates: (parameter, range, plotted species, expected type).

For any SBML / XPP / ODE model: the model is integrated to where it settles,
then every parameter the equations use is followed a decade either side of its
value (parameter_screen.py) and the Jacobian eigenvalues are watched. Each
parameter whose eigenvalues cross zero becomes a candidate, ranked by how close
to its current value that happens, with
  * range     a window covering its value and the crossing (what AUTO would use),
  * species   the state that moves most as the parameter approaches the crossing,
  * expected  LP (fold; two folds = bistable window), LP/BP, or HB (Hopf).
Nothing found within a decade -> three decades are searched, then SCREEN_DECADES[-1].
This is the "ranked candidates" step the human approval shows before AUTO runs.
"""
import json

import numpy as np

SCREEN_DECADES = (1.0, 3.0, 5.0)    # searched in turn until a crossing is found

_EXPECTED = {"fold (equilibrium disappears)": "LP (fold)",
             "fold or branch point (real eigenvalue crosses 0)": "LP or BP",
             "Hopf (complex pair crosses 0)": "HB (Hopf: oscillations)"}


def _responsive_species(model, start, param, near):
    """The state that changes most (relative) between the parameter's value and
    just before the crossing."""
    from scipy.optimize import fsolve
    from dynalyst.analysis.steady_states import _lambdify_model, conservation_reduce
    model = conservation_reduce({**model, "y0": {**model["y0"], **start}})  # as the screen does
    start = model["y0"]
    rhs_fn, _ = _lambdify_model(model)
    names, pnames = model["state_names"], model["param_names"]
    base = [model["param_defaults"][p] for p in pnames]
    y0 = np.array([start[s] for s in names], dtype=float)
    v0 = base[pnames.index(param)]
    target = v0 + 0.9 * (near - v0)
    pv = list(base)
    y = y0.copy()
    with np.errstate(all="ignore"):
        for v in np.linspace(v0, target, 12)[1:]:     # follow the equilibrium in small steps
            pv[pnames.index(param)] = float(v)
            y, _, ier, _ = fsolve(lambda z: np.asarray(rhs_fn(*z, *pv), dtype=float), y,
                                  full_output=True)
            if ier != 1:
                return None
    change = np.abs(y - y0) / (np.abs(y0) + 1e-12)
    return names[int(np.argmax(change))] if np.all(np.isfinite(change)) else None


def propose(model_name_or_path: str, n_candidates: int = 5) -> dict:
    from dynalyst.analysis.bifurcation_auto import _window_around
    from dynalyst.analysis.ode_model import resolve_model
    from dynalyst.analysis.parameter_screen import screen_parameters, settled_state
    model = resolve_model(model_name_or_path)
    start, note = settled_state(model)
    screen, decades = [], None
    for decades in SCREEN_DECADES:
        screen = screen_parameters(model, start, decades=decades)
        if any(r["events"] for r in screen):
            break
    cands = []
    for r in screen:
        if not r["events"]:
            continue
        e = r["events"][0]
        lo, hi = _window_around(r["value"], [e["near"]])
        try:
            species = _responsive_species(model, start, r["parameter"], e["near"])
        except Exception:
            species = None
        coupled = bool(e.get("at_model_value") or e.get("runaway"))
        cands.append({
            "parameter": r["parameter"], "value": r["value"],
            "range": [round(lo, 10), round(hi, 10)],
            "y_species": species,     # None: the run picks the species that changes most
            "expected": (("no steady state past it -- a state runs away (grows or vanishes "
                          "without bound): a balance the model needs breaks, not a fold"
                          if e.get("runaway") else
                          "no steady state just off the model's value: likely a parameter the "
                          "model ties to others (change them together), not a bifurcation")
                         if coupled else _EXPECTED.get(e["type"], e["type"])),
            **({"not_a_bifurcation": True} if coupled else {}),
            "crossing_near": e["near"],
            "decades_from_value": (round(e["decades_from_value"], 3)
                                   if e["decades_from_value"] is not None else None),
            "reason": f"{e['type']} at {r['parameter']} ~ {e['near']:.4g} "
                      f"(model value {r['value']:.4g})"})
        if len(cands) >= n_candidates:
            break
    res = {"model": model_name_or_path, "starting_state": note,
           "searched": f"{decades:g} decade(s) either side of each parameter's value",
           "candidates": cands,
           "method": "eigenvalue screen along the settled equilibrium (analysis/parameter_screen.py); "
                     "ranked by closeness of the crossing to the model's own value"}
    if not cands:
        import sympy as sp
        used = {str(s) for e in model["rhs"] for s in sp.sympify(e).free_symbols}
        listed = [p for p in model["param_names"] if p in used]
        res["note"] = (f"No parameter shows an eigenvalue crossing within {decades:g} decades of "
                       "its value: the model may have no bifurcation nearby. The model's own "
                       "parameters are listed instead (unscreened, in the file's order); a wider "
                       "LHS scan or an explicit parameter and range can still be tried.")
        if len(listed) > n_candidates:
            res["other_parameters"] = listed[n_candidates:]
            res["note"] += (" The model's other parameters, any of which can be named: "
                            + ", ".join(listed[n_candidates:]) + ".")
        unscreened = []
        for p in listed:
            v = float(model["param_defaults"][p])
            lo, hi = (v / 10, v * 10) if v > 0 else (v - 1, v + 1)
            unscreened.append({
                "parameter": p, "value": v, "range": [lo, hi],
                "y_species": None, "expected": "unknown (no crossing found)",
                "screened": False, "reason": f"a parameter of the model ({(model.get('param_kinds') or {}).get(p, 'parameter')}); "
                                             "the screen found no crossing for it"})
        res["candidates"] += unscreened[:n_candidates]
        if unscreened[n_candidates:]:
            # not shown as numbered choices, but a reply naming one of them runs it
            res["other_candidates"] = unscreened[n_candidates:]
        if not model["param_names"]:
            res["note"] = "This model exposes no tunable quantity, so there is nothing to continue in."
    return res


def propose_bifurcation_candidates(model_name_or_path: str, n_candidates: int = 5) -> str:
    """Tool: ranked (parameter, range, plotted species, expected type) for a model."""
    try:
        return json.dumps(propose(model_name_or_path, n_candidates), indent=2, default=float)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})


def format_candidates(res: dict, species: bool = True) -> str:
    """The candidates as a numbered list for the user. species=False: no y-axis species (a
    one-diagram run picks it after the run, and a species listed here reads as requested)."""
    lines = []
    for i, c in enumerate(res.get("candidates") or [], start=1):
        rng = c["range"] if isinstance(c["range"], str) else f"[{c['range'][0]:.4g}, {c['range'][1]:.4g}]"
        plotted = c.get('y_species') or "the species that changes most"
        lines.append(f"{i}. {c['parameter']} over {rng}" + (f", plotting {plotted}" if species else "")
                     + f" -- expected {c['expected']}; {c['reason']}")
    return "\n".join(lines)


TOOL_DEFINITIONS = [
    {"name": "propose_bifurcation_candidates",
     "description": "Ranked candidates for a bifurcation analysis of a model: which parameter to "
                    "continue in, over which range, which species to plot, and which bifurcation "
                    "to expect (fold/bistability, branch point, Hopf), from an eigenvalue screen "
                    "of the model. Use for 'which parameter could make it bistable / oscillate?'.",
     "input_schema": {"type": "object", "properties": {
         "model_name_or_path": {"type": "string"},
         "n_candidates": {"type": "integer", "description": "Default 5."}},
         "required": ["model_name_or_path"]}},
]
TOOL_FUNCTIONS = {"propose_bifurcation_candidates": propose_bifurcation_candidates}
