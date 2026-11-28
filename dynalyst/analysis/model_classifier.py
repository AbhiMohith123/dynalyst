"""
Model classifier: how many dimensions does a model really have?

Reports the number of state variables, the linear conservation laws among
them (e.g. total enzyme or total substrate), and the resulting INDEPENDENT
dimension -- the number of state variables that can actually move once the
conserved totals are fixed. That number is what determines which dynamics
and bifurcations are possible at all:

  1D  -- only steady-state bifurcations (fold/LP, transcritical or pitchfork/BP);
         no Hopf, no oscillations. Bistability is possible.
  2D  -- additionally Hopf (HB) and limit cycles; phase-plane analysis
         applies; no chaos (Poincare-Bendixson).
  3D+ -- all of the above, plus higher-codimension and chaotic dynamics.

Works on any SBML file (including ones with assignment rules, via the LHS / AUTO
pipeline's rule-substituting parser) or a directly typed ODE
system. load_any_model() is also used by the reviewer to re-check results.
"""
import json
from pathlib import Path

import numpy as np
import sympy as sp

from dynalyst.analysis.ode_model import ode_strings, resolve_model
from dynalyst.analysis.steady_states import _detect_conservation_laws, _lambdify_model


def _model_from_regime_parser(model_name_or_path: str) -> dict:
    """SBML -> ode_model-style dict via the regime parser, which substitutes
    assignment rules (the research pipeline's reader rejects those)."""
    from dynalyst.analysis.generic_sbml import resolve_sbml_path
    from dynalyst.regime.parser import parse_sbml

    path = resolve_sbml_path(model_name_or_path)
    parsed = parse_sbml(str(path))
    state_names = list(parsed["species"])
    state_symbols = [sp.Symbol(s) for s in state_names]
    param_names = list(parsed["params"])
    param_symbols = [sp.Symbol(p) for p in param_names]
    rhs = [sp.sympify(parsed["ode_rhs"][s]) for s in state_names]
    unknown = set().union(*(e.free_symbols for e in rhs)) - set(state_symbols) - set(param_symbols)
    if unknown:
        raise ValueError(f"{path.name}: symbols {sorted(map(str, unknown))} have no value "
                         f"(not species or global parameters).")
    y0 = {s: float(parsed["initial_guess"].get(s, 0.0)) for s in state_names}
    return {
        "label": Path(path).stem, "state_names": state_names, "state_symbols": state_symbols,
        "param_names": param_names, "param_symbols": param_symbols,
        "param_defaults": {p: float(v) for p, v in parsed["param_values"].items()},
        "rhs": rhs, "y0": y0, "source": "sbml (assignment rules substituted)",
        "source_path": str(path),
    }


def load_any_model(model_name_or_path=None, species=None, odes=None, parameters=None,
                   initial_conditions=None, allow_delay: bool = False) -> tuple[dict, str]:
    """(model, reader) for any supported input. Tries the research reader
    first, then the rule-substituting parser for SBML the research reader rejects.
    allow_delay: a delay differential equation model is returned as written (for reading it)."""
    try:
        return resolve_model(model_name_or_path, species, odes, parameters,
                             initial_conditions, allow_delay=allow_delay), "research SBML/ODE reader"
    except (ValueError, KeyError, NotImplementedError) as first_error:
        if not model_name_or_path:
            raise
        try:
            return _model_from_regime_parser(model_name_or_path), "rule-substituting SBML parser"
        except Exception:
            raise first_error


def conservation_laws(model: dict, seed: int = 0) -> list[np.ndarray]:
    """Linear conserved quantities c.y, as float rows of the exact laws
    (steady_states.exact_conservation_laws)."""
    from dynalyst.analysis.steady_states import exact_conservation_laws
    return [np.array([float(x) for x in c]) for c in exact_conservation_laws(model)]


def _stoichiometric_classification(model_name_or_path: str) -> dict:
    """Fallback for SBML neither symbolic reader handles (e.g. SBML function
    definitions): species from jaxkineticmodel, conservation laws from the
    left null space of the stoichiometric matrix (valid for reaction-based
    models)."""
    import contextlib
    import io
    from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
    from dynalyst.analysis.generic_sbml import resolve_sbml_path

    path = resolve_sbml_path(model_name_or_path)
    with contextlib.redirect_stdout(io.StringIO()):
        m = SBMLModel(str(path))
        S = m._get_stoichiometric_matrix()
    names = list(S.index)
    from dynalyst.analysis.steady_states import exact_number, normalised_law
    vals = np.asarray(S.values, dtype=float)
    mat = sp.Matrix([[exact_number(x) for x in row] for row in vals]) if vals.size else None
    laws = [normalised_law(list(v)) for v in mat.T.nullspace()] if mat is not None else []
    return {"label": path.stem, "state_names": names, "laws": laws,
            "n_parameters": len(m.parameters)}


def _readable(c, names: list[str]) -> str:
    from dynalyst.analysis.steady_states import law_text
    return law_text(c, names)


IMPLICATIONS = {
    1: ("1D: only steady-state bifurcations are possible -- fold/saddle-node (LP) and "
        "transcritical or pitchfork branch points (BP). No Hopf bifurcation, no "
        "oscillations. Bistability (two stable states) is possible."),
    2: ("2D: in addition to LP/BP, Hopf bifurcations (HB) and limit cycles are possible; "
        "phase-plane analysis applies. No chaos (Poincare-Bendixson theorem)."),
}
IMPLICATIONS_HIGHER = ("{n}D: all of the above (LP, BP, HB, limit cycles) plus "
                       "higher-codimension and potentially chaotic dynamics.")


def _advice(n_states, n_laws, timescales) -> str:
    parts = []
    if n_laws:
        parts.append(f"{n_laws} conservation law(s) can be eliminated exactly "
                     f"({n_states}D -> {n_states - n_laws}D); the bifurcation and steady-state "
                     f"tools already do this internally")
    if timescales and timescales.get("qssa_recommended"):
        parts.append(f"a quasi-steady-state approximation on {timescales['fast_species']} is "
                     f"justified by the timescale gap; reduce_model can build and validate it")
    elif timescales and timescales.get("qssa_recommended") is False:
        parts.append("no timescale separation large enough for QSSA")
    return "; ".join(parts) or "no reduction indicated"


def classify_model(model_name_or_path: str | None = None, species: list[str] | None = None,
                   odes: dict | None = None, parameters: dict | None = None,
                   initial_conditions: dict | None = None) -> str:
    """Tool: classify a model by dimension (1D / 2D / 3D+), after removing
    linear conservation laws."""
    odes_out, timescales, delay_note = None, None, None
    try:
        model, reader = load_any_model(model_name_or_path, species, odes, parameters,
                                       initial_conditions)
        from dynalyst.analysis.steady_states import exact_conservation_laws
        laws = exact_conservation_laws(model)
        label, names, n_params = model["label"], model["state_names"], len(model["param_names"])
        odes_out = ode_strings(model)
        try:
            from dynalyst.analysis.model_reduction import _clean, timescale_analysis
            ts = timescale_analysis({**model, "rhs": [_clean(e) for e in model["rhs"]]},
                                    n_conservation_laws=len(laws))
            timescales = {k: ts.get(k) for k in ("qssa_recommended", "fast_species",
                                                 "slow_species", "spectral_gap", "reason")}
        except Exception as e:
            timescales = {"qssa_recommended": None, "reason": f"timescale analysis failed: {e}"}
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        if not model_name_or_path:
            return json.dumps({"error": str(e)})
        try:
            info = _stoichiometric_classification(model_name_or_path)
        except Exception as e2:
            return json.dumps({"error": f"{e}; jaxkineticmodel fallback also failed: {e2}"})
        reader = "jaxkineticmodel (conservation laws from the stoichiometric matrix)"
        from dynalyst.analysis.generic_sbml import DELAY_MESSAGE, DelayModel
        if isinstance(e, DelayModel):
            delay_note = ("This is a DELAY differential equation model: " + DELAY_MESSAGE
                          + " The count of state variables below holds, but what a dimension "
                          "allows does not: a delay equation can oscillate even in 1D.")
        label, names, n_params, laws = (info["label"], info["state_names"],
                                        info["n_parameters"], info["laws"])
    n_states = len(names)
    n_indep = n_states - len(laws)
    dim_class = f"{n_indep}D" if n_indep <= 3 else f"{n_indep}D (higher-dimensional)"
    return json.dumps({
        "model": label,
        "reader": reader,
        "n_state_variables": n_states,
        "state_variables": names,
        "n_conservation_laws": len(laws),
        "conservation_laws": [_readable(c, names) for c in laws],
        "independent_dimension": n_indep,
        "dimension_class": dim_class,
        "implications": (delay_note or IMPLICATIONS.get(n_indep,
                                                        IMPLICATIONS_HIGHER.format(n=n_indep))),
        **({"delay_differential_equation": True} if delay_note else {}),
        "n_parameters": n_params,
        "timescales": timescales,
        "reduction_advice": _advice(n_states, len(laws), timescales),
        "odes": odes_out,
    }, indent=2, default=float)


TOOL_DEFINITIONS = [{
    "name": "classify_model",
    "description": (
        "Classify a model by its dimension: counts state variables, detects linear "
        "conservation laws (e.g. total enzyme or substrate) and reports the INDEPENDENT "
        "dimension (1D / 2D / 3D+) plus which dynamics and bifurcations that dimension "
        "allows (e.g. no Hopf in 1D). Works for any SBML file (including ones with "
        "assignment rules) or a directly typed ODE system."),
    "input_schema": {"type": "object", "properties": {
        "model_name_or_path": {"type": "string", "description": "SBML model name or path."},
        "species": {"type": "array", "items": {"type": "string"}, "description": "For a typed ODE system."},
        "odes": {"type": "object", "description": "{species: rhs_expression} for a typed ODE system."},
        "parameters": {"type": "object", "description": "{parameter: value} for a typed ODE system."},
        "initial_conditions": {"type": "object", "description": "{species: value} for a typed ODE system."},
    }, "required": []},
}]

TOOL_FUNCTIONS = {"classify_model": lambda **kw: classify_model(**kw)}
