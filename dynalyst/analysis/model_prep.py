"""Model preparation for continuation, in one place: the model as asked for, its conserved
totals, which states must stay >= 0, an initial state consistent with both, and the model with
the conservation laws removed. Every rule about pools and signs lives here, so a fix reaches
every step that uses the prepared model."""
import numpy as np
import sympy as sp

from dynalyst.analysis.conservation_parameters import expose_totals
from dynalyst.analysis.ode_model import resolve_model
from dynalyst.analysis.steady_states import conservation_reduce, kept_nonnegative


def _keep_model_totals(model, own, initial_conditions, fixed_params) -> dict:
    """{total: {"model_value", "from_initial_conditions"}} for pools the given initial values
    would move. A pool keeps the model's own total unless every state in it is given or the
    total itself is passed."""
    kept = {}
    for n, d in (own.get("conserved_totals") or {}).items():
        given = (model.get("conserved_totals") or {}).get(n, {}).get("default")
        if (given is None or n in (fixed_params or {})
                or set(d["coefficients"]) <= set(initial_conditions)
                or np.isclose(given, d["default"], rtol=1e-9, atol=1e-12)):
            continue
        kept[n] = {"model_value": d["default"], "from_initial_conditions": given}
    return kept


def _admissible(model, params, fixed_initials, nonneg) -> dict:
    """The initial state moved onto the requested pool totals: given initial values stay, the
    other states of each pool adjust. Only states the model keeps >= 0 must stay >= 0 (an
    extracellular-calcium bookkeeping state may be negative)."""
    from scipy.optimize import lsq_linear
    definitions = model.get("conserved_totals") or {}
    if not definitions:
        return model
    names = model["state_names"]
    C = np.array([[float(sp.Rational(d["coefficients"].get(n, "0"))) for n in names]
                  for d in definitions.values()])
    totals = np.array([params[n] for n in definitions], dtype=float)
    signed = np.array([n in nonneg for n in names])
    # a pool of states that are all kept >= 0 (with coefficients >= 0) cannot have a negative total
    pool_signed = np.all((C == 0) | (signed & (C > 0)), axis=1)
    if not np.all(np.isfinite(totals)) or np.any(pool_signed & (totals < 0)):
        raise ValueError("Conserved pool totals must be finite, and nonnegative for pools of "
                         "species the model keeps nonnegative.")
    y = np.array([model["y0"][n] for n in names], dtype=float)
    if np.any(signed & (y < 0)):
        bad = [n for n, s, v in zip(names, signed, y) if s and v < 0]
        raise ValueError(f"Initial values of {bad} are negative, but the model keeps them >= 0.")
    if np.allclose(C @ y, totals, rtol=1e-9, atol=1e-10):
        return model
    fixed = {names.index(n) for n in (fixed_initials or {}) if n in names}
    free = [i for i in range(len(names)) if i not in fixed and np.any(C[:, i] != 0)]
    rhs = totals - (C[:, sorted(fixed)] @ y[sorted(fixed)] if fixed else 0.0)
    if free:
        lower = np.where(signed[free], 0.0, -np.inf)
        y[free] = lsq_linear(C[:, free], rhs, bounds=(lower, np.inf), tol=1e-12, max_iter=1000).x
    if not np.allclose(C @ y, totals, rtol=1e-8, atol=1e-9):
        raise ValueError("The requested conserved totals and fixed initial conditions have no "
                         "compatible state (states the model keeps >= 0 stay >= 0).")
    return {**model, "y0": dict(zip(names, map(float, y)))}


def sbml_species(path) -> set | None:
    """The ids of the SBML file's species (None when it cannot be read)."""
    try:
        import libsbml
        m = libsbml.readSBMLFromFile(str(path)).getModel() if path else None
        return {s.getId() for s in m.getListOfSpecies()} if m is not None else None
    except Exception:
        return None


def prepare(model_name_or_path, species=None, odes=None, parameters=None,
            initial_conditions=None, fixed_params=None, start_guess=None) -> dict:
    """{"full": model with totals as parameters and an admissible y0, "reduced": the model
    without its conservation laws, "nonneg": states kept >= 0, "totals_kept": pools the given
    initial values would have moved}. Raises ValueError / FileNotFoundError for a bad request."""
    fixed_params = fixed_params or {}
    model = expose_totals(resolve_model(model_name_or_path, species, odes, parameters,
                                        initial_conditions))
    totals_kept = {}
    if initial_conditions and model.get("conserved_totals"):
        own = expose_totals(resolve_model(model_name_or_path, species, odes, parameters, None))
        totals_kept = _keep_model_totals(model, own, initial_conditions, fixed_params)
        if totals_kept:
            model = {**model,
                     "param_defaults": {**model["param_defaults"],
                                        **{n: v["model_value"] for n, v in totals_kept.items()}},
                     "conserved_totals": {n: ({**d, "default": totals_kept[n]["model_value"]}
                                              if n in totals_kept else d)
                                          for n, d in model["conserved_totals"].items()}}
    if start_guess and not initial_conditions:
        # start where the system settles (parameter_screen.settled_state); given values stay
        model = {**model, "y0": {**model["y0"],
                                 **{k: v for k, v in start_guess.items() if k in model["y0"]}}}
    params = {**model["param_defaults"], **fixed_params}
    # typed equations may have variables of either sign; SBML species follow the model's rates
    from dynalyst.analysis.typed_equations import is_typed_file
    sbml = (str(model.get("source", "")).startswith("sbml")
            and not is_typed_file(model.get("source_path") or ""))
    nonneg = kept_nonnegative(model, params) if sbml else set()
    sbml_ids = sbml_species(model.get("source_path")) if sbml else None
    if sbml_ids and nonneg & sbml_ids:
        # only an SBML species is an amount or a concentration; a state set by a rate rule on
        # a parameter or compartment (a voltage, a signed deviation) may have either sign
        nonneg &= sbml_ids
    model = _admissible(model, params, initial_conditions, nonneg)
    return {"full": model, "reduced": conservation_reduce(model), "nonneg": nonneg,
            "totals_kept": totals_kept}
