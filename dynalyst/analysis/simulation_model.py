"""One model and override contract for deterministic and stochastic simulation."""
import hashlib
import json
import math

from dynalyst.analysis.ode_model import (canonical_param, model_from_odes, model_from_xpp,
                                       relative_value, resolve_model_file,
                                       unknown_parameter_error)


def load_model(model_name_or_path=None, species=None, odes=None, parameters=None,
               initial_conditions=None):
    if model_name_or_path:
        path = resolve_model_file(str(model_name_or_path))
        if path.suffix.lower() in (".ode", ".txt"):
            model = model_from_xpp(path)
        else:
            from dynalyst.analysis.generic_sbml import build_symbolic_model
            model = {**build_symbolic_model(path, keep_time=True, allow_delay=True),
                     "source": "sbml", "source_path": str(path), "label": path.stem}
    else:
        model = model_from_odes(species, odes, parameters, initial_conditions)
    return apply_values(model, parameters if model_name_or_path else None, initial_conditions)


def apply_values(model, parameters=None, initial_conditions=None):
    """Copy, resolve aliases, and apply relative overrides without mutating a model."""
    values = dict(model["param_defaults"])
    changed = {}
    for name, value in (parameters or {}).items():
        key = canonical_param(model, name)
        if key is None:
            raise ValueError(json.dumps(unknown_parameter_error(model, name)))
        used = relative_value(value, values[key])
        changed[key] = (values[key], used)
        values[key] = used
    start = dict(model["y0"])
    unknown = set(initial_conditions or {}) - set(model["state_names"])
    if unknown:
        raise ValueError(f"Unknown species {sorted(unknown)}. Species: {model['state_names']}")
    for name, value in (initial_conditions or {}).items():
        start[name] = relative_value(value, start[name])
    for name, value in [*values.items(), *start.items()]:
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must have a finite value.")
    return {**model, "y0": start}, values, changed


def provenance(model, values):
    """Identify the equations and numerical inputs used by either solver."""
    contract = {"species": model["state_names"],
                "rhs": [str(e) for e in model.get("canonical_rhs", model["rhs"])],
                "parameters": {k: float(v) for k, v in values.items()},
                "initial_conditions": {k: float(v) for k, v in model["y0"].items()},
                "state_metadata": model.get("state_metadata", {}),
                "compartments": model.get("compartments", {}),
                "events": json.loads(json.dumps(model.get("events", []), default=str))}
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return {"model_signature": signature, "model_source": model.get("source"),
            "model_path": model.get("source_path"), **contract}


def selected_methods(methods, comparison):
    default = ["ode", "ssa"] if comparison else ["ssa"]
    if methods is None:
        return default
    if not isinstance(methods, (list, tuple)) or not methods or any(not isinstance(m, str) for m in methods):
        raise ValueError("methods must be a nonempty list.")
    methods = list(dict.fromkeys(m.lower() for m in methods))
    allowed = {"ode", "ssa", "cle"} if comparison else {"ssa", "cle"}
    if set(methods) - allowed:
        raise ValueError("stochastic_simulation accepts only 'ssa' and 'cle'; use compare_simulations for an ODE reference."
                         if not comparison else "Use 'ode', 'ssa' and/or 'cle'.")
    if comparison and ("ode" not in methods or not set(methods) & {"ssa", "cle"}):
        raise ValueError("compare_simulations requires 'ode' and at least one of 'ssa', 'cle'.")
    return methods


