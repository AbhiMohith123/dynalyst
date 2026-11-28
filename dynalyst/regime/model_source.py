"""Which model an LHS / scan tool runs on: any SBML / XPP model file.

The sampling and categorisation of a scan need the parameters and states the model is
continued in. They are read from the project's shared ODE layer (analysis.ode_model) after
conservation laws are eliminated, as the AUTO runner does.
"""
from pathlib import Path

import sympy as sp

WINDOW_DECADES = 1.0        # default window: a decade either side of the baseline

_XPP_SUFFIXES = ('.ode', '.txt')
_CACHE = {}


def require_model(model) -> str:
    """The model file a scan runs on: always given (an SBML / XPP file)."""
    if not model:
        raise ValueError("model is required: name the SBML or XPP model file to scan.")
    return model


def _model_dict(model: str):
    """(shared ODE-layer model dict, source path) for an SBML path/name or an
    XPP .ode/.txt file."""
    from dynalyst.analysis import generic_sbml, ode_model, xpp_to_sbml
    path = ode_model.resolve_model_file(model)
    if path.suffix.lower() in _XPP_SUFFIXES:
        species, odes, parameters, initials = xpp_to_sbml.parse_xpp(path.read_text())
        if not species:
            raise ValueError(f"{path.name} has no derivative (NAME'=...) lines: not an XPP model.")
        return ode_model.model_from_odes(species, odes, parameters, initials, label=path.stem), path
    try:
        return ode_model.model_from_sbml(str(path)), path
    except generic_sbml.TimeDependentModel as e:
        raise generic_sbml.TimeDependentModel(f"{path.name} cannot be scanned: {e}") from None


def require_variation(resolution, bif_param, label):
    """Refuse a batch that would sample nothing: every sample would be the same
    parameter set, i.e. one diagram run n times. Says why."""
    if resolution.varied():
        return
    held = [p for p in resolution.bounds]
    if not held:
        raise ValueError(
            f"Nothing to sample in {label}: the only parameter its equations use is {bif_param}, "
            f"the continuation parameter, which is swept, not sampled. For its diagram run "
            f"run_bifurcation_analysis once; to sample, the model needs another parameter.")
    reasons = sorted({resolution.notes.get(p, 'held') for p in held})
    raise ValueError(
        f"Nothing to sample in {label}: every parameter other than {bif_param} (the "
        f"continuation parameter) is held -- {', '.join(held)} ({'; '.join(reasons)}). "
        f"Let at least one vary (vary_only / fixed_params / a window with room).")


def load_model(model: str, keep=None) -> dict:
    """{'label', 'path', 'full', 'parsed', 'eliminated'} for a generic model,
    cached per file version."""
    from dynalyst.analysis.steady_states import conservation_reduce
    full, path = _model_dict(model)
    from dynalyst.analysis.conservation_parameters import expose_totals
    full = expose_totals(full)
    key = (str(path.resolve()), path.stat().st_mtime, keep)
    if key not in _CACHE:
        red = conservation_reduce(full, keep=[keep] if keep else [])
        rhs = {s: sp.sympify(e) for s, e in zip(red['state_names'], red['rhs'])}
        used = {str(x) for e in rhs.values() for x in e.free_symbols}
        _CACHE[key] = {
            'label': full['label'], 'path': str(path), 'full': full,
            'eliminated': [name for name, _ in red.get('eliminated', [])],
            'parsed': {
                'species': list(red['state_names']),
                'params': [p for p in red['param_names'] if p in used],
                'param_values': dict(red['param_defaults']),
                'ode_rhs': rhs,
                'initial_guess': dict(red['y0']),
                'reconstruction': dict(red.get('eliminated', [])),
                'conserved_totals': full.get('conserved_totals', {}),
                'source_path': str(path),
            },
        }
    return _CACHE[key]


_SCREEN_CACHE = {}


def _screened_parameter(src):
    """The parameter the model screen ranks first (cached per model file version)."""
    path = Path(src.get('path'))
    key = (str(path), path.stat().st_mtime if path.exists() else None)
    if key not in _SCREEN_CACHE:
        _SCREEN_CACHE[key] = _screen_first(src)
    return _SCREEN_CACHE[key]


def _screen_first(src):
    """The parameter the eigenvalue screen ranks first (screened_choice)."""
    return screened_choice(src)[0]


def screened_choice(src):
    """(parameter, where its eigenvalue crosses) for the parameter the eigenvalue screen
    ranks first, searched as widely as the candidate list is (one decade either side, then
    wider while nothing is found); (None, None) when nothing crosses."""
    try:
        from dynalyst.analysis.candidates import SCREEN_DECADES
        from dynalyst.analysis.parameter_screen import screen_parameters, settled_state
        start, _ = settled_state(src['full'])
        for decades in SCREEN_DECADES:
            ranked = [(r['parameter'], r['events'][0].get('near'))
                      for r in screen_parameters(src['full'], start, decades=decades) if r['events']]
            pick = next(((p, near) for p, near in ranked if p in src['parsed']['params']), None)
            if pick:
                return pick
        return None, None
    except Exception:
        return None, None


def resolve_target(model, bif_param=None, y_species=None):
    """(bif_param, y_species) to use, validated against the model: bif_param as the
    user named it (a bare reaction-local name too), else the parameter the eigenvalue
    screen ranks first, else the first parameter the equations use (the file's order)
    -- discovered from the model, never guessed from a name; y_species else the first
    state variable (the same default run_bifurcation_analysis uses)."""
    require_model(model)
    from dynalyst.analysis.ode_model import NO_PARAMETERS, canonical_param, parameter_listing
    src = load_model(model, keep=y_species)
    parsed = src['parsed']
    if not parsed['params']:
        raise ValueError(NO_PARAMETERS if not src['full'].get('param_names') else
                         f"None of {src['label']}'s parameters appears in its equations, so "
                         "none can change its steady states.")
    if bif_param:
        bif = canonical_param(src['full'], bif_param)
        if bif not in parsed['params']:
            raise ValueError(f"'{bif_param}' is not a parameter the equations of {src['label']} "
                             f"use (a species that changes over time cannot be one); they are "
                             f"{parameter_listing({**src['full'], 'param_names': parsed['params']})}")
    else:
        bif = _screened_parameter(src) or parsed['params'][0]
    y = y_species or parsed['species'][0]
    if y not in parsed['species']:
        note = (f" ({y} was eliminated through a conservation law; ask for one of the others)"
                if y in src['eliminated'] else "")
        raise ValueError(f"'{y}' is not a state variable {src['label']} is continued in; they are "
                         f"{parsed['species']}{note}")
    return bif, y


def default_window(parsed, bif_param):
    """Continuation window for a generic model: a decade either side of the baseline value."""
    value = float(parsed['param_values'][bif_param])
    if value <= 0:
        return (0.1, 200.0)
    return (value / 10 ** WINDOW_DECADES, value * 10 ** WINDOW_DECADES)


def prepare(model, bif_param, y_species):
    """Parsed model plus the resolved target: {'label', 'parsed', 'bif_param', 'y_species', 'path'}."""
    src = load_model(model, keep=y_species)
    bif, y = resolve_target(model, bif_param, y_species)
    from dynalyst.analysis.ode_model import time_inputs_note
    return {'label': src['label'], 'parsed': src['parsed'], 'bif_param': bif, 'y_species': y,
            'path': src['path'], 'time_note': time_inputs_note(src['full'])}


def window_for(target, param_min=None, param_max=None):
    """(param_range, source) for a prepare() target and optional user limits."""
    lo, hi = default_window(target['parsed'], target['bif_param'])
    if param_min is None and param_max is None:
        return (lo, hi), f"one decade either side of {target['bif_param']}'s baseline"
    return ((param_min if param_min is not None else lo),
            (param_max if param_max is not None else hi)), "explicitly requested"


def identity_line(model) -> str:
    """One line saying which model a result is about, with the BioModels ID the
    file itself declares (or that it declares none), so an answer never borrows
    another model's ID from earlier in the conversation."""
    from dynalyst.analysis.annotations import model_identity
    try:
        ident = model_identity(model)
    except Exception:
        ident = None
    if not ident:     # an XPP file or typed ODEs
        return f"Model: {display(model)}; it has no BioModels ID -- do not attach one to it."
    if ident["biomodels_id"]:
        return (f"Model: {ident['file']} (\"{ident['model_name']}\"), BioModels ID "
                f"{ident['biomodels_id']} (declared in the file).")
    return (f"Model: {ident['file']} (\"{ident['model_name']}\"); the file declares no BioModels "
            f"ID -- do not attach one to it.")


def display(model):
    return Path(str(model)).name if model else "(no model given)"
