"""The tools' own input checks (deterministic): run by tools.execute before a tool runs,
and reported as that tool's error, so the agent reads the reason and decides what to do.

  parameter   a continuation parameter (param_name / bif_param) must be one of the model's
              eligible parameters; the error lists the ones it has
  no change   values that are all the model's own change nothing; runs meant to be compared
              must differ

These are facts about the model and the arguments, never about the user's wording.
"""
_MODELS = {}     # model argument -> (file mtime, resolved model), for the name check


def _model_for(model_arg):
    from dynalyst.analysis.ode_model import resolve_model
    try:
        m = resolve_model(model_arg)
        from dynalyst.analysis.conservation_parameters import expose_totals
        m = expose_totals(m)
    except Exception:
        return None
    return m


def parameter_problem(args: dict) -> str | None:
    """A continuation parameter named in the call (param_name / bif_param) that the
    model does not have: rejected before anything runs, with the model's real
    parameters, so a guessed name (e.g. one from another model) costs no
    run. Also rejects constants unused by the dynamics. None when the parameter
    is eligible, or the model cannot be read here (the tool then reports why)."""
    name = args.get("param_name") or args.get("bif_param")
    model_arg = args.get("model") or args.get("model_name_or_path") or args.get("sbml_path")
    if not name or not model_arg or not isinstance(name, str) or not isinstance(model_arg, str):
        return None
    from pathlib import Path
    from dynalyst.analysis.ode_model import (canonical_param, continuation_parameter_error,
                                            parameter_listing)
    key = str(model_arg)
    try:
        from dynalyst.analysis.ode_model import resolve_model_file
        p = resolve_model_file(key)
        stamp = Path(p).stat().st_mtime
    except Exception:
        stamp = None
    if key not in _MODELS or _MODELS[key][0] != stamp:
        _MODELS[key] = (stamp, _model_for(key))
    model = _MODELS[key][1]
    if model is None:
        return None
    if canonical_param(model, name) is not None:
        problem = continuation_parameter_error(model, name)
        return (f"{problem['error']} Eligible parameters: {problem['available_parameters']}."
                if problem else None)
    if not model.get("param_names"):
        return (f"'{name}' is not a parameter of {model_arg}, and this model exposes no tunable "
                "quantity at all, so it has no continuation parameter.")
    return (f"'{name}' is not a parameter of {model_arg} (a species that changes over time "
            f"cannot be one). Its parameters: {parameter_listing(model, 40)}.")


def _cached_model(model_arg):
    from pathlib import Path
    key = str(model_arg)
    try:
        from dynalyst.analysis.ode_model import resolve_model_file
        stamp = Path(resolve_model_file(key)).stat().st_mtime
    except Exception:
        stamp = None
    if key not in _MODELS or _MODELS[key][0] != stamp:
        _MODELS[key] = (stamp, _model_for(key))
    return _MODELS[key][1]


def unchanged_value_problem(args: dict) -> str | None:
    """A run whose given values are all the model's own changes nothing, and runs meant to
    be compared that come out identical compare nothing: rejected, so an answer cannot
    report a change that was never made. Relative values ('*10', '/2') are applied by the
    tool and always count as a change."""
    model_arg = args.get("model") or args.get("model_name_or_path") or args.get("sbml_path")
    runs = [args] + [c for c in (args.get("conditions") or []) if isinstance(c, dict)]         + [{"initial_conditions": c} for c in (args.get("initial_conditions_list") or [])
           if isinstance(c, dict)]
    given = [{**(r.get("initial_conditions") or {}), **(r.get("parameters") or {}),
              **(r.get("parameter_overrides") or {})} for r in runs]
    given = [g for g in given if isinstance(g, dict)]
    if not isinstance(model_arg, str) or not any(given):
        return None
    model = _cached_model(model_arg)
    if not model:
        return None
    own = {**(model.get("param_defaults") or {}), **(model.get("y0") or {})}

    def changes(g):
        out = {}
        for k, v in g.items():
            if isinstance(v, str) or k not in own:
                out[k] = v
                continue
            try:
                if abs(float(v) - float(own[k])) > 1e-9 * max(1.0, abs(float(own[k]))):
                    out[k] = v
            except (TypeError, ValueError):
                out[k] = v
        return out
    made = [changes(g) for g in given]
    if any(made) and (len(runs) < 3 or len({repr(sorted(m.items())) for m in made[1:]}) > 1):
        return None
    same = {k: v for g in given for k, v in g.items() if k in own and k not in
            {kk for m in made for kk in m}}
    return (f"These values are the model's own ({same}), so {'these runs are identical' if len(runs) > 2 else 'this run changes nothing'}. "
            "For the model's own run leave them out. For a change relative to the model pass it "
            "relative, e.g. {'X': '*10'} for ten times, {'k': '/2'} for half (the tool applies it "
            "to the model's value); otherwise give the new value.")


def problem(tool: str, args: dict) -> str | None:
    if not isinstance(args, dict):
        return None
    return parameter_problem(args) or unchanged_value_problem(args)
