"""An SBML file as species, parameters and right-hand sides."""


def parse_sbml(sbml_path):
    """The canonical symbolic SBML model (analysis/ode_model.py) as plain fields."""
    from dynalyst.analysis.ode_model import eligible_parameters, model_from_sbml
    model = model_from_sbml(sbml_path)
    params = eligible_parameters(model)
    return {
        'species': list(model['state_names']),
        'params': params,
        'param_values': dict(model['param_defaults']),
        'ode_rhs': dict(zip(model['state_names'], model['rhs'])),
        'initial_guess': dict(model['y0']),
    }
