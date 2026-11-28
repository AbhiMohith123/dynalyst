"""
LangChain tool wrapper around batch_lhs_runner.run_and_report, so the agent
can trigger an LHS parameter scan of a model.

Every call names its `model` (required): an SBML / XPP file or model name
(see regime/model_source.py).
Every range request -- percentages, folds, offsets, exact bounds, changed
baselines, fixed / vary-only parameters -- is resolved by range_spec.py.
"""
from dynalyst.regime.scan_docs import scan_doc
import json
from langchain.tools import tool
from dynalyst.regime import model_source
from dynalyst.regime.batch_lhs_runner import run_and_report
from dynalyst.regime.range_spec import extras_from_tool_args


def _parse_manual_bounds(manual_bounds_json):
    """manual_bounds_json: an optional JSON string like
    '{"p1": [50, 300], "p2": [1, 20]}' -- lets you hand-set exact LHS
    bounds for specific parameters directly from a chat query. Returns
    None if not given; raises a clear error (shown back to the LLM/user,
    not a silent failure) if the JSON is malformed."""
    if not manual_bounds_json:
        return None
    try:
        parsed = json.loads(manual_bounds_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"manual_bounds must be valid JSON like "
            f'\'{{"p1": [50, 300]}}\', got: {manual_bounds_json!r} ({e})'
        )
    return {k: tuple(v) for k, v in parsed.items()}


def _split_names(names):
    return [v.strip() for v in names.split(',')] if names else None


def _param_range(param_min, param_max):
    """The continuation window, or None (chosen by the scan) when not given or empty
    (min == max, e.g. 0 and 0 written for 'not given')."""
    if param_min is not None and param_max is not None and param_min == param_max:
        return None
    return (param_min, param_max) if (param_min is not None or param_max is not None) else None


def _width(value):
    """A spread, offset or relative window of 0 widens nothing: it means 'not given'."""
    return None if value is not None and value == 0 else value


def _extras(baseline_values, fixed_params, param_windows):
    return extras_from_tool_args(baseline_values, fixed_params, param_windows)


@tool
@scan_doc()
def lhs_bifurcation_scan(model: str, n_samples: int = 20, bif_param: str = None,
                          y_species: str = None, param_min: float = None,
                          param_max: float = None,
                          additive_offset: float = None, manual_bounds: str = None,
                          seed: int = None, spread_factor: float = None,
                          rel_window: float = None, allowed_bounds: str = None,
                          vary_only: str = None, baseline_values: str = None,
                          fixed_params: str = None, param_windows: str = None,
                          goal: str = None, separate: bool = False,
                          log_probes: bool = False) -> str:
    """MANY parameter sets (Latin hypercube) around a model's values, a full AUTO-07p
    bifurcation diagram for each, overlaid on ONE plot coloured by category (bistable,
    branch_point, hopf, single_fold, ultrasensitive, graded, incomplete, no_bifurcation), with a table
    of every set and its outcome. Any model file. Only when the user asks for sampling,
    LHS, N parameter sets or an overlay; ONE diagram is run_bifurcation_analysis."""
    report, _, _ = run_and_report(
        n_samples=n_samples, bif_param=bif_param, y_species=y_species,
        param_range=_param_range(param_min, param_max),
        additive_offset=_width(additive_offset), manual_bounds=_parse_manual_bounds(manual_bounds),
        seed=seed, model=model_source.require_model(model), spread_factor=_width(spread_factor),
        rel_window=_width(rel_window), allowed_bounds=_parse_manual_bounds(allowed_bounds),
        vary_only=_split_names(vary_only),
        extras=_extras(baseline_values, fixed_params, param_windows), goal=goal,
        separate=separate, log_probes=log_probes)
    return report
