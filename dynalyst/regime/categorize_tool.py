"""
LangChain tool wrapper around categorize_lhs.py: categorized_lhs_scan, an LHS scan of
a model split by category (one subplot panel and one Excel file per category).

For ONE plot of everything plus an optional per-category split, see
lhs_tool.lhs_bifurcation_scan (separate=True).
"""
from dynalyst.regime.scan_docs import scan_doc
from langchain.tools import tool
from dynalyst.regime.categorize_lhs import run_and_categorize
from dynalyst.regime import lhs_goals, model_source
from dynalyst.regime.lhs_tool import _parse_manual_bounds, _split_names, _param_range, _extras
from dynalyst.regime.batch_lhs_runner import summarize_windows


def _goal_text(goals, results, bif_param, y_species):
    text = lhs_goals.report(goals, results, bif_param, y_species,
                            window_text=summarize_windows(results))
    return f"\n{text}" if text else ""


@tool
@scan_doc(**{'separate': 'always on (this tool always splits by category).'})
def categorized_lhs_scan(model: str, n_samples: int = 20, bif_param: str = None,
                          y_species: str = None, param_min: float = None,
                          param_max: float = None,
                          additive_offset: float = None, manual_bounds: str = None,
                          seed: int = None, ultrasensitive_threshold: float = 81.0,
                          spread_factor: float = None,
                          rel_window: float = None, allowed_bounds: str = None,
                          vary_only: str = None, baseline_values: str = None,
                          fixed_params: str = None, param_windows: str = None,
                          goal: str = None, separate: bool = True,
                          log_probes: bool = False) -> str:
    """An LHS scan whose results are split by category: one subplot panel and one Excel file
    (with XPPAUT par lines) per category -- bistable, branch_point, hopf, single_fold,
    ultrasensitive, graded, incomplete, no_bifurcation; a sample appears under every feature it shows.
    Use when the user wants the results split by type; one overlay is lhs_bifurcation_scan."""
    model = model_source.require_model(model)
    bif_param, y_species = model_source.resolve_target(model, bif_param, y_species)
    goals, extras, control = lhs_goals.prepare(goal, _extras(baseline_values, fixed_params,
                                                             param_windows))
    plot_path, counts_plot_path, excel_paths, buckets, results, out_dir = run_and_categorize(
        n_samples=n_samples, bif_param=bif_param, y_species=y_species,
        param_range=_param_range(param_min, param_max), additive_offset=additive_offset, manual_bounds=_parse_manual_bounds(manual_bounds),
        seed=seed, ultrasensitive_threshold=ultrasensitive_threshold,
        model=model, spread_factor=spread_factor, rel_window=rel_window,
        allowed_bounds=_parse_manual_bounds(allowed_bounds), vary_only=_split_names(vary_only),
        extras=extras, control=control, log_probes=log_probes)
    counts = {cat: len(entries) for cat, entries in buckets.items()}
    excel_list = "\n".join(f"  {cat}: {path}" for cat, path in excel_paths.items())
    return (f"Categorized LHS scan complete ({n_samples} samples of {model}).\n"
            f"Continuation window: {summarize_windows(results)}\n"
            f"{results.identity}\n"
            f"{results.resolution.describe()}\n"
            f"Category counts: {counts}\n"
            f"Subplot figure: {_shown(plot_path)}\n"
            f"Category-counts bar chart: {counts_plot_path}\n"
            f"Per-category Excel files:\n{excel_list}"
            + _goal_text(goals, results, bif_param, y_species))


def _shown(path):
    """A file path as shown to the user: relative to the project when inside it."""
    from dynalyst import config
    return config.display_path(path) if path else path
