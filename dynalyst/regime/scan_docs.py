"""Descriptions of the LHS scan tools' arguments, written once.

The language model reads every tool's description on every call, so each scan tool's
description is its own purpose plus one line per argument it takes (scan_doc), built from
the entries below; the range rules shared by all of them are stated once at the end.
"""
import inspect

ARG_DOCS = {
    "model": "the model to scan (required): a model file or name.",
    "n_samples": "how many parameter sets to draw; each is a full AUTO run (10-30 to start).",
    "bif_param": "the continuation parameter, the same for every sample (never sampled). Left "
                 "out: the model's own default, else the one its eigenvalue screen picks. "
                 "Never a name from another model.",
    "y_species": "species on the y-axis. Left out: the model's own default.",
    "param_min": "lower end of the continuation range. Left out: automatic.",
    "param_max": "upper end of the continuation range. Left out: a decade above the baseline.",
    "spread_factor": "a fold f: each parameter log-uniform in baseline/f .. baseline*f.",
    "rel_window": "a fraction r: each parameter within +-r of its baseline, clipped to "
                  "allowed_bounds.",
    "additive_offset": "an absolute, one-sided window [baseline, baseline + offset]; only when "
                       "the user asks for it.",
    "manual_bounds": "JSON, exact bounds for some parameters, e.g. '{\"p1\": [50, 300]}'.",
    "allowed_bounds": "JSON, each parameter's full allowed range '{\"p\": [lo, hi]}'; "
                      "rel_window is then taken of its width and never leaves it.",
    "vary_only": "comma-separated parameters to vary; every other one stays at its baseline.",
    "seed": "random seed for reproducible samples.",
    "ultrasensitive_threshold": "response ratio separating ultrasensitive from graded samples "
                                "(default 81, the Goldbeter-Koshland value).",
    "baseline_values": "JSON, changed baselines that still vary, e.g. '{\"p2\": 12.5}'.",
    "fixed_params": "JSON, parameters held constant: '{\"p1\": 0.2}' (at that value) or "
                    "'[\"p1\", \"p2\"]' (at their baselines).",
    "param_windows": "JSON, a window per parameter overriding the global one: "
                     "'{\"p1\": {\"fold\": 1.5}, \"p2\": {\"rel\": 0.1}, \"p3\": {\"offset\": 5}, "
                     "\"p4\": {\"abs\": [10, 30]}}'.",
    "goal": "what the scan looks for: 'bistability', 'branch point', 'hopf' (needs >= 2 "
            "independent states), 'ultrasensitivity', or several. It sets the default window "
            "where nothing else is given, runs the baseline as a control and reports what was "
            "and was not found, with how to widen or narrow; it never searches further itself.",
    "log_probes": "true only when the user asks for log scaling or log-spaced probes: branches "
                  "off the start's are then looked for at log-spaced values of bif_param (one "
                  "per decade) instead of 4 evenly spaced ones; slower, finds branches that "
                  "exist only at low values.",
    "separate": "also split the samples into one subplot and one Excel file per category "
                "(only when the user asks); the overlay is always made.",
}

RANGE_RULES = """
    Range of each parameter, first match wins: fixed_params; vary_only (the others stay at
    baseline); manual_bounds or an "abs" window; its own param_windows entry; then the
    global rel_window, additive_offset or spread_factor; else a fold of 2 around the baseline
    (or the goal's own). Windows centre on the baseline (baseline_values changes it), stay
    inside allowed_bounds, and contradictions are reported, not resolved silently.
"""


def scan_doc(**overrides):
    """Complete fn's description: its own docstring (the tool's purpose), then one line per
    argument it takes, then the range rules. `overrides` replaces an argument's line for
    this tool."""
    def wrap(fn):
        names = list(inspect.signature(fn).parameters)
        lines = [f"        {n}: {overrides.get(n, ARG_DOCS[n])}" for n in names]
        fn.__doc__ = ("    " + inspect.cleandoc(fn.__doc__ or "").replace("\n", "\n    ")
                      + "\n\n    Args:\n" + "\n".join(lines) + "\n"
                      + (RANGE_RULES if "fixed_params" in names else ""))
        return fn
    return wrap
