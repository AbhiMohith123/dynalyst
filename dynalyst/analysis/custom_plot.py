"""
Custom plotter: simulate a model and plot exactly what was asked for --
chosen species, against time or against another species, one or several
conditions overlaid, optional log axes and time window.

Simulation goes through jaxkm_tools._get_trajectory, so it works for every
model simulate_model can run (jaxkineticmodel SBML import, or the hand-coded
custom models).
"""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynalyst import config
from dynalyst.analysis.jaxkm_tools import _get_trajectory

PLOTS_DIR = config.PLOTS_DIR
SETTLED_TOLERANCE = 1e-3    # settled: the last 10% of a run moves under this share of its largest value (a choice)
MAX_LEGEND_ENTRIES = 20     # above this a legend covers the plot; a scan's right panel names the values
ONE_BRANCH_NOTE = ("Every run starts from the same state, so in a bistable model this shows one branch only. "
                   "It does not give a bistable region or fold points: those need a bifurcation diagram.")


def _settled(values) -> bool:
    tail = values[int(len(values) * 0.9):]
    return bool(np.ptp(tail) <= SETTLED_TOLERANCE * max(np.max(np.abs(values)), 1e-12))


def _scanned_parameter(runs) -> str | None:
    """The one parameter the runs vary, as numbers, when there are several runs and nothing else differs."""
    overrides = [r.get("parameter_overrides") or {} for r in runs]
    names = {k for o in overrides for k in o}
    if len(runs) < 3 or len(names) != 1 or any(r.get("initial_conditions") != runs[0].get("initial_conditions") for r in runs):
        return None
    name = names.pop()
    values = [o.get(name) for o in overrides]
    return name if all(isinstance(v, (int, float)) for v in values) else None


def _plot_series(model_name_or_path, series, x_label, y_label, log_x, log_y, title) -> str:
    """Plot points given as numbers (e.g. fold values from several bifurcation diagrams against
    a second parameter). Nothing is simulated. The agent checks the numbers against earlier
    tool results before this runs."""
    if not isinstance(series, list):
        return json.dumps({"error": "series must be a list of {'label', 'x': [...], 'y': [...]}."})
    clean = []
    for i, s in enumerate(series):
        if not isinstance(s, dict) or not isinstance(s.get("x"), list) or not isinstance(s.get("y"), list):
            return json.dumps({"error": f"series[{i}] needs 'x' and 'y' lists of numbers."})
        try:
            x, y = [float(v) for v in s["x"]], [float(v) for v in s["y"]]
        except (TypeError, ValueError):
            return json.dumps({"error": f"series[{i}]: every x and y must be a number."})
        if not x or len(x) != len(y) or not np.all(np.isfinite(x + y)):
            return json.dumps({"error": f"series[{i}]: x and y must be finite and of the same, non-zero length."})
        clean.append({"label": str(s.get("label") or f"series {i + 1}"), "x": x, "y": y})
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for s in clean:
        order = np.argsort(s["x"])
        ax.plot(np.array(s["x"])[order], np.array(s["y"])[order], marker="o", label=s["label"])
    ax.set_xlabel(x_label or "x")
    ax.set_ylabel(y_label or "y")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title or f"{model_name_or_path} -- {y_label or 'y'} vs {x_label or 'x'}")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(json.dumps([clean, x_label, y_label, log_x, log_y, title],
                                 sort_keys=True).encode()).hexdigest()[:8]
    stem = str(model_name_or_path or "data").replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
    plot_path = PLOTS_DIR / f"{stem.replace(' ', '_')}_series_{key}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    rel = config.display_path(plot_path)
    return json.dumps({"model": model_name_or_path, "series": clean, "x_axis": x_label, "y_axis": y_label,
                       "plot_path": rel, "plot_note": f"![{stem} {y_label or 'y'} vs {x_label or 'x'}]({rel})",
                       "note": "The points given were plotted as they are; nothing was simulated."}, indent=2)


def custom_plot(model_name_or_path: str, species: list[str] | None = None, x_axis: str = "time",
                t_end: float = 100.0, n_points: int = 500, t_start: float = 0.0,
                initial_conditions: dict | None = None, parameter_overrides: dict | None = None,
                conditions: list | None = None, log_x: bool = False, log_y: bool = False,
                title: str | None = None, series: list | None = None,
                x_label: str | None = None, y_label: str | None = None) -> str:
    """Tool: simulate and plot selected species, or plot given points (see TOOL_DEFINITIONS)."""
    if series:      # points from earlier results: no simulation
        return _plot_series(model_name_or_path, series, x_label, y_label, log_x, log_y, title)
    if isinstance(species, str):
        species = [species]
    plot_all = not species or [s.lower() for s in species] == ["all"]   # every species
    runs = conditions or [{"label": None, "initial_conditions": initial_conditions,
                           "parameter_overrides": parameter_overrides}]

    scanned = _scanned_parameter(runs) if x_axis == "time" else None
    fig, axes = plt.subplots(1, 2 if scanned else 1, figsize=(12.5 if scanned else 7.5, 4.8), squeeze=False)
    ax = axes[0][0]
    summary = []
    model_label = None
    for i, cond in enumerate(runs):
        try:
            df, all_species, model_label, _ = _get_trajectory(
                model_name_or_path, t_end, n_points,
                cond.get("initial_conditions"), cond.get("parameter_overrides"))
        except (ValueError, FileNotFoundError) as e:
            if i == len(runs) - 1 and not any("final" in c for c in summary):
                plt.close(fig)
                return json.dumps({"error": str(e)})
            summary.append({"condition": cond.get("label") or f"condition {i + 1}", "error": str(e)})
            continue
        if plot_all:
            species = [s for s in all_species if s != x_axis]
        wanted = list(species) + ([x_axis] if x_axis != "time" else [])
        missing = [s for s in wanted if s not in all_species]
        if missing:
            plt.close(fig)
            return json.dumps({"error": f"Unknown species {missing} for model '{model_label}'. "
                                        f"Valid: {all_species}"})
        window = df[df["time"] >= t_start]
        if window.empty:
            plt.close(fig)
            return json.dumps({"error": f"t_start={t_start} is beyond t_end={t_end}."})
        x = window["time"] if x_axis == "time" else window[x_axis]
        tag = cond.get("label") or (f"condition {i + 1}" if len(runs) > 1 else None)
        for s in species:
            label = f"{s} ({tag})" if tag else s
            ax.plot(x, window[s], label=label, linestyle=["-", "--", "-.", ":"][i % 4])
        summary.append({
            "condition": tag or "default",
            "final": {s: float(window[s].iloc[-1]) for s in species},
            "min": {s: float(window[s].min()) for s in species},
            "max": {s: float(window[s].max()) for s in species},
            "finite": bool(np.all(np.isfinite(window[species].to_numpy()))),
            "settled": {s: _settled(window[s].to_numpy()) for s in species},
        })

    ax.set_xlabel("Time" if x_axis == "time" else x_axis)
    ax.set_ylabel("Concentration" if x_axis == "time" else ", ".join(species))
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title or f"{model_label} -- {', '.join(species)}"
                 + (" vs time" if x_axis == "time" else f" vs {x_axis}"))
    if len(ax.get_lines()) <= MAX_LEGEND_ENTRIES:
        ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)
    scan = None
    if scanned:
        done = [(r["parameter_overrides"][scanned], c) for r, c in zip(runs, summary) if "final" in c]
        done.sort(key=lambda pair: pair[0])
        xs = [v for v, _ in done]
        scan = {"parameter": scanned, "values": xs,
                "final": {s: [c["final"][s] for _, c in done] for s in species},
                "settled": {s: [c["settled"][s] for _, c in done] for s in species},
                "failed": [c["condition"] for c in summary if "error" in c]}
        for s in species:
            axes[0][1].plot(xs, scan["final"][s], marker="o", label=s)
        axes[0][1].set_xlabel(scanned)
        axes[0][1].set_ylabel(f"Value at t = {t_end:g}")
        axes[0][1].legend(loc="best", fontsize="small")
        axes[0][1].grid(alpha=0.3)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(json.dumps([species, x_axis, t_start, t_end, runs, log_x, log_y],
                                 sort_keys=True, default=str).encode()).hexdigest()[:8]
    plot_path = PLOTS_DIR / f"{str(model_label).replace(' ', '_')}_custom_{key}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    rel = config.display_path(plot_path)
    return json.dumps({
        "model": model_label, "plotted_species": species, "x_axis": x_axis,
        "time_window": [t_start, t_end], "n_conditions": len(runs), "conditions": summary,
        "plot_path": rel, "plot_note": f"![{model_label} custom plot]({rel})",
        **({"final_vs_parameter": scan, "note": ONE_BRANCH_NOTE} if scan else {}),
    }, indent=2, default=float)


TOOL_DEFINITIONS = [{
    "name": "custom_plot",
    "description": (
        "Simulate a model and make a custom plot: chosen species against time or against "
        "another species (x_axis), optionally several conditions overlaid (different "
        "initial conditions and/or parameter values), a time window (t_start..t_end) and "
        "log axes. Use when the user asks for a specific plot rather than the standard "
        "simulation plot, or a time-course scan: several values of one parameter, all in one "
        "call through conditions, which also gives final_vs_parameter and its plot panel. "
        "With series it plots numbers that earlier tool results gave instead (e.g. the two fold "
        "values of each bifurcation diagram against the second parameter it was run at) and "
        "simulates nothing; every number must come from an earlier result."),
    "input_schema": {"type": "object", "properties": {
        "model_name_or_path": {"type": "string", "description": "Model name or SBML path."},
        "species": {"type": "array", "items": {"type": "string"}, "description": "Species to plot (y-axis), by their names in the model; omit to plot every species."},
        "x_axis": {"type": "string", "description": "'time' (default) or a species name for a phase-style plot."},
        "t_end": {"type": "number", "description": "Simulation end time, in the model's own time unit (inspect_model's time_unit; convert the user's hours / minutes / days to it). Default 100."},
        "n_points": {"type": "integer", "description": "Time points. Default 500."},
        "t_start": {"type": "number", "description": "Start of the plotted time window, in the model's own time unit. Default 0."},
        "initial_conditions": {"type": "object", "description": "{species: value} overrides (single condition)."},
        "parameter_overrides": {"type": "object", "description": "{parameter: value} overrides (single condition)."},
        "conditions": {"type": "array", "items": {"type": "object"}, "description":
            "Overlay several runs: [{'label': str, 'initial_conditions': {...}, 'parameter_overrides': {...}}, ...]."},
        "log_x": {"type": "boolean"}, "log_y": {"type": "boolean"},
        "title": {"type": "string"},
        "series": {"type": "array", "items": {"type": "object"}, "description":
            "Only to plot numbers from earlier results, no simulation: [{'label': str, 'x': [numbers], "
            "'y': [numbers]}, ...]. Leave out for a simulation."},
        "x_label": {"type": "string", "description": "x-axis name for series, e.g. 'total_M'."},
        "y_label": {"type": "string", "description": "y-axis name for series, e.g. 'MAPKK at the fold'."},
    }, "required": ["model_name_or_path"]},
}]

TOOL_FUNCTIONS = {"custom_plot": lambda **kw: custom_plot(**kw)}
