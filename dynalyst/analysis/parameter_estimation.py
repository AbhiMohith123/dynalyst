"""
Parameter estimation with jaxkineticmodel: fit chosen model parameters to
measured time courses.

Uses jaxkineticmodel's own machinery (parameter_estimation/training.py):
TrainingData for the dataset, Trainer for the optimisation -- several Latin
hypercube start points, gradient descent in log2 parameter space with
AdaBelief + gradient clipping, and the log-mean-centred loss (or MSE in
linear space). The only addition is holding the parameters that are NOT
being fitted fixed, which Trainer does not support directly: its loss is
swapped for one that merges the fixed values back in.

Data format: a CSV with one row per time point, a time column (first column
or one named 'time'/'t') and one column per measured species (names must
match the model's species). Missing values are allowed except at the first
time point, which sets the initial condition of every measured species.
"""
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dynalyst import config

PLOTS_DIR = config.PLOTS_DIR
_TIME_NAMES = ("time", "t", "Time", "T")


def _resolve_data_path(data_path: str) -> Path:
    p = Path(data_path).expanduser()
    if p.is_file():
        return p
    for root in (config.DATA_DIR, config.PROJECT_ROOT, config.JKM_REPO_PATH / "datasets"):
        for c in root.rglob(p.name):
            if c.is_file():
                return c
    raise FileNotFoundError(f"Dataset '{data_path}' not found (checked the path, {config.DATA_DIR} "
                            f"and the jaxkineticmodel datasets folder).")


def load_time_course(data_path: str) -> tuple[Path, np.ndarray, pd.DataFrame]:
    path = _resolve_data_path(data_path)
    df = pd.read_csv(path)
    time_col = next((c for c in df.columns if c in _TIME_NAMES), df.columns[0])
    ts = df[time_col].to_numpy(dtype=float)
    values = df.drop(columns=[time_col]).apply(pd.to_numeric, errors="coerce")
    if not np.all(np.isfinite(ts)):
        raise ValueError(f"{path.name}: every timestamp in '{time_col}' must be finite.")
    if len(ts) < 2 or np.any(np.diff(ts) <= 0):
        raise ValueError(f"{path.name}: the time column '{time_col}' must be strictly increasing "
                         f"with at least 2 time points.")
    return path, ts, values


def _assignment_rule_targets(sbml_path) -> set[str]:
    import libsbml
    model = libsbml.readSBMLFromFile(str(sbml_path)).getModel()
    return {r.getVariable() for r in model.getListOfRules()
            if r.getTypeCode() == libsbml.SBML_ASSIGNMENT_RULE}


def inspect_dataset(data_path: str, model_name_or_path: str | None = None) -> str:
    """Tool (read-only): what a dataset contains, and how it matches a model."""
    try:
        path, ts, values = load_time_course(data_path)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    out = {"data_file": str(path), "n_time_points": len(ts), "time_range": [ts[0], ts[-1]],
           "columns": list(values.columns),
           "missing_values": {c: int(values[c].isna().sum()) for c in values.columns}}
    if model_name_or_path:
        try:
            from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
            from dynalyst.analysis.generic_sbml import resolve_sbml_path
            from dynalyst.analysis.jaxkm_compat import jaxkm_ready_sbml
            sbml = resolve_sbml_path(model_name_or_path)
            with contextlib.redirect_stdout(io.StringIO()):
                m = SBMLModel(str(jaxkm_ready_sbml(sbml)))
            species = list(m.species_names)
            skip = _assignment_rule_targets(sbml)
            out.update({
                "model_species": species,
                "matched_species": [c for c in values.columns if c in species],
                "unmatched_columns": [c for c in values.columns if c not in species],
                "fittable_parameters": {k: float(v) for k, v in m.parameters.items()
                                        if k not in skip and float(v) > 0},
            })
        except Exception as e:
            out["model_error"] = str(e)
    return json.dumps(out, indent=2, default=float)


def estimate_parameters(model_name_or_path: str, data_path: str,
                        parameters_to_fit: list[str] | None = None,
                        n_initializations: int = 10, lower_bound: float = 0.1,
                        upper_bound: float = 10.0, n_iter: int = 600,
                        learning_rate: float = 5e-2, optim_space: str = "log",
                        loss_threshold: float = 1e-12, seed: int = 0,
                        parameter_bounds: dict | None = None, lhs_space: str = "log",
                        good_fit_factor: float = 10.0, max_solver_steps: int = 5000) -> str:
    # Defaults checked on synthetic data (two parameters x2 and x0.5 from nominal):
    # a looser stop (1e-6) halts 10-15% short of the true values; 1e-12 with
    # lr 0.05 recovers them to 3 significant figures in ~600 steps.
    """Tool: fit parameters of an SBML model to a time-course CSV with
    jaxkineticmodel (see module docstring)."""
    import jax
    import jax.numpy as jnp
    from scipy.stats import qmc
    from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
    from jaxkineticmodel.parameter_estimation.training import (
        Trainer, TrainingData, log_mean_centered_loss_func, mse_loss_func,
        update_linear, update_log)
    from dynalyst.analysis.generic_sbml import resolve_sbml_path

    if optim_space not in ("log", "linear"):
        return json.dumps({"error": "optim_space must be 'log' or 'linear'."})
    if lhs_space not in ("log", "linear"):
        return json.dumps({"error": "lhs_space must be 'log' or 'linear'."})
    if int(max_solver_steps) < 1 or int(n_iter) < 1 or int(n_initializations) < 1:
        return json.dumps({'error':'Solver steps, iterations and initialization count must be positive.'})
    from dynalyst.analysis.jaxkm_compat import jaxkm_ready_sbml
    from dynalyst.analysis.ode_model import xpp_file
    xpp = xpp_file(model_name_or_path)
    if xpp is not None:
        return json.dumps({"error": f"{xpp.name} is an XPP file; fitting runs on SBML. Convert it "
                                    "first (xpp_to_sbml) and fit the converted model."})
    try:
        sbml = resolve_sbml_path(model_name_or_path)
        data_file, ts, values = load_time_course(data_path)
        rule_targets = _assignment_rule_targets(sbml)
        sim_sbml = jaxkm_ready_sbml(sbml)
        with contextlib.redirect_stdout(io.StringIO()):
            sbml_model = SBMLModel(str(sim_sbml))
            kin_model = sbml_model.get_kinetic_model()
            kin_model.max_steps = int(max_solver_steps)
            # The installed model defaults to dt0=1e-12. On the decay regression,
            # Kvaerno5 stalls for ordinary k values with that forced step; its
            # adaptive initial-step selection succeeds at the same tolerances.
            kin_model.dt0 = None
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"jaxkineticmodel could not load '{model_name_or_path}': {e}"})

    species = list(sbml_model.species_names)
    measured = [c for c in values.columns if c in species]
    unmatched = [c for c in values.columns if c not in species]
    if not measured:
        return json.dumps({"error": f"No dataset column matches a model species. Columns: "
                                    f"{list(values.columns)}; model species: {species}"})
    if values[measured].iloc[0].isna().any():
        return json.dumps({"error": "The first time point must have a value for every measured "
                                    "species (it sets their initial conditions)."})

    nominal = {k: float(v) for k, v in sbml_model.parameters.items()}
    if parameters_to_fit:
        unknown = [p for p in parameters_to_fit if p not in nominal]
        if unknown:
            return json.dumps({"error": f"Unknown parameter(s) {unknown}.",
                               "available_parameters": sorted(nominal)})
        bad = [p for p in parameters_to_fit if p in rule_targets or nominal[p] <= 0]
        if bad and optim_space == "log":
            return json.dumps({"error": f"Cannot fit {bad} in log space: they are assignment-rule "
                                        f"variables or have a non-positive nominal value."})
        free = list(parameters_to_fit)
    else:
        free = [p for p, v in nominal.items() if p not in rule_targets and v > 0]
    if not free:
        return json.dumps({"error": "No parameters to fit."})
    fixed = {p: v for p, v in nominal.items() if p not in free}

    dataset = values[measured].to_numpy(dtype=float)[None, :, :]
    data = TrainingData(list(ts), dataset, [measured], sbml_model)
    trainer = Trainer(kin_model, data, n_iter=n_iter, learning_rate=learning_rate,
                      loss_threshold=loss_threshold, optim_space=optim_space)

    # Hold non-fitted parameters fixed: Trainer optimises every parameter it
    # is given, so feed it only the free ones and merge the fixed ones back
    # into the loss (in log2 space for log optimisation, like Trainer does).
    if optim_space == "log":
        fixed_opt = {k: jnp.log2(jnp.asarray(v)) for k, v in fixed.items()}
        base_loss, update = log_mean_centered_loss_func, update_log
    else:
        fixed_opt = {k: jnp.asarray(v) for k, v in fixed.items()}
        base_loss, update = mse_loss_func, update_linear

    def loss_with_fixed(params, ts, ys, model):
        return base_loss({**fixed_opt, **params}, ts, ys, model)

    trainer.loss_func = trainer._create_loss_func(loss_with_fixed)
    def step(opt_state, params, ts, ys):  # Trainer.train calls it with keywords
        return update(opt_state, params, ts, ys, trainer.loss_func, trainer.optimizer)

    trainer.make_step = jax.jit(step) if optim_space == "log" else step

    # Latin hypercube design of start points. Per-parameter bounds default to
    # [nominal*lower_bound, nominal*upper_bound]. 'log' (default) samples each
    # parameter log-uniformly -- rate constants span orders of magnitude, and a
    # linear design over 0.1x-10x puts ~90% of the starts above nominal.
    # 'linear' with no per-parameter bounds uses jaxkineticmodel's own
    # Trainer.latinhypercube_sampling unchanged.
    parameter_bounds = parameter_bounds or {}
    unknown = [p for p in parameter_bounds if p not in free]
    if unknown:
        return json.dumps({"error": f"parameter_bounds given for {unknown}, which are not being "
                                    f"fitted ({free})."})
    bounds = {p: tuple(map(float, parameter_bounds.get(p, (nominal[p] * lower_bound,
                                                           nominal[p] * upper_bound))))
              for p in free}
    bad = [p for p, (lo_, hi_) in bounds.items() if not (0 < lo_ < hi_)]
    if bad:
        return json.dumps({"error": f"Bounds for {bad} must satisfy 0 < lower < upper."})
    if lhs_space == "linear" and not parameter_bounds:
        with contextlib.redirect_stdout(io.StringIO()):
            np.random.seed(seed)
            sets = trainer.latinhypercube_sampling({p: nominal[p] for p in free},
                                                   lower_bound, upper_bound, n_initializations)
        sampler_name = "jaxkineticmodel Trainer.latinhypercube_sampling (linear, not seedable)"
    else:
        unit = qmc.LatinHypercube(d=len(free), seed=seed).random(n_initializations)
        lo = np.array([bounds[p][0] for p in free])
        hi = np.array([bounds[p][1] for p in free])
        vals = (np.exp(np.log(lo) + unit * (np.log(hi) - np.log(lo))) if lhs_space == "log"
                else lo + unit * (hi - lo))
        sets = pd.DataFrame(vals, columns=free)
        sampler_name = f"scipy.stats.qmc.LatinHypercube ({lhs_space}-uniform, seed={seed})"
    trainer.parameter_sets = sets

    from dynalyst.analysis.fitting_runtime import train_diagnosed, dependency_info
    dependency = dependency_info(trainer)
    import uuid
    job_dir = PLOTS_DIR / (sbml.stem + "_fit_" + uuid.uuid4().hex[:10])
    job_dir.mkdir(parents=True, exist_ok=True)
    optimized, runs = train_diagnosed(trainer, job_dir / 'optimizer_steps.jsonl')
    trace_path = job_dir / "optimizer_trace.json"
    trace_path.write_text(json.dumps({"dependency": dependency, "runs": runs}, indent=2), encoding="utf-8")
    runs = [{k:v for k,v in r.items() if k != "trace"} for r in runs]
    ok = [r for r in runs if not r["failed"] and np.isfinite(r["final_loss"])]
    if not ok:
        return json.dumps({"error": "Every initialization failed to optimise.", "runs": runs,
                           "dependency": dependency, "diagnostics_path": str(trace_path)})
    best = min(ok, key=lambda r: r["final_loss"])
    best_params = {k: float(v) for k, v in optimized[best["init"]].items()}
    full = {**fixed, **best_params}

    y0 = jnp.asarray(data.dataset[0, 0, :])
    try:
        y_fit = np.asarray(kin_model(jnp.asarray(ts), y0, full))
        if not np.isfinite(y_fit).all():
            raise ValueError("Fitted forward trajectory contains non-finite values.")
    except Exception as exc:
        return json.dumps({"error": "The fitted forward trajectory failed validation.",
                           "forward_error": f"{type(exc).__name__}: {exc}",
                           "best_parameters": best_params, "runs": runs, "dependency": dependency,
                           "diagnostics_path": config.display_path(trace_path)})
    nominal_warning = None
    try:
        y_nom = np.asarray(kin_model(jnp.asarray(ts), y0, nominal))
        if not np.isfinite(y_nom).all():
            raise ValueError("Nominal forward trajectory contains non-finite values.")
    except Exception as exc:
        y_nom = None
        nominal_warning = f"Nominal comparison unavailable: {type(exc).__name__}: {exc}"
    rmse = {}
    for j, s in enumerate(measured):
        obs = values[s].to_numpy(dtype=float)
        mask = np.isfinite(obs)
        rmse[s] = float(np.sqrt(np.mean((y_fit[mask, species.index(s)] - obs[mask]) ** 2)))

    fig, axes = plt_subplots(len(measured))
    for ax, s in zip(axes, measured):
        ax.plot(ts, values[s], "o", color="black", markersize=4, label="data")
        if y_nom is not None:
            ax.plot(ts, y_nom[:, species.index(s)], "--", color="grey", label="nominal parameters")
        ax.plot(ts, y_fit[:, species.index(s)], "-", color="tab:red", label="fitted")
        ax.set_ylabel(s)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize="small")
    axes[-1].set_xlabel("Time")
    axes[0].set_title(f"{sbml.stem} -- parameter fit ({len(free)} parameter(s))")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = job_dir / f"{sbml.stem}_parameter_fit.png"
    fig.savefig(plot_path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    rel = config.display_path(plot_path)

    fitted = {}
    for p in free:
        v, n0 = best_params[p], nominal[p]
        lo_, hi_ = bounds[p]
        fitted[p] = {"nominal": n0, "fitted": v, "fitted_over_nominal": v / n0,
                     "bounds": [lo_, hi_], "bounds_role": "initialization only; optimizer is not box-constrained",
                     "at_bound": bool(v <= lo_ * 1.01 or v >= hi_ * 0.99)}
    multistart = _multistart_analysis(runs, optimized, free, best["final_loss"], good_fit_factor,
                                      loss_floor=10 * loss_threshold)
    for p in free:
        fitted[p]["identifiability"] = multistart["parameters"][p]["status"]
    return json.dumps({
        "model_file": str(sbml), "simulated_with": str(sim_sbml), "data_file": str(data_file),
        "measured_species": measured, "ignored_columns": unmatched,
        "fitted_parameters": fitted, "fixed_parameters": sorted(fixed),
        "optim_space": optim_space, "n_iter": n_iter, "learning_rate": learning_rate,
        "bounds_factor": [lower_bound, upper_bound], "n_initializations": n_initializations,
        "lhs_design": {"sampler": sampler_name, "space": lhs_space, "n_starts": n_initializations,
                       "bounds": {p: list(b) for p, b in bounds.items()},
                       "start_points": sets.to_dict(orient="records")},
        "dependency": dependency, "diagnostics_path": config.display_path(trace_path),
        "multistart": multistart,
        "runs": runs, "best_init": best["init"],
        "best_start_converged": best["converged"], "nominal_warning": nominal_warning,
        "initial_loss": best["initial_loss"], "final_loss": best["final_loss"],
        "loss_reduction_factor": (best["initial_loss"] / best["final_loss"]
                                  if best["final_loss"] else None),
        "rmse_per_species": rmse,
        "plot_path": rel, "plot_note": f"![{sbml.stem} parameter fit]({rel})",
    }, indent=2, default=float)


def _multistart_analysis(runs, optimized, free, best_loss, factor, loss_floor=0.0):
    """How well the LHS start points agree. 'Good fits' are runs whose final
    loss is within `factor` of the best, or below `loss_floor` (runs that
    reached the stopping threshold are all perfect fits, however their tiny
    losses compare). A parameter is identifiable from
    these data if all good fits give (nearly) the same value; if good fits
    spread over a wide range the data do not pin it down."""
    good = [r for r in runs if not r["failed"] and r["final_loss"] is not None
            and np.isfinite(r["final_loss"])
            and r["final_loss"] <= max(factor * best_loss, loss_floor) + 1e-300]
    values = {p: np.array([float(optimized[r["init"]][p]) for r in good]) for p in free}
    per = {}
    for p in free:
        v = values[p]
        spread = float(v.max() / v.min()) if len(v) and v.min() > 0 else None
        if len(v) < 2:
            status = "undetermined (fewer than 2 good fits)"
        elif spread is not None and spread <= 1.2:
            status = "identifiable"
        else:
            status = "not identifiable from these data"
        per[p] = {"good_fit_values": [float(x) for x in v],
                  "min": float(v.min()) if len(v) else None,
                  "max": float(v.max()) if len(v) else None,
                  "fold_spread": spread, "status": status}
    # Correlation only means something for parameters whose good fits
    # actually spread (identical values give arbitrary correlations).
    spread_params = [p for p in free if per[p]["fold_spread"] and per[p]["fold_spread"] > 1.2]
    correlated = []
    if len(good) >= 3 and len(spread_params) >= 2:
        r = np.corrcoef(np.log(np.array([values[p] for p in spread_params])))
        for i in range(len(spread_params)):
            for j in range(i + 1, len(spread_params)):
                if abs(r[i, j]) >= 0.95:
                    correlated.append({"parameters": [spread_params[i], spread_params[j]],
                                       "log_correlation": float(r[i, j])})
    return {"n_good_fits": len(good), "n_starts": len(runs),
            "good_fit_criterion": f"final loss <= max({factor:g} x best, {loss_floor:g})",
            "parameters": per,
            "correlated_pairs": correlated}


def plt_subplots(n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.6 * n + 0.6), sharex=True, squeeze=False)
    return fig, list(axes[:, 0])


TOOL_DEFINITIONS = [
    {
        "name": "estimate_parameters",
        "description": (
            "Fit parameters of an SBML model to measured time-course data with "
            "jaxkineticmodel (gradient-based, several Latin-hypercube start points, log-space "
            "optimisation). Data: CSV with a time column and one column per measured species. "
            "Fits the listed parameters (default: every positive global parameter) and keeps the "
            "rest fixed. The start points are a Latin-hypercube design (log-uniform by default, "
            "per-parameter bounds optional); the multi-start results are analysed for "
            "identifiability (do all good fits agree? are parameters correlated?). Returns fitted "
            "vs nominal values, the LHS design, per-start losses, identifiability, RMSE per "
            "species and a data-vs-fit plot."),
        "input_schema": {"type": "object", "properties": {
            "model_name_or_path": {"type": "string", "description": "SBML model name or path."},
            "data_path": {"type": "string", "description": "Time-course CSV (path or file name)."},
            "parameters_to_fit": {"type": "array", "items": {"type": "string"},
                                  "description": "Parameters to estimate; all others stay at their model values."},
            "n_initializations": {"type": "integer", "description": "Latin-hypercube start points. Default 10 (each takes seconds to tens of seconds)."},
            "parameter_bounds": {"type": "object", "description": "Per-parameter LHS bounds {name: [lower, upper]}; others use nominal*lower_bound..nominal*upper_bound."},
            "lhs_space": {"type": "string", "enum": ["log", "linear"], "description": "LHS scaling. 'log' (default): log-uniform, right for rate constants. 'linear' without parameter_bounds uses jaxkineticmodel's own Trainer.latinhypercube_sampling."},
            "lower_bound": {"type": "number", "description": "Start points drawn from nominal*lower_bound. Default 0.1."},
            "upper_bound": {"type": "number", "description": "...up to nominal*upper_bound. Default 10."},
            "n_iter": {"type": "integer", "description": "Optimisation steps per start point. Default 600."},
            "max_solver_steps": {"type":"integer", "description":"Per-forward-solve step limit; default 5000. Failures are recorded per initialization."},
            "learning_rate": {"type": "number", "description": "AdaBelief learning rate. Default 0.05."},
            "loss_threshold": {"type": "number", "description": "Stop a start point once the loss drops below this. Default 1e-12."},
            "optim_space": {"type": "string", "enum": ["log", "linear"], "description": "Default 'log'."},
            "seed": {"type": "integer"},
        }, "required": ["model_name_or_path", "data_path"]},
    },
    {
        "name": "inspect_dataset",
        "description": ("Read-only: describe a time-course CSV (columns, time points, missing "
                        "values) and, given a model, which columns match its species and which "
                        "parameters can be fitted."),
        "input_schema": {"type": "object", "properties": {
            "data_path": {"type": "string"},
            "model_name_or_path": {"type": "string"},
        }, "required": ["data_path"]},
    },
]

TOOL_FUNCTIONS = {
    "estimate_parameters": lambda **kw: estimate_parameters(**kw),
    "inspect_dataset": lambda **kw: inspect_dataset(**kw),
}
