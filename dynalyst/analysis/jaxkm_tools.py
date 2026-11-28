"""
Wraps the real, installed jaxkineticmodel package as callable tool functions,
so the agent can actually SIMULATE models and RETRIEVE ODEs rather than just
describe them from text.

API confirmed from https://abeellab.github.io/jaxkineticmodel/SBML/ :

    from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
    model = SBMLModel(filepath)
    S = model._get_stoichiometric_matrix()
    JaxKmodel = model.get_kinetic_model()
    ys = JaxKmodel(ts=ts, y0=model.y0, params=model.parameters)

Run these functions on the GPU node — JAX will pick up CUDA automatically if
`jax[cuda12]` is installed and a GPU is visible.
"""
import hashlib
import json
import re
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed on a server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynalyst import config

# jaxkineticmodel must be importable (pip install -e jaxkineticmodel_src/ from setup_env.sh)
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from jaxkineticmodel.load_sbml.sbml_load_utils import construct_param_point_dictionary, separate_params

PLOTS_DIR = config.PLOTS_DIR


def _save_plot(df: pd.DataFrame, species: list[str], title: str, filename_stem: str,
                species_filter: list[str] | None = None) -> str:
    """
    Shared plotting helper for every simulation tool.
    species_filter: if given, only these species are drawn (still validated
    against the model's real species list by the caller before this runs).
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # named after the data too, so a second run with other values cannot overwrite the first
    tag = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:6]
    plot_path = PLOTS_DIR / f"{filename_stem}_simulation_{tag}.png"

    to_plot = species_filter if species_filter else species

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sp in to_plot:
        ax.plot(df["time"], df[sp], label=sp)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return config.display_path(plot_path)


def _resolve_sbml_path(model_name_or_path: str) -> Path:
    """
    Accepts either a full path to an .xml file, or a bare model name that will
    be searched for across config.SBML_SEARCH_DIRS (local downloaded models
    checked first, then the jaxkineticmodel repo's own example models).
    """
    return _jaxkm_ready(_find_sbml(model_name_or_path))


# files jaxkineticmodel cannot be given (its conversion of their assignment rules failed):
# read by generic_sbml instead
_NOT_FOR_JAXKM: set = set()


def _jaxkm_ready(path: Path) -> Path:
    # SBML with assignment-rule fluxes simulates as a flat
    # line in jaxkineticmodel; use a converted copy (analysis/jaxkm_compat.py).
    # When that conversion fails (jaxkineticmodel's exporter checks its own round trip
    # and can stop on an assertion), the file itself is returned, marked so that
    # jaxkineticmodel is not used for it: given the rules as they are, it would
    # simulate a flat line without saying so.
    from dynalyst.analysis.jaxkm_compat import jaxkm_ready_sbml
    try:
        return jaxkm_ready_sbml(path)
    except Exception:
        _NOT_FOR_JAXKM.add(str(path))
        return path


def generic_sbml_time_unit(model_name_or_path: str) -> str | None:
    """The model file's declared time unit (generic_sbml.time_unit_of), or None."""
    from dynalyst.analysis.generic_sbml import time_unit_of
    try:
        return time_unit_of(_find_sbml(model_name_or_path))
    except Exception:
        return None


def _find_sbml(model_name_or_path: str) -> Path:
    """The model's file: SBML, or an XPP / ODE text file (.ode, .txt) by name or path."""
    from dynalyst.analysis.ode_model import resolve_model_file
    return resolve_model_file(model_name_or_path)


def _xpp_ode_files() -> dict:
    """XPP / ODE text models (.ode, and .txt files with derivative lines such as
    S'=... or dS/dt=...) in the model folders: they are models too, read by every
    tool that takes model_name_or_path (simulation, stochastic, steady states, ...)."""
    import re
    deriv = re.compile(r"^\s*[A-Za-z_]\w*\s*'\s*=|^\s*d[A-Za-z_]\w*\s*/\s*dt\s*=", re.M)
    out = {}
    for root in config.SBML_SEARCH_DIRS:
        if not root.exists():
            continue
        found = []
        for p in sorted(list(root.rglob("*.ode")) + list(root.rglob("*.txt"))):
            try:
                if p.suffix == ".ode" or deriv.search(p.read_text(errors="replace")[:20000]):
                    found.append(str(p.relative_to(root)))
            except OSError:
                continue
        if found:
            out[str(root)] = found
    return out


def list_available_models() -> str:
    """Tool: list the model files available: SBML (.xml) and XPP / ODE text (.ode, .txt)."""
    from dynalyst.analysis.generic_sbml import available_sbml_models
    result = available_sbml_models()
    total = sum(len(v) for v in result.values())
    xpp = _xpp_ode_files()

    return json.dumps({
        "sbml_models_by_directory": result,
        "sbml_count": total,
        "xpp_ode_models_by_directory": xpp,
        "xpp_ode_count": sum(len(v) for v in xpp.values()),
        "note": "Every model is read from its own file (jaxkineticmodel, or generic_sbml "
                "for files jaxkineticmodel cannot load). XPP / ODE files (.ode, .txt) are "
                "used by their file name, e.g. 'my_model.ode'. A file not listed here can "
                "still be a path the user gives: pass it to the tool, which reports if it is "
                "missing.",
    })


def get_model_odes(model_name_or_path: str) -> str:
    """
    Tool: a model's structural definition (species, reactions, stoichiometry,
    parameters), read from its file -- jaxkineticmodel when it can load the file,
    else generic_sbml (which reads any SBML, including rules and local constants).
    """
    path = _resolve_sbml_path(model_name_or_path)
    try:
        if str(path) in _NOT_FOR_JAXKM:
            raise ValueError("not for jaxkineticmodel")
        model = SBMLModel(str(path))
        S = model._get_stoichiometric_matrix()
    except Exception:
        from dynalyst.analysis.ode_model import ode_strings, parameter_listing, resolve_model
        m = resolve_model(str(path))
        return json.dumps({"model_file": str(path), "species": m["state_names"],
                           "parameters": parameter_listing(m), "odes": ode_strings(m),
                           "source": "generic_sbml (jaxkineticmodel could not load this file)"},
                          indent=2, default=str)
    return json.dumps({
        "model_file": str(path),
        "species": list(S.index),
        "reactions": list(S.columns),
        "stoichiometric_matrix": S.values.tolist(),
        "parameter_names": list(model.parameters.keys()) if hasattr(model.parameters, "keys") else list(model.parameters),
        "n_species": len(S.index),
        "n_reactions": len(S.columns),
    }, indent=2)


from dynalyst.analysis.ode_model import relative_value  # noqa: E402  (one definition)


def _generic_trajectory(model_name_or_path, t_end, n_points, initial_conditions=None,
                        parameter_overrides=None):
    """The model read from its own file (analysis/generic_sbml: any SBML, rules,
    fixed species, local constants) integrated with LSODA. Used when
    jaxkineticmodel cannot load a file; nothing about any model is coded here."""
    from dynalyst.analysis.simulation_model import load_model, provenance
    from dynalyst.analysis.deterministic import integrate
    m, values, changed = load_model(model_name_or_path, parameters=parameter_overrides,
                                    initial_conditions=initial_conditions)
    names = m["state_names"]
    ts = np.linspace(0, t_end, n_points)
    ys, info = integrate(m, values, ts)
    df = pd.DataFrame(ys, columns=names)
    df.insert(0, "time", ts)
    src = {"source": "canonical model (" + info["backend"] + ")",
           "method": info["method"], "backend": info["backend"],
           "model_inputs": provenance(m, values), "params_used": values, "changed": changed}
    if m.get("events"):
        src["events_fired"] = [
            {"event": e, "time": round(tt, 6),
             "changed": {var: f"{before:.6g} -> {after:.6g}" for var, before, after in ch}}
            for e, tt, ch in info["events_fired"][:50]]
        if len(info["events_fired"]) > 50:
            src["events_fired"].append({"note": f"{len(info['events_fired']) - 50} more not listed"})
    if info.get("delays"):
        src["delay_differential_equation"] = (
            "simulated as a delay differential equation (method of steps; before time 0 "
            "every quantity has its starting value): " + ", ".join(info["delays"]))
    if info.get("notes"):
        src["simulation_notes"] = info["notes"]
    return df, names, m["label"], src


def _get_trajectory(model_name_or_path: str, t_end: float, n_points: int,
                     initial_conditions: dict | None = None,
                     parameter_overrides: dict | None = None):
    """
    Shared core: (df, species_list, model_label, source_dict) for ANY model file,
    with optional initial-condition and parameter overrides. simulate_model,
    compare_initial_conditions, plot_phase_plane and plot_rate_of_change all use
    it. jaxkineticmodel runs the file when it can load it; otherwise the model is
    read from the file by generic_sbml and integrated (_generic_trajectory).
    Raises ValueError on unknown species / parameters.
    """
    if _uses_time(_find_sbml(model_name_or_path)) or _needs_canonical_initial_state(_find_sbml(model_name_or_path)):
        # inputs that change with time: integrated as such by generic_sbml (keep_time)
        return _generic_trajectory(str(_find_sbml(model_name_or_path)), t_end, n_points,
                                   initial_conditions, parameter_overrides)
    from dynalyst.analysis.simulation_model import load_model, provenance
    canonical, canonical_values, _ = load_model(
        model_name_or_path, parameters=parameter_overrides, initial_conditions=initial_conditions)
    path = _resolve_sbml_path(model_name_or_path)
    try:
        if str(path) in _NOT_FOR_JAXKM:
            raise ValueError("not for jaxkineticmodel")
        model = SBMLModel(str(path))
        S = model._get_stoichiometric_matrix()
        JaxKmodel = model.get_kinetic_model()
    except Exception:
        return _generic_trajectory(str(path), t_end, n_points, initial_conditions,
                                   parameter_overrides)
    all_species = list(S.index)
    if set(all_species) != set(canonical["state_names"]):
        return _generic_trajectory(model_name_or_path, t_end, n_points,
                                   initial_conditions, parameter_overrides)
    y0 = jnp.array([canonical["y0"][n] for n in all_species])
    params_used = dict(model.parameters) if hasattr(model.parameters, "keys") else model.parameters
    changed = {}
    for k, v in (parameter_overrides or {}).items():
        key = override_key(params_used, k)
        if key is None:
            # e.g. a fixed (boundary) species, which jaxkineticmodel keeps as a species but
            # the generic reader treats as the parameter it is
            return _generic_trajectory(str(path), t_end, n_points, initial_conditions,
                                       parameter_overrides)
        new = relative_value(v, params_used[key])
        changed[key] = (float(params_used[key]), new)
        params_used[key] = new
    for name, value in canonical_values.items():
        key = override_key(params_used, name)
        if key is None:
            return _generic_trajectory(model_name_or_path, t_end, n_points,
                                       initial_conditions, parameter_overrides)
        params_used[key] = value
    ts = jnp.linspace(0, t_end, n_points)
    # jaxkm compiles its solver once per output length (~2.5 min each): pad the output times
    # to a power of 2 with copies of t_end, so a few lengths cover every request
    padded = 1 << max(int(n_points) - 1, 1).bit_length()
    ts_run = jnp.concatenate([ts, jnp.full(padded - int(n_points), t_end)])
    try:
        ys = JaxKmodel(ts=ts_run, y0=y0, params=params_used)[:int(n_points)]
    except Exception:
        # the JAX solver stops at its step limit on long runs of stiff or fast oscillating
        # models: the same file integrated with LSODA instead
        return _generic_trajectory(str(path), t_end, n_points, initial_conditions,
                                   parameter_overrides)
    if _frozen_but_moving(path, list(S.index), np.asarray(ys), initial_conditions,
                          parameter_overrides):
        # jaxkineticmodel kept a species constant that the model's own equations move
        # (e.g. rules it ignores, species in other compartments): its run is wrong
        return _generic_trajectory(str(path), t_end, n_points, initial_conditions,
                                   parameter_overrides)
    df = pd.DataFrame(ys, columns=S.index)
    df.insert(0, "time", ts)
    return df, all_species, path.stem, {"source": "SBML (jaxkineticmodel)",
                                        "method": "deterministic ODE",
                                        "backend": "JaxKineticModel / Diffrax",
                                        "model_inputs": provenance(canonical, canonical_values),
                                        "params_used": params_used, "changed": changed}


def _needs_canonical_initial_state(path) -> bool:
    """Use our canonical reader for rules and non-unit volume semantics.

    The alternate importer must not silently freeze rate-rule parameters or
    interpret an initial amount as a concentration.
    """
    import libsbml
    if Path(path).suffix.lower() not in (".xml", ".sbml"):
        return False
    doc = libsbml.readSBMLFromFile(str(path))
    model = doc.getModel()
    if model is None:
        return True  # canonical reader supplies the parse diagnostic
    return (model.getNumInitialAssignments() > 0
            or any(r.isRate() for r in model.getListOfRules())
            or any(c.isSetSize() and c.getSize() != 1.0 for c in model.getListOfCompartments()))


def _uses_time(path) -> bool:
    """Does the SBML file use one of SBML's special symbols in a formula -- time (a forcing or
    dose protocol), delay, rateOf or avogadro -- or have events? jaxkineticmodel does not give
    them their meaning (a delay, e.g., is simulated as something else without a word; events
    are skipped), so such a model is read by generic_sbml and simulated by hybrid_sim, which
    applies each one or says plainly that it cannot."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return (any(f"sbml.org/sbml/symbols/{s}" in text for s in ("time", "delay", "rateOf", "avogadro"))
            or "<listOfEvents" in text)


def _frozen_but_moving(path, names, ys, initial_conditions, parameter_overrides) -> bool:
    """An independent check of a jaxkineticmodel run: a species it kept exactly constant
    while the model's equations, read by generic_sbml, change it at the start. (Each of
    jaxkineticmodel's silent failures seen so far freezes the species it mishandles.)"""
    # constant up to rounding (a species no reaction term reaches drifts by ~1e-16)
    frozen = [n for i, n in enumerate(names) if ys.shape[0] > 1 and np.all(np.isfinite(ys[:, i]))
              and np.ptp(ys[:, i]) <= 1e-12 * (1.0 + float(np.max(np.abs(ys[:, i]))))]
    if not frozen:
        return False
    try:
        from dynalyst.analysis.ode_model import resolve_model
        from dynalyst.analysis.steady_states import _lambdify_model, _term_scale_fn, flux_scale
        m = resolve_model(str(path), parameters=parameter_overrides,
                          initial_conditions=initial_conditions)
        rhs_fn, _ = _lambdify_model(m)
        y = [m["y0"][s] for s in m["state_names"]]
        pv = [m["param_defaults"][p] for p in m["param_names"]]
        with np.errstate(all="ignore"):
            f = np.abs(np.asarray(rhs_fn(*y, *pv), dtype=float))
            scale = flux_scale(np.asarray(_term_scale_fn(m)(*y, *pv), dtype=float))
    except Exception:
        return False        # cannot judge: keep jaxkineticmodel's run
    moving = {s for s, fi, si in zip(m["state_names"], f, scale)
              if np.isfinite(fi) and fi > 1e-6 * si}
    return any(n in moving for n in frozen)


def agent_name(key: str) -> str:
    """jaxkineticmodel's 'lp_<reaction>__<name>' as the rest of the agent writes it."""
    return key[3:].replace("__", "_", 1) if key.startswith("lp_") and "__" in key else key


def override_key(params: dict, name: str):
    """The model's key for a parameter the user names: the exact name, else the
    one reaction-local parameter ending in '_<name>'."""
    if name in params:
        return name
    # jaxkineticmodel names a reaction-local constant 'lp_<reaction>__<name>'; the rest of
    # the agent (generic_sbml) names it '<reaction>_<name>': both refer to the same one
    for k in params:
        if k.startswith("lp_") and "__" in k and k[3:].replace("__", "_", 1) == name:
            return k
    hits = [k for k in params if k.endswith("_" + name)]
    return hits[0] if len(hits) == 1 else None


def time_course_behaviour(df, species) -> dict:
    """What a simulated time course shows, read from the series itself: settled (each
    species changes < 1% over the last fifth), oscillating (in the second half at least 3
    peaks of a species whose swing is > 5% of its level, with the period between peaks),
    or still changing at the end -- then the run was too short to judge."""
    import numpy as _np
    t = _np.asarray(df["time"], dtype=float)
    # a run with NaN / inf values failed: nothing about settling or oscillation can be read
    # from it (the species left are not the whole system)
    broken = [s for s in species if not _np.all(_np.isfinite(_np.asarray(df[s], dtype=float)))]
    if broken:
        return {"verdict": "integration failed", "non_finite_species": broken,
                "note": f"The simulation produced non-finite values (NaN or infinity) for "
                        f"{broken}: the run failed, so it shows neither settling nor oscillation. "
                        "Check the model's rate laws at these values, or simulate a shorter time."}
    half, last = t >= t[0] + 0.5 * (t[-1] - t[0]), t >= t[0] + 0.8 * (t[-1] - t[0])
    oscillating, changing = {}, []
    for sname in species:
        y = _np.asarray(df[sname], dtype=float)
        level = max(float(_np.max(_np.abs(y[half]))), 1e-12)
        yh, th = y[half], t[half]
        # the main peaks only (prominence): the small wiggles of a bursting oscillation
        # between its spikes are not periods
        from scipy.signal import find_peaks
        peaks = list(find_peaks(yh, prominence=0.3 * float(_np.ptp(yh)))[0]) if _np.ptp(yh) > 0 else []
        if len(peaks) >= 3 and (float(_np.max(yh)) - float(_np.min(yh))) > 0.05 * level:
            amplitudes = [float(yh[a] - _np.min(yh[a:b+1]))
                          for a, b in zip(peaks, peaks[1:])]
            ratio = amplitudes[-1] / amplitudes[0] if amplitudes[0] > 0 else None
            trend = ("undetermined" if ratio is None else "decreasing" if ratio < 0.95
                     else "increasing" if ratio > 1.05 else "approximately constant")
            gaps = _np.diff(th[peaks])

            def up_and_down(x, tol):     # a damped or growing oscillation changes one way only
                d = _np.diff(x)
                return bool(_np.any(d > tol) and _np.any(d < -tol))
            # one period repeats: evenly spaced main peaks (5%); spacings going up and down mean
            # bursting, period doubling or chaos, and the mean spacing is not a period. Heights
            # are not used: a sharp spike's sampled height varies with the sampling (Oregonator)
            complex_ = up_and_down(gaps, 0.05 * float(_np.mean(gaps)))
            oscillating[sname] = {"peaks": len(peaks),
                                  "period": float(_np.mean(gaps)),
                                  "pattern": "complex" if complex_ else "simple",
                                  "peak_to_trough_amplitudes": amplitudes,
                                  "last_over_first_amplitude": ratio,
                                  "amplitude_trend": trend}
        elif float(_np.max(y[last]) - _np.min(y[last])) > 0.01 * max(float(_np.max(_np.abs(y[last]))), 1e-12):
            changing.append(sname)
    if oscillating:
        name, o = next(iter(oscillating.items()))
        # an amplitude still changing over a few periods is the approach to the long-term
        # behaviour, not yet a sign of damping (a repressilator's first peaks shrink too)
        short = (o["amplitude_trend"] in ("decreasing", "increasing") and o["peaks"] < 10)
        return {"verdict": "oscillating", "oscillating_species": oscillating,
                "limit_cycle_established": False,
                "note": f"Oscillations observed in the second half of the run (e.g. {name}: "
                        f"{o['peaks']} peaks, period ~{o['period']:.4g}; peak-to-trough "
                        f"amplitude is {o['amplitude_trend']}). A finite time course does "
                        "not establish a sustained limit cycle."
                        + (f" With only {o['peaks']} peaks and the amplitude still changing, "
                           "this run cannot tell damped from sustained oscillation: run it "
                           "several times longer to decide." if short else "")}
    if changing:
        return {"verdict": "still changing at t_end", "changing_species": changing,
                "note": "The run ended before the dynamics settled: it is too short to say "
                        "whether the model settles or oscillates. Run it longer (a larger t_end)."}
    return {"verdict": "settled", "note": "Every species varies by less than 1% in the final "
            "fifth of this trajectory. This describes the observed window and initial "
            "condition; it does not rule out other attractors."}


POINTS_PER_PERIOD = 40          # samples per cycle needed to read the true peaks of an oscillation
MAX_BEHAVIOUR_POINTS = 200_000


def resolved_behaviour(df, species, rerun) -> dict:
    """time_course_behaviour read on a grid fine enough for the oscillation it finds. With few
    samples per cycle the sampled peaks miss the true ones: that reads as a changing amplitude
    and a wrong period. rerun(n) gives the same run on n points (the plot keeps the asked ones)."""
    b, asked = time_course_behaviour(df, species), len(df)
    for _ in range(2):          # the coarse period can be aliased: refine at most twice
        if b.get("verdict") != "oscillating":
            return b
        t = np.asarray(df["time"], dtype=float)
        period = min(o["period"] for o in b["oscillating_species"].values())
        if period >= POINTS_PER_PERIOD * (t[1] - t[0]):
            return b
        n = int(min(MAX_BEHAVIOUR_POINTS, 1.5 * POINTS_PER_PERIOD * (t[-1] - t[0]) / period)) + 1
        if n <= len(df):
            return b
        try:
            df = rerun(n)
        except Exception:
            return b
        b = time_course_behaviour(df, species)
        b["resolution_note"] = (f"read from the same run on {n} time points: the {asked} asked "
                                "for gave too few samples per cycle to place the peaks (the "
                                "plot and sampled_time_series use the asked points)")
    return b


def simulate_model(model_name_or_path: str, t_end: float = 100.0, n_points: int = 200,
                    species: list[str] | None = None,
                    initial_conditions: dict | None = None,
                    parameters: dict | None = None) -> str:
    """
    Tool: simulate a model and return its time series + a saved plot path.
    The model is read from its own file (jaxkineticmodel, or generic_sbml when
    jaxkineticmodel cannot load it).

    species: optional list of species names -- restricts the PLOT to only
      these (the returned data still includes all species).
    initial_conditions: optional {species_name: value} overriding the model's
      default starting concentrations for just those species.
    parameters: optional {parameter_name: value} overriding the model's
      parameter values for this run only (the file is not changed; save_sbml
      writes a changed copy).
    """
    try:
        df, all_species, label, src = _get_trajectory(model_name_or_path, t_end, n_points,
                                                      initial_conditions, parameters)
    except (FileNotFoundError, ValueError, NotImplementedError) as e:
        return json.dumps({"error": str(e)})
    if species:
        unknown = [name for name in species if name not in all_species]
        if unknown:
            return json.dumps({"error": f"Unknown species {unknown} for model '{label}'. "
                                        f"Valid species: {all_species}"})
    sample = df.iloc[:: max(1, len(df) // 20)]   # ~20 rows: the full series stays out of context
    relative_plot_path = _save_plot(df, all_species, f"{label} - simulated dynamics",
                                    label.replace(" ", "_"), species_filter=species)
    changed = src.get("changed") or {}
    return json.dumps({
        "model_file": str(_resolve_sbml_path(model_name_or_path)),
        "source": src["source"],
        "method": src["method"], "backend": src["backend"],
        "model_inputs": src["model_inputs"],
        "t_end": t_end,
        "time_unit": (generic_sbml_time_unit(model_name_or_path) or
                      "not declared: time is in the unit the rate constants use"),
        "n_points": n_points,
        "species": all_species,
        "plotted_species": species if species else all_species,
        "initial_conditions_used": initial_conditions if initial_conditions else "model defaults",
        "parameters_changed": ({k: {"model_value": a, "used": b} for k, (a, b) in changed.items()}
                               or "none (model values)"),
        "final_concentrations": df.iloc[-1].to_dict(),
        "behaviour": resolved_behaviour(
            df, all_species, lambda n: _get_trajectory(model_name_or_path, t_end, n,
                                                       initial_conditions, parameters)[0]),
        # what the run applied beyond the ODEs: events (when they fired), delays, and anything
        # it could not apply
        **{k: src[k] for k in ("events_fired", "delay_differential_equation", "simulation_notes")
           if k in src},
        "sampled_time_series": sample.to_dict(orient="records"),
        "plot_path": relative_plot_path,
        "plot_note": (
            "A real plot was generated and saved to this path. To show it, "
            "embed it in your answer as markdown: "
            f"![{label} dynamics]({relative_plot_path})"
        ),
    }, indent=2, default=float)


def _phase_plane_states(model_name_or_path, x_species, y_species, source_info) -> tuple[list, str | None]:
    """The model's steady states projected on (x, y), for marking on a phase plane:
    [{x, y, stability, kind}] (kind: steady state / negative values / delay model)."""
    from dynalyst.analysis.steady_states import find_steady_states
    changed = {k: new for k, (_, new) in (source_info.get("changed") or {}).items()}
    data = json.loads(find_steady_states(str(_find_sbml(model_name_or_path)),
                                         fixed_params=changed or None))
    if "error" in data:
        return [], f"steady states not marked: {data['error'][:200]}"
    out = []
    for kind, items in (("steady state", data.get("steady_states", [])),
                        ("steady state with negative values (not an amount)",
                         data.get("non_physical_solutions", []))):
        for e in items:
            st = e.get("state") or {}
            if x_species in st and y_species in st:
                out.append({x_species: st[x_species], y_species: st[y_species],
                            "stability": str(e.get("stability", "")).split(" (")[0], "kind": kind})
    return out, None


def plot_phase_plane(model_name_or_path: str, x_species: str, y_species: str,
                      t_end: float = 100.0, n_points: int = 500,
                      initial_conditions: dict | None = None,
                      parameter_overrides: dict | None = None,
                      mark_steady_states: bool = True,
                      extra_trajectories: bool = True) -> str:
    """
    Tool: plot a phase-plane trajectory (one species vs another, not vs time)
    for ANY model file. Works generically
    since it only needs the simulated trajectory. Marks the start (circle)
    and end (square) points so direction of travel is clear; the model's steady
    states (filled: stable, open: unstable; found by find_steady_states); and two more
    trajectories, from the starting point with the plotted pair scaled by 0.5 and 1.5,
    to show the flow around them.
    """
    try:
        df, all_species, model_label, source_info = _get_trajectory(
            model_name_or_path, t_end, n_points, initial_conditions, parameter_overrides
        )
    except ValueError as e:
        return json.dumps({"error": str(e)})

    for sp in (x_species, y_species):
        if sp not in all_species:
            return json.dumps({
                "error": f"Unknown species '{sp}' for model '{model_label}'. Valid: {all_species}"
            })

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_label.replace(" ", "_")
    plot_path = PLOTS_DIR / f"{safe_name}_phase_{x_species}_vs_{y_species}.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    others = []
    if extra_trajectories:
        start = {**{k: float(df[k].iloc[0]) for k in all_species}}
        for f in (0.5, 1.5):
            ic = {**(initial_conditions or {})}
            for k in (x_species, y_species):
                v, size = start[k], max(1.0, float(df[k].abs().max()))
                # a start at (or negligibly near) 0 is moved by a share of the variable's range
                ic[k] = v * f if abs(v) > 1e-9 * size else (f - 0.5) * size
            try:
                d2, *_ = _get_trajectory(model_name_or_path, t_end, n_points, ic, parameter_overrides)
            except (ValueError, NotImplementedError):
                continue
            others.append({x_species: ic[x_species], y_species: ic[y_species]})
            ax.plot(d2[x_species], d2[y_species], "-", color="#9AB8D8", linewidth=1.0, zorder=1)
            ax.plot(d2[x_species].iloc[0], d2[y_species].iloc[0], "o", color="#9AB8D8",
                    markersize=5, zorder=2)
    ax.plot(df[x_species], df[y_species], "-", color="#185FA5", linewidth=1.5, zorder=3)
    marked, mark_note = ([], None)
    if mark_steady_states:
        try:
            marked, mark_note = _phase_plane_states(model_name_or_path, x_species, y_species,
                                                    source_info)
        except Exception as e:          # the plot does not depend on it
            marked, mark_note = [], f"steady states not marked: {type(e).__name__}: {e}"[:200]
    labelled = set()
    for st in marked:
        stable = st["stability"].startswith("stable")
        if st["kind"].startswith("steady state with negative"):
            style, label = dict(marker="X", color="#7A7A7A", markersize=10), "steady state (negative values)"
        elif "not determined" in st["stability"]:
            style, label = dict(marker="D", color="#6B3FA0", markersize=9), "steady state (stability not determined)"
        elif stable:
            style, label = dict(marker="o", color="#111111", markersize=10), "stable steady state"
        else:
            style, label = dict(marker="o", color="#111111", markersize=10, markerfacecolor="white",
                                markeredgewidth=2), "unstable steady state"
        ax.plot(st[x_species], st[y_species], linestyle="none", zorder=6,
                label=None if label in labelled else label, **style)
        labelled.add(label)
    ax.plot(df[x_species].iloc[0], df[y_species].iloc[0], "o", color="#1D9E75",
             markersize=9, label="start", zorder=5)
    ax.plot(df[x_species].iloc[-1], df[y_species].iloc[-1], "s", color="#993C1D",
             markersize=9, label="end", zorder=5)
    ax.set_xlabel(x_species)
    ax.set_ylabel(y_species)
    ax.set_title(f"{model_label} — phase plane: {y_species} vs {x_species}")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    relative_plot_path = config.display_path(plot_path)

    return json.dumps({
        "model": model_label,
        "x_species": x_species,
        "y_species": y_species,
        "start_point": {x_species: float(df[x_species].iloc[0]), y_species: float(df[y_species].iloc[0])},
        "end_point": {x_species: float(df[x_species].iloc[-1]), y_species: float(df[y_species].iloc[-1])},
        **({"steady_states_marked": marked} if marked else {}),
        **({"steady_states_note": mark_note} if mark_note else {}),
        **({"extra_trajectories_from": others} if others else {}),
        "plot_path": relative_plot_path,
        "plot_note": (
            f"A real phase-plane plot was saved. Embed it as: "
            f"![{model_label} phase plane]({relative_plot_path})"
        ),
    }, indent=2, default=float)


def plot_rate_of_change(model_name_or_path: str, species: list[str] | None = None,
                         t_end: float = 100.0, n_points: int = 500,
                         initial_conditions: dict | None = None,
                         parameter_overrides: dict | None = None) -> str:
    """
    Tool: plot the net rate of change (d[species]/dt) over time for ANY
    model — the "rate balance" view: where each species is being produced
    (positive) vs consumed (negative) as the system evolves. Computed by
    numerical differentiation of the simulated trajectory (works generically
    for any model, since it doesn't require access to individual per-reaction
    flux functions — it shows the NET rate for each species, not individual
    reaction rates).
    """
    try:
        df, all_species, model_label, source_info = _get_trajectory(
            model_name_or_path, t_end, n_points, initial_conditions, parameter_overrides
        )
    except ValueError as e:
        return json.dumps({"error": str(e)})

    plot_species = species if species else all_species
    unknown = [s for s in plot_species if s not in all_species]
    if unknown:
        return json.dumps({
            "error": f"Unknown species {unknown} for model '{model_label}'. Valid: {all_species}"
        })

    t = df["time"].to_numpy()
    rates = {sp: np.gradient(df[sp].to_numpy(), t) for sp in plot_species}

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_label.replace(" ", "_")
    plot_path = PLOTS_DIR / f"{safe_name}_rate_of_change.png"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    for sp in plot_species:
        ax.plot(t, rates[sp], label=f"d[{sp}]/dt")
    ax.set_xlabel("Time")
    ax.set_ylabel("Net rate of change")
    ax.set_title(f"{model_label} — rate balance (production above 0, consumption below)")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    relative_plot_path = config.display_path(plot_path)

    peak_rates = {sp: {"max_production": float(np.max(rates[sp])),
                        "max_consumption": float(np.min(rates[sp]))}
                  for sp in plot_species}

    return json.dumps({
        "model": model_label,
        "plotted_species": plot_species,
        "peak_rates": peak_rates,
        "note": (
            "These are NET rates per species (production minus consumption "
            "combined), computed by numerical differentiation of the simulated "
            "trajectory — not individual per-reaction fluxes. Positive = net "
            "production, negative = net consumption at that instant."
        ),
        "plot_path": relative_plot_path,
        "plot_note": (
            f"A real rate-balance plot was saved. Embed it as: "
            f"![{model_label} rate balance]({relative_plot_path})"
        ),
    }, indent=2, default=float)


def compare_initial_conditions(model_name_or_path: str, initial_conditions_list: list,
                                t_end: float = 100.0, n_points: int = 200,
                                species: list[str] | None = None) -> str:
    """
    Tool: run the SAME model from several starting conditions and plot them
    together, optionally restricted to some species -- for any model file. Use it
    to see bistability by simulation (different starts settling at different
    steady states); find_steady_states gives the same answer analytically.

    initial_conditions_list: list of dicts, each with ONLY the species changed
      for that run; the others keep the model's own starting values.
    species: optional species to plot (default: all).
    """
    if not initial_conditions_list:
        return json.dumps({"error": "initial_conditions_list must contain at least one condition dict."})
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", "-.", ":"]
    results, label, plot_species = [], None, None
    for cond_idx, y0_override in enumerate(initial_conditions_list):
        try:
            df, all_species, label, _ = _get_trajectory(model_name_or_path, t_end, n_points,
                                                        y0_override or None)
        except (FileNotFoundError, ValueError) as e:
            plt.close(fig)
            return json.dumps({"error": f"initial_conditions_list[{cond_idx}]: {e}"})
        plot_species = species or all_species
        unknown = [name for name in plot_species if name not in all_species]
        if unknown:
            plt.close(fig)
            return json.dumps({"error": f"Unknown species {unknown} for model '{label}'. "
                                        f"Valid species: {all_species}"})
        label_desc = ", ".join(f"{k}={v}" for k, v in (y0_override or {}).items()) or "model defaults"
        for sp_idx, sname in enumerate(plot_species):
            ax.plot(df["time"], df[sname], color=colors[cond_idx % len(colors)],
                    linestyle=linestyles[sp_idx % len(linestyles)],
                    label=f"{sname} (start: {label_desc})")
        results.append({"initial_condition": y0_override or "model defaults",
                        "final_state": df.iloc[-1].to_dict()})
    plot_path = PLOTS_DIR / f"{label}_compare_conditions.png"
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title(f"{label} - {', '.join(plot_species)} across {len(initial_conditions_list)} "
                 "initial conditions")
    ax.legend(loc="best", fontsize="x-small")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    relative_plot_path = config.display_path(plot_path)
    return json.dumps({
        "model": label,
        "plotted_species": plot_species,
        "n_conditions": len(initial_conditions_list),
        "results": results,
        "plot_path": relative_plot_path,
        "plot_note": (f"A real plot comparing all conditions was saved. Embed it as: "
                      f"![{label} condition comparison]({relative_plot_path})"),
    }, indent=2, default=float)


# --- Tool schema for Claude's tool-use API ---
TOOL_DEFINITIONS = [
    {
        "name": "list_available_models",
        "description": ("The model files available locally: SBML (.xml) and XPP (.ode / .txt), by folder (uploads included)."),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_model_odes",
        "description": (
            "The structure of an SBML model only: species, reactions, stoichiometric matrix "
            "and parameter names (no values). For a model's equations with their values "
            "use get_odes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name_or_path": {
                    "type": "string",
                    "description": "Model name (a file in the model folders, with or without .xml) or full path to an .xml SBML file.",
                }
            },
            "required": ["model_name_or_path"],
        },
    },
    {
        "name": "simulate_model",
        "description": (
            "One deterministic simulation of a model (jaxkineticmodel): the time series and "
            "a concentration-vs-time plot. species restricts the plot to the species named; "
            "initial_conditions and parameters change this run only. A bistable model reaches "
            "ONE of its steady states per run: several starting points on one plot is "
            "compare_initial_conditions; a plot with chosen axes, overlaid conditions or log "
            "axes is custom_plot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name_or_path": {
                    "type": "string",
                    "description": "Model name (a file in the model folders, with or without .xml) or full path to an .xml SBML file.",
                },
                "t_end": {"type": "number", "description": "End time for simulation, in the model's own time unit (inspect_model's time_unit; convert the user's hours / minutes / days to it). Default 100."},
                "n_points": {"type": "integer", "description": "Number of time points. Default 200."},
                "species": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of species names to restrict the PLOT to (e.g. ['S1']). Returned data still includes all species; only the drawn lines are filtered. Omit to plot all species.",
                },
                "initial_conditions": {
                    "type": "object",
                    "description": "Optional dict of {species_name: value} overriding the model's default starting concentrations for just those species, e.g. {'S1': 500}. Species not mentioned keep their normal default.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Optional dict of {parameter_name: value} overriding the model's parameter values for this run only, e.g. {'k2cat': 5}.",
                },
            },
            "required": ["model_name_or_path"],
        },
    },
    {
        "name": "plot_phase_plane",
        "description": (
            "Plot a phase-plane trajectory (one species vs another, NOT vs time) for "
            "ANY model file. Marks start and end points so "
            "direction of travel is visible. Use this when the user asks for a 'phase "
            "plot', 'phase plane', 'phase portrait', or wants to see how two species "
            "relate to each other rather than each vs time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name_or_path": {"type": "string", "description": "Any model name or SBML path."},
                "x_species": {"type": "string", "description": "Species for the x-axis."},
                "y_species": {"type": "string", "description": "Species for the y-axis."},
                "t_end": {"type": "number", "description": "End time for simulation, in the model's own time unit (inspect_model's time_unit; convert the user's hours / minutes / days to it). Default 100."},
                "n_points": {"type": "integer", "description": "Number of time points. Default 500."},
                "initial_conditions": {"type": "object", "description": "Optional {species: value} overrides."},
                "parameter_overrides": {"type": "object", "description": "Optional {parameter: value} overrides."},
            },
            "required": ["model_name_or_path", "x_species", "y_species"],
        },
    },
    {
        "name": "plot_rate_of_change",
        "description": (
            "Plot the net rate of change (d[species]/dt) over time for ANY model — "
            "shows where each species is being produced (positive) vs consumed "
            "(negative) as the system evolves. Use this when the user asks for a "
            "'rate plot', 'rate balance', 'flux plot', or 'production vs consumption'. "
            "NOTE: this gives the NET rate per species (numerically differentiated "
            "from the trajectory), not individual per-reaction fluxes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name_or_path": {"type": "string", "description": "Any model name or SBML path."},
                "species": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Optional list of species to plot rates for. Omit for all species.",
                },
                "t_end": {"type": "number", "description": "End time for simulation, in the model's own time unit (inspect_model's time_unit; convert the user's hours / minutes / days to it). Default 100."},
                "n_points": {"type": "integer", "description": "Number of time points. Default 500 (higher helps numerical differentiation accuracy)."},
                "initial_conditions": {"type": "object", "description": "Optional {species: value} overrides."},
                "parameter_overrides": {"type": "object", "description": "Optional {parameter: value} overrides."},
            },
            "required": ["model_name_or_path"],
        },
    },
    {
        "name": "compare_initial_conditions",
        "description": (
            "Run the SAME model multiple times with DIFFERENT user-specified starting "
            "conditions and plot them together on one graph, optionally restricted to "
            "specific species. Use this to visually confirm bistability/multistability "
            "by simulation, or to plot only a subset of species (e.g. 'plot just one species "
            "for different initial conditions'). Works for any model file. find_steady_states (works for any SBML model) gives "
            "the same bistability answer analytically without needing to guess initial "
            "conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name_or_path": {
                    "type": "string",
                    "description": "Model name or path to an SBML file.",
                },
                "initial_conditions_list": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of dicts, each specifying only the species being changed for that run, e.g. [{'S1': 500, 'S2': 0}, {'S2': 500}]. Missing species use model defaults.",
                },
                "t_end": {"type": "number", "description": "End time for simulation, in the model's own time unit (inspect_model's time_unit; convert the user's hours / minutes / days to it). Default 100."},
                "n_points": {"type": "integer", "description": "Number of time points. Default 200."},
                "species": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of species names to plot (e.g. ['S1']). Omit to plot all species for every condition.",
                },
            },
            "required": ["model_name_or_path", "initial_conditions_list"],
        },
    },
]

TOOL_FUNCTIONS = {
    "list_available_models": lambda **kwargs: list_available_models(),
    "get_model_odes": lambda **kwargs: get_model_odes(**kwargs),
    "simulate_model": lambda **kwargs: simulate_model(**kwargs),
    "plot_phase_plane": lambda **kwargs: plot_phase_plane(**kwargs),
    "plot_rate_of_change": lambda **kwargs: plot_rate_of_change(**kwargs),
    "compare_initial_conditions": lambda **kwargs: compare_initial_conditions(**kwargs),
}

# --- Analysis tools that work generically for ANY-dimensional model (SBML or
# directly-specified ODEs), each in their own module: ---
#   ode_model.py             -- get_odes (SBML->ODE extraction, any dimension)
#   jaxkm_ode_to_sbml.py      -- odes_to_sbml (ODE->SBML conversion, via
#                                jaxkineticmodel's OWN NeuralODEBuild/
#                                SBMLExporter -- replaces the former
#                                hand-rolled libsbml writer in
#                                ode_sbml_convert.py, deleted, which wrote
#                                raw <rateRule> SBML directly instead of
#                                going through the actual jaxkineticmodel
#                                package)
#   steady_states.py         -- find_steady_states (+ stability classification)
#   bifurcation_auto.py      -- run_bifurcation_analysis (real AUTO-07p continuation)
#   pdf_tools.py              -- read_pdf / list_available_pdfs
#
# This replaces ALL previous bifurcation code (bifurcation_1d.py,
# bifurcation_classifier.py, bifurcation_preview.py -- all deleted), which
# only ever handled strictly 1D models via hand-rolled symbolic root-finding.
# AUTO-07p (https://github.com/auto-07p/auto-07p) now does real numerical
# pseudo-arclength continuation for a model of any dimension.
from dynalyst.analysis import bifurcation_auto
from dynalyst.analysis import jaxkm_ode_to_sbml
from dynalyst.analysis import ode_model
from dynalyst.analysis import pdf_tools
from dynalyst.analysis import steady_states

for _module in (ode_model, jaxkm_ode_to_sbml, steady_states, bifurcation_auto, pdf_tools):
    TOOL_DEFINITIONS += _module.TOOL_DEFINITIONS
    TOOL_FUNCTIONS.update(_module.TOOL_FUNCTIONS)

# IMPORTANT: keep the bifurcation dispatch dynamic.  The old implementation
# copied a lambda that captured the then-current run_bifurcation_analysis
# function object.  Reloading/replacing bifurcation_auto.py therefore did not
# change what the Ask agent executed: direct Python calls used the new module,
# while agent tool calls could continue invoking the stale function object.
# Resolve the function from the live module at call time instead.
TOOL_FUNCTIONS["run_bifurcation_analysis"] = (
    lambda **kwargs: bifurcation_auto.run_bifurcation_analysis(**kwargs)
)
