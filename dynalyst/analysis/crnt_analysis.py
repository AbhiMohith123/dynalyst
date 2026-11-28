"""Chemical reaction network theory (crnt4sbml): can the network be bistable?

1. The model (SBML / XPP / typed ODEs) is turned into an explicit mass-action
   network (analysis/crnt_network.py) and written as the SBML crnt4sbml
   reads (outputs/crnt/<model>_crnt.xml).
2. crnt4sbml reports its structure (species, complexes, reactions,
   deficiency, linkage classes, weak reversibility, conservation laws) and
   checks the deficiency zero / one theorems. If either holds, the network
   cannot have more than one positive steady state for ANY rate constants.
3. Otherwise its mass-conservation optimisation searches for rate constants
   and concentrations at which several steady states exist.
4. Every parameter set it returns is checked here independently: the full
   steady state is rebuilt, the conservation totals taken from it, and all
   steady states in that class are found (find_steady_states); only sets
   with two or more stable states count as bistable. Each verified set is
   saved as a model file for AUTO (run_bifurcation_analysis).
"""
import contextlib
import io
import json
import re
import time

import numpy as np

from dynalyst import config
from dynalyst.analysis.crnt_network import (NotMassAction, check_conversion, crnt_dir,
                                            import_crnt4sbml,
                                            law_totals, reconstruct, write_crnt_sbml,
                                            write_reduced_sbml)
from dynalyst.analysis.ode_model import resolve_model


class CrntStopped(RuntimeError):
    """crnt4sbml gave up on the network (it calls sys.exit() when a method
    does not apply); the message is what it printed."""


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            out = fn(*a, **kw)
    except SystemExit:
        lines = [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip()]
        raise CrntStopped(" ".join(lines[-4:]) or "crnt4sbml stopped without a message") from None
    return out, buf.getvalue()


def crnt_bistability(model_name_or_path: str | None = None, species: list | None = None,
                     odes: dict | None = None, parameters: dict | None = None,
                     initial_conditions: dict | None = None, iterations: int = 20,
                     max_candidates: int = 5, seed: int = 0, structure_only: bool = False,
                     semi_diffusive: bool = False, draw_network: bool = False,
                     job_id: str | None = None, _in_worker: bool = False,
                     _pause_after: float | None = None) -> str:
    """Tool: CRNT analysis of whether a model's reaction network can have several
    steady states (crnt4sbml) -- CRNT only: no continuation, no stability check, and
    the network graphs only with draw_network.
    The search runs in a worker process limited to config.CRNT_MAX_SECONDS:
    crnt4sbml's symbolic steps grow very fast with network size and cannot be
    interrupted inside this process.
    job_id: the result of a search an earlier call left running (still_running), waited for.
    _pause_after (set by the agent when a person is chatting, never by a model): after this
    many seconds a search not finished yet is left running and reported as still_running."""
    if job_id:
        return _collect_job(str(job_id))
    if not _in_worker and (config.CRNT_MAX_SECONDS > 0 or _pause_after):
        # also for the structure alone: crnt4sbml's own construction of a network can hang
        return _with_time_limit(dict(
            model_name_or_path=model_name_or_path, species=species, odes=odes,
            parameters=parameters, initial_conditions=initial_conditions, iterations=iterations,
            max_candidates=max_candidates, seed=seed, semi_diffusive=semi_diffusive,
            draw_network=draw_network), structure_only=structure_only, pause_after=_pause_after)
    t0 = time.time()
    try:
        m = resolve_model(model_name_or_path, species, odes, parameters, initial_conditions)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    label = m.get("label") or "user_model"
    try:
        net = reconstruct(m["state_names"], m["rhs"], m["param_names"], m["param_defaults"], m["y0"])
    except NotMassAction as e:
        return json.dumps({
            "model": label, "mass_action": False, "bistable": None,
            "conclusion": f"CRNT does not apply: the model is not a mass-action reaction network "
                          f"({e}). This is no verdict on bistability: the model may still be bistable, "
                          "which a bifurcation diagram can check.",
        })
    crnt_path = write_crnt_sbml(net, crnt_dir() / f"{label}_crnt.xml")
    converted = converted_model(m, net, crnt_path)
    try:
        return _analyse(label, net, crnt_path, iterations, max_candidates, seed, t0,
                        structure_only, converted, draw_network)
    except CrntStopped as e:
        return json.dumps({
            "model": label, "mass_action": True, "crnt_sbml": config.display_path(crnt_path),
            "converted_model": converted,
            "bistable": None, "conclusion": (
                f"crnt4sbml cannot analyse this network ({explain_refusal(str(e))}). Bistability "
                "is neither shown nor ruled out.")},
            indent=2, default=float)


# CRNT searches running as background jobs (job id -> process): a chat can pause and ask the
# user whether to wait for a long search, and pick its result up in its next request
_JOBS: dict = {}


def _job_dir(job_id: str):
    # the id reaches shutil.rmtree: only an id _start_job made (12 hex characters), so it
    # can never name a folder outside jobs/
    if not re.fullmatch(r"[0-9a-f]{12}", str(job_id)):
        raise ValueError(f"'{job_id}' is not a CRNT job id")
    return crnt_dir() / "jobs" / job_id


def _start_job(kwargs) -> str:
    """Start the analysis in a worker process; its files live in the job's folder."""
    import os
    import subprocess
    import sys
    import uuid
    job_id = uuid.uuid4().hex[:12]
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "args.json").write_text(json.dumps(kwargs, default=str), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(config.PROJECT_ROOT)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
    code = ("import sys; from dynalyst.analysis.crnt_analysis import _worker_main; "
            "_worker_main(*sys.argv[1:])")
    # stderr to a file, not a pipe: a long search could fill a pipe nobody reads and block
    popen = dict(env=env, cwd=str(config.PROJECT_ROOT), stdout=subprocess.DEVNULL,
                 stderr=open(d / "stderr.txt", "w", encoding="utf-8", errors="replace"))
    if os.name == "nt":
        popen["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen["start_new_session"] = True
    proc = subprocess.Popen([sys.executable, "-c", code, str(d / "args.json"),
                             str(d / "partial.json"), str(d / "result.json")], **popen)
    (d / "job.json").write_text(json.dumps({"started": time.time(), "pid": proc.pid,
                                            "model": kwargs.get("model_name_or_path")}),
                                encoding="utf-8")
    _JOBS[job_id] = proc
    return job_id


def _job_started(job_id: str) -> float:
    try:
        return float(json.loads((_job_dir(job_id) / "job.json").read_text(encoding="utf-8"))["started"])
    except (OSError, ValueError, KeyError):
        return time.time()


def _wait_job(job_id: str, seconds: float) -> bool:
    """Wait up to `seconds` for the job; True when it has finished."""
    import subprocess
    proc = _JOBS.get(job_id)
    if proc is None:      # started by another process: its result file says it is done
        end = time.time() + max(0.0, seconds)
        while not (_job_dir(job_id) / "result.json").exists() and time.time() < end:
            time.sleep(2)
        return (_job_dir(job_id) / "result.json").exists()
    try:
        proc.wait(timeout=max(0.0, seconds))
        return True
    except subprocess.TimeoutExpired:
        return False


def stop_crnt_job(job_id: str) -> None:
    """Stop a running CRNT search (e.g. the user chose not to wait for it)."""
    proc = _JOBS.pop(job_id, None)
    if proc is not None:
        _kill_tree(proc)
        return
    try:
        pid = int(json.loads((_job_dir(job_id) / "job.json").read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError):
        return
    import os
    import subprocess
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, timeout=60)
        else:
            import signal
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        pass


def _job_partial(job_id: str) -> dict:
    f = _job_dir(job_id) / "partial.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except (OSError, ValueError):
        return {}


def _network_words(res: dict) -> str:
    net = res.get("network") or {}
    return (f" ({net.get('n_species')} species, {net.get('n_reactions')} reactions, "
            f"{net.get('conservation_laws')} conservation law(s))") if net else ""


def _job_outcome(job_id: str) -> str:
    """The finished job's result, or why it stopped without one."""
    import shutil
    d = _job_dir(job_id)
    _JOBS.pop(job_id, None)
    if (d / "result.json").exists():
        out = (d / "result.json").read_text(encoding="utf-8")
        shutil.rmtree(d, ignore_errors=True)     # the result is returned; sets are saved elsewhere
        return out
    try:
        err = (d / "stderr.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        err = ""
    shutil.rmtree(d, ignore_errors=True)
    # the exception line first: a traceback tail cut short shows only its carets
    lines = [ln.strip() for ln in err.splitlines() if ln.strip()]
    cause = next((ln for ln in reversed(lines) if re.match(r"[\w.]*(?:Error|Exception)\b", ln)),
                 lines[-1] if lines else "no message")
    return json.dumps({"error": f"The CRNT analysis stopped without a result: {cause}",
                       "traceback_tail": "\n".join(lines[-12:])})


def _timed_out(job_id: str, limit: float) -> str:
    """Stop the job and report the structure it got to, with the reason."""
    stop_crnt_job(job_id)
    res = _job_partial(job_id)
    net = res.get("network") or {}
    # how far crnt4sbml got: its full structure (deficiency ...) is there only when it
    # finished building the network
    stage = ("crnt4sbml's search for rate constants with several steady states"
             if "deficiency" in net else
             "crnt4sbml's own construction of the network (a symbolic step it runs "
             "before any analysis)")
    res.update(bistable=None, timed_out=True, seconds=round(time.time() - _job_started(job_id), 1),
               conclusion=(
        f"Not decided: {stage} did not finish within {limit / 60:g} min on this "
        f"network{_network_words(res)}. Its symbolic method grows very fast with network size. "
        "Bistability is neither shown nor ruled out."))
    import shutil
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    return json.dumps(res, indent=2, default=float)


def _with_time_limit(kwargs, structure_only=False, pause_after=None) -> str:
    """Run the analysis in a worker process; past config.CRNT_MAX_SECONDS stop it and
    report the network structure (written by the worker first) with the reason.
    structure_only: the worker stops after the structure. pause_after (a person is
    chatting): after this many seconds the search is left running and the result says
    so (still_running, job_id), so the user can choose whether to wait for it."""
    job_id = _start_job({**kwargs, "structure_only": structure_only})
    if pause_after and not structure_only:
        if _wait_job(job_id, pause_after):
            return _job_outcome(job_id)
        res = _job_partial(job_id)
        waited = time.time() - _job_started(job_id)
        res.update(still_running=True, job_id=job_id, bistable=None, seconds=round(waited, 1),
                   conclusion=(
            f"Still searching after {waited / 60:.0f} min on this network{_network_words(res)}: "
            "crnt4sbml's search for rate constants with several steady states has not finished. "
            "It keeps running until the user decides whether to wait for it."))
        return json.dumps(res, indent=2, default=float)
    limit = float(config.CRNT_MAX_SECONDS)
    if _wait_job(job_id, limit):
        return _job_outcome(job_id)
    return _timed_out(job_id, limit)


def _collect_job(job_id: str) -> str:
    """The result of a search left running: waited for, up to
    config.CRNT_CONTINUE_MAX_SECONDS from its start."""
    if not (_job_dir(job_id) / "job.json").exists():
        return json.dumps({"error": f"No CRNT search with job_id '{job_id}' exists. Run "
                                    "crnt_bistability on the model again."})
    limit = float(config.CRNT_CONTINUE_MAX_SECONDS)
    if _wait_job(job_id, limit - (time.time() - _job_started(job_id))):
        return _job_outcome(job_id)
    return _timed_out(job_id, limit)


def _kill_tree(proc):
    """Stop the worker and everything it started (e.g. an AUTO run)."""
    import os
    import signal
    import subprocess
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=60)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()


def _worker_main(args_file, partial_file, result_file):
    """In the worker: the network's size from this project's own reconstruction,
    then crnt4sbml's structure, each kept if the worker is stopped; then the full
    analysis."""
    from pathlib import Path
    kwargs = json.loads(Path(args_file).read_text(encoding="utf-8"))
    try:
        Path(partial_file).write_text(json.dumps(_network_size(kwargs)), encoding="utf-8")
    except Exception:
        pass     # crnt_bistability below reports why the model cannot be read
    only_structure = kwargs.pop("structure_only", False)
    draw = kwargs.pop("draw_network", False)
    structure = crnt_bistability(**kwargs, structure_only=True, draw_network=draw and only_structure,
                                 _in_worker=True)
    Path(partial_file).write_text(structure, encoding="utf-8")
    Path(result_file).write_text(structure if only_structure else crnt_bistability(
        **kwargs, draw_network=draw, _in_worker=True), encoding="utf-8")


def _network_size(kwargs) -> dict:
    """Species, reactions and (exact) number of conservation laws of the
    mass-action network, from this project's own reconstruction: seconds, where
    crnt4sbml's construction of a large network can take far longer."""
    import sympy as sp
    m = resolve_model(kwargs.get("model_name_or_path"), kwargs.get("species"), kwargs.get("odes"),
                      kwargs.get("parameters"), kwargs.get("initial_conditions"))
    net = reconstruct(m["state_names"], m["rhs"], m["param_names"], m["param_defaults"], m["y0"])
    S = sp.Matrix([[r["products"].get(s, 0) - r["reactants"].get(s, 0) for r in net["reactions"]]
                   for s in net["species"]])
    return {"model": m.get("label") or "user_model", "mass_action": True,
            "network": {"n_species": len(net["species"]), "n_reactions": len(net["reactions"]),
                        "conservation_laws": len(net["species"]) - int(S.rank()),
                        "reactions": [f"{r['id']}: " + _complex(r["reactants"]) + " -> "
                                      + _complex(r["products"]) for r in net["reactions"]],
                        "restored_species": net["hidden"]}}


def converted_model(m, net, crnt_path) -> dict:
    """The converted mass-action model as crnt4sbml reads it, read back from
    the SBML file that was written: its ODEs, its reactions, the value of every
    rate constant with what it stands for in the original model, and a check
    that it reproduces the original ODEs (crnt_network.check_conversion)."""
    from dynalyst.analysis.ode_model import model_from_sbml, ode_report
    report = ode_report(model_from_sbml(str(crnt_path)))
    for k in ("model", "source", "source_path", "time_dependent_inputs"):
        report.pop(k, None)
    report["rate_constants"] = {
        f"k_{r['id']}": {"value": r["k_value"], "stands_for": r["rate_constant"],
                         "original_parameters": r["model_parameters"]}
        for r in net["reactions"]}
    report.pop("parameter_meanings", None)          # the same information, now per constant
    if net.get("hidden"):
        report["restored_species"] = {h: f"{e} (restored from a conservation law)"
                                      for h, e in net["hidden"].items()}
    report["original_parameters"] = {p: m["param_defaults"][p] for p in m["param_names"]
                                     if any(p in r["model_parameters"] for r in net["reactions"])}
    try:
        report["conversion_check"] = check_conversion(net, m["state_names"], m["rhs"],
                                                      m["param_names"])
    except Exception as e:   # a failed check is reported, never a failed analysis
        report["conversion_check"] = {"reproduces_original_odes": None,
                                      "error": f"{type(e).__name__}: {e}"}
    return report


def _versions() -> dict:
    """The library versions this result was computed with (crnt4sbml pins far older ones)."""
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ("crnt4sbml", "sympy", "numpy", "scipy", "networkx", "python-libsbml"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _as_written_for(fn):
    """Run crnt4sbml under the sympy behaviour it was written for (sympy 1.4). Since sympy
    1.6, row reduction simplifies every intermediate product (dotprodsimp); on crnt4sbml's
    symbolic matrices (rate constants as symbols) that expands without end, so even building
    a 20-species network does not finish. Without it the results are the same."""
    import functools
    from sympy.matrices.utilities import dotprodsimp

    @functools.wraps(fn)
    def run(*a, **kw):
        with dotprodsimp(False):
            return fn(*a, **kw)
    return run


@_as_written_for
def _analyse(label, net, crnt_path, iterations, max_candidates, seed, t0, structure_only=False,
             converted=None, draw_network=False):
    crnt4sbml = import_crnt4sbml()
    network, _ = _quiet(crnt4sbml.CRNT, str(crnt_path))
    cg = network.get_c_graph()
    ld = network.get_low_deficiency_approach()
    species, complexes = [str(x) for x in cg.get_species()], [str(x) for x in cg.get_complexes()]
    reactions = [str(x) for x in cg.get_reactions()]
    S, Y = cg.get_s(), cg.get_y()
    dim_class = _quiet(cg.get_network_dimensionality_classification)[0]
    structure = {
        "n_species": len(species), "n_complexes": len(complexes), "n_reactions": len(reactions),
        "linkage_classes": len(cg.get_linkage_classes()),
        "stoichiometric_rank": int(S.rank()),
        # deficiency = complexes - linkage classes - rank(S)
        "deficiency": int(cg.get_deficiency()),
        "linkage_class_deficiencies": [int(d) for d in cg.get_linkage_classes_deficiencies()],
        "terminal_strong_linkage_classes_per_linkage_class":
            [int(x) for x in cg.get_number_of_terminal_strong_lc_per_lc()],
        "weakly_reversible": bool(cg.get_if_cgraph_weakly_reversible()),
        "weakly_reversible_per_linkage_class":
            [bool(x) for x in cg.get_weak_reversibility_of_linkage_classes()],
        "conservation_laws": len(species) - int(S.rank()),
        "equilibrium_manifold_dimension": int(cg.get_dim_equilibrium_manifold()),
        "dimensionality": dim_class[0] if isinstance(dim_class, (list, tuple)) else str(dim_class),
    }
    matrices = {
        "species_order": species, "complex_order": complexes, "reaction_order": reactions,
        "stoichiometric_matrix": [[int(v) for v in S.row(i)] for i in range(S.rows)],
        "complex_composition_matrix_Y": [[int(v) for v in Y.row(i)] for i in range(Y.rows)],
    }
    out = {
        "model": label, "mass_action": True, "crnt_sbml": config.display_path(crnt_path),
        "network": {"reactions": [f"{r['id']}: " + _complex(r["reactants"]) + " -> " +
                                  _complex(r["products"]) + f"   (k = {r['rate_constant']})"
                                  for r in net["reactions"]],
                    "restored_species": net["hidden"], **structure},
        "converted_model": converted,
        "matrices": matrices,
        "environment": _versions(),
    }
    if draw_network:
        # one connected picture of the network (species and reactions as nodes) ...
        out["reaction_graph_plot"] = _plot_reaction_graph(
            net, crnt_dir() / f"{label}_reaction_graph.png")
        # ... and CRNT's complex graph, one island per linkage class by definition
        out["complex_graph_plot"] = _plot_c_graph(cg, crnt_dir() / f"{label}_complex_graph.png")
    def0 = bool(_quiet(ld.does_satisfy_deficiency_zero_theorem)[0])
    def1 = bool(_quiet(ld.does_satisfy_deficiency_one_theorem)[0])
    out["deficiency_zero_theorem"], out["deficiency_one_theorem"] = def0, def1
    if structure_only:
        out.update(bistable=False if def0 or def1 else None, proof=bool(def0 or def1), conclusion=(
            "Network structure reported" + (
                "; the deficiency {} theorem rules out multiple positive steady states within a stoichiometric compatibility class under its mass-action assumptions."
                .format("zero" if def0 else "one") if def0 or def1 else
                "; the deficiency theorems do not decide bistability (run without "
                "structure_only to search for bistable rate constants).")))
        out["seconds"] = round(time.time() - t0, 1)
        return json.dumps(out, indent=2, default=float)
    if def0 or def1:
        out.update(bistable=False, proof=True, conclusion=(
            f"The network satisfies the deficiency {'zero' if def0 else 'one'} theorem, so it "
            "cannot have more than one positive steady state within a stoichiometric "
            "compatibility class under its mass-action assumptions. This conclusion is "
            "about positive steady states, not all boundary states or all attractors."))
        out["seconds"] = round(time.time() - t0, 1)
        return json.dumps(out, indent=2, default=float)

    reaction_ids = {r["id"] for r in net["reactions"]}
    try:
        approach, log = _quiet(network.get_mass_conservation_approach)
        why_not = None if approach is not None else "it returned no approach"
    except Exception as e:   # crnt4sbml stops (CrntStopped) or raises when the method cannot apply
        approach, log, why_not = None, "", str(e)
    if approach is not None:
        method = "mass-conservation"
        out["conservation_laws"] = [ln.strip() for ln in
                                    str(approach.get_conservation_laws()).splitlines() if ln.strip()]
        bounds, cbounds = _quiet(approach.get_optimization_bounds)[0]
        decision = [str(x) for x in approach.get_decision_vector()]
        cspecies = [str(x) for x in approach.get_concentration_bounds_species()]
        funs = approach.get_concentration_funs()
        names = [str(s_) for s_ in cg.get_species()]

        def search(scope):
            b, cb = ((_near_model(decision, bounds, net), _near_model(cspecies, cbounds, net))
                     if scope == "model" else (bounds, cbounds))
            (params, objs), _ = _quiet(approach.run_optimization, bounds=b,
                                       concentration_bounds=cb, iterations=iterations, seed=seed)
            found = []
            for x, o in zip(params, objs):
                x = np.asarray(x, dtype=float)
                found.append((float(o),
                              {d: float(v) for d, v in zip(decision, x) if d in reaction_ids},
                              {n: float(f(*x)) for n, f in zip(names, funs)}))
            return {"candidates": found}
        out["optimisation"] = {"method": method, "iterations": iterations,
                               "decision_vector": decision}
    else:
        mc_reason = explain_refusal(why_not, cg)
        out["mass_conservation_approach"] = f"does not apply: {mc_reason}"
        if structure["conservation_laws"] == 0:
            # no conservation law: the semi-diffusive method, written for open networks
            sd = _semi_diffusive(network, iterations, seed, cg)
            if sd.get("bistable") is None and not sd.get("multistationarity_possible"):
                sd["conclusion"] = (f"crnt4sbml cannot decide bistability for this network. Its "
                                    f"mass-conservation method does not apply ({mc_reason}); "
                                    + sd["conclusion"][0].lower() + sd["conclusion"][1:])
            out.update(sd, crnt_log=log[-400:])
            out["seconds"] = round(time.time() - t0, 1)
            return json.dumps(out, indent=2, default=float)
        # the general approach needs conservation laws but not a uniterminal network
        method = "general"

        def search(scope):
            return _general(network, cg, iterations, seed, net if scope == "model" else None)
        out["optimisation"] = {"method": method, "iterations": iterations}

    # First near the model's own values (rate constants within a factor of NEAR_RATE_FACTOR,
    # concentrations on the scale of its totals); only when nothing turns up there, over
    # crnt4sbml's generic ranges, which say what the network can do, not what this model does.
    model_totals = law_totals(net, net.get("initial") or {})
    found_by, sets, scope = {}, [], None
    for scope in ("model", "generic"):
        res = search(scope)
        if res.get("error"):
            if scope == "model":
                out.update(bistable=None, conclusion=(
                    f"crnt4sbml cannot decide bistability for this network. Its mass-conservation "
                    f"method does not apply ("
                    f"{out.get('mass_conservation_approach', '').removeprefix('does not apply: ')}"
                    f"), and its general approach could not be run ({res['error']})."))
                out["seconds"] = round(time.time() - t0, 1)
                return json.dumps(out, indent=2, default=float)
            break
        if res.get("summary", {}).get("rate_bounds_note"):
            out["optimisation"]["rate_bounds_note"] = res["summary"]["rate_bounds_note"]
        found_by[SEARCH_RANGES[scope]] = len(res["candidates"])
        sets = _report_sets(label, net, res["candidates"], max_candidates, method,
                            SEARCH_RANGES[scope], model_totals)
        if sets:
            break
    out["optimisation"]["parameter_sets_found"] = sum(found_by.values())
    out["optimisation"]["by_search_range"] = found_by
    out["model_totals"] = model_totals
    out["multistationary_sets"] = sets
    where = "its mass-conservation method" if method == "mass-conservation" else (
        "its general approach (its mass-conservation method does not apply)")
    if sets:
        def describe(i, v):
            compare = ", ".join(f"{t} = {c['this_set']:.6g} (model {c['model']:.6g})"
                                for t, c in v["compared_with_model"]["totals"].items()
                                if c["model"] is not None)
            factor = v["compared_with_model"]["largest_rate_constant_factor"]
            return (f"set {i}: {compare}" + (f"; rate constants up to {factor:.3g}x from the "
                                             f"model's" if factor else "")
                    + f"; saved as {v['model_file']}")
        found_text = (
            f"crnt4sbml found {len(sets)} candidate bistability parameter set(s): rate constants "
            "and concentrations at which the network sits on a fold of its steady states, so "
            "that nearby it has more than one positive steady state. "
            + " | ".join(describe(i, v) for i, v in enumerate(sets, start=1))
            + ". They are candidates: whether two of the steady states are stable (bistability) "
            "is confirmed by running AUTO-07p on a set's file, along one of its totals.")
        if scope == "model":
            out.update(bistable=None, multistationary=True, near_model=True, conclusion=(
                f"Candidate bistability near the model's own values: with {where}, "
                f"searching rate constants within a factor of {NEAR_RATE_FACTOR:g} of the model's "
                f"and concentrations on the scale of its totals, {found_text}"))
        else:
            out.update(bistable=None, multistationary=True, near_model=False, conclusion=(
                f"Candidate bistability for the reaction network, but not near the model's own "
                f"values: with {where}, nothing turned up with rate constants "
                f"within a factor of {NEAR_RATE_FACTOR:g} of the model's; over crnt4sbml's "
                f"generic ranges (rate constants by reaction type, concentrations 0.5 to 5e5) "
                f"{found_text} These are not the model's values: they show what the network can "
                "do, not what this model does."))
    else:
        out.update(bistable=None, proof=False, multistationary=None, search_status='no_witness_found', conclusion=(
            f"No candidate bistability parameter set was found: with {where}, near the "
            "model's own values or over crnt4sbml's generic ranges. This search does not "
            "establish presence or absence of multistationarity or bistability."))
    out["seconds"] = round(time.time() - t0, 1)
    return json.dumps(out, indent=2, default=float)


def _report_sets(label, net, candidates, max_candidates, method, search_range, model_totals):
    """crnt4sbml's parameter sets (best objective first, all concentrations positive), each
    saved as a model file with one parameter per conservation total, so a bifurcation
    analysis can continue a total from it."""
    own_k = {r["id"]: float(r["k_value"]) for r in net["reactions"]
             if r.get("k_value") is not None and float(r["k_value"]) > 0}
    sets = []
    for obj, ks, conc in sorted(candidates, key=lambda c: c[0]):
        if len(sets) >= min(max_candidates, 3):
            break
        if not all(np.isfinite(v) and v > 0 for v in conc.values()):
            continue
        path, totals = write_reduced_sbml(
            net, crnt_dir() / f"{label}_crnt_set_{len(sets) + 1}.xml", ks, conc)
        factors = [max(v / own_k[r], own_k[r] / v) for r, v in ks.items() if r in own_k and v > 0]
        sets.append({
            "method": method, "search_range": search_range, "objective": obj,
            "rate_constants": ks, "model_parameters": _model_parameter_values(net, ks),
            "steady_state": conc, "conservation_totals": totals,
            "compared_with_model": {
                "largest_rate_constant_factor": max(factors) if factors else None,
                "totals": {t: {"this_set": v, "model": model_totals.get(t)}
                           for t, v in totals.items()}},
            "model_file": config.display_path(path),
            # read back from that file: what run_bifurcation_analysis will use
            **_saved_model_values(path)})
    return sets


# Where crnt4sbml searches: near the model's own values first (each rate constant within a
# factor of NEAR_RATE_FACTOR of the model's; each concentration between NEAR_CONC_SPAN times
# the model's largest conservation total), then its own generic ranges.
NEAR_RATE_FACTOR = 10.0
NEAR_CONC_SPAN = (1e-4, 10.0)
SEARCH_RANGES = {"model": "near the model's own values", "generic": "crnt4sbml's generic ranges"}


def _model_scale(net) -> tuple[dict, dict]:
    """The model's own values: its rate constant per reaction, and the scale of its
    concentrations (its largest conservation total, else its largest initial value) for
    every species. One scale for all: the totals themselves are what a bifurcation
    analysis varies, so a species is not held to the total of its own law."""
    ks = {r["id"]: float(r["k_value"]) for r in net["reactions"]
          if r.get("k_value") is not None and float(r["k_value"]) > 0}
    init = net.get("initial") or {}
    totals = [v for v in law_totals(net, init).values() if v > 0]
    level = max(totals) if totals else max([float(v) for v in init.values()] + [0.0])
    return ks, ({s_: level for s_ in net["species"]} if level > 0 else {})


def _near_model(names, generic, net) -> list:
    """Search bounds near the model's own values for crnt4sbml's variables (`names`, in
    its order); a variable without a model value keeps crnt4sbml's bound."""
    ks, scale = _model_scale(net)
    out = []
    for n, b in zip(names, generic):
        if n in ks:
            out.append((ks[n] / NEAR_RATE_FACTOR, ks[n] * NEAR_RATE_FACTOR))
        elif n in scale:
            out.append((scale[n] * NEAR_CONC_SPAN[0], scale[n] * NEAR_CONC_SPAN[1]))
        else:
            out.append(b)
    return out


# crnt4sbml's own rate ranges (complex formation .. catalysis): the bounds used for a reaction
# whose kind it could not classify
_UNTYPED_RATE_BOUNDS = (1e-8, 1.0)


def _general(network, cg, iterations, seed, near=None) -> dict:
    """crnt4sbml's general approach: rate constants and concentrations for which the
    network has several steady states, for networks with conservation laws that the
    mass-conservation method rejects (e.g. not uniterminal). `near`: the network whose
    own values bound the search (_near_model); None for crnt4sbml's generic ranges.
    Returns {"candidates": [(objective, rate constants, concentrations)], "summary"}
    or {"error"}."""
    names = [str(s_) for s_ in cg.get_species()]
    rids = [str(r) for r in cg.get_reactions()]
    try:
        ga, _ = _quiet(network.get_general_approach)
        _quiet(ga.initialize_general_approach, signal="C1", response=names[0], fix_reactions=True)
        bounds = _quiet(ga.get_optimization_bounds)[0]
        untyped = [str(v) for v, b in zip(ga.get_input_vector(), bounds) if b is None]
        bounds = [b if b is not None else _UNTYPED_RATE_BOUNDS for b in bounds]
        if near is not None:
            bounds = _near_model([str(v) for v in ga.get_input_vector()], bounds, near)
            untyped = [u for u in untyped if u not in _model_scale(near)[0]]
        (params, objs), _ = _quiet(ga.run_optimization, bounds=bounds, iterations=iterations,
                                   seed=seed)
    except CrntStopped as e:
        return {"error": explain_refusal(str(e), cg)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    order = [str(v) for v in ga.get_input_vector()]
    candidates = []
    for x, o in zip(params, objs):
        vals = dict(zip(order, np.asarray(x, dtype=float)))
        candidates.append((float(o), {r: float(vals[r]) for r in rids if r in vals},
                           {n: float(vals[n]) for n in names if n in vals}))
    summary = {"method": "general", "iterations": iterations,
               "parameter_sets_found": len(candidates)}
    if untyped:
        summary["rate_bounds_note"] = (f"crnt4sbml could not classify {untyped}; they were "
                                       f"searched over its full rate range {_UNTYPED_RATE_BOUNDS}")
    return {"candidates": candidates, "summary": summary}

SEMI_DIFFUSIVE_ZERO = 1e-12   # objective value counted as zero (a multistationarity witness)


def explain_refusal(message, cg=None) -> str:
    """crnt4sbml's terse refusal, in plain words, naming the part of the network
    that breaks the requirement where it can."""
    msg = str(message or "")
    low = msg.lower()
    if "uniterminal" in low:
        detail = _non_uniterminal_detail(cg) if cg is not None else ""
        return ("the network is not uniterminal: the method needs every connected part of the "
                "reaction network to end in exactly ONE terminal group of complexes (a group "
                "the reactions can reach but never leave)" + (f"; {detail}" if detail else ""))
    if "rank" in low and ("greater than zero" in low or "species" in low):
        return ("the network has no conservation law: every species can be produced or "
                "removed, so no total amount stays fixed, and the method is built on fixed "
                "totals")
    if "boundary species is not present" in low:
        return ("the network has no inflow or outflow (nothing is produced from or removed to "
                "the environment), which the semi-diffusive method needs")
    if "conservation laws present" in low:
        return ("the network has conservation laws, and the semi-diffusive method only applies "
                "to networks without any")
    return msg or "crnt4sbml gave no reason"


def _inflow_species(cg) -> set:
    """Species produced directly from a boundary complex (inflow from the environment)."""
    try:
        g = cg.get_graph()
        out = set()
        for a, b in g.edges:
            if g.nodes[a].get("species_bc") and not g.nodes[b].get("species_bc"):
                out |= {str(x) for x in g.nodes[b].get("species") or []}
        return out
    except Exception:
        return set()


def _non_uniterminal_detail(cg) -> str:
    """Which connected parts end in more than one terminal group, and in which."""
    import networkx
    try:
        parts = []
        for lc in cg.get_linkage_classes():
            # the boundary complex is the environment (inflow / outflow), not a species
            name = lambda c: ("the environment" if lc.nodes[c].get("species_bc") else c) or "0"
            cond = networkx.condensation(lc)
            terminal = [cond.nodes[n]["members"] for n in cond.nodes if cond.out_degree(n) == 0]
            if len(terminal) > 1:
                groups = [" / ".join(sorted(name(c) for c in group)) for group in terminal]
                parts.append("the part containing " + ", ".join(sorted(
                                 name(c) for c in lc.nodes if not lc.nodes[c].get("species_bc")))
                             + " ends in " + " and in ".join("{" + g + "}" for g in groups))
        return "; ".join(parts[:3]) + (f" (and {len(parts) - 3} more)" if len(parts) > 3 else "")
    except Exception:
        return ""


def _semi_diffusive(network, iterations, seed, cg=None) -> dict:
    """crnt4sbml's semi-diffusive approach, for networks without a conservation
    law (open networks with inflow / outflow). Its optimisation runs over the
    reaction FLUXES; an objective of zero means a flux vector exists for which the
    network has several positive steady states. It returns no rate constants and
    the continuation it offers needs AUTO 2000 through PyDSTool, so this is
    evidence that the network CAN be multistationary, not a confirmed bistable
    parameter set."""
    fed = _inflow_species(cg) if cg is not None else set()
    if cg is not None and fed and fed >= {str(x) for x in cg.get_species()}:
        # crnt4sbml's method splits the species into those fed from the environment and
        # the rest, and solves for the rest: with none left it fails internally
        return {"bistable": None, "approach": "semi-diffusive",
                "conclusion": "Its semi-diffusive method does not apply either: every species "
                              "receives inflow directly from the environment, a case the method "
                              "does not handle (it needs at least one species without direct "
                              "inflow)."}
    try:
        approach, log = _quiet(network.get_semi_diffusive_approach)
        if approach is None:
            return {"bistable": None, "approach": "semi-diffusive",
                    "conclusion": "Its semi-diffusive method does not apply either.",
                    "crnt_log": log[-400:]}
        bounds = _quiet(approach.get_optimization_bounds)[0]
        (params, objs), _ = _quiet(approach.run_optimization, bounds=bounds,
                                   iterations=iterations, seed=seed)
    except CrntStopped as e:
        return {"bistable": None, "approach": "semi-diffusive",
                "conclusion": "Its semi-diffusive method does not apply either: "
                              + explain_refusal(str(e), cg) + "."}
    except Exception as e:      # an error inside crnt4sbml, not a stated requirement
        return {"bistable": None, "approach": "semi-diffusive",
                "conclusion": "Its semi-diffusive method could not be run on this network: it "
                              f"failed inside crnt4sbml ({type(e).__name__}: {e})."}
    objs = [float(o) for o in objs]
    zero = [o for o in objs if o <= SEMI_DIFFUSIVE_ZERO]
    res = {"approach": "semi-diffusive",
           "optimisation": {"iterations": iterations, "decision_vector": [
               str(x) for x in approach.get_decision_vector()],
               "flux_vectors_found": len(params), "zero_objective": len(zero),
               "best_objective": min(objs) if objs else None}}
    if zero:
        res.update(bistable=None, multistationarity_possible=True, conclusion=(
            f"Multistationarity is possible: crnt4sbml's semi-diffusive approach found "
            f"{len(zero)} flux vector(s) with objective 0, i.e. reaction rates for which this "
            "open network may have several positive steady states. This does not establish "
            "their stability; it gives fluxes, not rate constants, so no bistable parameter set was "
            "confirmed."))
    else:
        res.update(bistable=None, proof=False, multistationarity_possible=None, search_status='no_witness_found', conclusion=(
            "The semi-diffusive optimisation found no flux vector with several steady states "
            f"(best objective {min(objs) if objs else 'n/a'}). Evidence, not proof."))
    return res


def _saved_model_values(path) -> dict:
    """The parameter values and eliminated species (conservation laws, as
    assignment rules) of a saved bistable model file, read back from the file."""
    from dynalyst.analysis.ode_model import model_from_sbml, ode_report
    try:
        r = ode_report(model_from_sbml(str(path)))
    except Exception as e:   # the file is still reported; its values just are not
        return {"file_read_error": f"{type(e).__name__}: {e}"}
    out = {"parameters_in_file": r["parameters"], "odes_in_file": r["odes"]}
    if r.get("assignment_rules"):
        out["eliminated_species_in_file"] = r["assignment_rules"]
    return out


def _plot_reaction_graph(net, path):
    """The species-reaction graph (networkx): species (circles) and reactions
    (boxes) are both nodes; an arrow runs from each reactant into its reaction
    and from the reaction to each product, labelled with the stoichiometry when
    it is not 1. Reactions that share a species are joined through it, so the
    network is drawn as one connected graph (unlike the complex graph, where
    every linkage class is a separate island)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    try:
        g = nx.DiGraph()
        species, reactions, stoich = set(), [], {}
        for r in net["reactions"]:
            rid = f"{r['id']}\n{r['rate_constant']}"
            reactions.append(rid)
            g.add_node(rid, kind="reaction")
            for s, n in r["reactants"].items():
                species.add(s)
                g.add_edge(s, rid)
                stoich[(s, rid)] = n
            for s, n in r["products"].items():
                species.add(s)
                g.add_edge(rid, s)
                stoich[(rid, s)] = n
        for s in species:
            g.nodes[s]["kind"] = "species"
        # one layout for the whole graph; components (if any) packed side by side
        pos, x0 = {}, 0.0
        for comp in sorted(nx.weakly_connected_components(g), key=len, reverse=True):
            sub = g.subgraph(comp)
            p_ = (nx.kamada_kawai_layout(sub.to_undirected()) if len(sub) > 2
                  else {n: (i, 0.0) for i, n in enumerate(sub)})
            xs = [v[0] for v in p_.values()]
            for n_, (x, y) in p_.items():
                pos[n_] = (x - min(xs) + x0, y)
            x0 += (max(xs) - min(xs)) + 1.0
        size = max(8.0, min(22.0, 1.2 * len(g) ** 0.5 * 3))
        fig, ax = plt.subplots(figsize=(size, size * 0.72))
        sp_nodes = [n for n in g if g.nodes[n].get("kind") == "species"]
        nx.draw_networkx_nodes(g, pos, nodelist=sp_nodes, ax=ax, node_color="#dde7f5",
                               node_size=1800, edgecolors="#3b5b8c")
        nx.draw_networkx_nodes(g, pos, nodelist=reactions, ax=ax, node_color="#f7e6c4",
                               node_shape="s", node_size=1300, edgecolors="#9a6b12")
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=7)
        nx.draw_networkx_edges(g, pos, ax=ax, arrowsize=14, edge_color="0.35",
                               node_size=1800, connectionstyle="arc3,rad=0.06")
        nx.draw_networkx_edge_labels(g, pos, ax=ax, font_size=7, edge_labels={
            e: f"{n:g}" for e, n in stoich.items() if n != 1})
        n_comp = nx.number_weakly_connected_components(g)
        ax.set_title(f"Reaction network: {len(sp_nodes)} species (circles), {len(reactions)} "
                     f"reactions (boxes)" + ("" if n_comp == 1 else
                                             f" -- {n_comp} unconnected parts"))
        ax.axis("off")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
        return config.display_path(path)
    except Exception as e:
        return f"not drawn ({type(e).__name__}: {e})"


def _plot_c_graph(cg, path):
    """CRNT's complex graph: complexes as nodes, reactions as arrows. Each linkage
    class is its own component, so this graph is disjoint by definition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    try:
        g = cg.get_graph()
        labels = {n: g.nodes[n].get("label", n) for n in g.nodes}
        fig, ax = plt.subplots(figsize=(10, 7))
        # each linkage class laid out on its own, side by side
        pos, x0 = {}, 0.0
        for comp in sorted(nx.weakly_connected_components(g), key=len, reverse=True):
            sub = g.subgraph(comp)
            p_ = nx.kamada_kawai_layout(sub) if len(sub) > 2 else {n: (i, 0) for i, n in enumerate(sub)}
            xs = [v[0] for v in p_.values()]
            for n_, (x, y) in p_.items():
                pos[n_] = (x - min(xs) + x0, y)
            x0 += (max(xs) - min(xs)) + 1.0
        nx.draw_networkx(g, pos, ax=ax, labels=labels, node_color="#dde7f5", node_size=2200,
                         font_size=8, arrowsize=14, edge_color="0.35")
        nx.draw_networkx_edge_labels(g, pos, ax=ax, font_size=7, edge_labels={
            e: g.edges[e].get("label", "") for e in g.edges})
        n_lc = nx.number_weakly_connected_components(g)
        ax.set_title(f"Complex graph (CRNT): {len(g)} complexes, {n_lc} linkage class"
                     + ("es -- one part each" if n_lc != 1 else ""))
        ax.axis("off")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
        return config.display_path(path)
    except Exception as e:
        return f"not drawn ({type(e).__name__}: {e})"


def _complex(c):
    return " + ".join(f"{n}{s}" if n > 1 else s for s, n in c.items()) or "0"


def _model_parameter_values(net, ks):
    """Rate constants that are single model parameters, in the model's names."""
    out = {}
    for r in net["reactions"]:
        if r["id"] in ks and len(r["model_parameters"]) == 1 and r["rate_constant"] == r["model_parameters"][0]:
            out[r["model_parameters"][0]] = ks[r["id"]]
    return out


def draw_reaction_network(model_name_or_path: str | None = None, species: list | None = None,
                          odes: dict | None = None, parameters: dict | None = None,
                          initial_conditions: dict | None = None) -> str:
    """Tool: the model's reaction network drawn -- the whole network (species and
    reactions as nodes) and CRNT's complex graph -- with its structure, no search."""
    d = json.loads(crnt_bistability(model_name_or_path, species, odes, parameters,
                                    initial_conditions, structure_only=True, draw_network=True))
    keep = ["model", "error", "mass_action", "network", "reaction_graph_plot",
            "complex_graph_plot"]
    if "reaction_graph_plot" not in d:      # why it could not be drawn
        keep.append("conclusion")
    return json.dumps({k: d[k] for k in keep if k in d}, indent=2, default=float)


_MODEL_INPUT = {
    "model_name_or_path": {"type": "string", "description": "SBML / XPP model name or path."},
    "species": {"type": "array", "items": {"type": "string"}},
    "odes": {"type": "object", "description": "{species: rhs} for typed ODEs."},
    "parameters": {"type": "object"},
    "initial_conditions": {"type": "object"},
}
TOOL_DEFINITIONS = [{
    "name": "crnt_bistability",
    "description": (
        "Chemical reaction network theory (crnt4sbml) only: the model's reaction network "
        "(reactions, complexes, stoichiometric matrix, rank, deficiency, linkage classes, weak "
        "reversibility, conservation laws) and whether it can have several steady states for "
        "SOME rate constants. Converts the model to a mass-action network and a "
        "crnt4sbml-compatible SBML automatically, applies the deficiency zero/one theorems "
        "(which rule several steady states out for all rate constants), otherwise searches "
        "for rate constants with several steady states, near the model's own values first, "
        "then over crnt4sbml's generic ranges. No continuation and no stability check: whether "
        "two of the states are stable (bistability) is a separate bifurcation analysis. Sets "
        "found are saved as model files for it."),
    "input_schema": {
        "type": "object",
        "properties": {
            **_MODEL_INPUT,
            "iterations": {"type": "integer", "description": "Optimisation starts (default 20)."},
            "structure_only": {"type": "boolean", "description": "Only the network structure "
                               "(deficiency, stoichiometric matrix, linkage classes, theorems), "
                               "no search for rate constants."},
            "semi_diffusive": {"type": "boolean", "description": "For a network with no "
                               "conservation law (open, with inflow/outflow): run crnt4sbml's "
                               "semi-diffusive approach (slower) to test multistationarity."},
            "job_id": {"type": "string", "description": "Only the job_id of an earlier call that "
                       "returned still_running and that the user chose to wait for: collects "
                       "that search's result (nothing else is needed)."},
        },
    },
}, {
    "name": "draw_reaction_network",
    "description": (
        "Draw the model's reaction network: the whole network with species and reactions as "
        "nodes, and CRNT's complex graph (one part per linkage class). Only when the user asks "
        "to see the reaction network; no analysis."),
    "input_schema": {"type": "object", "properties": dict(_MODEL_INPUT)},
}]
TOOL_FUNCTIONS = {"crnt_bistability": crnt_bistability,
                  "draw_reaction_network": draw_reaction_network}
