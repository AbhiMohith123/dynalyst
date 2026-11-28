"""Run every analysis tool on every local model, without the agent (no model calls).

    python scripts/tool_sweep.py                    # all models, all tools
    python scripts/tool_sweep.py --tools find_steady_states,simulate_model
    python scripts/tool_sweep.py --match Markevich  # only models whose file name contains this
    python scripts/tool_sweep.py --workers 4 --timeout 300

Each (model, tool) call runs in its own process with a wall-clock limit, so a hang shows
up as TIMEOUT instead of stopping the sweep. Per call: ok / ERROR (the tool's own error
message) / CRASH (an exception escaped the tool) / TIMEOUT / BAD (non-finite numbers in
the result), with the time taken. A 'consistency' task cross-checks tools against each
other on the same model:
  * every steady state found is a root of the equations,
  * a long simulation that settles ends near one of the stable steady states found,
  * a parameter or starting value passed in is the one the model uses.
Saves outputs/tool_sweep.json and prints the problems grouped by message.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# one call per tool, with only the model given (plus settings that keep a call short)
TOOL_ARGS = {
    "classify_model": {},
    "find_steady_states": {},
    "simulate_model": {"t_end": 200},
    "plot_rate_of_change": {"t_end": 50},
    "jacobian_network": {"which": "all", "parameter_jacobian": False},
    "propose_bifurcation_candidates": {},
    "crnt_bistability": {"structure_only": True},
    "propensity_table": {},
    "stochastic_simulation": {"n_runs": 5},
    "reduce_model": {"method": "auto"},
    "run_bifurcation_analysis": {},
    "get_annotations": {},
}
# derived files the agent writes (copies, CRNT conversions and sets, reductions, typed
# systems): the same networks again
_DERIVED = re.compile(r"(_crnt|_crnt_set_\d+|_reduced|_copy|^typed_)", re.I)


def _non_finite(x) -> bool:
    if isinstance(x, float):
        return math.isnan(x) or math.isinf(x)
    if isinstance(x, dict):
        return any(_non_finite(v) for v in x.values())
    if isinstance(x, list):
        return any(_non_finite(v) for v in x)
    return False


def run_tool(tool: str, model: str) -> dict:
    """In a worker process: one tool call, classified."""
    from dynalyst.orchestration import tools as registry
    out = registry.execute(tool, {"model_name_or_path": model, **TOOL_ARGS[tool]})
    try:
        data = json.loads(out)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict) and data.get("error"):
        return {"status": "ERROR", "detail": str(data["error"])[:300]}
    if data is None:
        low = (out or "").lower()
        if "traceback" in low or "failed:" in low:
            return {"status": "ERROR", "detail": (out or "")[:300]}
        return {"status": "ok"}
    if _non_finite(data):
        return {"status": "BAD", "detail": "non-finite numbers in the result"}
    return {"status": "ok"}


def consistency(model: str) -> dict:
    """In a worker process: tools checked against each other on one model."""
    import numpy as np
    from dynalyst.analysis.jaxkm_tools import simulate_model
    from dynalyst.analysis.ode_model import resolve_model
    from dynalyst.analysis.steady_states import _lambdify_model
    from dynalyst.orchestration import tools as registry
    problems = []
    m = resolve_model(model)
    names = m["state_names"]
    rhs_fn, _ = _lambdify_model(m)
    pv = [m["param_defaults"][p] for p in m["param_names"]]
    # 1. overrides reach the model
    if m["param_names"]:
        p = m["param_names"][0]
        new = 2.0 * float(m["param_defaults"][p]) + 1.0
        if float(resolve_model(model, parameters={p: new})["param_defaults"][p]) != new:
            problems.append(f"parameter override of {p} not applied")
    if names:
        s = names[0]
        new = 2.0 * float(m["y0"][s]) + 1.0
        if float(resolve_model(model, initial_conditions={s: new})["y0"][s]) != new:
            problems.append(f"initial-condition override of {s} not applied")
    # 2. every steady state found is a root
    ss = json.loads(registry.execute("find_steady_states", {"model_name_or_path": model}))
    states = [x for x in ss.get("steady_states") or [] if isinstance(x, dict)]
    stable = []
    for x in states:
        vec = x.get("state") or x.get("values") or {k: x[k] for k in names if k in x}
        if not isinstance(vec, dict) or not all(k in vec for k in names):
            continue
        y = np.array([float(vec[k]) for k in names])
        with np.errstate(all="ignore"):
            f = np.abs(np.asarray(rhs_fn(*y, *pv), dtype=float))
        if not np.all(np.isfinite(f)) or float(np.max(f)) > 1e-5 * (1 + float(np.max(np.abs(y)))):
            problems.append(f"steady state is not a root (max |dx/dt| {float(np.max(f)):.2e})")
        if str(x.get("stability", "")).startswith("stable"):
            stable.append(y)
    # 3. a long simulation that settles ends at a stable steady state found
    sim = json.loads(simulate_model(model, t_end=2000, n_points=400))
    if stable and (sim.get("behaviour") or {}).get("verdict") == "settled":
        end = np.array([float(sim["final_concentrations"][k]) for k in names])
        gap = min(float(np.max(np.abs(end - y)) / (1 + np.max(np.abs(y)))) for y in stable)
        if gap > 1e-2:
            problems.append(f"simulation settles at a state no stable steady state matches "
                            f"(relative gap {gap:.2e})")
    return {"status": "PROBLEM" if problems else "ok", "detail": "; ".join(problems)[:300]}


def _worker(task: str, model: str) -> None:
    import warnings
    warnings.filterwarnings("ignore")
    try:
        res = consistency(model) if task == "consistency" else run_tool(task, model)
    except Exception as e:     # an exception the tool did not turn into an error message
        res = {"status": "CRASH", "detail": f"{type(e).__name__}: {str(e)[:250]}"}
    print("RESULT " + json.dumps(res), flush=True)


def _call(python, task, model, timeout) -> dict:
    t0 = time.time()
    env = {**os.environ, "DYNALYST_AUTO_INDEX_MODELS": "0", "PYTHONIOENCODING": "utf-8"}
    try:
        p = subprocess.run([python, "-W", "ignore", __file__, "--worker", task, model],
                           capture_output=True, text=True, errors="replace", timeout=timeout,
                           cwd=str(ROOT), env=env)
        line = next((ln for ln in p.stdout.splitlines()[::-1] if ln.startswith("RESULT ")), None)
        res = (json.loads(line[7:]) if line else
               {"status": "CRASH", "detail": (p.stderr.strip().splitlines() or ["no output"])[-1][:300]})
    except subprocess.TimeoutExpired:
        res = {"status": "TIMEOUT", "detail": f"over {timeout} s"}
    return {"model": model, "task": task, "seconds": round(time.time() - t0, 1), **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2)
    ap.add_argument("--tools", default=",".join([*TOOL_ARGS, "consistency"]))
    ap.add_argument("--match", default="")
    ap.add_argument("--models", default="",
                    help="model files to sweep instead of data/sbml_models (comma-separated "
                         "paths, any folder), e.g. a fixed set of models that must work")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--rerun", default="", help="a saved sweep (json): rerun only its problems")
    ap.add_argument("--task-timeout", action="append", default=[],
                    help="a time limit for one tool, e.g. stochastic_simulation=1200 (repeatable)")
    ap.add_argument("--only-status", default="",
                    help="with --rerun: only rows with this status (e.g. TIMEOUT)")
    a = ap.parse_args()
    if a.worker:
        return _worker(*a.worker)
    from dynalyst import config
    # the file's path, not its bare name: two folders may hold models of the same name
    if a.models:
        models = [m.strip() for m in a.models.split(",") if m.strip()]
    else:
        models = sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                        for p in Path(config.LOCAL_SBML_MODELS_DIR).glob("*.xml")
                        if not _DERIVED.search(p.stem) and a.match.lower() in p.name.lower())
    tasks = [t for t in a.tools.split(",") if t]
    if a.rerun:
        old = json.loads(Path(a.rerun).read_text(encoding="utf-8"))
        by_name = {Path(m).name: m for m in models}
        jobs = [(r["task"], by_name.get(Path(r["model"]).name, r["model"])) for r in old
                if r["status"] != "ok" and r["task"] in tasks
                and (not a.only_status or r["status"] == a.only_status)]
    else:
        jobs = [(t, m) for m in models for t in tasks]
    print(f"{len(jobs)} calls", flush=True)
    limits = {k: int(v) for k, v in (x.split("=", 1) for x in a.task_timeout)}
    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_call, sys.executable, t, m, limits.get(t, a.timeout)) for t, m in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            if r["status"] != "ok":
                print(f"[{i}/{len(jobs)}] {r['status']:8} {r['task']:30} {r['model']:40} "
                      f"{r.get('detail', '')[:120]}", flush=True)
    out = Path(config.OUTPUT_DIR) / "tool_sweep.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"\n{len(rows) - len(bad)} ok, {len(bad)} problems; saved {out}")
    groups = {}
    for r in bad:     # the same message on many models is one bug
        key = (r["task"], r["status"], re.sub(r"[\d.]+(e-?\d+)?", "#", r.get("detail", ""))[:90])
        groups.setdefault(key, []).append(r["model"])
    for (task, status, msg), ms in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"{len(ms):3d}x {status:8} {task:30} {msg}  e.g. {', '.join(ms[:3])}")


if __name__ == "__main__":
    main()
