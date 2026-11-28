"""AUTO-07p problem types that a model file cannot describe, from typed equations: discrete maps
(IPS=-1), boundary value problems (IPS=4), algebraic optimization (IPS=5), periodic orbits and
rotations started from numerical data (IPS=2) and travelling waves (IPS=11, then 12).

Only the free parameters take PAR slots; every other value is written into the equations as a
number, so AUTO's reserved slots (PAR(10), PAR(11), the diffusion constants) are never in the way.
AUTO differentiates numerically (JAC=0). PDE time stepping (IPS=14, 16, 17) is left out on purpose.
"""
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from dynalyst import config
from dynalyst.fortran import FortranError, fortran_assign

KINDS = {"map": -1, "boundary_value": 4, "optimization": 5, "orbit": 2, "wave": 11}
MAX_FREE = 9        # PAR(1)..PAR(9): AUTO uses PAR(10) and up for its own values in these problems
_HEADER = ("      IMPLICIT NONE\n"
           "      INTEGER, INTENT(IN) :: NDIM, ICP(*)\n")

_RUNNER = r"""
import json
import os
import shutil
import sys
import numpy as np

sys.path.insert(0, os.environ["AUTO_DIR_PY"])
from auto import run

with open("runner_config.json") as f:
    CONFIG = json.load(f)
out = {"branches": [], "errors": []}
SPECIAL = ("BP", "LP", "HB", "PD", "TR", "GH", "BT", "ZH", "CP", "R1", "R2", "R3", "R4",
           "LPD", "LTR", "PTR")


def keep(result, what):
    for br in result:
        arr = np.asarray(br.toarray(), dtype=float)
        points = [{"index": int(i), "type": str(ty), "label": info.get("LAB")}
                  for i, val in br.labels.sortByIndex() for ty, info in val.items()
                  if ty in SPECIAL]
        out["branches"].append({"run": what, "coordnames": [str(c) for c in br.coordnames],
                                "data": arr.tolist(), "stability": [int(x) for x in br.stability()],
                                "points": points})


def go(tag, *args, **kw):
    try:
        result = run(*args, **kw)
    except Exception as e:
        out["errors"].append(f"{tag}: {e}")
        return None
    keep(result, tag)
    return result


kind = CONFIG["kind"]
if kind == "optimization":
    # each extremum (LP) of the objective frees one more parameter (AUTO demo opt)
    r = go("objective", e="model", c="model")
    for i in range(CONFIG["n_free"] - 1):
        lps = [lab for b in (r or []) for lab in
               [info["LAB"] for _, val in b.labels.sortByIndex() for ty, info in val.items() if ty == "LP"]]
        if not lps:
            break
        r = go(f"extremum {i + 1}", r(lps[0]))
elif kind == "wave":
    uniform = go("uniform states", e="model", c="model", IPS=11)
    for _ in [0] if uniform else []:
        hopfs = [info["LAB"] for b in uniform for _, val in b.labels.sortByIndex()
                 for ty, info in val.items() if ty == "HB"]
        for lab in hopfs[:2]:
            go(f"waves from HB {lab}", uniform(lab), IPS=12, ICP=[CONFIG["icp"][0], 11], ILP=0, ISP=0,
               NMX=CONFIG["nmx"], NPR=CONFIG["nmx"])
else:
    kw = {"dat": "start"} if kind == "orbit" else {}
    forward = go("forward", e="model", c="model", **kw)
    backward = go("backward", e="model", c="model", DS="-", **kw)
    # branch switching where AUTO allows it: branch points, and period doublings of maps and orbits
    for name, r in (("forward", forward), ("backward", backward)):
        labs = [(info["LAB"], ty) for b in (r or []) for _, val in b.labels.sortByIndex()
                for ty, info in val.items() if ty in ("BP", "PD")]
        for lab, ty in labs[:2]:
            go(f"switch at {ty} of {name}", r(lab), ISW=-1)
with open("result.json", "w") as f:
    json.dump(out, f)
"""


def _symbols(states, free):
    sub = {sp.Symbol(s): sp.Symbol(f"U({i + 1})") for i, s in enumerate(states)}
    sub.update({sp.Symbol(p): sp.Symbol(f"PAR({i + 1})") for i, p in enumerate(free)})
    return sub


def _exprs(texts, numbers, sub, what):
    """Each text as AUTO names: free parameters and states as PAR(i)/U(i), the rest numbers."""
    return [(sp.sympify(t, locals={n: sp.Symbol(n) for n in [*numbers, *map(str, sub)]})
             .xreplace({sp.Symbol(k): sp.Float(v) for k, v in numbers.items()}).xreplace(sub),
             f"{what} {i + 1}") for i, t in enumerate(texts)]


def _assign(items, target):
    return [fortran_assign(e, target.format(i + 1), where) for i, (e, where) in enumerate(items)]


def _write_equations(path, kind, states, free, equations, parameters, initial_state, start_values,
                     boundary_conditions, integral_conditions, objective, diffusion, wave_speed):
    sub = _symbols(states, free)
    fixed = {k: v for k, v in parameters.items() if k not in free}
    lines = ["      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)", _HEADER.rstrip(),
             "      INTEGER, INTENT(IN) :: IJAC",
             "      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)",
             "      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)",
             "      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)"]
    lines += _assign(_exprs([equations[s] for s in states], fixed, sub, "equation"), "F({})")
    lines += ["      END SUBROUTINE FUNC", "",
              "      SUBROUTINE STPNT(NDIM,U,PAR,T)",
              "      IMPLICIT NONE", "      INTEGER, INTENT(IN) :: NDIM",
              "      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)",
              "      DOUBLE PRECISION, INTENT(IN) :: T"]
    lines += [f"      PAR({i + 1}) = {float(start_values[p])!r}" for i, p in enumerate(free)]
    if kind == "wave":
        lines.append(f"      PAR(10) = {float(wave_speed)!r}")
        lines += [f"      PAR({15 + i}) = {float(diffusion[s])!r}" for i, s in enumerate(states)]
    t_sub = {**sub, sp.Symbol("t"): sp.Symbol("U(0)")}     # the profile's t, swapped back below
    for i, s in enumerate(states):
        text = str(initial_state.get(s, 0.0))
        line = fortran_assign(_exprs([text], fixed, t_sub, f"initial value of {s}")[0][0], f"U({i + 1})",
                              f"initial value of {s}")
        lines.append(line.replace("U(0)", "T"))
    lines += ["      END SUBROUTINE STPNT", ""]
    if boundary_conditions:
        ends = dict(sub)
        for i, s in enumerate(states):
            ends[sp.Symbol(f"{s}_0")] = sp.Symbol(f"U(0{i + 1})")
            ends[sp.Symbol(f"{s}_1")] = sp.Symbol(f"U(1{i + 1})")
        body = _assign(_exprs(boundary_conditions, fixed, ends, "boundary condition"), "FB({})")
        for i in range(len(states)):
            body = [b.replace(f"U(0{i + 1})", f"U0({i + 1})").replace(f"U(1{i + 1})", f"U1({i + 1})")
                    for b in body]
        lines += ["      SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC)", _HEADER.rstrip(),
                  "      INTEGER, INTENT(IN) :: NBC, IJAC",
                  "      DOUBLE PRECISION, INTENT(IN) :: PAR(*), U0(NDIM), U1(NDIM)",
                  "      DOUBLE PRECISION, INTENT(OUT) :: FB(NBC)",
                  "      DOUBLE PRECISION, INTENT(INOUT) :: DBC(NBC,*)", *body,
                  "      END SUBROUTINE BCND"]
    else:
        lines += ["      SUBROUTINE BCND", "      END SUBROUTINE BCND"]
    if integral_conditions:
        lines += ["      SUBROUTINE ICND(NDIM,PAR,ICP,NINT,U,UOLD,UDOT,UPOLD,FI,IJAC,DINT)", _HEADER.rstrip(),
                  "      INTEGER, INTENT(IN) :: NINT, IJAC",
                  "      DOUBLE PRECISION, INTENT(IN) :: PAR(*)",
                  "      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), UOLD(NDIM), UDOT(NDIM), UPOLD(NDIM)",
                  "      DOUBLE PRECISION, INTENT(OUT) :: FI(NINT)",
                  "      DOUBLE PRECISION, INTENT(INOUT) :: DINT(NINT,*)",
                  *_assign(_exprs(integral_conditions, fixed, sub, "integral condition"), "FI({})"),
                  "      END SUBROUTINE ICND"]
    else:
        lines += ["      SUBROUTINE ICND", "      END SUBROUTINE ICND"]
    if objective:
        lines += ["      SUBROUTINE FOPT(NDIM,U,ICP,PAR,IJAC,FS,DFDU,DFDP)", _HEADER.rstrip(),
                  "      INTEGER, INTENT(IN) :: IJAC",
                  "      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)",
                  "      DOUBLE PRECISION, INTENT(OUT) :: FS",
                  "      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM), DFDP(*)",
                  *_assign(_exprs([objective], fixed, sub, "objective"), "FS"),
                  "      END SUBROUTINE FOPT"]
    else:
        lines += ["      SUBROUTINE FOPT", "      END SUBROUTINE FOPT"]
    lines += ["      SUBROUTINE PVLS", "      END SUBROUTINE PVLS"]
    path.write_text("\n".join(lines) + "\n", newline="\n")


def _write_constants(path, kind, states, free, n_bc, n_int, lo, hi, nmx):
    span = (hi - lo) if lo is not None and hi is not None else 10.0
    n = len(states)
    names = {i + 1: p for i, p in enumerate(free)}
    icp = [10, *range(1, len(free) + 1)] if kind == "optimization" else (
        [1, 11] if kind == "orbit" else list(range(1, len(free) + 1)) if kind == "boundary_value" else [1])
    if kind == "optimization":
        names[10] = "objective"
    stop = "" if lo is None else f"\nUZSTOP = {{1: [{lo!r}, {hi!r}]}}"
    text = (f"unames = {{{', '.join(f'{i + 1}: {s!r}' for i, s in enumerate(states))}}}\n"
            f"parnames = {names!r}\n"
            f"NDIM= {n}, IPS = {KINDS[kind]}, IRS = 0, ILP = 1\n"
            f"ICP = {icp!r}\n"
            f"NTST= 20, NCOL= 4, IAD = 3, ISP = 2, ISW = 1, IPLT= 0, NBC= {n_bc}, NINT= {n_int}\n"
            f"NMX= {nmx}, NPR= {nmx}, MXBF= 5, IID = 2, ITMX= 8, ITNW= 7, NWTN= 3, JAC= 0\n"
            "EPSL= 1e-07, EPSU = 1e-07, EPSS = 1e-05\n"
            f"DS = {span / 500!r}, DSMIN= {span / 1e7!r}, DSMAX= {span / 50!r}, IADS= 1\n"
            f"NPAR = {max(icp + [14 + n if kind == 'wave' else 1])}{stop}\n"
            + ("THL = {11: 0.0}\n" if kind == "orbit" else ""))
    path.write_text(text, newline="\n")


def _summary(data, free, states):
    rows, points = [], []
    for b in data.get("branches") or []:
        names = b["coordnames"]
        arr = np.array(b["data"], dtype=float)
        if arr.ndim != 2 or not arr.shape[1]:
            continue
        row = {"run": b["run"], "n_points": int(arr.shape[1])}
        for p in free:
            if p in names:
                v = arr[names.index(p)]
                row[p] = [float(np.nanmin(v)), float(np.nanmax(v))]
        rows.append(row)
        for pt in b["points"]:
            if pt["index"] < arr.shape[1]:
                points.append({"type": pt["type"], "run": b["run"],
                               **{n: float(arr[names.index(n)][pt["index"]]) for n in names
                                  if n in free or n in states or n in ("objective", "PERIOD", "PAR(11)")}})
    return rows, points


def _plot(data, free, states, label):
    """The first free parameter against the first state (or the objective)."""
    y_name = states[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    drew = False
    for b in data.get("branches") or []:
        names, arr = b["coordnames"], np.array(b["data"], dtype=float)
        x_name = free[0] if free[0] in names else None
        y = y_name if y_name in names else ("objective" if "objective" in names else None)
        if arr.ndim != 2 or not x_name or not y:
            continue
        xs, ys = arr[names.index(x_name)], arr[names.index(y)]
        ax.plot(xs, ys, color="black", linewidth=1.5)
        for pt in b["points"]:
            if pt["index"] < xs.size:
                ax.plot(xs[pt["index"]], ys[pt["index"]], "o", color="tab:red")
                ax.annotate(pt["type"], (xs[pt["index"]], ys[pt["index"]]), fontsize=8)
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.set_xlabel(free[0])
    ax.set_ylabel(y_name)
    ax.grid(alpha=0.3)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mktemp(prefix=f"{label}_", suffix=".png", dir=str(config.PLOTS_DIR)))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return config.display_path(path)


def run_auto_problem(kind: str, states: list, equations: dict, free_parameters: list,
                     parameters: dict | None = None, initial_state: dict | None = None,
                     param_min: float | None = None, param_max: float | None = None,
                     boundary_conditions: list | None = None, integral_conditions: list | None = None,
                     objective: str | None = None, start_orbit_file: str | None = None,
                     diffusion: dict | None = None, wave_speed: float | None = None,
                     nmx: int = 500) -> str:
    """Tool: one AUTO-07p problem from typed equations."""
    from dynalyst.analysis.bifurcation_auto import _run_auto
    if kind not in KINDS:
        return json.dumps({"error": f"kind must be one of {sorted(KINDS)}."})
    parameters = dict(parameters or {})
    initial_state = dict(initial_state or {})
    boundary_conditions, integral_conditions = boundary_conditions or [], integral_conditions or []
    problems = []
    missing = [s for s in states if s not in equations]
    if missing:
        problems.append(f"no equation for {missing}")
    if not free_parameters or len(free_parameters) > MAX_FREE:
        problems.append(f"give 1 to {MAX_FREE} free_parameters")
    if [p for p in free_parameters if p not in parameters]:
        problems.append("every free parameter needs its starting value in parameters")
    if kind == "boundary_value":
        if not boundary_conditions:
            problems.append("a boundary value problem needs boundary_conditions (write the "
                            "ends as x_0 and x_1 for a state x)")
        needed = len(boundary_conditions) + len(integral_conditions) - len(states) + 1
        if boundary_conditions and needed != len(free_parameters):
            problems.append(f"{len(boundary_conditions)} boundary and {len(integral_conditions)} "
                            f"integral conditions on {len(states)} states need {needed} free "
                            f"parameter(s), not {len(free_parameters)}")
    if kind == "optimization" and not objective:
        problems.append("an optimization needs an objective")
    if kind == "orbit" and not start_orbit_file:
        problems.append("an orbit or rotation run needs start_orbit_file: rows of time and states "
                        "over one period (for a rotation, a state gains a multiple of 2*pi)")
    if kind == "wave" and (wave_speed is None or set(diffusion or {}) != set(states)):
        problems.append("a wave run needs wave_speed and a diffusion constant for every state")
    if problems:
        return json.dumps({"error": "; ".join(problems)})

    config.AUTO_WORK_DIR.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="auto_problem_", dir=str(config.AUTO_WORK_DIR)))
    try:
        _write_equations(work / "model.f90", kind, states, free_parameters, equations, parameters,
                         initial_state, parameters, boundary_conditions, integral_conditions,
                         objective, diffusion, wave_speed)
        _write_constants(work / "c.model", kind, states, free_parameters, len(boundary_conditions),
                         len(integral_conditions), param_min, param_max, int(nmx))
        if start_orbit_file:
            shutil.copyfile(start_orbit_file, work / "start.dat")
        (work / "runner_config.json").write_text(json.dumps(
            {"kind": kind, "n_free": len(free_parameters), "nmx": int(nmx),
             "icp": list(range(1, len(free_parameters) + 1))}))
        data = _run_auto(work, "model", script=_RUNNER)
    except (FortranError, ValueError, OSError, RuntimeError, sp.SympifyError) as e:
        return json.dumps({"error": f"AUTO-07p {kind} run failed: {e}"})
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rows, points = _summary(data, free_parameters, states)
    if not rows:
        return json.dumps({"error": f"AUTO-07p produced no {kind} branch.", "details": data.get("errors")})
    return json.dumps({"kind": kind, "auto_ips": KINDS[kind], "branches": rows, "special_points": points,
                       "plot_path": _plot(data, free_parameters, states, f"auto_{kind}"),
                       "errors": data.get("errors") or []}, indent=2, default=float)


TOOL_DEFINITIONS = [{
    "name": "run_auto_problem",
    "description": (
        "AUTO-07p on typed equations that a model file cannot describe: kind 'map' (discrete "
        "x(k+1) = f(x)), 'boundary_value' (ODEs on [0,1] with boundary and integral conditions), "
        "'optimization' (extrema of an objective along a family), 'orbit' (a periodic orbit or "
        "rotation from numerical data), 'wave' (travelling waves of a reaction-diffusion system). "
        "A model file or ordinary ODE steady states and cycles: run_bifurcation_analysis. "
        "PDE time stepping is not available."),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": sorted(KINDS)},
            "states": {"type": "array", "items": {"type": "string"}, "description": "State names."},
            "equations": {"type": "object", "description": "{state: right-hand side}; for a map, the next value."},
            "free_parameters": {"type": "array", "items": {"type": "string"}, "description": "Parameters AUTO varies, the continuation parameter first."},
            "parameters": {"type": "object", "description": "{name: value} of every parameter, free ones at their starting values."},
            "initial_state": {"type": "object", "description": "{state: value at the start}; a boundary value problem may use t on [0,1]."},
            "param_min": {"type": "number", "description": "Optional lower end for the first free parameter."},
            "param_max": {"type": "number", "description": "Optional upper end for the first free parameter."},
            "boundary_conditions": {"type": "array", "items": {"type": "string"}, "description": "Expressions that must vanish; x_0 and x_1 are state x at t = 0 and t = 1."},
            "integral_conditions": {"type": "array", "items": {"type": "string"}, "description": "Integrands whose integral over [0,1] must vanish."},
            "objective": {"type": "string", "description": "For 'optimization': the expression to extremize."},
            "start_orbit_file": {"type": "string", "description": "For 'orbit': a text file with rows of time and states over one period."},
            "diffusion": {"type": "object", "description": "For 'wave': {state: diffusion constant}."},
            "wave_speed": {"type": "number", "description": "For 'wave': the wave speed."},
            "nmx": {"type": "integer", "description": "Optional maximum continuation steps. Default 500."},
        },
        "required": ["kind", "states", "equations", "free_parameters"],
    },
}]

TOOL_FUNCTIONS = {"run_auto_problem": run_auto_problem}
