"""
Bifurcation analysis via AUTO-07p (https://github.com/auto-07p/auto-07p),
replacing ALL previous bifurcation code in this project (the old
bifurcation_1d.py / bifurcation_classifier.py / bifurcation_preview.py,
which only handled strictly 1D models via hand-rolled symbolic root-finding).

This module works for a model of ANY dimension:
  1. Build the symbolic ODE model (ode_model.py -- SBML of any dimension, or
     directly-specified ODEs).
  2. Find a starting equilibrium at the sweep's starting parameter value
     (steady_states.py's constrained solver, so models with conservation
     laws still get an isolated starting point rather than an arbitrary
     point on a manifold).
  3. Auto-generate the AUTO-07p Fortran equations file: F (the RHS), DFDU
     (analytic Jacobian w.r.t. state), and DFDP (analytic Jacobian w.r.t.
     parameters) -- all produced from the SAME sympy expressions via
     sympy.fcode, so there is no hand-transcription step to get wrong.
  4. Run real numerical pseudo-arclength continuation (forward AND backward
     from the start point, merged) using AUTO's Python interface, in an
     isolated subprocess (AUTO compiles a fresh .exe per run and uses
     cwd-relative files/global state, so each call gets its own scratch dir).
  5. AUTO itself detects and labels special points along the branch:
       LP  = Limit Point   -> saddle-node / fold bifurcation
       BP  = Branch Point  -> transcritical or pitchfork bifurcation
             (two branches cross at this point; which of the two it is
             depends on whether the crossing branch is symmetric -- AUTO
             calls both "BP", so both are reported, described as
             "transcritical/pitchfork (branch point)")
       HB  = Hopf Bifurcation -> birth of a periodic orbit (oscillation)
     This is the real generalization the old classifier only faked for 1D
     models: AUTO detects these from the actual continuation, not from
     local normal-form derivative tests on a hand-solved critical point.
  6. Plot: stability-colored branches (from AUTO's own stability segments)
     with special points marked and labeled.
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import time
import sys
import tempfile
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from dynalyst import config
from dynalyst.auto_env import build_failure, find_lib_dir, posix_path, unavailable_message
from dynalyst.fortran import (AUTO_RESERVED_PAR, FortranError, auto_par_index, auto_symbols,
                               fortran_assign)
from dynalyst.analysis.ode_model import (NO_PARAMETERS, canonical_param, continuation_parameter_error,
                                         eligible_parameters,
                                         jacobian, ode_strings, resolve_model,
                                         unknown_parameter_error)
from dynalyst.analysis.parameter_screen import screen_parameters, settled_state
from dynalyst.analysis.steady_states import (
    _candidate_starts,
    _term_scale_fn,
    flux_scale,
    _lambdify_model,
    across_a_pole,
    reconstruct_eliminated,
)
from scipy.optimize import fsolve, least_squares

PLOTS_DIR = config.PLOTS_DIR


def _write_lf(path, text: str) -> None:
    """Write UTF-8 text with LF endings on every OS (AUTO/Fortran dislike CRLF)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _find_xpp_source(model_name_or_path: str | None) -> Path | None:
    """Locate the exact XPP source associated with a model.

    Never uses substring/fuzzy matching. This is critical because models often
    have related names (e.g. ``model_a`` and ``model_a_extended``). Uploaded files are also searched under
    ``/mnt/data`` and the project tree by exact basename/stem.
    """
    if not model_name_or_path:
        return None

    try:
        p = Path(model_name_or_path).expanduser()
    except Exception:
        return None

    suffixes = (".ode", ".txt")
    candidates = []
    if p.is_file():
        candidates.append(p)
    else:
        name = p.name
        stem = p.stem
        # the model folders first: an XPP file next to its converted SBML
        for root in (p.parent, *config.SBML_SEARCH_DIRS, Path.cwd(), Path("/mnt/data"),
                     Path("/mnt/user-data/uploads"),
                     config.PROJECT_ROOT, config.DATA_DIR):
            if not root.exists():
                continue
            for suffix in suffixes:
                c = root / f"{stem}{suffix}"
                if c.is_file():
                    candidates.append(c)
            exact = root / name
            if exact.is_file() and exact.suffix.lower() in suffixes:
                candidates.append(exact)
            # Recursive exact match only (same stem, XPP suffix).
            for suffix in suffixes:
                for c in root.rglob(f"{stem}{suffix}"):
                    if c.is_file() and "vendor" not in c.parts:
                        candidates.append(c)

    seen = set()
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            continue
        if c in seen:
            continue
        seen.add(c)
        if c.is_file():
            return c
    return None


def _find_exact_input_path(model_name_or_path: str) -> Path:
    """Resolve an uploaded .txt/.ode or SBML filename by exact name only."""
    p = Path(model_name_or_path).expanduser()
    if p.is_file():
        return p.resolve()
    name = p.name
    roots = [p.parent, Path.cwd(), Path("/mnt/data"),
             Path("/mnt/user-data/uploads"),
             config.PROJECT_ROOT, config.DATA_DIR]
    matches = []
    for root in roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file():
            matches.append(direct)
        try:
            matches.extend(c for c in root.rglob(name) if c.is_file())
        except (OSError, PermissionError):
            pass
    unique = []
    seen = set()
    for c in matches:
        try:
            c = c.resolve()
        except Exception:
            continue
        if c not in seen:
            seen.add(c)
            unique.append(c)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        # Prefer the shallowest exact match; never choose by partial name.
        return min(unique, key=lambda x: (len(x.parts), str(x)))
    raise FileNotFoundError(
        f"No exact input file named '{model_name_or_path}' was found. "
        f"Refusing to substitute a similarly named model."
    )


def _parse_xpp_metadata(source_path: Path | None) -> dict:
    """Parse XPP AUTO/plot metadata from an optional .ode/.txt source.

    Only uncommented lines beginning with '@' are considered. Typical fields:
    autovar, parmin, parmax, autoxmin/autoxmax, autoymin/autoymax, xp, yp1, yp2.
    """
    meta = {}
    if source_path is None:
        return meta

    try:
        lines = source_path.read_text(errors="replace").splitlines()
    except Exception:
        return meta

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or not line.startswith("@"):
            continue

        body = line[1:].strip()

        # XPP normally writes comma-separated key=value pairs. Accept whitespace
        # around commas without accidentally treating a value as another token.
        tokens = [tok.strip() for tok in re.split(r"\s*,\s*", body) if tok.strip()]

        for token in tokens:
            if "=" not in token:
                continue

            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()

            if key in {"autovar", "xp", "yp1", "yp2"}:
                meta[key] = value
            elif key in {
                "parmin", "parmax",
                "autoxmin", "autoxmax",
                "autoymin", "autoymax"
            }:
                try:
                    meta[key] = float(value)
                except ValueError:
                    pass

    meta["source_path"] = str(source_path.resolve())
    return meta



def _xpp_declared_parameter_order(source_path: Path | None) -> list[str]:
    """Return active XPP parameter names in first-declaration order."""
    if source_path is None:
        return []
    try:
        text = source_path.read_text(errors="replace")
    except Exception:
        return []
    order = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("par ") or low.startswith("p "):
            body = line.split(None, 1)[1]
            for name in re.findall(r"([A-Za-z_]\w*)\s*=", body):
                if name not in order:
                    order.append(name)
    return order


def _infer_parameter_range(model: dict, param_name: str, requested_min: float | None,
                           requested_max: float | None, xpp_meta: dict) -> tuple[float, float, str]:
    """Infer a range when the caller did not provide one.

    An explicit range is only meaningful when the continuation parameter is
    explicit. The public wrapper enforces that rule so the same numeric range
    is never accidentally applied to unrelated parameters during an automatic
    search.
    """
    if requested_min is not None or requested_max is not None:
        if requested_min is None or requested_max is None:
            # Allow the unspecified side to be inferred, but keep this explicit
            # because it is tied to the one authoritative parameter.
            pass
        else:
            return float(requested_min), float(requested_max), "explicitly requested"

    # Preserve XPP's actual AUTO range when the source is available.
    if "parmin" in xpp_meta and "parmax" in xpp_meta:
        lo, hi = float(xpp_meta["parmin"]), float(xpp_meta["parmax"])
        if requested_min is not None:
            lo = float(requested_min)
        if requested_max is not None:
            hi = float(requested_max)
        return lo, hi, "inferred from XPP @ parmin/parmax"

    # Some XPP files persist AUTO's displayed x-axis limits even when
    # PARMIN/PARMAX are absent. They are not guaranteed to be the continuation
    # limits, so use them only as a clearly labelled fallback before inventing
    # a model-based range.
    if "autoxmin" in xpp_meta and "autoxmax" in xpp_meta:
        lo = float(xpp_meta["autoxmin"])
        hi = float(xpp_meta["autoxmax"])
        if requested_min is not None:
            lo = float(requested_min)
        if requested_max is not None:
            hi = float(requested_max)
        if hi > lo:
            return lo, hi, "inferred from XPP @ autoxmin/@autoxmax fallback"

    # Generic fallback: a decade either side of the parameter's current value
    # (the span the parameter screen examines), so the diagram shows the region
    # the model actually sits in. Explicit user ranges and XPP @ parmin/@ parmax
    # metadata take precedence above.
    default = float(model["param_defaults"].get(param_name, 1.0))
    if default != 0:
        lo, hi = sorted((default / 10.0, default * 10.0))
    else:  # no scale to go by
        lo, hi = -1.0, 1.0
    if requested_min is not None:
        lo = float(requested_min)
    if requested_max is not None:
        hi = float(requested_max)
    if hi <= lo:
        raise ValueError(f"Invalid continuation range [{lo}, {hi}] for '{param_name}'.")
    return lo, hi, "automatic range heuristic"


def _choose_plot_state(model: dict, requested: str | None, xpp_meta: dict) -> tuple[str | None, str]:
    """Choose the y-axis exactly from user/XPP metadata, not a variance heuristic."""
    if requested:
        if requested not in model["state_names"]:
            raise ValueError(
                f"{requested!r} is not a state variable. "
                f"Available state variables: {model['state_names']}"
            )
        return requested, "explicitly requested"

    # XPP uses autovar/xp/yp1/yp2 for plotting hints. Prefer yp1 when present
    # because it is the actual y-axis variable used by the XPP plotting setup;
    # autovar is the next-best explicit model hint.
    # XPP's AUTOVAR is specifically the variable AUTO uses for the
    # bifurcation-diagram y-axis.  Generic XPP YP1/YP2 plotting settings
    # must not override an explicit AUTO variable.
    for key, reason in (
        ("autovar", "XPP @ autovar"),
        ("yp1", "XPP @ yp1 fallback"),
        ("yp2", "XPP @ yp2 fallback"),
    ):
        value = xpp_meta.get(key)
        if value in model["state_names"]:
            return value, reason

    # There is no scientifically justified way to infer the user's preferred
    # observable from an arbitrary model. Use the first state deterministically
    # rather than selecting whichever state happens to have the largest numeric
    # variance on our branch.
    if model["state_names"]:
        return model["state_names"][0], "first state variable (no XPP/user plot variable supplied)"
    return None, "no state variable available"


_BIF_TYPE_DESCRIPTIONS = {
    "LP": "Saddle-Node / Fold (Limit Point) -- two steady states collide and annihilate",
    "BP": "Transcritical or Pitchfork (Branch Point) -- two solution branches cross/exchange stability",
    "HB": "Hopf point -- oscillatory eigenvalue crossing; periodic orbits and their stability require further checks",
    "NS": "Neutral saddle -- AUTO's Hopf test fired, but the eigenvalues there are real (two summing to "
          "zero): not a Hopf point, no cycles are born and stability does not change here",
}


def _is_hopf(ev) -> bool:
    """Whether eigenvalues at a point AUTO labelled HB contain a complex pair on the imaginary
    axis (auto_diagnostics.hopf_pair). No eigenvalues (None): keep AUTO's label."""
    if ev is None or not len(ev):
        return True
    from dynalyst.analysis.auto_diagnostics import hopf_pair
    return hopf_pair(ev)


def _onset_period(ev):
    """The period of the cycles born at a Hopf point, 2*pi/omega, omega the imaginary part of
    the complex pair nearest the imaginary axis; None without eigenvalues. The period further
    along the cycle branch differs: this one belongs to the Hopf point only."""
    if ev is None or not len(ev):
        return None
    pairs = [e for e in ev if e.imag > 1e-12]
    if not pairs:
        return None
    omega = min(pairs, key=lambda e: abs(e.real)).imag
    return float(2 * np.pi / omega)


START_RELATIVE_TOLERANCE = 1e-8


def _find_start_state(model: dict, param_name: str, param_start_value: float,
                       fixed_params: dict, nonneg: set | None = None) -> tuple[np.ndarray, dict]:
    """Find ONE equilibrium at the sweep's starting parameter value. The model's
    own default y0 is the first initial guess (so continuation starts from the
    branch the model's defaults sit on), then where the model settles from it
    and from zero, then spread-out starts; each with fsolve, then least squares.
    A point is an equilibrium when every rate of change is negligible next to the
    rate terms it balances (relative, so it does not depend on the units).
    When the model's initial state is non-negative, a non-negative equilibrium is
    preferred. Assumes `model` has already been through conservation_reduce."""
    param_values = dict(model["param_defaults"])
    param_values.update(fixed_params or {})
    param_values[param_name] = param_start_value

    rhs_fn, jac_fn = _lambdify_model(model)
    scale_fn = _term_scale_fn(model)
    param_vec = [param_values[p] for p in model["param_names"]]

    def f(y):
        return np.array(rhs_fn(*y, *param_vec), dtype=float)

    def jac(y):
        return np.array(jac_fn(*y, *param_vec), dtype=float)

    def imbalance(y):
        """max_i |dy_i/dt| / (sum of |rate terms| in equation i)."""
        with np.errstate(all="ignore"):
            r = np.abs(f(y))
            scale = np.abs(np.array(scale_fn(*y, *param_vec), dtype=float))
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(scale))):
            return np.inf
        scale = flux_scale(scale)      # an equation whose terms all vanish there: system scale
        ratio = np.where(r == 0, 0.0, r / np.where(scale > 0, scale, np.inf))
        return float(np.max(ratio)) if ratio.size else 0.0

    y0_default = np.array([model["y0"][s] for s in model["state_names"]], dtype=float)
    nonneg_model = bool(np.all(y0_default >= 0))
    # with model_prep's sign rule: the states kept >= 0, eliminated ones included
    checks = []
    if nonneg is not None:
        syms = model["state_symbols"] + model["param_symbols"]
        checks = [sp.lambdify(syms, sp.sympify(ex), "numpy")
                  for n, ex in (model.get("eliminated") or []) if n in nonneg]
        signed = np.array([s in nonneg for s in model["state_names"]])

    def physical(y):
        if nonneg is None:
            return not nonneg_model or np.all(y >= -START_RELATIVE_TOLERANCE * max(1.0, np.max(np.abs(y))))
        tol = START_RELATIVE_TOLERANCE * max(1.0, float(np.max(np.abs(y))))
        with np.errstate(all="ignore"):
            derived = [float(fn(*y, *param_vec)) for fn in checks]
        return bool(np.all(y[signed] >= -tol) and all(v >= -tol for v in derived))

    def starts():       # the simulations behind the other starts run only when needed
        yield y0_default
        yield from _candidate_starts({**model, "param_defaults": param_values}, f, 3,
                                     np.random.default_rng(0))[1:]

    best, fallback, closest, n_tried = None, None, np.inf, 0
    for y_init in starts():
        n_tried += 1
        if not np.all(np.isfinite(y_init)):
            continue
        tries = []
        with np.errstate(all="ignore"):
            try:
                tries.append(fsolve(f, y_init, fprime=jac, xtol=1e-12, full_output=True)[0])
            except Exception:
                pass
            try:
                tries.append(least_squares(f, y_init, jac=jac, xtol=1e-15, ftol=1e-15,
                                           gtol=1e-15, x_scale="jac").x)
            except Exception:
                pass
        for y in tries:
            err = imbalance(y)
            closest = min(closest, err)
            if err > START_RELATIVE_TOLERANCE:
                continue
            if physical(y):
                best = y
                break
            fallback = y if fallback is None else fallback
        if best is not None:
            break

    sol = best if best is not None else fallback
    if sol is None:
        raise ValueError(
            f"Could not find an equilibrium at {param_name}={param_start_value}: tried the "
            f"model's default initial conditions, where the model settles, and {n_tried} "
            f"starts in all (the closest point still had a rate of change "
            f"{closest:.2g} times the rates it should balance). Try find_steady_states "
            f"first to locate a valid steady state, or pass a different param range."
        )
    return np.where(np.abs(sol) < 1e-300, 0.0, sol), param_values


AUTO_BRANCH_COLUMNS = 6     # AUTO writes at most this many states per point to its output


def _output_slots(model: dict) -> dict:
    """{state name: PAR index} that carries every state into AUTO's output. AUTO writes
    at most 7 columns per point (io.f90, WRLINE), so beyond a few states the rest never
    reach the branch data; parameters listed in ICP are always written, so PVLS copies
    each state into one (AUTO's documented way of monitoring extra quantities). Empty
    when every state fits."""
    names = model["state_names"]
    if len(names) <= AUTO_BRANCH_COLUMNS:
        return {}
    n_par = len(model["param_names"])
    first = max(auto_par_index(n_par - 1) if n_par else 0, max(AUTO_RESERVED_PAR)) + 1
    return {s: first + i for i, s in enumerate(names)}


def _write_equations_file(model: dict, param_name: str, y_start: np.ndarray,
                           param_values: dict, path: Path) -> bool:
    """Write the AUTO equations file. States and parameters appear only as
    U(i)/PAR(i) (dynalyst/fortran.py), so no model name can clash with
    Fortran. Returns whether the analytic Jacobians were written: when a
    derivative cannot be expressed in Fortran (e.g. of max/min/abs, which
    give Heaviside/sign terms), AUTO differentiates numerically (JAC=0)."""
    state_symbols = model["state_symbols"]
    param_symbols = model["param_symbols"]
    rhs = model["rhs"]
    n = len(state_symbols)
    sub = auto_symbols(state_symbols, param_symbols)
    names = model.get("state_names") or [str(s) for s in state_symbols]

    f_lines = [fortran_assign(sp.sympify(expr).xreplace(sub), f"F({i + 1})", f"d{names[i]}/dt")
               for i, expr in enumerate(rhs)]
    jac_lines = []
    try:
        jac = jacobian(model)
        jacp = sp.Matrix(rhs).jacobian(param_symbols) if param_symbols else sp.zeros(n, 0)
        jac_lines.append("      IF(IJAC.EQ.0)RETURN")
        for i in range(n):
            for j in range(n):
                jac_lines.append(fortran_assign(jac[i, j].xreplace(sub), f"DFDU({i + 1},{j + 1})",
                                                f"d(d{names[i]}/dt)/d{names[j]}"))
        jac_lines.append("      IF(IJAC.EQ.1)RETURN")
        for i in range(n):
            for j in range(len(param_symbols)):
                jac_lines.append(fortran_assign(jacp[i, j].xreplace(sub),
                                                f"DFDP({i + 1},{auto_par_index(j)})",
                                                f"d(d{names[i]}/dt)/d{param_symbols[j]}"))
    except FortranError:
        jac_lines = []

    lines = []
    lines.append("      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)")
    lines.append("      IMPLICIT NONE")
    lines.append("      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC")
    lines.append("      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)")
    lines.append("      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)")
    lines.append("      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)")
    lines += f_lines + jac_lines
    lines.append("      END SUBROUTINE FUNC")
    lines.append("")
    lines.append("      SUBROUTINE STPNT(NDIM,U,PAR,T)")
    lines.append("      IMPLICIT NONE")
    lines.append("      INTEGER, INTENT(IN) :: NDIM")
    lines.append("      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)")
    lines.append("      DOUBLE PRECISION, INTENT(IN) :: T")
    for i, p in enumerate(param_symbols):
        lines.append(f"      PAR({auto_par_index(i)}) = {param_values[str(p)]!r}")
    for i in range(n):
        lines.append(f"      U({i + 1}) = {float(y_start[i])!r}")
    lines.append("      END SUBROUTINE STPNT")
    lines.append("")
    lines.append("      SUBROUTINE BCND")
    lines.append("      END SUBROUTINE BCND")
    lines.append("      SUBROUTINE ICND")
    lines.append("      END SUBROUTINE ICND")
    lines.append("      SUBROUTINE FOPT")
    lines.append("      END SUBROUTINE FOPT")
    slots = _output_slots(model)
    if slots:
        lines.append("      SUBROUTINE PVLS(NDIM,U,PAR)")
        lines.append("      IMPLICIT NONE")
        lines.append("      INTEGER, INTENT(IN) :: NDIM")
        lines.append("      DOUBLE PRECISION, INTENT(IN) :: U(NDIM)")
        lines.append("      DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)")
        for i, s_name in enumerate(model["state_names"]):
            lines.append(f"      PAR({slots[s_name]}) = U({i + 1})")
        lines.append("      END SUBROUTINE PVLS")
    else:
        lines.append("      SUBROUTINE PVLS")
        lines.append("      END SUBROUTINE PVLS")

    _write_lf(path, "\n".join(lines) + "\n")
    return bool(jac_lines)


_AUTO_LINE_LIMIT = 1900   # characters: below the 2048 AUTO's Fortran reads of one line


def _write_constants_file(model: dict, param_name: str, param_min: float, param_max: float,
                           path: Path, n_pts: int, overrides: dict | None = None,
                           analytic_jacobian: bool = True):
    state_names = model["state_names"]
    param_names = model["param_names"]
    n = len(state_names)
    icp_index = auto_par_index(param_names.index(param_name))
    unames = {i + 1: s for i, s in enumerate(state_names)}
    slots = _output_slots(model)
    # only the names AUTO reports back (the swept parameter and the state slots): every
    # parameter's name on one line passes AUTO's 2048-character line limit in a large model
    # (Chassagnole2002: 170 parameters) and AUTO stops with "bad value on line 3"
    parnames = {icp_index: param_name, **{idx: s for s, idx in slots.items()}}
    icp_list = [icp_index, *slots.values()]
    span = max(abs(param_max - param_min), 1e-6)
    # AUTO's own manual (doc/auto.tex, "Efficiency"): DSMAX "should be
    # sufficiently small, in order to prevent stepping over bifurcations
    # without detecting them." The old DSMAX=span/20 let AUTO take steps as
    # coarse as 5% of the whole sweep -- plenty of room to step clean over a
    # narrow/sharp fold (SN) or branch point without ever bisecting onto it,
    # which silently produces a smooth-looking but bifurcation-free plot.
    # Tie the max step to the requested point density instead, so asking
    # for more resolution (n_pts) actually buys finer bifurcation detection,
    # not just a smoother-looking curve.
    # bounded either way: a retry that multiplies the resolution must not ask AUTO for
    # hundreds of millions of steps (it then produces nothing at all)
    n_pts = min(max(n_pts, 50), config.AUTO_MAX_POINTS)
    dsmax = span / n_pts
    ds = dsmax / 10
    # DSMIN is a FLOOR on how small AUTO is allowed to shrink the step when
    # the corrector struggles to converge -- once it can't converge even at
    # DSMIN, it gives up on that stretch (this can show up as the branch
    # simply stopping well short of param_max). A stiff patch near a real
    # bifurcation (transcritical/branch-point crossings are notorious for
    # this) can need steps far smaller than a "reasonable-looking" floor
    # like ds/100, so keep DSMIN close to negligible and let NMX (below)
    # be the real budget control instead.
    dsmin = max(dsmax / 1e6, 1e-12)
    # More steps are needed now that DSMAX is much smaller, so NMX must grow
    # to match or long/steep branches get cut off before reaching param_max
    # (AUTO calls this an "MX" / abnormal termination -- see below). Bumped
    # further to give a stiff crossing region room to grind through with
    # many small retried steps rather than hitting the ceiling.
    nmx = max(n_pts * 40, 20000)
    itmx, itnw, nwtn = 12, 12, 5
    epsl, epsu, epss = 1e-7, 1e-7, 1e-5

    # Every auto-computed value above is just a sensible starting point
    # (the same spirit as XPPAUT's AUTO Numerics panel picking defaults
    # from your parameter range) -- any of them can be pinned explicitly
    # via `overrides` instead of trusting the heuristic, e.g. when a model
    # needs a tighter DSMIN or more Newton iterations than the default
    # guess provides.
    o = overrides or {}
    ds = o.get("ds", ds)
    dsmin = o.get("dsmin", dsmin)
    dsmax = o.get("dsmax", dsmax)
    nmx = min(int(o.get("nmx", nmx)), config.AUTO_MAX_NMX)
    dsmin = max(float(dsmin), config.AUTO_MIN_DSMIN)
    itmx = o.get("itmx", itmx)
    itnw = o.get("itnw", itnw)
    nwtn = o.get("nwtn", nwtn)
    epsl = o.get("epsl", epsl)
    epsu = o.get("epsu", epsu)
    epss = o.get("epss", epss)

    # AUTO copies the unames line into fort.2, which its Fortran reads with a 2048-character
    # line buffer: a longer line (many species) shifts what it reads and AUTO stops with
    # "bad value on line 3". Names are written compactly and cut to the 13 characters AUTO
    # keeps anyway; if that is still too long they are left out (AUTO then calls the states
    # U(1)...; their values are read from the named output slots in parnames, not these).
    unames_text = "{" + ",".join(f"{i}:{s[:13]!r}" for i, s in unames.items()) + "}"
    if len(unames_text) > _AUTO_LINE_LIMIT:
        unames_text = "{}"
    content = textwrap.dedent(f"""\
        unames = {unames_text}
        parnames = {parnames!r}
        NDIM={n:4d}, IPS =   1, IRS =   0, ILP =   1
        ICP =  {icp_list}
        NTST=  35, NCOL=   4, IAD =   3, ISP =   2, ISW = 1, IPLT= 0, NBC= 0, NINT= 0
        NMX= {nmx:6d}, NPR= {nmx:6d}, MXBF=   5, IID =   1, ITMX= {itmx}, ITNW= {itnw}, NWTN= {nwtn}, JAC= {1 if analytic_jacobian else 0}
        EPSL= {epsl!r}, EPSU = {epsu!r}, EPSS = {epss!r}
        DS  = {ds!r}, DSMIN=  {dsmin!r}, DSMAX=  {dsmax!r}, IADS=   1
        NPAR = {max([auto_par_index(len(param_names) - 1), *parnames])}, THL =  {{}}, THU =  {{}}
        UZR = {{}}
        RL0 = {param_min!r}, RL1 = {param_max!r}, A0 = -1e9, A1 = 1e9
        """)
    _write_lf(path, content)
    return {"ds": ds, "dsmin": dsmin, "dsmax": dsmax, "nmx": nmx, "itmx": itmx,
            "itnw": itnw, "nwtn": nwtn, "epsl": epsl, "epsu": epsu, "epss": epss}


_RUNNER_SCRIPT = r"""
import contextlib
import io
import json
import os
import re
import shutil
import sys
import numpy as np

sys.path.insert(0, os.environ["AUTO_DIR_PY"])
import auto as auto_module
from auto import run

label = sys.argv[1]
errors = []
run_records = []
RAW = {}
MAX_SWITCHES = 200
try:
    with open("runner_config.json") as f:
        CONFIG = json.load(f)
except Exception:
    CONFIG = {}

diagnostics = {
    "auto_module_file": getattr(auto_module, "__file__", None),
    "auto_dir_env": os.environ.get("AUTO_DIR"),
    "auto_executable_found": bool(
        os.environ.get("PATH", "").split(os.pathsep) and
        any(os.path.exists(os.path.join(p, "auto"))
            for p in os.environ.get("PATH", "").split(os.pathsep))
    ),
    "cwd": os.getcwd(),
}


def _parse_auto_special_points(stdout):
    # AUTO equilibrium table:
    # BR PT TY LAB PAR L2-NORM <state values...>
    parsed = []
    if not stdout:
        return parsed

    row_re = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+(EP|BP|LP|HB|UZ|MX)\s+(\d+)"
        r"\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
        r"\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
        r"\s*(.*)$"
    )

    state_names = None

    for line in stdout.splitlines():
        line = line.rstrip()

        if "TY" in line and "LAB" in line and "L2-NORM" in line:
            parts = line.split()
            try:
                k = parts.index("L2-NORM")
                state_names = parts[k + 1:]
            except ValueError:
                pass
            continue

        m = row_re.match(line)
        if not m:
            continue

        tail = m.group(7).split()
        states = {}

        if state_names and len(tail) >= len(state_names):
            for name, value in zip(state_names, tail[:len(state_names)]):
                try:
                    states[name] = float(value)
                except ValueError:
                    pass

        parsed.append({
            "branch": int(m.group(1)),
            "point": int(m.group(2)),
            "type": m.group(3),
            "label": int(m.group(4)),
            "parameter": float(m.group(5)),
            "l2_norm": float(m.group(6)),
            "states": states,
        })

    return parsed


def safe_run(tag, *args, **kwargs):
    # Each AUTO invocation is kept as its own result. In particular, the two
    # ISW=-1 directions from a BP are NOT merged.
    buf = io.StringIO()
    # AUTO's own eigenvalues / Floquet multipliers (fort.9) are overwritten by the next run:
    # keep each run's copy as d.<tag>, a file copy only (reading AUTO's objects breaks restarts)
    # on Windows the previous AUTO process can still hold fort.9 for a moment: retry, then go
    # on (only this run's eigenvalue copy d.<tag> may be stale then; the run itself is fine)
    for _ in range(20):
        try:
            if os.path.exists("fort.9"):
                os.remove("fort.9")
            break
        except OSError:
            import time as _time
            _time.sleep(0.1)

    try:
        with contextlib.redirect_stdout(buf):
            try:
                result = run(*args, **kwargs)
            finally:
                try:
                    if os.path.exists("fort.9"):
                        shutil.copyfile("fort.9", "d." + tag)
                except OSError:      # a locked file costs the eigenvalue copy, not the run
                    pass

        stdout = buf.getvalue()
        branches = list(result)
        RAW[tag] = result          # the whole run: a restart point is taken from it by label

        run_records.append({
            "tag": tag,
            "kwargs": {k: str(v) for k, v in kwargs.items()},
            "n_branches": len(branches),
            "branch_sizes": [
                len(b.toarray()[0]) if len(b.toarray()) else 0
                for b in branches
            ],
            "special_points": _parse_auto_special_points(stdout),
            "stdout": stdout,
        })

        return branches

    except Exception as e:
        stdout = buf.getvalue()
        errors.append(f"{tag}: {e}")
        run_records.append({
            "tag": tag,
            "kwargs": {k: str(v) for k, v in kwargs.items()},
            "n_branches": 0,
            "branch_sizes": [],
            "special_points": _parse_auto_special_points(stdout),
            "stdout": stdout,
            "error": str(e),
            "traceback": __import__("traceback").format_exc()[-1500:],
        })
        return []


def _switch_bps(branches, record, source_tag, visited):
    # IMPORTANT: use AUTO's terminal table as a fallback for BP discovery.
    # AUTO's terminal table can expose branch points even when br.labels omits them.
    switched = []

    for sp in record.get("special_points", []):
        if sp.get("type") != "BP" or not sp.get("label"):
            continue

        if len(visited) >= MAX_SWITCHES:
            return switched

        branch_number = int(sp["branch"])
        branch_index = branch_number - 1

        if branch_index < 0 or branch_index >= len(branches):
            errors.append(
                f"{source_tag}: AUTO reported BP LAB={sp['label']} on "
                f"BR={branch_number}, but {len(branches)} branch objects were returned."
            )
            continue

        try:
            source_branch = branches[branch_index]
            # the restart is taken from the whole run, as for Hopf points: a point taken from
            # a branch object lacks AUTO's branch/label fields (_mbr, _mlab) and cannot restart
            bp_point = (RAW[source_tag](int(sp["label"])) if source_tag in RAW
                        else source_branch(int(sp["label"])))

            key = (
                int(sp["label"]),
                round(float(sp["parameter"]), 7),
                branch_number,
                # a coordinate comes back as a 1-element array: numpy >= 2.x refuses float() on it
                tuple(round(float(np.ravel(bp_point[c])[0]), 7) for c in bp_point.coordnames),
            )
        except Exception as e:
            errors.append(
                f"{source_tag}: could not retrieve BP LAB={sp['label']}: {e}"
            )
            continue

        if key in visited:
            continue

        visited.add(key)

        # the equations and constants of this run go along: a restart point alone leaves
        # some constants unset (AUTO then fails comparing None with a number)
        plus = safe_run(
            f"BP-{sp['label']}-plus",
            bp_point,
            e=label, c=label,
            ISW=-1,
        )
        plus_record = run_records[-1]

        minus = safe_run(
            f"BP-{sp['label']}-minus",
            bp_point,
            e=label, c=label,
            ISW=-1,
            DS="-",
        )
        minus_record = run_records[-1]

        switched.extend(plus)
        switched.extend(minus)

        # Recursively inspect the newly switched runs for additional BPs.
        switched.extend(
            _switch_bps(
                plus,
                plus_record,
                f"BP-{sp['label']}-plus",
                visited,
            )
        )
        switched.extend(
            _switch_bps(
                minus,
                minus_record,
                f"BP-{sp['label']}-minus",
                visited,
            )
        )

    return switched


# 1) Main continuation.
main_forward = safe_run("main-forward", label)
main_forward_record = run_records[-1] if run_records else {}

main_backward = safe_run("main-backward", label, DS="-")
main_backward_record = run_records[-1] if run_records else {}

combined = list(main_forward) + list(main_backward)

# 2) Detect BPs from AUTO's actual terminal output, even when br.labels does
#    not expose them, and perform the user's exact XPP branch-switch recipe.
visited = set()
switched = []

switched.extend(
    _switch_bps(
        main_forward,
        main_forward_record,
        "main-forward",
        visited,
    )
)

switched.extend(
    _switch_bps(
        main_backward,
        main_backward_record,
        "main-backward",
        visited,
    )
)

combined.extend(switched)

diagnostics["n_raw_branches_from_auto"] = len(combined)
diagnostics["raw_branch_point_counts"] = [
    len(br.toarray()[0]) if len(br.toarray()) else 0
    for br in combined
]
diagnostics["n_bp_switch_attempts"] = len(visited)
diagnostics["hit_switch_cap"] = len(visited) >= MAX_SWITCHES
diagnostics["run_records"] = run_records

terminal_special = []
for rec in run_records:
    for sp in rec.get("special_points", []):
        terminal_special.append(dict(sp, run_tag=rec["tag"]))

diagnostics["terminal_special_points"] = terminal_special

out = {
    "branches": [],
    "errors": errors,
    "diagnostics": diagnostics,
}

for br_idx, br in enumerate(combined):
    coordnames = [str(c) for c in br.coordnames]
    arr = np.asarray(br.toarray(), dtype=float)
    stab = [int(x) for x in br.stability()]
    special = []

    for idx, val in br.labels.sortByIndex():
        for ty, info in val.items():
            if ty == "No Label":
                continue
            special.append({
                "index": int(idx),
                "type": str(ty),
                "label": (
                    int(info.get("LAB"))
                    if info.get("LAB") is not None else None
                ),
            })

    out["branches"].append({
        "coordnames": coordnames,
        "data": arr.tolist(),
        "stability_segments": stab,
        "special_points": special,
        "branch_index": br_idx,
        "n_points": int(arr.shape[1]) if arr.ndim == 2 else 0,
    })

# 3) Periodic orbits: from each Hopf point, continue the cycles it gives birth to (IPS=2,
#    the period in PAR(11)). Each stored cycle keeps its parameter value, period, stability
#    (Floquet multipliers) and every species over one period, so the caller can draw the
#    max/min envelope of any species. A failure here never costs the steady-state result.
out["periodic"] = []
with open("result.json", "w") as f:     # the steady states are kept if the cycles run out of time
    json.dump(out, f)


def _last_end(nmx):
    # how AUTO ended the run just made, from its own last table row (its point count and EP
    # or MX label; the parameter there can be NaN, so it is not parsed as a number)
    ends = re.findall(r"^\s*\d+\s+-?(\d+)\s+(EP|MX)\s+\d+", run_records[-1].get("stdout") or "",
                      re.M)
    return {"point": int(ends[-1][0]), "type": ends[-1][1], "nmx": nmx} if ends else None


def _period_doublings(cyc):
    # (label, parameter, type) of each period doubling and branch point, where AUTO can switch
    # branch, read from the labels only: reading a solution (pb(LAB), stability()) corrupts what
    # AUTO writes for a later restart from it
    out_pd = []
    for pb in cyc:
        barr = np.asarray(pb.toarray(), dtype=float)
        for idx, val in pb.labels.sortByIndex():
            for ty in ("PD", "BP"):
                info = val.get(ty)
                if info and info.get("LAB") is not None:
                    out_pd.append((int(info["LAB"]), float(barr[0, idx]), ty))
    return out_pd


def _keep_cycles(cyc, what, origin, end):
    # stores each cycle branch of a run in out["periodic"]; call it only after every restart
    # from that run is done (see _period_doublings)
    for pb in cyc:
        try:
            barr = np.asarray(pb.toarray(), dtype=float)
            bnames = [str(c) for c in pb.coordnames]
            cycles = []
            for idx, val in pb.labels.sortByIndex():
                for ty, info in val.items():
                    if ty == "No Label" or info.get("LAB") is None:
                        continue
                    try:
                        sol = pb(int(info["LAB"]))
                        orbit = {str(n): np.asarray(sol[n], dtype=float).tolist()
                                 for n in sol.coordnames}
                    except Exception:
                        continue
                    cycles.append({"index": int(idx), "type": str(ty), "orbit": orbit})
            out["periodic"].append({
                **origin, "coordnames": bnames, "data": barr.tolist(), "end": end,
                "stability_segments": [int(x) for x in pb.stability()],
                "cycles": cycles})
        except Exception as e:
            errors.append(f"cycles from {what}: {e}")


# AUTO's own Hopf type: its first Lyapunov coefficient (GH test function, equilibrium.f90
# FNGHEQ) is computed only while it follows the Hopf point in two parameters (ISW=2), so a
# 3-step Hopf continuation from each Hopf point; l1 < 0 supercritical, l1 > 0 subcritical.
# A run that stops before AUTO writes l1 is retried with another second parameter.
out["hopf_l1"] = []
if CONFIG.get("second_parameters"):
    typed = []
    for tag, rec in {r.get("tag"): r for r in run_records
                     if str(r.get("tag")).startswith(("main-", "BP-"))}.items():
        for sp in rec.get("special_points", []):
            if (sp.get("type") != "HB" or not sp.get("label") or len(typed) >= 5 or any(
                    abs(sp["parameter"] - t) <= 1e-6 * max(abs(t), 1e-12) for t in typed)):
                continue
            typed.append(sp["parameter"])
            l1 = None
            for second in CONFIG["second_parameters"]:
                l1_tag = f"L1-{tag}-{int(sp['label'])}-{int(second)}"
                try:
                    safe_run(l1_tag, RAW[tag](int(sp["label"])), ISW=2,
                             ICP=[int(CONFIG["icp"]), int(second)], NMX=3, NPR=0,
                             IID=2)     # AUTO writes the GH value to fort.9 only from IID 2
                    with open("d." + l1_tag) as fh:
                        m = re.search(r"GH\s+Function:\s*(\S+)", fh.read())
                    if m:
                        l1 = float(m.group(1))
                        break
                except Exception as e:
                    errors.append(f"Hopf type at {sp['parameter']}: {e}")
            out["hopf_l1"].append({"parameter": sp["parameter"], "l1": l1})


pd_runs = [0]               # period-doubled branches followed in this run, at most 4
if CONFIG.get("periodic"):
    # Hopf points on every equilibrium branch (the main ones first, then the branches switched
    # to at branch points). A tag can repeat (labels restart in each run) and RAW keeps the last
    # run with it, so a Hopf point is taken only from the run RAW holds for its tag
    last = {r.get("tag"): r for r in run_records if str(r.get("tag")).startswith(("main-", "BP-"))}
    hopfs = []
    for tag, rec in last.items():
        for sp in rec.get("special_points", []):
            if sp.get("type") == "HB" and sp.get("label") and not any(
                    abs(sp["parameter"] - h[2]) <= 1e-6 * max(abs(h[2]), 1e-12) for h in hopfs):
                hopfs.append((tag, int(sp["label"]), float(sp["parameter"])))
    out["hopf_not_followed"] = [h[2] for h in hopfs[5:]]
    for tag, lab, at in hopfs[:5]:
        try:
            start = RAW[tag](lab)
        except Exception as e:
            errors.append(f"cycles from HB LAB={lab}: could not take the Hopf point: {e}")
            continue
        dsmax = float(CONFIG.get("dsmax") or 0.01)
        # AUTO measures a step in (parameter, cycle, period) together, so a step sized to the
        # parameter range alone stalls when the period is long: allow large steps and let
        # AUTO adapt them (IADS). The first step is small: a large one jumps past the cycles
        # next to the Hopf point (their stability, a period doubling just beyond it)
        nmx = int(CONFIG.get("periodic_nmx", 500))
        steps = dict(IPS=2, ICP=[int(CONFIG["icp"]), 11], ISP=2, ILP=1, NMX=nmx, NPR=2,
                     DS=dsmax / 10, DSMAX=max(10 * dsmax, 0.1), IADS=1)
        cyc = safe_run(f"HB-{lab}-cycles", start, ISW=1, **steps)
        end = _last_end(nmx)
        if end and end["point"] >= nmx:
            # stopped at the step limit (a canard, a near-homoclinic branch): the same run once
            # more with 4x the steps; most branches end sooner and keep the short run
            if run_records and run_records[-1].get("tag") == f"HB-{lab}-cycles":
                run_records.pop()
            nmx = 4 * nmx
            steps["NMX"] = nmx
            cyc = safe_run(f"HB-{lab}-cycles", start, ISW=1, **steps)
            end = _last_end(nmx)
        # the cycles of twice the period born at a period doubling, and the other branch at a
        # branch point of the cycles (AUTO's branch switch, ISW=-1): whether oscillations go on
        # past it, and how, is on that branch
        twins = []
        for pd_lab, pd_at, pd_type in _period_doublings(cyc)[:2]:
            if pd_runs[0] >= 4:
                break
            pd_runs[0] += 1
            try:
                # from the whole run, not the branch object: AUTO cannot restart from a point
                # taken out of a single branch (its solution lacks the branch number)
                twice = safe_run(f"{pd_type}-{pd_lab}-cycles", RAW[f"HB-{lab}-cycles"](pd_lab), ISW=-1, **steps)
            except Exception as e:
                errors.append(f"cycles from {pd_type} LAB={pd_lab}: could not take the point: {e}")
                continue
            twins.append((twice, pd_lab, pd_at, _last_end(nmx), pd_type))
        _keep_cycles(cyc, f"HB LAB={lab}", {"hopf_parameter": at}, end)
        for twice, pd_lab, pd_at, pd_end, pd_type in twins:
            _keep_cycles(twice, f"{pd_type} LAB={pd_lab}",
                         {"hopf_parameter": at, ("pd_parameter" if pd_type == "PD" else "bp_parameter"): pd_at},
                         pd_end)

with open("result.json", "w") as f:
    json.dump(out, f)
"""






# In automatic multi-parameter mode, cap the entire search rather than
# multiplying the per-AUTO timeout by the number of candidates.
_AUTO_SEARCH_TOTAL_TIMEOUT = 900


def _auto_timeout_seconds(work_dir: Path) -> int:
    """Choose a bounded timeout from generated AUTO settings/model size."""
    cfg = work_dir / "runner_config.json"
    nmx = 200
    n_pts = 500
    if cfg.exists():
        try:
            meta = json.loads(cfg.read_text())
            nmx = int(meta.get("nmx") or nmx)
            n_pts = int(meta.get("n_pts") or n_pts)
        except Exception:
            pass
    # Base + modest scaling; never below 180s or above 900s.
    return max(180, min(900, 180 + int(0.08 * nmx + 0.15 * n_pts)))


def _wsl_opt_in() -> bool:
    """AUTO_USE_WSL=1 forces WSL on Windows. Default: native everywhere."""
    return os.name == "nt" and os.environ.get("AUTO_USE_WSL", "0") == "1"


def _wsl_available() -> bool:
    return os.name == "nt" and shutil.which("wsl") is not None and bool(os.environ.get("AUTO_DIR_WSL"))


def _native_auto_dir() -> Path:
    """AUTO install dir: env AUTO_DIR wins, else config.AUTO_DIR. Same on Windows/Linux."""
    env_dir = os.environ.get("AUTO_DIR")
    return Path(env_dir) if env_dir else Path(config.AUTO_DIR)


def _wsl_auto_dir() -> str:
    d = os.environ.get("AUTO_DIR_WSL")
    if not d:
        raise RuntimeError("AUTO_DIR_WSL not set (Linux path of AUTO-07p inside WSL).")
    return d.rstrip("/")


def _preflight(use_wsl: bool) -> list[str]:
    """NON-FATAL checks. Returns warnings; never raises.

    A missing tool here does not stop the run (AUTO may still find it another
    way). Warnings are attached to the error message only if the run then fails.
    """
    warnings: list[str] = []
    try:
        if use_wsl:
            if shutil.which("wsl") is None:
                warnings.append("'wsl' not found on PATH.")
            return warnings
        for tool in ("gfortran",):      # AUTO's Python interface builds models without make
            if shutil.which(tool) is None:
                warnings.append(f"'{tool}' not found on PATH.")
        if not (_native_auto_dir() / "python").is_dir():
            warnings.append(f"AUTO dir has no 'python' subfolder: {_native_auto_dir()}")
    except Exception as e:  # never let a check break the run
        warnings.append(f"preflight check error: {e}")
    return warnings


def _runner_command_and_env(work_dir: Path, label: str, use_wsl: bool) -> tuple[list[str], dict]:
    """Build (cmd, env) for the AUTO runner, native or via WSL."""
    env = dict(os.environ)
    if use_wsl:
        auto_dir = _wsl_auto_dir()
        wsl_dir = subprocess.check_output(
            ["wsl", "wslpath", "-a", str(work_dir)], text=True,
            encoding="utf-8", errors="replace",
        ).strip()
        script = (
            f"export AUTO_DIR={shlex.quote(auto_dir)}; "
            f"export AUTO_DIR_PY={shlex.quote(auto_dir + '/python')}; "
            f"export PATH={shlex.quote(auto_dir + '/bin')}:{shlex.quote(auto_dir + '/cmds')}:$PATH; "
            f"cd {shlex.quote(wsl_dir)} && python3 _runner.py {shlex.quote(label)}"
        )
        return ["wsl", "bash", "-lc", script], env
    auto_dir = _native_auto_dir()
    # the library compiled for this machine; RuntimeError (a setup problem) if none
    env["AUTO_LIB_DIR"] = posix_path(find_lib_dir(auto_dir))
    env["AUTO_DIR"] = posix_path(auto_dir)
    env["AUTO_DIR_PY"] = str(auto_dir / "python")
    env["PATH"] = os.pathsep.join([
        str(auto_dir / "bin"),
        str(auto_dir / "cmds"),
        env.get("PATH", ""),
    ])
    return [sys.executable, "_runner.py", label], env


AUTO_LOG_MAX_BYTES = 200 * 2**20   # an AUTO output file this large means a diverged run
_AUTO_LOGS = ("fort.7", "fort.8", "fort.9", "_stdout.txt", "_stderr.txt")


def _kill_tree(proc) -> None:
    """Stop the runner and everything it started: AUTO's compiled model program runs as a
    child process, which outlives a parent killed on its own (and keeps writing)."""
    import os
    import signal
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=60)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def _watch_auto(cmd, work_dir: Path, env, out_f: Path, err_f: Path, timeout_seconds) -> str | None:
    """Run AUTO, watching it: None when it finished, "time" when it ran past
    timeout_seconds, "diverged" when one of its output files grew past AUTO_LOG_MAX_BYTES
    (a NaN / singular-matrix loop writes a note at every step). A stopped run is killed
    with its whole process tree and its oversized files are deleted."""
    import os
    popen = {}
    if os.name == "nt":
        popen["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen["start_new_session"] = True

    def size(f):        # AUTO removes and rewrites its files while it runs
        try:
            return (work_dir / f).stat().st_size
        except OSError:
            return 0
    with open(out_f, "wb") as out, open(err_f, "wb") as err:
        proc = subprocess.Popen(cmd, cwd=str(work_dir), env=env, stdout=out, stderr=err, **popen)
        began, stopped = time.monotonic(), None
        while proc.poll() is None:
            time.sleep(1.0)
            if timeout_seconds and time.monotonic() - began > timeout_seconds:
                stopped = "time"
            elif any(size(f) > AUTO_LOG_MAX_BYTES for f in _AUTO_LOGS):
                stopped = "diverged"
            if stopped:
                _kill_tree(proc)
                break
    if stopped:
        for f in _AUTO_LOGS:
            p = work_dir / f
            try:
                if p.exists() and p.stat().st_size > AUTO_LOG_MAX_BYTES // 4:
                    p.unlink()
            except OSError:
                pass
    return stopped


def _run_auto_once(work_dir: Path, label: str, timeout_seconds: int, use_wsl: bool,
                   script: str = _RUNNER_SCRIPT) -> dict:
    warnings = _preflight(use_wsl)
    _write_lf(work_dir / "_runner.py", script)
    result_path = work_dir / "result.json"
    if result_path.exists():
        result_path.unlink()
    cmd, env = _runner_command_and_env(work_dir, label, use_wsl)
    mode = "wsl" if use_wsl else "native"
    out_f, err_f = work_dir / "_stdout.txt", work_dir / "_stderr.txt"

    def _tail(path, n):
        try:
            with open(path, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - n))
                return fh.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    stopped = _watch_auto(cmd, work_dir, env, out_f, err_f, timeout_seconds)
    if stopped:
        if result_path.exists():
            # the steady-state continuation finished; only the cycles were stopped
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
                data.setdefault("errors", []).append(
                    "cycles from the Hopf point(s): " + (
                        f"stopped at the {timeout_seconds}s time limit" if stopped == "time"
                        else "AUTO diverged (NaN / singular matrix: its log grew past "
                             f"{AUTO_LOG_MAX_BYTES // 2**20} MB) and was stopped"))
                data["auto_stdout_log"] = _tail(out_f, 5000)
                data["run_mode"] = mode
                return data
            except (OSError, ValueError):
                pass
        if stopped == "time":
            raise RuntimeError(
                f"AUTO-07p timed out after {timeout_seconds}s ({mode}). "
                f"Try reducing n_pts/NMX or increasing the AUTO timeout for a large/stiff model.\n"
                f"stdout:\n{_tail(out_f, 5000)}\nstderr:\n{_tail(err_f, 5000)}")
        raise RuntimeError(
            "AUTO-07p diverged: the continuation hit NaN / a singular matrix and kept "
            f"iterating (its log grew past {AUTO_LOG_MAX_BYTES // 2**20} MB), so it was stopped. "
            "The model may have no steady state AUTO can follow from this start, or be badly "
            "scaled; try another continuation parameter or range.")
    if not result_path.exists():
        raise RuntimeError(
            f"AUTO-07p run failed in {mode} mode (no result.json produced).\n"
            f"preflight warnings: {warnings or 'none'}\n"
            f"stdout:\n{_tail(out_f, 3000)}\nstderr:\n{_tail(err_f, 3000)}"
        )
    data = json.loads(result_path.read_text(encoding="utf-8"))
    data["auto_stdout_log"] = _tail(out_f, 20000)
    stderr_text = _tail(err_f, 5000)
    if stderr_text.strip():
        data["auto_stderr_log"] = stderr_text
    data["run_mode"] = mode
    if warnings:
        data["preflight_warnings"] = warnings
    return data


def _run_auto(work_dir: Path, label: str, timeout_override: int | None = None,
              script: str = _RUNNER_SCRIPT) -> dict:
    """Run AUTO. Works on Windows and Linux without configuration changes.

    Order: native first (default on both OSes). On Windows, if native fails and
    WSL is usable (AUTO_DIR_WSL set, 'wsl' present), retry once inside WSL.
    AUTO_USE_WSL=1 forces WSL first. Timeouts are not retried.
    """
    timeout_seconds = int(timeout_override) if timeout_override is not None else _auto_timeout_seconds(work_dir)
    timeout_seconds = max(1, timeout_seconds)

    modes = [False]
    if _wsl_opt_in():
        modes = [True]
    if _wsl_available() and True not in modes:
        modes.append(True)  # fallback only

    last_err: Exception | None = None
    for use_wsl in modes:
        try:
            return _run_auto_once(work_dir, label, timeout_seconds, use_wsl, script)
        except RuntimeError as e:
            last_err = e
            if "timed out" in str(e):
                raise
    raise last_err  # type: ignore[misc]


def _stability_word(sign, physical=True) -> str:
    if not physical:
        return "non-physical"
    return "stable" if sign > 0 else "unstable" if sign < 0 else "unknown"


def _segment(xs, ys, requested, param_name, plot_state) -> dict | None:
    """One stretch of a branch, inside the requested range: where it starts and ends in the
    parameter and in the plotted species, and its lowest and highest plotted value."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    keep = (xs >= requested[0]) & (xs <= requested[1]) & np.isfinite(ys)
    if not keep.any():
        return None
    x, y = xs[keep], ys[keep]
    g = lambda v: float(f"{float(v):.6g}")
    return {f"{param_name}_from": g(x[0]), f"{param_name}_to": g(x[-1]),
            f"{plot_state}_from": g(y[0]), f"{plot_state}_to": g(y[-1]),
            f"{plot_state}_range": [g(y.min()), g(y.max())], "n_points": int(keep.sum())}


def _write_branch_csv(path, branch_rows, cycle_rows):
    """Every computed point: the equilibrium branches, then the cycles (if any)."""
    import csv
    rows = branch_rows + [{"branch": "cycle", **r} for r in cycle_rows]
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _stretches(points: list, stability, within: tuple | None = None) -> list:
    """[[first, last] parameter value] of each run of consecutive cycle points (in branch
    order) whose stability is `stability` (True, False or None for unknown). `within` cuts each
    run to a range: a run that crosses the range covers it, with or without a saved point there."""
    out, run = [], []
    low, high = within or (-float("inf"), float("inf"))
    for q in points + [None]:
        if q is not None and q["stable"] is stability:
            run.append(q["p"])
        elif run:
            if max(run) >= low and min(run) <= high:
                out.append([max(min(run), low), min(max(run), high)])
            run = []
    return out


def _branch_in_order(points: list) -> list:
    """The cycle branch as one curve, stretch by stretch in AUTO's order and outside the requested
    range too: how its parts connect. A stretch starts at the last point of the one before."""
    out, start = [], 0
    for i in range(1, len(points) + 1):
        if i == len(points) or points[i]["stable"] is not points[start]["stable"]:
            first, last = points[max(start - 1, 0)], points[i - 1]
            out.append({"stability": {True: "stable", False: "unstable"}.get(points[start]["stable"], "unknown"),
                        "from": first["p"], "to": last["p"],
                        "period_from": first["period"], "period_to": last["period"]})
            start = i
    return out


def _cycles_by_stability(points: list, plot_state: str) -> dict:
    """Period and amplitude ranges of the stable, the unstable and the unknown-stability
    cycles, each on its own: one range over all of them would mix them."""
    out = {}
    for key, flag in (("stable", True), ("unstable", False), ("stability_unknown", None)):
        qs = [q for q in points if q["stable"] is flag]
        if not qs:
            continue
        periods = [q["period"] for q in qs if q["period"] is not None]
        out[key] = {"n_points": len(qs),
                    "period_range": [min(periods), max(periods)] if periods else None,
                    f"{plot_state}_range": [min(q["min"] for q in qs), max(q["max"] for q in qs)]}
    return out


def _l1_type(data: dict, hopf_p: float) -> str:
    """The Hopf type AUTO computed at the Hopf point at hopf_p: its first Lyapunov coefficient
    l1 (< 0 supercritical, > 0 subcritical), from the runner's short Hopf continuation."""
    near = [h for h in data.get("hopf_l1") or [] if h.get("l1") is not None
            and abs(h["parameter"] - hopf_p) <= 1e-4 * max(1.0, abs(hopf_p))]
    if not near:
        return "unknown: AUTO did not compute the first Lyapunov coefficient here"
    l1 = near[0]["l1"]
    # l1 describes only the two Hopf directions: another unstable pair leaves the cycles unstable
    floquet = "; whether the cycles are stable is in the Floquet results (periodic_orbits), not in l1"
    return (f"supercritical (AUTO first Lyapunov coefficient l1 = {l1:.4g} < 0){floquet}"
            if l1 < 0 else
            f"subcritical (AUTO first Lyapunov coefficient l1 = {l1:.4g} > 0){floquet}" if l1 > 0 else
            "degenerate (AUTO l1 = 0): a generalized Hopf (Bautin) point")


def _settle_cycle_entries(entries: list, param_name: str, tol: float) -> list:
    """The cycle entries as the diagram shows them. A failed run is dropped when another run from
    the same Hopf point followed its cycles, and turned into a pointer when a followed branch ends
    at its Hopf point (the same cycles, reached from the other end). Repeats are dropped."""
    followed = [e for e in entries if "error" not in e and "same_cycles_as" not in e
                and "from_period_doubling_at" not in e and e.get("cycles_over")]
    out, seen = [], set()
    for e in entries:
        key = json.dumps(e, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        hb = e.get("from_hopf_at")
        if "error" in e and "from_period_doubling_at" not in e and hb is not None:
            if any(abs(f["from_hopf_at"] - hb) <= tol for f in followed):
                continue        # another run from this Hopf point followed its cycles
            reach = next((f for f in followed if min(abs(x - hb) for x in f["cycles_over"]) <= tol), None)
            if reach is not None:
                end = min(reach["cycles_over"], key=lambda x: abs(x - hb))
                e = {"from_hopf_at": hb, "same_cycles_as": (
                    f"the branch from the Hopf point at {reach['from_hopf_at']:.6g}: it reaches "
                    f"{param_name} = {end:.6g}, next to this Hopf point, so these are the same "
                    "cycles. A separate run from this Hopf point stalled; that says nothing "
                    "against those cycles.")}
        if "same_cycles_as" in e and any("same_cycles_as" in o and o["from_hopf_at"] == hb for o in out):
            continue            # one pointer per Hopf point
        out.append(e)
    return out


def _folds_of_no_size(points: list) -> list:
    """Folds of cycles whose cycle has almost no amplitude (under 1% of the branch's largest).
    Such a cycle is the Hopf point itself, where the branch begins or ends."""
    biggest = max((q["max"] - q["min"] for q in points), default=0.0)
    return sorted({float(f"{q['p']:.5g}") for q in points
                   if q["type"] == "LP" and q["max"] - q["min"] < 0.01 * biggest})


def _auto_problems(data: dict) -> dict:
    """{"auto_problems": [...]} for the parts of the AUTO run that failed, or {}: a diagram
    with branches can still miss a direction or a branch switch, and that must be visible."""
    meaning = {"main-forward": "the forward continuation from the start failed",
               "main-backward": "the backward continuation from the start failed: only one "
                                "direction was computed"}
    diag = data.get("diagnostics") or {}
    out = []
    for r in diag.get("run_records") or []:
        if r.get("error"):
            tag = r.get("tag", "?")
            what = meaning.get(tag) or ("a branch switch at a branch point failed: that branch "
                                        "is missing" if tag.startswith("BP-") else
                                        "the cycles from this Hopf point were not computed"
                                        if tag.startswith("HB-") else
                                        "the cycles of twice the period from this period "
                                        "doubling were not computed" if tag.startswith("PD-") else "")
            out.append(f"{tag}: {what}{'; ' if what else ''}{str(r['error'])[:200]}")
    logged = " ".join(out)
    out += [str(e)[:200] for e in data.get("errors") or [] if str(e).split(":")[0] not in logged]
    if diag.get("hit_switch_cap"):
        out.append("the branch-switch limit was reached: further branches were not followed")
    return {"auto_problems": out} if out else {}


def _cycle_end(end, stop, diverged, last_p, requested, param_name) -> dict:
    """Why a cycle branch ends where it does: {"branch_end": cause} and, when the end is
    where the computation stopped rather than where the cycles end, an "incomplete" note.
    The cause is read from what the run did, not from AUTO's label alone: an EP can also
    mean that NMX ran out.
      hopf          the cycles shrank into a Hopf point          (the cycles end there)
      range_end     the branch left the requested range          (complete within it)
      zero_period   the period fell to 0: no real cycles past it
      diverged      later points were not finite numbers
      no_convergence AUTO's corrector failed (MX)
      step_limit    all NMX continuation steps were used
      unexplained   AUTO ended normally (EP) inside the range before NMX, giving no reason"""
    tol = 1e-3 * (requested[1] - requested[0])
    if stop == "hopf":
        return {"branch_end": "hopf"}
    if diverged:
        cause = "diverged"
    elif stop == "zero_period":
        cause = "zero_period"
    elif end and end.get("type") == "MX":
        cause = "no_convergence"
    elif end and end.get("point", 0) >= end.get("nmx", float("inf")):
        cause = "step_limit"
    elif not requested[0] + tol < last_p < requested[1] - tol:
        return {"branch_end": "range_end"}
    else:
        cause = "unexplained"
    why = {"diverged": f"the continuation broke down ({diverged} later points were not finite numbers)",
           "zero_period": "the period fell to zero, so there are no real cycles past this point",
           "no_convergence": "AUTO's corrector did not converge (MX)",
           "step_limit": f"AUTO used all {end['nmx'] if end else '?'} continuation steps (NMX)",
           "unexplained": "AUTO ended the branch (EP) before its step limit without saying why"}[cause]
    return {"branch_end": cause, "stopped_at": last_p, "incomplete": (
        f"This cycle branch stops at {param_name} = {last_p:.6g}, inside the range: {why}. "
        "Cycles may go on past it, with unknown size and stability; the end of cycles_over is "
        "where the computation stopped, not where the cycles or their stability end.")}


def _point_stability_signs(n_points: int, stability_segments: list[int]) -> np.ndarray:
    """Expand AUTO's compact stability() segment-boundary list into a
    per-point sign array (+1 stable, -1 unstable, 0 unknown: a point no segment
    covers, e.g. past the end of a truncated run)."""
    signs = np.zeros(n_points)
    start = 0
    for seg_end in stability_segments:
        stable = seg_end < 0
        end = abs(seg_end)
        signs[start:end] = 1 if stable else -1
        start = end
    return signs


def _format_auto_terminal(data: dict, model_label: str, param_name: str,
                          param_min: float, param_max: float) -> str:
    lines = [
        "",
        "=" * 72,
        "AUTO-07p CONTINUATION LOG",
        "=" * 72,
        f"Model              : {model_label}",
        f"Continuation       : {param_name}",
        f"Requested range    : [{param_min:g}, {param_max:g}]",
        "",
    ]

    diag = data.get("diagnostics") or {}
    lines += [
        f"Branches returned   : {diag.get('n_raw_branches_from_auto', '?')}",
        f"BP switch attempts  : {diag.get('n_bp_switch_attempts', '?')}",
        f"Switch cap reached  : {diag.get('hit_switch_cap', False)}",
        "",
    ]

    points = diag.get("terminal_special_points") or []
    if points:
        lines.append("Special points reported by AUTO:")
        for sp in points:
            state_txt = " ".join(
                f"{k}={v:.6g}" for k, v in sp.get("states", {}).items()
            )
            lines.append(
                f"  {sp['type']:>2}  LAB={sp['label']:<5d}  "
                f"{param_name}={sp['parameter']:.8g}"
                + (f"  {state_txt}" if state_txt else "")
            )
        lines.append("")

    records = diag.get("run_records") or []
    lines.append("Continuation runs:")
    for rec in records:
        err = f"  ERROR={rec['error']}" if rec.get("error") else ""
        lines.append(
            f"  {rec.get('tag', 'run'):<24} "
            f"branches={rec.get('n_branches', 0):<2} "
            f"points={rec.get('branch_sizes', [])}{err}"
        )
    lines.append("")

    raw_chunks = []
    for rec in records:
        raw = (rec.get("stdout") or "").strip()
        if raw:
            raw_chunks.append(f"[{rec['tag']}]")
            raw_chunks.extend(raw.splitlines())

    if raw_chunks:
        max_lines = 180
        if len(raw_chunks) > max_lines:
            omitted = len(raw_chunks) - max_lines
            raw_chunks = [f"... ({omitted} earlier lines omitted) ..."] + raw_chunks[-max_lines:]
        lines.extend(raw_chunks)
    else:
        lines.append("[AUTO produced no stdout transcript]")

    lines += ["", "=" * 72, "END AUTO-07p LOG", "=" * 72, ""]
    return "\n".join(lines)



def _as_float(value):
    """A value as a float, or None when it is not a number (e.g. '*2' before it is applied)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _monotone_pieces(xs):
    """Index arrays of the stretches of one branch between its turning points in the
    parameter (a fold turns the branch back): each stretch is single-valued in the parameter."""
    d = np.sign(np.diff(xs))
    for k in range(1, d.size):          # a flat step keeps the direction before it
        if d[k] == 0:
            d[k] = d[k - 1]
    turns = [k + 1 for k in range(d.size - 1) if d[k] and d[k + 1] and d[k] != d[k + 1]]
    bounds = [0] + turns + [xs.size - 1]
    return [np.arange(a, b + 1) for a, b in zip(bounds, bounds[1:])]


def _separating_species(collected, names, requested):
    """Where two or more branch stretches exist at the same parameter values (both sides of a
    fold, the branches through a branch point), the species that keeps them furthest apart: a
    species with equal values on two branches draws one over the other and hides a fold or a
    branch. Score of a species: the smallest median distance (relative to the species' scale)
    over all pairs of overlapping stretches. None when no two stretches overlap, or when every
    species leaves some pair on top of each other."""
    pieces = []
    for c in collected:
        xs = np.asarray(c["arr"][c["p_idx"]], dtype=float)
        for idx in _monotone_pieces(xs):
            px = xs[idx]
            keep = (px >= requested[0]) & (px <= requested[1]) & np.isfinite(px)
            if keep.sum() < 3:
                continue
            order = np.argsort(px[keep])
            ys = {n: np.asarray(c["arr"][c["coordnames"].index(n)], dtype=float)[idx][keep][order]
                  for n in names if n in c["coordnames"]}
            pieces.append((px[keep][order], {n: y for n, y in ys.items() if np.all(np.isfinite(y))}))
    scale = {}
    for _, ys in pieces:
        for n, y in ys.items():
            scale[n] = max(scale.get(n, 0.0), float(np.max(np.abs(y))))
    width = requested[1] - requested[0]
    score = {}
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            (xa, ya), (xb, yb) = pieces[i], pieces[j]
            lo, hi = max(xa[0], xb[0]), min(xa[-1], xb[-1])
            if hi - lo <= 1e-6 * width:
                continue
            grid = np.linspace(lo, hi, 50)
            seps = {n: float(np.median(np.abs(np.interp(grid, xa, ya[n]) - np.interp(grid, xb, yb[n]))))
                    / (scale[n] + 1e-300) for n in ya if n in yb}
            if not seps or max(seps.values()) < 1e-6:   # the same curve followed twice
                continue
            for n, s in seps.items():
                score[n] = min(score.get(n, np.inf), s)
    if not score:
        return None
    best = max(score, key=score.get)
    return best if score[best] > 1e-3 else None


_OTHER_STATE_PROBES = (0.25, 0.5, 0.75, 1.0)   # where in the range other states are looked for
_LOG_PROBES = 5                               # log-spaced values per sign of the range (diagrams)
_MAX_OTHER_STARTS = 4                         # continuations from states off the followed branches


def _probe_values(lo, hi, own=None, log_spaced=False) -> list:
    """Where steady states off the followed branches are looked for. A diagram (or an LHS scan
    with log_probes): the model's own value plus log-spaced values across the range (each sign
    apart; a range from 0 starts at 1e-3 of its end), so a state that exists only in one decade
    is not missed. The LHS scans by default: four evenly spaced values."""
    if not log_spaced:
        return [lo + f * (hi - lo) for f in _OTHER_STATE_PROBES]
    pts = [own] if own is not None and lo <= own <= hi else []
    if hi > 0:
        pts += list(np.geomspace(lo if lo > 0 else hi * 1e-3, hi, _LOG_PROBES))
    if lo < 0:
        pts += list(-np.geomspace(-hi if hi < 0 else -lo * 1e-3, -lo, _LOG_PROBES))
    return [float(v) for i, v in enumerate(pts) if not any(np.isclose(v, w) for w in pts[:i])]


def _other_state_probes(requested, own, log_spaced, log_of) -> list:
    """_probe_values over the requested range; a range in log10 of a parameter is probed at the
    log10 of the probes of the actual range."""
    if not log_of:
        return _probe_values(*requested, own, log_spaced)
    return [float(np.log10(v)) for v in _probe_values(10 ** requested[0], 10 ** requested[1],
                                                      None if own is None else 10 ** own, True)]


def _on_branches(branches, names_all, param_name, state_names, value, y, rel_tol=1e-2) -> bool:
    """Does the state y at param_name = value lie on one of these AUTO branches (linear
    interpolation between the branch points around value, every state within rel_tol)?"""
    y = np.asarray(y, dtype=float)
    for br in branches:
        names = _full_coordnames(br["coordnames"], names_all)
        arr = np.array(br["data"])
        if (param_name not in names or arr.ndim != 2 or arr.shape[1] < 2
                or any(s not in names for s in state_names)):
            continue
        p = arr[names.index(param_name)]
        rows = arr[[names.index(s) for s in state_names]]
        for i in np.flatnonzero((p[:-1] - value) * (p[1:] - value) <= 0):
            w = 0.0 if p[i + 1] == p[i] else (value - p[i]) / (p[i + 1] - p[i])
            on = rows[:, i] + w * (rows[:, i + 1] - rows[:, i])
            if np.linalg.norm(on - y) <= rel_tol * (1 + np.linalg.norm(y)):
                return True
    return False


def _continue_from_other_states(data, model, param_name, param_values, requested, window,
                                model_args, fixed_params, label, n_pts, run_overrides,
                                runner_cfg, timeout, log_spaced=False, log_of=None) -> list:
    """Steady states the continuation from the start never reached: find_steady_states at a few
    values of the parameter inside the requested range (_probe_values); every state that lies
    on none of the branches followed so far is continued from (both directions, over the same
    run window), and its branches are added to data["branches"]. Returns where each extra
    continuation started. log_of: param_name is log10 of this parameter of the model the
    search reads (the probes are log-spaced in it)."""
    from dynalyst.analysis.steady_states import find_steady_states
    search = getattr(find_steady_states, "func", find_steady_states)
    state_names = model["state_names"]
    names_all = [param_name, *model["param_names"], *state_names]
    starts = []
    for value in _other_state_probes(requested, param_values.get(param_name), log_spaced, log_of):
        if len(starts) >= _MAX_OTHER_STARTS:    # no more continuations: no more searches
            break
        try:
            given = ({**fixed_params, param_name: value} if not log_of else
                     {**{k: v for k, v in fixed_params.items() if k != param_name}, log_of: 10 ** value})
            found = json.loads(search(**model_args, fixed_params=given))
        except Exception:
            continue
        for s in found.get("steady_states") or []:
            state = s.get("state") or {}
            if len(starts) >= _MAX_OTHER_STARTS or any(n not in state for n in state_names):
                continue
            y = np.array([float(state[n]) for n in state_names])
            if not np.all(np.isfinite(y)) or _on_branches(data["branches"], names_all, param_name,
                                                          state_names, value, y):
                continue
            if log_spaced and across_a_pole(model, y, {**param_values, param_name: value}):
                continue
            work_dir = Path(tempfile.mkdtemp(prefix="auto_", dir=str(config.AUTO_WORK_DIR)))
            try:
                pv = {**param_values, param_name: value}
                analytic = _write_equations_file(model, param_name, y, pv, work_dir / f"{label}.f90")
                used = _write_constants_file(model, param_name, window[0], window[1],
                                             work_dir / f"c.{label}", n_pts, run_overrides,
                                             analytic_jacobian=analytic)
                _write_lf(work_dir / "runner_config.json", json.dumps({
                    **runner_cfg, "nmx": int(used["nmx"]), "dsmax": float(used["dsmax"])}))
                more = _run_auto(work_dir, label, timeout_override=timeout)
            except Exception:
                continue
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
            if more.get("branches"):
                data["branches"] += more["branches"]
                starts.append({param_name: value, "stability": s.get("stability"),
                               "state": {n: float(v) for n, v in zip(state_names, y)}})
    return starts


def _run_single_bifurcation_analysis(model_name_or_path: str | None = None,
                              species: list[str] | None = None,
                              odes: dict | None = None,
                              parameters: dict | None = None,
                              initial_conditions: dict | None = None,
                              param_name: str | None = None,
                              was_auto_selected: bool = False,
                              param_min: float | None = None,
                              param_max: float | None = None,
                              fixed_params: dict | None = None,
                              plot_state: str | None = None,
                              n_pts: int = 500,
                              ds: float | None = None,
                              dsmin: float | None = None,
                              dsmax: float | None = None,
                              nmx: int | None = None,
                              itmx: int | None = None,
                              itnw: int | None = None,
                              nwtn: int | None = None,
                              epsl: float | None = None,
                              epsu: float | None = None,
                              epss: float | None = None,
                              show_auto_log: bool = True,
                              auto_timeout_seconds: int | None = None,
                              xpp_source_path: Path | None = None,
                              start_guess: dict | None = None,
                              probe_only: bool = False,
                              cycles: bool = True,
                              cycle_folds: bool = False,
                              find_other_branches: bool = False,
                              log_probes: bool | None = None,
                              log_axis: bool = False) -> str:
    """
    Tool: run a real numerical bifurcation analysis (AUTO-07p pseudo-
    arclength continuation of steady states) for a model of ANY dimension.

    This tool is SELF-CONTAINED: callers may provide only an SBML model/path
    (or a directly-specified ODE system). It will internally choose a sweep
    parameter/range when they were not supplied, apply conservation reduction,
    find a valid starting equilibrium, generate AUTO equations/Jacobians, run
    continuation, switch branches at BPs, and make the plot. Upstream calls to
    get_odes or find_steady_states are useful for inspection but are NOT required.

    For models associated with an XPP .ode/.txt source, the original source is
    explicitly carried through conversion and used for XPP AUTO metadata when available (e.g. @ autovar, @ parmin, @ parmax).
    The equations being continued still come from the resolved model itself.

    Detects and classifies LP (saddle-node/fold), BP (transcritical/pitchfork),
    and HB (Hopf) from actual AUTO continuation. fixed_params overrides any
    OTHER parameter's default value. plot_state selects the y-axis variable;
    when omitted, XPP @ autovar wins, then XPP @ yp1/yp2 are fallbacks, then the species that
    changes most over the range.

    AUTO's continuation constants (DS, DSMIN, DSMAX, NMX, ITMX, ITNW, NWTN,
    EPSL, EPSU, EPSS) are computed automatically from param_min/param_max
    and n_pts, the same way XPPAUT's AUTO Numerics panel seeds sensible
    defaults from your parameter range. Leave any of the ds/dsmin/dsmax/
    nmx/itmx/itnw/nwtn/epsl/epsu/epss arguments as None to keep the
    computed default, or pass a specific value to pin just that one --
    e.g. if a stiff crossing keeps terminating early (AUTO's "MX" abnormal
    termination), try a smaller dsmin and/or a larger nmx.

    show_auto_log=True includes AUTO's raw continuation log (the BR/PT/TY/
    LAB/L2-NORM/U(...) table it prints as it runs, same as XPPAUT's AUTO
    console) in the result as "auto_stdout_log". It is enabled by default;
    pass show_auto_log=False when a shorter result is preferred.

    probe_only=True runs AUTO without cycles or a plot and returns the special points and
    the branches: the parameter search and the LHS scans use it.

    find_other_branches=True also looks for steady states off the branches followed from the
    start (find_steady_states at a few parameter values in the range) and continues from each
    one found, so a branch not connected to the start (a fold whose branch point has unfolded,
    in a sampled parameter set) is part of the result. Off by default: it costs extra runs.

    log_axis=True (a positive range): AUTO continues log10_<param> (the model with the
    parameter written as 10**log10_<param>), the x axis is logarithmic, and every parameter
    value in the result is log10 (the wrapper adds the actual values, _add_actual_values).
    """
    # A bifurcation request may target either SBML or an uploaded XPP text file.
    # For XPP input, convert THAT EXACT FILE here so model identity is carried
    # through the whole operation instead of relying on a second, fuzzy lookup.
    if model_name_or_path and Path(model_name_or_path).suffix.lower() in {".txt", ".ode"}:
        try:
            source = _find_exact_input_path(model_name_or_path)
            from dynalyst.analysis.xpp_to_sbml import parse_xpp
            from dynalyst.analysis.jaxkm_ode_to_sbml import odes_to_sbml
            text = source.read_text(errors="replace")
            xs, xo, xp, xic = parse_xpp(text)
            if not xs:
                raise ValueError(f"No XPP derivative equations found in '{source.name}'.")
            converted = json.loads(odes_to_sbml(
                species=xs, odes=xo, parameters=xp,
                initial_conditions=xic, model_name=source.stem,
            ))
            if "error" in converted:
                raise ValueError(converted["error"])
            model_name_or_path = converted["path"]
        except Exception as e:
            return json.dumps({"error": f"Could not prepare the exact XPP model for bifurcation: {e}"})

    # model preparation (pools, signs, initial state, reduction): dynalyst/analysis/model_prep.py
    from dynalyst.analysis.model_prep import prepare
    try:
        prepared = prepare(model_name_or_path, species, odes, parameters, initial_conditions,
                           fixed_params, start_guess)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    log_of = None       # the actual parameter, when AUTO continues its log10
    if log_axis:
        if not (param_name and param_min is not None and param_max is not None and 0 < param_min < param_max):
            return json.dumps({"error": "Internal error: log_axis needs param_name and a positive range."})
        log_of = canonical_param(prepared["full"], param_name) or param_name
        q = f"log10_{log_of}"
        fixed_params = dict(fixed_params or {})
        value = float(fixed_params.pop(log_of, prepared["full"]["param_defaults"][log_of]))
        # a value of 0 (or one outside the window) cannot be a log start: start at the window's end
        q_value = (float(np.log10(value)) if param_min <= value <= param_max
                   else float(np.log10(param_min if value < param_min else param_max)))
        prepared = {**prepared, "full": _log_parameter(prepared["full"], log_of, q, q_value),
                    "reduced": _log_parameter(prepared["reduced"], log_of, q, q_value)}
        param_name, param_min, param_max = q, float(np.log10(param_min)), float(np.log10(param_max))
    model, totals_kept = prepared["full"], prepared["totals_kept"]

    # Preserve the ORIGINAL XPP source across the TXT/ODE -> SBML conversion.
    # The wrapper may pass an XML path to the worker, so rediscovering the XPP
    # file from model_name_or_path alone is not reliable.
    xpp_source = xpp_source_path or _find_xpp_source(model_name_or_path)
    # Direct worker calls with an XPP path may still create `source` locally.
    if not xpp_source and 'source' in locals():
        xpp_source = source
    xpp_meta = _parse_xpp_metadata(xpp_source)

    # _run_single_bifurcation_analysis is an internal worker. Its caller has
    # already selected the continuation parameter, so do not run a second
    # parameter-selection path here. Keeping one authoritative selection path
    # prevents stale/default choices from leaking into individual trials.
    if not param_name:
        return json.dumps({"error": "Internal error: continuation parameter was not supplied to the AUTO worker."})
    problem = continuation_parameter_error(model, param_name)
    if problem:
        return json.dumps(problem)
    param_name = canonical_param(model, param_name)
    try:
        param_min, param_max, range_selection_reason = _infer_parameter_range(
            model, param_name, param_min, param_max, xpp_meta
        )
        requested_plot_state, plot_selection_reason = _choose_plot_state(model, plot_state, xpp_meta)
        plot_state = requested_plot_state
    except ValueError as e:
        return json.dumps({"error": str(e)})
    parameter_selection_reason = (
        "selected automatically from the supplied model using its XPP/AUTO hint or declared parameter order"
        if was_auto_selected else
        "explicitly supplied by the user"
    )

    fixed_params = fixed_params or {}
    unknown = [k for k in fixed_params if k not in model["param_defaults"]]
    if unknown:
        return json.dumps({"error": f"Unknown parameter(s) {unknown}.",
                            "available_parameters": list(model["param_defaults"].keys())})

    state_names = model["state_names"]
    if plot_state is not None and plot_state not in state_names:
        return json.dumps({"error": f"'{plot_state}' is not a state variable.", "state_variables": state_names})

    full_model = model
    model = prepared["reduced"]
    reduced_state_names = model["state_names"]
    if model.get("eliminated"):
        eliminated_names = [name for name, _ in model["eliminated"]]
    else:
        eliminated_names = []

    # XPP-AUTO starts continuation from the equilibrium at the model's current
    # parameter value (the value in the uploaded model), not from the lower
    # sweep boundary. Starting at param_min can land on a different equilibrium
    # branch and is a major source of diagrams that disagree with XPP-AUTO.
    # a value given for the continuation parameter itself (fixed_params) is where it starts:
    # e.g. inside a window where another branch exists than at the model's own value
    param_start = float(fixed_params.get(param_name, model["param_defaults"][param_name]))
    requested = (param_min, param_max)
    extended = None
    if not (param_min <= param_start <= param_max):
        # As XPP-AUTO would: start at the model's own value and continue into the requested
        # range; only the requested range is reported and drawn. The run window has a margin
        # of its own width on each side: a branch may have to turn at a fold outside the
        # requested range before it comes back into it. A window of non-negative values stays
        # non-negative.
        lo, hi = min(param_min, param_start), max(param_max, param_start)
        width = hi - lo
        extended = (max(lo - width, 0.0) if lo >= 0 else lo - width, hi + width)
        param_min, param_max = extended

    try:
        y_start, param_values = _find_start_state(model, param_name, param_start, fixed_params,
                                                  prepared["nonneg"])
        # AUTO cannot flag a Hopf point it starts on (no sign change before its first step):
        # start 1% of the window away, inside the range, and say so
        start_moved = None
        _, jac_fn = _lambdify_model(model)
        from dynalyst.analysis.auto_diagnostics import hopf_pair
        with np.errstate(all="ignore"):
            J = np.array(jac_fn(*y_start, *[param_values[p] for p in model["param_names"]]), dtype=float)
        if np.all(np.isfinite(J)) and hopf_pair(np.linalg.eigvals(J)):
            step = 0.01 * (param_max - param_min)
            moved = param_start + step if param_start + step <= param_max else param_start - step
            y_start, param_values = _find_start_state(model, param_name, moved,
                                                      {**fixed_params, param_name: moved},
                                                      prepared["nonneg"])
            start_moved = (param_start, moved)
            param_start = moved
    except ValueError as e:
        return json.dumps({"error": str(e)})

    config.AUTO_WORK_DIR.mkdir(parents=True, exist_ok=True)
    label = "model"
    work_dirs = []

    def covers_requested(data) -> bool:
        """Did some branch reach both ends of the requested range?"""
        names_all = [param_name, *model["param_names"], *reduced_state_names]
        lo_ok = hi_ok = False
        tol = max(abs(requested[1] - requested[0]), 1e-6) * 1e-3
        for br in data.get("branches") or []:
            names = _full_coordnames(br["coordnames"], names_all)
            arr = np.array(br["data"])
            if param_name not in names or arr.ndim != 2 or not arr.shape[1]:
                continue
            p = arr[names.index(param_name)]
            lo_ok |= bool(np.min(p) - requested[0] < tol)
            hi_ok |= bool(requested[1] - np.max(p) < tol)
        return lo_ok and hi_ok

    def unstable_at_edge(data) -> set:
        """The ends of the requested range ("lo", "hi") that a branch reaches while unstable.
        An unstable equilibrium goes on past that end, so the fold (or other point) that ends
        it lies beyond: a diagram cut there would miss it."""
        names_all = [param_name, *model["param_names"], *reduced_state_names]
        tol = max(abs(requested[1] - requested[0]), 1e-6) * 1e-3
        sides = set()
        for br in data.get("branches") or []:
            names = _full_coordnames(br["coordnames"], names_all)
            arr = np.array(br["data"])
            if param_name not in names or arr.ndim != 2 or not arr.shape[1]:
                continue
            p = arr[names.index(param_name)]
            unstable = np.asarray(_point_stability_signs(arr.shape[1], br["stability_segments"])) < 0
            if np.any(unstable & (np.abs(p - requested[0]) < tol)):
                sides.add("lo")
            if np.any(unstable & (np.abs(p - requested[1]) < tol)):
                sides.add("hi")
        return sides

    def steps_ran_out(data, limit) -> bool:
        return any(np.array(br["data"]).ndim == 2 and np.array(br["data"]).shape[1] >= 0.99 * limit
                   for br in data.get("branches") or [])

    try:
        overrides = {k: v for k, v in dict(ds=ds, dsmin=dsmin, dsmax=dsmax, nmx=nmx, itmx=itmx,
                                            itnw=itnw, nwtn=nwtn, epsl=epsl, epsu=epsu,
                                            epss=epss).items() if v is not None}
        # A continuation that does not reach both ends of the requested range is run once
        # more over a wider window (and with more steps): a branch inside the range may only
        # be reachable by turning at a fold outside it. Only the requested range is reported.
        # The window stays non-negative for SBML parameters (rates, amounts), not for typed
        # equations, whose parameters may take any sign.
        from dynalyst.analysis.typed_equations import is_typed_file
        any_sign = is_typed_file(model.get("source_path") or "")
        base_nmx = None
        chase_edge = set()     # ends of the range a branch left while unstable
        for attempt in range(2):
            work_dir = Path(tempfile.mkdtemp(prefix="auto_", dir=str(config.AUTO_WORK_DIR)))
            work_dirs.append(work_dir)
            analytic_jac = _write_equations_file(model, param_name, y_start, param_values,
                                                 work_dir / f"{label}.f90")
            run_overrides = dict(overrides)
            if attempt:
                width = param_max - param_min
                low = param_min - width
                param_min = low if any_sign or param_min < 0 else max(low, 0.0)
                param_max = param_max + width
                extended = (param_min, param_max)
                if nmx is None:
                    run_overrides["nmx"] = base_nmx * 4
                if dsmax is None and steps_ran_out(data, base_nmx):
                    # a step counts the parameter and the states together: span/n_pts crawls
                    # when the states are far larger than the range of a small parameter
                    run_overrides["dsmax"] = max(param_max - param_min, float(np.max(np.abs(y_start)))) / n_pts
            constants_used = _write_constants_file(model, param_name, param_min, param_max,
                                                    work_dir / f"c.{label}", n_pts, run_overrides,
                                                    analytic_jacobian=analytic_jac)
            base_nmx = base_nmx or int(constants_used["nmx"])
            runner_cfg = {
                "n_pts": int(n_pts),
                # cycles from Hopf points: for the diagram, not for a window probe
                "periodic": (not probe_only) and cycles,
                "icp": auto_par_index(model["param_names"].index(param_name)),
                # any other parameters: the Hopf run only needs a second free one
                # (a one-parameter model: the next free slot, which no equation uses)
                "second_parameters": ([auto_par_index(i) for i, q in enumerate(model["param_names"])
                                       if q != param_name][:3]
                                      or [auto_par_index(len(model["param_names"]))])
                                     if not probe_only else [],
            }
            _write_lf(work_dir / "runner_config.json", json.dumps({
                **runner_cfg, "nmx": int(constants_used["nmx"]),
                "dsmax": float(constants_used["dsmax"])}))
            data = _run_auto(work_dir, label, timeout_override=auto_timeout_seconds)
            # A range covered with no fold can still hide one: on an S-curve the branch from the
            # start runs past the range to its fold, and the part inside lies beyond that turn.
            # Runs with cycles are not repeated for this: they are slow.
            # A branch cut at an end while unstable is followed further even with cycles (slow,
            # but the diagram must show the point that ends it).
            cut_unstable = unstable_at_edge(data) if not attempt else set()
            chase_edge |= cut_unstable
            look_further = not attempt and (
                ((probe_only or not cycles)
                 and not any(sp.get("type") == "LP" for sp in data.get("special_points") or []))
                or cut_unstable)
            if not data["branches"] or (covers_requested(data) and not look_further):
                break
    except Exception as e:
        # a missing program means AUTO cannot run here; a missing file of the run itself
        # (AUTO's fort.* in its work folder) is a failure of this run, not of the machine
        own_file = isinstance(e, FileNotFoundError) and any(
            str(e.filename or "").startswith(str(wd)) for wd in work_dirs)
        reason = None if own_file else build_failure(str(e))
        if reason or (isinstance(e, FileNotFoundError) and not own_file):
            return json.dumps({"error": unavailable_message(reason or str(e)),
                               "auto_unavailable": True})
        return json.dumps({"error": f"AUTO-07p run failed: {e}"})
    finally:
        for wd in work_dirs:
            shutil.rmtree(wd, ignore_errors=True)

    # a branch the range cut while unstable ends at a point past the range: every point the
    # wider run found is shown in the one diagram, the drawn and reported range grows to
    # include them (with a margin). A wider run made only to reach the range from the model's
    # own value (outside it) does not widen what is shown.
    widened_from = None
    if extended and chase_edge:
        beyond = []
        names_all = [param_name, *model["param_names"], *reduced_state_names]
        for br in data.get("branches") or []:
            names = _full_coordnames(br["coordnames"], names_all)
            arr = np.array(br["data"])
            if param_name not in names or arr.ndim != 2 or not arr.shape[1]:
                continue
            p = arr[names.index(param_name)]
            beyond += [float(p[sp["index"]]) for sp in br.get("special_points") or []
                       if sp.get("type") in ("LP", "BP", "HB") and 0 <= sp.get("index", -1) < p.size
                       # only past an end that a branch left while unstable
                       and ((float(p[sp["index"]]) < requested[0] and "lo" in chase_edge)
                            or (float(p[sp["index"]]) > requested[1] and "hi" in chase_edge))
                       and extended[0] <= float(p[sp["index"]]) <= extended[1]]
        if beyond:
            lo, hi = min(requested[0], *beyond), max(requested[1], *beyond)
            pad = 0.1 * (hi - lo)
            widened_from = requested
            requested = (max(extended[0], lo - pad), min(extended[1], hi + pad))

    if not data["branches"]:
        reason = build_failure("\n".join([
            data.get("auto_stdout_log") or "", data.get("auto_stderr_log") or "",
            json.dumps(data.get("errors"), default=str),
            json.dumps((data.get("diagnostics") or {}).get("run_records"), default=str)]))
        if reason:   # the model could not even be compiled: a setup problem
            return json.dumps({"error": unavailable_message(reason), "auto_unavailable": True})
        return json.dumps({
            "error": "AUTO-07p produced no branches.",
            "details": data.get("errors"),
            "diagnostics": {
                k: v for k, v in (data.get("diagnostics") or {}).items()
                if k != "run_records"
            },
            "run_errors": [
                {"tag": r.get("tag"), "error": r.get("error")}
                for r in ((data.get("diagnostics") or {}).get("run_records") or [])
                if r.get("error")
            ],
            "auto_stdout_tail": (data.get("auto_stdout_log") or "")[-3000:],
            "auto_stderr_tail": (data.get("auto_stderr_log") or "")[-2000:],
        })

    other_starts = []    # steady states off the followed branches that were continued from
    # diagrams probe log-spaced; LHS samples (probe_only) evenly, unless log_probes is asked for
    log_spaced = (not probe_only) if log_probes is None else log_probes
    if find_other_branches:
        other_starts = _continue_from_other_states(
            data, model, param_name, param_values, requested, (param_min, param_max),
            dict(model_name_or_path=model_name_or_path, species=species, odes=odes,
                 parameters=parameters, initial_conditions=initial_conditions),
            fixed_params, label, n_pts, run_overrides, runner_cfg, auto_timeout_seconds,
            log_spaced=log_spaced, log_of=log_of)

    param_names = model["param_names"]

    # Collect branch data first (without plotting) so we can auto-pick the
    # most informative state variable by variance, when the caller didn't
    # specify one -- some state variables are invariant under a given sweep
    # (e.g. the Brusselator's x* doesn't depend on b) and would produce a
    # flat, uninformative plot if chosen blindly.
    collected = []
    abnormal_terminations = 0
    reached_param_min = False
    reached_param_max = False
    # coverage is judged on the range that was asked for (a run window widened around the
    # model's own value is wider than that)
    cover_lo, cover_hi = requested
    edge_tol = max(abs(cover_hi - cover_lo), 1e-6) * 1e-3
    full_names = [param_name, *param_names, *reduced_state_names]
    for br in data["branches"]:
        coordnames = br["coordnames"] = _full_coordnames(br["coordnames"], full_names)
        if param_name not in coordnames:
            continue
        arr = np.array(br["data"])
        p_idx = coordnames.index(param_name)
        signs = _point_stability_signs(arr.shape[1], br["stability_segments"])
        collected.append({"coordnames": coordnames, "arr": arr, "p_idx": p_idx,
                           "signs": signs, "special_points": br["special_points"]})
        # AUTO labels a branch "MX" ("abnormal termination") when it stops
        # because NMX (max steps) was hit or the corrector gave up -- NOT
        # because it reached param_min/param_max. If that happens, the
        # branch may be truncated before ever reaching a real fold/branch
        # point further along, and the resulting plot can look deceptively
        # smooth/complete. Track it so we can warn instead of staying silent.
        abnormal_terminations += sum(1 for sp in br["special_points"] if sp["type"] == "MX")
        if arr.shape[1]:
            pvals = arr[p_idx]
            reached_param_min = reached_param_min or bool(np.min(pvals) - cover_lo < edge_tol)
            reached_param_max = reached_param_max or bool(cover_hi - np.max(pvals) < edge_tol)

    if not collected:
        return json.dumps({"error": f"AUTO-07p produced no branch containing '{param_name}'."})

    if plot_selection_reason.startswith("first state variable"):
        # no species named: the one that changes most (relative) over the requested range,
        # so the diagram shows the response rather than a state the parameter barely moves
        spread = {}
        for c in collected:
            xs = np.asarray(c["arr"][c["p_idx"]], dtype=float)
            inside = (xs >= requested[0]) & (xs <= requested[1])
            if not np.any(inside):
                continue
            for s_name in reduced_state_names:
                if s_name not in c["coordnames"]:
                    continue
                ys = np.asarray(c["arr"][c["coordnames"].index(s_name)], dtype=float)[inside]
                if ys.size and np.all(np.isfinite(ys)):
                    rel = float(np.ptp(ys) / (np.mean(np.abs(ys)) + 1e-300))
                    spread[s_name] = max(spread.get(s_name, 0.0), rel)
        apart = _separating_species(collected, reduced_state_names, requested)
        if apart:
            plot_state = apart
            plot_selection_reason = ("the species that keeps the coexisting branches furthest "
                                     "apart (no species was named)")
        elif spread and max(spread.values()) > 0:
            plot_state = max(spread, key=spread.get)
            plot_selection_reason = ("the species that changes most over the range "
                                     "(no species was named)")

    range_warning = None
    if abnormal_terminations or not (reached_param_min and reached_param_max):
        range_warning = (
            "The continuation did not cleanly cover the full requested "
            f"[{cover_lo:g}, {cover_hi:g}] range for '{param_name}' "
            f"(reached down to {'param_min' if reached_param_min else 'somewhere short of param_min'}, "
            f"up to {'param_max' if reached_param_max else 'somewhere short of param_max'}"
            + (f"; {abnormal_terminations} branch(es) stopped abnormally (AUTO 'MX' -- "
               f"hit the step limit or the corrector failed) rather than reaching a "
               f"boundary or a genuine end point" if abnormal_terminations else "")
            + "). Any bifurcation past the point where continuation stopped would be "
              "missed. Re-run the same range with more continuation steps (nmx) or a "
              "smaller step size (dsmax); the range itself is the user's choice."
        )

    # If the requested/auto-picked plot_state was eliminated via a
    # conservation law, it isn't one of AUTO's own coordinates -- reconstruct
    # its value at every continuation point from the ones that ARE.
    plot_state_elim_expr = None
    if plot_state in eliminated_names:
        plot_state_elim_expr = dict(model["eliminated"])[plot_state]
        elim_fn = sp.lambdify(model["state_symbols"] + model["param_symbols"], plot_state_elim_expr, "numpy")

    # Every continuation point counts, whatever the sign of its states: negative values can be
    # meaningful, so no branch is labelled or left out for being negative (the user's decision).
    kept = set()

    def physical(coordnames, arr, xs, n):
        return np.ones(n, dtype=bool)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    all_special = []
    non_physical_points = []
    physical_ys = []
    any_non_physical = False
    plotted_anything = False
    legend_seen = set()
    curves = []          # (xs, ys, stability signs) per branch, for the classification
    state_curves = []    # (xs, all states per point, signs): coexistence counted on whole states
    neutral_saddles = []  # points AUTO labelled HB whose eigenvalues show no Hopf pair
    stepped_over = []     # where a branch changes stability with no special point: one was missed
    try:
        _, jac_red = _lambdify_model(model)
    except Exception:
        jac_red = None

    def eigs_at(state, p):
        """Jacobian eigenvalues of the continued (reduced) model at a state, or None."""
        try:
            J = np.array(jac_red(*[state[s] for s in model["state_names"]],
                                 *[p if q == param_name else param_values[q] for q in model["param_names"]]),
                         dtype=float)
            return np.linalg.eigvals(J) if np.all(np.isfinite(J)) else None
        except Exception:
            return None

    def plot_values(coordnames, arr, xs):
        """The plotted state along a branch (rebuilt when it was eliminated), or None."""
        if plot_state_elim_expr is not None:
            reduced_rows = [arr[coordnames.index(s)] for s in reduced_state_names if s in coordnames]
            if len(reduced_rows) != len(reduced_state_names):
                return None
            pvals = [xs if p == param_name else param_values[p] for p in model['param_names']]
            return np.broadcast_to(np.asarray(elim_fn(*reduced_rows, *pvals), dtype=float), xs.shape)
        return arr[coordnames.index(plot_state)] if plot_state in coordnames else None

    if probe_only:
        probe_points, probe_branches = [], []
        for c in collected:
            coordnames, arr = c["coordnames"], c["arr"]
            xs, ys = arr[c["p_idx"]], plot_values(coordnames, arr, arr[c["p_idx"]])
            n_points = arr.shape[1]
            for sp_pt in c["special_points"]:
                if sp_pt["type"] in ("LP", "BP", "HB") and sp_pt["index"] < n_points:
                    probe_points.append([sp_pt["type"], float(xs[sp_pt["index"]])])
            if ys is None:
                continue
            n = min(xs.shape[0], c["signs"].shape[0])
            phys = physical(coordnames, arr, xs, n)
            points = []
            for sp_pt in c["special_points"]:
                idx, ty = sp_pt["index"], sp_pt["type"]
                if ty in ("LP", "BP", "HB") and idx < n and phys[idx] and (ty != "HB" or _is_hopf(eigs_at(
                        {s: float(arr[coordnames.index(s)][idx]) for s in reduced_state_names
                         if s in coordnames}, float(xs[idx])))):
                    points.append({"type": ty, "index": idx})
            probe_branches.append({"parameter": xs[:n].tolist(), "state": ys[:n].tolist(),
                                   "stable": ((c["signs"][:n] > 0) & phys).tolist(),
                                   "physical": np.asarray(phys[:n], dtype=bool).tolist(), "points": points})
        return json.dumps({
            "probe": True,
            "swept_range": list(requested),
            "special_points": probe_points,
            "branches": probe_branches,
            "abnormal_terminations": abnormal_terminations,
            "range_covered": bool(reached_param_min and reached_param_max),
            **({"other_starts": other_starts} if other_starts else {}),
        })

    # the numbers the plot is drawn from, for the result: every stretch of one stability on
    # each equilibrium branch, and every computed point (equilibria and cycles) for the CSV
    branch_segments, branch_rows, cycle_rows = [], [], []

    for branch_no, c in enumerate(collected, 1):
        coordnames, arr, p_idx, signs = c["coordnames"], c["arr"], c["p_idx"], c["signs"]
        xs = arr[p_idx]
        ys = plot_values(coordnames, arr, xs)
        if ys is None:
            continue
        # AUTO may report a branch and stability segments with different
        # lengths after an abnormal/truncated run. Never index past the
        # shorter representation while still retaining the available curve.
        n_points = min(xs.shape[0], signs.shape[0])
        if n_points == 0:
            continue
        xs = xs[:n_points]
        ys = ys[:n_points]
        signs = signs[:n_points]
        phys = physical(coordnames, arr, xs, n_points)
        in_range = (np.asarray(xs, float) >= requested[0]) & (np.asarray(xs, float) <= requested[1])
        any_non_physical |= bool(np.any(~phys & in_range))
        physical_ys.extend(np.asarray(ys, dtype=float)[phys].tolist())
        curves.append((xs, ys, np.where(phys, signs, 0)))   # non-physical: never counted stable
        state_rows = [np.asarray(arr[coordnames.index(s)][:n_points], dtype=float)
                      for s in reduced_state_names if s in coordnames]
        if state_rows:      # whole states, for counting coexisting stable states
            state_curves.append((xs, np.vstack(state_rows).T, np.where(phys, signs, 0)))
        sp_idx = [sp["index"] for sp in c["special_points"]]
        stepped_over += [float((xs[j - 1] + xs[j]) / 2) for j in range(1, n_points)
                         if 0 != signs[j - 1] != signs[j] != 0 and phys[j - 1] and phys[j]
                         and requested[0] <= xs[j] <= requested[1]
                         and not any(abs(k - j) <= 2 for k in sp_idx)]
        for j in range(n_points):
            branch_rows.append({
                "branch": branch_no, param_name: float(xs[j]), plot_state: float(ys[j]),
                "stability": _stability_word(signs[j], phys[j]),
                **{n: float(arr[k][j]) for k, n in enumerate(coordnames) if n != param_name}})

        seg_start = seg_from = 0
        cuts = sorted(sp["index"] for sp in c["special_points"] if sp["type"] in ("HB", "LP", "BP"))
        for i in range(1, n_points + 1):
            if i == n_points or signs[i] != signs[seg_start] or phys[i] != phys[seg_start]:
                stable = signs[seg_start] > 0
                unknown = signs[seg_start] == 0
                lbl_text = (("Stability unknown" if unknown else "Stable" if stable else "Unstable")
                            if phys[seg_start] else "Non-physical (a negative concentration)")
                lbl = lbl_text if lbl_text not in legend_seen else None
                legend_seen.add(lbl_text)
                ax.plot(xs[seg_start:i], ys[seg_start:i],
                        color=(("0.45" if unknown else "black" if stable else "red")
                               if phys[seg_start] else "0.6"),
                        linestyle=(":" if unknown else "-") if phys[seg_start] else "--",
                        linewidth=2.2, label=lbl)
                plotted_anything = True
                # to the next stretch's first point, so that no gap opens between two stretches
                # the stretch ends at AUTO's labelled point when the change sits on one
                end = i if i == n_points else next(
                    (k for k in cuts if max(seg_start, i - 2) <= k <= i), i)
                seg = _segment(xs[seg_from:end + 1], ys[seg_from:end + 1], requested, param_name, plot_state)
                if seg:
                    branch_segments.append({"branch": branch_no,
                                            "stability": _stability_word(signs[seg_start],
                                                                         phys[seg_start]),
                                            **seg})
                seg_start, seg_from = i, end

        for sp_pt in c["special_points"]:
            idx = sp_pt["index"]
            if idx >= n_points:
                continue
            ty = sp_pt["type"]
            if ty in ("EP", "MX"):
                continue
            if not phys[idx]:
                non_physical_points.append({"type": ty, param_name: float(xs[idx])})
                continue
            reduced_state_at_pt = {s: float(arr[coordnames.index(s)][idx])
                                    for s in reduced_state_names if s in coordnames}
            ev_at_pt = eigs_at(reduced_state_at_pt, float(xs[idx])) if ty == "HB" else None
            if ty == "HB" and not _is_hopf(ev_at_pt):
                ty = "NS"           # AUTO's Hopf test also fires at a neutral saddle
                neutral_saddles.append(float(xs[idx]))
            full_state_at_pt = reconstruct_eliminated(model, reduced_state_at_pt,
                                                       {**param_values, param_name: float(xs[idx])})
            all_special.append({
                "type": ty,
                "description": _BIF_TYPE_DESCRIPTIONS.get(ty, ty),
                param_name: float(xs[idx]),
                plot_state: float(ys[idx]),
                "full_state": full_state_at_pt,
                **({"period_at_onset": _onset_period(ev_at_pt)} if ty == "HB" else {}),
            })
            marker = {"LP": "s", "BP": "^", "HB": "o"}.get(ty, "x")
            ax.plot(xs[idx], ys[idx], marker=marker, markersize=10,
                    markerfacecolor="gold", markeredgecolor="black", zorder=5)
            ax.annotate(ty, (xs[idx], ys[idx]), textcoords="offset points",
                        xytext=(8, 8), fontsize=10, fontweight="bold")

    # Periodic orbits born at the Hopf points: max and min of the plotted species over one
    # cycle, green (solid = stable cycle, dashed = unstable), folds of cycles marked.
    periodic_summary = []
    followed = []                   # (how, where, origin) of each branch already summarised
    for pbr in data.get("periodic") or []:
        pd_at = pbr.get("pd_parameter")       # a branch of period-doubled cycles
        origin = pd_at if pd_at is not None else pbr["hopf_parameter"]
        # matched at AUTO's printed precision (6 digits): the runner reads Hopf points from its table
        if pd_at is None and any(abs(pbr["hopf_parameter"] - x) <= 1e-4 * max(1.0, abs(x))
                                 for x in neutral_saddles):
            continue                # not a Hopf point: no cycles are born there
        if not any(abs(pbr["hopf_parameter"] - pt[param_name]) <= 1e-4 * max(1.0, abs(pt[param_name]))
                   for pt in all_special if pt["type"] == "HB"):
            continue                # a Hopf point on a non-physical branch (negative states)
        try:
            barr = np.asarray(pbr["data"], dtype=float)
            if barr.ndim != 2 or not barr.shape[1]:
                continue
            bnames = [str(c).upper() for c in pbr["coordnames"]]
            period_col = next((i for i, c in enumerate(bnames) if c in ("PERIOD", "PAR(11)")), None)
            signs_p = _point_stability_signs(barr.shape[1], pbr["stability_segments"])
            pts, diverged, biggest, stop = [], 0, 0.0, None
            for cyc in pbr["cycles"]:
                idx, orbit = cyc["index"], cyc["orbit"]
                if idx >= barr.shape[1]:
                    continue
                if plot_state_elim_expr is not None:
                    rows = [np.asarray(orbit[s], dtype=float) for s in reduced_state_names if s in orbit]
                    if len(rows) != len(reduced_state_names):
                        continue
                    pvals = [float(barr[0, idx]) if p == param_name
                             else param_values[p] for p in model['param_names']]
                    ys = np.asarray(elim_fn(*rows, *pvals), dtype=float)
                elif plot_state in orbit:
                    ys = np.asarray(orbit[plot_state], dtype=float)
                else:
                    continue
                period = float(barr[period_col, idx]) if period_col is not None else None
                if period is not None and period <= 0:
                    stop = "zero_period"
                    break           # past a zero period the continuation has left real cycles
                if not np.all(np.isfinite([barr[0, idx], *ys, period or 0.0])):
                    diverged += 1   # a diverged point (NaN): no cycle, and no stability to read
                    continue
                amp = float(np.max(ys) - np.min(ys))
                if pts and amp <= 1e-6 * biggest:
                    # the cycles shrank to a point: the branch reached a Hopf point, and past it
                    # AUTO follows the steady state with a made-up period, not cycles
                    stop = "hopf"
                    break
                biggest = max(biggest, amp)
                pts.append({"p": float(barr[0, idx]), "max": float(np.max(ys)), "min": float(np.min(ys)),
                            "period": period, "type": cyc["type"],
                            "stable": None if signs_p[idx] == 0 else bool(signs_p[idx] > 0)})
            # stability and labels as AUTO gives them (its Floquet multipliers): no smoothing
            tol = 0.01 * (requested[1] - requested[0])
            # the same cycles: an earlier branch shrank into this Hopf point, or the two branches
            # join the same pair of Hopf points (one may stop short of the other's end). Drawn
            # twice, two dashed lines on top of each other look like one solid (stable) line
            same = next((began for how, where, began in followed if pd_at is None
                         and abs(where - origin) <= tol
                         and (how == "hopf" or (pts and abs(pts[-1]["p"] - began) <= tol))), None)
            if same is not None:
                periodic_summary.append({"from_hopf_at": pbr["hopf_parameter"], "same_cycles_as": (
                    f"the branch from the Hopf point at {same:.6g}: it joins this Hopf point, so "
                    "these are the same cycles, listed and drawn once under that branch.")})
                continue
            born = (f"this period doubling (on the cycles from the Hopf point at "
                    f"{pbr['hopf_parameter']:.6g})" if pd_at is not None else "this Hopf point")
            where_from = ({"from_period_doubling_at": pd_at, "on_cycles_from_hopf_at": pbr["hopf_parameter"]}
                          if pd_at is not None else {"from_hopf_at": pbr["hopf_parameter"]})
            span_p = max(q["p"] for q in pts) - min(q["p"] for q in pts) if pts else 0.0
            if len(pts) < 2 or span_p < 1e-4 * (requested[1] - requested[0]):
                # the run never left its start: its stability and folds are noise
                periodic_summary.append({
                    **where_from,
                    "error": f"AUTO could not follow the cycles away from {born} (the "
                             + (f"continuation broke down: {diverged} points were not finite "
                                "numbers" if diverged else "continuation stalled there")
                             + "): their size and stability are not known from this run."})
                continue
            followed.append((stop, pts[-1]["p"], origin))
            start = 0
            for i in range(1, len(pts) + 1):
                if i == len(pts) or pts[i]["stable"] != pts[start]["stable"]:
                    seg = pts[max(start - 1, 0):i]
                    stable = pts[start]["stable"]
                    lbl_text = ("Cycle, stability unknown (max/min)" if stable is None else
                                "Stable cycle (max/min)" if stable else "Unstable cycle (max/min)")
                    lbl = lbl_text if lbl_text not in legend_seen else None
                    legend_seen.add(lbl_text)
                    for key in ("max", "min"):
                        ax.plot([q["p"] for q in seg], [q[key] for q in seg], color="tab:green",
                                linestyle=":" if stable is None else "-" if stable else "--",
                                linewidth=1.8,
                                label=lbl if key == "max" else None)
                    start = i
            for q in pts:
                # folds of cycles are always computed (AUTO needs them to follow the branch)
                # but labelled only when the task asks for them
                if q["type"] in ("PD", "TR", "BP") or (q["type"] == "LP" and cycle_folds):
                    ax.plot(q["p"], q["max"], marker="D", markersize=7, markerfacecolor="white",
                            markeredgecolor="tab:green", zorder=5)
                    ax.annotate({"LP": "fold of cycles", "PD": "period doubling", "BP": "branch point",
                                 "TR": "torus"}[q["type"]], (q["p"], q["max"]),
                                textcoords="offset points", xytext=(6, 6), fontsize=8)
            plotted_anything = True
            inside = [q for q in pts if requested[0] <= q["p"] <= requested[1]]
            by_stability = _cycles_by_stability(inside, plot_state)
            hopf_type = {"hopf_type": _l1_type(data, origin)} if pd_at is None else {}
            # where the stability changes with no labelled point between: AUTO stepped over a
            # fold, a period doubling or a torus point there
            unlabelled = [round((a["p"] + b["p"]) / 2, 6) for a, b in zip(pts, pts[1:])
                          if None not in (a["stable"], b["stable"]) and a["stable"] != b["stable"]
                          and not {a["type"], b["type"]} & {"LP", "PD", "TR"}
                          and requested[0] <= (a["p"] + b["p"]) / 2 <= requested[1]]
            cycle_rows += [{**where_from, param_name: q["p"],
                            "period": q["period"], f"{plot_state}_min": q["min"],
                            f"{plot_state}_max": q["max"],
                            "stability": {True: "stable", False: "unstable"}.get(q["stable"],
                                                                                  "unknown"),
                            "point_type": q["type"]} for q in pts]
            periodic_summary.append({
                **where_from,
                **hopf_type,
                "cycles_over": next(iter(_stretches([{**q, "stable": True} for q in pts], True, requested)), None),
                # each stretch of stable cycles apart: two stable stretches with unstable
                # cycles between them are not one interval
                "stable_cycles_over": _stretches(pts, True, requested),
                **({"cycles_of_unknown_stability_over": _stretches(pts, None, requested)}
                   if any(q["stable"] is None for q in pts) else {}),
                "branch_in_order": _branch_in_order(pts),
                # period and amplitude of stable and of unstable cycles kept apart
                "cycles_by_stability": by_stability,
                # how the period changes across the branch, not only its range
                **({"period_at_ends": {
                    f"{param_name} = {q['p']:.6g}": q["period"]
                    for q in (min(inside, key=lambda q: q["p"]), max(inside, key=lambda q: q["p"]))}}
                   if inside and inside[0]["period"] is not None else {}),
                **({"folds_of_cycles_at": sorted({float(f"{q['p']:.5g}") for q in pts if q["type"] == "LP"})}
                   if cycle_folds else {}),     # not asked for: not reported, not mentioned
                **({"folds_of_no_size_at": _folds_of_no_size(pts), "folds_of_no_size_note": (
                    "AUTO labelled these as folds, but their cycles have almost no amplitude: they "
                    "sit on a Hopf point, where the branch begins or ends. Do not count them as "
                    "folds between cycles of real size.")} if cycle_folds and _folds_of_no_size(pts) else {}),
                "other_cycle_points": [{"type": q["type"], param_name: q["p"]} for q in pts
                                       if q["type"] in ("PD", "TR", "BP")],
                **({"period_doubling": (
                    "At a period doubling (PD) this branch's cycles become unstable and cycles "
                    "of twice the period are born there (their own entry, from_period_doubling_at, "
                    "when AUTO could follow them): a PD is not, by itself, where oscillations stop.")}
                   if any(q["type"] == "PD" for q in pts) else {}),
                **({"unlabelled_stability_changes_at": unlabelled, "unlabelled_note": (
                    "The cycles change stability here with no labelled point between: AUTO "
                    "stepped over a fold, period doubling or torus point, of unknown kind.")}
                   if unlabelled else {}),
                # a value outside cycles_over is not a value without oscillations
                "scope": (f"These are only the cycles born at {born}. A branch begins or ends at its "
                          "Hopf value itself; its nearest saved point lies just past it. Other cycles, "
                          "e.g. large ones born elsewhere, can exist at the same "
                          f"{param_name} values and are not listed: whether the model "
                          "oscillates at a value is shown by a simulation there."),
                **_cycle_end(pbr.get("end"), stop, diverged, pts[-1]["p"], requested, param_name),
            })
        except Exception as e:      # the cycles are an addition: never cost the diagram
            periodic_summary.append({"from_hopf_at": pbr.get("hopf_parameter"),
                                     **({"from_period_doubling_at": pd_at} if pd_at is not None else {}),
                                     "error": f"{type(e).__name__}: {e}"})

    periodic_summary = _settle_cycle_entries(periodic_summary, param_name,
                                             0.01 * (requested[1] - requested[0]))

    if not plotted_anything:
        return json.dumps({
            "error": f"AUTO-07p ran but produced no plottable points for "
                     f"param='{param_name}' vs state='{plot_state}'.",
        })

    # where the model sits now: which side of each bifurcation it is on
    ax.axvline(param_start, color="0.4", linestyle=":", linewidth=1.5,
               label=f"start ({10 ** param_start:.4g})" if log_of else f"model value ({param_start:.4g})")
    if extended:
        ax.set_xlim(*requested)
    if log_of:
        # x is log10 of the parameter: ticks show the parameter itself, one per decade, and the
        # view is the special points with a decade either side (the whole window if none)
        from matplotlib.ticker import FuncFormatter, MultipleLocator
        inside = [pt[param_name] for pt in all_special if requested[0] <= pt[param_name] <= requested[1]]
        ax.set_xlim(max(requested[0], min(inside) - 1), min(requested[1], max(inside) + 1)) \
            if inside else ax.set_xlim(*requested)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{v:g}}}$"))
        if not (any_non_physical and physical_ys):
            # y from what is in view: a branch followed far past the window would squash it
            x0, x1 = ax.get_xlim()
            seen = [y for ln in ax.lines if ln.get_transform() == ax.transData   # not axvline
                    for x, y in zip(*[np.asarray(a, float).ravel() for a in ln.get_data()])
                    if x0 <= x <= x1 and np.isfinite(y)]
            if seen:
                lo, hi = min(seen), max(seen)
                pad = 0.05 * (hi - lo) if hi > lo else 0.05 * max(1.0, abs(hi))
                ax.set_ylim(lo - pad, hi + pad)
    if any_non_physical and physical_ys:     # scale to the physical states
        lo, hi = min(physical_ys), max(physical_ys)
        pad = 0.05 * (hi - lo) if hi > lo else 0.05 * max(1.0, abs(hi))
        ax.set_ylim(lo - pad, hi + pad)
    shown = f"{log_of} (log scale)" if log_of else param_name
    ax.set_xlabel(shown, fontweight="bold", fontsize=14)
    ax.set_ylabel(plot_state, fontweight="bold", fontsize=14)
    ax.set_title(f"{model['label']} -- bifurcation diagram ({log_of or param_name} sweep)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize="small")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # one file per species, range and changed parameter values, so another diagram of the
    # model (e.g. the same sweep with one parameter doubled) never overwrites this one
    tag = f"{requested[0]:.4g}_{requested[1]:.4g}".replace("+", "")
    changed = {**{k: v for k, v in (parameters or {}).items() if k != param_name},
               **{k: v for k, v in fixed_params.items() if k != param_name
                  and _as_float(v) != _as_float(model["param_defaults"].get(k, v))}}
    if changed:
        text = "_".join(f"{k}-{_as_float(v):g}" if _as_float(v) is not None else f"{k}-{v}"
                        for k, v in sorted(changed.items()))
        text = re.sub(r"[^\w.-]", "", text)
        if len(text) > 40:
            import hashlib
            text = hashlib.md5(text.encode()).hexdigest()[:8]
        tag += f"_{text}"
    # the same diagram with and without its cycles, another resolution or another start is
    # another file: an earlier answer's plot is never replaced by a different picture
    import hashlib
    tag += "_" + hashlib.md5(json.dumps([n_pts, bool(periodic_summary), initial_conditions, ds, dsmin, dsmax,
                                         nmx], sort_keys=True, default=str).encode()).hexdigest()[:6]
    plot_path = PLOTS_DIR / f"{model['label']}_auto_bifurcation_{param_name}_{plot_state}_{tag}.png"
    fig.tight_layout()
    fig._dynalyst_no_twin = True        # the interactive twin is drawn from the branch data below
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    relative_plot_path = config.display_path(plot_path)
    data_path = plot_path.with_name(plot_path.stem + "_branches.csv")
    _write_branch_csv(data_path, branch_rows, cycle_rows)

    # de-duplicate special points found from both forward/backward runs
    dedup = []
    for pt in all_special:
        if not any(pt["type"] == d["type"] and abs(pt[param_name] - d[param_name]) <= 1e-6 * max(abs(d[param_name]), 1e-12) for d in dedup):
            dedup.append(pt)
    if extended:        # only what lies in the range that was asked for
        dedup = [pt for pt in dedup if requested[0] <= pt[param_name] <= requested[1]]
    dedup = _without_folds_on_branch_points(dedup, param_name, requested[1] - requested[0])
    fold_notes = []
    dedup = _refine_folds(dedup, model, param_name, param_values, reduced_state_names,
                          requested, notes=fold_notes)
    for pt in dedup:
        if pt["type"] == "HB":
            pt["hopf_type"] = _l1_type(data, pt[param_name])
    unphysical = []
    for pt in non_physical_points:
        if (not extended or requested[0] <= pt[param_name] <= requested[1]) and not any(
                pt["type"] == d["type"] and abs(pt[param_name] - d[param_name]) <= 1e-6 * max(abs(d[param_name]), 1e-12)
                for d in unphysical):
            unphysical.append(pt)

    conservation_note = None
    if full_model is not model:
        conservation_note = (
            f"Detected conservation law(s); {', '.join(eliminated_names)} "
            f"were algebraically eliminated before continuation (standard "
            f"practice for moiety-conserved models -- otherwise the "
            f"continuation Jacobian is rank-deficient at every point, not "
            f"just at real bifurcations) and reconstructed in the reported "
            f"state values below."
        )

    plot_html = None
    from dynalyst import interactive_plots
    if interactive_plots.enabled():
        try:        # XPPAUT-style page: y-variable menu, step to each point, hover with all states
            plot_html = config.display_path(interactive_plots.save(interactive_plots.bifurcation_figure(
                branch_rows, cycle_rows, dedup, param_name, plot_state,
                title=f"{full_model['label']} -- bifurcation diagram ({param_name})",
                x_range=tuple(requested)), plot_path))
        except Exception:       # an extra: never costs the diagram
            plot_html = None

    # a stability change right next to a labelled point is that point's own exchange of
    # stability seen on another branch (e.g. the branch switched onto at a BP), not a miss
    near = 0.005 * (requested[1] - requested[0])
    stepped_over = [x for x in stepped_over
                    if not any(abs(x - pt[param_name]) <= near for pt in dedup)]

    return json.dumps({
        "model": full_model["label"],
        "odes": ode_strings(full_model),
        "swept_parameter": param_name,
        **({"other_branches_followed": (
            f"Also followed: the branches through {len(other_starts)} steady state(s) the start did "
            f"not reach, found by find_steady_states inside the range ({param_name} = "
            + ", ".join(f"{s[param_name]:.6g}" for s in other_starts) + ").")}
           if other_starts else {}),
        "branches_followed": (
            f"The branches connected to the start at {param_name} = {param_start:.6g}, and those "
            "through every steady state a multi-start search found at " + param_name + " = "
            + ", ".join(f"{v:.4g}" for v in _other_state_probes(requested, param_values.get(param_name),
                                                                log_spaced, log_of))
            + ". A branch with no steady state at any of these values can still be missed, so a "
            "statement about the model as a whole (no fold, not bistable) is not proven."
            if find_other_branches else
            f"Only the branches connected to the start at {param_name} = {param_start:.6g}, so a "
            "statement about the model (no fold, not bistable) does not follow from them. Others may "
            "exist: find_steady_states with param_name, param_min and param_max lists the states it finds, and "
            "this tool continues from one with fixed_params (its parameter value) and "
            "initial_conditions (that state)."),
        "swept_range": list(requested),
        **({"run_window": (
            f"The continuation was followed over [{extended[0]:g}, {extended[1]:g}], wider than "
            f"the requested range [{requested[0]:g}, {requested[1]:g}]"
            + (f" (the model's own value {param_name} = {param_start:g} lies outside it, and the "
               "continuation starts there)" if not requested[0] <= param_start <= requested[1]
               else " (branches inside it turn at folds outside it)")
            + "; only the requested range is reported and drawn.")} if extended and not widened_from else {}),
        **({"range_widened": (
            f"The requested range [{widened_from[0]:g}, {widened_from[1]:g}] was widened to "
            f"[{requested[0]:g}, {requested[1]:g}]: a branch left it unstable, and the wider run "
            "found the bifurcation point(s) past its end. Every point found is drawn in this one "
            "diagram.")} if widened_from else {}),
        "fixed_parameters": {k: v for k, v in param_values.items() if k != param_name},
        **_auto_problems(data),
        **({"unlabelled_steady_state_changes_at": sorted(round(x, 6) for x in stepped_over),
            "unlabelled_note": (
                "The steady state changes stability here with no special point found: AUTO stepped "
                "over a bifurcation (e.g. a narrow fold pair or a Hopf point). A larger n_pts or a "
                "narrower range around it finds it.")} if stepped_over else {}),
        **({"cycles_not_followed_from": data["hopf_not_followed"], "cycles_not_followed_note": (
            "More Hopf points than the cycle continuation follows in one run (5): the cycles born "
            "at these were not computed. A diagram over a narrower range around them follows them.")}
           if data.get("hopf_not_followed") else {}),
        "conservation_laws_detected": conservation_note,
        "fold_notes": fold_notes or None,
        # one short line per special point, kept whole in a language model's view even when the
        # full point list is shortened: the corrected fold values must reach the answer
        "special_points_summary": [
            f"{pt['type']} at {param_name} = {pt[param_name]:.6g}"
            + (f" (AUTO-07P reported {pt['reported_at']:.6g}; corrected by the fold equations)"
               if pt.get("reported_at") is not None else "")
            for pt in dedup] or None,
        **({"non_physical": (
            "Part of the continuation passes through states with a negative concentration "
            "(drawn grey and dashed). Those parts are not states the system can be in: their "
            "stability does not count and their special points are not bifurcations of the model"
            + (": " + ", ".join(f"{p['type']} at {param_name} = {p[param_name]:.6g}"
                                for p in unphysical) if unphysical else "")
            + ".")} if any_non_physical else {}),
        "range_coverage_warning": range_warning,
        "range_coverage": {
            "reached_param_min": reached_param_min,
            "reached_param_max": reached_param_max,
            "abnormal_terminations": abnormal_terminations,
        },
        "swept_parameter_start": param_start,
        **({"start_moved": (f"{param_name} = {start_moved[0]:.6g} is at a Hopf point, which AUTO "
                            f"cannot detect where it starts: the continuation starts at "
                            f"{start_moved[1]:.6g} instead.")} if start_moved else {}),
        "auto_constants_used": constants_used,
        "xpp_metadata": xpp_meta if xpp_meta else None,
        "xpp_source": str(xpp_source) if xpp_source else None,
        "xpp_source_handoff": (
            "original uploaded XPP source explicitly carried through TXT/ODE->SBML conversion"
            if xpp_source_path else
            "source rediscovered from analysis input"
        ),
        "parameter_selection": parameter_selection_reason,
        "range_selection": range_selection_reason,
        "plot_selection": plot_selection_reason,
        "auto_diagnostics": data.get("diagnostics"),
        "auto_stdout_log": data.get("auto_stdout_log") if show_auto_log else None,
        "auto_terminal_log": (
            _format_auto_terminal(
                data, full_model["label"], param_name, param_min, param_max
            )
            if show_auto_log else None
        ),
        "starting_equilibrium": reconstruct_eliminated(
            model, {s: float(v) for s, v in zip(reduced_state_names, y_start)}, param_values
        ),
        "conserved_totals": full_model.get("conserved_totals", {}),
        **({"conserved_totals_kept": (
            "The initial_conditions given would change these conserved totals; each keeps the "
            "model's own value and the other states in it were adjusted (pass the total_* "
            f"parameter to change one): {totals_kept}")} if totals_kept else {}),
        "reconstruction": {n: str(e) for n, e in model.get('eliminated', [])},
        "bifurcation_points": dedup,
        "n_bifurcation_points": len(dedup),
        "classification": _classification(dedup, curves, param_name, plot_state,
                                          bool(reached_param_min and reached_param_max), tuple(requested),
                                          state_curves=state_curves),
        **_coexistence(state_curves or curves, dedup, param_name, requested,
                       state_tol=1e-3 if state_curves else None),
        # the numbers behind the plot: each stretch of one stability on each branch, in the
        # requested range (the full point list, equilibria and cycles, is in the CSV)
        "equilibrium_branches": branch_segments,
        "branch_data_csv": config.display_path(data_path),
        **({"periodic_orbits": periodic_summary} if periodic_summary else
           {"cycle_continuation": "available, not run in this call: this tool continues the "
                                  "limit cycles born at the Hopf point(s) -- their period, "
                                  "amplitude, stability (Floquet multipliers) and period "
                                  "doublings. If the task asks for those, call this tool again "
                                  "with cycles=true. If not, the user is asked after the answer: "
                                  "report it as 'offered, not yet run'. Period, amplitude and cycle "
                                  "stability across the range come from it, not from simulations."}
           if not cycles and any(p["type"] == "HB" for p in dedup) else
           {"periodic_orbits": (
               "Hopf point(s) found, but AUTO could not continue the cycles born there"
               + (f" ({'; '.join(e for e in data.get('errors') or [] if 'cycles' in e)[:300]})"
                  if any('cycles' in e for e in data.get('errors') or []) else "")
               + ": the diagram shows where oscillations start, not their size.")}
           if any(p["type"] == "HB" for p in dedup) else {}),
        "plot_state": plot_state,
        "plot_path": relative_plot_path,
        **({"plot_html": plot_html, "plot_html_note": (
            "Interactive version of the diagram (open in a browser, or shown in the notebook): zoom, "
            "pan, a menu for the y variable, buttons to step to each labelled point, hover with all "
            "states.")} if plot_html else {}),
        "plot_note": (
            f"A real AUTO-07p continuation diagram was computed and saved. Embed it as: "
            f"![{full_model['label']} bifurcation diagram]({relative_plot_path})"
        ),
        "interpretation": (
            "Black = stable branch, red = unstable branch (from AUTO's own linear "
            "stability tracking along the continuation, not a re-derived heuristic). "
            "LP = saddle-node/fold, BP = transcritical/pitchfork (branch point), "
            "HB = Hopf (onset of oscillation). Green = the periodic orbits born at a Hopf "
            "point (AUTO periodic-orbit continuation): the max and min of the plotted species "
            "over one cycle, solid where the cycle is stable, dashed where it is unstable. "
            "A Hopf point's period_at_onset (2 pi / omega from its eigenvalues) is the period "
            "of the cycles born there; periods elsewhere on the cycle branch differ."
        ),
    }, indent=2, default=float)


def _without_folds_on_branch_points(points, param_name, width) -> list:
    """AUTO labels the turn of a pitchfork's two arms a fold, at the branch point itself: the same
    point, not a fold of its own."""
    def same(a, b):
        keys = [k for k in a if k in b and k != "type" and isinstance(a[k], float) and isinstance(b[k], float)]
        return abs(a[param_name] - b[param_name]) <= 1e-4 * max(abs(width), 1e-12) and all(
            abs(a[k] - b[k]) <= 1e-3 * (1 + abs(b[k])) for k in keys if k != param_name)
    return [p for p in points if not (p["type"] == "LP" and any(
        q["type"] == "BP" and same(p, q) for q in points))]


def _coexistence(curves, points, param_name, requested, state_tol=None) -> dict:
    """Where two or more stable steady states coexist, as windows in the parameter (from
    AUTO's own stability on the branches followed), so an answer states the window instead of
    guessing it from the special points. A window end within one grid step of a special point
    is that point's value; an end at the range's end says so."""
    from dynalyst.regime.taxonomy import coexistence_windows
    try:
        windows = coexistence_windows(curves, within=tuple(requested), state_tol=state_tol)
    except Exception:       # a summary must never cost the diagram
        return {}
    if not windows:
        return {}
    step = 1.5 * (requested[1] - requested[0]) / 400
    out = []
    for w in windows:
        row = {}
        for end in ("from", "to"):
            x = w[end]
            snap = [pt for pt in points if abs(pt[param_name] - x) <= step]
            if snap:
                pt = min(snap, key=lambda q: abs(q[param_name] - x))
                row[end] = pt[param_name]
                row[f"{end}_point"] = pt["type"]
            elif abs(x - requested[1 if end == "to" else 0]) <= step:
                row[end] = requested[1 if end == "to" else 0]
                row[f"{end}_point"] = "end of the range"
            else:
                row[end] = x
        row["stable_states"] = w["stable_states"]
        if out and row["from"] - out[-1]["to"] <= step:
            # windows that touch (at a branch point the count dips for one grid value, where
            # two branches exchange stability) are one window
            out[-1].update({k: v for k, v in row.items() if k.startswith("to")},
                           stable_states=max(out[-1]["stable_states"], row["stable_states"]))
            continue
        out.append(row)
    return {"coexisting_stable_states": out, "coexisting_note": (
        "Parameter windows where two or more stable steady states coexist on the branches "
        "followed (AUTO's own stability). An end marked 'end of the range' may continue past it; "
        "an end with no point named is accurate to within one grid step.")}


def _refine_folds(points, model, param_name, param_values, reduced_state_names, window=None,
                  notes=None):
    """Each fold AUTO reported, solved exactly (f = 0, zero eigenvalue) from where it was
    reported. A fold that moves by more than fold_refine.ACCEPT_REL is replaced by the solved
    value, and the reported one is kept as reported_at; one that cannot be solved is marked."""
    if not any(p.get("type") == "LP" for p in points) or param_name not in model["param_names"]:
        return points
    try:
        from dynalyst.analysis.fold_refine import refine_fold
        from dynalyst.analysis.steady_states import _lambdify_model
        rhs_fn, jac_fn = _lambdify_model(model)
    except Exception:
        return points
    names_p = model["param_names"]
    k = names_p.index(param_name)
    base = [param_values[q] for q in names_p]
    def vec(p):
        v = list(base); v[k] = p; return v
    f = lambda y, p: np.array(rhs_fn(*y, *vec(p)), dtype=float)
    j = lambda y, p: np.array(jac_fn(*y, *vec(p)), dtype=float)
    out = []
    for pt in points:
        if pt.get("type") != "LP":
            out.append(pt)
            continue
        state = pt.get("full_state") or {}
        if any(s not in state for s in reduced_state_names):
            out.append(pt)
            continue
        p0 = float(pt[param_name])
        try:
            res = refine_fold(f, j, [state[s] for s in reduced_state_names], p0)
        except Exception as e:
            out.append({**pt, "fold_check": f"not checked ({type(e).__name__})"})
            continue
        new = dict(pt)
        new["fold_check"] = res["detail"]
        if res["converged"] and not res["accepted"]:
            p_new = float(res["p"])
            if window and not (window[0] <= p_new <= window[1]):
                # the true fold lies outside the range that was asked for: say so, do not drop it silently
                if notes is not None:
                    notes.append(f"AUTO-07P reported a fold at {param_name} = {p0:.6g}; the fold "
                                 f"equations put it at {p_new:.10g}, outside the requested range "
                                 f"[{window[0]:.6g}, {window[1]:.6g}], so it is not listed")
                continue
            new["reported_at"] = p0
            new[param_name] = p_new
            y_new = dict(zip(reduced_state_names, res["y"]))
            try:
                new["full_state"] = reconstruct_eliminated(model, y_new,
                                                           {**param_values, param_name: p_new})
            except Exception:
                new["full_state"] = {**state, **y_new}
        out.append(new)
    merged = []
    for pt in out:
        if pt.get("type") == "LP":
            twin = next((m for m in merged if m.get("type") == "LP" and abs(m[param_name] - pt[param_name])
                         <= 1e-6 * max(abs(pt[param_name]), 1e-12)), None)
            if twin is not None:
                seen = twin.get("also_reported_at", [])
                if "reported_at" in pt:
                    seen.append(pt["reported_at"])
                twin["also_reported_at"] = seen
                twin["fold_check"] = (twin.get("fold_check", "") +
                                      "; the same fold was also reached on another branch")
                continue
        merged.append(pt)
    return merged


def _classification(points, curves, param_name, plot_state, covered=True, within=None,
                    state_curves=None):
    """The same labels the LHS scans use (regime/taxonomy.py), from AUTO's own
    stability: bistable only where two stable states coexist."""
    from dynalyst.regime.taxonomy import classify_diagram
    try:
        c = classify_diagram([p["type"] for p in points], curves, param_name, plot_state, covered=covered,
                             within=within, state_curves=state_curves or None)
    except Exception as e:      # a label must never cost the diagram
        return {"category": "unclassified", "reason": f"{type(e).__name__}: {e}"}
    ev = c["evidence"]
    if "bistable" in c["features"] and ev.get("at") is not None:
        n = int(ev["max_coexisting_stable"])
        # the category 'bistable' (shared with the LHS scans) means 2 or more stable states:
        # the diagram names how many
        c["multistability"] = {2: "bistable", 3: "tristable", 4: "tetrastable"}.get(
            n, f"multistable ({n} stable states)")
        c["reason"] = (f"{c['multistability']}: {n} stable steady states coexist at "
                       f"{param_name} = {ev['at']:.6g}")
        if n > 2:
            c["category_note"] = (f"category 'bistable' covers 2 or more coexisting stable states; "
                                  f"this diagram is {c['multistability']}: call it that")
    elif ev.get("note"):
        c["reason"] = ev["note"]
    folds = sorted(float(p[param_name]) for p in points
                   if p.get("type") == "LP" and param_name in p)
    if "bistable" in c["features"] and len(folds) >= 2 and folds[0] > 0:
        # a window's width depends on the parameter's units; the ratio of its ends does not
        c["fold_ratio"] = round(folds[-1] / folds[0], 4)
        c["fold_ratio_note"] = (f"upper fold / lower fold = {folds[-1]:.6g} / {folds[0]:.6g}. "
                                "Compare bistable windows of different parameters by "
                                "fold_ratio: absolute widths depend on units and scale.")
    c["states_counted_on"] = "only the branches followed (branches_followed): not the whole model"
    return c


def _select_generic_candidate_parameters(model: dict, xpp_source: Path | None,
                                          requested: str | None,
                                          max_parameter_trials: int | None = None) -> list[str]:
    """Select continuation parameters from the supplied model only.

    This follows the XPP-AUTO mental model: if the user gives a parameter, use
    exactly that parameter; otherwise prefer an explicit XPP AUTO hint, then
    the model's declared parameter order. No biochemical parameter names are
    assumed.
    """
    names = list(model.get("param_names", []))
    if requested:
        if requested not in names:
            raise ValueError(
                f"'{requested}' is not a parameter of this model. Available parameters: {names}"
            )
        return [requested]
    if not names:
        return []

    ordered: list[str] = []
    if xpp_source is not None:
        meta = _parse_xpp_metadata(xpp_source)
        for key in ("autopar", "parameter", "par", "icp"):
            hinted = meta.get(key)
            if hinted in names:
                ordered.append(hinted)
                break
        for name in _xpp_declared_parameter_order(xpp_source):
            if name in names and name not in ordered:
                ordered.append(name)

    # Preserve the model/parser declaration order. This is much closer to how
    # XPP presents the model to the user than optimizing for guessed sensitivity.
    ordered.extend(name for name in names if name not in ordered)

    if max_parameter_trials is None:
        # An unspecified continuation parameter is an automated exploratory
        # request. XPP-AUTO itself requires an ICP choice, so our generic
        # wrapper boundedly tries several parameters from THIS model until it
        # finds an actual special point. Explicit param_name remains a single
        # authoritative run.
        max_parameter_trials = 5
    max_parameter_trials = max(1, int(max_parameter_trials))
    return ordered[:max_parameter_trials]

_SELECTION_RULE = (
    "An XPP/AUTO parameter hint wins. Otherwise every parameter the equations use is "
    "screened: the equilibrium the model settles to is followed a decade either side of "
    "the parameter's value, and parameters whose Jacobian eigenvalues cross zero (fold / "
    "branch point / Hopf) closest to their current value are tried first, then the model's "
    "declared order. The first AUTO run with a real LP/BP/HB is returned."
)


def _screen_summary(screen, top=5):
    """The screen's evidence, compact enough for the agent to quote."""
    out = []
    for r in screen[:top]:
        e = r["events"][0] if r["events"] else None
        out.append({"parameter": r["parameter"], "value": r["value"],
                    "nearest_crossing": e["type"] if e else "none within a decade",
                    **({"crossing_near": round(e["near"], 6),
                        "decades_from_value": round(e["decades_from_value"], 3)
                        if e["decades_from_value"] is not None else None} if e else {})})
    return out


def run_bifurcation_analysis(model_name_or_path: str | None = None,
                              species: list[str] | None = None,
                              odes: dict | None = None,
                              parameters: dict | None = None,
                              initial_conditions: dict | None = None,
                              param_name: str | None = None,
                              param_min: float | None = None,
                              param_max: float | None = None,
                              fixed_params: dict | None = None,
                              plot_state: str | None = None,
                              n_pts: int = 500,
                              ds: float | None = None,
                              dsmin: float | None = None,
                              dsmax: float | None = None,
                              nmx: int | None = None,
                              itmx: int | None = None,
                              itnw: int | None = None,
                              nwtn: int | None = None,
                              epsl: float | None = None,
                              epsu: float | None = None,
                              epss: float | None = None,
                              show_auto_log: bool = True,
                              max_parameter_trials: int | None = None,
                              cycles: bool | None = None,
                              cycle_folds: bool | None = None,
                              model_range: bool = False,
                              _cycles: bool = True) -> str:
    """Generic bifurcation entry point for arbitrary XPP/SBML/ODE models.

    With an explicit parameter this performs one authoritative AUTO run. With
    no parameter, it introspects the supplied model, honors an explicit
    XPP/AUTO parameter hint when present, otherwise uses the model's declared
    parameter order, and performs a bounded search over several candidate
    parameters, stopping when AUTO reports an actual special point. If no
    candidate produces a special point, the best successful branch is returned
    with that fact stated explicitly. No biochemical parameter names or
    model-family assumptions are used.
    """
    # First resolve/parse the exact model once so automatic parameter ranking
    # is based on THIS input and never on a previous model.
    source = None
    effective_model_arg = model_name_or_path
    if effective_model_arg and Path(effective_model_arg).suffix.lower() in {".txt", ".ode"}:
        try:
            source = _find_exact_input_path(effective_model_arg)
            from dynalyst.analysis.xpp_to_sbml import parse_xpp
            from dynalyst.analysis.jaxkm_ode_to_sbml import odes_to_sbml
            xs, xo, xp, xic = parse_xpp(source.read_text(errors="replace"))
            if not xs:
                return json.dumps({"error": f"No XPP derivative equations found in '{source.name}'."})
            converted = json.loads(odes_to_sbml(
                species=xs, odes=xo, parameters=xp,
                initial_conditions=xic, model_name=source.stem,
            ))
            if "error" in converted:
                return json.dumps({"error": converted["error"]})
            effective_model_arg = converted["path"]
        except Exception as e:
            return json.dumps({"error": f"Could not prepare the exact XPP model for bifurcation: {e}"})

    try:
        base_model = resolve_model(effective_model_arg, species, odes, parameters, initial_conditions)
        from dynalyst.analysis.conservation_parameters import expose_totals
        base_model = expose_totals(base_model)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    eligible = eligible_parameters(base_model)
    if param_name is not None:
        problem = continuation_parameter_error(base_model, param_name)
        if problem:
            return json.dumps(problem)
        param_name = canonical_param(base_model, param_name)
    if not eligible:
        return json.dumps({"error": NO_PARAMETERS})
    # param_names stays the full list: every numeric function is compiled with all the
    # parameters (param_symbols); the eligible ones only choose and show candidates
    from dynalyst.analysis.model_check import sanity
    check = sanity(base_model)
    if not check["ok"]:      # a problem in the equations: AUTO cannot fix it, so do not run it
        return json.dumps({"error": "The model failed the pre-AUTO sanity check: "
                                    + "; ".join(check["problems"]), "model_sanity": check})
    if fixed_params:
        unknown = [k for k in fixed_params if canonical_param(base_model, k) is None]
        if unknown:
            return json.dumps(unknown_parameter_error(base_model, unknown[0]))
        fixed_params = {canonical_param(base_model, k): v for k, v in fixed_params.items()}
        # '*2' / '/2': relative to the model's own value, as in the simulation tools
        from dynalyst.analysis.ode_model import relative_value
        fixed_params = {k: relative_value(v, base_model["param_defaults"][k])
                        for k, v in fixed_params.items()}

    if param_name is None and (param_min is not None or param_max is not None):
        return json.dumps({
            "error": (
                "param_min/param_max were supplied without param_name. An explicit "
                "range must belong to one specific continuation parameter; otherwise "
                "the same range would be incorrectly applied to every automatic candidate."
            ),
            "available_parameters": list(eligible),
            "hint": "Provide param_name together with param_min/param_max, or omit the range and let it be inferred per candidate."
        })

    xpp_source = source or _find_xpp_source(effective_model_arg) or _find_xpp_source(model_name_or_path)
    # Like XPP: integrate first, then continue from where the system settles.
    # With no parameter named, screen every parameter the equations use for an
    # eigenvalue crossing near its current value (analysis/parameter_screen.py)
    # and try the closest ones first; the model's own order is the fallback.
    screen, start_guess, start_note = [], None, None
    try:
        # settled at the values this analysis uses (fixed_params), not the file's
        at_values = {**base_model, "param_defaults": {**base_model["param_defaults"],
                                                       **(fixed_params or {})}}
        start_guess, start_note = settled_state(at_values)
        if param_name is None:
            screen = screen_parameters(at_values, start_guess, candidates=eligible)
    except Exception as e:  # the screen is an aid; never block the analysis
        start_note = f"not used ({type(e).__name__}: {e})"
    # candidate choice only (no numerics): the eligible parameters, screened ones first
    ranked = [r["parameter"] for r in screen if r["events"]]
    select_model = {**base_model, "param_names": ranked + [p for p in eligible if p not in ranked]}
    candidates = _select_generic_candidate_parameters(select_model, xpp_source, param_name, max_parameter_trials)
    if not candidates:
        return json.dumps({"error": "This model has no free parameters available for bifurcation continuation."})

    # Search with plot-free probes, then make ONE full run (one diagram) for
    # the chosen parameter. Windows: a range the user gave (or, with model_range, the XPP
    # file's own) is used exactly; otherwise the log window, the parameter's value times
    # 10^-3 .. 10^3 stretched over a positive XPP range, continued in log10 (_log_window).
    # A parameter with a negative value has no log window: a decade either side, then three,
    # then five, tightened around the special points found.
    top_meta = _parse_xpp_metadata(xpp_source)
    range_given = (param_min is not None or param_max is not None
                   or (model_range and ({"parmin", "parmax"} <= set(top_meta)
                                        or {"autoxmin", "autoxmax"} <= set(top_meta))))
    base_kwargs = dict(
        model_name_or_path=effective_model_arg,
        species=species, odes=odes, parameters=parameters,
        initial_conditions=initial_conditions,
        was_auto_selected=(param_name is None),
        param_min=param_min, param_max=param_max, fixed_params=fixed_params,
        plot_state=plot_state, n_pts=n_pts, ds=ds, dsmin=dsmin, dsmax=dsmax,
        nmx=nmx, itmx=itmx, itnw=itnw, nwtn=nwtn, epsl=epsl, epsu=epsu,
        epss=epss, show_auto_log=show_auto_log,
        xpp_source_path=source, start_guess=start_guess, cycles=_cycles and cycles is not False,
        cycle_folds=bool(cycle_folds),
    )
    deadline = time.monotonic() + _AUTO_SEARCH_TOTAL_TIMEOUT

    def value_of(p):
        return float({**base_model["param_defaults"], **(fixed_params or {})}[p])

    def probe(p, window=None, log_axis=False):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            return {"error": "automatic search wall-clock budget exhausted"}
        kw = {**base_kwargs, "param_name": p, "auto_timeout_seconds": min(180, remaining)}
        if window:
            kw.update(param_min=window[0], param_max=window[1], log_axis=log_axis)
        try:
            return json.loads(_run_single_bifurcation_analysis(probe_only=True, **kw))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def log_window(p, decades=LOG_DECADES):
        return None if range_given else _log_window(value_of(p), top_meta, decades)

    attempts, chosen, window, found_at, log_run = [], None, None, [], False
    if param_name and not range_given and log_window(param_name):
        passes = []         # named parameter, no range: straight to the log window
        chosen, window, log_run = param_name, log_window(param_name), True
    elif range_given:
        passes = [] if param_name else [None]
    else:       # no parameter named: each candidate's log window; a named negative one: decades
        passes = ["log"] if not param_name else [None, 3, 5]
    for decades in passes:
        for cand in candidates:
            # a candidate with a negative value has no log window: the default one for it
            lw = log_window(cand) if decades == "log" else None
            w = lw or (_decade_window(value_of(cand), decades)
                       if decades not in (None, "log") else None)
            r = probe(cand, w, bool(lw))
            if r.get("auto_unavailable"):   # no other parameter or window can help
                return json.dumps(r)
            pts = [float(x[1]) for x in r.get("special_points") or []]
            attempts.append({"parameter": cand, "window": list(w) if w else "default",
                             **({"log_axis": True} if lw else {}),
                             "n_bifurcation_points": len(pts), "error": r.get("error")})
            if not r.get("error") and pts:
                chosen, found_at = cand, pts
                if lw:
                    window, log_run = lw, True
                elif w:
                    window = _window_around(value_of(cand), pts)
                else:
                    window, found_at = _past_unstable_edges(r, value_of(cand), pts,
                                                            lambda wide, c=cand: probe(c, wide))
                break
        if chosen:
            break
    if chosen is None:
        clean = [a for a in attempts if not a["error"]]
        chosen = param_name or (clean[0]["parameter"] if clean else candidates[0])
        if log_window(chosen):      # nothing found anywhere: the diagram is still the log window
            window, log_run = log_window(chosen), True

    # the diagram: states off the followed branches are searched for and continued from
    final = {**base_kwargs, "param_name": chosen, "find_other_branches": True}
    if window:
        final.update(param_min=window[0], param_max=window[1], log_axis=log_run)
    result = json.loads(_run_single_bifurcation_analysis(**final))
    log_d = LOG_DECADES
    if log_run and "error" not in result and not result.get("bifurcation_points"):
        # nothing in 10^-3 .. 10^3 (pitchfork_model: BP at 1300x the value): look wider once.
        # shortcut: only the chosen parameter is widened, not every candidate; loop candidates if that misses
        wide = log_window(chosen, LOG_WIDE_DECADES)
        wider = json.loads(_run_single_bifurcation_analysis(
            **{**final, "param_min": wide[0], "param_max": wide[1]}))
        if "error" not in wider and wider.get("bifurcation_points"):
            result, window, log_d = wider, wide, LOG_WIDE_DECADES
    if log_run and "error" not in result:
        result = _add_actual_values(result, chosen, f"log10_{chosen}")
    if base_model.get("caveats"):     # SBML features the reader did not apply: say so with the result
        result["model_caveats"] = base_model["caveats"]
    if log_run:
        result["range_selection"] = (
            f"default window {window[0]:.4g} .. {window[1]:.4g} ({chosen} = {value_of(chosen):.4g} "
            f"times 10^-{log_d} .. 10^{log_d}"
            + (", widened because 10^-3 .. 10^3 held no special point" if log_d != LOG_DECADES else "")
            + (", stretched to cover the model file's own range"
               if window != _log_window(value_of(chosen), {}, log_d) else "") + "), continued in log10. Nothing outside it was computed: a range given "
            "in the request (param_min, param_max) is used exactly.")
    elif window:
        result["range_selection"] = (
            f"widened automatically: the default window around {chosen}={value_of(chosen):.4g} "
            f"missed special points; a wider probe found them at {[round(x, 6) for x in found_at]}, "
            "so the window was set around them")
    if param_name is None:
        result["automatic_parameter_search"] = {
            "mode": "generic_introspection",
            "candidates_tried": list(dict.fromkeys(a["parameter"] for a in attempts)),
            "attempts": attempts,
            "selected_parameter": chosen,
            "parameters_in_model": list(base_model.get("param_names", [])),
            "parameter_screen": _screen_summary(screen),
            "starting_state": start_note,
            "selection_rule": _SELECTION_RULE + (
                "" if found_at else " No candidate showed a special point; the top-ranked "
                "one is shown without claiming a bifurcation."),
        }
    return json.dumps(result, indent=2, default=float)


def _physical_points(probe_result: dict) -> list[float]:
    """Where a probe found folds, branch points and Hopf points on physical branches (its run
    goes a little past the window, so some may lie outside it)."""
    return sorted({float(br["parameter"][q["index"]]) for br in probe_result.get("branches") or []
                   for q in br.get("points") or [] if q["index"] < len(br["parameter"])})


def _unstable_edges(probe_result: dict) -> list[str]:
    """The ends ('lo', 'hi') of a probe's window where a physical steady state is still
    unstable: its stability changes somewhere past that end (a Hopf point or a fold),
    so the window cut off a special point."""
    lo, hi = probe_result.get("swept_range") or (None, None)
    if lo is None:
        return []
    tol = 1e-3 * (hi - lo)
    edges = set()
    for br in probe_result.get("branches") or []:
        for x, stable, phys in zip(br["parameter"], br["stable"], br.get("physical") or br["stable"]):
            if phys and not stable:
                if lo - tol <= x <= lo + tol:
                    edges.add("lo")
                elif hi - tol <= x <= hi + tol:
                    edges.add("hi")
    return sorted(edges)


def _past_unstable_edges(probe_result: dict, value: float, points: list, probe) -> tuple:
    """(window, points) for the full run when the default window cut special points off:
    the probe already found some past its ends, or its steady state is still unstable at an
    end and one more probe, a decade further on that side, finds more. The window is then
    set around all of them. Otherwise (None, points): the default window stays."""
    lo, hi = probe_result["swept_range"]
    found = _physical_points(probe_result)
    if any(x < lo or x > hi for x in found):
        return _window_around(value, found), found
    edges = _unstable_edges(probe_result)
    if not edges:
        return None, points
    wide = (lo / 10 if "lo" in edges and lo > 0 else lo - (hi - lo) if "lo" in edges else lo,
            hi * 10 if "hi" in edges and hi > 0 else hi + (hi - lo) if "hi" in edges else hi)
    r = probe(wide)
    more = _physical_points(r) if not r.get("error") else []
    if len(more) <= len(found):
        return None, points
    return _window_around(value, more), more


LOG_DECADES = 3     # default window: the parameter's value times 10^-3 .. 10^3
LOG_WIDE_DECADES = 5    # when that window holds no special point: 10^-5 .. 10^5, as the old decade passes did


def _log_window(value: float, xpp_meta: dict, decades: int = LOG_DECADES) -> tuple[float, float] | None:
    """The default window when no range is given: value * 10^-3 .. value * 10^3 (10^-3 .. 10^3
    for a value of 0), stretched to cover a positive range an XPP file declares (@ parmin /
    parmax, or autoxmin / autoxmax). None for a negative value: no log window."""
    if value < 0:
        return None
    lo, hi = (value / 10 ** decades, value * 10 ** decades) if value > 0 else \
        (10.0 ** -decades, 10.0 ** decades)
    for a, b in (("parmin", "parmax"), ("autoxmin", "autoxmax")):
        try:
            mn, mx = float(xpp_meta[a]), float(xpp_meta[b])
        except (KeyError, TypeError, ValueError):
            continue
        hi = max(hi, mx)
        lo = min(lo, mn) if mn > 0 else lo
        break
    return lo, hi


def _log_parameter(model: dict, p: str, q: str, value_q: float) -> dict:
    """The model with parameter p written as 10**q: AUTO then continues q = log10(p), so every
    decade of p gets the same resolution and the tolerances, set as fractions of the range, are
    relative in p."""
    P, Q = sp.Symbol(p), sp.Symbol(q)
    sub = {P: 10 ** Q}
    out = {**model,
           "rhs": [sp.sympify(e).xreplace(sub) for e in model["rhs"]],
           "param_names": [q if n == p else n for n in model["param_names"]],
           "param_symbols": [Q if s == P else s for s in model["param_symbols"]],
           "param_defaults": {(q if k == p else k): (value_q if k == p else v)
                              for k, v in model["param_defaults"].items()},
           "param_kinds": {(q if k == p else k): v for k, v in (model.get("param_kinds") or {}).items()}}
    if model.get("eliminated"):
        out["eliminated"] = [(n, sp.sympify(e).xreplace(sub)) for n, e in model["eliminated"]]
    return out


def _add_actual_values(result: dict, p: str, q: str) -> dict:
    """A result computed in q = log10(p): every value of q gets its actual value p = 10**q next
    to it (bifurcation points, branch stretches, coexistence windows, cycle intervals), so the
    answer can state both."""
    act = lambda v: float(f"{10 ** float(v):.6g}")
    pair = lambda r: [act(r[0]), act(r[1])] if isinstance(r, (list, tuple)) and len(r) == 2 else r

    def walk(x):
        if isinstance(x, list):
            for v in x:
                walk(v)
            return
        if not isinstance(x, dict):
            return
        for k, v in list(x.items()):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if k == q:
                    x[p] = act(v)
                elif k in (f"{q}_from", f"{q}_to"):
                    x[p + k[len(q):]] = act(v)
                elif k in ("from_hopf_at", "from_period_doubling_at", "on_cycles_from_hopf_at",
                           "reported_at"):
                    x[f"{k}_{p}"] = act(v)
            elif k in ("stable_cycles_over", "cycles_over", "cycles_of_unknown_stability_over") and v:
                x[f"{k}_{p}"] = [pair(r) for r in v] if isinstance(v[0], (list, tuple)) else pair(v)
            elif k in ("folds_of_cycles_at", "unlabelled_steady_state_changes_at",
                       "unlabelled_stability_changes_at", "also_reported_at") and isinstance(v, list):
                x[f"{k}_{p}"] = [act(r) for r in v]
            else:
                walk(v)
    walk(result)
    if result.get("special_points_summary"):      # the short lines the answer is written from
        result["special_points_summary"] = [
            f"{pt['type']} at {p} = {act(pt[q])} ({q} = {pt[q]:.6g})"
            + (f" (AUTO-07P reported {p} = {act(pt['reported_at'])}; corrected by the fold equations)"
               if pt.get("reported_at") is not None else "")
            for pt in result.get("bifurcation_points") or []]
    for row in result.get("coexisting_stable_states") or []:
        for end in ("from", "to"):
            if isinstance(row.get(end), (int, float)):
                row[f"{p}_{end}"] = act(row[end])
    # the reviewer, the session state and the final check read these in the model's own terms
    result["swept_parameter"] = p
    if isinstance(result.get("swept_range"), list):
        result[f"swept_range_{q}"] = result["swept_range"]
        result["swept_range"] = pair(result["swept_range"])
    if isinstance(result.get("swept_parameter_start"), (int, float)):
        result[f"swept_parameter_start_{q}"] = result["swept_parameter_start"]
        result["swept_parameter_start"] = act(result["swept_parameter_start"])
    c = result.get("classification") or {}
    folds = sorted(float(pt[q]) for pt in result.get("bifurcation_points") or [] if pt.get("type") == "LP")
    if "fold_ratio" in c and len(folds) >= 2:      # a ratio of p, not of log10(p)
        c["fold_ratio"] = round(10 ** (folds[-1] - folds[0]), 4)
        c["fold_ratio_note"] = (f"upper fold / lower fold = {act(folds[-1])} / {act(folds[0])} in "
                                f"{p}. Compare bistable windows of different parameters by fold_ratio.")
    csv = result.get("branch_data_csv")
    if csv:
        try:
            import pandas as pd
            path = Path(csv) if Path(csv).is_absolute() else config.PROJECT_ROOT / csv
            df = pd.read_csv(path)
            if q in df and p not in df:
                df.insert(df.columns.get_loc(q) + 1, p, 10 ** df[q].astype(float))
                df.to_csv(path, index=False)
        except Exception:       # the CSV is an extra: never cost the result
            pass
    result["log_scale"] = {
        "parameter": p, "continued_in": q,
        "note": (f"AUTO continued {q} = log10({p}) and the diagram's x axis is logarithmic (its "
                 f"ticks are values of {p}). Every parameter value in this result is log10 of "
                 f"{p} (named {q}, also inside the text and the AUTO log) unless it is named {p}; "
                 f"the actual value is given beside it as {p}. State both in the answer, e.g. "
                 f"'HB at {p} = 1.642 ({q} = 0.2155)'.")}
    return result


def _decade_window(value: float, decades: float) -> tuple[float, float]:
    """`decades` either side of a parameter's value (a unit-scaled window
    around 0 when the value is 0)."""
    if value == 0:
        return (-10.0 ** (decades - 1), 10.0 ** (decades - 1))
    lo, hi = sorted((value / 10 ** decades, value * 10 ** decades))
    return (lo, hi)


def _full_coordnames(coordnames: list, full_names: list) -> list:
    """AUTO-07p writes at most 13 characters of a parameter or state name in its
    output ('Reaction12_kcat_f' -> 'Reaction12_kc'), so a long name -- every
    reaction-local SBML parameter is '<reaction>_<name>' -- was never found and the
    run reported 'no branch containing ...'. Map each cut name back to the one full
    name it is the start of."""
    out = []
    for c in coordnames:
        if c in full_names:
            out.append(c)
            continue
        match = {n for n in full_names if len(c) >= 10 and n.startswith(c)}
        out.append(match.pop() if len(match) == 1 else c)
    return out


def _window_around(value: float, points: list) -> tuple[float, float]:
    """A window covering the parameter's value and the special points, with
    a margin, so the diagram shows both. A model value of 0 gives no scale and no sign to go
    by: the window covers both signs equally, as far as the farthest point and a margin, so
    a point found on one side does not leave the other side unexplored. Negative values are
    kept: physicality is judged on the branches, not by cutting the parameter range."""
    if value == 0 and points:
        reach = 1.25 * max(abs(float(p)) for p in points) or 1.0
        return (-reach, reach)
    lo, hi = min([value, *points]), max([value, *points])
    span = hi - lo or abs(hi) or 1.0
    lo = lo * 0.5 if lo > 0 else 0.0 if lo == 0 else lo - 0.25 * span
    hi = hi * 1.5 if hi > 0 else hi + 0.25 * span
    return (lo, hi)


TOOL_DEFINITIONS = [
    {
        "name": "run_bifurcation_analysis",
        "description": (
            "ONE AUTO-07p bifurcation diagram of any model (SBML, XPP, typed ODEs): steady "
            "states continued in one parameter with their stability, the special points LP "
            "(fold), BP (branch point) and HB (Hopf), and the cycles born at a Hopf point. "
            "Self-contained (finds a start state itself). Leave param_name out when none is "
            "named: the tool screens the model. Many parameter sets: lhs_bifurcation_scan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name_or_path": {"type": "string", "description": "SBML model name or path."},
                "species": {"type": "array", "items": {"type": "string"}, "description": "For a directly-specified ODE system."},
                "odes": {"type": "object", "description": "{species_name: rhs_expression_string} for a directly-specified system."},
                "parameters": {"type": "object", "description": "{parameter_name: default_value} for a directly-specified system."},
                "initial_conditions": {"type": "object", "description": "{species_name: value} for a directly-specified system."},
                "param_name": {"type": "string", "description": "Optional. Bifurcation parameter to sweep. If omitted, the tool chooses one automatically."},
                "param_min": {"type": "number", "description": "Optional sweep range minimum, only when the task gives a range: then it is used exactly. If omitted: the parameter's value times 10^-3 .. 10^3, continued in log10 (log x axis)."},
                "param_max": {"type": "number", "description": "Optional sweep range maximum, only when the task gives a range: then it is used exactly."},
                "model_range": {"type": "boolean", "description": "Optional. true only when the task asks for the model file's own range (XPP @ parmin/parmax): used exactly instead of the log window. Default false."},
                "fixed_params": {"type": "object", "description": "Optional override of any parameter's value for this analysis. A value for param_name itself is where the continuation starts (e.g. inside a window where another branch exists; give initial_conditions near that branch's steady state)."},
                "plot_state": {"type": "string", "description": "Which state variable to plot on the y-axis. Defaults to the species that changes most along the branch."},
                "n_pts": {"type": "integer", "description": "Target number of continuation points. Default 500."},
                "ds": {"type": "number", "description": "Optional AUTO initial arclength step override."},
                "dsmin": {"type": "number", "description": "Optional AUTO minimum arclength step override."},
                "dsmax": {"type": "number", "description": "Optional AUTO maximum arclength step override."},
                "nmx": {"type": "integer", "description": "Optional AUTO maximum continuation steps."},
                "itmx": {"type": "integer", "description": "Optional AUTO maximum iterations."},
                "itnw": {"type": "integer", "description": "Optional AUTO Newton iterations."},
                "nwtn": {"type": "integer", "description": "Optional AUTO Newton tolerance iterations."},
                "epsl": {"type": "number", "description": "Optional AUTO EPSL override."},
                "epsu": {"type": "number", "description": "Optional AUTO EPSU override."},
                "epss": {"type": "number", "description": "Optional AUTO EPSS override."},
                "cycles": {"type": "boolean", "description": "Optional. The limit cycles born at Hopf points are continued by default (period, amplitude and stability along the parameter). false: skip them, only when the task says so."},
                "cycle_folds": {"type": "boolean", "description": "Optional, with cycles=true. true: also report folds of cycles (saddle-node of limit cycles, where a cycle branch turns back). Only when the task asks for folds of cycles; default false."},
                "max_parameter_trials": {"type": "integer", "description": "Optional maximum number of continuation parameters to try when param_name is omitted. Defaults to 5 for bounded automatic parameter exploration; explicit param_name always runs once."},
                "show_auto_log": {"type": "boolean", "description": "Include a compact XPP/AUTO-style terminal continuation log in the result. Defaults to true."},
            },
            "required": [],
        },
    },
]

TOOL_FUNCTIONS = {
    "run_bifurcation_analysis": lambda **kwargs: run_bifurcation_analysis(**kwargs),
}
