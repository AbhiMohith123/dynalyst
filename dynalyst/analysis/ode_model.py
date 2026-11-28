"""
Shared symbolic ODE model layer.

Every analysis tool in this project (steady-state finding, stability
classification, AUTO-07p bifurcation continuation, parameter scans, the
ODE<->SBML converters) operates on one common representation, built here:

    {
      "label": str,                     # model name / stem, for filenames & messages
      "state_names": [str, ...],
      "state_symbols": [sp.Symbol, ...],
      "param_names": [str, ...],
      "param_symbols": [sp.Symbol, ...],
      "param_defaults": {name: float},
      "rhs": [sp.Expr, ...],            # dstate_i/dt, same order as state_names
      "y0": {name: float},
      "source": "sbml" | "user_odes",
      "source_path": str | None,
    }

Two ways to build one:
  1. `model_from_sbml(model_name_or_path)` -- any SBML file, any dimension,
     via generic_sbml.build_symbolic_model (libsbml -> sympy, no hand
     transcription).
  2. `model_from_odes(species, odes, parameters=None, initial_conditions=None)`
     -- user types the ODEs directly (e.g. "answer a query about a system I
     just typed in" or as the target of odes_to_sbml). `odes` is a
     dict/list of RHS strings parsed with sympy.sympify; any symbol not in
     `species` is treated as a free parameter (given a default from
     `parameters`, or 1.0 if unspecified).

`resolve_model` is the single entry point every tool should call: pass an
SBML model_name_or_path OR the (species, odes) pair, never both.
"""
import re
from pathlib import Path

import sympy as sp

from dynalyst import config
from dynalyst.analysis import generic_sbml


def model_from_sbml(model_name_or_path: str, allow_delay: bool = False) -> dict:
    path = generic_sbml.resolve_sbml_path(model_name_or_path)
    m = generic_sbml.build_symbolic_model(path, allow_delay=allow_delay)
    return {
        "label": path.stem,
        "state_names": m["state_names"],
        "state_symbols": m["state_symbols"],
        "param_names": m["param_names"],
        "param_symbols": m["param_symbols"],
        "param_defaults": m["param_defaults"],
        "param_aliases": m.get("param_aliases", {}),
        "param_kinds": m.get("param_kinds", {}),
        "caveats": m.get("caveats", []),
        "rhs": m["rhs"],
        "y0": m["y0"],
        "source": "sbml",
        "source_path": str(path),
        # time-dependent inputs frozen into constants (generic_sbml.freeze_time_input)
        "time_inputs": m.get("time_inputs", []),
        "display": m.get("display"),     # the readable form, see ode_report
        **({"delays": m["delays"]} if m.get("delays") else {}),   # allow_delay only
    }


def canonical_param(model: dict, name: str | None) -> str | None:
    """The model's own name for a parameter the user names: itself, or the one
    reaction-local parameter a bare name stands for (k1 -> 'R3_k1'). None when
    the model has no such parameter."""
    if not name:
        return None
    if name in model.get("param_defaults", {}):
        return name
    return (model.get("param_aliases") or {}).get(name)


def eligible_parameters(model: dict) -> list[str]:
    """Parameters that occur in the model RHS and can affect its dynamics."""
    if model.get("conserved_totals"):
        from dynalyst.analysis.steady_states import conservation_reduce
        model = conservation_reduce(model)
    used = {
        str(symbol)
        for expr in model.get("rhs", [])
        for symbol in sp.sympify(expr).free_symbols
    }
    return [name for name in model.get("param_names", []) if name in used]


def parameter_listing(model: dict, limit: int = 60) -> dict:
    """The model's parameters, grouped by kind, for error messages and the agent."""
    kinds = model.get("param_kinds") or {}
    out = {}
    for p in model.get("param_names", []):
        out.setdefault(kinds.get(p, "parameter"), []).append(p)
    return {k: (v if len(v) <= limit else v[:limit] + [f"... {len(v) - limit} more"])
            for k, v in out.items()}


def unknown_parameter_error(model: dict, name: str) -> dict:
    if not model.get("param_names"):
        return {"error": f"'{name}' is not a parameter of this model, and the model exposes "
                         "no tunable quantity at all (no parameters, no fixed species, no "
                         "reaction-local constants), so there is nothing to continue in.",
                "available_parameters": {}}
    return {"error": f"'{name}' is not a parameter of this model (a species that changes over "
                     "time cannot be a continuation parameter).",
            "available_parameters": parameter_listing(model)}


def continuation_parameter_error(model: dict, name: str) -> dict | None:
    """Validate a free parameter's role, without predicting a bifurcation.

    Rate constants and exposed conserved totals are both eligible. An observable-only
    constant cannot change the ODE dynamics; a dynamic species is not a free parameter.
    """
    canonical = canonical_param(model, name)
    if canonical is None:
        return unknown_parameter_error(model, name)
    eligible = eligible_parameters(model)
    if canonical not in eligible:
        return {
            "error": f"'{name}' is a model parameter, but it does not occur in the ODE "
                     "right-hand sides. Varying it cannot change the model's dynamics.",
            "available_parameters": eligible,
            "hint": "Choose a free parameter that affects the equations. Kinetic constants "
                    "and explicitly parameterized conserved totals can both qualify; "
                    "neither guarantees a bifurcation in a given range.",
        }
    return None


NO_PARAMETERS = ("This model exposes no tunable quantity (no parameters, no fixed species, no "
                 "reaction-local constants), so it has no continuation parameter.")


XPP_SUFFIXES = (".ode", ".txt")


def _xpp_by_stem(name: str) -> Path | None:
    """An XPP file in the model folders whose name without its extension is `name`."""
    stem = Path(str(name)).name.lower()
    found = sorted((p for d in config.SBML_SEARCH_DIRS if d.exists()
                    for suffix in XPP_SUFFIXES for p in d.rglob(f"*{suffix}")
                    if p.is_file() and p.stem.lower() == stem),
                   key=lambda p: (len(str(p)), str(p)))
    return found[0] if found else None


def xpp_file(model_name_or_path) -> Path | None:
    """The XPP file a name or path refers to, or None (an SBML file, or no file)."""
    try:
        path = resolve_model_file(str(model_name_or_path))
    except (FileNotFoundError, ValueError):
        return None
    return path if path.suffix.lower() in XPP_SUFFIXES else None


def resolve_model_file(model_name_or_path: str) -> Path:
    """The model file a name or path refers to: an XPP file (.ode / .txt) by its own
    lookup, else an SBML file; a name without an extension that matches no SBML file
    is also looked for as an XPP file. Raises the SBML lookup's error when neither exists."""
    name = str(model_name_or_path)
    if Path(name).suffix.lower() in XPP_SUFFIXES:
        from dynalyst.analysis.xpp_to_sbml import _resolve_ode_path
        return _resolve_ode_path(name)
    try:
        return generic_sbml.resolve_sbml_path(name)
    except FileNotFoundError:
        if not Path(name).suffix:
            xpp = _xpp_by_stem(name)
            if xpp is not None:
                return xpp
        raise


def model_from_xpp(path_or_name: str) -> dict:
    """An XPP .ode/.txt file, parsed with the same parser xpp_to_sbml uses.
    Parameters the equations never use are kept aside as 'unused_parameters'."""
    from dynalyst.analysis.xpp_to_sbml import _resolve_ode_path, parse_xpp
    path = _resolve_ode_path(path_or_name)
    species, odes, parameters, initials = parse_xpp(path.read_text(errors="replace"))
    if not species:
        raise ValueError(f"{path.name} has no derivative (NAME'=...) lines: not an XPP model.")
    used = set()
    for s_ in species:
        names = {n: sp.Symbol(n) for n in [*species, *parameters]}
        used |= {str(x) for x in sp.sympify(odes[s_], locals=names).free_symbols}
    m = model_from_odes(species, odes, {k: v for k, v in parameters.items() if k in used},
                        {k: v for k, v in initials.items() if k in species}, label=path.stem)
    m.update(source="xpp", source_path=str(path),
             unused_parameters={k: v for k, v in parameters.items() if k not in used})
    return m


_TIME_CSYMBOL = "sbml.org/sbml/symbols/time"
_NOTE_CACHE = {}


def time_note_for(model_name_or_path: str) -> str | None:
    """time_inputs_note for an SBML model given by name or path, or None when
    it is not an SBML file or does not use time (a quick text check first, so
    ordinary models are not parsed again)."""
    try:
        path = generic_sbml.resolve_sbml_path(model_name_or_path)
    except (FileNotFoundError, ValueError):
        return None
    if path.suffix.lower() != ".xml":
        return None
    key = (str(path.resolve()), path.stat().st_mtime)
    if key not in _NOTE_CACHE:
        text = path.read_text(errors="replace")
        _NOTE_CACHE[key] = time_inputs_note(model_from_sbml(str(path))) \
            if _TIME_CSYMBOL in text else None
    return _NOTE_CACHE[key]


def time_inputs_note(model: dict) -> str | None:
    """One sentence on the time-dependent inputs a model's analysis froze, or None."""
    items = model.get("time_inputs") or []
    if not items:
        return None
    parts = []
    for t in items:
        if t["value"] is None:
            parts.append(f"{t['input']} (uses time; taken at {t['basis']})")
        else:
            lt = t.get("long_time_expression")
            parts.append(f"{t['input']} = {t['expression']} frozen at {t['value']:.6g}, "
                         f"{t['basis']}" + (f" = {lt}" if lt and 'long-time' in t['basis']
                                            and any(c.isalpha() for c in lt) else ""))
    return ("This model is time-dependent: its inputs were frozen into constant parameters so it "
            "has steady states -- " + "; ".join(parts) + ". Each frozen input is an ordinary "
            "parameter now: sweep it, sample it or set its value to study other stimulus levels.")


# "dX/dt", "d X / d t", "d[X]/dt", "d(X)/dt" and "X'" all name the state X
_DERIVATIVE_KEY = re.compile(
    r"^\s*(?:d\s*[\[(]?\s*(?P<a>[A-Za-z_]\w*)\s*[\])]?\s*/\s*d\s*t"
    r"|(?P<b>[A-Za-z_]\w*)\s*')\s*$")


def normalise_odes(species: list[str] | None, odes: dict | list) -> tuple[list[str], dict | list]:
    """(species, odes) with derivative-style keys ("dX/dt", "X'") turned into
    plain state names. When `species` is not given and `odes` is a dict, the
    states are its keys, in the order written."""
    if isinstance(odes, dict):
        declared = set(species or [])
        renamed = {}
        for key, rhs in odes.items():
            name = str(key).strip()
            m = None if name in declared else _DERIVATIVE_KEY.match(name)
            if m:
                name = m.group("a") or m.group("b")
            if name in renamed:
                raise ValueError(f"Two right-hand sides were given for {name!r}.")
            renamed[name] = rhs
        odes = renamed
        if not species:
            species = list(odes)
    if not species:
        raise ValueError("Give the state names (species), or the odes as a "
                         "{state: right-hand side} mapping.")
    return [str(s).strip() for s in species], odes


def model_from_odes(species: list[str] | None, odes: dict | list[str],
                     parameters: dict | None = None,
                     initial_conditions: dict | None = None,
                     label: str = "user_model") -> dict:
    """
    species: ordered list of state variable names, e.g. ["x", "y"].
    odes: either a dict {species_name: rhs_string} or a list of rhs strings
      in the same order as `species`, e.g. {"x": "a - (b+1)*x + x**2*y"}.
    parameters: optional {name: default_value} for any symbol appearing in
      the ODEs that is not itself a species. Symbols not listed here default
      to 1.0 (still returned in param_defaults so the caller can see this).
    initial_conditions: optional {species_name: value}; defaults to 0.0.
    Keys written as derivatives ("dx/dt", "x'") are accepted, and `species`
    may be omitted when `odes` is a dict (see normalise_odes).
    """
    species, odes = normalise_odes(species, odes)
    state_symbols = [sp.Symbol(s) for s in species]
    local_dict = dict(zip(species, state_symbols))
    # declared parameters are plain symbols too (Ge, S, E, N... are sympy objects otherwise)
    local_dict.update({p: sp.Symbol(p) for p in (parameters or {}) if p not in local_dict})

    if isinstance(odes, dict):
        missing = [s for s in species if s not in odes]
        if missing:
            raise ValueError(f"No RHS provided for species {missing}.")
        rhs_strings = [odes[s] for s in species]
    else:
        if len(odes) != len(species):
            raise ValueError(
                f"Got {len(odes)} ODE(s) for {len(species)} species -- "
                f"lengths must match when `odes` is a list."
            )
        rhs_strings = list(odes)

    rhs = []
    all_free = set()
    for expr_str in rhs_strings:
        expr = sp.sympify(expr_str, locals=local_dict)
        if not isinstance(expr, sp.Expr) or not expr.is_commutative:
            raise ValueError(f"Each ODE right-hand side must be a scalar expression: {expr_str!r}")
        rhs.append(expr)
        all_free |= expr.free_symbols

    param_symbols = sorted((s for s in all_free if s not in state_symbols), key=str)
    param_names = [str(s) for s in param_symbols]

    parameters = parameters or {}
    unknown_params = [k for k in parameters if k not in param_names]
    if unknown_params:
        raise ValueError(
            f"parameters {unknown_params} don't appear in any ODE. "
            f"Free parameters actually used: {param_names}"
        )
    param_defaults = {p: float(parameters.get(p, 1.0)) for p in param_names}

    initial_conditions = initial_conditions or {}
    unknown_ic = [k for k in initial_conditions if k not in species]
    if unknown_ic:
        raise ValueError(f"initial_conditions {unknown_ic} are not in species list {species}.")
    y0 = {s: float(initial_conditions.get(s, 0.0)) for s in species}

    return {
        "label": label,
        "state_names": list(species),
        "state_symbols": state_symbols,
        "param_names": param_names,
        "param_symbols": param_symbols,
        "param_defaults": param_defaults,
        "rhs": rhs,
        "y0": y0,
        "source": "user_odes",
        "source_path": None,
    }


def relative_value(given, model_value) -> float:
    """A value as given, or relative to the model's own: '*10' (or 'x10') is ten times the
    model's value, '/2' half of it. Lets a request like '10 times more X' be passed exactly,
    without anyone computing the number."""
    if isinstance(given, str):
        m = re.fullmatch(r"\s*([*x×/])\s*([0-9.]+(?:[eE][+-]?\d+)?)\s*", given)
        if m:
            factor = float(m.group(2))
            return float(model_value) / factor if m.group(1) == "/" else float(model_value) * factor
    return float(given)


def with_overrides(model: dict, parameters: dict | None = None,
                   initial_conditions: dict | None = None) -> dict:
    """A file's model with the caller's parameter values and starting state in place of
    its own (for this run only; the file is not changed). Unknown names are an error,
    never silently ignored."""
    if initial_conditions:
        unknown = [k for k in initial_conditions if k not in model["y0"]]
        if unknown:
            raise ValueError(f"Unknown species {unknown} in initial_conditions. "
                             f"Species: {list(model['y0'])}")
        model = {**model, "y0": {**model["y0"], **{
            k: relative_value(v, model["y0"][k]) for k, v in initial_conditions.items()}}}
    if parameters:
        named = {k: canonical_param(model, k) for k in parameters}
        unknown = [k for k, c in named.items() if c is None]
        if unknown:
            raise ValueError(f"Unknown parameter(s) {unknown}. "
                             f"Parameters: {list(model['param_defaults'])}")
        model = {**model, "param_defaults": {**model["param_defaults"], **{
            named[k]: relative_value(v, model["param_defaults"][named[k]])
            for k, v in parameters.items()}}}
    return model


def resolve_model(model_name_or_path: str | None = None,
                   species: list[str] | None = None,
                   odes: dict | list | None = None,
                   parameters: dict | None = None,
                   initial_conditions: dict | None = None,
                   allow_delay: bool = False) -> dict:
    """Single dispatcher: pass model_name_or_path for an SBML file, OR
    species+odes for a directly-specified ODE system. allow_delay: a delay
    differential equation model is returned as written (for reading it only)."""
    if model_name_or_path and (species or odes):
        raise ValueError("Pass either model_name_or_path OR (species, odes), not both.")
    if model_name_or_path:
        path = resolve_model_file(model_name_or_path)
        if path.suffix.lower() in XPP_SUFFIXES:
            model = model_from_xpp(str(path))
        else:
            model = model_from_sbml(str(path), allow_delay=allow_delay)
        return with_overrides(model, parameters, initial_conditions)
    if odes and (species or isinstance(odes, dict)):
        return model_from_odes(species, odes, parameters, initial_conditions)
    raise ValueError(
        "Provide either model_name_or_path (an SBML file) or odes (a "
        "directly-specified ODE system: a {state: right-hand side} mapping, or a "
        "list of right-hand sides together with species)."
    )


def ode_strings(model: dict) -> dict:
    """{'dX/dt': 'formatted rhs string'} for display / tool JSON output."""
    return {
        f"d{name}/dt": sp.sstr(expr)
        for name, expr in zip(model["state_names"], model["rhs"])
    }


def jacobian(model: dict) -> sp.Matrix:
    return sp.Matrix(model["rhs"]).jacobian(model["state_symbols"])


# --- Tool: extract/derive the ODEs for any model (SBML, arbitrary dimension,
#     OR a directly-specified system) --------------------------------------
import json


def tidy(expr) -> str:
    """An expression as text without float clutter: 1.0*x -> x, 2.0 -> 2."""
    expr = sp.sympify(expr)
    whole = {f: sp.Integer(int(f)) for f in expr.atoms(sp.Float) if float(f).is_integer()}
    return sp.sstr(expr.xreplace(whole) if whole else expr)


READABLE_MAX_OPS = 400        # larger right-hand sides are shown as written (checked before
READABLE_MAX_TERMS = 200      # expanding, which can blow up), as are larger expanded ones


def _negatives(f) -> int:
    return sum(t.could_extract_minus_sign() for t in sp.Add.make_args(f))


def _sum_text(f) -> str:
    """A sum with its positive terms first: T - A - B."""
    terms = sorted(sp.Add.make_args(f), key=lambda t: t.could_extract_minus_sign())
    text = ""
    for t in terms:
        neg = t.could_extract_minus_sign()
        body = sp.sstr(-t if neg else t)
        text += (" - " if neg else " + ") + body if text else ("-" if neg else "") + body
    return text


def _term_text(term) -> tuple[bool, str]:
    """(negative, text) of one product, its sums written for reading: a sum of only negative
    terms has its sign taken out (-(k33 + k4)), and a 'total minus parts' sum
    (A + B - T) is turned round to (T - A - B), the term's sign following."""
    coeff, factors = term.as_coeff_mul()
    neg = bool(coeff < 0)
    sums = [f for f in factors if isinstance(f, sp.Add)]
    flip = {f for f in sums if _negatives(f) == len(f.args)}      # all negative
    neg ^= len(flip) % 2 == 1
    one = [f for f in sums if f not in flip and _negatives(f) == 1]
    flip |= set(one)
    neg ^= len(one) % 2 == 1
    parts = [] if abs(coeff) == 1 else [sp.sstr(abs(coeff))]
    for f in factors:
        if isinstance(f, sp.Add):
            parts.append(f"({_sum_text(-f if f in flip else f)})")
        elif isinstance(f, sp.Pow) and isinstance(f.base, sp.Add):
            parts.append(f"({_sum_text(f.base)})**{sp.sstr(f.exp)}")
        else:
            parts.append(sp.sstr(f))
    return neg, "*".join(parts) or "1"


def readable(expr, state_symbols, param_symbols) -> str | None:
    """A right-hand side regrouped for reading: the terms of the rate constant that
    multiplies the most of them are summed and factored, and so on; then one state monomial
    under several rate constants is collected (C*(k1 + k2)). Only for a polynomial in the
    states (a file may store a whole ODE fully expanded) of limited size; None otherwise, or
    when the text would not read back as exactly the same expression (checked)."""
    expr = sp.sympify(expr)
    whole = {f: sp.Integer(int(f)) for f in expr.atoms(sp.Float) if float(f).is_integer()}
    expr = expr.xreplace(whole) if whole else expr
    try:
        if sp.count_ops(expr) > READABLE_MAX_OPS or not expr.is_polynomial(*state_symbols):
            return None
        expanded = sp.expand(expr)
        if len(sp.Add.make_args(expanded)) > READABLE_MAX_TERMS:
            return None
        # greedy: the parameter that multiplies the most remaining terms takes them all,
        # and their sum is factored (k*(T - A - B)*(S - A - ...))
        left, terms = list(sp.Add.make_args(expanded)), []
        while left:
            counts = {p: sum(t.has(p) for t in left) for p in param_symbols}
            p = max(counts, key=counts.get) if counts else None
            if p is None or counts[p] < 2:
                terms += left
                break
            group = [t for t in left if t.has(p)]
            left = [t for t in left if not t.has(p)]
            terms.append(sp.factor(sp.Add(*group)))
        # one state monomial under several rate constants: C*k1 + C*k2 -> C*(k1 + k2)
        by_state, kept = {}, []
        for t in terms:
            rate, state_part = t.as_independent(*state_symbols, as_Add=False)
            if state_part != 1 and not state_part.has(sp.Add) and not rate.has(*state_symbols):
                by_state.setdefault(state_part, []).append(rate)
            else:
                kept.append(t)
        kept += [s * (r[0] if len(r) == 1 else sp.factor(sp.Add(*r))) for s, r in by_state.items()]
        kept.sort(key=lambda t: _term_text(t)[0])                 # gains before losses
        text = ""
        for t in kept:
            neg, body = _term_text(t)
            text += (" - " if neg else " + ") + body if text else ("-" if neg else "") + body
        names = {str(s): s for s in [*state_symbols, *param_symbols]}
        if sp.expand(sp.sympify(text, locals=names) - expanded) != 0:
            return None
    except (sp.PolynomialError, sp.SympifyError, TypeError, ValueError, RecursionError):
        return None
    # only a real regrouping (fewer terms than written), not a reordering of the same terms
    return text if len(kept) < len(sp.Add.make_args(expr)) else None


def _ode_encoded(reactions, rhs_by_state) -> bool:
    """A file that writes each ODE as a 'reaction' (nothing -> X, rate = the whole right-hand
    side, which has loss terms) rather than as a reaction network."""
    if not reactions:
        return False
    products = [r["products"][0][0] for r in reactions
                if not r["reactants"] and len(r["products"]) == 1]
    return (len(products) == len(reactions) == len(set(products))
            and all(any(t.could_extract_minus_sign() for t in sp.Add.make_args(sp.sympify(r["rate"])))
                    for r in reactions))


def _side(items) -> str:
    return " + ".join((f"{int(n) if float(n).is_integer() else n} {s}" if n != 1 else s)
                      for s, n in items) or "(nothing)"


def ode_report(model: dict) -> dict:
    """Everything needed to read a model: its ODEs as written (fixed species
    and compartments by name, each reaction's rate visible), for an SBML model
    also each reaction with its rate law and the ODEs as sums of reaction
    rates, the assignment rules, and the value of every parameter, constant and
    initial condition the equations use (parameters they do not use listed
    apart). Works for SBML (incl. crnt-converted SBML), XPP and typed ODEs."""
    names = model["state_names"]
    disp = model.get("display") or {}
    rhs = disp.get("rhs") or model["rhs"]
    out = {"model": model["label"], "source": model["source"],
           "source_path": model.get("source_path"), "state_variables": names,
           "n_state_variables": len(names),
           "odes": {f"d{s}/dt": tidy(e) for s, e in zip(names, rhs)}}
    used = set()
    for e in rhs:
        used |= {str(x) for x in sp.sympify(e).free_symbols}
    # the same equations regrouped for reading, when the file stores them expanded
    nice = {f"d{s}/dt": readable(e, model["state_symbols"], model.get("param_symbols", []))
                        or out["odes"][f"d{s}/dt"] for s, e in zip(names, rhs)}
    if nice != out["odes"]:
        out["odes_as_written"], out["odes"] = out["odes"], nice
        out["odes_note"] = ("'odes' are the file's equations with terms grouped by rate constant "
                            "and factored (exactly equal, checked); 'odes_as_written' as stored.")

    reactions = disp.get("reactions") or []
    if _ode_encoded(reactions, rhs):
        out["reactions_note"] = ("The file writes each ODE as a 'reaction' that produces its "
                                 "species at the whole right-hand side, not as a reaction "
                                 "network, so no reactions are listed: the ODEs above are the model.")
        reactions = []
    if reactions:
        out["reactions"] = []
        for r in reactions:
            arrow = " <-> " if r.get("reversible") else " -> "
            item = {"id": r["id"],
                    "equation": _side(r["reactants"]) + arrow + _side(r["products"]),
                    "rate": tidy(r["rate"])}
            if r.get("name") and r["name"] != r["id"]:
                item["name"] = r["name"]
            if r.get("local_parameters"):
                item["local_parameters"] = r["local_parameters"]
            out["reactions"].append(item)
        flux = {s: [] for s in names}
        for r in reactions:
            for s_, n in r["reactants"]:
                if s_ in flux:
                    flux[s_].append((-n, r["id"]))
            for s_, n in r["products"]:
                if s_ in flux:
                    flux[s_].append((n, r["id"]))
        as_rates = {}
        for s_ in names:
            if s_ in (disp.get("rate_rules") or {}):
                as_rates[f"d{s_}/dt"] = tidy(disp["rate_rules"][s_])
                continue
            terms = []
            for n, rid in flux[s_]:
                mag = abs(n)
                coef = "" if mag == 1 else f"{int(mag) if float(mag).is_integer() else mag}*"
                terms.append(("- " if n < 0 else "+ ") + f"{coef}v_{rid}")
            text = " ".join(terms).lstrip("+ ").strip()
            as_rates[f"d{s_}/dt"] = (text if not text.startswith("- ") else "-" + text[2:]) or "0"
        out["odes_as_reaction_rates"] = as_rates

    rules = disp.get("assignment_rules") or {}
    if rules:
        out["assignment_rules"] = {v: tidy(e) for v, e in rules.items()}
        for e in rules.values():
            used |= {str(x) for x in sp.sympify(e).free_symbols}
    for r in reactions:
        used |= {str(x) for x in sp.sympify(r["rate"]).free_symbols}

    params = model["param_defaults"]
    out["parameters"] = {p: params[p] for p in model["param_names"] if p in used}
    unused = {p: params[p] for p in model["param_names"] if p not in used}
    unused.update(model.get("unused_parameters") or {})
    if unused:
        out["unused_parameters"] = unused
    kinds = model.get("param_kinds") or {}
    fixed = {p: v for p, v in out["parameters"].items() if kinds.get(p) == "fixed species"}
    if fixed:           # clamped species (enzymes, sources): parameters, but named as species
        out["fixed_species"] = fixed
    local = [p for p in out["parameters"] if kinds.get(p) == "local parameter"]
    if local:
        out["local_parameters"] = (f"{len(local)} constants local to a reaction, named "
                                   "<reaction>_<name> (a bare name works when it is unique)")
    titles = {p: t for p, t in (disp.get("param_titles") or {}).items() if p in out["parameters"]}
    if titles:
        out["parameter_meanings"] = titles
    constants = {c: v for c, v in (disp.get("constants") or {}).items() if c in used}
    if constants:
        out["constants"] = constants
    out["initial_conditions"] = model["y0"]
    note = time_inputs_note(model)
    if note:
        out["time_dependent_inputs"] = note
    if model.get("caveats"):
        out["caveats"] = model["caveats"]
    if model.get("delays"):
        out["delay_differential_equation"] = {
            "delayed_terms": {d["term"]: f"{d['variable']} at time t - ({d['delay']})"
                              for d in model["delays"]},
            "note": ("These are delay differential equations, not ODEs: delay(x, tau) is the "
                     "value x had tau time units earlier, x(t - tau). The equations are shown "
                     "exactly as the model defines them. The state at any time is the whole "
                     "history over the last tau, so the system is infinite-dimensional even "
                     "though it has n_state_variables variables; a reduction to fewer ODEs "
                     "does not apply. " + generic_sbml.DELAY_MESSAGE[0].upper()
                     + generic_sbml.DELAY_MESSAGE[1:]),
        }
    return out


def get_odes(model_name_or_path: str | None = None,
             species: list[str] | None = None,
             odes: dict | None = None,
             parameters: dict | None = None,
             initial_conditions: dict | None = None) -> str:
    """
    Tool: return the symbolic ODE system for a model -- works for an SBML
    file of ANY dimension (parsed generically via libsbml, no hand
    transcription, no 1D restriction) or for a system of ODEs given directly.
    """
    try:
        model = resolve_model(model_name_or_path, species, odes, parameters, initial_conditions,
                              allow_delay=True)
    except (FileNotFoundError, ValueError, KeyError, NotImplementedError) as e:
        return json.dumps({"error": str(e)})
    return json.dumps(ode_report(model), indent=2, default=float)


TOOL_DEFINITIONS = [
    {
        "name": "get_odes",
        "description": (
            "A model's ODEs and parameter values, read from the file: the ODEs as written "
            "(fixed species and compartments by name), for SBML also each reaction with its "
            "rate law and the ODEs as sums of reaction rates (v_<id>), the assignment rules, "
            "and the value of every parameter / constant / initial condition the equations "
            "use (unused parameters listed apart; a crnt-converted file says what each "
            "k_reN stands for). Works for an SBML file of ANY dimension (including the "
            "crnt-converted SBML under outputs/crnt/), an XPP .ode/.txt file, or ODEs the user "
            "specifies directly (species names + one RHS string per species, e.g. "
            "odes={'x':'a-(b+1)*x+x**2*y','y':'b*x-x**2*y'}). Use this whenever the "
            "user asks for 'the ODEs', 'the equations', or the structure of a model -- "
            "including as the first step before finding steady states or running a "
            "bifurcation analysis, so the actual free parameters are known rather "
            "than guessed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name_or_path": {
                    "type": "string",
                    "description": "SBML model name or path. Omit if supplying species+odes directly instead.",
                },
                "species": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Ordered list of state-variable names, for a directly-specified ODE system.",
                },
                "odes": {
                    "type": "object",
                    "description": "{species_name: rhs_expression_string}, e.g. {'x': 'a - x**2'}. Required together with species if not using an SBML model.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Optional {parameter_name: default_value} for a directly-specified system.",
                },
                "initial_conditions": {
                    "type": "object",
                    "description": "Optional {species_name: value} for a directly-specified system. Defaults to 0.0.",
                },
            },
            "required": [],
        },
    },
]

TOOL_FUNCTIONS = {
    "get_odes": lambda **kwargs: get_odes(**kwargs),
}
