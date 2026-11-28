"""Equations typed in a request, read by code.

A user may write a model straight into the message ("dX/dt = a - b*X with a = 0.05,
b = 0.45 and X(0) = 0"). If every role of the agent re-typed it into tool arguments,
each could transcribe it differently (a value written into the equation, a parameter
taken for a species). Instead this module reads it once, deterministically:

  derivatives    dX/dt = ..., d X / d t = ..., d[X]/dt = ..., d(X)/dt = ..., X' = ...
  values         name = number           (a parameter, or a species' initial value)
  initial values X(0) = number, X0 = number when X is a state

and the agent saves it once as a model file (save_typed_system) that every step uses.
When the text cannot be read unambiguously, parse_typed_system returns None and the
request is handled as before. Nothing is guessed: parameters the text gives no value
are reported, not filled in.
"""
import contextlib
import hashlib
import io
import json
import re

import sympy as sp

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_DERIV = re.compile(
    rf"(?<![\w.])(?:d\s*[\[(]?\s*(?P<a>{_NAME})\s*[\])]?\s*/\s*d\s*t|(?P<b>{_NAME})\s*')\s*=")
# a value may end a sentence ("I(0) = 0.01."): a full stop after it is not part of a name
_INIT = re.compile(rf"(?<![\w.])(?P<n>{_NAME})\s*\(\s*0\s*\)\s*=\s*(?P<v>{_NUM})(?![\w(]|\.\w)")
_VALUE = re.compile(rf"(?<![\w.(])(?P<n>{_NAME})\s*=\s*(?P<v>{_NUM})(?![\w(]|\.\w)")
# where an equation's right-hand side ends: the next definition, a sentence or clause end
_END = re.compile(r";|\n|\.\s|\.$|[?!]|,\s*(?:parameters?|with|where|and|for|using|in)\b|\s(?:with|where|for)\s"
                  # "... in mu from -1 to 1", "... over k between 0 and 2": a range, not the equation
                  r"|\s(?:in|over)\s+[A-Za-z_]\w*\s+(?:from|between)\b")
_TRAILING = re.compile(r"[\s,]*(?:\band\b|\bwith\b|\bwhere\b)?[\s,.:?!]*$", re.I)
_FUNCS = {f: getattr(sp, f) for f in ("exp", "log", "sqrt", "sin", "cos", "tanh", "Abs",
                                      "Max", "Min", "Heaviside")}


def _rhs_end(text, start, stops):
    """End of a right-hand side starting at `start`: the first stop (another definition)
    or clause boundary after it, outside parentheses."""
    depth, i = 0, start
    nxt = min([s for s in stops if s > start] + [len(text)])
    while i < nxt:
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0:
            m = _END.match(text, i)
            if m:
                return i
        i += 1
    return nxt


_RANGE = re.compile(rf"(?<![\w.])(?P<n>{_NAME})\s+(?:from|between)\s+(?P<a>{_NUM})\s+(?:to|and)\s+"
                    rf"(?P<b>{_NUM})(?![\w(]|\.\w)")
# one value given to several names at once: "x(0) = y(0) = 1", "a = b = 0.5"
_TARGET = rf"{_NAME}\s*(?:\(\s*0\s*\))?"
_CHAIN = re.compile(rf"(?<![\w.(/])((?:{_TARGET}\s*=\s*){{2,}})(?P<v>{_NUM})(?![\w(]|\.\w)")


def _expand_chains(text: str) -> str:
    """'x(0) = y(0) = 1' written as 'x(0) = 1, y(0) = 1', so each name gets the value."""
    def one(m):
        targets = re.findall(rf"({_TARGET})\s*=", m.group(1))
        return ", ".join(f"{t.strip()} = {m.group('v')}" for t in targets)
    return _CHAIN.sub(one, text)


def parse_typed_system(text: str) -> dict | None:
    """{species, odes, parameters, initial_conditions, unvalued} read from the text, or
    None when it holds no complete, unambiguous system of equations."""
    text = _expand_chains(text or "")
    derivs = list(_DERIV.finditer(text))
    if not derivs:
        return None
    defs = sorted({m.start() for m in derivs} | {m.start() for m in _INIT.finditer(text)}
                  | {m.start() for m in _VALUE.finditer(text)})
    odes, spans = {}, []
    for m in derivs:
        name = m.group("a") or m.group("b")
        end = _rhs_end(text, m.end(), defs)
        rhs = _TRAILING.sub("", text[m.end():end]).strip()
        if not rhs or name in odes:
            return None                       # empty, or two equations for one state
        odes[name] = rhs.replace("^", "**")
        spans.append((m.start(), end))
    species = list(odes)
    inside = lambda pos: any(a <= pos < b for a, b in spans)

    initial, parameters = {}, {}
    in_equations = {t for rhs in odes.values() for t in re.findall(_NAME, rhs)}
    for m in _INIT.finditer(text):
        if not inside(m.start()) and m.group("n") in odes:
            initial[m.group("n")] = float(m.group("v"))
    for m in _VALUE.finditer(text):
        if inside(m.start()):
            continue
        n, v = m.group("n"), float(m.group("v"))
        if n in odes:
            initial.setdefault(n, v)          # "X = 1" for a state: its starting value
        elif n.endswith("0") and n[:-1] in odes and n not in in_equations:
            initial.setdefault(n[:-1], v)     # "X0 = 1" (unless the equations use X0 itself)
        elif n in parameters and parameters[n] != v:
            return None                       # two different values for one name
        else:
            parameters[n] = v

    # a parameter given only as a range ("in mu from -1 to 1"): the parameter to vary, which
    # needs no value of its own -- it starts at the lower end of its range
    ranges = {}
    for m in _RANGE.finditer(text):
        lo, hi = sorted((float(m.group("a")), float(m.group("b"))))
        ranges.setdefault(m.group("n"), (lo, hi))
    for n, (lo, _hi) in ranges.items():
        if n not in parameters and n not in odes:
            parameters[n] = lo

    symbols = {n: sp.Symbol(n) for n in [*species, *parameters]}
    used = set()
    for name, rhs in odes.items():
        try:
            expr = sp.sympify(rhs, locals={**_FUNCS, **symbols})
        except Exception:
            return None                       # not an expression: let the models read it
        if not isinstance(expr, sp.Expr) or not expr.is_commutative:
            return None                       # tuples, matrices and relations are not scalar ODEs
        used |= {str(s) for s in expr.free_symbols}
    return {"species": species, "odes": odes,
            "parameters": {k: v for k, v in parameters.items() if k in used},
            "initial_conditions": {s: initial.get(s, 0.0) for s in species},
            "initial_given": sorted(initial),
            "unvalued": sorted(used - set(species) - set(parameters))}


def without_system(text: str) -> str:
    """The text with its typed equations and value assignments blanked out: what the user
    says ABOUT the system (which parameter to vary, which range), since the equations
    themselves mention every parameter."""
    text = text or ""
    derivs = list(_DERIV.finditer(text))
    defs = sorted({m.start() for m in derivs} | {m.start() for m in _INIT.finditer(text)}
                  | {m.start() for m in _VALUE.finditer(text)})
    spans = [(m.start(), _rhs_end(text, m.end(), defs)) for m in derivs]
    spans += [(m.start(), m.end()) for m in _INIT.finditer(text)]
    spans += [(m.start(), m.end()) for m in _VALUE.finditer(text)]
    out = list(text)
    for a, b in spans:
        out[a:b] = " " * (b - a)
    return "".join(out)


def typed_model_name(system: dict) -> str:
    """A file name from the system's content: the same equations and values always give
    the same name, different ones never overwrite each other."""
    key = json.dumps({k: system[k] for k in ("odes", "parameters", "initial_conditions")},
                     sort_keys=True)
    return "typed_" + hashlib.sha1(key.encode()).hexdigest()[:8]


TYPED_MARK = "dynalyst: written from equations typed by the user; its variables may take any sign."
_TYPED = {}


def is_typed_file(model_name_or_path) -> bool:
    """Was this model file written from typed equations (save_typed_system)? Its
    variables are not concentrations, so checks for negative values do not apply."""
    try:
        from dynalyst.analysis.generic_sbml import resolve_sbml_path
        path = resolve_sbml_path(str(model_name_or_path))
        mtime = path.stat().st_mtime
        if _TYPED.get(path, (None,))[0] != mtime:
            _TYPED[path] = (mtime, TYPED_MARK in path.read_text(encoding="utf-8", errors="replace"))
        return _TYPED[path][1]
    except Exception:
        return False


def save_typed_system(system: dict) -> dict:
    """Save the system once as an SBML model file (jaxkm_ode_to_sbml.odes_to_sbml) and
    confirm that the file gives back the same equations. Returns {"file", "name", ...}
    or {"error"}."""
    from dynalyst.analysis.jaxkm_ode_to_sbml import odes_to_sbml
    from dynalyst.analysis.ode_model import resolve_model
    name = typed_model_name(system)
    with contextlib.redirect_stdout(io.StringIO()):
        out = json.loads(odes_to_sbml(system["species"], system["odes"], system["parameters"],
                                      system["initial_conditions"], model_name=name))
    if out.get("error"):
        return {"error": out["error"]}
    try:     # mark the file: its variables may be negative (not concentrations)
        import libsbml
        doc = libsbml.readSBMLFromFile(out["path"])
        doc.getModel().setNotes('<body xmlns="http://www.w3.org/1999/xhtml"><p>'
                                + TYPED_MARK + "</p></body>")
        libsbml.writeSBMLToFile(doc, out["path"])
    except Exception:
        pass
    from dynalyst.analysis.steady_states import _exact_expr, _vanishes
    try:     # the file must reproduce the typed equations exactly (floats such as the 2.0 in
        # X**2.0 that the export writes are read as the exact numbers they denote)
        saved = resolve_model(out["path"])
        names = {n: sp.Symbol(n) for n in [*system["species"], *saved["param_names"]]}
        for s, rhs in zip(saved["state_names"], saved["rhs"]):
            typed = sp.sympify(system["odes"][s], locals={**_FUNCS, **names})
            read = sp.sympify(rhs, locals=names)
            if not (_vanishes(_exact_expr(read) - _exact_expr(typed))
                    or _agrees_to_machine_precision(read, typed)):
                return {"error": f"the saved file does not reproduce d{s}/dt"}
    except Exception as e:
        return {"error": f"the saved file could not be checked ({type(e).__name__}: {e})"}
    return {"file": f"{name}.xml", "name": name, "path": out["path"]}


def _agrees_to_machine_precision(a, b, n_points: int = 12) -> bool:
    """The file writes numbers with 15 significant digits (1/3 -> 0.333333333333333), so it
    reproduces a typed fraction only to machine precision: equal within 1e-12 (relative)
    at random points of every symbol."""
    import random
    names = sorted({str(x) for x in (a.free_symbols | b.free_symbols)})
    rng = random.Random(0)
    for _ in range(n_points):
        point = {sp.Symbol(n): rng.uniform(0.1, 3.0) for n in names}
        try:
            x, y = complex(a.subs(point).evalf()), complex(b.subs(point).evalf())
        except (TypeError, ValueError):
            return False
        if abs(x - y) > 1e-12 * max(1.0, abs(x), abs(y)):
            return False
    return True
