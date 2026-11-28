"""
XPPAUT .ode file -> SBML, wired into the SAME pipeline odes_to_sbml() already
uses -- this module does NOT re-implement export; it only does the piece
that was missing: turning XPP's plain-text ODE syntax into the
(species, odes, parameters) triple that odes_to_sbml() / model_from_odes()
already expect. Everything downstream (jaxkineticmodel's own
NeuralODEBuild/SBMLExporter, the round-trip check through
ode_model.model_from_sbml) is reused unchanged.

Why this exists: previously there was NO reader for XPP's .ode format
anywhere in this project. Converting a .ode file meant an agent (or a human)
had to manually retype species/odes as tool arguments -- easy to get wrong,
and in practice produced generic/mismatched placeholder systems instead of
the real model. This module removes that manual step entirely.

Handles the subset of XPP syntax these models actually use:
  - '#' full-line and trailing comments (a line is dropped entirely if its
    first non-space character is '#'; other lines are truncated at '#')
  - par/p lines: par a=1,b=2   (comma-separated; LATER uncommented
    assignments override earlier ones -- this is how the .ode files in this
    project encode "try model A, or model B, or model C" as three blocks of
    `#par ...` / `#par ...` / `par ...`, only the last (uncommented) one
    active)
  - algebraic ("auxiliary") assignments: NAME = expr  (no prime) --
    substituted forward, in file order, into everything after them. This is
    what resolves conservation relations (A = ATOT-(AP+APP+...)) and
    intermediate rate expressions (v1 = k1*A*K-k11*AK) down into pure
    state+parameter expressions before they ever reach odes_to_sbml.
  - derivative lines: NAME' = expr  -- these define the state variables, in
    the order first encountered.
  - initial values: init NAME=expr or NAME(0)=expr, evaluated using the
    parameters already declared.
  - '@' AUTO/numerics settings and a bare 'done' line are ignored (XPP
    solver/continuation config, not part of the ODE system itself).

Other declaration types (aux, table, global, markov, ...) are rejected with
a line number. Supporting them requires extending parse_xpp().
"""
import json
import re
from pathlib import Path

import sympy as sp

from dynalyst import config

_DERIV_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\'\s*=\s*(.*?)\s*$")
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$")
_INIT_RE = re.compile(r"^\s*init\s+(.+?)\s*$", re.IGNORECASE)
_INITIAL_EXPR_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(\s*0\s*\)\s*=\s*(.+)$")

# Mirrors pdf_tools._resolve_pdf_path / generic_sbml.resolve_sbml_path: a
# bare filename from the LLM (e.g. "model.txt", with no
# directory) should still resolve if the file was uploaded through the chat
# UI rather than typed as a real path on this machine. Without this, every
# .ode file the user "uploads" instead of paths-on-disk fails with a
# confusing "No such file" even though the file plainly exists somewhere.
_ODE_SEARCH_DIRS = [
    config.PROJECT_ROOT,
    config.DATA_DIR,
    Path("/mnt/user-data/uploads"),
    Path.cwd(),
]


def _resolve_ode_path(path_or_name: str) -> Path:
    candidate = Path(path_or_name)
    if candidate.exists():
        return candidate

    query = candidate.name.lower()  # tolerate a wrong/nonexistent directory prefix too
    stem = query[:-4] if query.endswith(".ode") else (
        query[:-4] if query.endswith(".txt") else query
    )
    searched = []
    for d in _ODE_SEARCH_DIRS:
        if not d.exists():
            continue
        searched.append(str(d))
        # exact filename first (handles .txt-named .ode files like this project's)
        exact = d / candidate.name
        if exact.exists():
            return exact
        # IMPORTANT: never use substring/fuzzy matching here. If the user asks
        # for `model.txt`, silently selecting
        # `model_plus_extra.txt` produces a valid-looking but WRONG
        # SBML model and makes all downstream bifurcation results meaningless.
        # Prefer exact filename matches only.
        exact_recursive = [
            p for p in d.rglob(candidate.name)
            if p.is_file() and p.suffix.lower() in (".ode", ".txt")
        ]
        if exact_recursive:
            exact_recursive.sort(key=lambda x: (len(str(x)), str(x)))
            return exact_recursive[0]

        # Also allow an exact stem match across .ode/.txt when the user gives
        # the same model name with the other XPP extension.
        stem_matches = [
            p for p in d.rglob("*")
            if p.is_file()
            and p.stem.lower() == stem
            and p.suffix.lower() in (".ode", ".txt")
        ]
        if stem_matches:
            stem_matches.sort(key=lambda x: (0 if x.suffix.lower() == candidate.suffix.lower() else 1, len(str(x)), str(x)))
            return stem_matches[0]

    raise FileNotFoundError(
        f"No .ode/.txt file matching '{path_or_name}' found (checked the given "
        f"path, and {searched or _ODE_SEARCH_DIRS}). If it was uploaded through "
        f"the chat UI, confirm it actually lands in one of those directories."
    )


def _normalise_xpp_expr(expr: str) -> str:
    expr = expr.strip().replace('^', '**')
    expr = re.sub(r"\bln\b", "log", expr, flags=re.IGNORECASE)
    return expr


def _sympify_xpp(expr: str, known_names: set[str] | None = None) -> sp.Expr:
    expr = _normalise_xpp_expr(expr)
    known_names = known_names or set()
    tokens = set(re.findall(r"[A-Za-z_]\w*", expr))
    reserved = {'exp','log','sqrt','sin','cos','tan','asin','acos','atan','sinh','cosh',
                'tanh','abs','min','max','floor','ceil','pi','E','oo','e'}
    locals_map = {tok: sp.Symbol(tok) for tok in tokens if tok not in reserved}
    locals_map.update({name: sp.Symbol(name) for name in known_names})
    return sp.sympify(expr, locals=locals_map)


def _substitute_algebraics(expr: sp.Expr, algebraic: dict[sp.Symbol, sp.Expr]) -> sp.Expr:
    cur = expr
    for _ in range(max(1, len(algebraic) + 2)):
        nxt = cur.subs(algebraic, simultaneous=False)
        if nxt == cur:
            break
        cur = nxt
    return cur


def parse_xpp(text: str) -> tuple[list[str], dict[str, str], dict[str, float], dict[str, float]]:
    par_defaults: dict[str, float] = {}
    init_values: dict[str, float] = {}
    algebraic: dict[sp.Symbol, sp.Expr] = {}
    odes_raw: dict[str, str] = {}
    order: list[str] = []

    for line_number, raw_line in enumerate(text.splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue
        low = line.lower()
        if low == 'done' or line.startswith('@'):
            continue

        if low.startswith('par ') or low.startswith('p '):
            body = line.split(None, 1)[1]
            for assign in re.split(r'\s*,\s*|(?<=\S)\s+(?=[A-Za-z_]\w*\s*=)', body):
                if not assign.strip():
                    continue
                if '=' not in assign:
                    raise ValueError(f"Invalid XPP parameter declaration at line {line_number}: {line}")
                name, val = assign.split('=', 1)
                name, val = name.strip(), val.strip()
                try:
                    par_defaults[name] = float(sp.N(_sympify_xpp(val)))
                except Exception as e:
                    raise ValueError(f"Invalid parameter value for {name}: {val!r}: {e}") from e
            continue

        if low.startswith('aux ') and '=' in line:
            line = line.split(None, 1)[1].strip()

        initial_expr = _INITIAL_EXPR_RE.match(line)
        if initial_expr:
            name, value = initial_expr.groups()
            expr = _sympify_xpp(value, set(par_defaults))
            init_values[name] = float(expr.subs({sp.Symbol(k): v for k, v in par_defaults.items()}))
            continue

        m = _INIT_RE.match(line)
        if m:
            for assign in re.split(r'\s*,\s*|(?<=\S)\s+(?=[A-Za-z_]\w*\s*=)', m.group(1)):
                if not assign.strip():
                    continue
                if '=' not in assign:
                    raise ValueError(f"Invalid XPP initial declaration at line {line_number}: {line}")
                name, val = assign.split('=', 1)
                try:
                    init_values[name.strip()] = float(sp.N(_sympify_xpp(val.strip())))
                except Exception as e:
                    raise ValueError(f"Invalid initial value for {name.strip()}: {val.strip()!r}: {e}") from e
            continue

        m = _DERIV_RE.match(line)
        if m:
            name, rhs = m.group(1), m.group(2)
            if name not in odes_raw:
                order.append(name)
            odes_raw[name] = rhs
            continue

        m = _ASSIGN_RE.match(line)
        if m:
            name, rhs = m.group(1), m.group(2)
            try:
                expr = _sympify_xpp(rhs, set(odes_raw) | set(par_defaults) | {name})
                algebraic[sp.Symbol(name)] = _substitute_algebraics(expr, algebraic)
            except Exception as e:
                raise ValueError(f"Invalid XPP assignment at line {line_number}: {line}") from e
            continue
        raise ValueError(f"Unsupported XPP declaration at line {line_number}: {line}")

    if not order:
        return [], {}, par_defaults, init_values

    known = set(order) | set(par_defaults) | {str(k) for k in algebraic}
    odes: dict[str, str] = {}
    for name in order:
        expr = _sympify_xpp(odes_raw[name], known)
        expr = _substitute_algebraics(expr, algebraic)
        odes[name] = str(sp.expand(expr))

    used_params: set[str] = set()
    for rhs in odes.values():
        used_params |= {str(sym) for sym in _sympify_xpp(rhs, set(order)).free_symbols}
    used_params -= set(order)
    parameters = {p: par_defaults[p] for p in used_params if p in par_defaults}
    return order, odes, parameters, init_values


def looks_like_xpp(text: str) -> bool:
    return any(_DERIV_RE.match(line.split('#', 1)[0].strip()) for line in text.splitlines())



def xpp_text_to_sbml(text: str, model_name: str = "uploaded_xpp_model",
                     initial_conditions: dict | None = None,
                     compartment_size: float = 1.0) -> str:
    """Auto-detect XPP text and convert it directly to SBML."""
    if not isinstance(text, str) or not text.strip():
        return json.dumps({"error": "Uploaded text is empty."})
    if not looks_like_xpp(text):
        return json.dumps({"error": "Uploaded text does not look like an XPP/XPPAUT ODE model (no NAME'=... derivative lines found)."})
    try:
        from dynalyst.analysis.jaxkm_ode_to_sbml import odes_to_sbml
        species, odes, parameters, parsed_initials = parse_xpp(text)
        merged_initials = dict(parsed_initials)
        if initial_conditions:
            merged_initials.update(initial_conditions)
        result = json.loads(odes_to_sbml(
            species=species, odes=odes, parameters=parameters,
            initial_conditions=merged_initials, model_name=model_name,
            compartment_size=compartment_size,
        ))
        if "error" not in result:
            result["source"] = "uploaded_xpp_text"
            result["parsed_species"] = species
            result["parsed_parameters"] = parameters
            result["parsed_initial_conditions"] = merged_initials
        return json.dumps(result, indent=2, default=float)
    except Exception as e:
        return json.dumps({"error": f"Could not convert uploaded XPP text: {e}"})

def xpp_to_sbml(path: str, model_name: str | None = None,
                 initial_conditions: dict | None = None,
                 compartment_size: float = 1.0) -> str:
    """
    Tool: read an XPPAUT .ode file from disk, parse it (par/algebraic/
    derivative lines -- see module docstring), and convert it straight to a
    real SBML file via the existing odes_to_sbml() pipeline (jaxkineticmodel's
    own NeuralODEBuild/SBMLExporter). This is the single entry point an
    agent should use for ANY .ode file -- do NOT hand-retype species/odes
    from the file's contents as odes_to_sbml arguments; that step is exactly
    what this tool replaces, and manual retyping is how mismatched/placeholder
    models have slipped through before.
    """
    try:
        p = _resolve_ode_path(path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

    try:
        from dynalyst.analysis.jaxkm_ode_to_sbml import odes_to_sbml
        text = p.read_text()
        species, odes, parameters, parsed_initials = parse_xpp(text)
    except Exception as e:
        return json.dumps({"error": f"Could not parse {p.name} as an XPP .ode file: {e}"})

    if not species:
        return json.dumps({
            "error": f"Parsed {p.name} but found no derivative (NAME'=...) lines -- "
                     f"is this actually an XPP .ode file?"
        })

    name = model_name or p.stem
    merged_initials = dict(parsed_initials)
    if initial_conditions:
        merged_initials.update(initial_conditions)
    result = odes_to_sbml(
        species=species, odes=odes, parameters=parameters,
        initial_conditions=merged_initials, model_name=name,
        compartment_size=compartment_size,
    )
    parsed_result = json.loads(result)
    if "error" not in parsed_result:
        parsed_result["source_ode_file"] = str(p)
        parsed_result["parsed_species"] = species
        parsed_result["parsed_parameters"] = parameters
        parsed_result["note"] = (
            f"Parsed directly from {p.name} (no manual transcription) and "
            f"exported via odes_to_sbml. " + parsed_result.get("note", "")
        )
    return json.dumps(parsed_result, indent=2, default=float)


TOOL_DEFINITIONS = [
    {
        "name": "xpp_text_to_sbml",
        "description": (
            "Auto-detect raw XPP/XPPAUT ODE text and convert it directly to a real "
            "SBML file. Use this for uploaded/pasted .txt or .ode contents when the "
            "content itself is available. It detects derivative lines automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Raw XPP/XPPAUT text."},
                "model_name": {"type": "string"},
                "initial_conditions": {"type": "object"},
                "compartment_size": {"type": "number"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "xpp_to_sbml",
        "description": (
            "Convert an XPPAUT .ode file (plain-text ODE model: 'par' lines, "
            "algebraic/conservation assignments, NAME'=rhs derivative lines, '@' "
            "AUTO settings) directly into a real SBML file, by parsing it and handing "
            "the result to odes_to_sbml() -- no manual retyping of species/equations "
            "needed or wanted. ALWAYS use this tool (not manual species/odes "
            "transcription) when the user gives you a .ode file or XPP-format text "
            "and wants it as SBML, wants to run the analysis pipeline on it, or asks "
            "'convert this .ode file'. If the .ode file has multiple par blocks "
            "(commented out with '#par' vs active 'par'), only the active "
            "(uncommented) one is used, matching XPP's own semantics. Pass just the "
            "filename the user gave you (e.g. 'model.txt') even if you don't know its "
            "exact directory -- this tool searches the project data directory and the "
            "chat upload directory for it, so do not guess or fabricate a full path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the .ode file on disk."},
                "model_name": {
                    "type": "string",
                    "description": "Name for the model / output filename. Defaults to the .ode file's stem.",
                },
                "initial_conditions": {
                    "type": "object",
                    "description": "Optional {species_name: value} overrides for initial values parsed from init or NAME(0) declarations. Species without an initial value default to 0.0.",
                },
                "compartment_size": {"type": "number", "description": "Single default compartment's size. Default 1.0."},
            },
            "required": ["path"],
        },
    },
]

TOOL_FUNCTIONS = {
    "xpp_to_sbml": lambda **kwargs: xpp_to_sbml(**kwargs),
    "xpp_text_to_sbml": lambda **kwargs: xpp_text_to_sbml(**kwargs),
}
