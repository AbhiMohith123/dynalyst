"""Tool registry: every tool the agent can use, as LangChain tools, grouped by
who may call it.

  UTILITY   -- the orchestrator's read-only lookups: list_available_models and
               list_available_pdfs (which files are local), model_parameters (a
               model's parameters and their part in the rate laws), inspect_model
               (its species, equations, reactions) and inspect_dataset (a data
               file's columns); everything else belongs to a sub-agent
  CONTEXT   -- read-only tools any sub-agent may call while working out its
               selection (inspect a model, list models, inspect a dataset)
  OBJECTIVE -- the analysis tools; each belongs to exactly one sub-agent
               (see subagents.py), is selected by it, checked by the guard
               and run here

execute() runs any tool by name and always returns text: errors come back
as a JSON {"error": ...} string, never as an exception (a call refused by the
file-safety or input checks too), and the tools'
own console output (AUTO-07p prints a lot) is captured.
"""
import contextlib
import io
import json
import threading
import traceback

from langchain_core.tools import StructuredTool

from dynalyst import config


# --- read-only context tools written for the agent ---------------------------

def inspect_model(model_name_or_path: str) -> str:
    """Species, parameters (with values), ODEs, dimension, and which
    bifurcation tool family can read the file."""
    from dynalyst.analysis.generic_sbml import time_unit_of
    from dynalyst.analysis.model_classifier import conservation_laws, load_any_model
    from dynalyst.analysis.ode_model import eligible_parameters, ode_strings, parameter_listing
    try:
        # read as written, delays included: reading a delay equation is fine, only the
        # analyses written for ordinary differential equations refuse it
        model, reader = load_any_model(model_name_or_path, allow_delay=True)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    from dynalyst.analysis.generic_sbml import DELAY, DELAY_MESSAGE
    import sympy as sp
    delayed = any(isinstance(e, sp.Basic) and e.atoms(DELAY) for e in model["rhs"])
    if not delayed:
        from dynalyst.analysis.conservation_parameters import expose_totals
        model = expose_totals(model)
    try:
        n_laws = len(conservation_laws(model))
    except Exception:
        n_laws = None
    uses_rules = reader == "rule-substituting SBML parser"
    odes, readable = ode_strings(model), {}
    if not uses_rules:   # the research reader keeps reactions, rules and constants for reading
        from dynalyst.analysis.ode_model import ode_report
        report = ode_report(model)
        odes = report["odes"]
        readable = {k: report[k] for k in ("reactions", "odes_as_reaction_rates", "assignment_rules",
                                           "constants", "unused_parameters", "parameter_meanings")
                    if k in report}
    return json.dumps({
        "model": model["label"], "path": model.get("source_path"), "reader": reader,
        "species": model["state_names"], "initial_conditions": model["y0"],
        "parameters": model["param_defaults"], "odes": odes, **readable,
        "parameter_kinds": parameter_listing(model),
        "continuation_parameters": eligible_parameters(model),
        "conserved_totals": model.get("conserved_totals", {}),
        "continuation_parameter_note": (
            "These free parameters occur in the ODEs; eligibility does not predict a bifurcation. "
            "Kinetic constants are valid choices. A conserved total is a choice only if it is "
            "explicitly exposed as a free parameter in these equations. Dynamic species and "
            "initial conditions are not themselves continuation parameters."),
        "time_unit": (time_unit_of(model.get("source_path") or "") or
                      "not declared: time is in the unit the rate constants use"),
        "n_state_variables": len(model["state_names"]),
        "independent_dimension": (len(model["state_names"]) - n_laws) if n_laws is not None else None,
        "uses_assignment_rules": uses_rules,
        **({"delay_differential_equation": DELAY_MESSAGE} if delayed else {}),
    }, indent=2, default=float)


def _rate_role(rate, name: str, values: dict) -> str:
    """How one parameter enters one rate law, read from the expression: an exponent, a
    factor the rate is proportional to, or else the direction the rate moves when the
    parameter rises (at the model's own values)."""
    import sympy as sp
    s = sp.Symbol(name)
    if any(s in a.exp.free_symbols for a in rate.atoms(sp.Pow)):
        return "exponent in the rate law"
    try:
        if s not in (rate / s).free_symbols or s not in sp.cancel(rate / s).free_symbols:
            return "the rate is proportional to it"
    except Exception:
        pass
    try:
        slope = float(sp.diff(rate, s).subs(values))
    except Exception:
        slope = float("nan")
    if slope < 0:
        return "the rate falls as it rises (not proportionally)"
    if slope > 0:
        return "the rate rises as it rises (not proportionally)"
    return "occurs in the rate law"


def model_parameters(model_name_or_path: str) -> str:
    """Each parameter that can be varied, with its value, kind, the name the file gives it,
    and its part in each reaction rate it occurs in (read from the rate laws)."""
    import sympy as sp
    from dynalyst.analysis.generic_sbml import DELAY
    from dynalyst.analysis.model_classifier import load_any_model
    from dynalyst.analysis.ode_model import eligible_parameters
    try:
        model, _ = load_any_model(model_name_or_path, allow_delay=True)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    if not any(isinstance(e, sp.Basic) and e.atoms(DELAY) for e in model["rhs"]):
        from dynalyst.analysis.conservation_parameters import expose_totals
        model = expose_totals(model)
    disp = model.get("display") or {}
    values = {sp.Symbol(k): (v.get("value") if isinstance(v, dict) else v)
              for k, v in (disp.get("constants") or {}).items()}
    values.update({sp.Symbol(k): v for k, v in model["param_defaults"].items()})
    y0 = model["y0"] if isinstance(model["y0"], dict) else dict(zip(model["state_names"],
                                                                     model["y0"]))
    values.update({sp.Symbol(k): (v or 1.0) for k, v in y0.items()})   # 0 would hide a slope
    rules = {sp.Symbol(k): sp.sympify(e) for k, e in (disp.get("assignment_rules") or {}).items()}
    reactions = []
    for r in disp.get("reactions") or []:
        try:
            rate = sp.sympify(r["rate"])
            for _ in range(len(rules)):     # the rate as the parameters write it
                if not rate.free_symbols & rules.keys():
                    break
                rate = rate.subs(rules)
            reactions.append((r, rate))
        except Exception:
            pass
    kinds, titles = model.get("param_kinds") or {}, disp.get("param_titles") or {}
    totals = model.get("conserved_totals") or {}
    rows = []
    for p in eligible_parameters(model):
        row = {"name": p, "value": model["param_defaults"].get(p),
               "kind": kinds.get(p, "parameter")}
        if titles.get(p) and titles[p] != p:
            row["name_in_file"] = titles[p]
        if p in totals:
            t = totals[p]
            row["conserved_total_of"] = str(t.get("expression", t) if isinstance(t, dict) else t)
        found = [f"{r['id']}{' (' + r['name'] + ')' if r.get('name') else ''}: "
                 f"{_rate_role(rate, p, values)}"
                 for r, rate in reactions if sp.Symbol(p) in rate.free_symbols]
        if found:
            row["in_reactions"] = found
        elif not reactions:
            row["in_equations"] = [f"d{s}/dt" for s, e in zip(model["state_names"], model["rhs"])
                                   if sp.Symbol(p) in sp.sympify(e).free_symbols]
        rows.append(row)
    return json.dumps({"model": model["label"], "parameters": rows, "note": (
        "Every parameter that occurs in the equations, with its part in each reaction rate "
        "(read from the rate laws). A description in words can fit several of them.")},
        indent=1, default=float)


_CONTEXT_DEFS = [
    ({"name": "model_parameters",
      "description": ("Read-only. The parameters of a model that can be varied: value, kind "
                      "(constant, fixed species such as an enzyme amount, conserved total), "
                      "the name the file gives it, and its part in each reaction rate (the "
                      "rate is proportional to it, falls or rises as it rises, or an "
                      "exponent). Use it when the user describes a parameter instead of "
                      "naming it, to see which parameters fit."),
      "input_schema": {"type": "object", "properties": {
          "model_name_or_path": {"type": "string"}}, "required": ["model_name_or_path"]}},
     model_parameters),
    ({"name": "inspect_model",
      "description": ("Read-only. Species, initial conditions, parameters with values, ODEs, "
                      "dimension, the model's time unit, and which bifurcation tool can read "
                      "this model. Use it to get exact species/parameter names (and the time "
                      "unit t_end is in) before selecting an analysis tool."),
      "input_schema": {"type": "object", "properties": {
          "model_name_or_path": {"type": "string"}}, "required": ["model_name_or_path"]}},
     inspect_model),
]


# --- building the registry ----------------------------------------------------

# Tools that analyse the symbolic model (generic_sbml.build_symbolic_model), in
# which time-dependent inputs are frozen. Simulation tools use the real time course.
FROZEN_INPUT_TOOLS = {"find_steady_states", "run_bifurcation_analysis", "classify_model",
                      "reduce_model", "get_odes", "inspect_model"}


def model_context(model, tool_name=None) -> dict:
    """What every result about an SBML model should carry: 'model_identity'
    (the file, its own model name and the BioModels ID the file declares, or
    None -- so an answer never has to guess or borrow one) and, for
    FROZEN_INPUT_TOOLS, 'time_dependent_inputs' (which time-dependent input
    was frozen at which value, generic_sbml.freeze_time_input)."""
    from dynalyst.analysis.annotations import model_identity
    from dynalyst.analysis.ode_model import time_note_for
    extra = {}
    try:
        identity = model_identity(model)
    except Exception:
        identity = None
    if identity:
        extra["model_identity"] = identity
    if tool_name in FROZEN_INPUT_TOOLS:
        try:
            note = time_note_for(model)
        except Exception:
            note = None
        if note:
            extra["time_dependent_inputs"] = note
    return extra


def _with_model_context(name, fn):
    """Adds model_context(...) to a successful JSON result of a tool that takes
    model_name_or_path."""
    def call(**kw):
        out = fn(**kw)
        model = kw.get("model_name_or_path")
        if not model or not isinstance(out, str):
            return out
        try:
            data = json.loads(out)
        except Exception:
            return out
        if not isinstance(data, dict) or "error" in data:
            return out
        extra = model_context(model, name)
        if not extra:
            return out
        data.update(extra)
        return json.dumps(data, indent=2, default=float)
    return call


def _json_tool(defn: dict, fn) -> StructuredTool:
    wrapped = _with_model_context(defn["name"], fn)
    return StructuredTool.from_function(func=lambda **kw: wrapped(**kw), name=defn["name"],
                                        description=defn["description"],
                                        args_schema=defn["input_schema"])


def _build():
    from dynalyst.agents import research_agent
    from dynalyst.analysis import (annotations, auto_problems, biomodels, candidates, crnt_analysis,
                                   custom_plot, jaxkm_tools, paper_map,
                                   model_classifier, model_reduction,
                                   jacobian_network, literature, omnipath_check,
                                   parameter_estimation, save_model, stochastic, xpp_to_sbml)
    from dynalyst.rag import model_index
    from dynalyst.regime.categorize_tool import categorized_lhs_scan
    from dynalyst.regime.generic_tool import generic_bifurcation_model
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan

    json_defs = {}
    for module in (auto_problems, jaxkm_tools, xpp_to_sbml, model_classifier, annotations, custom_plot,
                   parameter_estimation, model_reduction, crnt_analysis, biomodels,
                   literature, model_index, save_model, candidates, paper_map, stochastic,
                   omnipath_check, jacobian_network):
        for d in module.TOOL_DEFINITIONS:
            json_defs[d["name"]] = (d, module.TOOL_FUNCTIONS[d["name"]])
    json_defs["retrieve_knowledge"] = (
        research_agent.RETRIEVE_TOOL_DEF,
        lambda **kw: research_agent._execute_tool("retrieve_knowledge", kw))
    # run_bifurcation_analysis goes through the research agent's dispatch
    # (live-module lookup + error wrapping), as it always has
    json_defs["run_bifurcation_analysis"] = (
        json_defs["run_bifurcation_analysis"][0],
        lambda **kw: research_agent._execute_tool("run_bifurcation_analysis", kw))
    for defn, fn in _CONTEXT_DEFS:
        json_defs[defn["name"]] = (defn, fn)

    tools = {name: _json_tool(d, fn) for name, (d, fn) in json_defs.items()}
    for t in (generic_bifurcation_model, lhs_bifurcation_scan, categorized_lhs_scan):
        tools[t.name] = t
    return tools


_TOOLS = None


def all_tools() -> dict:
    global _TOOLS
    if _TOOLS is None:
        _TOOLS = _build()
    return _TOOLS


# the orchestrator's own tools, all read-only: which model files and PDFs exist, so it can
# tell a local file from one to search for or download; a model's parameters and equations
# and a data file's columns, so it can tell which ones a user's description fits and ask;
# every analysis tool and every tool that writes a file belongs to a sub-agent
UTILITY = ["list_available_models", "list_available_pdfs", "model_parameters", "inspect_model",
           "inspect_dataset"]
CONTEXT = ["inspect_model", "list_available_models", "get_model_odes", "inspect_dataset"]


def get(names) -> list:
    tools = all_tools()
    return [tools[n] for n in names]


def schema_of(tool) -> dict:
    from langchain_core.utils.function_calling import convert_to_openai_tool
    return convert_to_openai_tool(tool)["function"].get("parameters", {})


# Tools that may run at the same time as other tools (parallel delegations): network
# look-ups that neither plot (matplotlib's pyplot is one global state), change the working
# directory (the LHS / AUTO pipeline does), nor write shared files. Every other tool holds
# _EXECUTION_LOCK while it runs, so parallel delegations overlap their language-model
# reasoning and these look-ups, never two computations.
PARALLEL_SAFE = {"search_biomodels", "get_biomodels_info", "find_biomodels_by_paper",
                 "find_models_for_pubmed", "get_uniprot_entry", "search_uniprot",
                 "search_literature", "get_pubmed_article"}
_EXECUTION_LOCK = threading.RLock()


def _types(prop: dict) -> set:
    t = prop.get("type")
    out = set(t) if isinstance(t, list) else {t} if t else set()
    for alt in prop.get("anyOf") or []:
        out |= _types(alt)
    return out


# numeric arguments for which 0 is a real value (a seed, a start time, a bound); for every
# other optional number -- counts, sizes, steps, tolerances, system sizes -- 0 is a filler
ZERO_IS_A_VALUE = {"seed", "burn_in", "t_start", "lower_bound", "upper_bound"}
# strings a language model writes for "nothing" in an optional argument
FILLER_STRINGS = {"", "none", "null", "nil", "n/a"}


def coerce_args(name: str, args):
    """A one-item list given for an argument that takes a single string (a model
    language model writing ["model.xml"] for "model.xml") becomes that string.
    Fillers a language model writes for arguments it means to leave out are dropped, so
    the tool's own default applies: for an optional argument an empty string or a filler
    word ("none", "null", ...: FILLER_STRINGS), an empty
    list or mapping, or a 0 for a number where 0 is no value (a count, size, step,
    tolerance; not a seed or bound: ZERO_IS_A_VALUE), and a <name>_min / <name>_max pair
    that is an empty window (min == max, e.g. 0 and 0). Anything else is left for the
    tool's own validation."""
    tool = all_tools().get(name)
    if tool is None or not isinstance(args, dict):
        return args
    schema = schema_of(tool)
    props, required = schema.get("properties", {}), set(schema.get("required", []))
    out = dict(args)
    for key, val in args.items():
        types = _types(props.get(key) or {})
        if ("string" in types and "array" not in types and isinstance(val, list)
                and len(val) == 1 and isinstance(val[0], str)):
            out[key] = val[0]
        elif key in required:
            continue
        elif isinstance(val, str) and val.strip().lower() in FILLER_STRINGS:
            del out[key]
        elif isinstance(val, (list, dict)) and not val:          # no list given: the default
            del out[key]
        elif (isinstance(val, (int, float)) and not isinstance(val, bool) and val == 0
              and key not in ZERO_IS_A_VALUE and not key.endswith(("_min", "_max"))
              and ("number" in types or "integer" in types)):
            del out[key]      # a count, size, step or tolerance of 0 is a filler, not a request
    for key in list(out):
        if key.endswith("_min") and key[:-4] + "_max" in out:
            lo, hi = out[key], out[key[:-4] + "_max"]
            if (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo == hi
                    and key not in required and key[:-4] + "_max" not in required):
                del out[key], out[key[:-4] + "_max"]
    return out


def input_problem(name: str, args: dict) -> str | None:
    """Why this call cannot run, found before running it: file safety (safety.py) and the
    tool's own input checks (inputs.py). None when it may run."""
    from dynalyst.orchestration import inputs, safety
    return safety.problem(name, args) or inputs.problem(name, args)


def execute(name: str, args: dict, capture_output: bool = True) -> str:
    tools = all_tools()
    if name not in tools:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    args = coerce_args(name, args)
    problem = input_problem(name, args)
    if problem:      # refused before it runs, and reported as this tool's own error
        return json.dumps({"error": problem, "tool": name})
    lock = contextlib.nullcontext() if name in PARALLEL_SAFE else _EXECUTION_LOCK
    with lock:
        return _execute(tools, name, args, capture_output)


def _execute(tools, name, args, capture_output):
    buf = io.StringIO()
    try:
        # redirect_stdout swaps the process-wide sys.stdout: only safe under the lock
        redirect = capture_output and name not in PARALLEL_SAFE
        with contextlib.redirect_stdout(buf) if redirect else contextlib.nullcontext():
            out = tools[name].invoke(args)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}", "tool": name,
                           "traceback": traceback.format_exc()[-3000:]})
    out = out if isinstance(out, str) else json.dumps(out, default=str)
    return _index_saved_model(name, out)


# tools that save a new model, and the result field holding its path: every such
# model goes into the knowledge base at once (rag/model_index.py)
SAVES_MODEL = {"fetch_biomodels_model": "saved_as", "xpp_to_sbml": "path",
               "xpp_text_to_sbml": "path", "odes_to_sbml": "path", "save_sbml": "path"}


def _index_saved_model(name, out):
    if name not in SAVES_MODEL or not config.AUTO_INDEX_MODELS:
        return out
    try:
        data = json.loads(out)
    except ValueError:
        return out
    path = isinstance(data, dict) and not data.get("error") and data.get(SAVES_MODEL[name])
    if not path:
        return out
    from dynalyst.rag.model_index import auto_index
    with contextlib.redirect_stdout(io.StringIO()):
        data.update(auto_index(path))
    return json.dumps(data, indent=2, default=str)


_HEAVY_KEYS = {"auto_stdout_log", "auto_terminal_log", "auto_stderr_log", "auto_diagnostics",
               "traceback", "sampled_time_series", "reloaded_odes"}


# the fields that state what a result found: shortened last, so a long result keeps them
_OUTCOME_KEYS = {"special_points_summary", "fold_notes", "conclusion", "conclusions", "interpretation", "summary", "verdict", "period_note",
                 "stability_verdicts", "behaviour", "warnings", "error", "note", "plot_path",
                 "plot_note", "final_vs_parameter", "results", "models", "hopf_points", "fold_points",
                 "bifurcation_points", "n_bifurcation_points", "classification", "category",
                 "categories", "goal_check", "steady_states", "fitted_parameters", "bistable",
                 "multistationary_sets", "independent_dimension", "stability_changes",
                 "tracking_stopped", "range_achieved", "equilibrium_branches",
                 "periodic_orbits", "branch_data_csv", "auto_problems",
                 "unlabelled_steady_state_changes_at",
                 "cycles_not_followed_from"}


def _shorten(value, chars):
    """`value` with every string longer than `chars` cut to that length (lists and
    dicts kept whole), so a list of records keeps its IDs and titles when it is too long."""
    if isinstance(value, str):
        return value if len(value) <= chars else value[:chars] + "..."
    if isinstance(value, list):
        return [_shorten(v, chars) for v in value]
    if isinstance(value, dict):
        return {k: _shorten(v, chars) for k, v in value.items()}
    return value


def _compact(value, n: int, depth: int = 0, top: int = 12):
    """`value` with every nested list cut to its first n items (the field's own list to
    `top`, so each of its records stays: e.g. every steady state), a note saying how many
    more there are, and long texts cut: a result made of matrices and long lists keeps its
    structure and leading entries (e.g. the top sensitivities) instead of being dropped."""
    if isinstance(value, str):
        return value if len(value) <= 150 else value[:150] + "..."
    if isinstance(value, list):
        keep = top if depth == 0 else n
        if depth == 0 and len(value) > keep and _is_sweep(value):
            # a sweep: rows spread over the whole range (first and last included), not only
            # its start, so the trend across the range stays visible
            idx = sorted({round(i * (len(value) - 1) / (keep - 1)) for i in range(keep)})
            return ([_compact(value[i], n, depth + 1, top) for i in idx]
                    + [f"{len(idx)} of {len(value)} rows, evenly spaced over the range "
                       "(all kept in the run record)"])
        head = [_compact(v, n, depth + 1, top) for v in value[:keep]]
        return head + ([f"... {len(value) - keep} more (kept in the run record)"]
                       if len(value) > keep else [])
    if isinstance(value, dict):
        return {k: _compact(v, n, depth + 1, top) for k, v in value.items()}
    return value


def _is_sweep(rows) -> bool:
    """Records ordered by a number that only rises or only falls (a scan's parameter, a
    time): a sweep, not a ranked list (papers, candidates), where the first items matter."""
    if not (len(rows) > 2 and all(isinstance(r, dict) for r in rows)):
        return False
    for key, first in rows[0].items():
        if not isinstance(first, (int, float)) or isinstance(first, bool):
            continue
        vals = [r.get(key) for r in rows]
        if all(isinstance(v, (int, float)) for v in vals):
            steps = [b - a for a, b in zip(vals, vals[1:])]
            if all(s > 0 for s in steps) or all(s < 0 for s in steps):
                return True
    return False


def _without_largest(value):
    """A list of records with the largest field of each record left out (named), so every
    record stays with its smaller, usually decisive, fields."""
    if not (isinstance(value, list) and value and all(isinstance(v, dict) for v in value)):
        return value
    out = []
    for rec in value:
        sizes = {k: len(json.dumps(v, default=str)) for k, v in rec.items()}
        big = max(sizes, key=sizes.get)
        out.append({**rec, big: f"<omitted from this view: {sizes[big]} characters>"})
    return out


def for_model(output: str, limit: int | None = None) -> str:
    """A tool output trimmed for a language model's context: bulky log /
    trace / time-series fields dropped (they stay in the run record)."""
    limit = limit or config.TOOL_OUTPUT_CHARS
    try:
        data = json.loads(output)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        slim = {k: ("<omitted>" if k in _HEAVY_KEYS else v) for k, v in data.items()}
        text = json.dumps(slim, default=str)
        # too long: leave out the largest fields first, saying so, rather than
        # cutting the end off -- where the conclusion of a result often is
        # details go before the fields that state the outcome (conclusion, verdicts, plots)
        sizes = sorted(((len(json.dumps(v, default=str)), k) for k, v in slim.items()),
                       key=lambda sk: (sk[1] in _OUTCOME_KEYS, -sk[0]))
        for size, key in sizes:
            if len(text) <= limit:
                break
            # a small detail field gains little and may matter: a larger field (an outcome
            # list) is shortened first; only the last pass below leaves small fields out
            if size < max(200, limit // 20):
                continue
            # a list or record of items keeps its items with long texts cut (e.g. abstracts);
            # one long text (a matrix dump, a log) is left out whole, never cut part-way
            candidates = []
            if isinstance(slim[key], (list, dict)):
                candidates = ([_shorten(slim[key], c) for c in (400, 150)]
                              + [_compact(slim[key], k) for k in (12, 6, 3, 1)])
                lean = slim[key]
                for _ in range(3):           # then the largest field of each record, in turn
                    lean = _without_largest(lean)
                    candidates.append(_compact(lean, 3))
            for short in candidates:
                if len(json.dumps({**slim, key: short}, default=str)) <= limit:
                    slim[key] = short
                    break
            else:
                items = _shorten(slim[key], 150) if isinstance(slim[key], list) else []
                while items and len(json.dumps({**slim, key: items + [f"... {len(slim[key]) - len(items)} more (kept in the run record)"]}, default=str)) > limit:
                    items = items[:-1]
                slim[key] = (items + [f"... {len(slim[key]) - len(items)} more (kept in the run record)"]
                             if items else
                             f"<omitted from this view: {size} characters (kept in the run record)>")
            text = json.dumps(slim, default=str)
        # still too long: whole fields left out, details first -- never a cut through the
        # JSON, which loses every field after the cut (a plot path, the interpretation)
        for size, key in sizes:
            if len(text) <= limit:
                break
            if not str(slim[key]).startswith("<omitted"):
                slim[key] = f"<omitted from this view: {size} characters (kept in the run record)>"
                text = json.dumps(slim, default=str)
    else:
        text = str(output)
    return text if len(text) <= limit else text[:limit] + f"... [truncated, {len(text)} chars]"
