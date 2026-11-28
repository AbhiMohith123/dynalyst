"""
Generic, model-agnostic SBML -> ODE builder.

This is the automatic fallback used when jaxkineticmodel's own SBML importer
can't load a file (e.g. constant species used only as rate-law modifiers, or
a rateRule-only file with no reaction network at all). Every model is read
from its own file; there are no hand-typed model equations anywhere.

Nothing here is model-specific. Everything -- species, parameters,
compartments, rateRules, reaction stoichiometry, kinetic laws, local
reaction parameters -- is read directly out of the SBML file via libsbml and
assembled into a numeric right-hand-side function, then integrated with
scipy's LSODA (stiff-capable, so it doesn't matter whether the underlying
model happens to be stiff -- no per-model solver choice needed either).

Two structural cases are handled, and can coexist in the same model:
  1. Species governed by a <rateRule>: dspecies/dt = rateRule's math, taken
     directly.
  2. Species produced/consumed by <reaction>s: dspecies/dt = sum over
     reactions the species participates in of (stoichiometry * kinetic law
     rate). Species marked constant="true" or boundaryCondition="true" are
     held fixed at their initial value and used as plain modifiers in rate
     laws -- never integrated -- which is exactly the pattern
     jaxkineticmodel's importer rejects.

Rules and the rest of SBML's model structure:
  - assignment rules the dynamics read are substituted in dependency order;
    ones no rate law reads (observables such as a fluorescence readout) are
    skipped. An algebraic rule that determines one variable is solved for it
    and becomes an assignment rule; any other (a true differential-algebraic
    system) is refused with the reason.
  - events and delays are applied by simulations (hybrid_sim); the other
    analyses say they use the continuous dynamics only.
  - a compartment whose size changes (a rate rule) is a state variable, and
    the concentrations in it are diluted as it grows (build_symbolic_model).
What cannot be applied is refused or reported, never silently dropped.
"""
import html
import math
import re
from pathlib import Path

import libsbml
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from dynalyst import config
from dynalyst.sbml_io import read_sbml_expanded


_NAMED_SECONDS = {1.0: "second", 60.0: "minute", 3600.0: "hour", 86400.0: "day"}
_TIME_UNITS = {}    # path -> (mtime, description)


def time_unit_of(path) -> str | None:
    """The model's time unit as its SBML file declares it ('hour (3600 s)', 'second'), or
    None when the file declares none (time is then in whatever unit its rate constants use)
    or is not SBML."""
    path = Path(path)
    if path.suffix.lower() != ".xml" or not path.is_file():
        return None
    key = str(path.resolve())
    stamp = path.stat().st_mtime
    if key in _TIME_UNITS and _TIME_UNITS[key][0] == stamp:
        return _TIME_UNITS[key][1]
    text = None
    try:
        model = libsbml.readSBML(str(path)).getModel()
        if model is not None:
            unit_id = (model.getTimeUnits() if model.isSetTimeUnits() else "") or "time"
            if unit_id == "second":
                text = "second"
            else:
                ud = model.getUnitDefinition(unit_id)
                units = list(ud.getListOfUnits()) if ud is not None else []
                if len(units) == 1 and units[0].getKind() == libsbml.UNIT_KIND_SECOND \
                        and units[0].getExponent() == 1:
                    seconds = units[0].getMultiplier() * 10.0 ** units[0].getScale()
                    name = next((n for s, n in _NAMED_SECONDS.items()
                                 if abs(seconds - s) <= 1e-9 * s), None)
                    text = (name if name == "second" else
                            f"{name} ({seconds:g} s)" if name else f"{seconds:g} s")
    except Exception:
        text = None
    _TIME_UNITS[key] = (stamp, text)
    return text


def resolve_sbml_path(model_name_or_path: str) -> Path:
    """
    Centralized SBML path resolution, used by every tool module in this
    project (previously duplicated separately in jaxkm_tools.py and
    bifurcation_1d.py). Accepts a full path to an .xml file, or a bare model
    name searched across config.SBML_SEARCH_DIRS.
    """
    candidate = config.existing_file(model_name_or_path)
    if candidate is not None:
        return candidate

    query = model_name_or_path.lower()
    if query.endswith(".xml"):
        query = query[:-4]
    config.ensure_unambiguous(query)

    searched = []
    for search_root in config.SBML_SEARCH_DIRS:
        if not search_root.exists():
            continue
        searched.append(str(search_root))
        match = config.best_name_match(query, search_root.rglob("*.xml"))
        if match:
            return match

    by_name = _match_declared_name(model_name_or_path)
    if by_name is not None:
        return by_name

    raise FileNotFoundError(
        f"No SBML file matching '{model_name_or_path}' found in: {searched or config.SBML_SEARCH_DIRS}. "
        f"Pass a full path instead, or check list_available_models()."
    )


_MODEL_TAG = re.compile(r"<(?:\w+:)?model\b([^>]*)>", re.IGNORECASE)
_ATTR = re.compile(r"\b(id|name)\s*=\s*(\"[^\"]*\"|'[^']*')")
_DECLARED = {}      # path -> (mtime, (model id, model name))


def _flat(text: str) -> str:
    return re.sub(r"[\W_]+", " ", str(text)).strip().lower()


def _declared_names(path: Path) -> tuple:
    """The id and name a file's <model> element declares (read from its start)."""
    mtime = path.stat().st_mtime
    hit = _DECLARED.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(20000)
    m = _MODEL_TAG.search(head)
    attrs = dict((k.lower(), v[1:-1]) for k, v in _ATTR.findall(m.group(1))) if m else {}
    names = (attrs.get("id", ""), html.unescape(attrs.get("name", "")))
    _DECLARED[path] = (mtime, names)
    return names


def _match_declared_name(query: str) -> "Path | None":
    """A local SBML file whose <model> id or name is `query` (ignoring case,
    spaces and punctuation), when no file name matches it. Several different
    files declaring that name is an error, not a guess."""
    q = _flat(Path(str(query)).stem if str(query).lower().endswith(".xml") else query)
    if not q:
        return None
    found = {}
    for root in config.SBML_SEARCH_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*.xml"):
            try:
                if q in {_flat(n) for n in _declared_names(path) if n}:
                    found.setdefault(path.read_bytes(), path)
            except OSError:
                continue
    if len(found) > 1:
        paths = [config.display_path(p) for p in found.values()]
        raise ValueError(f"'{query}' is the declared name of {len(paths)} different models: "
                         f"{', '.join(paths)}. Say which one by passing its path.")
    return next(iter(found.values()), None)


def available_sbml_models() -> dict[str, list[str]]:
    """Return the canonical model inventory used by every SBML tool."""
    result = {}
    for search_root in config.SBML_SEARCH_DIRS:
        if not search_root.exists():
            continue
        result[str(search_root)] = sorted(
            str(path.relative_to(search_root))
            for path in search_root.rglob("*.xml")
        )
    return result

# ---------------------------------------------------------------------------
# Generic MathML/AST evaluator -- walks any libsbml ASTNode and computes a
# numeric value given a flat {name: value} environment. This is what lets the
# rest of this module stay generic: it never needs to know what a formula
# "means", only how to evaluate it.
# ---------------------------------------------------------------------------

_UNARY_MATH_FUNCS = {
    libsbml.AST_FUNCTION_ABS: abs,
    libsbml.AST_FUNCTION_EXP: math.exp,
    libsbml.AST_FUNCTION_LN: math.log,
    libsbml.AST_FUNCTION_SIN: math.sin,
    libsbml.AST_FUNCTION_COS: math.cos,
    libsbml.AST_FUNCTION_TAN: math.tan,
    libsbml.AST_FUNCTION_SINH: math.sinh,
    libsbml.AST_FUNCTION_COSH: math.cosh,
    libsbml.AST_FUNCTION_TANH: math.tanh,
    libsbml.AST_FUNCTION_FLOOR: math.floor,
    libsbml.AST_FUNCTION_CEILING: math.ceil,
    # the rest of MathML's functions, with MathML's own definitions (arccot x = arctan 1/x)
    libsbml.AST_FUNCTION_SEC: lambda x: 1.0 / math.cos(x),
    libsbml.AST_FUNCTION_CSC: lambda x: 1.0 / math.sin(x),
    libsbml.AST_FUNCTION_COT: lambda x: 1.0 / math.tan(x),
    libsbml.AST_FUNCTION_ARCSIN: math.asin,
    libsbml.AST_FUNCTION_ARCCOS: math.acos,
    libsbml.AST_FUNCTION_ARCTAN: math.atan,
    libsbml.AST_FUNCTION_ARCSEC: lambda x: math.acos(1.0 / x),
    libsbml.AST_FUNCTION_ARCCSC: lambda x: math.asin(1.0 / x),
    libsbml.AST_FUNCTION_ARCCOT: lambda x: math.atan(1.0 / x),
    libsbml.AST_FUNCTION_SECH: lambda x: 1.0 / math.cosh(x),
    libsbml.AST_FUNCTION_CSCH: lambda x: 1.0 / math.sinh(x),
    libsbml.AST_FUNCTION_COTH: lambda x: 1.0 / math.tanh(x),
    libsbml.AST_FUNCTION_ARCSINH: math.asinh,
    libsbml.AST_FUNCTION_ARCCOSH: math.acosh,
    libsbml.AST_FUNCTION_ARCTANH: math.atanh,
    libsbml.AST_FUNCTION_ARCSECH: lambda x: math.acosh(1.0 / x),
    libsbml.AST_FUNCTION_ARCCSCH: lambda x: math.asinh(1.0 / x),
    libsbml.AST_FUNCTION_ARCCOTH: lambda x: math.atanh(1.0 / x),
    libsbml.AST_FUNCTION_FACTORIAL: lambda x: math.gamma(x + 1.0),   # x! = Gamma(x + 1)
}

def _is_rate_of_call(node) -> bool:
    """A call of a user function 'rateOf' with one argument: SBML Level 2's stand-in for the
    Level 3 rateOf (its NaN placeholder definition is removed on reading, sbml_io)."""
    return (node.getType() == libsbml.AST_FUNCTION and (node.getName() or "").lower() == "rateof"
            and node.getNumChildren() == 1)


AVOGADRO = 6.02214076e23     # SBML's avogadro symbol: the exact SI value (mol^-1)


class DelayModel(NotImplementedError):
    """The model uses SBML's delay(x, tau): a delay differential equation, which the
    analyses here (built for ordinary differential equations) do not treat."""


DELAY_MESSAGE = ("the model uses delay(x, tau): it is a delay differential equation, where the "
                 "rates depend on earlier states. It can be simulated (simulate_model integrates "
                 "the delays) and its steady states found (find_steady_states; their stability "
                 "is not given). Bifurcation, CRNT, reduction and stochastic analyses here are "
                 "written for ordinary differential equations and would misread it, so they are "
                 "not run on it.")


def eval_ast(node, env: dict) -> float:
    """Recursively evaluate a libsbml ASTNode against a {name: value} env."""
    if node is None:
        return 0.0

    if node.isNumber():
        return float(node.getValue())
    if node.getType() == libsbml.AST_NAME_AVOGADRO:
        return AVOGADRO
    if node.getType() == libsbml.AST_FUNCTION_DELAY:
        raise DelayModel(DELAY_MESSAGE)

    if node.isName():
        name = node.getName()
        if name not in env:
            raise KeyError(
                f"Symbol '{name}' used in SBML math has no value (not a known "
                f"species, parameter, or compartment)."
            )
        return float(env[name])

    ntype = node.getType()
    if ntype == libsbml.AST_CONSTANT_PI:
        return math.pi
    if ntype == libsbml.AST_CONSTANT_E:
        return math.e
    if ntype == libsbml.AST_CONSTANT_TRUE:
        return 1.0
    if ntype == libsbml.AST_CONSTANT_FALSE:
        return 0.0

    n = node.getNumChildren()
    args = [eval_ast(node.getChild(i), env) for i in range(n)]

    if ntype == libsbml.AST_PLUS:
        return sum(args) if args else 0.0
    if ntype == libsbml.AST_MINUS:
        return (args[0] - args[1]) if len(args) == 2 else -args[0]
    if ntype == libsbml.AST_TIMES:
        result = 1.0
        for a in args:
            result *= a
        return result
    if ntype == libsbml.AST_DIVIDE:
        return args[0] / args[1]
    if ntype in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
        return args[0] ** args[1]
    if ntype == libsbml.AST_FUNCTION_ROOT:
        # root(degree, x) in SBML MathML child order
        return args[1] ** (1.0 / args[0]) if len(args) == 2 else math.sqrt(args[0])
    if ntype == libsbml.AST_FUNCTION_LOG:
        # log(x) base 10, or log(base, x) with two children
        return math.log10(args[0]) if len(args) == 1 else math.log(args[1], args[0])
    if ntype == libsbml.AST_FUNCTION_PIECEWISE:
        for i in range(0, n - 1, 2):
            if args[i + 1]:
                return args[i]
        return args[-1] if n % 2 == 1 else 0.0
    if ntype in _RELATIONAL_MATH:
        # SBML allows chains (a < b < c): every neighbouring pair must hold
        rel = _RELATIONAL_MATH[ntype]
        return float(all(rel(args[i], args[i + 1]) for i in range(len(args) - 1)))
    if ntype == libsbml.AST_LOGICAL_AND:
        return float(all(args))
    if ntype == libsbml.AST_LOGICAL_OR:
        return float(any(args))
    if ntype == libsbml.AST_LOGICAL_XOR:
        return float(sum(bool(a) for a in args) % 2 == 1)
    if ntype == libsbml.AST_LOGICAL_IMPLIES:
        return float((not args[0]) or bool(args[1]))
    if ntype == libsbml.AST_LOGICAL_NOT:
        return float(not args[0])
    if ntype == libsbml.AST_FUNCTION_MIN:
        return float(min(args))
    if ntype == libsbml.AST_FUNCTION_MAX:
        return float(max(args))
    if ntype == libsbml.AST_FUNCTION_REM:            # remainder, sign of the dividend
        return math.fmod(args[0], args[1])
    if ntype == libsbml.AST_FUNCTION_QUOTIENT:       # integer part of a / b, toward zero
        return float(math.trunc(args[0] / args[1]))
    if ntype == libsbml.AST_FUNCTION_RATE_OF or _is_rate_of_call(node):
        raise NotImplementedError("rateOf(x) is only evaluated in the symbolic model "
                                  "(it is replaced by x's own rate of change)")
    if ntype in _UNARY_MATH_FUNCS:
        return float(_UNARY_MATH_FUNCS[ntype](args[0]))

    raise NotImplementedError(
        f"Unsupported SBML math construct '{node.getName() or ntype}' "
        f"(node type {ntype}) -- this formula uses a feature the generic "
        f"evaluator doesn't cover yet (e.g. a user-defined functionDefinition)."
    )


# ---------------------------------------------------------------------------
# SBML -> structured model data (species, params, rules, reactions)
# ---------------------------------------------------------------------------

def _ast_names(node) -> set:
    if node is None:
        return set()
    names = ({node.getName()} if node.isName()
             and node.getType() != libsbml.AST_NAME_AVOGADRO else set())
    for i in range(node.getNumChildren()):
        names |= _ast_names(node.getChild(i))
    return names


def load_generic_sbml(path) -> dict:
    """
    Parse any SBML file into a plain-data description, reading everything
    (species, parameters, compartments, rateRules, reaction stoichiometry and
    kinetic laws) directly from the XML. Raises ValueError with a specific
    reason if the file can't be turned into an ODE system this way (e.g. it
    only has assignment/algebraic rules, or no dynamics at all).
    """
    path = Path(path) if Path(path).exists() else resolve_sbml_path(str(path))
    doc = read_sbml_expanded(path)
    n_errors = doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR)
    if n_errors > 0:
        msgs = [doc.getError(i).getMessage() for i in range(doc.getNumErrors())
                if doc.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR]
        raise ValueError(f"SBML parse errors in {path.name}: {'; '.join(msgs)}")

    model = doc.getModel()
    if model is None:
        raise ValueError(f"No <model> element found in {path.name}.")

    compartments = {
        c.getId(): (c.getSize() if c.isSetSize() else 1.0)
        for c in model.getListOfCompartments()
    }

    species_info = {}
    for s in model.getListOfSpecies():
        if s.isSetInitialConcentration():
            init = s.getInitialConcentration()
        elif s.isSetInitialAmount():
            init = s.getInitialAmount()
        else:
            init = 0.0
        species_info[s.getId()] = {
            "initial": float(init),
            "initial_is_amount": bool(s.isSetInitialAmount()),
            "constant": bool(s.getConstant()),
            "boundary": bool(s.getBoundaryCondition()),
            "compartment": s.getCompartment(),
            # SBML: a reaction rate is an AMOUNT per time; a species in concentration
            # units changes by rate / (its compartment's volume)
            "amount_units": bool(s.getHasOnlySubstanceUnits() or
                                 (model.getCompartment(s.getCompartment()).isSetSpatialDimensions()
                                  and model.getCompartment(s.getCompartment()).getSpatialDimensions() == 0)),
            "name": s.getName() or s.getId(),
        }

    global_params = {p.getId(): float(p.getValue()) for p in model.getListOfParameters()}
    param_titles = {p.getId(): p.getName() for p in model.getListOfParameters()
                    if p.getName() and p.getName() != p.getId()}

    rate_rules = {}
    assign = {}
    algebraic = []
    for r in model.getListOfRules():
        if r.isRate():
            target = model.getElementBySId(r.getVariable())
            if target is None or not hasattr(target, "getConstant"):
                raise ValueError(f"Unsupported rate-rule target: {r.getVariable()}")
            if target.getConstant():
                raise ValueError(f"Rate-rule target '{r.getVariable()}' is declared constant.")
            rate_rules[r.getVariable()] = r.getMath()
        elif r.isAssignment():
            assign[r.getVariable()] = r.getMath()
        else:
            algebraic.append(r.getMath())
    # an algebraic rule (0 = expression) determines the one variable nothing else does:
    # solved for it, it is an assignment rule (refused, with the reason, when it cannot be)
    assign.update(_solve_algebraic_rules(path, model, algebraic, rate_rules, assign))
    # Initial assignments set starting values / parameter values / compartment sizes from
    # expressions: evaluated here, in dependency order (they were silently ignored before).
    caveats = _apply_initial_assignments(model, compartments, species_info, global_params, assign)
    caveats += _unsupported_constructs(model, set(assign))
    # Assignment rules the dynamics read (directly, or through other rules)
    # are substituted, in dependency order; the rest are observables (e.g. a
    # fluorescence readout) that cannot change the dynamics.
    read = set()
    for m in [*rate_rules.values(), *(rx.getKineticLaw().getMath() for rx in model.getListOfReactions()
                                      if rx.getKineticLaw() is not None)]:
        read |= _ast_names(m)
    for event in model.getListOfEvents():
        for part in (event.getTrigger(), event.getDelay(), event.getPriority(),
                     *event.getListOfEventAssignments()):
            if part is not None and part.isSetMath():
                read |= _ast_names(part.getMath())
    needed, frontier = set(), read & set(assign)
    while frontier:
        needed |= frontier
        frontier = {n for v in frontier for n in _ast_names(assign[v])} & set(assign) - needed
    observables = [v for v in assign if v not in needed]
    assignment_rules, done = [], set()
    while len(done) < len(needed):
        ready = [v for v in assign if v in needed and v not in done
                 and not ((_ast_names(assign[v]) & needed) - done)]
        if not ready:
            raise ValueError(f"{path.name}: assignment rules for {sorted(needed - done)} "
                             f"depend on each other in a cycle.")
        for v in ready:
            assignment_rules.append((v, assign[v]))
            done.add(v)
    for v in assign:  # rule-defined quantities are not free parameters
        global_params.pop(v, None)

    reactions = []
    for rxn in model.getListOfReactions():
        reactants = [
            (sr.getSpecies(), sr.getStoichiometry() if sr.isSetStoichiometry() else 1.0)
            for sr in rxn.getListOfReactants()
        ]
        products = [
            (sr.getSpecies(), sr.getStoichiometry() if sr.isSetStoichiometry() else 1.0)
            for sr in rxn.getListOfProducts()
        ]
        kl = rxn.getKineticLaw()
        math_ast = kl.getMath() if kl is not None else None
        local_params = {}
        if kl is not None:
            get_local = getattr(kl, "getListOfLocalParameters", None)
            if get_local is not None:
                for lp in kl.getListOfLocalParameters():
                    local_params[lp.getId()] = float(lp.getValue())
            for lp in kl.getListOfParameters():  # SBML L2-style, usually empty in L3
                local_params.setdefault(lp.getId(), float(lp.getValue()))
        reactions.append({
            "id": rxn.getId(), "name": rxn.getName() or "", "reactants": reactants,
            "products": products, "math": math_ast, "local_params": local_params,
            "reversible": bool(rxn.getReversible()),
        })

    reaction_species = {sp for rxn in reactions for sp, _ in rxn["reactants"] + rxn["products"]}

    dynamic_species = [
        sp_id for sp_id in species_info
        if sp_id not in assign and (
            sp_id in rate_rules
            or (sp_id in reaction_species and not species_info[sp_id]["constant"]
                and not species_info[sp_id]["boundary"]))
    ]

    dynamic_parameters = {p.getId(): global_params[p.getId()]
                          for p in model.getListOfParameters() if p.getId() in rate_rules}
    for name in dynamic_parameters:
        if model.getParameter(name).getConstant():
            raise ValueError(f"Rate-rule target '{name}' is declared constant.")
        global_params.pop(name)
    unknown_rates = set(rate_rules) - set(species_info) - set(compartments) - set(dynamic_parameters)
    if unknown_rates:
        raise ValueError(f"Unsupported rate-rule targets: {sorted(unknown_rates)}")
    if not dynamic_species and not dynamic_parameters and not (set(compartments) & set(rate_rules)):
        raise ValueError(
            f"{path.name} has no rateRules and no reactions that produce a "
            f"dynamic species -- nothing to integrate."
        )

    return {
        "path": path, "compartments": compartments, "species_info": species_info,
        "global_params": global_params, "param_titles": param_titles, "rate_rules": rate_rules,
        "reactions": reactions, "dynamic_species": dynamic_species,
        "dynamic_parameters": dynamic_parameters,
        "ignored_observables": observables,
        "assignment_rules": assignment_rules,
        "events": _read_events(model),
        # compartments whose size a rate rule changes: state variables (build_symbolic_model)
        "dynamic_compartments": [c.getId() for c in model.getListOfCompartments()
                                 if c.getId() in rate_rules and not c.getConstant()],
        # Keep the SBMLDocument alive for as long as model_data is alive: the
        # ASTNode objects stashed above (rate_rules values, reactions[i]["math"])
        # are raw pointers into memory owned by this document. Once `doc` is
        # garbage-collected, libsbml frees that memory and those ASTNodes
        # become dangling -- using them then segfaults the interpreter rather
        # than raising a catchable Python exception. This reference is the fix.
        "_doc": doc,
        "caveats": caveats,
    }


def _solve_algebraic_rules(path, model, rules, rate_rules, assign) -> dict:
    """{variable: math} for each algebraic rule 0 = f, solved for the one variable it
    determines: a non-constant species, parameter or compartment in f that no reaction,
    rate rule, assignment rule or earlier algebraic rule sets. A rule with no such variable,
    several, or no unique closed-form solution is a differential-algebraic system this
    reader does not handle: ValueError with the reason."""
    if not rules:
        return {}
    changed_by_reactions = {
        ref.getSpecies() for rx in model.getListOfReactions()
        for ref in list(rx.getListOfReactants()) + list(rx.getListOfProducts())}
    fixed = {s.getId() for s in model.getListOfSpecies() if s.getConstant()}
    fixed |= {p.getId() for p in model.getListOfParameters() if p.getConstant()}
    fixed |= {c.getId() for c in model.getListOfCompartments() if c.getConstant()}
    boundary = {s.getId() for s in model.getListOfSpecies() if s.getBoundaryCondition()}
    set_already = set(rate_rules) | set(assign) | (changed_by_reactions - boundary) | fixed
    solved = {}
    for math in rules:
        text = libsbml.formulaToL3String(math)
        names = sorted(_ast_names(math) - set_already - set(solved))
        if len(names) != 1:
            raise ValueError(
                f"{path.name} has an algebraic rule (0 = {text}) that does not determine exactly "
                f"one variable (undetermined: {names or 'none'}): a differential-algebraic system "
                "this reader does not solve.")
        var = names[0]
        symbols = {n: sp.Symbol(n) for n in _ast_names(math)}
        expr = eval_ast_sympy(math, {**symbols, SBML_TIME.name: SBML_TIME})
        roots = sp.solve(expr, sp.Symbol(var))
        if len(roots) != 1:
            raise ValueError(
                f"{path.name} has an algebraic rule (0 = {text}) with "
                f"{'no' if not roots else len(roots)} closed-form solution(s) for {var}: a "
                "differential-algebraic system this reader does not solve.")
        formula = sp.sstr(roots[0].subs(SBML_TIME, sp.Symbol("time"))).replace("**", "^")
        node = libsbml.parseL3Formula(formula)
        if node is None:
            raise ValueError(f"{path.name}: the algebraic rule 0 = {text}, solved for {var}, "
                             f"could not be read back ({formula}).")
        solved[var] = node
    return solved


def _read_events(model) -> list[dict]:
    """The model's events as plain data (math as ASTs, owned by the document): trigger,
    the trigger's value before time 0 (initialValue; SBML L2: true, so a trigger already
    true at the start does not fire), persistence, delay, whether assignments use the
    values at trigger time, and the assignments, in document order (SBML's order when no
    priorities decide)."""
    events = []
    for ev in model.getListOfEvents():
        trig = ev.getTrigger()
        if trig is None or not trig.isSetMath():
            continue
        events.append({
            "id": ev.getId() or f"event_{len(events) + 1}",
            "trigger": trig.getMath(),
            "initial_value": bool(trig.getInitialValue()) if trig.isSetInitialValue() else True,
            "persistent": bool(trig.getPersistent()) if trig.isSetPersistent() else True,
            "delay": ev.getDelay().getMath() if ev.isSetDelay() and ev.getDelay().isSetMath() else None,
            "use_trigger_values": (bool(ev.getUseValuesFromTriggerTime())
                                   if ev.isSetUseValuesFromTriggerTime() else True),
            "has_priority": ev.isSetPriority(),
            "assignments": [(ea.getVariable(), ea.getMath()) for ea in ev.getListOfEventAssignments()
                            if ea.isSetMath()],
        })
    return events


def _apply_initial_assignments(model, compartments, species_info, global_params, assignment_rules=None) -> list:
    """Resolve initial values by dependency, in SBML symbol units (never XML order)."""
    initial = {ia.getSymbol(): ia.getMath() for ia in model.getListOfInitialAssignments()
               if ia.isSetMath()}
    rules = (assignment_rules if assignment_rules is not None else
             {r.getVariable(): r.getMath() for r in model.getListOfRules() if r.isAssignment()})
    expressions = {**rules, **initial}
    defaults = {**compartments, **global_params,
                **{k: v["initial"] for k, v in species_info.items()}, "time": 0.0}
    values, visiting = {}, set()

    def resolve(name):
        if name in values:
            return values[name]
        if name in visiting:
            raise ValueError(f"Cyclic initial-value dependency involving '{name}'.")
        visiting.add(name)
        if name in expressions:
            node = expressions[name]
            env = {n: resolve(n) for n in _ast_names(node)}
            value = float(eval_ast(node, env))
        else:
            if name not in defaults:
                raise ValueError(f"Initial value references unknown symbol '{name}'.")
            value = float(defaults[name])
            info = species_info.get(name)
            if info and info["initial_is_amount"] != info["amount_units"]:
                volume = resolve(info["compartment"])
                if not math.isfinite(volume) or volume <= 0:
                    raise ValueError(f"Invalid initial compartment size for '{name}': {volume}")
                value = value * volume if info["amount_units"] else value / volume
        if not math.isfinite(value):
            raise ValueError(f"Non-finite initial value for '{name}'.")
        visiting.remove(name)
        values[name] = value
        return value

    # Assignment-rule observables need not have an initial value unless another
    # initial expression reads them. Compartment rules affect species unit conversion.
    for name in list(initial) + list(compartments) + [s for s in species_info if s not in rules]:
        resolve(name)
    for name, value in values.items():
        if name in species_info:
            species_info[name]["initial"] = value
        elif name in compartments:
            compartments[name] = value
        elif name in global_params:
            global_params[name] = value
    return []


def _unsupported_constructs(model, dynamic_targets: set) -> list:
    """SBML features this ODE reader does not apply, so a result may not be the model's
    full behaviour. Reported with every analysis, never silently dropped."""
    out = []
    if model.getNumEvents():
        out.append(f"{model.getNumEvents()} event(s) (discrete jumps / switches): applied in "
                   "simulations; steady-state, bifurcation, CRNT and stochastic analyses use "
                   "only the continuous dynamics (events NOT applied there)")
    if model.isSetConversionFactor() or any(s.isSetConversionFactor() for s in model.getListOfSpecies()):
        out.append("a conversion factor is declared and NOT applied to the reaction rates")
    changing = [c.getId() for c in model.getListOfCompartments()
                if c.getId() in dynamic_targets and not c.getConstant()]
    if changing:      # (a size set by a rate rule is a state variable: build_symbolic_model)
        out.append(f"compartment size(s) {changing} are set by an assignment rule; they are "
                   "treated as constants (the dilution of their concentrations is not applied)")
    for r in model.getListOfReactions():
        for ref in list(r.getListOfReactants()) + list(r.getListOfProducts()):
            if ref.isSetId() and (ref.getId() in dynamic_targets or model.getInitialAssignmentBySymbol(ref.getId())):
                out.append(f"stoichiometry of a species reference in reaction '{r.getId()}' is variable; "
                           "its file value is used")
                break
    for r in model.getListOfReactions():
        if r.isSetFast() and r.getFast():
            out.append(f"reaction '{r.getId()}' is marked fast (a quasi-steady-state assumption) and is "
                       "integrated like any other")
            break
    return out


def volume_of(model_data: dict, sp_id: str, symbolic: bool = False):
    """What a reaction's rate (amount / time) is divided by for species sp_id:
    its compartment's volume when it is in concentration units, else 1."""
    info = model_data["species_info"][sp_id]
    if info.get("amount_units"):
        return 1
    comp = info.get("compartment")
    if comp in model_data.get("dynamic_compartments", []):
        return sp.Symbol(comp)
    size = model_data["compartments"].get(comp)
    if size is None or size == 0 or size == 1:
        return 1
    return sp.Symbol(comp) if symbolic else float(size)


def _build_env(model_data: dict, y: np.ndarray, params_used: dict) -> dict:
    dynamic_species = model_data["dynamic_species"]
    env = dict(model_data["compartments"])
    env.update(params_used)
    for sp_id, info in model_data["species_info"].items():
        env[sp_id] = float(y[dynamic_species.index(sp_id)]) if sp_id in dynamic_species else info["initial"]
    for var, math_ast in model_data.get("assignment_rules", []):
        env[var] = eval_ast(math_ast, env)
    return env


def _make_rhs(model_data: dict, params_used: dict):
    dynamic_species = model_data["dynamic_species"]
    idx_of = {sp: i for i, sp in enumerate(dynamic_species)}

    def rhs(t, y):
        env = _build_env(model_data, y, params_used)
        dydt = np.zeros(len(dynamic_species))
        for sp_id, math_ast in model_data["rate_rules"].items():
            if sp_id in idx_of:
                dydt[idx_of[sp_id]] += eval_ast(math_ast, env)
        for rxn in model_data["reactions"]:
            if rxn["math"] is None:
                continue
            local_env = dict(env)
            local_env.update(rxn["local_params"])
            rate = eval_ast(rxn["math"], local_env)
            for sp, stoich in rxn["reactants"]:
                if sp in idx_of:
                    dydt[idx_of[sp]] -= stoich * rate / volume_of(model_data, sp)
            for sp, stoich in rxn["products"]:
                if sp in idx_of:
                    dydt[idx_of[sp]] += stoich * rate / volume_of(model_data, sp)
        return dydt

    return rhs


def simulate_generic_sbml(path, t_end: float = 100.0, n_points: int = 200,
                           y0: dict | None = None, params: dict | None = None):
    """
    Returns (ts, ys, species_names, params_used) for ANY SBML file loadable by
    load_generic_sbml -- ys has shape (n_points, len(species_names)). Raises
    ValueError on an unknown override name, RuntimeError if the integrator
    fails to converge.
    """
    model = build_symbolic_model(path, keep_time=True, allow_delay=True)
    dynamic_species = model["state_names"]

    params_used = dict(model["param_defaults"])
    if params:
        for k, v in params.items():
            if k not in params_used:
                raise ValueError(f"Unknown parameter '{k}'. Valid: {list(params_used.keys())}")
            params_used[k] = v

    y0_arr = np.array([model["y0"][name] for name in dynamic_species])
    if y0:
        for sp, val in y0.items():
            if sp not in dynamic_species:
                raise ValueError(f"Unknown or non-dynamic species '{sp}'. Dynamic species: {dynamic_species}")
            y0_arr[dynamic_species.index(sp)] = val

    ts = np.linspace(0.0, t_end, n_points)
    from dynalyst.analysis.hybrid_sim import simulate
    ys, _ = simulate(model, y0_arr, params_used, t_end, ts)
    return ts, ys, dynamic_species, params_used


def get_generic_odes(path) -> dict:
    """
    Structural description of a model read generically from its SBML file --
    for the get_model_odes tool's fallback path. Rate laws are rendered as
    formula strings via libsbml (no hand transcription).
    """
    model_data = load_generic_sbml(path)
    dynamic_species = model_data["dynamic_species"]

    odes = {}
    for sp in dynamic_species:
        terms = []
        if sp in model_data["rate_rules"]:
            terms.append(libsbml.formulaToL3String(model_data["rate_rules"][sp]))
        for rxn in model_data["reactions"]:
            if rxn["math"] is None:
                continue
            formula = libsbml.formulaToL3String(rxn["math"])
            for r_sp, stoich in rxn["reactants"]:
                if r_sp == sp:
                    terms.append(f"-{stoich if stoich != 1 else ''}({formula})")
            for p_sp, stoich in rxn["products"]:
                if p_sp == sp:
                    terms.append(f"+{stoich if stoich != 1 else ''}({formula})")
        odes[f"d{sp}/dt"] = " ".join(terms) if terms else "0"

    constant_species = [
        sp for sp, info in model_data["species_info"].items()
        if sp not in dynamic_species
    ]

    return {
        "model_file": str(model_data["path"]),
        "dynamic_species": dynamic_species,
        "constant_or_boundary_species_used_as_modifiers": constant_species,
        "odes": odes,
        "parameters": model_data["global_params"],
        "default_initial_conditions": {
            sp: model_data["species_info"][sp]["initial"] for sp in dynamic_species
        },
        "source": "auto-derived from the SBML file's rateRules/reactions via libsbml "
                   "(generic_sbml.py) -- not hand-coded.",
    }


# ---------------------------------------------------------------------------
# SYMBOLIC (SymPy) build -- generalizes bifurcation_1d.py's approach to ANY
# dimension. This is the shared foundation for get_odes / steady-state finding
# / AUTO-07p bifurcation continuation / parameter scans: everything downstream
# works off one sympy expression per dynamic species, instead of hand-rolled
# per-tool SBML parsing.
#
# Mirrors eval_ast's node-by-node walk, but builds sp.Expr instead of a float,
# against a {name: sympy.Symbol} environment instead of {name: float}. Same
# structural limitations as load_generic_sbml apply (no assignment/algebraic
# rules, no events, no functionDefinitions).
# ---------------------------------------------------------------------------

_RELATIONAL_MATH = {
    libsbml.AST_RELATIONAL_EQ: lambda a, b: a == b,
    libsbml.AST_RELATIONAL_NEQ: lambda a, b: a != b,
    libsbml.AST_RELATIONAL_LT: lambda a, b: a < b,
    libsbml.AST_RELATIONAL_LEQ: lambda a, b: a <= b,
    libsbml.AST_RELATIONAL_GT: lambda a, b: a > b,
    libsbml.AST_RELATIONAL_GEQ: lambda a, b: a >= b,
}

# placeholders the symbolic model resolves once its rates are known
RATE_OF = sp.Function("rateOf")
DELAY = sp.Function("delay")

_UNARY_SYMPY_FUNCS = {
    libsbml.AST_FUNCTION_ABS: sp.Abs,
    libsbml.AST_FUNCTION_EXP: sp.exp,
    libsbml.AST_FUNCTION_LN: sp.log,
    libsbml.AST_FUNCTION_SIN: sp.sin,
    libsbml.AST_FUNCTION_COS: sp.cos,
    libsbml.AST_FUNCTION_TAN: sp.tan,
    libsbml.AST_FUNCTION_SINH: sp.sinh,
    libsbml.AST_FUNCTION_COSH: sp.cosh,
    libsbml.AST_FUNCTION_TANH: sp.tanh,
    libsbml.AST_FUNCTION_FLOOR: sp.floor,
    libsbml.AST_FUNCTION_CEILING: sp.ceiling,
    libsbml.AST_FUNCTION_SEC: sp.sec,
    libsbml.AST_FUNCTION_CSC: sp.csc,
    libsbml.AST_FUNCTION_COT: sp.cot,
    libsbml.AST_FUNCTION_ARCSIN: sp.asin,
    libsbml.AST_FUNCTION_ARCCOS: sp.acos,
    libsbml.AST_FUNCTION_ARCTAN: sp.atan,
    libsbml.AST_FUNCTION_ARCSEC: lambda x: sp.acos(1 / x),
    libsbml.AST_FUNCTION_ARCCSC: lambda x: sp.asin(1 / x),
    libsbml.AST_FUNCTION_ARCCOT: lambda x: sp.atan(1 / x),
    libsbml.AST_FUNCTION_SECH: sp.sech,
    libsbml.AST_FUNCTION_CSCH: sp.csch,
    libsbml.AST_FUNCTION_COTH: sp.coth,
    libsbml.AST_FUNCTION_ARCSINH: sp.asinh,
    libsbml.AST_FUNCTION_ARCCOSH: sp.acosh,
    libsbml.AST_FUNCTION_ARCTANH: sp.atanh,
    libsbml.AST_FUNCTION_ARCSECH: lambda x: sp.acosh(1 / x),
    libsbml.AST_FUNCTION_ARCCSCH: lambda x: sp.asinh(1 / x),
    libsbml.AST_FUNCTION_ARCCOTH: lambda x: sp.atanh(1 / x),
    libsbml.AST_FUNCTION_FACTORIAL: lambda x: sp.gamma(x + 1),   # x! = Gamma(x + 1)
}


_RELATIONAL_SYMPY = {
    libsbml.AST_RELATIONAL_LT: sp.StrictLessThan,
    libsbml.AST_RELATIONAL_LEQ: sp.LessThan,
    libsbml.AST_RELATIONAL_GT: sp.StrictGreaterThan,
    libsbml.AST_RELATIONAL_GEQ: sp.GreaterThan,
    libsbml.AST_RELATIONAL_EQ: sp.Eq,
    libsbml.AST_RELATIONAL_NEQ: sp.Ne,
}


class TimeDependentModel(ValueError):
    """The model's equations use SBML's `time` symbol in a way that cannot be
    frozen into a constant input (no long-time value: periodic or unbounded
    forcing), so it has no fixed steady states to continue."""


TIME_DEPENDENT_MESSAGE = (
    "its equations depend explicitly on time (SBML's `time` symbol). Steady states and "
    "bifurcation diagrams need an autonomous model whose equations do not contain time.")

# SBML's `time` csymbol inside symbolic expressions (a name no SBML id can take)
SBML_TIME = sp.Symbol("__sbml_time__")


def _as_number(value):
    try:
        v = complex(value)
    except (TypeError, ValueError):
        return None
    if abs(v.imag) > 1e-12 or not np.isfinite(v.real):
        return None
    return float(v.real)


def _long_time_value(expr):
    """The value of a numeric expression of SBML_TIME as time -> infinity, or
    None when it has none (periodic, unbounded, undecidable)."""
    try:
        v = _as_number(expr.subs(SBML_TIME, sp.oo))
        if v is not None:
            return v
    except Exception:
        pass
    try:
        v = _as_number(sp.limit(expr, SBML_TIME, sp.oo))
        if v is not None:
            return v
    except Exception:
        pass
    # numerically: the value must settle at large times
    try:
        vals = [_as_number(expr.subs(SBML_TIME, 10.0 ** k)) for k in (6, 9, 12)]
    except Exception:
        return None
    if all(v is not None for v in vals) and abs(vals[2] - vals[1]) <= 1e-9 * (1 + abs(vals[2])) \
            and abs(vals[1] - vals[0]) <= 1e-6 * (1 + abs(vals[1])):
        return vals[2]
    return None


def freeze_time_input(name, expr, numeric_subs):
    """Freeze a time-dependent input `name` = expr(time, parameters) into a
    constant. Returns (value, basis, long_time_expression_text).

    The value is the input's long-time value (time -> infinity: after every
    stimulus has switched on, e.g. a step input at its final level). When that
    is 0 -- a dose that decays away, so the long-time model has no input at all
    -- the value at time 0 (the dose applied) is used instead. An input with no
    long-time value (periodic or unbounded forcing) cannot be frozen: raises
    TimeDependentModel."""
    num = expr.subs(numeric_subs)
    long_time = _long_time_value(num)
    symbolic = None
    try:
        pos = {s: sp.Symbol(s.name, positive=True) for s in expr.free_symbols if s != SBML_TIME}
        lt = expr.xreplace(pos).subs(SBML_TIME, sp.oo)
        if lt.is_finite and not lt.has(sp.nan):
            symbolic = sp.sstr(lt.xreplace({v: k for k, v in pos.items()}))
    except Exception:
        pass
    if long_time is None:
        raise TimeDependentModel(
            f"The input '{name}' = {sp.sstr(expr.subs(SBML_TIME, sp.Symbol('time')))} has no "
            f"long-time value (periodic or unbounded forcing), so it cannot be frozen into a "
            f"constant: the model has no fixed steady states to continue.")
    if long_time != 0:
        return long_time, "its long-time value (time -> infinity)", symbolic
    at_zero = _as_number(num.subs(SBML_TIME, 0))
    if at_zero not in (None, 0.0):
        return at_zero, ("its value at time 0 (it decays to 0, which would remove the input "
                         "altogether)"), symbolic
    return 0.0, "its long-time value (time -> infinity)", symbolic


def eval_ast_sympy(node, env: dict):
    """Symbolic twin of eval_ast: walks a libsbml ASTNode and returns a sympy
    expression, given a {name: sympy.Symbol-or-number} environment."""
    if node is None:
        return sp.Integer(0)

    if node.isNumber():
        return sp.Float(node.getValue())
    if node.getType() == libsbml.AST_NAME_AVOGADRO:
        return sp.Float(AVOGADRO)

    if node.getType() == libsbml.AST_NAME_TIME:
        if SBML_TIME.name in env:
            return env[SBML_TIME.name]
        raise TimeDependentModel(f"This model cannot be analysed here: {TIME_DEPENDENT_MESSAGE}")

    if node.isName():
        name = node.getName()
        if name not in env:
            raise KeyError(
                f"Symbol '{name}' used in SBML math has no symbolic binding "
                f"(not a known species, parameter, or compartment)."
            )
        return env[name]

    ntype = node.getType()
    if ntype == libsbml.AST_CONSTANT_PI:
        return sp.pi
    if ntype == libsbml.AST_CONSTANT_E:
        return sp.E
    if ntype == libsbml.AST_CONSTANT_TRUE:
        return sp.Integer(1)
    if ntype == libsbml.AST_CONSTANT_FALSE:
        return sp.Integer(0)

    n = node.getNumChildren()
    args = [eval_ast_sympy(node.getChild(i), env) for i in range(n)]

    if ntype == libsbml.AST_PLUS:
        return sp.Add(*args) if args else sp.Integer(0)
    if ntype == libsbml.AST_MINUS:
        return (args[0] - args[1]) if len(args) == 2 else -args[0]
    if ntype == libsbml.AST_TIMES:
        result = sp.Integer(1)
        for a in args:
            result *= a
        return result
    if ntype == libsbml.AST_DIVIDE:
        return args[0] / args[1]
    if ntype in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
        return args[0] ** args[1]
    if ntype == libsbml.AST_FUNCTION_ROOT:
        return args[1] ** (1 / args[0]) if len(args) == 2 else sp.sqrt(args[0])
    if ntype == libsbml.AST_FUNCTION_LOG:
        return sp.log(args[0], 10) if len(args) == 1 else sp.log(args[1], args[0])
    if ntype == libsbml.AST_FUNCTION_PIECEWISE:
        pairs = []
        for i in range(0, n - 1, 2):
            pairs.append((args[i], args[i + 1]))
        if n % 2 == 1:
            pairs.append((args[-1], True))
        return sp.Piecewise(*pairs)
    if ntype in _UNARY_SYMPY_FUNCS:
        return _UNARY_SYMPY_FUNCS[ntype](args[0])
    if ntype in _RELATIONAL_SYMPY:
        # SBML allows chains (a < b < c): every neighbouring pair must hold
        rel = _RELATIONAL_SYMPY[ntype]
        pairs = [rel(args[i], args[i + 1]) for i in range(len(args) - 1)]
        return sp.And(*pairs) if len(pairs) > 1 else pairs[0]
    if ntype == libsbml.AST_LOGICAL_AND:
        return sp.And(*args)
    if ntype == libsbml.AST_LOGICAL_OR:
        return sp.Or(*args)
    if ntype == libsbml.AST_LOGICAL_XOR:
        return sp.Xor(*args)
    if ntype == libsbml.AST_LOGICAL_NOT:
        return sp.Not(args[0])
    if ntype == libsbml.AST_LOGICAL_IMPLIES:
        return sp.Implies(args[0], args[1])
    if ntype == libsbml.AST_FUNCTION_MIN:
        return sp.Min(*args)
    if ntype == libsbml.AST_FUNCTION_MAX:
        return sp.Max(*args)
    if ntype in (libsbml.AST_FUNCTION_REM, libsbml.AST_FUNCTION_QUOTIENT):
        q = args[0] / args[1]
        whole = sp.sign(q) * sp.floor(sp.Abs(q))          # integer part, toward zero
        return whole if ntype == libsbml.AST_FUNCTION_QUOTIENT else args[0] - args[1] * whole
    if ntype == libsbml.AST_FUNCTION_RATE_OF or _is_rate_of_call(node):
        return RATE_OF(args[0])       # replaced by the variable's own rate (build_symbolic_model)
    if ntype == libsbml.AST_FUNCTION_DELAY:
        return DELAY(args[0], args[1])

    raise NotImplementedError(
        f"Unsupported SBML math construct '{node.getName() or ntype}' "
        f"(node type {ntype}) in symbolic mode -- this formula uses a feature "
        f"not covered by the generic symbolic builder yet."
    )


_UNIT_FLOATS = {sp.Float(1.0): sp.Integer(1), sp.Float(-1.0): sp.Integer(-1)}


def _resolve_rate_of(rhs, state_symbols):
    """rateOf(x) replaced by x's right-hand side (a state) or 0 (a constant: parameter,
    compartment, fixed species). Repeated, since one rate may use another's rateOf."""
    rate = dict(zip(state_symbols, rhs))
    out = list(rhs)
    for _ in range(len(out) + 1):
        if not any(e.has(RATE_OF) for e in out):
            return out
        out = [e.replace(lambda x: getattr(x, "func", None) == RATE_OF,
                         lambda x: rate.get(x.args[0], sp.Integer(0))) for e in out]
        rate = dict(zip(state_symbols, out))
    raise NotImplementedError("rateOf(x) refers back to itself through the rates (a "
                              "circular definition): the model's equations are not explicit")


def _set_by_rule_or_boundary(model_data, s_id) -> bool:
    """SBML: reactions do not change a boundary species, and a species with a rate rule is
    changed by that rule alone. Adding the reactions as well counts its change twice
    (BIOMD0000000006 gives u and v both a rate rule and the equivalent reactions)."""
    info = model_data["species_info"].get(s_id, {})
    return bool(info.get("boundary")) or s_id in model_data["rate_rules"]


def build_symbolic_model(path, keep_time: bool = False, allow_delay: bool = False) -> dict:
    """
    Build a fully symbolic, arbitrary-dimensional ODE model straight from an
    SBML file, reusing load_generic_sbml's structural parse. Returns:

        {
          "path": Path,
          "state_names": [str, ...],              # dynamic species, in order
          "state_symbols": [sp.Symbol, ...],
          "param_names": [str, ...],               # every other free symbol
          "param_symbols": [sp.Symbol, ...],
          "param_defaults": {name: float},
          "rhs": [sp.Expr, ...],                   # dstate_i/dt, same order as state_names
          "y0": {name: float},                     # default initial conditions
        }

    This is the shared representation used by steady_states.py,
    and bifurcation_auto.py -- build it once per model,
    reuse everywhere.

    A model that uses delay(x, tau) raises DelayModel, unless allow_delay (only for
    reading the equations): then the right-hand sides keep delay(x, tau) and "delays"
    lists each delayed term.
    """
    model_data = load_generic_sbml(path)
    dyn_comps = model_data.get("dynamic_compartments", [])
    dyn_params = model_data.get("dynamic_parameters", {})
    dynamic_species = model_data["dynamic_species"] + dyn_comps + list(dyn_params)
    state_symbols = [sp.Symbol(s) for s in dynamic_species]
    env = {name: sym for name, sym in zip(dynamic_species, state_symbols)}

    # Every quantity the user can tune is a parameter -- discovered from the file,
    # never guessed: (1) the global parameters, in the file's own order; (2) species
    # held fixed (boundaryCondition / constant, e.g. a clamped substrate), which are
    # the model's inputs; (3) reaction-local
    # parameters, named '<reaction>_<parameter>' (unique; a bare local name that is
    # unambiguous also works, see param_aliases). Only compartment sizes stay numbers.
    numeric_env = {c: v for c, v in model_data["compartments"].items() if c not in dyn_comps}
    rule_targets = {var for var, _ in model_data["assignment_rules"]}
    param_names = list(model_data["global_params"].keys())
    param_defaults = dict(model_data["global_params"])
    param_titles = dict(model_data.get("param_titles", {}))
    param_kinds = {p: "global parameter" for p in param_names}
    for sp_id, info in model_data["species_info"].items():
        if sp_id in dynamic_species or sp_id in rule_targets:
            continue
        param_names.append(sp_id)
        param_defaults[sp_id] = float(info["initial"])
        param_kinds[sp_id] = "fixed species"
        param_titles.setdefault(sp_id, f"fixed (boundary) species {sp_id}")
    local_symbols, bare = {}, {}
    taken = set(param_names) | set(dynamic_species) | set(numeric_env) | rule_targets
    for rxn in model_data["reactions"]:
        local_symbols[id(rxn)] = {}
        for lp_id, value in rxn["local_params"].items():
            name, k = f"{rxn['id']}_{lp_id}", 2
            while name in taken:
                name, k = f"{rxn['id']}_{lp_id}_{k}", k + 1
            taken.add(name)
            param_names.append(name)
            param_defaults[name] = float(value)
            param_kinds[name] = "local parameter"
            param_titles.setdefault(name, f"{lp_id} of reaction {rxn['id']}")
            local_symbols[id(rxn)][lp_id] = sp.Symbol(name)
            bare.setdefault(lp_id, []).append(name)
    # a bare local name that is unambiguous (and not itself a model id) -> its parameter
    param_aliases = {b: names[0] for b, names in bare.items()
                     if len(names) == 1 and b not in param_defaults and b not in taken}
    env.update({name: sp.Symbol(name) for name in param_names})
    env.update(numeric_env)
    env[SBML_TIME.name] = SBML_TIME

    # Time-dependent inputs (an assignment rule using SBML's `time`: a stimulus
    # or dose protocol) are frozen into a constant parameter of the same name,
    # so the model is autonomous and the input can itself be swept or sampled.
    time_inputs = []
    numeric_subs = {sp.Symbol(k): v for k, v in param_defaults.items()}
    numeric_subs.update({sp.Symbol(k): v for k, v in numeric_env.items()})
    for var, math_ast in model_data["assignment_rules"]:  # already in dependency order
        expr = eval_ast_sympy(math_ast, env)
        if keep_time and isinstance(expr, sp.Basic) and expr.has(SBML_TIME):
            # simulation: the input follows its own time course (rhs depends on SBML_TIME)
            time_inputs.append({"input": var, "value": None, "basis": "kept as a function of time",
                                "expression": sp.sstr(expr.subs(SBML_TIME, sp.Symbol("time")))})
            env[var] = expr
            continue
        if isinstance(expr, sp.Basic) and expr.has(SBML_TIME):
            value, basis, long_time = freeze_time_input(var, expr, numeric_subs)
            time_inputs.append({
                "input": var, "value": value, "basis": basis,
                "expression": sp.sstr(expr.subs(SBML_TIME, sp.Symbol("time"))),
                "long_time_expression": long_time})
            if var not in param_defaults:
                param_names.append(var)
            param_defaults[var] = value
            numeric_subs[sp.Symbol(var)] = value
            expr = sp.Symbol(var)
        env[var] = expr

    def _no_time(expr, where):
        """Direct uses of time outside an assignment rule are frozen in place (kept, for
        a simulation: keep_time)."""
        if keep_time or not (isinstance(expr, sp.Basic) and expr.has(SBML_TIME)):
            return expr
        frozen = expr.subs(SBML_TIME, sp.oo)
        if frozen.has(sp.nan, sp.zoo, sp.oo, -sp.oo, SBML_TIME):
            raise TimeDependentModel(
                f"{where} uses time in a way that has no long-time value, so it cannot be frozen "
                f"into an autonomous model: {TIME_DEPENDENT_MESSAGE}")
        time_inputs.append({"input": where, "value": None,
                            "basis": "time -> infinity, in place",
                            "expression": sp.sstr(expr.subs(SBML_TIME, sp.Symbol("time"))),
                            "long_time_expression": sp.sstr(frozen)})
        return frozen

    # The same equations for PEOPLE to read (display_env): fixed species and
    # compartments stay named instead of being folded into numbers, and the
    # right-hand sides are not simplified, so each reaction stays visible.
    display_env = dict(env)
    display_env.update({name: sp.Symbol(name) for name in numeric_env})

    def _display(expr):
        return expr.subs(SBML_TIME, sp.oo) if isinstance(expr, sp.Basic) and expr.has(SBML_TIME) else expr

    frozen_inputs = {t["input"] for t in time_inputs}
    display_rules = {}
    for var, math_ast in model_data["assignment_rules"]:
        # rule variables stay named in the equations and are listed on their own
        if var not in frozen_inputs:
            display_rules[var] = _display(eval_ast_sympy(math_ast, display_env))
        display_env[var] = sp.Symbol(var)

    param_symbols = [sp.Symbol(p) for p in param_names]
    rhs = [sp.Integer(0)] * len(dynamic_species)
    rhs_display = [sp.Integer(0)] * len(dynamic_species)
    idx_of = {s: i for i, s in enumerate(dynamic_species)}
    display_rate_rules, display_reactions = {}, []

    for sp_id, math_ast in model_data["rate_rules"].items():
        if sp_id in idx_of:
            rhs[idx_of[sp_id]] += _no_time(eval_ast_sympy(math_ast, env), f"rate rule of {sp_id}")
            display_rate_rules[sp_id] = _display(eval_ast_sympy(math_ast, display_env))
            rhs_display[idx_of[sp_id]] += display_rate_rules[sp_id]

    for rxn in model_data["reactions"]:
        if rxn["math"] is None:
            continue
        local_env = dict(env)
        local_env.update(local_symbols.get(id(rxn), {}))   # local parameters stay symbols
        rate = _no_time(eval_ast_sympy(rxn["math"], local_env), f"rate of reaction {rxn.get('id', '?')}")
        rate_display = _display(eval_ast_sympy(rxn["math"],
                                               {**display_env, **local_symbols.get(id(rxn), {})}))
        display_reactions.append({
            "id": rxn["id"], "name": rxn.get("name", ""), "rate": rate_display,
            "reactants": rxn["reactants"], "products": rxn["products"],
            "reversible": rxn.get("reversible", False),
            "local_parameters": {str(sym): rxn["local_params"][lp]
                                 for lp, sym in local_symbols.get(id(rxn), {}).items()}})
        for s_id, stoich in rxn["reactants"]:
            if s_id in idx_of and not _set_by_rule_or_boundary(model_data, s_id):
                rhs[idx_of[s_id]] -= stoich * rate / volume_of(model_data, s_id)
                rhs_display[idx_of[s_id]] -= stoich * rate_display / volume_of(model_data, s_id, True)
        for s_id, stoich in rxn["products"]:
            if s_id in idx_of and not _set_by_rule_or_boundary(model_data, s_id):
                rhs[idx_of[s_id]] += stoich * rate / volume_of(model_data, s_id)
                rhs_display[idx_of[s_id]] += stoich * rate_display / volume_of(model_data, s_id, True)

    # a compartment that grows or shrinks dilutes or concentrates what is in it: a species
    # whose reactions change its amount (rates / V) also changes as d[S]/dt -= [S] (dV/dt) / V
    for s_id in model_data["dynamic_species"]:
        comp = model_data["species_info"][s_id].get("compartment")
        if (comp in dyn_comps and s_id not in model_data["rate_rules"]
                and not model_data["species_info"][s_id].get("amount_units")):
            i, v = idx_of[s_id], idx_of[comp]
            rhs[i] -= state_symbols[i] * rhs[v] / state_symbols[v]
            rhs_display[i] -= state_symbols[i] * rhs_display[v] / state_symbols[v]

    # delay(x, tau): a delay differential equation, not an ODE -- said, never misread.
    # Only reading the equations (allow_delay) gets them, with each delayed term listed
    delays = sorted({d for e in [*rhs_display, *display_rules.values()] if isinstance(e, sp.Basic)
                     for d in e.atoms(DELAY)}, key=sp.sstr)
    if any(e.has(DELAY) for e in rhs) and not allow_delay:
        raise DelayModel(DELAY_MESSAGE)
    # rateOf(x): x's own rate of change (0 for a constant); rates may refer to each other
    rhs, rhs_display = _resolve_rate_of(rhs, state_symbols), _resolve_rate_of(rhs_display,
                                                                              state_symbols)
    # the right-hand sides stay as the reactions give them: simplify is cosmetic and can run
    # without end on large models. Only unit factors and exponents written as floats
    # (1.0*x, (x + 1)**1.0) are dropped: an exact, linear-time rewrite
    rhs = [e.xreplace(_UNIT_FLOATS) for e in rhs]
    constants = {}
    for sp_id, info in model_data["species_info"].items():
        if (sp_id not in dynamic_species and sp_id not in param_defaults
                and sp_id not in {t["input"] for t in time_inputs}):
            constants[sp_id] = {"value": info["initial"], "kind": "fixed species"}
    for c_id, size in model_data["compartments"].items():
        if c_id not in dyn_comps:
            constants[c_id] = {"value": size, "kind": "compartment"}

    # events, for a simulation (keep_time): trigger, delay and assignments as expressions in
    # the model's own symbols (time kept), each assignment's target classified
    events = []
    if keep_time:
        own_rate = dict(zip(state_symbols, rhs))     # rateOf(x) in an event: x's own rate
        for ev in model_data.get("events", []):
            def expr(ast):
                e = eval_ast_sympy(ast, env)
                if isinstance(e, sp.Basic) and e.has(RATE_OF):
                    e = e.replace(lambda x: getattr(x, "func", None) == RATE_OF,
                                  lambda x: own_rate.get(x.args[0], sp.Integer(0)))
                return e
            targets = []
            for var, ast in ev["assignments"]:
                kind = ("state" if var in idx_of else "param" if var in param_defaults
                        else "compartment" if var in model_data["compartments"] else "other")
                targets.append((var, kind, expr(ast)))
            events.append({"id": ev["id"], "trigger": expr(ev["trigger"]),
                           "initial_value": ev["initial_value"], "persistent": ev["persistent"],
                           "delay": expr(ev["delay"]) if ev["delay"] is not None else None,
                           "use_trigger_values": ev["use_trigger_values"],
                           "has_priority": ev["has_priority"], "assignments": targets})

    return {
        "path": Path(path),
        "state_names": dynamic_species,
        "state_metadata": {
            name: {"kind": "species" if name in model_data["species_info"] else
                           "parameter" if name in dyn_params else "compartment",
                   **{k: v for k, v in model_data["species_info"].get(name, {}).items()
                      if k in ("compartment", "amount_units")}}
            for name in dynamic_species},
        "compartments": dict(model_data["compartments"]),
        "state_symbols": state_symbols,
        "param_names": param_names,
        "param_symbols": param_symbols,
        "param_defaults": param_defaults,
        "param_aliases": param_aliases,      # bare local name -> '<reaction>_<name>'
        "caveats": model_data.get("caveats", []),
        "param_kinds": param_kinds,          # global parameter / fixed species / local parameter
        "rhs": rhs,
        "y0": {sp_: (model_data["species_info"][sp_]["initial"] if sp_ in model_data["species_info"]
                     else float(dyn_params[sp_] if sp_ in dyn_params else
                                model_data["compartments"][sp_])) for sp_ in dynamic_species},
        "time_inputs": time_inputs,
        # keep_time: the right-hand sides depend on time through this symbol
        **({"time_symbol": SBML_TIME} if keep_time else {}),
        **({"events": events} if events else {}),
        # allow_delay: each delay(x, tau) the equations use (x and tau as written)
        **({"delays": [{"term": sp.sstr(d), "variable": sp.sstr(d.args[0]),
                        "delay": sp.sstr(d.args[1])} for d in delays]} if delays else {}),
        # for reading the model (ode_model.ode_report); analysis uses rhs above
        "display": {
            "rhs": rhs_display, "reactions": display_reactions,
            "rate_rules": display_rate_rules, "assignment_rules": display_rules,
            "constants": constants, "param_titles": param_titles,
            "ignored_observables": list(model_data.get("ignored_observables", [])),
        },
    }
