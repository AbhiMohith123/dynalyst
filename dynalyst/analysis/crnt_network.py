"""ODE model -> mass-action reaction network, written as the SBML crnt4sbml
reads.

crnt4sbml reads only the network structure (reactants, products,
stoichiometries) and assumes mass action, so any model is first turned into
an explicit mass-action network, from its equations alone:

1. Species removed through a conservation law come back. Models written by
   hand often replace a species by ``TOTAL - (sum of states)`` (e.g.
   ``E = ET - (E1 + E2 + C)``); every such sub-expression (one
   parameter minus a positive combination of states) becomes a species
   ``<TOTAL>_free`` again, with d/dt = -(the combination's derivative).
2. Every right-hand side is expanded into monomial rate terms
   ``coefficient * rate-constant * prod(species**n)``. One rate term shared
   by several equations is one reaction: its reactant complex is the
   monomial, its product complex the reactant complex plus the term's
   stoichiometric coefficients (the canonical mass-action realisation).
3. A term that cannot come from a mass-action reaction (a non-integer or
   negative exponent, a rational rate law, or a species consumed that its own
   rate does not depend on) is reported, naming the term: CRNT then does not
   apply to the model as written.
"""
from pathlib import Path

import libsbml
import numpy as np
import sympy as sp

from dynalyst import config


class NotMassAction(ValueError):
    """The model is not a mass-action reaction network."""


def _is_total_form(f, state_set, param_set):
    """TOTAL - sum(c_j * x_j), c_j > 0 (either overall sign): (TOTAL, sign) or None."""
    for sign in (1, -1):
        g = sp.expand(sign * f)
        if not isinstance(g, sp.Add):
            continue
        const = [t for t in g.args if t.free_symbols and t.free_symbols <= param_set]
        rest = [t for t in g.args if t not in const]
        if len(const) != 1 or not isinstance(const[0], sp.Symbol) or not rest:
            continue
        if all((lambda c, x: x in state_set and c.is_number and c < 0)(*t.as_coeff_Mul())
               for t in rest):
            return const[0], sign
    return None


def _restore_conserved(rhs, states, params):
    """Bring back species removed through conservation laws.

    Terms are grouped by their rate-constant monomial; each group is factored,
    and factors TOTAL - sum(c_j x_j) become a species <TOTAL>_free. Works on
    expanded equations (as XPP converters write them) and on factored ones.
    Returns (rewritten rhs, {hidden symbol: total expression})."""
    state_set = set(states)
    hidden = {}   # expanded total form -> symbol

    def rewrite(e, rate_params):
        groups = {}
        for term in sp.Add.make_args(sp.expand(e)):
            rate = sp.Mul(*[f for f in sp.Mul.make_args(term)
                            if f.free_symbols and f.free_symbols <= rate_params])
            groups[rate] = groups.get(rate, 0) + term / rate
        out = sp.Integer(0)
        for rate, g in groups.items():
            parts = []
            for f in sp.Mul.make_args(sp.factor(g)):
                base, ex = f.as_base_exp()
                hit = _is_total_form(base, state_set, set(params)) if isinstance(base, sp.Add) else None
                if hit is None:
                    parts.append(f)
                    continue
                total, sign = hit
                key = sp.expand(sign * base)
                hidden.setdefault(key, sp.Symbol(f"{total}_free"))
                parts.append((sign * hidden[key]) ** ex)
            out += rate * sp.Mul(*parts)
        return out

    # pass 1 finds the totals: the coefficient of each single parameter,
    # factored, shows TOTAL - sum(...) factors (e.g. a rate constant's coefficient
    # X*ET - X*C1 - X*C2 = X*(ET - C1 - C2)); pass 2 then groups by
    # the remaining (rate) parameters only
    for e in rhs:
        ex = sp.expand(e)
        for p_ in ex.free_symbols & set(params):
            for f in sp.Mul.make_args(sp.factor(ex.coeff(p_))):
                base = f.as_base_exp()[0]
                hit = _is_total_form(base, state_set, set(params)) if isinstance(base, sp.Add) else None
                if hit:
                    total, sign = hit
                    hidden.setdefault(sp.expand(sign * base), sp.Symbol(f"{total}_free"))
    if not hidden:
        return list(rhs), {}
    totals = {sym for k in hidden for sym in k.free_symbols if sym not in state_set}
    new_rhs = [rewrite(e, set(params) - totals) for e in rhs]
    return new_rhs, {h: k for k, h in hidden.items()}


def _split_term(term, species, params):
    """(stoich coefficient, rate constant expr, exponent tuple) of one monomial."""
    num, rest = term.as_coeff_Mul()
    exps = [0] * len(species)
    k = sp.Integer(1)
    idx = {s: i for i, s in enumerate(species)}
    for f in sp.Mul.make_args(rest):
        base, e = f.as_base_exp()
        if base in idx:
            whole = e.is_number and float(e) > 0 and float(e).is_integer()
            if not whole:
                raise NotMassAction(f"the term {term} has {base}**{e}; mass action needs a "
                                    "positive integer power")
            exps[idx[base]] += int(float(e))
        elif f.free_symbols <= set(params) and not f.free_symbols & set(species):
            if isinstance(f, sp.Pow) and f.exp.is_negative and base.free_symbols & set(species):
                raise NotMassAction(f"the term {term} divides by a species")
            k = k * f
        else:
            raise NotMassAction(f"the term {term} is not a mass-action rate (factor {f})")
    return num, k, tuple(exps)


def reconstruct(state_names, rhs, param_names, param_values, y0):
    """The mass-action network behind an ODE model.

    Returns {species, initial, reactions: [{id, reactants, products, rate_constant,
    k_value, model_parameters}], hidden: {species: expression}}.
    """
    states = [sp.Symbol(s) for s in state_names]
    params = [sp.Symbol(p) for p in param_names]
    # exact numbers (1.0*AK**2.0 -> AK**2): sympy only factors exact polynomials
    rhs = [sp.nsimplify(sp.sympify(e), rational=True) for e in rhs]

    rhs, hidden_expr = _restore_conserved(rhs, states, params)
    hidden_syms = list(hidden_expr)
    hidden_rhs = []
    for h in hidden_syms:
        # h = TOTAL - sum(c_j x_j)  =>  dh/dt = -sum(c_j dx_j/dt)
        d = sp.Integer(0)
        for t in sp.Add.make_args(hidden_expr[h]):
            c, x = t.as_coeff_Mul()
            if x in states:
                d += c * rhs[states.index(x)]
        hidden_rhs.append(d)
    species = states + hidden_syms
    all_rhs = rhs + hidden_rhs

    flux = {}  # (exponents, rate constant) -> stoichiometry vector
    for i, e in enumerate(all_rhs):
        for term in sp.Add.make_args(sp.expand(e)):
            if term == 0:
                continue
            num, k, exps = _split_term(term, species, params)
            if not num.is_number:
                raise NotMassAction(f"the term {term} has a non-numeric coefficient")
            key = (exps, k)
            flux.setdefault(key, [0.0] * len(species))[i] += float(num)

    pv = {sp.Symbol(p): v for p, v in param_values.items()}
    reactions = []
    for n, ((exps, k), stoich) in enumerate(flux.items(), start=1):
        st = np.round(np.array(stoich), 9)
        if np.allclose(st, 0):
            continue
        if not np.allclose(st, np.round(st)):
            raise NotMassAction(f"the rate term {k}*{_mono(exps, species)} changes species by "
                                f"non-integer amounts {st.tolist()}")
        st = np.round(st).astype(int)
        prod = np.array(exps) + st
        if np.any(prod < 0):
            bad = [str(species[j]) for j in np.where(prod < 0)[0]]
            raise NotMassAction(f"the rate term {k}*{_mono(exps, species)} consumes {bad}, but "
                                "its rate does not depend on them (no mass-action reaction "
                                "can do that)")
        kval = float(k.xreplace(pv)) if k.free_symbols else float(k)
        if kval < 0:
            raise NotMassAction(f"the rate constant {k} is negative ({kval})")
        reactions.append({
            "id": f"re{n}",
            "reactants": {str(species[j]): int(c) for j, c in enumerate(exps) if c},
            "products": {str(species[j]): int(c) for j, c in enumerate(prod) if c},
            "rate_constant": str(k), "k_value": kval,
            "model_parameters": sorted(str(s) for s in k.free_symbols),
        })
    if not reactions:
        raise NotMassAction("no reactions could be formed")

    y = {sp.Symbol(s): v for s, v in y0.items()}
    initial = {s: float(y0[s]) for s in state_names}
    for h in hidden_syms:
        initial[str(h)] = float(hidden_expr[h].xreplace(pv).xreplace(y))
    return {"species": [str(s) for s in species], "initial": initial, "reactions": reactions,
            "hidden": {str(h): sp.sstr(hidden_expr[h]) for h in hidden_syms}}


def _unused_id(base, taken):
    name, k = base, 1
    while name in taken:
        name, k = f"{base}_{k}", k + 1
    return name


def _mono(exps, species):
    return "*".join(f"{s}^{e}" if e > 1 else str(s) for s, e in zip(species, exps) if e) or "1"


def write_crnt_sbml(net: dict, path, rate_constants: dict | None = None,
                    initial: dict | None = None) -> Path:
    """Mass-action SBML of the network (one irreversible reaction per rate
    term, rate constant k_<id>) -- the form crnt4sbml reads, and a normal
    model every other tool can analyse."""
    rate_constants = rate_constants or {}
    initial = {**net["initial"], **(initial or {})}
    doc = libsbml.SBMLDocument(2, 4)
    m = doc.createModel()
    m.setId(Path(path).stem.replace("-", "_"))
    c = m.createCompartment()
    c.setId("cell"); c.setSize(1.0); c.setConstant(True)
    for s in net["species"]:
        x = m.createSpecies()
        x.setId(s); x.setCompartment("cell"); x.setInitialConcentration(float(initial.get(s, 0.0)))
        x.setBoundaryCondition(False); x.setHasOnlySubstanceUnits(False)
        if s in net.get("hidden", {}):   # a species restored from a conservation law
            x.setName(f"{s} = {net['hidden'][s]} (restored from a conservation law)")
    # Inflow (nothing -> X) and outflow (X -> nothing): crnt4sbml expects the "nothing"
    # side as a BOUNDARY species, which it leaves out of the species, so the complex is
    # still the zero complex (deficiency and conservation laws unchanged) -- and its
    # semi-diffusive approach, written for such open networks, refuses to run without one.
    boundary = None
    if any(not r["reactants"] or not r["products"] for r in net["reactions"]):
        boundary = _unused_id("boundary", set(net["species"]))
        x = m.createSpecies()
        x.setId(boundary); x.setCompartment("cell"); x.setInitialConcentration(1.0)
        x.setBoundaryCondition(True); x.setConstant(True); x.setHasOnlySubstanceUnits(False)
        x.setName("boundary: the environment (inflow / outflow), held at 1 so rates are unchanged")
    for r in net["reactions"]:
        kid = f"k_{r['id']}"
        p = m.createParameter()
        p.setId(kid); p.setValue(float(rate_constants.get(r["id"], r["k_value"]))); p.setConstant(True)
        # what the rate constant is in the original model, so the file can be read on its own
        p.setName(f"{r['rate_constant']} in the original model")
        rx = m.createReaction()
        rx.setId(r["id"]); rx.setReversible(False)
        rx.setName(f"{r['rate_constant']}")
        reactants = r["reactants"] or ({boundary: 1} if boundary else {})
        products = r["products"] or ({boundary: 1} if boundary else {})
        for s, n in reactants.items():
            sr = rx.createReactant(); sr.setSpecies(s); sr.setStoichiometry(n)
        for s, n in products.items():
            sr = rx.createProduct(); sr.setSpecies(s); sr.setStoichiometry(n)
        # mass action over the real reactants only (the boundary is held at 1 anyway)
        law = "*".join([kid] + [f"{s}^{n}" if n > 1 else s for s, n in r["reactants"].items()])
        rx.createKineticLaw().setMath(libsbml.parseL3Formula(law))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    libsbml.writeSBMLToFile(doc, str(path))
    return path


def check_conversion(net: dict, state_names, rhs, param_names, n_points=6, seed=0) -> dict:
    """Does the mass-action network reproduce the model's own ODEs? Each rate
    constant is put back as the original expression it stands for and each
    restored species as its conservation-law expression; the network's
    right-hand sides must then equal the original ones. Compared numerically
    at random positive points (states and parameters), so no algebra can hide
    a difference."""
    # every name as a plain symbol: model names like Ge, S, E or N would otherwise
    # be read as sympy objects (GreaterThan, S, E, ...)
    names = {n: sp.Symbol(n) for n in [*net["species"], *state_names, *param_names]}

    def parse(text):
        return sp.parse_expr(str(text), local_dict=names)

    species = [sp.Symbol(s) for s in net["species"]]
    hidden = {sp.Symbol(h): parse(e) for h, e in net.get("hidden", {}).items()}
    conv = {s: sp.Integer(0) for s in species}
    for r in net["reactions"]:
        flux = parse(r["rate_constant"])
        for s_, n in r["reactants"].items():
            flux *= sp.Symbol(s_) ** n
        for s_ in set(r["reactants"]) | set(r["products"]):
            change = r["products"].get(s_, 0) - r["reactants"].get(s_, 0)
            if change:
                conv[sp.Symbol(s_)] += change * flux
    states = [sp.Symbol(s) for s in state_names]
    params = [sp.Symbol(p) for p in param_names]
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_points):
        point = {x: float(v) for x, v in zip(states + params, rng.uniform(0.5, 2.0, len(states) + len(params)))}
        for s, orig in zip(states, rhs):
            a = float(sp.sympify(orig).xreplace(point))
            b = float(conv[s].xreplace(hidden).xreplace(point))
            worst = max(worst, abs(a - b) / max(abs(a), abs(b), 1e-12))
    return {"reproduces_original_odes": bool(worst < 1e-8),
            "max_relative_difference": worst, "random_points_checked": n_points,
            "how": "each k_reN replaced by the original expression it stands for and each restored "
                   "species by its conservation-law expression; the right-hand sides then compared "
                   "with the original ODEs at random positive states and parameters"}


def conservation_laws(net: dict) -> list:
    """Exact conservation laws of the network: [{species: coefficient}]."""
    names = net["species"]
    N = sp.zeros(len(names), len(net["reactions"]))
    for j, r in enumerate(net["reactions"]):
        for s_, c in r["reactants"].items():
            N[names.index(s_), j] -= c
        for s_, c in r["products"].items():
            N[names.index(s_), j] += c
    basis = N.T.nullspace()
    physical = _non_negative_laws(N, len(basis))
    if len(physical) == len(basis):
        return [{names[i]: float(x) for i, x in enumerate(v) if x != 0} for v in physical]
    laws = []
    for v in basis:
        v = v / max(abs(x) for x in v if x != 0)
        laws.append({names[i]: float(x) for i, x in enumerate(v) if x != 0})
    return laws


def _non_negative_laws(N, k):
    """Up to k independent conservation laws with non-negative integer coefficients (total
    amounts, e.g. an enzyme free plus bound); fewer when the network has no full such set.
    For each species the smallest non-negative law containing it is found by linear
    programming, checked exactly (c^T N = 0), and laws are taken, smallest first, while
    they stay independent."""
    if k == 0:
        return []
    from scipy.optimize import linprog
    n = N.rows
    Nf = np.array(N.tolist(), dtype=float)
    found = []
    for i in range(n):
        bounds = [(0, None)] * n
        bounds[i] = (1, None)
        res = linprog(np.ones(n), A_eq=Nf.T, b_eq=np.zeros(Nf.shape[1]), bounds=bounds,
                      method="highs")
        if not res.success:
            continue
        v = [sp.nsimplify(float(x), rational=True, tolerance=1e-9) for x in res.x]
        if any(x < 0 for x in v) or any(e != 0 for e in (sp.Matrix([v]) * N)):
            continue
        den = sp.ilcm(*[sp.Rational(x).q for x in v]) if len(v) > 1 else 1
        v = [int(x * den) for x in v]
        g = sp.igcd(*[x for x in v if x]) if sum(1 for x in v if x) > 1 else max(v)
        found.append([x // g for x in v])
    chosen = []
    for v in sorted({tuple(v) for v in found}, key=lambda v: (sum(1 for x in v if x), sum(v))):
        if sp.Matrix(chosen + [list(v)]).rank() > len(chosen):
            chosen.append(list(v))
        if len(chosen) == k:
            break
    return chosen


def named_laws(net: dict) -> list:
    """The conservation laws with the species each one eliminates and the name of its
    total: [(total, eliminated species, {species: coefficient})]."""
    laws, out, eliminated = conservation_laws(net), [], set()
    for k, law in enumerate(laws, start=1):
        cands = [s_ for s_ in law if s_ not in eliminated and all(
            s_ not in l2 or l2 is law for l2 in laws[:k - 1])]
        cands = cands or [s_ for s_ in law if s_ not in eliminated]
        pick = sorted(cands, key=lambda s_: (not s_.endswith("_free"), -abs(law[s_])))[0]
        eliminated.add(pick)
        out.append((pick[:-5] if pick.endswith("_free") else f"C{k}", pick, law))
    return out


def law_totals(net: dict, point: dict) -> dict:
    """The total of each conservation law at a point, by the names named_laws gives."""
    return {total: sum(c * float(point.get(s_, 0.0)) for s_, c in law.items())
            for total, _, law in named_laws(net)}


def write_reduced_sbml(net: dict, path, rate_constants: dict, point: dict) -> tuple[Path, dict]:
    """The network with one species per conservation law written as an
    assignment rule ``X = TOTAL - (...)`` and each TOTAL a parameter, so the
    totals can be continued (AUTO) like any other parameter. A restored
    species ``<T>_free`` is eliminated first and its total is named ``T``
    (the model's own name); otherwise ``C<k>``. Returns (path, {total: value})."""
    eliminated, totals = {}, {}
    for total, pick, law in named_laws(net):
        totals[total] = sum(c * point[s_] for s_, c in law.items())
        rest = " + ".join(f"{c!r} * {s_}" for s_, c in law.items() if s_ != pick)
        eliminated[pick] = f"({total} - ({rest or 0})) / {law[pick]!r}"
    path = write_crnt_sbml(net, path, rate_constants=rate_constants, initial=point)
    doc = libsbml.readSBMLFromFile(str(path))
    m = doc.getModel()
    for t, v in totals.items():
        p = m.createParameter(); p.setId(t); p.setValue(float(v)); p.setConstant(True)
    for s_, formula in eliminated.items():
        m.getSpecies(s_).setBoundaryCondition(True)
        rule = m.createAssignmentRule()
        rule.setVariable(s_); rule.setMath(libsbml.parseL3Formula(formula))
    libsbml.writeSBMLToFile(doc, str(path))
    return path, totals


def crnt_dir() -> Path:
    return config.OUTPUT_DIR / "crnt"


def import_crnt4sbml():
    """crnt4sbml 0.0.15 pins numpy 1.16 / matplotlib 3.1; it runs on current
    versions once the names they removed are aliased back."""
    import matplotlib
    import matplotlib.cbook
    import numpy
    if not hasattr(matplotlib.cbook, "mplDeprecation"):
        matplotlib.cbook.mplDeprecation = matplotlib.MatplotlibDeprecationWarning
    for name, value in (("ComplexWarning", getattr(numpy, "exceptions", numpy).ComplexWarning),
                        ("Inf", numpy.inf), ("PINF", numpy.inf), ("float", float)):
        if name not in numpy.__dict__:
            setattr(numpy, name, value)
    import crnt4sbml
    _alias_dual_annealing_options()
    _faster_equilibrium_manifold()
    _faster_conservation_matrix()
    return crnt4sbml


def _faster_conservation_matrix():
    """crnt4sbml builds the conservation-law matrix B^T from the nullspace of (Y A)^T with
    every rate constant a symbol; on networks with many elementary reactions that symbolic
    row reduction runs for many minutes. The rate constants enter only as generic positive
    values, so the same nullspace is computed with each one replaced by a fixed random
    positive fraction, in exact arithmetic (the symbolic result is the generic one, which a
    random point reproduces except on a set of measure zero). The rest is crnt4sbml's own."""
    import random
    import sys as _sys
    import numpy
    import sympy
    from crnt4sbml.c_graph import Cgraph
    if getattr(Cgraph, "_faster_bt", False):
        return
    P = "_Cgraph__"

    def create_bt(self):
        M = (getattr(self, P + "Y") * getattr(self, P + "A")).T
        rng = random.Random(20240607)
        point = {s: sympy.Rational(rng.randint(101, 997), rng.randint(101, 991))
                 for s in sorted(M.free_symbols, key=str)}
        the_null_space = M.xreplace(point).nullspace()
        sizes = len(the_null_space)
        bt_temp = getattr(self, P + "create_non_negative_b_matrix")(the_null_space, sizes)
        bt = sympy.Matrix(bt_temp[~numpy.all(bt_temp == 0, axis=1)]).evalf(10)
        bt = bt[0:self.get_dim_equilibrium_manifold(), :]
        _, vals = bt.T.rref()
        if len(vals) != self.get_dim_equilibrium_manifold():
            print("\nError: Not all rows of B^T are linearly independent!\n")
            _sys.exit()
        setattr(self, P + "B", bt)

    setattr(Cgraph, P + "create_BT_matrix", create_bt)
    Cgraph._faster_bt = True


def _faster_equilibrium_manifold():
    """crnt4sbml's mass-conservation approach picks the variables to solve for by trying
    every combination of species and testing each equation for joint linearity in it. Two
    parts of that are slow without need, and are replaced by exact equivalents (same
    answers, same order, same choice):
    - the linearity test computed every second derivative of the equation, for every
      combination; here each equation is split into its terms once, and "jointly linear"
      is read from the exponents (no term of degree two or more in the variables);
    - 'combination already tried' searched a list growing to hundreds of thousands of
      entries; here a set."""
    import sympy
    from crnt4sbml.mass_conservation_approach import MassConservationApproach as M
    if getattr(M, "_faster_manifold", False):
        return
    P = "_MassConservationApproach__"
    original_is_linear = getattr(M, P + "is_linear")

    def is_linear(self, expr, variables):
        cache = self.__dict__.setdefault("_term_cache", {})
        entry = cache.get(id(expr))
        if entry is None or entry[0] is not expr:
            try:
                terms = [t.as_powers_dict() for t in sympy.Add.make_args(sympy.expand(expr))]
            except Exception:
                return original_is_linear(self, expr, variables)
            entry = cache[id(expr)] = (expr, terms)
        wanted = set(variables)
        for powers in entry[1]:
            degree = 0
            for base, exp in powers.items():
                if base in wanted:
                    if not getattr(exp, "is_Integer", False):
                        return original_is_linear(self, expr, variables)   # not a monomial
                    if exp < 0:
                        return False          # in a denominator: second derivative not zero
                    degree += int(exp)
                elif getattr(base, "free_symbols", set()) & wanted:
                    return original_is_linear(self, expr, variables)       # inside a function
            if degree > 1:
                return False
        return True

    def choose(self, temp_vec, indicies_to_skip, indp_vars):
        get = lambda n: getattr(self, P + n)
        leng = get("M") - get("ell") - get("delta")
        explored, counts = get("indices_explored"), get("counts_n_indices")
        seen = self.__dict__.setdefault("_explored_set", set(explored))
        skip = set(indicies_to_skip)
        for jj in range(len(indp_vars)):
            if jj not in skip and jj not in seen:
                n_linear = [is_linear(self, temp_vec[j], indp_vars[jj][0:leng])
                            for j in range(temp_vec.shape[0])].count(True)
                counts.append([n_linear, jj])
                explored.append(jj)
                seen.add(jj)
                if n_linear == get("M"):
                    break
        chosen = max((c for c in counts if c[1] not in skip), key=lambda item: item[0])[1]
        setattr(self, P + "fixed_pars", indp_vars[chosen])
        fixed = get("fixed_pars")
        setattr(self, P + "free_pars", [i for i in get("deficiency_pars") + get("concentration_pars")
                                        if i not in fixed])
        out = [is_linear(self, temp_vec[j], indp_vars[chosen]) for j in range(temp_vec.shape[0])]
        return ([i for i, x in enumerate(out) if x] + [i for i, x in enumerate(out) if not x],
                chosen)

    setattr(M, P + "is_linear", is_linear)
    setattr(M, P + "create_fixed_free_pars_and_reordered_ind", choose)
    M._faster_manifold = True


def _alias_dual_annealing_options():
    """crnt4sbml 0.0.15 calls two scipy.optimize functions the way scipy 1.4 had them:
    dual_annealing(local_search_options=...), now minimizer_kwargs, and
    linprog(method='simplex'), deprecated and later removed (the HiGHS solver solves the
    same linear program). Its modules (only they) get a scipy that takes both and passes
    everything else to scipy unchanged."""
    import importlib
    import inspect
    import types
    import scipy
    import scipy.optimize

    def dual_annealing(*args, local_search_options=None, **kwargs):
        if (local_search_options is not None and "local_search_options"
                not in inspect.signature(scipy.optimize.dual_annealing).parameters):
            kwargs.setdefault("minimizer_kwargs", local_search_options)
        elif local_search_options is not None:
            kwargs["local_search_options"] = local_search_options
        return scipy.optimize.dual_annealing(*args, **kwargs)

    def linprog(*args, method="highs", **kwargs):
        return scipy.optimize.linprog(*args, method="highs" if method == "simplex" else method,
                                      **kwargs)

    class _Proxy(types.ModuleType):       # everything else is scipy's own
        def __init__(self, real, **own):
            super().__init__(real.__name__)
            self._real = real
            self.__dict__.update(own)

        def __getattr__(self, name):
            return getattr(self._real, name)

    optimize = _Proxy(scipy.optimize, dual_annealing=dual_annealing, linprog=linprog)
    proxy = _Proxy(scipy, optimize=optimize, _crnt_alias=True)
    for name in ("c_graph", "general_approach", "mass_conservation_approach",
                 "semi_diffusive_approach", "bistability_finder", "bistability_analysis"):
        try:
            mod = importlib.import_module(f"crnt4sbml.{name}")
        except Exception:
            continue
        if getattr(getattr(mod, "scipy", None), "_crnt_alias", False):
            continue
        if hasattr(mod, "scipy"):
            mod.scipy = proxy
