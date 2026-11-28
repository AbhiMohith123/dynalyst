"""Explicit conserved-pool parameters for continuation, with recorded definitions."""
import sympy as sp


def expose_totals(model):
    if model.get("conserved_totals") or model.get("delays"):
        return model
    from dynalyst.analysis.steady_states import exact_conservation_laws, law_holds, law_is_exact
    names, symbols = model["state_names"], model["state_symbols"]
    definitions = {}
    occupied = set(names) | set(model["param_names"])
    for law in exact_conservation_laws(model):
        if not law_is_exact(law) or any(c < 0 for c in law):
            continue
        # A relation true only at the nominal kinetic constants is not a pool.
        if not law_holds(law, model["rhs"]):
            continue
        support = [n for n, c in zip(names, law) if c]
        if len(support) < 2:
            continue
        # the free form, inside two or more other names (K of K, AK, APK); else the first state
        inside = {s: sum(s.lower() in o.lower() for o in support if o != s) for s in support}
        free = max(support, key=inside.get)
        base = "total_" + (free if inside[free] >= 2 else support[0])
        name, suffix = base, 2
        while name in occupied:
            name, suffix = f"{base}_{suffix}", suffix + 1
        occupied.add(name)
        definitions[name] = {
            "coefficients": {n: str(c) for n, c in zip(names, law) if c},
            "expression": str(sum(c*s for c, s in zip(law, symbols))),
            "default": float(sum(c*model["y0"][n] for n, c in zip(names, law))),
            "identity_verified": True,
        }
    return {**model, "conserved_totals": definitions,
            "param_names": list(model["param_names"]) + list(definitions),
            "param_symbols": list(model["param_symbols"]) + [sp.Symbol(n) for n in definitions],
            "param_defaults": {**model["param_defaults"], **{n:d["default"] for n,d in definitions.items()}},
            "param_kinds": {**model.get("param_kinds", {}), **{n:"conserved total" for n in definitions}}}
