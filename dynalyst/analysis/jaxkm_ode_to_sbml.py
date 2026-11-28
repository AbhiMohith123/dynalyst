"""
ODE -> SBML conversion via the ACTUAL jaxkineticmodel package.

Replaces the project's former hand-rolled libsbml rateRule writer
(ode_sbml_convert.py). Instead of writing raw <rateRule> SBML directly, this
builds a real jaxkineticmodel.building_models.JaxKineticModelBuild.NeuralODEBuild
object -- one synthetic Reaction per species, each carrying its full
right-hand side as a single custom Mechanism -- and exports it through
jaxkineticmodel.load_sbml.export_sbml.SBMLExporter, the package's own
reaction/kineticLaw-based SBML writer. This is the same exporter
jaxkineticmodel itself uses for models built with JaxKineticModelBuild, so
the emitted file reflects the real package's own semantics (and could be
loaded straight back into jaxkineticmodel itself later) rather than a
side-channel format only this project understood. It's still read back here
via this project's generic_sbml.py, which already handles
reaction+kineticLaw-based SBML generically (alongside rateRules) -- exactly
the shape this exporter writes -- so every downstream tool (get_odes,
find_steady_states, run_bifurcation_analysis) keeps working
unchanged.

Design notes
------------
- A user-specified ODE system (species, {species: rhs_string}) is not
  itself a reaction network -- there's no natural split into individual
  mass-action steps. So each species' full RHS is wrapped as ONE synthetic
  reaction "d<species>_dt" that produces/consumes that species at
  stoichiometry 1 (rate = the whole RHS expression).
- Any OTHER species referenced in that RHS (coupled systems -- e.g. the
  Brusselator's dx/dt depending on y) is included in the same reaction at
  stoichiometry 0, which jaxkineticmodel's own Reaction/exporter machinery
  represents as a modifier: present in the rate law, not stoichiometrically
  changed by it. This is exactly the right semantics, and (unlike leaving
  it out of the reaction's species list entirely) it does NOT get
  misclassified as a fixed parameter by Reaction's own
  "free symbols minus species = parameters" logic.
- RawExpressionMechanism (a real jaxkineticmodel Mechanism subclass, not a
  bypass of it) wraps one arbitrary sympy expression as a mechanism
  `compute()`: called with sympy Symbols (from Mechanism.symbolic(), used
  by the exporter to build the SBML kinetic law) it substitutes and returns
  the expression symbolically; called with numbers/JAX arrays (from
  Mechanism.__call__, used if the model is actually simulated/trained
  through jaxkineticmodel directly) it evaluates via a JAX-lambdified
  version of the same expression, built once at construction time.
"""
import json
import math

import jax.numpy as jnp
import re

import sympy as sp

from dynalyst import config
from dynalyst.analysis import ode_model
from dynalyst.analysis.ode_model import model_from_odes, normalise_odes

from jaxkineticmodel.building_models.JaxKineticModelBuild import (
    JaxKineticModelBuild, NeuralODEBuild, Reaction,
)
from jaxkineticmodel.kinetic_mechanisms.JaxKineticMechanisms import Mechanism
from jaxkineticmodel.load_sbml.export_sbml import SBMLExporter

_COMPARTMENT_NAME = "compartment"


class RawExpressionMechanism(Mechanism):
    """Wraps one arbitrary sympy expression as a jaxkineticmodel Mechanism,
    so a directly-specified ODE's full right-hand side can be handed to
    jaxkineticmodel's own Reaction/export machinery as a single kinetic
    law, rather than trying to decompose a user-typed equation into the
    package's built-in mechanism library (Michaelis-Menten, mass-action,
    etc.), which it generally isn't an instance of.
    """

    def __init__(self, expr: sp.Expr):
        self._expr = expr
        # deterministic order so every call (symbolic or numeric) agrees
        names = sorted(str(s) for s in expr.free_symbols)
        self._names = names
        symbols = [sp.Symbol(n) for n in names]
        # built once here, not per call: numeric/JAX evaluation path, used
        # by Mechanism.__call__ if the model is ever simulated directly
        self._numeric_fn = (
            sp.lambdify(symbols, expr, modules="jax") if names else (lambda: float(expr))
        )
        # Mechanism.__init__ wants {local_name: actual_name}; every name
        # here is already a real species/parameter name (no renaming,
        # unlike jaxkineticmodel's built-in mechanisms which map e.g. a
        # local 'substrate' to a model-specific species name).
        super().__init__(**{n: n for n in names})

    def compute(self, **kwargs):
        if not self._names:
            return self._numeric_fn()
        values = [kwargs[n] for n in self._names]
        if any(isinstance(v, sp.Basic) for v in values):
            # symbolic call, from Mechanism.symbolic() at export time
            subs = dict(zip((sp.Symbol(n) for n in self._names), values))
            return self._expr.subs(subs)
        # numeric/JAX call, from Mechanism.__call__ at simulation time
        return self._numeric_fn(*values)


def _build_jkm_model(species: list[str], odes: dict, compartment_size: float) -> NeuralODEBuild:
    """Build a real jaxkineticmodel NeuralODEBuild from a directly-specified
    ODE system: one synthetic Reaction per species wrapping its full RHS."""
    # every name not called as a function is a symbol of the model: never one of sympy's
    # built-in constants (I is the imaginary unit, E Euler's number, N / S / O / Q functions)
    local_dict = {n: sp.Symbol(n) for text in odes.values()
                  for n in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_]*\s*\()", str(text))}
    local_dict.update({s: sp.Symbol(s) for s in species})
    reactions = []
    if not math.isfinite(compartment_size) or compartment_size <= 0:
        raise ValueError("compartment_size must be finite and positive")
    for target in species:
        expr = sp.sympify(odes[target], locals=local_dict)
        # ODE states are concentrations; SBML kinetic laws are amount/time.
        expr *= sp.Rational(str(compartment_size))
        free = {str(s) for s in expr.free_symbols}
        # owner species first (stoichiometry +1); any OTHER species this
        # RHS references come next, at stoichiometry 0 (modifiers) -- free
        # symbols that are not in `species` at all are simply never listed
        # here, so Reaction's own logic classifies them as parameters.
        rxn_species = [target] + [s for s in species if s in free and s != target]
        stoichiometry = [1] + [0] * (len(rxn_species) - 1)
        compartments = [_COMPARTMENT_NAME] * len(rxn_species)
        mechanism = RawExpressionMechanism(expr)
        reactions.append(Reaction(
            name=f"d{target}_dt", species=rxn_species, stoichiometry=stoichiometry,
            compartments=compartments, mechanism=mechanism,
        ))

    func = JaxKineticModelBuild(reactions, compartments={_COMPARTMENT_NAME: compartment_size})
    return NeuralODEBuild(func)


def odes_to_sbml(species: list[str] | None, odes: dict, parameters: dict | None = None,
                  initial_conditions: dict | None = None, model_name: str | None = None,
                  compartment_size: float = 1.0) -> str:
    """
    Tool: convert a directly-specified ODE system into a real SBML file,
    written via jaxkineticmodel's OWN model-building and export machinery
    (JaxKineticModelBuild / NeuralODEBuild / SBMLExporter) instead of a
    hand-rolled libsbml writer -- so the file reflects exactly how
    jaxkineticmodel itself represents (and would re-simulate) the system.
    Saved under this project's local SBML models directory so it's
    immediately usable by name in every other tool (get_odes,
    find_steady_states, run_bifurcation_analysis).
    """
    # Validate against the SAME semantics every other tool relies on --
    # ode_model.model_from_odes is the canonical parser (species membership,
    # parameter defaulting to 1.0, etc.) -- before touching jaxkineticmodel.
    try:
        species, odes = normalise_odes(species, odes)
        if not model_name:     # named by content: never overwrites a different system's file
            from dynalyst.analysis.typed_equations import typed_model_name
            model_name = typed_model_name({"odes": odes, "parameters": parameters or {},
                                           "initial_conditions": initial_conditions or {}})
        parsed = model_from_odes(species, odes, parameters, initial_conditions, label=model_name)
    except Exception as e:
        return json.dumps({"error": f"Could not parse this ODE system: {e}"})

    try:
        jkm_model = _build_jkm_model(species, odes, compartment_size)
    except Exception as e:
        return json.dumps({"error": f"jaxkineticmodel could not build this system: {e}"})

    # export() zips initial_conditions against jkm_model.species_names,
    # which comes from the stoichiometric matrix's row order -- not
    # necessarily the caller's `species` order -- so build the vector in
    # THAT order, not the input order.
    y0_map = parsed["y0"]
    # SBMLExporter writes initialAmount while the declared species symbols and
    # the typed ODE use concentrations.
    y0_vec = jnp.array([float(y0_map.get(s, 0.0)) * compartment_size
                       for s in jkm_model.species_names])

    # Every free parameter jaxkineticmodel actually found across all
    # reactions must get a value or the exported SBML would reference an
    # undefined parameter -- default to 1.0 for anything not given,
    # matching model_from_odes's own default so both paths agree.
    param_defaults = parsed["param_defaults"]
    export_params = {p: float(param_defaults.get(p, 1.0)) for p in jkm_model.parameter_names}

    config.LOCAL_SBML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = re.sub(r"\.(xml|sbml)$", "", str(model_name).strip(), flags=re.I)  # never .xml.xml
    out_path = config.LOCAL_SBML_MODELS_DIR / f"{model_name}.xml"

    try:
        exporter = SBMLExporter(jkm_model)
        exporter.export(initial_conditions=y0_vec, parameters=export_params, output_file=str(out_path))
    except Exception as e:
        return json.dumps({"error": f"jaxkineticmodel SBML export failed: {e}"})

    # Round-trip sanity check: confirm this project's OWN SBML reader (used
    # by every downstream tool) can actually load the file back into the
    # same ODE system before reporting success.
    try:
        reloaded = ode_model.model_from_sbml(str(out_path))
        if set(reloaded["state_names"]) != set(parsed["state_names"]):
            raise ValueError("Export changed the state variables")
        defaults = {sp.Symbol(k): sp.Rational(str(v)) for k, v in reloaded["param_defaults"].items()}
        expected_defaults = {sp.Symbol(k): sp.Rational(str(v)) for k, v in parsed["param_defaults"].items()}
        def exact_decimals(expr):
            # SBML serializes integer powers as floats (x^3.0). Compare their
            # values before parameter arithmetic; allow serialization roundoff
            # in fractions, with relative tolerance so tiny rates stay nonzero.
            replacements = {}
            for value in expr.atoms(sp.Float):
                tolerance = min(abs(float(value)) * 1e-12, 1e-12)
                replacements[value] = (sp.nsimplify(value, rational=True, tolerance=tolerance)
                                       if tolerance else sp.Rational(str(value)))
            return expr.xreplace(replacements)
        for name, expected in zip(parsed["state_names"], parsed["rhs"]):
            actual = reloaded["rhs"][reloaded["state_names"].index(name)]
            a = exact_decimals(actual).subs(defaults)
            b = exact_decimals(expected).subs(expected_defaults)
            if sp.simplify(a-b) != 0:
                # SBML decimal serialization can move a coefficient by one ULP.
                # Compare every symbolic term's coefficient relatively: a missing
                # tiny rate still fails (there is deliberately no absolute floor).
                ac = sp.expand(a).as_coefficients_dict()
                bc = sp.expand(b).as_coefficients_dict()
                if not all(math.isclose(float(ac.get(term,0)), float(bc.get(term,0)),
                                        rel_tol=1e-12, abs_tol=0)
                           for term in set(ac) | set(bc)):
                    raise ValueError(f"Export changed the derivative of {name}")
            if not math.isclose(reloaded["y0"][name], parsed["y0"][name], rel_tol=1e-12, abs_tol=0):
                raise ValueError(f"Export changed the initial value of {name}")
    except Exception as e:
        return json.dumps({
            "error": f"Exported {out_path.name}, but it failed to load back "
                     f"through this project's SBML reader: {e}",
            "path": str(out_path),
        })

    return json.dumps({
        "model_name": model_name,
        "path": str(out_path),
        "species": species,
        "parameters": export_params,
        "initial_conditions": {s: float(y0_map.get(s, 0.0)) for s in species},
        "reloaded_odes": ode_model.ode_strings(reloaded),
        "note": (
            f"Written to {out_path} using jaxkineticmodel's own "
            f"NeuralODEBuild/SBMLExporter (not a hand-rolled writer), and "
            f"verified to round-trip back through get_odes. It can now be "
            f"used by name ('{model_name}') in get_odes, find_steady_states, "
            f"and run_bifurcation_analysis, exactly like "
            f"any other SBML model."
        ),
    }, indent=2, default=float)


TOOL_DEFINITIONS = [
    {
        "name": "odes_to_sbml",
        "description": (
            "Convert a directly-specified system of ODEs into a real, valid SBML file, "
            "built and written via jaxkineticmodel's OWN model machinery "
            "(JaxKineticModelBuild/NeuralODEBuild + its SBMLExporter) rather than a "
            "hand-rolled writer -- saved so it can immediately be used by name in every "
            "other tool (get_odes, find_steady_states, run_bifurcation_analysis). "
            "Use this when the user gives you equations directly and "
            "wants an SBML model out of them, or wants to save/share/reuse a "
            "directly-specified system as a standard SBML file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "species": {"type": "array", "items": {"type": "string"}, "description": "Ordered list of state-variable/species names. Optional: defaults to the keys of odes."},
                "odes": {"type": "object", "description": "{species_name: rhs_expression_string}, e.g. {'x': 'a - x**2'}. Keys may also be written 'dx/dt' or \"x'\"."},
                "parameters": {"type": "object", "description": "{parameter_name: default_value}."},
                "initial_conditions": {"type": "object", "description": "{species_name: value}. Defaults to 0.0 for any species not listed."},
                "model_name": {"type": "string", "description": "Name for the model / output filename. Default: a name made from the equations and values (typed_<hash>), so different systems never overwrite each other."},
                "compartment_size": {"type": "number", "description": "Single default compartment's size. Default 1.0."},
            },
            "required": ["odes"],
        },
    },
]

TOOL_FUNCTIONS = {
    "odes_to_sbml": lambda **kwargs: odes_to_sbml(**kwargs),
}
