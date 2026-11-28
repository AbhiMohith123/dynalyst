"""
Registry of generic bifurcation models, covering two things by design:

1. "generic" variant: minimal normal-form SBML models built directly from
   the standard textbook table (dx/dt = mu + x^2 for saddle-node, etc.) --
   built on demand via libsbml, one species (two for the Hopf models),
   parameter 'mu'.
2. "pdp" variant: the three phosphorylation-dephosphorylation reference
   models (Sriram 2026 examples 2-4) -- saddle_node.xml, transcritical.xml,
   pitchfork.xml -- which already exhibit the same bifurcation types via a
   biologically-motivated Hill/MM autoregulation mechanism instead of the
   bare normal form. Parameter 'alpha'.

Both variants run through run_bifurcation_analysis, like any other model, so a Hopf model
also gets its cycles. Generalized Hopf (Bautin) and limit-point cycle have no reference model
here: resolving either raises a ValueError saying so rather than producing a misleading result.
"""
import os
import re
import libsbml

from dynalyst import config

# The PDP reference SBML files (saddle_node.xml, transcritical.xml,
# pitchfork.xml) live in the project's models/ folder.
MODELS_DIR = str(config.MODELS_DIR)


# ---------------------------------------------------------------------------
# Generic normal-form SBML builders
# ---------------------------------------------------------------------------

def _new_doc(model_id):
    document = libsbml.SBMLDocument(2, 4)
    model = document.createModel(model_id)
    comp = model.createCompartment()
    comp.setId("c")
    comp.setConstant(True)
    comp.setSize(1)
    return document, model


def _add_species(model, sid, ic):
    s = model.createSpecies()
    s.setId(sid)
    s.setCompartment("c")
    s.setInitialConcentration(ic)
    s.setHasOnlySubstanceUnits(False)
    s.setBoundaryCondition(False)
    s.setConstant(False)


def _add_param(model, pid, value):
    p = model.createParameter()
    p.setId(pid)
    p.setValue(value)
    p.setConstant(True)


def _add_rate_rule(model, var, formula):
    rule = model.createRateRule()
    rule.setVariable(var)
    rule.setMath(libsbml.parseL3Formula(formula))


def _finalize_and_write(document, out_path):
    document.checkConsistency()
    severe = [document.getError(i).getMessage() for i in range(document.getNumErrors())
              if document.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR]
    if severe:
        raise ValueError(f"Generated SBML failed validation: {severe}")
    # normal forms: x is no concentration, so checks for negative values do not apply
    from dynalyst.analysis.typed_equations import TYPED_MARK
    document.getModel().setNotes(f'<body xmlns="http://www.w3.org/1999/xhtml"><p>{TYPED_MARK}</p></body>')
    libsbml.writeSBMLToFile(document, out_path)
    return out_path


def build_saddle_node(out_path, mu0=-4.0, x0=-2.0):
    """dx/dt = mu + x^2. Two equilibria x=+-sqrt(-mu) exist for mu<0 and
    collide/annihilate at mu=0. x0 starts on the stable branch (x<0)."""
    document, model = _new_doc('generic_saddle_node')
    _add_species(model, 'x', x0)
    _add_param(model, 'mu', mu0)
    _add_rate_rule(model, 'x', 'mu + x^2')
    return _finalize_and_write(document, out_path)


def build_transcritical(out_path, mu0=-2.0, x0=0.05):
    """dx/dt = mu*x - x^2. Branches x=0 and x=mu cross and exchange
    stability at mu=0. x0 starts near the (stable, for mu<0) x=0 branch."""
    document, model = _new_doc('generic_transcritical')
    _add_species(model, 'x', x0)
    _add_param(model, 'mu', mu0)
    _add_rate_rule(model, 'x', 'mu*x - x^2')
    return _finalize_and_write(document, out_path)


def build_pitchfork(out_path, mu0=-2.0, x0=0.05):
    """dx/dt = mu*x - x^3. Supercritical pitchfork: x=0 stable for mu<0,
    unstable for mu>0, with new stable branches x=+-sqrt(mu) for mu>0."""
    document, model = _new_doc('generic_pitchfork')
    _add_species(model, 'x', x0)
    _add_param(model, 'mu', mu0)
    _add_rate_rule(model, 'x', 'mu*x - x^3')
    return _finalize_and_write(document, out_path)


def build_hopf(out_path, subcritical=False, mu0=-2.0, x0=0.05, y0=0.05):
    """2D Cartesian Hopf normal form (polar r'=mu*r -+ r^3 converted to
    x,y): dx/dt = mu*x - y -+ x*(x^2+y^2); dy/dt = x + mu*y -+ y*(x^2+y^2).
    '-' (supercritical) or '+' (subcritical) sign on the cubic term. The
    origin (0,0) is an equilibrium for every mu; its eigenvalues cross the
    imaginary axis at mu=0 (the HB point), which AUTO's continuation on
    this trivial branch will detect and report -- but see this module's
    docstring: the emerging/vanishing limit-cycle amplitude branch itself
    is NOT traced, since periodic-orbit continuation isn't implemented."""
    kind = 'generic_hopf_subcritical' if subcritical else 'generic_hopf_supercritical'
    document, model = _new_doc(kind)
    _add_species(model, 'x', x0)
    _add_species(model, 'y', y0)
    _add_param(model, 'mu', mu0)
    sign = '+' if subcritical else '-'
    _add_rate_rule(model, 'x', f'mu*x - y {sign} x*(x^2 + y^2)')
    _add_rate_rule(model, 'y', f'x + mu*y {sign} y*(x^2 + y^2)')
    return _finalize_and_write(document, out_path)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY = {
    'saddle_node': {
        'aliases': ['saddle-node', 'saddle node', 'fold', 'lp'],
        'description': ("dx/dt = mu + x^2 (table normal form): two equilibria "
                         "collide and annihilate at mu=0. PDP variant: Hill-"
                         "autoregulation saddle-node at alpha=2 (Sriram 2026 Ex.3)."),
        'variants': {
            'generic': {'build': build_saddle_node, 'bif_param': 'mu',
                        'y_species': 'x', 'param_range': (-4, 4)},
            'pdp': {'file': 'saddle_node.xml', 'bif_param': 'alpha',
                    'y_species': 'P', 'param_range': (0.1, 5)},
        },
    },
    'transcritical': {
        'aliases': ['tc', 'transcritical'],
        'description': ("dx/dt = mu*x - x^2: branches x=0 and x=mu cross and "
                         "exchange stability at mu=0. PDP variant: MM-"
                         "autoregulation transcritical at alpha=1 (Sriram 2026 Ex.2)."),
        'variants': {
            'generic': {'build': build_transcritical, 'bif_param': 'mu',
                        'y_species': 'x', 'param_range': (-2, 4)},
            'pdp': {'file': 'transcritical.xml', 'bif_param': 'alpha',
                    'y_species': 'P', 'param_range': (0.1, 4)},
        },
    },
    'pitchfork': {
        'aliases': ['pf', 'pitchfork'],
        'description': ("dx/dt = mu*x - x^3: supercritical pitchfork at mu=0. "
                         "PDP variant: symmetric mutual-inhibition pitchfork at "
                         "alpha~1.013 (Sriram 2026 Ex.4)."),
        'variants': {
            'generic': {'build': build_pitchfork, 'bif_param': 'mu',
                        'y_species': 'x', 'param_range': (-2, 4)},
            'pdp': {'file': 'pitchfork.xml', 'bif_param': 'alpha',
                    'y_species': 'P1', 'param_range': (0.1, 3)},
        },
    },
    'hopf_supercritical': {
        'aliases': ['hopf', 'hb', 'supercritical hopf'],
        'description': ("2D Cartesian supercritical Hopf normal form; a stable "
                         "limit cycle emerges at mu=0. No PDP variant provided."),
        'variants': {
            'generic': {'build': lambda out_path: build_hopf(out_path, subcritical=False),
                        'bif_param': 'mu', 'y_species': 'x', 'param_range': (-2, 2)},
        },
    },
    'hopf_subcritical': {
        'aliases': ['subcritical hopf'],
        'description': ("2D Cartesian subcritical Hopf normal form; an unstable "
                         "limit cycle is involved at mu=0. No PDP variant provided."),
        'variants': {
            'generic': {'build': lambda out_path: build_hopf(out_path, subcritical=True),
                        'bif_param': 'mu', 'y_species': 'x', 'param_range': (-2, 2)},
        },
    },
    'generalized_hopf': {
        'aliases': ['gh', 'bautin', 'generalized hopf'],
        'description': ("No reference model: r' = r*(mu1 + mu2*r^2 - r^4) needs a "
                         "two-parameter continuation."),
        'variants': {},
    },
    'limit_point_cycle': {
        'aliases': ['lpc', 'limit-point cycle', 'saddle-node of cycles'],
        'description': ("No reference model: run_bifurcation_analysis reports a "
                         "saddle-node of limit cycles on any model that has one."),
        'variants': {},
    },
}


def resolve_generic_category(query):
    """Loose alias matching, using \\b word-boundary matching rather than plain substring `in` --
    short aliases like 'tc', 'pf', 'hb', 'lp' would otherwise false-positive
    inside unrelated words (e.g. 'tc' matches inside "pi[tc]hfork"). Exact
    key/alias match is tried first, then word-boundary substring match,
    else a ValueError listing every available category (including the two
    flagged-unsupported ones, so the caller sees WHY they're missing rather
    than a plain 'not found')."""
    q = query.strip().lower()
    for key, entry in REGISTRY.items():
        if q == key or q in entry['aliases']:
            return key
    matches = []
    for key, entry in REGISTRY.items():
        candidates = entry['aliases'] + [key.replace('_', ' ')]
        matches.extend((len(a), key) for a in candidates
                       if re.search(r'\b' + re.escape(a) + r'\b', q))
    if matches:
        longest = max(n for n, _ in matches)
        keys = {key for n, key in matches if n == longest}
        if len(keys) == 1:
            return keys.pop()
        raise ValueError(f"Ambiguous bifurcation category: {sorted(keys)}")
    available = {k: v['description'] for k, v in REGISTRY.items()}
    raise ValueError(f"Could not match '{query}' to a known bifurcation category. "
                      f"Available: {available}")


def get_variant(category_key, variant='generic'):
    entry = REGISTRY[category_key]
    spec = entry['variants'].get(variant)
    if not spec:
        available = list(entry['variants'].keys())
        raise ValueError(
            f"No '{variant}' variant available for '{category_key}'. "
            f"{entry['description']} Available variants for this category: {available}"
        )
    return spec


def build_model_file(category, variant, workdir):
    """Resolves `category` (loose text) + `variant` ('generic' or 'pdp')
    to an actual SBML file on disk -- building it fresh into `workdir` for
    the 'generic' variant, or pointing at the existing reference file
    (expected in config.MODELS_DIR) for the 'pdp' variant. Returns
    (resolved_key, sbml_path, spec)."""
    key = resolve_generic_category(category)
    spec = get_variant(key, variant)
    if 'build' in spec:
        out_path = os.path.join(workdir, f'{key}_{variant}.xml')
        spec['build'](out_path)
    else:
        out_path = os.path.abspath(os.path.join(MODELS_DIR, spec['file']))
        if not os.path.exists(out_path):
            raise FileNotFoundError(
                f"PDP reference file not found at '{out_path}'. Place "
                f"{spec['file']} in the models folder ({MODELS_DIR})."
            )
    return key, out_path, spec
