"""A boundary species with a rate rule is changed by the rule alone, not also by reactions.

BIOMD0000000006 (Tyson 1991) gives u and v both a rate rule and reactions that say the same;
counting both doubled du/dt, so the Hopf points stayed where they are (the trace is zero at
the same place) but every period came out wrong by a factor of up to sqrt(2).
"""
import libsbml
import pytest
import sympy as sp

from dynalyst.analysis.generic_sbml import build_symbolic_model


def _model(tmp_path, boundary):
    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel(); m.setId("rr_boundary")
    c = m.createCompartment(); c.setId("cell"); c.setSize(1); c.setConstant(True); c.setSpatialDimensions(3)
    for sid in ("X", "Src"):
        s = m.createSpecies(); s.setId(sid); s.setCompartment("cell"); s.setInitialAmount(1)
        s.setHasOnlySubstanceUnits(True); s.setConstant(False)
        s.setBoundaryCondition(boundary if sid == "X" else True)
    p = m.createParameter(); p.setId("k"); p.setValue(2.0); p.setConstant(True)
    r = m.createRateRule(); r.setVariable("X"); r.setMath(libsbml.parseL3Formula("k - X"))
    rx = m.createReaction(); rx.setId("R"); rx.setReversible(False)
    sr = rx.createReactant(); sr.setSpecies("Src"); sr.setStoichiometry(1); sr.setConstant(True)
    sp_ = rx.createProduct(); sp_.setSpecies("X"); sp_.setStoichiometry(1); sp_.setConstant(True)
    kl = rx.createKineticLaw(); kl.setMath(libsbml.parseL3Formula("k"))
    path = tmp_path / "rr.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    return build_symbolic_model(path)


def test_a_boundary_species_with_a_rate_rule_follows_the_rule_only(tmp_path):
    mod = _model(tmp_path, boundary=True)
    i = mod["state_names"].index("X")
    X, k = sp.Symbol("X"), sp.Symbol("k")
    rhs = mod["rhs"][i]
    val = float(rhs.subs({s: (1.0 if str(s) == "X" else 2.0) for s in rhs.free_symbols}))
    assert val == pytest.approx(1.0)          # k - X = 2 - 1, not 2*(k - X) or (k - X) + k
