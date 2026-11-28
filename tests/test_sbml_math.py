"""Every SBML (MathML) function is read with its standard meaning, by both readers: the
numeric one (generic_sbml.eval_ast) and the symbolic one (eval_ast_sympy)."""
import math

import libsbml
import pytest
import sympy as sp

from dynalyst.analysis.generic_sbml import (AVOGADRO, DelayModel, eval_ast, eval_ast_sympy)

X, Y = 0.7, 2.5
ENV_NUM = {"x": X, "y": Y}
ENV_SYM = {"x": sp.Symbol("x"), "y": sp.Symbol("y")}
SUBS = {sp.Symbol("x"): X, sp.Symbol("y"): Y}

CASES = [
    ("x + y * 2 - 1 / y", X + Y * 2 - 1 / Y),
    ("pow(x, y)", X ** Y), ("x^y", X ** Y), ("sqrt(y)", math.sqrt(Y)), ("root(3, y)", Y ** (1 / 3)),
    ("exp(x)", math.exp(X)), ("ln(y)", math.log(Y)), ("log10(y)", math.log10(Y)),
    ("log(2, y)", math.log(Y, 2)), ("abs(-x)", X),
    ("floor(y)", 2.0), ("ceil(y)", 3.0), ("factorial(3)", 6.0), ("factorial(y)", math.gamma(Y + 1)),
    ("sin(x)", math.sin(X)), ("cos(x)", math.cos(X)), ("tan(x)", math.tan(X)),
    ("sec(x)", 1 / math.cos(X)), ("csc(x)", 1 / math.sin(X)), ("cot(x)", 1 / math.tan(X)),
    ("asin(x)", math.asin(X)), ("acos(x)", math.acos(X)), ("atan(x)", math.atan(X)),
    ("arcsec(y)", math.acos(1 / Y)), ("arccsc(y)", math.asin(1 / Y)), ("arccot(y)", math.atan(1 / Y)),
    ("sinh(x)", math.sinh(X)), ("cosh(x)", math.cosh(X)), ("tanh(x)", math.tanh(X)),
    ("sech(x)", 1 / math.cosh(X)), ("csch(x)", 1 / math.sinh(X)), ("coth(x)", 1 / math.tanh(X)),
    ("arcsinh(x)", math.asinh(X)), ("arccosh(y)", math.acosh(Y)), ("arctanh(x)", math.atanh(X)),
    ("arcsech(x)", math.acosh(1 / X)), ("arccsch(x)", math.asinh(1 / X)),
    ("arccoth(y)", math.atanh(1 / Y)),
    ("min(x, y, 1)", min(X, Y, 1)), ("max(x, y, 1)", max(X, Y, 1)),
    ("rem(-7, 3)", math.fmod(-7, 3)), ("quotient(-7, 3)", -2.0),
    ("piecewise(1, x > y, 2, x < 1, 3)", 2.0), ("piecewise(1, x > y, 3)", 3.0),
    ("x < y && y > 1", 1.0), ("x > y || y > 1", 1.0), ("!(x > y)", 1.0),
    ("xor(x > 0, y > 0)", 0.0), ("implies(x > y, y < 0)", 1.0), ("x < 1 < y", 1.0),
    ("pi", math.pi), ("exponentiale", math.e), ("avogadro", AVOGADRO),
    ("true", 1.0), ("false", 0.0), ("INF", math.inf),
]


@pytest.mark.parametrize("formula,expected", CASES, ids=[c[0] for c in CASES])
def test_every_mathml_function_has_its_meaning(formula, expected):
    node = libsbml.parseL3Formula(formula)
    assert node is not None, libsbml.getLastParseL3Error()
    num = eval_ast(node, ENV_NUM)
    sym = eval_ast_sympy(node, ENV_SYM)
    if isinstance(sym, (sp.logic.boolalg.Boolean, sp.core.relational.Relational)):
        sym = 1.0 if bool(sym.subs(SUBS)) else 0.0
    else:
        sym = float(sp.N(sym.subs(SUBS)))
    if math.isinf(expected):
        assert math.isinf(num) and math.isinf(sym)
    else:
        assert num == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert sym == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_delay_is_refused_plainly_by_the_numeric_reader():
    with pytest.raises(DelayModel):
        eval_ast(libsbml.parseL3Formula("delay(x, 2)"), ENV_NUM)


def test_rate_of_is_the_variables_own_rate(tmp_path):
    from dynalyst.analysis.generic_sbml import build_symbolic_model
    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    c = m.createCompartment(); c.setId("cell"); c.setSize(1); c.setConstant(True)
    for sid in ("x", "y"):
        s = m.createSpecies(); s.setId(sid); s.setCompartment("cell"); s.setInitialConcentration(1)
        s.setHasOnlySubstanceUnits(False); s.setBoundaryCondition(False); s.setConstant(False)
    k = m.createParameter(); k.setId("k"); k.setValue(0.5); k.setConstant(True)
    r = m.createRateRule(); r.setVariable("x"); r.setMath(libsbml.parseL3Formula("-k * x"))
    r = m.createRateRule(); r.setVariable("y"); r.setMath(libsbml.parseL3Formula("rateOf(x) + rateOf(k)"))
    path = tmp_path / "rate_of.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    model = build_symbolic_model(path)
    rhs = dict(zip(model["state_names"], model["rhs"]))
    x, kk = sp.Symbol("x"), sp.Symbol("k")
    assert sp.simplify(rhs["y"] - (-kk * x)) == 0      # rateOf of a constant is 0


def test_delay_equations_are_shown_but_not_analysed(tmp_path):
    import json
    from dynalyst.analysis.generic_sbml import build_symbolic_model
    from dynalyst.analysis.ode_model import get_odes
    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    c = m.createCompartment(); c.setId("cell"); c.setSize(1); c.setConstant(True)
    s = m.createSpecies(); s.setId("x"); s.setCompartment("cell"); s.setInitialConcentration(1)
    s.setHasOnlySubstanceUnits(False); s.setBoundaryCondition(False); s.setConstant(False)
    for pid, val in (("k", 0.5), ("tau", 2.0)):
        p = m.createParameter(); p.setId(pid); p.setValue(val); p.setConstant(True)
    r = m.createRateRule(); r.setVariable("x"); r.setMath(libsbml.parseL3Formula("-k * delay(x, tau)"))
    path = tmp_path / "delayed.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    with pytest.raises(DelayModel):
        build_symbolic_model(path)
    out = json.loads(get_odes(str(path)))
    assert "error" not in out
    assert "delay(x, tau)" in out["odes"]["dx/dt"]
    assert list(out["delay_differential_equation"]["delayed_terms"]) == ["delay(x, tau)"]


def test_expanded_odes_are_regrouped_exactly_for_reading():
    from dynalyst.analysis.ode_model import readable
    E, S, C, kf, kr, kc, ET = sp.symbols("E S C kf kr kc ET")
    # binding to free enzyme (ET - C), written fully expanded as some files store it
    rhs = sp.expand(-kf * S * (ET - C) + kr * C)
    text = readable(rhs, [S, C], [kf, kr, kc, ET])
    assert text == "C*kr - S*kf*(ET - C)"
    assert sp.expand(sp.sympify(text, locals={"S": S, "C": C}) - rhs) == 0
    # not a polynomial in the states: left as written
    assert readable(kc * S / (1 + S), [S], [kc]) is None
