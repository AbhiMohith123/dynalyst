"""Writing model equations as Fortran for AUTO-07p, safely for any model.

Model names never reach the Fortran source: states become U(i) and
parameters PAR(i). Fortran is case-insensitive, limits names to 63
characters, forbids a leading underscore and reserves U, PAR, F, T, NDIM...,
so ``Km`` next to ``KM``, a species called ``F`` or an SBML id like
``_S1`` would otherwise break compilation.

Every expression is checked before it is written: anything left that is
not U(i)/PAR(i) (an undefined symbol, time, an unexpanded function) or that
Fortran cannot express raises FortranError naming the equation and the
construct, instead of a compiler error or a sympy printing error.
"""
import sympy as sp
from sympy.core.function import AppliedUndef


class FortranError(ValueError):
    """An equation cannot be written as Fortran."""


_INT_MAX = 2 ** 31 - 1      # Fortran's default INTEGER


# AUTO-07p writes PAR(11) itself (the period, estimated at every equilibrium step with
# complex eigenvalues and at Hopf points) and uses PAR(11)..PAR(20) for its own extended
# systems. A model parameter placed there is silently overwritten, so the continuation solves
# a different model: model parameters take PAR(1)..PAR(10), then PAR(21) onwards.
AUTO_RESERVED_PAR = range(11, 21)


def auto_par_index(j: int) -> int:
    """The AUTO PAR index of the j-th model parameter (j from 0), skipping AUTO's own."""
    return j + 1 if j < 10 else j + 1 + len(AUTO_RESERVED_PAR)


def auto_symbols(states, params) -> dict:
    """{model symbol: U(i) / PAR(auto_par_index(j))} for the given order."""
    sub = {sp.Symbol(str(s)): sp.Symbol(f"U({i + 1})") for i, s in enumerate(states)}
    sub.update({sp.Symbol(str(p)): sp.Symbol(f"PAR({auto_par_index(j)})")
                for j, p in enumerate(params)})
    return sub


def _auto_name(sym) -> bool:
    n = sym.name
    return (n.startswith("U(") or n.startswith("PAR(")) and n.endswith(")")


def fortran_assign(expr, target: str, where: str, standard: int = 95) -> str:
    """``target = expr`` as free-form Fortran (continuation lines included).
    `expr` must already be in U(i)/PAR(i) form."""
    expr = sp.sympify(expr)
    stray = sorted(str(s) for s in expr.free_symbols if not _auto_name(s))
    if stray:
        raise FortranError(f"{where} uses {stray}, which is neither a state variable nor a "
                           "parameter of the model")
    undef = sorted({f.func.__name__ for f in expr.atoms(AppliedUndef)})
    if undef:
        raise FortranError(f"{where} calls undefined function(s) {undef}")
    # an exact integer beyond Fortran's default (32-bit) integer does not compile as a
    # literal: write it as a double, the precision AUTO computes in anyway
    big = {n: sp.Float(int(n), 17) for n in expr.atoms(sp.Integer) if abs(int(n)) > _INT_MAX}
    if big:
        expr = expr.xreplace(big)
    try:
        return sp.fcode(expr, assign_to=target, source_format="free", standard=standard)
    except Exception as e:  # PrintMethodNotImplementedError, NotImplementedError, ...
        kinds = sorted({type(a).__name__ for a in sp.preorder_traversal(expr)
                        if isinstance(a, sp.Function) and not isinstance(a, AppliedUndef)})
        raise FortranError(f"{where} cannot be written as Fortran: {type(e).__name__} "
                           f"({str(e)[:200]}); functions in it: {kinds}") from None
