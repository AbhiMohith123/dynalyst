"""Locate a fold (limit point) exactly from an approximate one.

A continuation reports a fold at the step where the parameter turned, which can lie a step
away from the true turning point. Checking "is the smallest eigenvalue small next to the
largest?" does not separate the two in a stiff model: there the largest eigenvalue is so large
that any slow mode passes. Instead, the fold is solved for:

    f(y, p) = 0,   mu(y, p) = 0,

where mu is the real eigenvalue of the Jacobian closest to zero, by Newton's method from the
reported point. A reported fold is accepted only if the solution lies within a relative
tolerance of it in the parameter; otherwise the solved value is the fold, and the reported one
was off by the difference.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import root

ACCEPT_REL = 1e-4     # a reported fold within this relative distance of the solved one is accepted


def _real_mask(ev: np.ndarray) -> np.ndarray:
    """Eigenvalues whose imaginary part is zero to rounding."""
    return np.abs(ev.imag) <= 1e-7 * (1.0 + np.abs(ev))


def _mu_index(ev: np.ndarray) -> int:
    """Index of the real eigenvalue closest to zero. A fold has a real zero eigenvalue, so a
    slow, weakly damped complex pair nearer to zero must not be taken for it; only when no
    eigenvalue is real does this fall back to the eigenvalue closest to zero."""
    real = np.where(_real_mask(ev))[0]
    pool = real if real.size else np.arange(ev.size)
    return int(pool[np.argmin(np.abs(ev[pool]))])


def _mu(J: np.ndarray) -> float:
    """The real eigenvalue closest to zero (see _mu_index)."""
    ev = np.linalg.eigvals(J)
    return float(ev[_mu_index(ev)].real) if ev.size else 0.0


def refine_fold(f, jac, y0, p0: float, accept_rel: float = ACCEPT_REL) -> dict:
    """f(y, p) -> dy/dt and jac(y, p) -> df/dy for the (reduced) model; (y0, p0) the reported
    fold. Returns {converged, p, y, rel_shift, accepted, residual, mu, detail}."""
    y0 = np.asarray(y0, dtype=float)
    n = y0.size
    scale_y = np.maximum(np.abs(y0), 1e-12)

    def G(z):
        y, p = z[:n], z[n]
        with np.errstate(all="ignore"):
            r = np.asarray(f(y, p), dtype=float)
            m = _mu(np.asarray(jac(y, p), dtype=float))
        return np.append(r, m)

    z0 = np.append(y0, p0)
    best = None
    for method in ("hybr", "lm"):
        try:
            sol = root(G, z0, method=method, options={"xtol": 1e-13} if method == "hybr"
                       else {"xtol": 1e-14, "ftol": 1e-14})
        except Exception:
            continue
        z = sol.x
        if not np.all(np.isfinite(z)):
            continue
        y, p = z[:n], float(z[n])
        with np.errstate(all="ignore"):
            r = np.asarray(f(y, p), dtype=float)
            J = np.asarray(jac(y, p), dtype=float)
        jn = float(np.max(np.abs(J))) if J.size else 0.0
        residual = float(np.max(np.abs(r))) if r.size else 0.0
        ok_res = residual <= 1e-8 * (1.0 + jn * (1.0 + float(np.max(np.abs(y)))))
        ev = np.linalg.eigvals(J)
        k = _mu_index(ev) if ev.size else 0
        mu = float(ev[k].real) if ev.size else 0.0
        is_real = bool(ev.size and _real_mask(ev)[k])
        # mu must be zero against the eigenvalue next to it, not against the largest one,
        # and the zero eigenvalue must be real (a fold, not a Hopf point)
        others = np.sort(np.abs(np.delete(ev, k))) if ev.size > 1 else np.array([1.0])
        ok_mu = is_real and abs(mu) <= 1e-6 * max(float(others[0]) if others.size else 1.0, 1e-300)
        # the solution must stay near the reported state (not jump to another fold)
        near = bool(np.max(np.abs(y - y0) / (scale_y + np.max(np.abs(y0)))) < 0.5)
        cand = dict(converged=bool(ok_res and ok_mu and near), p=p, y=y.tolist(),
                    residual=residual, mu=mu)
        if cand["converged"]:
            best = cand
            break
        best = best or cand
    if best is None:
        return dict(converged=False, accepted=False, p=None, y=None, rel_shift=None,
                    detail="the fold equations could not be solved from the reported point")
    rel = abs(best["p"] - p0) / max(abs(best["p"]), abs(p0), 1e-12)
    best["rel_shift"] = float(rel)
    best["accepted"] = bool(best["converged"] and rel <= accept_rel)
    if not best["converged"]:
        best["detail"] = (f"not confirmed: Newton on f = 0, zero eigenvalue did not converge near "
                          f"the reported point (residual {best['residual']:.2g})")
    elif best["accepted"]:
        best["detail"] = f"confirmed: the fold equations give {best['p']:.10g} (relative shift {rel:.1g})"
    else:
        best["detail"] = (f"moved: the fold equations put the fold at {best['p']:.10g}, "
                          f"not {p0:.10g} (relative shift {rel:.2g})")
    return best
