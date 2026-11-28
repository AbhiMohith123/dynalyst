"""Is a point AUTO-07p labelled HB a Hopf point? Judged from the Jacobian eigenvalues."""


def hopf_pair(eigenvalues) -> bool:
    """Do these eigenvalues have a complex pair on the imaginary axis (a Hopf point), as opposed
    to two real ones summing to zero (a neutral saddle, where AUTO's test also fires)? Judged
    against the pair itself: a stiff model's other eigenvalues (1e6 in BIOMD5) do not count."""
    import numpy as np
    ev = np.asarray(eigenvalues, dtype=complex)
    ev = ev[np.isfinite(ev)]
    if not ev.size:
        return False
    tiny = 1e-9 * max(1.0, float(np.max(np.abs(ev))))
    return bool(np.any((np.abs(ev.imag) > tiny) & (np.abs(ev.real) <= 1e-3 * np.abs(ev.imag) + 1e-12)))


if __name__ == "__main__":      # self-check: python -m dynalyst.analysis.auto_diagnostics
    assert hopf_pair([-1e6, -0.6, 1e-5 + 0.19338j, 1e-5 - 0.19338j])     # BIOMD5: stiff, still Hopf
    assert not hopf_pair([-1e6, 0.4, -0.4])                                 # neutral saddle
