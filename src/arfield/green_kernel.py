"""Green's function for a point source and its spatial gradient.

Closed-form kernel of the DPSM: G(R) = exp(ikR) / (4*pi*R), consistent
with the e^{-i*omega*t} time convention (see docs/adr/0001-time-convention.md).
"""

import numpy as np


def green(r: np.ndarray, kf: float) -> np.ndarray:
    """Evaluate the Green's function G(r) = exp(i*kf*r) / (4*pi*r).

    Parameters
    ----------
    r : np.ndarray
        Distance between source and observation point, in meters. Must be
        strictly positive; r = 0 is not handled (sources are always placed
        away from observation points by construction, see r_s in the DPSM
        formulation).
    kf : float
        Wavenumber, 2*pi/wavelength, in rad/m.

    Returns
    -------
    np.ndarray
        Complex128 array, same shape as r.
    """
    return np.exp(1j * kf * r) / (4 * np.pi * r)


def gradient_green(r: np.ndarray, e_r: np.ndarray, kf: float) -> np.ndarray:
    """Evaluate the spatial gradient of the Green's function.

    grad G = G(r) * (i*kf - 1/r) * e_r

    Parameters
    ----------
    r : np.ndarray
        Distance between source and observation point, in meters.
    e_r : np.ndarray, shape (..., 3)
        Unit vector pointing FROM the source TO the observation point.
        Reversing this direction flips the sign of the result. Leading
        dimensions must match r.
    kf : float
        Wavenumber, in rad/m.

    Returns
    -------
    np.ndarray
        Complex128 array, shape matching e_r.
    """
    grad_r = green(r, kf) * (1j * kf - 1 / r)
    return grad_r[..., None] * e_r
