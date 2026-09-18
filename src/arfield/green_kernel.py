"""Green's function for a point source and its spatial derivatives.

Closed-form kernel of the DPSM: G(r) = exp(i*kf*r) / (4*pi*r), consistent
with the e^{-i*omega*t} time convention (see docs/adr/0001-time-convention.md).

Every function here is elementwise on ``r``: pass a scalar, an ``(N,)``
vector or an ``(M, N)`` table and get the same shape back. Nothing in this
module knows about clouds of points, collocation, or source strengths.

Each derived quantity comes in two spellings. The plain one (``d_green_dr``,
``gradient_green``, ...) takes ``r`` and computes everything it needs, so it
reads as the formula and is what tests compare against. The ``_from``
spelling takes its immediate predecessor in the chain already evaluated and
computes nothing twice; a caller that holds ``green(r, kf)`` can obtain every
other kernel from it with a single exponential. The chain is

    g -> dgdr -> {grad_green, aux_fun_h} -> aux_fun_s -> hessian_green

and each ``_from`` function receives exactly the link before it. The plain
spelling always calls the ``_from`` one, so each formula is written once.

Two scalars have no name of their own in the literature and are called by
letter here, as in the derivation notes (gorkov_dpsm.pdf, section 7.3):

    aux_fun_h = (dG/dr) / r          so that  grad G = R * aux_fun_h
    aux_fun_s = kf**2 * G + 3 * aux_fun_h

With them the Hessian is ``delta_ij * h - e_i * e_j * s``.
"""

import numpy as np

__all__ = [
    "aux_fun_h",
    "aux_fun_h_from",
    "aux_fun_s",
    "aux_fun_s_from",
    "d_green_dr",
    "d_green_dr_from",
    "gradient_green",
    "gradient_green_from",
    "green",
    "hessian_green",
    "hessian_green_from",
]


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


def d_green_dr_from(g: np.ndarray, r: np.ndarray, kf: float) -> np.ndarray:
    """Radial derivative of the Green's function, from G already evaluated.

    dG/dr = G(r) * (i*kf - 1/r)

    Parameters
    ----------
    g : np.ndarray
        ``green(r, kf)``, complex128, same shape as r.
    r : np.ndarray
        Distance between source and observation point, in meters.
    kf : float
        Wavenumber, in rad/m.

    Returns
    -------
    np.ndarray
        Complex128 array, same shape as r.
    """
    return g * (1j * kf - 1 / r)


def d_green_dr(r: np.ndarray, kf: float) -> np.ndarray:
    """Radial derivative of the Green's function.

    dG/dr = G(r) * (i*kf - 1/r)

    Evaluates G internally. Callers that already hold ``green(r, kf)``
    should use `d_green_dr_from` instead.

    Parameters
    ----------
    r : np.ndarray
        Distance between source and observation point, in meters.
    kf : float
        Wavenumber, in rad/m.

    Returns
    -------
    np.ndarray
        Complex128 array, same shape as r.
    """
    return d_green_dr_from(green(r, kf), r, kf)


def gradient_green_from(dgdr: np.ndarray, e_r: np.ndarray) -> np.ndarray:
    """Spatial gradient of the Green's function, from dG/dr already evaluated.

    grad G = (dG/dr) * e_r

    Parameters
    ----------
    dgdr : np.ndarray
        ``d_green_dr(r, kf)``, complex128, shape matching the leading
        dimensions of e_r.
    e_r : np.ndarray, shape (..., 3)
        Unit vector pointing FROM the source TO the observation point.
        Reversing this direction flips the sign of the result.

    Returns
    -------
    np.ndarray
        Complex128 array, same shape as e_r.
    """
    return dgdr[..., None] * e_r


def gradient_green(r: np.ndarray, e_r: np.ndarray, kf: float) -> np.ndarray:
    """Spatial gradient of the Green's function.

    grad G = G(r) * (i*kf - 1/r) * e_r

    Evaluates G and dG/dr internally. Callers that already hold dG/dr
    should use `gradient_green_from` instead.

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
        Complex128 array, same shape as e_r.
    """
    return gradient_green_from(d_green_dr(r, kf), e_r)


def aux_fun_h_from(dgdr: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Gradient scale h = (dG/dr) / r, from dG/dr already evaluated.

    Defined so that ``grad G = R * h`` with ``R = r * e_r`` the separation
    vector, which is the form the Hessian is derived from.

    Parameters
    ----------
    dgdr : np.ndarray
        ``d_green_dr(r, kf)``, complex128, same shape as r.
    r : np.ndarray
        Distance between source and observation point, in meters.

    Returns
    -------
    np.ndarray
        Complex128 array, same shape as r.
    """
    return dgdr / r


def aux_fun_h(r: np.ndarray, kf: float) -> np.ndarray:
    """Gradient scale h = (dG/dr) / r = exp(i*kf*r) * (i*kf*r - 1) / (4*pi*r**3).

    Evaluates G and dG/dr internally. Callers that already hold dG/dr
    should use `aux_fun_h_from` instead.

    Parameters
    ----------
    r : np.ndarray
        Distance between source and observation point, in meters.
    kf : float
        Wavenumber, in rad/m.

    Returns
    -------
    np.ndarray
        Complex128 array, same shape as r.
    """
    return aux_fun_h_from(d_green_dr(r, kf), r)


def aux_fun_s_from(g: np.ndarray, h: np.ndarray, kf: float) -> np.ndarray:
    """Hessian scale s = kf**2 * G + 3 * h, from G and h already evaluated.

    Scalar coefficient of the outer-product term of the Hessian; see
    `hessian_green_from`.

    Parameters
    ----------
    g : np.ndarray
        ``green(r, kf)``, complex128.
    h : np.ndarray
        ``aux_fun_h(r, kf)``, complex128, same shape as g.
    kf : float
        Wavenumber, in rad/m.

    Returns
    -------
    np.ndarray
        Complex128 array, same shape as g.
    """
    return kf**2 * g + 3 * h


def aux_fun_s(r: np.ndarray, kf: float) -> np.ndarray:
    """Hessian scale s = kf**2 * G + 3 * h.

    Evaluates G twice internally (once directly, once inside h). Callers
    that already hold G and h should use `aux_fun_s_from` instead.

    Parameters
    ----------
    r : np.ndarray
        Distance between source and observation point, in meters.
    kf : float
        Wavenumber, in rad/m.

    Returns
    -------
    np.ndarray
        Complex128 array, same shape as r.
    """
    return aux_fun_s_from(green(r, kf), aux_fun_h(r, kf), kf)


def hessian_green_from(h: np.ndarray, s: np.ndarray, e_r: np.ndarray) -> np.ndarray:
    """Hessian of the Green's function, from h and s already evaluated.

    d2G/dx_i dx_j = delta_ij * h - e_i * e_j * s

    Elementwise over every leading dimension: for an ``(M, N)`` table of
    pairs the result is the full ``(M, N, 3, 3)`` tensor, nothing summed.
    The contraction with the source strengths, which is what a field
    evaluation needs, is not done here; it belongs to whoever holds A.

    Two identities hold and make cheap tests: the result is symmetric in
    ``(i, j)``, and its trace equals ``-kf**2 * G`` (Helmholtz).

    Parameters
    ----------
    h : np.ndarray
        ``aux_fun_h(r, kf)``, complex128, shape matching the leading
        dimensions of e_r.
    s : np.ndarray
        ``aux_fun_s(r, kf)``, complex128, same shape as h.
    e_r : np.ndarray, shape (..., 3)
        Unit vector pointing FROM the source TO the observation point.
        The Hessian is even in e_r, so its direction does not affect the
        result.

    Returns
    -------
    np.ndarray
        Complex128 array, shape ``e_r.shape + (3,)``, i.e. ``(..., 3, 3)``.
    """
    outer_term = e_r[..., :, None] * e_r[..., None, :]
    return np.eye(3) * h[..., None, None] - outer_term * s[..., None, None]


def hessian_green(r: np.ndarray, e_r: np.ndarray, kf: float) -> np.ndarray:
    """Hessian of the Green's function.

    d2G/dx_i dx_j = delta_ij * h - e_i * e_j * s, with h and s as in
    `aux_fun_h` and `aux_fun_s`.

    Evaluates G three times internally. Reference spelling for tests and
    for single-point use; production code chains from ``green`` through
    the ``_from`` functions.

    Parameters
    ----------
    r : np.ndarray
        Distance between source and observation point, in meters.
    e_r : np.ndarray, shape (..., 3)
        Unit vector pointing FROM the source TO the observation point.
        Leading dimensions must match r.
    kf : float
        Wavenumber, in rad/m.

    Returns
    -------
    np.ndarray
        Complex128 array, shape ``(..., 3, 3)``.
    """
    return hessian_green_from(aux_fun_h(r, kf), aux_fun_s(r, kf), e_r)
