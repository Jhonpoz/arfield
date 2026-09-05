"""Influence matrices: the field of every source at every target point."""

import numpy as np

from .green_kernel import gradient_green, green
from .pairwise import separation

__all__ = ["compute_euler_gradn_green_TS", "compute_green_TS"]


def compute_green_TS(targets: np.ndarray, sources: np.ndarray, kf: float) -> np.ndarray:
    """Pressure contributed by every point source at every target point.

    Entry ``(m, n)`` is the Green's function evaluated between target ``m``
    and source ``n``, so the matrix maps source strengths to pressure:
    ``p = matrix @ A``. Nothing is summed here; the sum over sources is the
    matrix-vector product, and it belongs to whoever owns ``A``.

    Targets are an arbitrary cloud of points, unrelated to the sources. They
    may be a scan line, a volume, or the collocation points of the surface
    itself, and there is no requirement that ``M`` equal ``N``.

    Parameters
    ----------
    targets : ndarray, shape (M, 3)
        Points where the pressure is evaluated, in meters.
    sources : ndarray, shape (N, 3)
        Point-source positions, in meters. For a surface these are the
        retreated positions, ``Source.positions``, not the surface points.
    kf : float
        Wavenumber of the fluid, ``2 * pi / wavelength``, in rad/m.

    Returns
    -------
    ndarray, shape (M, N)
        Complex128. Entries carry units of inverse length, so the product
        with strengths in Pa*m gives pressure in Pa.

    Notes
    -----
    The ``1 / (4 * pi)`` of the free-space Green's function is kept inside
    the kernel. Placko and Kundu absorb it into the source strengths
    instead, so strengths computed here are not numerically comparable with
    the book's, although every field derived from them is.

    Separation returns unit vectors alongside the distances, and only the
    distances are used here. The unused ``(M, N, 3)`` array is three times
    the size of the distances and roughly doubles the peak allocation of
    this call. It is left that way on purpose: sharing one separation
    routine matters more than the allocation at the sizes this package
    targets, and the trade is worth revisiting only when ``M * N`` grows
    past about 1e7.

    Zero distance is not handled. Sources sit a distance ``rs`` behind the
    surface and collocation points are offset tangentially, so a target
    never coincides with a source by construction. A division warning from
    this call means that construction was bypassed.

    References
    ----------
    Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
    Wiley (2007), chapter 1.
    """
    r_TS, _ = separation(targets, sources)
    return green(r_TS, kf)


def compute_euler_gradn_green_TS(
    targets: np.ndarray,
    normals: np.ndarray,
    sources: np.ndarray,
    kf: float,
    c: float,
    rho: float,
) -> np.ndarray:
    """Normal velocity contributed by every point source at every target.

    Entry ``(m, n)`` is the Green's gradient projected on the normal at
    target ``m`` and divided by ``i * omega * rho``, so the matrix maps
    source strengths to normal velocity: ``v_n = matrix @ A``. Imposing a
    prescribed ``v_n`` at the collocation points of a surface turns that
    into the linear system the DPSM solves.

    The projection is what collapses the spatial axis: the gradient is a
    vector per pair, the normal picks one direction out of it, and the
    result is scalar per pair. The normal used is the one at the *target*,
    because the equation being written is the boundary condition at that
    point, not a property of the source. On a flat surface every normal is
    the same and the distinction is invisible; on a curved one it is not.

    Euler's equation is applied here and only here in the package, which is
    what the ``euler_`` prefix marks. Downstream code multiplies by
    strengths and never divides by ``i * omega * rho`` again.

    Parameters
    ----------
    targets : ndarray, shape (M, 3)
        Points where the normal velocity is evaluated, in meters. For a
        surface these are ``Source.collocation``, which are offset
        tangentially from the surface points.
    normals : ndarray, shape (M, 3)
        Unit normals at ``targets``, one per target. These belong to the
        targets, not to the sources.
    sources : ndarray, shape (N, 3)
        Point-source positions, in meters, typically ``Source.positions``.
    kf : float
        Wavenumber of the fluid, ``2 * pi / wavelength``, in rad/m.
    c : float
        Speed of sound in the fluid, in m/s. Enters only through
        ``omega = kf * c``; it describes the same wave as ``kf``, and the
        two are not independent. Passing values that disagree is not
        detected.
    rho : float
        Density of the fluid, in kg/m^3.

    Returns
    -------
    ndarray, shape (M, N)
        Complex128. Entries carry units of m*s/kg, so the product with
        strengths in Pa*m gives velocity in m/s.

    Notes
    -----
    The sign of the Euler factor follows the ``exp(-i * omega * t)`` time
    convention, under which ``v = grad(p) / (i * omega * rho)``. Under the
    opposite convention the factor is negated and every phase in the package
    reverses with it.

    Nothing here is special-cased for the diagonal. A source sits at
    ``sqrt(rs**2 + xi**2)`` from its own collocation point, which is
    strictly positive, so the self entry is computed by the same formula as
    the rest. Code that needs a diagonal branch is a sign that the source
    layer was built wrongly, not that the physics asks for one.

    The tangential offset of the collocation points is derived for the self
    term alone: it makes a single evaluation reproduce the velocity averaged
    over the cell that owns the source. Contributions from other sources are
    sampled at that same offset point, where the offset corrects nothing,
    which is part of why the analytic value overestimates the measured
    optimum. The error is second order as long as the offset stays small
    against the distance to the nearest neighbor.

    References
    ----------
    Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
    Wiley (2007), chapter 1.
    """
    r_TS, e_TSj = separation(targets, sources)
    grad_green_TSj = gradient_green(r_TS, e_TSj, kf)
    gradn_green_TS = np.einsum("mnj,mj->mn", grad_green_TSj, normals)
    factor = 1j * kf * c * rho
    return gradn_green_TS / factor
