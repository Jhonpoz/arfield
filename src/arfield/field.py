"""Field evaluation: pressure and velocity from known source strengths.

Step five of the DPSM, and the last one. Everything upstream builds
geometry, assembles matrices and solves for the strengths ``A``; this
module spends them. It is the only module in the package that knows ``A``
exists, which is why nothing here is reusable by the assembly code and
nothing there depends on this.

Targets are an arbitrary cloud of points ``(M, 3)``, never a grid. A flat
list has no axis order to get wrong, so the reshape between a grid and a
list lives in the script that plots, in one line each way, and never in
here. Gor'kov, array design and the spherical-harmonic bridge all hang off
this step.
"""

import numpy as np

from .influence import compute_grad_green_TSj, compute_green_TS

__all__ = ["pressure", "velocity"]


def pressure(
    targets: np.ndarray, source_strengths: np.ndarray, sources: np.ndarray, kf: float
) -> np.ndarray:
    """Acoustic pressure at every target point.

    Sums the contribution of every point source: ``p = green_TS @ A``. The
    matrix is built here and discarded, since a different set of targets
    needs a different matrix and there is nothing to reuse between calls.

    Needs no medium. Pressure from known strengths is geometry and
    wavenumber only; ``c`` and ``rho`` enter the package when the boundary
    condition is imposed, and again in `velocity`, but not here.

    Parameters
    ----------
    targets : ndarray, shape (M, 3)
        Points where the pressure is evaluated, in meters. Any cloud: a scan
        line, a flattened grid, a single point. ``M`` need not equal ``N``.
    source_strengths : ndarray, shape (N,)
        Complex source strengths in Pa*m, one per source, ordered as
        ``sources``. From `solver.solve_strength`, or assigned directly in
        the discretized Rayleigh branch.
    sources : ndarray, shape (N, 3)
        Point-source positions, in meters. For a surface these are
        ``Source.positions``, the retreated layer, and they must be the same
        array the strengths were solved against. Passing the surface points
        instead changes the geometry that ``A`` was computed for, and the
        result is a plausible-looking field that means nothing.
    kf : float
        Wavenumber of the fluid, ``2 * pi / wavelength``, in rad/m.

    Returns
    -------
    ndarray, shape (M,)
        Complex128, in Pa. One value per target, in target order.

    Notes
    -----
    The matrix returned by the assembly is ``16 * M * N`` bytes, but the
    peak of the call is higher: `influence.compute_green_TS` is also holding
    the distances and the unit vectors it has no use for. Measured at
    ``M = 20000`` and ``N = 480``, the peak is 614 MB against a 154 MB
    result. A volumetric evaluation is where that first becomes the limiting
    number, and blocking over targets is the answer when it does; it is not
    needed for a scan line or a plane.

    Phase follows the ``exp(-i * omega * t)`` convention, so the phase of
    ``p`` advances by ``kf * R`` with distance from a source. Comparing
    phases at two ranges through ``np.angle`` needs the values to be within
    half a wavelength of each other, or the comparison must unwrap; the
    wrap is not a sign error.

    References
    ----------
    Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
    Wiley (2007), chapter 1.
    """
    return compute_green_TS(targets, sources, kf) @ source_strengths


def velocity(
    targets: np.ndarray,
    source_strengths: np.ndarray,
    sources: np.ndarray,
    kf: float,
    c: float,
    rho: float,
) -> np.ndarray:
    """Acoustic particle velocity at every target point.

    The pressure gradient summed over sources, then Euler's equation:
    ``v = grad(p) / (i * omega * rho)``. All three components, not a
    projection: a target in the fluid carries no normal, and ``abs(v)**2``
    in the Gor'kov potential needs the whole vector.

    The Euler factor is applied to the contracted result, ``(M, 3)``, and
    not to the gradient array, ``(M, N, 3)``. That is ``3 * M``
    multiplications instead of ``3 * M * N``, and it keeps the transpose a
    view instead of forcing a full complex copy of the largest array the
    package builds.

    Parameters
    ----------
    targets : ndarray, shape (M, 3)
        Points where the velocity is evaluated, in meters.
    source_strengths : ndarray, shape (N,)
        Complex source strengths in Pa*m, one per source, ordered as
        ``sources``.
    sources : ndarray, shape (N, 3)
        Point-source positions, in meters, the same array the strengths were
        solved against.
    kf : float
        Wavenumber of the fluid, ``2 * pi / wavelength``, in rad/m.
    c : float
        Speed of sound in the fluid, in m/s. Enters only through
        ``omega = kf * c``, so it is not independent of ``kf``; values that
        disagree are not detected.
    rho : float
        Density of the fluid, in kg/m^3.

    Returns
    -------
    ndarray, shape (M, 3)
        Complex128, in m/s. Cartesian components, in target order.

    Notes
    -----
    The gradient array is ``48 * M * N`` bytes, three times the influence
    matrix of `pressure` for the same targets, and the call peaks at about
    twice that: 922 MB measured at ``M = 20000`` and ``N = 480``. On the
    shapes the package is actually called with, that is its memory ceiling
    and the first place blocking over targets will be needed. On equal
    shapes `influence.compute_euler_gradn_green_TS` peaks higher still, but
    it is only ever called with ``M`` equal to ``N`` over the surface
    itself, a few hundred points, so it never approaches this.

    For a single source the closed form is
    ``v = p * (i * kf - 1 / R) / (i * omega * rho)`` along the unit vector
    from source to target, and the ratio ``p / v_R`` is
    ``rho * c / (1 + i / (kf * R))``, which tends to the specific impedance
    of the fluid in the far field. A missing ``c`` or a mismatched ``omega``
    breaks that ratio by a factor large enough to see by eye.

    The sign of the Euler factor follows ``exp(-i * omega * t)``. Under the
    opposite convention it is negated, together with every phase in the
    package.

    References
    ----------
    Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
    Wiley (2007), chapter 1.
    """
    grad_green_TSj = compute_grad_green_TSj(targets, sources, kf)
    factor = 1 / (1j * kf * c * rho)
    return factor * (grad_green_TSj.transpose(0, 2, 1) @ source_strengths)
