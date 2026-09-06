"""Influence matrices: the field of every source at every target point."""

import numpy as np

from .green_kernel import gradient_green, green
from .pairwise import separation

__all__ = ["compute_euler_gradn_green_TS", "compute_grad_green_TSj", "compute_green_TS"]


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
    distances are used here. That unused ``(M, N, 3)`` float64 array costs
    ``24 * M * N`` bytes, half again the size of the complex matrix being
    returned, and it is the largest single allocation this call makes:
    measured at ``M = 20000`` and ``N = 480``, the peak is 614 MB, of which
    the unused array is 230 and the returned matrix 154. Quoted as a ratio
    the figure moves with whatever else the kernel is holding at the time;
    the bytes do not.

    Binding it to ``_`` does not free it. The name is an ordinary local and
    the array stays alive until this function returns. Python has no
    equivalent of MATLAB's ``nargout``, so a callee cannot learn that one of
    its outputs will be discarded and skip building it.

    It is left this way on purpose: one separation routine shared by the
    whole module is worth more than the allocation at the sizes this package
    targets. The fix, on the day a machine runs out of room, is a
    distances-only path in ``pairwise``, not a change here.

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


def compute_grad_green_TSj(
    targets: np.ndarray, sources: np.ndarray, kf: float
) -> np.ndarray:
    """Green's gradient contributed by every point source at every target.

    Entry ``(m, n, :)`` is the gradient of the Green's function between
    target ``m`` and source ``n``, a vector per pair, so the array maps
    source strengths to the pressure gradient. Contracting over sources
    with strengths ``A`` of shape ``(N,)`` gives ``grad(p)`` of shape
    ``(M, 3)``.

    Unprojected, unlike ``compute_euler_gradn_green_TS``. Nothing picks a
    direction here because there is no boundary condition being written: a
    target is a point in the fluid, not a point on a surface, and it carries
    no normal. That is why this function takes no ``normals``.

    No medium either. The Euler factor is not applied and neither ``c`` nor
    ``rho`` is asked for, because the array holds the bare kernel gradient,
    which belongs to the Green's function and not to the fluid. Whoever
    needs velocity divides by ``i * omega * rho`` once, after contracting,
    which is ``3 * M`` operations instead of ``3 * M * N``.

    Parameters
    ----------
    targets : ndarray, shape (M, 3)
        Points where the gradient is evaluated, in meters. An arbitrary
        cloud, unrelated to the sources; ``M`` need not equal ``N``.
    sources : ndarray, shape (N, 3)
        Point-source positions, in meters. For a surface these are the
        retreated positions, ``Source.positions``, not the surface points.
    kf : float
        Wavenumber of the fluid, ``2 * pi / wavelength``, in rad/m.

    Returns
    -------
    ndarray, shape (M, N, 3)
        Complex128. Entries carry units of inverse length squared, so the
        product with strengths in Pa*m gives a pressure gradient in Pa/m.

    Notes
    -----
    The gradient is taken with respect to the *target* coordinate. Taken
    with respect to the source it is the negative of this, and the two are
    easy to confuse because the only visible difference is a sign that
    survives every magnitude check.

    Contract with ``grad.transpose(0, 2, 1) @ A``. The transpose is a view,
    not a copy, and the product allocates only the result. ``tensordot`` and
    the broadcasting form both materialize a temporary the size of the input
    array, which at these shapes is the dominant allocation of the call.

    This function is deliberately not the building block of
    ``compute_euler_gradn_green_TS``. That one exists precisely because it
    never materializes the ``(M, N, 3)`` array: it projects pair by pair and
    keeps the peak at the size of the influence matrix. Rewriting it to call
    this function and project afterwards would triple its memory and remove
    its reason to exist. The two share a preamble of two lines, which is a
    second repetition, and the rule of three says to leave it alone.

    Memory is ``48 * M * N`` bytes, three times the influence matrix of the
    same shape. This is the array that makes a volumetric evaluation
    expensive, and the place where blocking over targets will be needed
    first.

    Zero distance is not handled, on the same construction argument as the
    rest of the module: sources sit behind the surface, and a target that
    coincides with one means that construction was bypassed.

    References
    ----------
    Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
    Wiley (2007), chapter 1.
    """
    r_TS, e_TSj = separation(targets, sources)
    return gradient_green(r_TS, e_TSj, kf)


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
    factor = 1 / (1j * kf * c * rho)
    return gradn_green_TS * factor
