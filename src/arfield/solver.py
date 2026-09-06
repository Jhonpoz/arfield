"""The two ways to obtain source strengths, and only those two.

Both functions return the same quantity in the same units -- one complex
strength per point source, in Pa*m -- and step five spends them
identically. What separates them is the route, and the routes have nothing
in common.

`solve_strength` inverts the influence matrix: it imposes the boundary
condition and lets every source see every other one. It is the only part of
the DPSM that inverts anything, and it contains no physics at all. It does
not know that the matrix came from a Green's function or that the vector is
a surface velocity, and it depends on nothing else in this package.

`rayleigh_strength` hands each element a strength from a closed form, with
no system and no interaction between sources. It is the opposite balance:
all physics, no algebra. It also carries a hypothesis the other one does
not -- a rigid infinite baffle -- which nothing in its arguments can check.

Having both is what makes the third curve possible. A benchmark plots the
closed-form solution against both branches, and the disagreement says where
a fault is: a solved curve that leaves the other two blames the linear
system, and two numerical curves that leave the closed form together blame
the kernel or the mesh.

The influence matrix is built elsewhere, in `influence`, and stays
reachable there on purpose. It is needed more than once -- to measure its
conditioning, to reuse a factorization across several right-hand sides, and
to compare the two branches -- and building it in here would put it out of
reach.
"""

import numpy as np

__all__ = ["rayleigh_strength", "solve_strength"]


def solve_strength(matrix: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """Solve for the source strengths that reproduce a prescribed velocity.

    The influence matrix maps strengths to normal velocity at the
    collocation points; this inverts that map. It is the whole of the
    linear-algebra step and none of the physics: the physics lives in how
    the matrix was built and in what ``v0`` means, both of which belong to
    the caller.

    Parameters
    ----------
    matrix : ndarray, shape (N, N)
        Influence matrix, as returned by
        `influence.compute_euler_gradn_green_TS` with the collocation points
        of the source layer as targets. Square: each source contributes one
        column and one equation.
    v0 : ndarray, shape (N,)
        Normal velocity imposed at each collocation point, in m/s, complex.
        A uniform piston is a constant vector, a phased array carries the
        per-element phase here, and a rigid passive surface asks for zero.
        The difference between an emitting and a reflecting surface lives
        entirely in this vector, not in the matrix.

    Returns
    -------
    ndarray, shape (N,)
        Complex128. Source strengths in Pa*m, one per source, in the same
        order as the columns of ``matrix``. Not comparable with those of
        Placko and Kundu, who absorb the ``1 / (4 * pi)`` of the Green's
        function into them; fields computed from them are.

    Notes
    -----
    A direct solve is used rather than a least-squares one. The system is
    square by construction, since every source carries exactly one
    collocation point, and least squares on a square system is a more
    expensive solve that also hides how badly conditioned the matrix is. The
    choice is worth revisiting once the conditioning has been measured
    against the retreat distance.

    A singular matrix means the sources were placed on the surface itself,
    ``alpha = 0``. Nothing else is validated: a matrix built with one
    geometry and a ``v0`` built with another will solve and return strengths
    that mean nothing.
    """
    return np.linalg.solve(matrix, v0)


def rayleigh_strength(
    v0: np.ndarray, cell_area: np.ndarray, kf: float, c: float, rho: float
) -> np.ndarray:
    """Source strengths assigned element by element, with no system to solve.

    The discretized Rayleigh-Sommerfeld integral: each element is handed the
    strength that a patch of surface of its own area, moving at its own
    velocity, would radiate on its own. ``A_m = B * dS_m`` with
    ``B = -2 i omega rho v_0``, which is Eq. (1.16a) of Placko and Kundu.
    Nothing here knows that the other sources exist, so nothing has to be
    inverted.

    **The hypothesis is a rigid infinite baffle, not a flat surface and not
    a uniform velocity.** The factor of two in ``B`` is the image of the
    rigid half-space; a radiator that does not sit in a baffle does not get
    it. Curvature is allowed: O'Neil's argument, quoted in section 1.3.3 of
    the book, is that the same integral holds over a gently curved surface,
    which is what makes the focused cap of Eq. (1.32) fair game. A
    per-element ``v0`` is allowed too, which is what makes a phased array
    fair game. What is not allowed is a surface radiating into free space --
    a pulsating sphere, for instance, where this overestimates the strength
    by exactly that factor of two in the quasi-static limit and by more as
    ``k a`` grows.

    Nothing in the arguments carries geometry, so the function cannot tell
    whether the caller's surface is baffled. Misuse returns plausible
    numbers of the right order and warns about nothing.

    Parameters
    ----------
    v0 : ndarray, shape (N,) or scalar
        Normal velocity of each surface element, in m/s, complex. A uniform
        piston passes one value; a phased array carries the per-element
        phase here. Unlike in `solve_strength` this is not a boundary
        condition imposed at a collocation point: it is the velocity the
        element is declared to have.
    cell_area : ndarray, shape (N,) or scalar
        Surface area each element stands for, in m^2. Broadcasts against
        ``v0``.
    kf : float
        Wavenumber of the fluid, ``2 * pi / wavelength``, in rad/m.
    c : float
        Speed of sound in the fluid, in m/s. Enters only through
        ``omega = kf * c``, so the two are not independent; values that
        disagree are not detected.
    rho : float
        Density of the fluid, in kg/m^3.

    Returns
    -------
    ndarray, shape (N,)
        Complex128. Source strengths in Pa*m, one per element, in the same
        order as ``cell_area``. Not comparable with those of Placko and
        Kundu, who absorb the ``1 / (4 * pi)`` of the Green's function into
        their strengths; fields computed from either are.

    Notes
    -----
    Shapes broadcast, which means both arguments scalar returns a scalar,
    not a length-one array, and `field.pressure` then rejects it: matmul
    will not take a zero-dimensional operand. The uniform piston on a
    uniform mesh is exactly the case where that happens, so one of the two
    arguments has to arrive as an array of length ``N``.

    This branch is not the crude one. On the flat piston it converges to
    about 0.75 percent with a lambda/40 mesh, against 2 to 3 percent for
    the solved branch, and the two agree to about 6 percent in mean
    strength. What it leaves out is the interaction between sources, which
    is a different thing from being inaccurate: the solved branch pays for
    that interaction with a linear system and, on this geometry, does not
    get its money back. Where the two disagree the useful reading is
    diagnostic -- if the solved curve leaves both this one and the closed
    form, the fault is in the system; if both numerical curves leave the
    closed form together, it is in the kernel or in the mesh.

    The strengths scale with the cell area: a larger element moving at the
    same velocity displaces more fluid. They scale with ``rho`` and with
    ``omega`` and with nothing else in the medium, and the whole of the
    physics sits in the sign and in the factor of two. A dropped two is an
    error of exactly a factor of two in amplitude, which no test on the
    shape of a curve will catch; the ratio against the solved branch will.

    References
    ----------
    Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
    Wiley (2007), Eqs. (1.14a), (1.16), (1.16a) and section 1.3.3.
    """
    return -2j * kf * c * rho * v0 * cell_area
