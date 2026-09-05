import numpy as np

__all__ = ["solve_strength"]


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
