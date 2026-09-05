"""Point sources standing in for a radiating surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["Source", "tangent_displacement"]


@dataclass(frozen=True, eq=False)
class Source:
    """A layer of point sources standing in for a radiating surface.

    Takes an already discretized surface -- points, normals, tangents and the
    area each point represents -- and places one point source behind each
    point, a short distance ``rs`` along the local inward normal. The offset
    keeps sources away from the points where the boundary condition is imposed
    and from any point where the field is later evaluated. Because it follows
    the normal at each point individually, flat and curved surfaces are
    handled the same way.

    Each collocation point is additionally displaced by ``xi`` along the local tangent.
    That distance is the one-point quadrature offset of `tangent_displacement`:
    it makes the boundary condition hold as an average over the cell rather
    than at its center alone, which is what brings the solved branch from a
    34 percent error down to a few percent.

    The class holds geometry and computes nothing else. It does not build the
    mesh and does not know what shape the surface has: it takes four arrays,
    so ``Source(mesh.points, mesh.normals, mesh.tangents, mesh.cell_area)``
    works without this module importing the mesh module. It does not move the
    surface either; a surface is placed and oriented before it becomes a
    source layer. Source strengths are unknown at this stage;
    `solver.solve_strength` finds them by imposing the surface velocity at
    ``collocation``, using ``positions`` as the origin of the field.
    SI units throughout.

    Frozen because ``positions`` is derived. Rebinding ``points`` on a mutable
    version leaves the sources radiating from where the surface used to be,
    with no error anywhere along the way: the solver still solves and a field
    still comes out. The arrays are also marked read-only, so writing into
    them in place is refused too. To change anything, build another instance.

    ``eq=False`` because a generated ``__eq__`` on a class holding arrays
    raises on comparison, and ``frozen=True`` would advertise a ``__hash__``
    that raises as well.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Points on the physical surface, where the boundary condition is
        imposed and where the surface areas below belong.
    normals : array_like, shape (N, 3)
        Unit normals at ``points``, pointing away from the surface into the
        fluid. Sources are placed on the opposite side, so a normal pointing
        the wrong way puts its source in the fluid rather than behind the
        surface.
    tangents : array_like, shape (N, 3)
        Unit vectors at ``points``, in the surface and perpendicular to
        ``normals``, giving the direction of the ``xi`` displacement. Which
        tangent of the circle of valid ones is irrelevant on a hexagonal
        lattice, where the representative point of a cell is a circle; on a
        square lattice it is not, and the wrong direction costs about 11 per
        cent. `Mesh.tangents` supplies these.
    cell_area : array_like, shape (N,) or scalar
        Surface area represented by each point. A scalar means every element
        has the same area and broadcasts against the other arrays.
    alpha : float, optional
        Sets how far behind the surface the sources go, as a fraction of the
        element size: ``rs = alpha * sqrt(cell_area)``. Default 0.25. It also
        fixes ``xi``, which is a function of ``rs`` and the cell area, so the
        two offsets are never set independently.

    Attributes
    ----------
    rs : ndarray
        Retreat distance of each source along the inward normal, shape
        ``(N,)`` or zero-dimensional, following ``cell_area``.
    xi : ndarray
        Tangential offset of each collocation point, shape ``(N,)`` or
        zero-dimensional, following ``cell_area``.
    positions : ndarray, shape (N, 3)
        Where the sources radiate from,
        ``points - rs * normals``.
    collocation : ndarray, shape (N, 3)
        Points where the boundary condition is imposed,
        offset tangentially from the surface by ``xi``:
        ``points + xi * tangents``.

    Notes
    -----
    ``alpha`` is a starting value, not a constant. Raising it degrades both
    field accuracy and the conditioning of the linear system; lowering it too
    far makes individual sources unstable. Values around 0.2-0.3 avoid both,
    but each new geometry is worth checking rather than assuming.

    ``alpha=0`` places the sources on the surface itself. The geometry is
    then usable for evaluating a field from known strengths, but not for
    solving for them: sources and surface points coincide and the system is
    singular; ``tangent_displacement`` warns about the division.

    Inputs are not validated. A ``cell_area`` of the wrong length, a normal
    that is not unit length, or a tangent that is not perpendicular to its
    normal will all propagate silently. `Mesh` checks the last two at
    construction; arrays assembled by hand are not checked anywhere.

    References
    ----------
    Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
    Wiley (2007).
    """

    points: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    cell_area: np.ndarray
    alpha: float = 0.25

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=np.float64, order="C")
        normals = np.array(self.normals, dtype=np.float64, order="C")
        tangents = np.array(self.tangents, dtype=np.float64, order="C")
        cell_area = np.asarray(self.cell_area, dtype=np.float64)

        # asarray, because alpha * sqrt(0-d array) comes back as a NumPy
        # scalar, and a scalar has no .flags to clear below.
        rs = np.asarray(self.alpha * np.sqrt(cell_area))
        xi = np.asarray(tangent_displacement(rs, cell_area))

        collocation = points + xi[..., None] * tangents
        positions = points - rs[..., None] * normals

        for array in (
            points,
            normals,
            tangents,
            cell_area,
            rs,
            xi,
            positions,
            collocation,
        ):
            array.flags.writeable = False

        # object.__setattr__ is `self.x = y` written the way a frozen
        # dataclass allows: frozen blocks plain assignment even in here.
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "tangents", tangents)
        object.__setattr__(self, "cell_area", cell_area)
        object.__setattr__(self, "rs", rs)
        object.__setattr__(self, "xi", xi)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "collocation", collocation)


def tangent_displacement(rs: ArrayLike, cell_area: ArrayLike) -> np.ndarray:
    """Offset that makes a one-point rule reproduce the cell-averaged velocity.

    A point source at a retreat distance ``rs`` behind the surface does not
    produce the same normal velocity everywhere on its cell: the velocity
    peaks right above the source and falls off across the cell. Imposing the
    boundary condition at the cell center therefore imposes the peak, not the
    average, and the assembled system asks every source for too little
    strength. Evaluating instead at a point offset by the returned distance,
    tangentially, reproduces the average over the cell exactly, with a single
    evaluation and no quadrature.

    The offset ``xi`` is the root of

    .. math::

        (\\xi^*{}^2 + r_s^2)^{3/2} = B,
        \\qquad B = \\frac{r_c^2}{2\\,(1/r_s - 1/R)}

    with ``r_c = sqrt(cell_area / pi)`` the radius of the disc of equal area
    and ``R = sqrt(r_c**2 + rs**2)`` the distance from the source to the rim
    of that disc. ``B`` does not contain the unknown, so the root is closed
    form: ``xi = sqrt(B ** (2 / 3) - rs ** 2)``.

    Symbols follow ``notacion.md`` section 2, which is the author's own
    notation from the derivation; ``xi`` is written there as ``xi*``.

    Parameters
    ----------
    rs : array_like
        Retreat distance of the source behind the surface, in meters.
        Typically ``alpha * sqrt(cell_area)``.
    cell_area : array_like
        Surface area the point stands for, in square meters. Broadcasts
        against ``rs``.

    Returns
    -------
    ndarray
        The offset, in meters, in the same shape the inputs broadcast to.

    Notes
    -----
    The result scales with the cell, not with the wavelength: it depends on
    ``rs`` and ``cell_area`` only. In units of ``r_c`` it runs from about
    0.44 at ``alpha = 0.1`` to a ceiling near 0.70 for large ``alpha``, and
    is 0.567 at the default ``alpha = 0.25``.

    Any tangential direction gives the same result on a hexagonal lattice,
    where the locus of valid offsets is a circle. On a square lattice the
    direction matters, and choosing it badly costs about 11 percent.

    ``rs = 0`` divides by zero, which raises a NumPy warning and then returns
    0.0: with the source on the surface there is nothing to correct for. The
    warning is worth reading anyway, because that geometry is singular for
    solving.

    References
    ----------
    Derived for this project; see ``HALLAZGOS_hito1.md``, section 2.
    """
    rs = np.asarray(rs, dtype=np.float64)
    cell_area = np.asarray(cell_area, dtype=np.float64)

    r_c = np.sqrt(cell_area / np.pi)
    R = np.sqrt(r_c**2 + rs**2)
    # The right-hand side of the equation above; the unknown is not in it.
    B = r_c**2 / (2.0 * (1.0 / rs - 1.0 / R))

    return np.sqrt(B ** (2.0 / 3.0) - rs**2)
