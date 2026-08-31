"""Point sources standing in for a radiating surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Source"]


@dataclass(frozen=True, eq=False)
class Source:
    """A layer of point sources standing in for a radiating surface.

    Takes an already discretised surface -- points, normals and the area each
    point represents -- and places one point source behind each point, a short
    distance ``rs`` along the local inward normal. The offset keeps sources
    away from the points where the boundary condition is imposed and from any
    point where the field is later evaluated. Because it follows the normal at
    each point individually, flat and curved surfaces are handled the same way.

    The class holds geometry and computes nothing else. It does not build the
    mesh and does not know what shape the surface has: it takes three arrays,
    so ``Source(mesh.points, mesh.normals, mesh.cell_area)`` works without this
    module importing the mesh module. It does not move the surface either; a
    surface is placed and oriented before it becomes a source layer. Source
    strengths are unknown at this stage; `Solver` finds them by imposing the
    surface velocity at ``points``, using ``positions`` as the origin of the
    field. SI units throughout.

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
    cell_area : array_like, shape (N,) or scalar
        Surface area represented by each point. A scalar means every element
        has the same area and broadcasts against the other arrays.
    alpha : float, optional
        Sets how far behind the surface the sources go, as a fraction of the
        element size: ``rs = alpha * sqrt(cell_area)``. Default 0.25.

    Attributes
    ----------
    rs : ndarray
        Retreat distance of each source, shape ``(N,)`` or zero-dimensional,
        following ``cell_area``. Equals the distance from a source to its own
        surface point exactly, whatever the surface shape.
    positions : ndarray, shape (N, 3)
        Where the sources radiate from, ``points - rs * normals``.

    Notes
    -----
    ``alpha`` is a starting value, not a constant. Raising it degrades both
    field accuracy and the conditioning of the linear system; lowering it too
    far makes individual sources unstable. Values around 0.2-0.3 avoid both,
    but each new geometry is worth checking rather than assuming.

    ``alpha=0`` places the sources on the surface itself. The geometry is
    then usable for evaluating a field from known strengths, but not for
    solving for them: sources and surface points coincide and the system is
    singular.

    Inputs are not validated. A ``cell_area`` of the wrong length or a normal
    that is not unit length will propagate silently.

    References
    ----------
    Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
    Wiley (2007).
    """

    points: np.ndarray
    normals: np.ndarray
    cell_area: np.ndarray
    alpha: float = 0.25

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=np.float64)
        normals = np.array(self.normals, dtype=np.float64)
        cell_area = np.array(self.cell_area, dtype=np.float64)

        # asarray, because alpha * sqrt(0-d array) comes back as a NumPy
        # scalar rather than an array, and scalars carry no flags.
        rs = np.asarray(self.alpha * np.sqrt(cell_area))
        positions = points - rs[..., np.newaxis] * normals

        for array in (points, normals, cell_area, rs, positions):
            array.flags.writeable = False

        # object.__setattr__ is `self.x = y` written the way a frozen
        # dataclass allows: frozen blocks plain assignment even in here.
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "cell_area", cell_area)
        object.__setattr__(self, "rs", rs)
        object.__setattr__(self, "positions", positions)
