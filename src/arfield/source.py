import numpy as np


class Source:
    """A layer of point sources standing in for a radiating surface.

    Takes an already discretised surface — points, normals and the area each
    point represents — and places one point source behind each point, a short
    distance ``rs`` along the local inward normal. The offset keeps sources
    away from the points where the boundary condition is imposed and from any
    point where the field is later evaluated. Because it follows the normal at
    each point individually, flat and curved surfaces are handled the same way.

    The class holds geometry and computes nothing else. It does not build the
    mesh and does not know what shape the surface has. Source strengths are
    unknown at this stage: `Solver` finds them by imposing the surface velocity
    at ``points``, using ``positions`` as the origin of the field. SI units
    throughout.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Points on the physical surface, where the boundary condition is
        imposed and where the surface areas below belong.
    normals : array_like, shape (N, 3)
        Unit normals at ``points``, pointing away from the surface into the
        fluid. Sources are placed on the opposite side, so a normal pointing
        the wrong way puts its source in the fluid without raising an error.
    cell_area : array_like, shape (N,) or scalar
        Surface area represented by each point. A scalar means every element
        has the same area and broadcasts against the other arrays.
    alpha : float, optional
        Sets how far behind the surface the sources go, as a fraction of the
        element size: ``rs = alpha * sqrt(cell_area)``. Default 0.25.

    Attributes
    ----------
    points, normals, cell_area : ndarray
        Copies of the inputs, marked read-only so that mutating them cannot
        silently invalidate the derived arrays below.
    rs : ndarray, shape (N,) or scalar
        Retreat distance of each source. Equals the distance from a source
        to its own surface point exactly, whatever the surface shape.
    positions : ndarray, shape (N, 3)
        Where the sources radiate from, ``points - rs * normals``.

    Notes
    -----
    ``rs`` and ``positions`` are computed once at construction and mean
    something only alongside the ``points`` that produced them. Reassigning
    an attribute afterwards leaves the object inconsistent; build a new
    instance instead.

    ``alpha`` is a starting value, not a constant. Raising it degrades both
    field accuracy and the conditioning of the linear system; lowering it too
    far makes individual sources unstable. Values around 0.2-0.3 avoid both,
    but each new geometry is worth checking rather than assuming.

    ``alpha=0`` places the sources on the surface itself. The geometry is
    then usable for evaluating a field from known strengths, but not for
    solving for them: sources and surface points coincide and the system is
    singular.

    Inputs are not validated. A ``cell_area`` of the wrong length or a
    normal that is not unit length will propagate silently.

    References
    ----------
    Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
    Wiley (2007).
    """

    def __init__(self, points, normals, cell_area, alpha=0.25):
        self.points = np.array(points, dtype=float)
        self.normals = np.array(normals, dtype=float)
        self.cell_area = np.array(cell_area, dtype=float)

        for arr in (self.points, self.normals, self.cell_area):
            arr.flags.writeable = False

        self.rs = alpha * np.sqrt(self.cell_area)
        self.positions = self.points - self.rs[..., None] * self.normals
