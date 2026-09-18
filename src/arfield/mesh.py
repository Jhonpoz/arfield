"""Point clouds for surfaces and observation paths.

This module is geometry only: it does not know whether the points it returns
will be used as DPSM sources, as collocation points, or as field observation
points.

Two return types, because there are two kinds of object.

Surfaces
    ``circle``, ``vogel_circle``, ``rectangle``, ``polygon`` and ``sphere``
    return a `Mesh`: the points plus
    the small amount of state that only means something alongside them, namely
    the surface normal, an in-plane tangent, the area one cell stands for,
    the exact area of the outline, and the grid shape when the points form one.

Paths
    ``line`` and ``arc`` return a plain ``(K, 3)`` array. A path has no cells,
    no area and no normal; a `Mesh` with four empty fields would be the wrong
    shape for it.

Conventions
-----------
Lattice half-extents
    ``hex_lattice`` and ``rect_lattice`` take *half-extents*: ``a`` and ``b``
    are measured from the origin, so the lattice fills ``[-a, a] x [-b, b]``.
    Both return every lattice node inside that box and no node outside it.

Surface plane
    ``plane`` selects which two axes the surface spans: ``'xy'``, ``'xz'`` or
    ``'yz'``. This is an axis permutation, exact and free of trigonometry, and
    it covers observation planes as well as radiating surfaces. For any other
    orientation, build the surface in one of the three and call
    `Mesh.rotated`; a rotation moves the points, the normal and the tangent
    together.

Boundary nodes
    A node lying on the outline of a shape, within a relative tolerance of
    1e-9, is kept. Border cells are therefore whole cells whose true area is
    smaller than the nominal cell area, so `Mesh.meshed_area` overshoots
    `Mesh.exact_area` by roughly half a cell per unit of perimeter.
    `Mesh.area_ratio` reports that bias instead of hiding it.

    ``vogel_circle`` is the exception, and the reason it exists. It places no
    nodes on a lattice and cuts nothing, so it has no border cells to bias:
    it partitions the disc into equal-area annuli and puts one node in each,
    and `Mesh.area_ratio` is 1 by construction. What that buys and what it
    costs is in its own docstring, and ``sphere`` inherits both, being the
    same construction mapped onto the sphere by Archimedes' hat-box theorem.

Flat and curved
    Every shape here but ``sphere`` is flat, and stores one ``normal`` and one
    ``tangent`` for the whole mesh. ``sphere`` stores a frame per node. Read
    `Mesh.normals` and `Mesh.tangents`, never the singular fields, and the two
    cases are the same object to a consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "Mesh",
    "arc",
    "cap_area",
    "circle",
    "cone",
    "cylinder",
    "fibonacci_cell_area",
    "fibonacci_count",
    "fibonacci_lattice",
    "hex_cell_area",
    "hex_lattice",
    "line",
    "polygon",
    "rect_cell_area",
    "rect_lattice",
    "rectangle",
    "rotation",
    "sphere",
    "vogel_cell_area",
    "vogel_circle",
    "vogel_count",
    "vogel_lattice",
]

# Relative tolerance used to decide whether a node sits on a boundary.
_RTOL = 1e-9

# Golden angle, pi * (3 - sqrt(5)) radians, about 137.507 degrees. Used by
# `vogel_lattice`.
#
# The angular step has to be an irrational fraction of a turn or the spiral
# closes on itself: were it 2 * pi * p / q, every q-th node would land on the
# same ray and the disc would fill as q spokes with gaps between them. The
# golden angle is the worst-approximable irrational -- its continued fraction
# is all ones -- so it is the one that takes longest to line up at any order.
_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))

# Absolute tolerance on the cosine between normal and tangent. Same scale as
# the orthogonality check in `Mesh.rotated`, and for the same reason: it has
# to absorb round-off from a chain of rotations without letting a genuinely
# skewed frame through.
_ATOL_ORTHOGONAL = 1e-9

# In-plane axes and outward normal for each supported plane.
_PLANES = {
    "xy": ((0, 1), (0.0, 0.0, 1.0)),
    "xz": ((0, 2), (0.0, 1.0, 0.0)),
    "yz": ((1, 2), (1.0, 0.0, 0.0)),
}


# --------------------------------------------------------------------------
# the mesh object
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Mesh:
    """A discretised surface: points plus the state that belongs with them.

    Frozen on purpose. Under a rotation ``points``, ``normal`` and ``tangent``
    all change, and they are only meaningful together: a normal that no longer
    matches its points, or a tangent that no longer matches its normal, is
    wrong in a way that raises nothing. ``cell_area``, ``exact_area`` and
    ``grid_shape`` do not change at all. Freezing the object means the only
    way to move a mesh is `rotated` or `translated`, which update the whole
    frame as a unit; rebinding ``points`` on its own is refused. The arrays
    are additionally marked read-only, so that writing into them in place is
    refused too. Both are one line to remove if they get in the way.

    ``eq=False`` is not decoration: a generated ``__eq__`` on a class holding
    arrays raises on comparison, and ``frozen=True`` would advertise a
    ``__hash__`` that raises as well.

    Attributes
    ----------
    points : ndarray, shape (K, 3)
        Node coordinates, in metres.
    normal : ndarray, shape (3,) or (K, 3)
        Unit normal of the surface. ``(3,)`` when it is the same for every
        node, which is every flat shape here; ``(K, 3)`` for a curved one,
        where each node has its own. Read `normals` rather than this field:
        it gives the per-node form either way, which is what makes the two
        cases interchangeable to a consumer.
    tangent : ndarray, shape (3,) or (K, 3)
        Unit vector in the plane of the surface, perpendicular to `normal`;
        the first in-plane axis of ``plane``. The tangent plane is
        two-dimensional, so this is one arbitrary pick from a circle of
        equally valid ones, and nothing downstream may depend on *which* one.
        It exists for the one-point quadrature that imposes the boundary
        condition as a cell average, where the collocation point sits a
        distance ``xi`` from the cell centre along a tangent; on a hexagonal
        lattice the representative point is a circle, so any tangent serves.

        Stored rather than derived from `normal` on demand, for two reasons:
        every closed-form recipe that reads only the normal degenerates for
        some normal, and a stored vector rotates with the mesh while a derived
        one would be recomputed from the rotated normal and come back
        different. On a sphere the degeneracy is not an accident of the
        recipe but a theorem -- no continuous non-vanishing tangent field
        exists on one -- so a stored field is the only honest option.
    cell_area : float
        Area one node stands for, in square metres. A scalar because every
        lattice here is uniform. Consumers may multiply by it but must not
        sum or index it, so that widening it to ``(K,)`` later stays a
        non-breaking change.
    exact_area : float
        Area of the outline in closed form, in square metres.
    grid_shape : tuple[int, int] or None
        Shape to reshape a per-node quantity back into a grid, or ``None``
        when the nodes do not form one. Only rectangles have it: a mask
        destroys the rectangular structure. ``points.reshape(*grid_shape, 3)``
        varies the first in-plane axis along axis 0.
    """

    points: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    cell_area: float
    exact_area: float
    grid_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        # object.__setattr__ is the documented escape hatch for a frozen
        # dataclass during construction; plain assignment would raise.
        points = np.ascontiguousarray(self.points, dtype=np.float64)
        normal = np.asarray(self.normal, dtype=np.float64)
        tangent = np.asarray(self.tangent, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (K, 3), got {points.shape}")
        for name, vector in (("normal", normal), ("tangent", tangent)):
            # (3,) for a flat surface, (K, 3) for a curved one. Both are
            # checked the same way along the last axis, so a curved mesh gets
            # every node validated rather than none.
            if vector.shape not in ((3,), points.shape):
                raise ValueError(
                    f"{name} must have shape (3,) or {points.shape}, got {vector.shape}"
                )
            norm = np.linalg.norm(vector, axis=-1)
            if not np.allclose(norm, 1.0):
                worst = float(norm.flat[np.argmax(np.abs(norm - 1.0))])
                raise ValueError(f"{name} must be a unit vector, got norm {worst!r}")
        # Absolute, and never == 0.0: both vectors are unit, so the dot
        # product is a direction cosine and 1e-9 is a hard limit on the angle.
        # An exact test would pass today, when every tangent is an axis
        # vector, and start rejecting valid meshes the first time `rotated`
        # leaves a residue of 1e-16.
        cosine = np.sum(normal * tangent, axis=-1)
        if np.abs(cosine).max() > _ATOL_ORTHOGONAL:
            worst = float(np.abs(cosine).max())
            raise ValueError(
                f"normal and tangent must be perpendicular, got cosine {worst!r}"
            )
        if not float(self.cell_area) > 0.0:
            raise ValueError(f"cell_area must be positive, got {self.cell_area!r}")
        if not float(self.exact_area) > 0.0:
            raise ValueError(f"exact_area must be positive, got {self.exact_area!r}")
        if self.grid_shape is not None:
            n_1, n_2 = self.grid_shape
            if n_1 * n_2 != points.shape[0]:
                raise ValueError(
                    f"grid_shape {self.grid_shape} does not match "
                    f"{points.shape[0]} points"
                )

        points.flags.writeable = False
        normal.flags.writeable = False
        tangent.flags.writeable = False
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "tangent", tangent)
        object.__setattr__(self, "cell_area", float(self.cell_area))
        object.__setattr__(self, "exact_area", float(self.exact_area))

    def __len__(self) -> int:
        return self.points.shape[0]

    def __repr__(self) -> str:
        # A curved mesh holds K normals; printing them fills the screen.
        frame = (
            f"normal={self.normal.tolist()}, tangent={self.tangent.tolist()}"
            if self.normal.ndim == 1
            else "frame=per-node"
        )
        return (
            f"Mesh(K={len(self)}, {frame}, "
            f"cell_area={self.cell_area:.4e}, area_ratio={self.area_ratio:.4f}, "
            f"grid_shape={self.grid_shape})"
        )

    @property
    def normals(self) -> np.ndarray:
        """Per-node view of `normal`, shape ``(K, 3)``.

        A broadcast view of `normal` when that is a single vector: no memory
        and read-only. When `normal` is already per-node, the stored array
        itself, also read-only but not a view. This is the form `Source`
        expects, so ``Source(m.points, m.normals, m.tangents, m.cell_area)``
        works without either module importing the other, and works the same
        for a flat surface and a curved one.
        """
        if self.normal.ndim == 1:
            return np.broadcast_to(self.normal, self.points.shape)
        return self.normal

    @property
    def tangents(self) -> np.ndarray:
        """Per-node view of `tangent`, shape ``(K, 3)``.

        A broadcast view when `tangent` is a single vector, the stored array
        when it is already per-node, exactly as `normals` is, and passed to
        `Source` alongside it. That the widening to ``(K, 3)`` left every
        caller of this property untouched is the whole point of consumers
        reading the property and not the field.
        """
        if self.tangent.ndim == 1:
            return np.broadcast_to(self.tangent, self.points.shape)
        return self.tangent

    @property
    def meshed_area(self) -> float:
        """Total area the mesh accounts for, ``K * cell_area``.

        This is the area the DPSM assembly actually uses, since every source
        carries the same nominal cell area.
        """
        return len(self) * self.cell_area

    @property
    def area_ratio(self) -> float:
        """`meshed_area` over `exact_area`.

        One when the discretisation loses nothing. Above one by the border
        cells kept whole; the excess grows with perimeter over area, so a
        long thin outline shows a larger bias than a disc at the same pitch.
        """
        return self.meshed_area / self.exact_area

    def rotated(self, matrix: ArrayLike) -> Mesh:
        """Return a new mesh rotated by a ``(3, 3)`` matrix.

        Points, normal and tangent are rotated by the same matrix, which is
        what keeps them a consistent frame. ``cell_area``, ``exact_area`` and
        ``grid_shape`` are invariant under rotation, so they are carried over
        untouched.

        The rotation is about the origin. To rotate about another point,
        translate to it, rotate, translate back.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        # Orthogonality also rules out the wrong shape. A scaling matrix is
        # the silent case: the nodes move apart and cell_area goes on
        # describing the cell they used to span.
        if matrix.shape != (3, 3) or not np.allclose(
            matrix @ matrix.T, np.eye(3), atol=1e-9
        ):
            raise ValueError("matrix must be orthogonal, or areas stop being areas")
        return replace(
            self,
            points=self.points @ matrix.T,
            # `v @ matrix.T` is `matrix @ v` for a single vector and the
            # row-wise rotation for a stack of them, so one spelling covers
            # a flat frame and a per-node one.
            normal=self.normal @ matrix.T,
            tangent=self.tangent @ matrix.T,
        )

    def translated(self, offset: ArrayLike) -> Mesh:
        """Return a new mesh shifted by a ``(3,)`` offset. Nothing else moves."""
        return replace(self, points=self.points + np.asarray(offset, float))


def rotation(axis: ArrayLike, angle: float) -> np.ndarray:
    """Rotation matrix about an arbitrary axis through the origin.

    Rodrigues' formula. ``axis`` need not be normalised; ``angle`` is in
    radians, positive by the right-hand rule.
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    # A zero axis divides into NaN, which travels far before anyone sees it.
    if norm == 0.0:
        raise ValueError("axis must not be the zero vector")
    k_1, k_2, k_3 = axis / norm

    cross = np.array([[0.0, -k_3, k_2], [k_3, 0.0, -k_1], [-k_2, k_1, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _as_center(center: ArrayLike | None) -> np.ndarray:
    """Return ``center`` as a ``(3,)`` float64 array; ``None`` means origin.

    A two-component centre is rejected rather than promoted, so that a caller
    who means a point in space never has a component silently dropped.
    """
    if center is None:
        return np.zeros(3, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"center must have shape (3,), got {center.shape}")
    return center


def _embed(
    u: np.ndarray, v: np.ndarray, plane: str, center: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lift in-plane coordinates into 3-D.

    Returns the points, the outward normal and an in-plane tangent.

    ``u`` and ``v`` are the two in-plane coordinates in the order the plane
    name gives them, so for ``'xz'`` they are the ``x1`` and ``x3``
    components. This is a permutation of axes, not a rotation: exact, with no
    trigonometry and no orientation left undefined.

    The tangent is the basis vector of the first in-plane axis. It is built
    from the index rather than derived from the normal: a permutation applied
    to the normal has a fixed point (``'xz'``, whose normal is the middle
    axis, maps to itself), and every closed-form recipe that reads only the
    normal degenerates somewhere. Reading the index cannot.
    """
    # A bad name raises KeyError naming it, which is guard enough.
    (axis_u, axis_v), normal = _PLANES[plane]
    tangent = [0.0, 0.0, 0.0]
    tangent[axis_u] = 1.0

    points = np.empty((u.size, 3), dtype=np.float64)
    points[:, axis_u] = u
    points[:, axis_v] = v
    # The third axis holds only the offset; the surface is flat.
    points[:, 3 - axis_u - axis_v] = 0.0
    points += center
    return (
        points,
        np.array(normal, dtype=np.float64),
        np.array(tangent, dtype=np.float64),
    )


# --------------------------------------------------------------------------
# lattice primitives
# --------------------------------------------------------------------------


def hex_lattice(a: float, b: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes of a hexagonal lattice filling the box ``[-a, a] x [-b, b]``.

    Every node has six neighbours at distance ``pitch``. Rows are spaced
    ``sqrt(3) / 2 * pitch`` apart along the second axis and are offset by half
    a pitch from one row to the next.

    Returns two flat float64 arrays of equal length.

    Notes
    -----
    The index box in ``(q, r)`` is generated larger than needed and then
    trimmed to the requested box. The shear ``u = pitch * (q + r / 2)``
    displaces row ``r`` by ``pitch * r / 2``, so a symmetric index box maps to
    a parallelogram: sizing ``q`` from ``a`` alone leaves the upper rows short
    on one side and long on the other.
    """

    row_spacing = np.sqrt(3.0) / 2.0 * pitch
    n_r = int(np.ceil(b / row_spacing)) + 1
    # Worst-case shear is pitch * n_r / 2, which the q range must absorb.
    n_q = int(np.ceil(a / pitch + n_r / 2.0)) + 1

    q, r = np.mgrid[-n_q : n_q + 1, -n_r : n_r + 1]
    u = pitch * (q + 0.5 * r)
    v = row_spacing * r

    atol = _RTOL * max(a, b, pitch)
    inside = (np.abs(u) <= a + atol) & (np.abs(v) <= b + atol)
    # Boolean indexing of a 2-D array already returns a flat array.
    return u[inside], v[inside]


def rect_lattice(
    a: float, b: float, du: float, dv: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes of a rectangular lattice filling the box ``[-a, a] x [-b, b]``.

    Returns two flat float64 arrays of equal length, ordered so that the
    first coordinate varies slowest: ``u.reshape(n_u, n_v)`` is a grid.
    """

    # floor, not ceil: no node may fall outside the requested box. The
    # tolerance keeps a / du = 5.0000000000000001 from flooring to 4.
    n_u = int(np.floor(a / du + _RTOL))
    n_v = int(np.floor(b / dv + _RTOL))

    q, r = np.mgrid[-n_u : n_u + 1, -n_v : n_v + 1]
    return (q * du).ravel(), (r * dv).ravel()


def hex_cell_area(pitch: float) -> float:
    """Area of one hexagonal cell, ``sqrt(3) / 2 * pitch ** 2``."""
    return float(np.sqrt(3.0) / 2.0 * pitch**2)


def rect_cell_area(du: float, dv: float) -> float:
    """Area of one rectangular cell, ``du * dv``."""
    return du * dv


def vogel_count(radius: float, pitch: float) -> int:
    """Node count whose cell area matches a hexagonal lattice of ``pitch``.

    The spiral has no lattice constant, so a spacing has to be converted into
    a count before it means anything. The reference is the hexagonal cell
    area: ``vogel_circle`` and ``circle(lattice='hex')`` of the same ``pitch``
    then carry the same ``cell_area``, hence the same source retreat
    ``rs = alpha * sqrt(cell_area)`` and the same tangential offset. All that
    is left between them is where the nodes sit, which is the comparison the
    spiral exists to make.

    Against ``lattice='rect'`` the comparison is not clean: a square cell of
    the same ``pitch`` has area ``pitch ** 2``, 15 percent larger, so ``rs``
    differs by 7 percent. The hexagonal lattice is the reference.

    Returns at least 1. The rounding is the only reason ``vogel_cell_area``
    is not exactly ``hex_cell_area(pitch)``; the gap is a few parts in a
    thousand on a coarse mesh and less on a fine one.
    """
    if not radius > 0.0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    if not pitch > 0.0:
        raise ValueError(f"pitch must be positive, got {pitch!r}")

    # float() so that a NumPy scalar radius still rounds to a Python int.
    return max(round(float(np.pi * radius**2 / hex_cell_area(pitch))), 1)


def vogel_cell_area(radius: float, pitch: float) -> float:
    """Area assigned to one node of a Vogel spiral, ``pi * radius**2 / n``.

    Exact, and the same for every node: the disc is split into ``n`` annuli
    of equal area and each holds one node. Unlike `hex_cell_area` it needs
    the radius, because the count comes from `vogel_count` and the area
    follows from the count rather than from the spacing.
    """
    return float(np.pi * radius**2 / vogel_count(radius, pitch))


def vogel_lattice(radius: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes of a Vogel spiral filling a disc of radius ``radius``.

    Splits the disc into ``n = vogel_count(radius, pitch)`` concentric annuli
    of *equal area* and puts one node in each, so the area assigned per node
    is exact and the assigned areas sum to the disc area exactly. Annulus
    ``i`` runs from ``radius * sqrt(i / n)`` to ``radius * sqrt((i + 1) / n)``,
    since the area enclosed by a circle grows linearly in ``r ** 2``.

    Unlike `hex_lattice` and `rect_lattice` this does not fill a box and is
    not meant to be masked afterwards. The outline is built in, which is the
    whole point: there are no border cells to cut.

    Returns two flat float64 arrays of length ``n``, ordered by increasing
    radius.

    Parameters
    ----------
    radius : float
        Disc radius, in metres. Not a half-extent: unlike `hex_lattice` and
        `rect_lattice` this fills a disc, not a box, so there is no second
        extent to pair it with.
    pitch : float
        Node spacing, in metres, read through `vogel_count`: it fixes the
        cell area and the node count follows. The nearest-neighbour distance
        that comes out is about 0.89 times this value, not equal to it.

    Notes
    -----
    The ``+ 0.5`` in the radius is not a tweak. It is the radius that halves
    its annulus into two pieces of equal area, which is what makes the node
    the representative of its cell rather than an arbitrary point inside it.
    Using ``+ 0`` would put every node on the inner rim of its annulus and
    bias the whole mesh towards the centre.

    Equal assigned area is not equal *territory*. Measured against the
    Voronoi cells clipped to the disc, the real area of a node's cell runs
    from 0.940 to 1.054 of the nominal value in the interior, and those
    extremes do not shrink with ``n``: they belong to the spiral, not to the
    discretization. At the rim it is worse. The outermost node sits at
    ``radius * sqrt(1 - 1 / (2 * n))``, within one part in ``4 * n`` of the
    edge, so its territory is a sliver of about half the nominal area, and
    the ring just inside carries the surplus, up to 1.28 times nominal. The
    bias that `circle` puts in ``area_ratio``, this one hides in the
    per-cell distribution, where nothing reports it.

    Nearest-neighbour spacing is about ``0.956 * sqrt(cell_area)``, against
    ``1.075`` for a hexagonal lattice of the same cell area: at equal area
    per node the neighbours here sit some 11 percent closer. That matters
    wherever an offset is assumed small against the distance to the nearest
    neighbour, which is the argument `source.tangent_displacement` rests on.

    Cells are not congruent either. In the interior 82 percent are hexagons
    and the rest are 5-7 defect pairs in equal numbers, so the claim that any
    tangent direction is as good as another -- true on a hexagonal lattice,
    where the locus of valid offsets is a circle -- is untested here.

    The angle is accumulated as ``i * _GOLDEN_ANGLE``, which reaches a few
    thousand radians at the counts this is used with. The absolute error
    there is around 1e-12 radians, far below anything geometric.
    """
    n = vogel_count(radius, pitch)

    index = np.arange(n, dtype=np.float64)
    r = radius * np.sqrt((index + 0.5) / n)
    angle = index * _GOLDEN_ANGLE
    return r * np.cos(angle), r * np.sin(angle)


def _lattice_in_box(
    a: float, b: float, pitch: float, lattice: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Dispatch to a lattice primitive by name and report its cell area."""
    if lattice == "hex":
        u, v = hex_lattice(a, b, pitch)
        return u, v, hex_cell_area(pitch)
    if lattice == "rect":
        u, v = rect_lattice(a, b, pitch, pitch)
        return u, v, rect_cell_area(pitch, pitch)
    raise ValueError(f"unknown lattice {lattice!r}, expected 'hex' or 'rect'")


# --------------------------------------------------------------------------
# curved-surface primitives
# --------------------------------------------------------------------------

# Exact rotation by pi about +x: it maps +z to -z, and every entry is 0 or
# +-1, so a southern cap is the mirror of a northern one to the last bit.
_FLIP_Z = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def cap_area(radius: float, half_angle: float) -> float:
    """Area of a spherical cap, ``2 * pi * radius**2 * (1 - cos(half_angle))``.

    ``half_angle = pi`` gives ``4 * pi * radius ** 2``: the closed sphere is
    the cap that shuts. This is the `Mesh.exact_area` of everything `sphere`
    builds.
    """
    if not radius > 0.0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    if not 0.0 < half_angle <= np.pi:
        raise ValueError(f"half_angle must be in (0, pi], got {half_angle!r}")
    return float(2.0 * np.pi * radius**2 * (1.0 - np.cos(half_angle)))


def fibonacci_count(radius: float, pitch: float, half_angle: float = np.pi) -> int:
    """Node count whose cell area matches a hexagonal lattice of ``pitch``.

    The same reference as `vogel_count`, for the same reason: it keeps
    ``rs = alpha * sqrt(cell_area)`` and the tangential offset equal to those
    of a flat mesh of the same ``pitch``, so a curved result can be compared
    against a flat one without the cell size moving underneath.

    Returns at least 1.
    """
    if not pitch > 0.0:
        raise ValueError(f"pitch must be positive, got {pitch!r}")
    return max(round(float(cap_area(radius, half_angle) / hex_cell_area(pitch))), 1)


def fibonacci_cell_area(
    radius: float, pitch: float, half_angle: float = np.pi
) -> float:
    """Area assigned to one node, ``cap_area / n``. Exact, and equal for all."""
    return cap_area(radius, half_angle) / fibonacci_count(radius, pitch, half_angle)


def _align_from_z(axis: ArrayLike) -> np.ndarray | None:
    """Rotation taking ``+z`` to ``axis``, or ``None`` when ``axis`` is ``+z``.

    Exact for the two poles and only for them: ``+z`` needs no rotation at all
    and ``-z`` is a sign flip. Any other axis goes through Rodrigues and
    carries the round-off of a sine and a cosine, which is the price of a free
    direction rather than the enumerated set ``plane`` gives flat shapes.
    """
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,):
        raise ValueError(f"axis must have shape (3,), got {axis.shape}")
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("axis must not be the zero vector")
    axis = axis / norm

    cosine = float(axis[2])
    if cosine >= 1.0:
        return None
    if cosine <= -1.0:
        return _FLIP_Z

    # Rodrigues about z x axis, whose norm is the sine of the angle. The
    # 1 + cosine in the denominator is the half-angle identity, and it is what
    # the branch above keeps away from zero.
    k_1, k_2 = -axis[1], axis[0]
    cross = np.array([[0.0, 0.0, k_2], [0.0, 0.0, -k_1], [-k_2, k_1, 0.0]])
    return np.eye(3) + cross + cross @ cross / (1.0 + cosine)


def fibonacci_lattice(
    radius: float, pitch: float, half_angle: float = np.pi
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes, unit normals and unit tangents on a sphere or spherical cap.

    Returns three ``(n, 3)`` arrays rather than the two in-plane coordinates
    the flat lattices return, and that difference is the point: a curved
    surface carries its frame at every node, and there is no plane to lift
    from. The cap is centred on ``+z``; `sphere` turns it afterwards.
    Ordered from the pole outwards.

    Splits the surface into ``n`` bands of equal ``z`` and puts one node in
    each, the azimuth advancing by `_GOLDEN_ANGLE`. Bands of equal ``z`` on a
    sphere have equal area -- Archimedes' hat-box theorem -- so this is the
    Vogel spiral of `vogel_lattice` mapped onto the sphere, with the same
    consequence: the area per node is exact and the areas sum to the cap area
    exactly. The node sits at the ``z`` that halves its band, which is the
    same ``+ 0.5`` that makes a node the representative of its cell rather
    than an arbitrary point in it.

    Notes
    -----
    This is not a plane projected onto a curved surface, which apodises the
    aperture and is what ARQUITECTURA.md section 4 rules out. It is built on
    the sphere.

    Equal assigned area is not equal territory, exactly as in the plane.
    Measured with geodesic Voronoi cells on the closed sphere, the real area
    of a node's cell runs from 0.940 to 1.054 of nominal, the same figures
    the Vogel disc gives in its interior, because it is the same
    construction. A cap has a rim and gets the rim problem back: the
    outermost node sits half a band inside the edge, so its territory is
    about half the nominal area while ``rs`` still uses the nominal one.

    The tangent is the tangent of the generating spiral, ``d(point)/di``,
    normalised. It is perpendicular to the normal by construction and not by
    correction: the points lie on a sphere, so ``|P|`` is constant and
    ``dP/di . P = 0`` identically; measured, the worst cosine is 3e-16. It
    does not vanish at the poles either, where the azimuthal part shrinks as
    ``sin(theta)`` and the polar part grows to meet it, so no special case is
    needed there. It is not a continuous field -- no such field exists on a
    sphere -- and it does not need to be: it is built per node from the
    spiral index, and consecutive indices are far apart in space.

    Nearest-neighbour spacing is about ``0.96 * sqrt(cell_area)``, against
    ``1.075`` for a hexagonal lattice of the same cell area, as in the plane.
    """
    n = fibonacci_count(radius, pitch, half_angle)

    span = 1.0 - np.cos(half_angle)
    index = np.arange(n, dtype=np.float64)
    z = 1.0 - span * (index + 0.5) / n
    rho = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    angle = index * _GOLDEN_ANGLE
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    normals = np.stack([rho * cos_a, rho * sin_a, z], axis=1)

    # d(rho)/di follows from rho ** 2 + z ** 2 being one. rho vanishes only
    # for a cap so small that it holds a single node, where the guard leaves
    # the polar term to carry the tangent on its own.
    dz = -span / n
    d_rho = np.divide(-z * dz, rho, out=np.zeros_like(rho), where=rho > 0.0)
    derivative = np.stack(
        [
            d_rho * cos_a - rho * _GOLDEN_ANGLE * sin_a,
            d_rho * sin_a + rho * _GOLDEN_ANGLE * cos_a,
            np.full(n, dz),
        ],
        axis=1,
    )
    tangents = derivative / np.linalg.norm(derivative, axis=1, keepdims=True)

    return radius * normals, normals, tangents


# --------------------------------------------------------------------------
# point-in-polygon, without a plotting library
# --------------------------------------------------------------------------


def _distance_to_edges(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Distance from every point to every polygon edge.

    Returns a ``(K, E)`` array for ``K`` points and ``E`` edges. Memory is
    ``O(K * E)``, which is why this is only used for boundary tolerance on
    polygons, where ``E`` is a handful of edges.
    """
    start = vertices[np.newaxis, :, :]
    edge = np.roll(vertices, -1, axis=0)[np.newaxis, :, :] - start
    offset = points[:, np.newaxis, :] - start

    length_sq = np.sum(edge * edge, axis=-1)
    t = np.sum(offset * edge, axis=-1) / np.where(length_sq > 0.0, length_sq, 1.0)
    t = np.clip(t, 0.0, 1.0)

    closest = start + t[:, :, np.newaxis] * edge
    return np.linalg.norm(points[:, np.newaxis, :] - closest, axis=-1)


def _points_in_polygon(
    points: np.ndarray, vertices: np.ndarray, atol: float
) -> np.ndarray:
    """Boolean mask of the points inside or on the outline of a polygon.

    Crossing-number test: a ray is cast along ``+u`` from each point and the
    edges it crosses are counted; an odd count means inside. The half-open
    rule on ``v`` counts each vertex exactly once, so a ray passing through a
    vertex is not double counted. Points on the outline are added back
    afterwards, since the crossing test decides them arbitrarily.

    Works for convex and concave outlines. Self-intersecting outlines follow
    the even-odd rule, which is a definition, not a failure.
    """
    u, v = points[:, 0], points[:, 1]
    u_a, v_a = vertices[:, 0], vertices[:, 1]
    u_b, v_b = np.roll(u_a, -1), np.roll(v_a, -1)

    # An edge is a candidate when v sits between its endpoints, half-open.
    straddles = (v_a[np.newaxis, :] <= v[:, np.newaxis]) != (
        v_b[np.newaxis, :] <= v[:, np.newaxis]
    )

    # u of the edge at height v. The denominator only vanishes on horizontal
    # edges, for which `straddles` is already False, so the guarded value is
    # never read.
    dv = v_b - v_a
    safe_dv = np.where(dv != 0.0, dv, 1.0)
    t = (v[:, np.newaxis] - v_a[np.newaxis, :]) / safe_dv[np.newaxis, :]
    u_cross = u_a[np.newaxis, :] + t * (u_b - u_a)[np.newaxis, :]

    crossings = np.sum(straddles & (u[:, np.newaxis] < u_cross), axis=1)
    inside = crossings % 2 == 1

    on_outline = np.any(_distance_to_edges(points, vertices) <= atol, axis=1)
    return inside | on_outline


def _shoelace_area(vertices: np.ndarray) -> float:
    """Area enclosed by a simple polygon, from the shoelace formula."""
    u, v = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(np.abs(np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v)))


# --------------------------------------------------------------------------
# planar shapes
# --------------------------------------------------------------------------


def circle(
    radius: float,
    pitch: float,
    *,
    center: ArrayLike | None = None,
    plane: str = "xy",
    lattice: str = "hex",
) -> Mesh:
    """Disc of radius ``radius``, centred on ``center``, spanning ``plane``.

    Parameters
    ----------
    radius
        Disc radius, in metres.
    pitch
        Node spacing, in metres.
    center
        ``(3,)`` position of the disc centre. Defaults to the origin.
    plane
        ``'xy'``, ``'xz'`` or ``'yz'``.
    lattice
        ``'hex'`` or ``'rect'``.

    Returns
    -------
    Mesh
        With ``grid_shape`` of ``None``: the circular mask destroys the
        rectangular structure of the lattice.
    """
    center = _as_center(center)

    u, v, cell = _lattice_in_box(radius, radius, pitch, lattice)
    atol = _RTOL * radius
    inside = u**2 + v**2 <= (radius + atol) ** 2

    points, normal, tangent = _embed(u[inside], v[inside], plane, center)
    return Mesh(points, normal, tangent, cell, np.pi * radius**2)


def vogel_circle(
    radius: float,
    pitch: float,
    *,
    center: ArrayLike | None = None,
    plane: str = "xy",
) -> Mesh:
    """Disc sampled on a Vogel spiral, with no area lost at the outline.

    Same shape as `circle` and a different way of covering it. `circle` fills
    a box with a lattice and cuts it with the outline, which keeps border
    cells whole and leaves ``area_ratio`` off by up to a few percent in either
    direction, non-monotonically with ``pitch``. This one builds the outline
    into the sampling: `vogel_lattice` partitions the disc into equal-area
    annuli, so the meshed area is the disc area exactly and ``area_ratio`` is
    1 to machine precision at every ``pitch``.

    Takes no ``lattice``: there is no lattice to choose, and that absence is
    what separates this constructor from `circle`. `circle` keeps ``'hex'``
    and ``'rect'``, which remain the branch this one is measured against.

    Parameters
    ----------
    radius
        Disc radius, in metres.
    pitch
        Node spacing, in metres, read through `vogel_count`. Chosen so that
        this mesh and ``circle(radius, pitch, lattice='hex')`` carry the same
        cell area, up to the rounding of the node count.
    center
        ``(3,)`` position of the disc centre. Defaults to the origin.
    plane
        ``'xy'``, ``'xz'`` or ``'yz'``.

    Returns
    -------
    Mesh
        With ``grid_shape`` of ``None``: a spiral has no grid structure at
        all, not even one destroyed by a mask.

    Notes
    -----
    Not centrally symmetric, and its nodes do not average to ``center``. Both
    hold for every mesh `circle` returns and neither holds here, so tests
    written against the lattice shapes do not carry over unexamined.

    ``area_ratio == 1`` is a property of the constructor, not evidence that
    the mesh is better: it is true by construction and would stay true if the
    nodes were placed wrongly within their annuli. What it does buy is that
    the total radiated strength no longer wanders with ``pitch``, which is
    what made convergence non-monotonic on the cut lattice. What it does not
    buy is a uniform cell *territory*, and the rim nodes are worse off here
    than anywhere in the interior; `vogel_lattice` gives the measured
    distribution.

    The tangent is the one `_embed` supplies, the first in-plane basis
    vector, as for every other planar shape.
    """
    center = _as_center(center)

    u, v = vogel_lattice(radius, pitch)
    points, normal, tangent = _embed(u, v, plane, center)
    return Mesh(
        points, normal, tangent, vogel_cell_area(radius, pitch), np.pi * radius**2
    )


def rectangle(
    width: float,
    height: float,
    du: float,
    dv: float,
    *,
    center: ArrayLike | None = None,
    plane: str = "xy",
) -> Mesh:
    """Rectangle centred on ``center``, spanning ``plane``.

    ``width`` and ``height`` are full side lengths, not half-extents, and
    they run along the first and second axis of ``plane`` respectively: with
    ``plane='xz'``, ``width`` is along ``x1`` and ``height`` along ``x3``.

    An axis-aligned rectangle needs no point-in-polygon test; the lattice
    primitive already fills exactly the requested box.

    Returns
    -------
    Mesh
        With ``grid_shape`` set, so a per-node quantity can be reshaped back
        into a grid. This is the constructor to use for an observation plane.
    """
    center = _as_center(center)

    u, v = rect_lattice(0.5 * width, 0.5 * height, du, dv)
    n_u = 2 * int(np.floor(0.5 * width / du + _RTOL)) + 1
    n_v = 2 * int(np.floor(0.5 * height / dv + _RTOL)) + 1

    points, normal, tangent = _embed(u, v, plane, center)
    return Mesh(
        points, normal, tangent, rect_cell_area(du, dv), width * height, (n_u, n_v)
    )


def polygon(
    vertices: ArrayLike,
    pitch: float,
    *,
    center: ArrayLike | None = None,
    plane: str = "xy",
    lattice: str = "hex",
) -> Mesh:
    """Arbitrary polygon spanning ``plane``.

    The outline is given by its vertices in the two in-plane coordinates, in
    order, without repeating the first one. The returned points keep those
    coordinates; ``center`` translates the whole shape on top of that.

    Returns
    -------
    Mesh
        With ``grid_shape`` of ``None``.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    # A third column would be dropped without a word.
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError(f"vertices must have shape (V, 2), got {vertices.shape}")
    center = _as_center(center)

    # Bounding box of the outline, per axis. Its centre is where the lattice
    # is anchored; the vertex mean would be a different point for any outline
    # that is not centrally symmetric.
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    box_center = 0.5 * (low + high)
    half = 0.5 * (high - low)

    u, v, cell = _lattice_in_box(half[0], half[1], pitch, lattice)
    candidates = np.stack((u + box_center[0], v + box_center[1]), axis=1)

    atol = _RTOL * float(np.max(half))
    inside = _points_in_polygon(candidates, vertices, atol)

    points, normal, tangent = _embed(
        candidates[inside, 0], candidates[inside, 1], plane, center
    )
    return Mesh(points, normal, tangent, cell, _shoelace_area(vertices))


# --------------------------------------------------------------------------
# paths, for observation
# --------------------------------------------------------------------------


def line(start: ArrayLike, end: ArrayLike, spacing: float) -> np.ndarray:
    """Straight segment between two points in space, as ``(K, 3)``.

    Both endpoints are included. The actual step is at most ``spacing``: the
    number of intervals is rounded up, so refining ``spacing`` never skips
    past the endpoint.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    # Scalars would return an (n, 1) array instead of (n, 3), silently.
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError(
            f"start and end must have shape (3,), got {start.shape} and {end.shape}"
        )

    # A negative spacing is absorbed by the max below and silently returns a
    # two-point line; zero raises on its own.
    if spacing < 0.0:
        raise ValueError(f"spacing must be positive, got {spacing!r}")

    length = float(np.linalg.norm(end - start))
    n_intervals = max(int(np.ceil(length / spacing)), 1)
    # n intervals need n + 1 points; linspace counts points, not intervals.
    t = np.linspace(0.0, 1.0, n_intervals + 1)
    return start + t[:, np.newaxis] * (end - start)


def arc(
    center: ArrayLike,
    radius: float,
    angle_start: float,
    angle_end: float,
    spacing: float,
) -> np.ndarray:
    """Circular arc in the ``x1-x3`` plane, as ``(K, 3)``.

    Angles are in radians, measured from ``+x1`` towards ``+x3``, so
    ``angle = pi / 2`` is the acoustic axis of a surface lying in ``x1-x2``.
    Both ends are included and ``spacing`` is an arc-length bound, not an
    angular step.
    """
    center = _as_center(center)

    if spacing < 0.0:
        raise ValueError(f"spacing must be positive, got {spacing!r}")

    arc_length = radius * abs(float(angle_end) - float(angle_start))
    n_intervals = max(int(np.ceil(arc_length / spacing)), 1)
    angle = np.linspace(float(angle_start), float(angle_end), n_intervals + 1)

    x_1 = center[0] + radius * np.cos(angle)
    x_2 = np.full(angle.shape, center[1], dtype=np.float64)
    x_3 = center[2] + radius * np.sin(angle)
    return np.stack((x_1, x_2, x_3), axis=1)


# --------------------------------------------------------------------------
# curved surfaces, deferred
# --------------------------------------------------------------------------

_DEFERRED = (
    "curved surfaces are deferred; a mesh projected from a plane apodises "
    "the aperture and must not be used, see ARQUITECTURA.md section 4"
)


def cylinder(radius: float, height: float, spacing: float) -> Mesh:
    """Not implemented. See ARQUITECTURA.md, section 4."""
    raise NotImplementedError(_DEFERRED)


def cone() -> Mesh:
    """Not implemented. See ARQUITECTURA.md, section 4."""
    raise NotImplementedError(_DEFERRED)


# --------------------------------------------------------------------------
# curved shapes
# --------------------------------------------------------------------------


def sphere(
    radius: float,
    pitch: float,
    half_angle: float = np.pi,
    *,
    center: ArrayLike | None = None,
    axis: ArrayLike | None = None,
    outward: bool = True,
) -> Mesh:
    """Sphere or spherical cap, sampled so that no area is lost at the rim.

    The first curved surface in the module, and the first `Mesh` whose frame
    is per-node: ``normal`` and ``tangent`` come back as ``(K, 3)``. Consumers
    that read `Mesh.normals` and `Mesh.tangents` rather than the fields see no
    difference, which is what those properties are for.

    ``half_angle = pi`` is the closed sphere -- a pulsating sphere -- and
    anything less is a cap. Both are one construction: a cap is a sphere with
    a shorter band of ``z``, and `cap_area` closes to ``4 * pi * radius ** 2``
    at ``pi``.

    Parameters
    ----------
    radius
        Sphere radius, in metres.
    pitch
        Node spacing, in metres, read through `fibonacci_count`. Chosen so
        that this mesh and a flat one of the same ``pitch`` carry the same
        cell area, and therefore the same ``rs`` and the same tangential
        offset.
    half_angle
        Cap half-angle in radians, measured from ``axis``. Default ``pi``.
    center
        ``(3,)`` position of the sphere centre. Defaults to the origin. This
        is the centre of curvature, not a point on the surface.
    axis
        Direction the cap points, needing no normalisation. Default ``+z``.
        ``(0, 0, -1)`` is the southern cap, and is exact: both poles avoid
        Rodrigues entirely. On a closed sphere this only turns the spiral and
        changes nothing observable.
    outward
        Direction of the normals. ``True``, the default, points away from
        `center`: a pulsating sphere, or a cap radiating from its convex
        face. ``False`` points at `center`, which is the concave face of a
        cap -- a focused bowl, and the case a levitator cares about.

    Returns
    -------
    Mesh
        With ``grid_shape`` of ``None`` and ``area_ratio`` of 1 to machine
        precision, at every ``pitch``.

    Notes
    -----
    ``outward`` is the argument that fails silently. `Source` places its point
    sources on the far side of the normal, so reversing it puts them in the
    fluid instead of behind the surface; the solve still solves and a
    plausible field still comes out. Nothing downstream can check it, because
    nothing downstream knows the surface is closed.

    **`solver.rayleigh_strength` does not apply to a pulsating sphere.** Its
    factor of two is the image of a rigid baffle, and a surface radiating into
    free space is the case its own docstring names as overestimated by exactly
    that factor in the quasi-static limit, and by more as ``ka`` grows. The
    closed sphere has a closed form and only one legitimate numerical branch,
    so the three-curve benchmark of the flat piston does not carry over. A
    concave cap is different: O'Neil's argument, section 1.3.3 of the book,
    extends the Rayleigh integral to a gently curved radiator, which is what
    makes the focused cap of Eq. (1.32) fair game.

    The collocation point is offset along a straight tangent, not along the
    surface, so it leaves the sphere by ``xi ** 2 / (2 * radius)`` to second
    order. At the default ``alpha`` on a 10 mm sphere at a quarter wavelength
    that is 0.14 percent of ``rs``, but it grows as the square of the cell and
    falls with the radius, and it has no counterpart on a flat surface.

    A cap keeps the rim problem the disc has: `fibonacci_lattice` gives the
    measured cell distribution.
    """
    points, normals, tangents = fibonacci_lattice(radius, pitch, half_angle)

    rotate = _align_from_z((0.0, 0.0, 1.0) if axis is None else axis)
    if rotate is not None:
        # The frame turns with the points or it stops describing them.
        points = points @ rotate.T
        normals = normals @ rotate.T
        tangents = tangents @ rotate.T

    return Mesh(
        points + _as_center(center),
        normals if outward else -normals,
        tangents,
        fibonacci_cell_area(radius, pitch, half_angle),
        cap_area(radius, half_angle),
    )
