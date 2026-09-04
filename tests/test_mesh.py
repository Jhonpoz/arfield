"""Unit tests for arfield.mesh.

Every test here is fast and runs on each push. Nothing in this file needs a
wavenumber, a medium or a solver: the module under test is pure geometry.
"""

import numpy as np
import pytest

from arfield import mesh

SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
TRIANGLE = np.array([[10.0, 10.0], [12.0, 10.0], [11.0, 11.5]])
L_SHAPE = np.array(
    [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]
)


def nearest_neighbour_distance(points: np.ndarray) -> float:
    """Smallest distance between two distinct points of a small cloud."""
    delta = points[:, None, :] - points[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


# --------------------------------------------------------------------------
# lattice invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pitch", [0.1, 0.037])
def test_hex_lattice_spacing_is_exact(pitch):
    x1, x2 = mesh.hex_lattice(0.5, 0.5, pitch)
    points = np.stack((x1, x2), axis=1)
    assert nearest_neighbour_distance(points) == pytest.approx(pitch)


def test_rect_lattice_spacing_is_exact():
    du, dv = 0.1, 0.25
    x1, x2 = mesh.rect_lattice(1.0, 1.0, du, dv)
    assert np.diff(np.unique(x1)) == pytest.approx(du)
    assert np.diff(np.unique(x2)) == pytest.approx(dv)


@pytest.mark.parametrize("builder", [mesh.hex_lattice, None])
@pytest.mark.parametrize(("a", "b"), [(1.0, 1.0), (2.0, 0.5), (0.3, 1.7)])
def test_lattice_covers_requested_box(builder, a, b):
    """No point of the box is further than one cell from a node.

    The box is deliberately non-square: with a == b and equal spacings, a
    lattice that is short on one side and long on the other still looks
    plausible, which is the failure mode this test exists for.
    """
    pitch = 0.05
    if builder is None:
        x1, x2 = mesh.rect_lattice(a, b, pitch, pitch)
        tolerance = pitch * np.sqrt(2.0)
    else:
        x1, x2 = builder(a, b, pitch)
        tolerance = pitch

    probe_1, probe_2 = np.meshgrid(
        np.linspace(-a, a, 31), np.linspace(-b, b, 37), indexing="ij"
    )
    probe = np.stack((probe_1.ravel(), probe_2.ravel()), axis=1)
    nodes = np.stack((x1, x2), axis=1)
    gap = np.linalg.norm(probe[:, None, :] - nodes[None, :, :], axis=-1).min(axis=1)
    assert gap.max() <= tolerance


@pytest.mark.parametrize(("a", "b"), [(1.0, 1.0), (2.0, 0.5)])
def test_lattice_stays_inside_requested_box(a, b):
    for x1, x2 in (mesh.hex_lattice(a, b, 0.07), mesh.rect_lattice(a, b, 0.07, 0.03)):
        assert np.abs(x1).max() <= a
        assert np.abs(x2).max() <= b


def test_cell_areas():
    assert mesh.hex_cell_area(0.2) == pytest.approx(np.sqrt(3.0) / 2.0 * 0.04)
    assert mesh.rect_cell_area(0.2, 0.5) == pytest.approx(0.1)


# --------------------------------------------------------------------------
# output contract
# --------------------------------------------------------------------------


def all_shapes():
    return {
        "circle_hex": mesh.circle(1.0, 0.1).points,
        "circle_rect": mesh.circle(1.0, 0.1, lattice="rect").points,
        "rectangle": mesh.rectangle(1.0, 2.0, 0.1, 0.2).points,
        "polygon": mesh.polygon(TRIANGLE, 0.05).points,
        "line": mesh.line([0.0, 0.0, 0.0], [0.0, 0.0, 0.1], 0.005),
        "arc": mesh.arc([0.0, 0.0, 0.0], 0.1, 0.0, np.pi, 1e-3),
    }


@pytest.mark.parametrize("name", list(all_shapes()))
def test_shapes_return_k3_float64(name):
    points = all_shapes()[name]
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert points.shape[0] > 0
    assert points.dtype == np.float64
    assert points.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("name", list(all_shapes()))
def test_no_duplicate_points(name):
    points = all_shapes()[name]
    assert np.unique(points.round(12), axis=0).shape[0] == points.shape[0]


def test_invalid_lattice_raises():
    with pytest.raises(ValueError, match="unknown lattice"):
        mesh.circle(1.0, 0.1, lattice="hexagonal")
    with pytest.raises(ValueError, match="unknown lattice"):
        mesh.polygon(SQUARE, 0.1, lattice="")


def test_two_component_center_raises():
    """A centre with a missing component is refused, never promoted."""
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        mesh.circle(1.0, 0.1, center=[0.0, 0.0])


def test_negative_spacing_raises():
    """Negative spacing is the silent one: max() turns it into two points."""
    with pytest.raises(ValueError, match="spacing"):
        mesh.line([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -1.0)


def test_zero_spacing_raises_on_its_own():
    with pytest.raises(ZeroDivisionError):
        mesh.line([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0)


@pytest.mark.parametrize("stub", [mesh.cylinder, mesh.sphere])
def test_deferred_surfaces_raise(stub):
    with pytest.raises(NotImplementedError):
        stub(1.0, 1.0, 1.0)


# --------------------------------------------------------------------------
# shape geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lattice", ["hex", "rect"])
def test_circle_count_matches_area(lattice):
    """Node count times cell area recovers the disc area as the mesh refines."""
    radius, pitch = 1.0, 0.01
    cell = (
        mesh.hex_cell_area(pitch)
        if lattice == "hex"
        else mesh.rect_cell_area(pitch, pitch)
    )
    count = len(mesh.circle(radius, pitch, lattice=lattice))
    assert count * cell == pytest.approx(np.pi * radius**2, rel=0.02)


def test_circle_points_are_inside():
    radius = 4.95e-3
    points = mesh.circle(radius, radius / 12.0).points
    assert np.hypot(points[:, 0], points[:, 1]).max() <= radius


@pytest.mark.parametrize(
    ("width", "height", "du", "dv"),
    [(1.0, 1.0, 0.1, 0.1), (2.0, 0.5, 0.05, 0.05), (1.0, 1.0, 0.1, 0.25)],
)
def test_rectangle_spans_requested_size(width, height, du, dv):
    """The rectangle delivered is the rectangle requested, and it is centred.

    The failure this catches is losing one edge row to an undefined
    boundary rule, which shifts the mesh half a cell off centre.
    """
    points = mesh.rectangle(width, height, du, dv).points
    columns = np.unique(points[:, 0].round(9))
    rows = np.unique(points[:, 1].round(9))

    assert columns.size == int(round(width / du)) + 1
    assert rows.size == int(round(height / dv)) + 1
    assert columns.min() == pytest.approx(-0.5 * width)
    assert columns.max() == pytest.approx(0.5 * width)
    assert rows.min() == pytest.approx(-0.5 * height)
    assert rows.max() == pytest.approx(0.5 * height)


@pytest.mark.parametrize("lattice", ["hex", "rect"])
def test_polygon_keeps_vertex_coordinates(lattice):
    """The outline is where the vertices say it is, not at the origin."""
    points = mesh.polygon(TRIANGLE, 0.05, lattice=lattice).points
    low = TRIANGLE.min(axis=0)
    high = TRIANGLE.max(axis=0)
    assert np.all(points[:, :2].min(axis=0) >= low)
    assert np.all(points[:, :2].max(axis=0) <= high)
    assert points[:, :2].mean(axis=0) == pytest.approx(TRIANGLE.mean(axis=0), abs=0.1)


def test_polygon_square_matches_rectangle():
    """A square outline and the rectangle constructor agree node for node."""
    from_polygon = mesh.polygon(SQUARE, 0.1, lattice="rect").points
    from_rectangle = mesh.rectangle(2.0, 2.0, 0.1, 0.1).points
    assert from_polygon.shape == from_rectangle.shape
    assert np.allclose(
        np.unique(from_polygon.round(9), axis=0),
        np.unique(from_rectangle.round(9), axis=0),
    )


def test_polygon_handles_a_concave_outline():
    """Even-odd crossing must exclude the notch of an L shape."""
    points = mesh.polygon(L_SHAPE, 0.05, lattice="rect").points
    in_notch = (points[:, 0] > 1.0) & (points[:, 1] > 1.0)
    assert not in_notch.any()
    # Border cells are kept whole, so the node count over-estimates the area
    # by roughly half a cell per unit of perimeter. The L shape has an area of
    # 3 and a perimeter of 8, which makes that bias visible rather than
    # negligible: it is the boundary-cell debt of ARQUITECTURA.md section 7,
    # not a meshing error.
    pitch, area, perimeter = 0.05, 3.0, 8.0
    estimate = points.shape[0] * mesh.rect_cell_area(pitch, pitch)
    assert area <= estimate <= area + perimeter * pitch


@pytest.mark.parametrize(
    "points",
    [
        mesh.circle(1.0, 0.05).points,
        mesh.circle(1.0, 0.05, lattice="rect").points,
        mesh.rectangle(1.0, 1.0, 0.1, 0.1).points,
        mesh.polygon(SQUARE, 0.1).points,
    ],
)
def test_shapes_are_centrally_symmetric(points):
    """Every point has its antipode, so no side of the mesh was truncated.

    This does not catch a sheared lattice on its own, which is why the box
    coverage test above exists as well.
    """
    forward = {tuple(row) for row in points.round(9)}
    assert all(tuple(row) in forward for row in (-points).round(9))


@pytest.mark.parametrize("center", [None, [0.1, -0.2, 0.3]])
def test_center_translates_the_whole_shape(center):
    offset = np.zeros(3) if center is None else np.asarray(center)
    points = mesh.circle(1e-3, 1e-4, center=center).points
    assert points.mean(axis=0) == pytest.approx(offset, abs=1e-9)
    assert points[:, 2] == pytest.approx(offset[2])


# --------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("length", "spacing"), [(1.0, 0.01), (1.5, 0.01), (0.9, 0.01), (1.0, 0.007)]
)
def test_line_endpoints_and_step(length, spacing):
    """Endpoints are exact and no step exceeds the requested spacing.

    The lengths are chosen not to be integer multiples of the spacing: with a
    round ratio, an off-by-one in the interval count gives an almost correct
    answer and hides.
    """
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.0, 0.0, length])
    points = mesh.line(start, end, spacing)

    assert points[0] == pytest.approx(start)
    assert points[-1] == pytest.approx(end)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert steps.max() <= spacing * (1.0 + 1e-12)
    assert points.shape[0] == int(np.ceil(length / spacing)) + 1


def test_line_is_not_restricted_to_one_axis():
    start = np.array([1.0, 2.0, 3.0])
    end = np.array([-1.0, 0.0, 4.0])
    points = mesh.line(start, end, 0.01)
    direction = (end - start) / np.linalg.norm(end - start)
    residual = points - start - np.outer((points - start) @ direction, direction)
    assert np.abs(residual).max() < 1e-12


def test_arc_points_lie_on_the_circle():
    center = np.array([0.0, 0.0, 0.01])
    radius, spacing = 0.1, 1e-3
    points = mesh.arc(center, radius, 0.0, np.pi, spacing)

    assert np.linalg.norm(points - center, axis=1) == pytest.approx(radius)
    assert points[:, 1] == pytest.approx(center[1])
    assert points[0] == pytest.approx(center + [radius, 0.0, 0.0])
    assert points[-1] == pytest.approx(center + [-radius, 0.0, 0.0])
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert steps.max() <= spacing


def test_arc_sweeps_the_other_way_when_limits_are_reversed():
    forward = mesh.arc([0.0, 0.0, 0.0], 1.0, 0.0, np.pi / 2, 0.01)
    backward = mesh.arc([0.0, 0.0, 0.0], 1.0, np.pi / 2, 0.0, 0.01)
    assert forward == pytest.approx(backward[::-1])


# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("lattice", "expected"), [("hex", 499), ("rect", 421)])
def test_known_point_counts(lattice, expected):
    """Frozen node counts for the transducer of the milestone.

    Radius 4.95 mm, pitch lambda / 20 at 40 kHz in air. Nothing physical
    depends on the exact number; the test exists so that a silent change in
    the meshing shows up here rather than as a shifted pressure curve.
    """
    radius = 4.95e-3
    pitch = 8.575e-3 / 20.0
    assert len(mesh.circle(radius, pitch, lattice=lattice)) == expected


# --------------------------------------------------------------------------
# the Mesh object
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plane", "normal"),
    [("xy", [0, 0, 1]), ("xz", [0, 1, 0]), ("yz", [1, 0, 0])],
)
def test_plane_sets_axes_and_normal(plane, normal):
    """An observation plane is an axis permutation, exact and unambiguous."""
    m = mesh.rectangle(0.02, 0.06, 1e-3, 2e-3, center=[0, 0, 0.03], plane=plane)
    assert m.normal == pytest.approx(normal)
    assert m.points.mean(axis=0) == pytest.approx([0, 0, 0.03], abs=1e-12)
    spread = np.ptp(m.points, axis=0)
    assert spread[np.argmax(normal)] == pytest.approx(0.0)


def test_invalid_plane_raises():
    """No guard: the KeyError names the bad value, which is guard enough."""
    with pytest.raises(KeyError, match="XY"):
        mesh.circle(1.0, 0.1, plane="XY")


def test_grid_shape_reshapes_back_to_a_grid():
    """The node order is a promise, because an observation grid depends on it."""
    m = mesh.rectangle(0.02, 0.01, 1e-3, 2.5e-4, plane="xz")
    n_u, n_v = m.grid_shape
    assert n_u * n_v == len(m)
    grid_u = m.points[:, 0].reshape(n_u, n_v)
    grid_v = m.points[:, 2].reshape(n_u, n_v)
    assert np.diff(grid_u, axis=0) == pytest.approx(1e-3)
    assert np.diff(grid_v, axis=1) == pytest.approx(2.5e-4)


def test_masked_shapes_have_no_grid_shape():
    assert mesh.circle(1.0, 0.1).grid_shape is None
    assert mesh.polygon(SQUARE, 0.1).grid_shape is None


def test_areas_and_ratio():
    m = mesh.circle(1.0, 0.01)
    assert m.exact_area == pytest.approx(np.pi)
    assert m.meshed_area == pytest.approx(len(m) * m.cell_area)
    assert m.area_ratio == pytest.approx(m.meshed_area / m.exact_area)
    assert 1.0 <= m.area_ratio <= 1.05


def test_polygon_exact_area_is_the_shoelace_area():
    """The L shape encloses 3, whatever the mesh does at its border."""
    assert mesh.polygon(L_SHAPE, 0.05).exact_area == pytest.approx(3.0)


def test_normals_is_a_free_read_only_view():
    m = mesh.circle(1.0, 0.1)
    assert m.normals.shape == m.points.shape
    assert not m.normals.flags.writeable
    assert m.normals.base is not None


@pytest.mark.parametrize(
    ("plane", "expected"),
    [("xy", [1, 0, 0]), ("xz", [1, 0, 0]), ("yz", [0, 1, 0])],
)
def test_tangent_is_a_unit_vector_in_the_surface(plane, expected):
    """Perpendicular to the normal in every plane, and it is the first
    in-plane axis. The `xz` case is the one that catches a tangent derived
    from the normal by permutation: that normal is the middle axis and maps
    to itself."""
    m = mesh.circle(5e-3, 1e-3, plane=plane)
    assert m.tangent == pytest.approx(expected)
    assert np.linalg.norm(m.tangent) == pytest.approx(1.0)
    assert float(m.normal @ m.tangent) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    "builder",
    [
        lambda p: mesh.circle(5e-3, 1e-3, plane=p),
        lambda p: mesh.rectangle(4e-3, 3e-3, 1e-3, 1e-3, plane=p),
        lambda p: mesh.polygon(SQUARE, 0.2, plane=p),
    ],
)
@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_every_constructor_sets_a_valid_frame(builder, plane):
    """The frame is built in _embed, so all three shapes must agree."""
    m = builder(plane)
    assert float(m.normal @ m.tangent) == pytest.approx(0.0, abs=1e-15)
    assert np.linalg.norm(m.tangent) == pytest.approx(1.0)


def test_tangents_is_a_free_read_only_view():
    """Same broadcast trick as normals: no memory, and it widens to (K, 3)
    for curved surfaces without any caller noticing."""
    m = mesh.circle(1.0, 0.1)
    assert m.tangents.shape == m.points.shape
    assert not m.tangents.flags.writeable
    assert m.tangents.base is not None
    assert m.tangents.strides[0] == 0


def test_mesh_is_frozen_at_both_levels():
    m = mesh.circle(1.0, 0.1)
    with pytest.raises(Exception):  # noqa: B017  FrozenInstanceError
        m.points = np.zeros((3, 3))
    with pytest.raises(ValueError, match="read-only"):
        m.points[0, 0] = 1.0


def test_rotation_moves_points_normal_and_tangent_together():
    """The failure this catches produces a field, not an exception."""
    m = mesh.circle(4.95e-3, 4e-4)
    matrix = mesh.rotation([1, 1, 0], np.deg2rad(35))
    turned = m.rotated(matrix)

    offset = turned.points - turned.points.mean(axis=0)
    assert np.abs(offset @ turned.normal).max() < 1e-14
    assert np.linalg.norm(turned.normal) == pytest.approx(1.0)
    assert m.normal == pytest.approx([0, 0, 1])

    # The tangent has to travel with them. Forgetting it in `rotated` leaves
    # a mesh whose frame is inconsistent and whose collocation points are
    # displaced in a direction that is no longer in the surface.
    assert turned.tangent == pytest.approx(matrix @ m.tangent)
    assert float(turned.normal @ turned.tangent) == pytest.approx(0.0, abs=1e-14)
    assert np.abs(offset @ turned.tangent).max() > 1e-6


def test_rotation_preserves_the_invariant_fields_and_the_lattice():
    m = mesh.circle(1.0, 0.05)
    turned = m.rotated(mesh.rotation([0, 1, 0], 0.7))
    assert turned.cell_area == m.cell_area
    assert turned.exact_area == m.exact_area
    assert turned.grid_shape == m.grid_shape
    assert len(turned) == len(m)
    before = np.linalg.norm(m.points[1] - m.points[0])
    after = np.linalg.norm(turned.points[1] - turned.points[0])
    assert after == pytest.approx(before)


def test_the_frame_survives_a_chain_of_rotations():
    """Round-off accumulates, so orthogonality is checked with a tolerance
    and never against zero. An exact test passes on a fresh mesh and starts
    rejecting valid ones here."""
    m = mesh.circle(1.0, 0.1)
    matrix = mesh.rotation([1, 1, 1], 0.7)
    for _ in range(200):
        m = m.rotated(matrix)
    assert abs(float(m.normal @ m.tangent)) < 1e-9
    assert np.linalg.norm(m.tangent) == pytest.approx(1.0)


def test_translated_leaves_the_frame_alone():
    """A translation moves points and nothing else."""
    m = mesh.circle(1.0, 0.1)
    shifted = m.translated([0.1, -0.2, 0.3])
    assert np.array_equal(shifted.tangent, m.tangent)
    assert np.array_equal(shifted.normal, m.normal)


def test_non_orthogonal_matrix_raises():
    """A scaling matrix would leave cell_area describing a different cell."""
    with pytest.raises(ValueError, match="orthogonal"):
        mesh.circle(1.0, 0.1).rotated(2.0 * np.eye(3))


def test_translated_moves_only_the_points():
    m = mesh.circle(1.0, 0.1)
    shifted = m.translated([0.1, -0.2, 0.3])
    assert np.allclose(shifted.points - m.points, [0.1, -0.2, 0.3])
    assert shifted.normal == pytest.approx(m.normal)
    assert m.points.mean(axis=0) == pytest.approx([0, 0, 0], abs=1e-12)


def test_rotation_matrix_is_orthogonal_and_right_handed():
    matrix = mesh.rotation([2.0, -1.0, 0.5], 1.234)
    assert matrix @ matrix.T == pytest.approx(np.eye(3))
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_zero_axis_raises():
    with pytest.raises(ValueError, match="zero vector"):
        mesh.rotation([0.0, 0.0, 0.0], 1.0)


def test_mesh_rejects_inconsistent_state():
    points = np.zeros((5, 3))
    unit_z, unit_x = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="unit vector"):
        mesh.Mesh(points, [0.0, 0.0, 2.0], unit_x, 1.0, 1.0)
    with pytest.raises(ValueError, match="unit vector"):
        mesh.Mesh(points, unit_z, [3.0, 0.0, 0.0], 1.0, 1.0)
    with pytest.raises(ValueError, match="perpendicular"):
        mesh.Mesh(points, unit_z, unit_z, 1.0, 1.0)
    with pytest.raises(ValueError, match="shape"):
        mesh.Mesh(points, unit_z, [1.0, 0.0], 1.0, 1.0)
    with pytest.raises(ValueError, match="grid_shape"):
        mesh.Mesh(points, unit_z, unit_x, 1.0, 1.0, (2, 2))
