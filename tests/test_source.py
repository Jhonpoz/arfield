"""Unit tests for arfield.source.

Source holds derived state, so most of these tests are about that state
staying true to the surface that produced it.
"""

import numpy as np
import pytest

from arfield import mesh
from arfield.source import Source, tangent_displacement

RADIUS = 4.95e-3
PITCH = 8.575e-3 / 20.0


def disc():
    return mesh.circle(RADIUS, PITCH)


def make_layer(**kwargs):
    m = disc()
    return Source(m.points, m.normals, m.tangents, m.cell_area, **kwargs)


def r_cell(cell_area):
    """Radius of the disc of equal area: r_c in notacion.md, section 2."""
    return np.sqrt(cell_area / np.pi)


def no_third_component(offset, layer, along_normal, along_tangent):
    """True when the offset lies in the plane the normal and tangent span.

    Both derived arrays are built from those two directions alone, so any
    residue is a component nobody put there on purpose.
    """
    residue = (
        offset
        - along_normal[:, None] * layer.normals
        - along_tangent[:, None] * layer.tangents
    )
    return np.max(np.abs(residue)) < 1e-18


# --------------------------------------------------------------------------
# the retreat geometry
# --------------------------------------------------------------------------


def test_source_takes_plain_arrays_from_a_mesh():
    """Source must not depend on mesh; it takes arrays, not a Mesh."""
    m = disc()
    made = Source(m.points, m.normals, m.tangents, m.cell_area)
    assert made.points.shape[0] == len(m)
    assert made.points == pytest.approx(m.points)


@pytest.mark.parametrize("alpha", [0.15, 0.25, 0.35])
def test_normal_offset_is_exact(alpha):
    """The retreat is rs along the normal, and nothing else.

    Since ADR 0007 the retreat is all ``positions`` carries: the tangential
    part of the geometry moved to ``collocation``. The two are checked apart
    because they answer different questions. How far the source sits behind
    the surface is what alpha means; how far the collocation point slides
    across the cell is a quadrature result.
    """
    layer = make_layer(alpha=alpha)
    offset = layer.positions - layer.points
    along_normal = np.sum(offset * layer.normals, axis=1)
    along_tangent = np.sum(offset * layer.tangents, axis=1)

    assert along_normal == pytest.approx(-alpha * np.sqrt(layer.cell_area))
    assert along_tangent == pytest.approx(0.0, abs=1e-18)
    assert no_third_component(offset, layer, along_normal, along_tangent)


@pytest.mark.parametrize("alpha", [0.15, 0.25, 0.35])
def test_collocation_slides_by_xi_and_stays_in_the_surface(alpha):
    """The collocation offset is xi along the tangent, with no normal part.

    A normal component here would lift the boundary condition off the
    surface, which is the one place v_n = v_0 means anything. Nothing raises
    if it does: the solver would impose the velocity of the fluid a fraction
    of a cell away from the piston and return a plausible field.
    """
    layer = make_layer(alpha=alpha)
    offset = layer.collocation - layer.points

    along_normal = np.sum(offset * layer.normals, axis=1)
    along_tangent = np.sum(offset * layer.tangents, axis=1)

    assert along_tangent == pytest.approx(float(layer.xi))
    assert along_normal == pytest.approx(0.0, abs=1e-18)
    assert no_third_component(offset, layer, along_normal, along_tangent)


def test_self_separation_survived_the_split():
    """Source to its own collocation point is sqrt(rs**2 + xi**2), as before.

    ADR 0007 moved xi from the sources to the collocation points, which flips
    the sign of the tangential term in the separation vector. ADR 0004 says a
    reversed tangent is as valid as the original, so the distance must come
    out unchanged -- and with it the diagonal of the influence matrix, and
    with it every Milestone 1 result obtained before the change. If this
    fails, the split was not the re-election of tangent it was argued to be.
    """
    layer = make_layer()
    separation = np.linalg.norm(layer.collocation - layer.positions, axis=1)
    expected = np.hypot(float(layer.rs), float(layer.xi))
    assert separation == pytest.approx(expected)


def test_the_source_cloud_is_not_shifted_across_the_surface():
    """Why xi moves the collocation points and not the sources.

    With a uniform tangent -- every flat surface here -- adding xi to the
    positions translates the whole cloud rigidly in the plane. The aperture
    is where the sources are, so the disc being simulated would sit off to
    one side of the disc that was asked for, and an on-axis curve would not
    be on its axis. The centroid offset must be pure retreat.
    """
    layer = make_layer()
    shift = layer.positions.mean(axis=0) - layer.points.mean(axis=0)
    assert np.dot(shift, layer.normals[0]) == pytest.approx(-float(layer.rs))
    assert np.dot(shift, layer.tangents[0]) == pytest.approx(0.0, abs=1e-18)


def test_points_are_kept_exactly_as_supplied():
    """Nothing downstream reads points; this is what it is kept for.

    It is the only reference the two derived arrays can be measured against,
    and the quadrature offset is the part of this package that most needs an
    independent check. A points that drifted would make both invariants above
    self-consistent and wrong.
    """
    m = disc()
    layer = Source(m.points, m.normals, m.tangents, m.cell_area)
    assert np.array_equal(layer.points, m.points)


def test_sources_sit_behind_the_surface():
    """Behind means opposite the outward normal, for every source."""
    layer = make_layer()
    behind = np.sum((layer.positions - layer.points) * layer.normals, axis=1)
    assert np.all(behind < 0.0)


def test_scalar_and_per_node_cell_area_agree():
    m = disc()
    scalar = Source(m.points, m.normals, m.tangents, m.cell_area)
    per_node = Source(m.points, m.normals, m.tangents, np.full(len(m), m.cell_area))
    assert per_node.rs.shape == (len(m),)
    assert per_node.xi.shape == (len(m),)
    assert per_node.positions == pytest.approx(scalar.positions)
    assert per_node.collocation == pytest.approx(scalar.collocation)


def test_alpha_zero_puts_sources_on_the_surface():
    """Allowed on purpose: usable to evaluate a field, not to solve for one.

    Both offsets vanish -- rs because it is proportional to alpha, xi because
    a source sitting on the surface has nothing to correct for. The division
    by rs warns on the way, which is why the filter is local to this test and
    not global: everywhere else that warning means a bug.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        layer = make_layer(alpha=0.0)
    assert float(layer.rs) == 0.0
    assert float(layer.xi) == 0.0
    assert layer.positions == pytest.approx(layer.points)
    assert layer.collocation == pytest.approx(layer.points)


def test_alpha_is_stored():
    """A layer must be able to say which value produced it."""
    assert make_layer(alpha=0.3).alpha == 0.3


# --------------------------------------------------------------------------
# the one-point quadrature
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(0.1, 0.4410), (0.25, 0.5668), (0.5, 0.6445), (0.8, 0.6768)],
)
def test_offset_matches_the_table_in_hallazgos(alpha, expected):
    """xi / r_c against the closed form, HALLAZGOS_hito1.md section 2.

    The 0.677 at alpha = 0.8 is the value quoted there; the other three pin
    the curve either side of it, so a sign slip or a swapped exponent cannot
    pass by matching one point.
    """
    cell_area = 1.0
    xi = tangent_displacement(alpha * np.sqrt(cell_area), cell_area)
    assert float(xi) / r_cell(cell_area) == pytest.approx(expected, abs=5e-4)


def test_offset_satisfies_the_equation_it_solves():
    """(xi**2 + rs**2)**1.5 == B, against the definition rather than a number.

    This catches an algebra slip in the rearrangement even if the tabulated
    values above were wrong in the same way.
    """
    cell_area, alpha = 3.7e-6, 0.25
    rs = alpha * np.sqrt(cell_area)
    xi = float(tangent_displacement(rs, cell_area))

    r_c = r_cell(cell_area)
    big_r = np.sqrt(r_c**2 + rs**2)
    b = r_c**2 / (2.0 * (1.0 / rs - 1.0 / big_r))
    assert (xi**2 + rs**2) ** 1.5 == pytest.approx(b)


def test_offset_scales_with_the_cell_and_with_nothing_else():
    """xi / r_c depends on alpha alone, over eight orders of magnitude in
    area. A wavelength or a radius leaking into the formula breaks this."""
    alpha = 0.25
    ratios = [
        float(tangent_displacement(alpha * np.sqrt(area), area)) / r_cell(area)
        for area in (1e-8, 1e-6, 1e-4, 1.0)
    ]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_offset_grows_with_alpha_towards_a_ceiling():
    """Monotonic, and it saturates. A deeper source spreads its own term more
    evenly over the cell, so the representative point stops moving out."""
    alphas = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
    ratios = np.array(
        [float(tangent_displacement(a, 1.0)) / r_cell(1.0) for a in alphas]
    )
    assert np.all(np.diff(ratios) > 0.0)
    assert ratios[-1] < 0.71


def test_offset_broadcasts_like_its_inputs():
    """Scalar in, zero-dimensional out; per-node in, per-node out. Source
    relies on this to index it with [..., None] in either case."""
    assert np.ndim(tangent_displacement(0.25, 1.0)) == 0
    per_node = tangent_displacement(np.full(4, 0.25), np.full(4, 1.0))
    assert per_node.shape == (4,)
    assert per_node == pytest.approx(float(tangent_displacement(0.25, 1.0)))


# --------------------------------------------------------------------------
# the invariant
# --------------------------------------------------------------------------


def test_a_sweep_is_a_loop_over_constructions():
    """No sweep helper: the script builds one layer per alpha."""
    m = disc()
    layers = [
        Source(m.points, m.normals, m.tangents, m.cell_area, alpha=a)
        for a in (0.15, 0.35)
    ]
    assert float(layers[1].rs) == pytest.approx(0.35 / 0.15 * float(layers[0].rs))
    assert layers[0].points == pytest.approx(layers[1].points)


def test_rebinding_a_field_is_refused():
    """The failure this catches: sources radiating from the old surface."""
    layer = make_layer()
    with pytest.raises(Exception):  # noqa: B017  FrozenInstanceError
        layer.points = np.zeros_like(layer.points)


def test_construction_does_not_touch_the_caller_arrays():
    """Source copies its inputs; it does not adopt them.

    This is the freeze of ADR 0003 seen from outside. Copying was implicit
    while the arrays went through np.array, and was briefly lost to
    np.ascontiguousarray, which returns its argument unchanged when the
    layout already conforms: constructing a Source then cleared the
    writeable flag on somebody else's array and shared its memory, so a
    caller who set the flag back could write into a frozen object. Passing
    arrays built by hand is the supported case, so it is the case to test.
    """
    m = disc()
    points = np.array(m.points)
    normals = np.array(m.normals)
    tangents = np.array(m.tangents)

    layer = Source(points, normals, tangents, m.cell_area)

    assert points.flags.writeable
    assert normals.flags.writeable
    assert tangents.flags.writeable
    assert not np.shares_memory(layer.points, points)
    assert not np.shares_memory(layer.normals, normals)
    assert not np.shares_memory(layer.tangents, tangents)


@pytest.mark.parametrize(
    "name", ["points", "normals", "tangents", "positions", "collocation"]
)
def test_stored_arrays_are_c_contiguous(name):
    """Uniform layout, so no consumer has to ask which array came in bent.

    Mesh.normals is a stride-zero broadcast view, and np.array defaults to
    order='K', which preserves the layout of whatever it is given -- so
    normals and tangents used to land Fortran-contiguous while their
    neighbours were C. It costs no measurable time either way; what it costs
    is a surprise the day these arrays reach numba or a raw buffer.
    """
    assert getattr(make_layer(), name).flags.c_contiguous


@pytest.mark.parametrize(
    "name",
    [
        "points",
        "normals",
        "tangents",
        "cell_area",
        "rs",
        "xi",
        "positions",
        "collocation",
    ],
)
def test_arrays_are_read_only(name):
    """Derived arrays too: writing into positions desyncs it from points."""
    array = getattr(make_layer(), name)
    assert not array.flags.writeable
