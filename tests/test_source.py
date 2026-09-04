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
    """The component along the normal is rs, whatever else the offset does.

    Split from the tangential component on purpose. The straight-line distance
    from a source to its own point is not rs but sqrt(rs**2 + xi**2), and how
    much of it should be tangential is a design question. That the normal part
    is exactly rs is not: it is what alpha means.
    """
    layer = make_layer(alpha=alpha)
    offset = layer.positions - layer.points
    along_normal = np.sum(offset * layer.normals, axis=1)
    assert along_normal == pytest.approx(-alpha * np.sqrt(layer.cell_area))


@pytest.mark.parametrize("alpha", [0.15, 0.25, 0.35])
def test_tangential_offset_is_xi_and_nothing_leaks(alpha):
    """The component along the tangent is xi, and the offset has no third
    component: it lives entirely in the plane the normal and tangent span."""
    layer = make_layer(alpha=alpha)
    offset = layer.positions - layer.points

    along_normal = np.sum(offset * layer.normals, axis=1)
    along_tangent = np.sum(offset * layer.tangents, axis=1)
    assert along_tangent == pytest.approx(float(layer.xi))

    residue = (
        offset
        - along_normal[:, None] * layer.normals
        - along_tangent[:, None] * layer.tangents
    )
    assert np.max(np.abs(residue)) < 1e-18


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


@pytest.mark.parametrize(
    "name",
    ["points", "normals", "tangents", "cell_area", "rs", "xi", "positions"],
)
def test_arrays_are_read_only(name):
    """Derived arrays too: writing into positions desyncs it from points."""
    array = getattr(make_layer(), name)
    assert not array.flags.writeable
