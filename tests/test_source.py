"""Unit tests for arfield.source.

Source holds derived state, so most of these tests are about that state
staying true to the surface that produced it.
"""

import numpy as np
import pytest

from arfield import mesh
from arfield.source import Source

RADIUS = 4.95e-3
PITCH = 8.575e-3 / 20.0


def disc():
    return mesh.circle(RADIUS, PITCH)


def make_layer(**kwargs):
    m = disc()
    return Source(m.points, m.normals, m.cell_area, **kwargs)


# --------------------------------------------------------------------------
# the retreat geometry
# --------------------------------------------------------------------------


def test_source_takes_three_arrays_from_a_mesh():
    """Source must not depend on mesh; it takes arrays, not a Mesh."""
    m = disc()
    made = Source(m.points, m.normals, m.cell_area)
    assert made.points.shape[0] == len(m)
    assert made.points == pytest.approx(m.points)


@pytest.mark.parametrize("alpha", [0.15, 0.25, 0.35])
def test_retreat_distance_is_exact(alpha):
    """rs is the distance from a source to its own surface point, exactly."""
    layer = make_layer(alpha=alpha)
    distance = np.linalg.norm(layer.points - layer.positions, axis=1)
    assert distance == pytest.approx(alpha * np.sqrt(layer.cell_area))


def test_sources_sit_behind_the_surface():
    """Behind means opposite the outward normal, for every source."""
    layer = make_layer()
    behind = np.sum((layer.positions - layer.points) * layer.normals, axis=1)
    assert np.all(behind < 0.0)


def test_scalar_and_per_node_cell_area_agree():
    m = disc()
    scalar = Source(m.points, m.normals, m.cell_area)
    per_node = Source(m.points, m.normals, np.full(len(m), m.cell_area))
    assert per_node.rs.shape == (len(m),)
    assert per_node.positions == pytest.approx(scalar.positions)


def test_alpha_zero_puts_sources_on_the_surface():
    """Allowed on purpose: usable to evaluate a field, not to solve for one."""
    layer = make_layer(alpha=0.0)
    assert layer.positions == pytest.approx(layer.points)


def test_alpha_is_stored():
    """A layer must be able to say which value produced it."""
    assert make_layer(alpha=0.3).alpha == 0.3


# --------------------------------------------------------------------------
# the invariant
# --------------------------------------------------------------------------


def test_a_sweep_is_a_loop_over_constructions():
    """No sweep helper: the script builds one layer per alpha."""
    m = disc()
    layers = [Source(m.points, m.normals, m.cell_area, alpha=a) for a in (0.15, 0.35)]
    assert float(layers[1].rs) == pytest.approx(0.35 / 0.15 * float(layers[0].rs))
    assert layers[0].points == pytest.approx(layers[1].points)


def test_rebinding_a_field_is_refused():
    """The failure this catches: sources radiating from the old surface."""
    layer = make_layer()
    with pytest.raises(Exception):  # noqa: B017  FrozenInstanceError
        layer.points = np.zeros_like(layer.points)


@pytest.mark.parametrize("name", ["points", "normals", "cell_area", "rs", "positions"])
def test_arrays_are_read_only(name):
    """Derived arrays too: writing into positions desyncs it from points."""
    array = getattr(make_layer(), name)
    assert not array.flags.writeable
