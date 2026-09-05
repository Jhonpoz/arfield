"""Unit tests for arfield.influence.

The two functions here are the whole of the geometry-to-matrix step, so
these tests are mostly about the two ways that step can go wrong without
raising: an axis in the wrong order, and a normal taken from the wrong end
of the pair.
"""

import numpy as np
import pytest

from arfield import mesh
from arfield.influence import compute_euler_gradn_green_TS, compute_green_TS
from arfield.source import Source

# Air at 40 kHz, the working point of every milestone in this project.
C = 343.0
RHO = 1.2
KF = 2.0 * np.pi * 40_000.0 / C

RADIUS = 4.95e-3
PITCH = 8.575e-3 / 20.0


def disc():
    return mesh.circle(RADIUS, PITCH)


def cloud(n, seed, spread=0.05, offset=0.0):
    """A reproducible scatter of points, away from the origin."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-spread, spread, size=(n, 3)) + offset


def unit_rows(vectors):
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def green_by_hand(target, source, kf):
    """G(r) written out, with no package call in it."""
    r = np.linalg.norm(target - source)
    return np.exp(1j * kf * r) / (4.0 * np.pi * r)


def euler_gradn_by_hand(target, normal, source, kf, c, rho):
    """n . grad G / (i omega rho), written out, with no package call in it."""
    delta = target - source
    r = np.linalg.norm(delta)
    e_r = delta / r
    grad = green_by_hand(target, source, kf) * (1j * kf - 1.0 / r) * e_r
    return np.dot(normal, grad) / (1j * kf * c * rho)


# --------------------------------------------------------------------------
# shapes and layout
# --------------------------------------------------------------------------


def test_green_has_one_row_per_target_and_one_column_per_source():
    """M and N are deliberately different, so a transposed axis cannot pass.

    With a square matrix every axis mistake in this module returns something
    of the right shape, and the first thing that notices is the solved field
    being wrong by an amount nobody can attribute.
    """
    matrix = compute_green_TS(cloud(7, seed=1), cloud(4, seed=2, offset=0.3), KF)
    assert matrix.shape == (7, 4)
    assert matrix.dtype == np.complex128


def test_euler_has_one_row_per_target_and_one_column_per_source():
    targets = cloud(7, seed=3)
    normals = unit_rows(cloud(7, seed=4, spread=1.0, offset=1.0))
    matrix = compute_euler_gradn_green_TS(
        targets, normals, cloud(4, seed=5, offset=0.3), KF, C, RHO
    )
    assert matrix.shape == (7, 4)
    assert matrix.dtype == np.complex128


def test_targets_need_not_be_the_sources():
    """M != N is the point of taking targets as a free array, ADR 0009.

    field evaluates on a scan line or a volume that has nothing to do with
    the source count, and the solver is the special case where the two happen
    to coincide -- not the other way round.
    """
    matrix = compute_green_TS(cloud(200, seed=6), cloud(9, seed=7, offset=0.4), KF)
    assert matrix.shape == (200, 9)


def test_read_only_inputs_are_accepted():
    """Source freezes its arrays, and they are what these functions receive.

    Nothing in the assembly needs to write into its inputs, but separation
    normalises in place, and an in-place step one array too early would fail
    only when fed a real Source -- never in a test built from fresh arrays.
    """
    m = disc()
    layer = Source(m.points, m.normals, m.tangents, m.cell_area)
    assert not layer.positions.flags.writeable

    compute_green_TS(layer.collocation, layer.positions, KF)
    compute_euler_gradn_green_TS(
        layer.collocation, layer.normals, layer.positions, KF, C, RHO
    )


# --------------------------------------------------------------------------
# the values, against the formulas rather than against the package
# --------------------------------------------------------------------------


def test_green_entry_matches_the_closed_form():
    """Every entry, computed one at a time with no package call involved.

    Written as a double loop on purpose: it is the only test here that cannot
    be satisfied by a consistent misreading of the kernel, because it does
    not use the kernel.
    """
    targets = cloud(5, seed=8)
    sources = cloud(3, seed=9, offset=0.25)
    matrix = compute_green_TS(targets, sources, KF)

    expected = np.array([[green_by_hand(t, s, KF) for s in sources] for t in targets])
    assert matrix == pytest.approx(expected)


def test_euler_entry_matches_the_closed_form():
    targets = cloud(5, seed=10)
    normals = unit_rows(cloud(5, seed=11, spread=1.0, offset=1.0))
    sources = cloud(3, seed=12, offset=0.25)
    matrix = compute_euler_gradn_green_TS(targets, normals, sources, KF, C, RHO)

    expected = np.array(
        [
            [euler_gradn_by_hand(t, n, s, KF, C, RHO) for s in sources]
            for t, n in zip(targets, normals, strict=True)
        ]
    )
    assert matrix == pytest.approx(expected)


def test_amplitude_falls_as_one_over_distance():
    """The 1 / (4 pi r) of the free-space Green's function, isolated."""
    source = np.zeros((1, 3))
    near = np.array([[0.0, 0.0, 0.01]])
    far = np.array([[0.0, 0.0, 0.02]])

    near_value = compute_green_TS(near, source, KF)[0, 0]
    far_value = compute_green_TS(far, source, KF)[0, 0]
    assert abs(far_value) == pytest.approx(0.5 * abs(near_value))


def test_phase_advances_with_distance_in_the_stated_convention():
    """G(r2) / G(r1) == (r1 / r2) * exp(i k (r2 - r1)), ADR 0001.

    Written as a ratio because it needs no unwrapping and because it pins the
    sign: under exp(+i omega t) the exponent is negated, every phase in the
    package reverses, and a standing wave built from two of these ends up
    with its nodes where the antinodes should be.
    """
    source = np.zeros((1, 3))
    r_near, r_far = 0.010, 0.012
    near = compute_green_TS(np.array([[0.0, 0.0, r_near]]), source, KF)[0, 0]
    far = compute_green_TS(np.array([[0.0, 0.0, r_far]]), source, KF)[0, 0]

    expected = (r_near / r_far) * np.exp(1j * KF * (r_far - r_near))
    assert far / near == pytest.approx(expected)


# --------------------------------------------------------------------------
# the projection
# --------------------------------------------------------------------------


def test_a_normal_across_the_ray_contributes_nothing():
    """The gradient is radial, so a perpendicular normal sees none of it.

    Isolates the projection from the kernel: this entry is zero whatever the
    Green's function does, and it is not zero if the contraction is summing
    the wrong axis.
    """
    source = np.zeros((1, 3))
    target = np.array([[0.02, 0.0, 0.0]])
    across = np.array([[0.0, 0.0, 1.0]])

    entry = compute_euler_gradn_green_TS(target, across, source, KF, C, RHO)[0, 0]
    assert entry == pytest.approx(0.0, abs=1e-12)


def test_a_normal_along_the_ray_gives_the_whole_gradient():
    source = np.zeros((1, 3))
    distance = 0.02
    target = np.array([[0.0, 0.0, distance]])
    along = np.array([[0.0, 0.0, 1.0]])

    entry = compute_euler_gradn_green_TS(target, along, source, KF, C, RHO)[0, 0]
    expected = (
        green_by_hand(target[0], source[0], KF)
        * (1j * KF - 1.0 / distance)
        / (1j * KF * C * RHO)
    )
    assert entry == pytest.approx(expected)


def test_the_projecting_normal_belongs_to_the_target():
    """Two targets at the same place, with different normals.

    The contraction is 'mnj,mj->mn': m is the target. Written with n instead,
    every row of a block gets the same normal, and on a flat piston -- where
    all normals are equal -- the result is identical and the mistake survives
    until the first curved surface. Here the two rows must differ: one normal
    lies along the ray and one across it, so a shared normal makes them
    equal and this fails.
    """
    source = np.zeros((1, 3))
    same_point = np.array([[0.0, 0.0, 0.02], [0.0, 0.0, 0.02]])
    normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    matrix = compute_euler_gradn_green_TS(same_point, normals, source, KF, C, RHO)
    assert abs(matrix[0, 0]) > 0.0
    assert matrix[1, 0] == pytest.approx(0.0, abs=1e-12)


def test_reversing_the_normal_reverses_the_sign():
    """The velocity is signed: which side of the surface is the fluid on.

    A normal pointing into the surface instead of out of it produces a field
    that solves and looks plausible, with the piston sucking where it should
    blow.
    """
    source = np.zeros((1, 3))
    target = np.array([[0.0, 0.0, 0.02]])
    out = np.array([[0.0, 0.0, 1.0]])

    forward = compute_euler_gradn_green_TS(target, out, source, KF, C, RHO)
    backward = compute_euler_gradn_green_TS(target, -out, source, KF, C, RHO)
    assert backward == pytest.approx(-forward)


# --------------------------------------------------------------------------
# the medium
# --------------------------------------------------------------------------


def test_euler_scales_inversely_with_the_impedance_factors():
    """c and rho enter only through the division by i * omega * rho.

    Neither appears anywhere else in the matrix, so doubling either one has
    to halve every entry exactly. A c that leaked into the wavenumber, or a
    rho that reached the kernel, breaks this and nothing else notices.
    """
    targets = cloud(4, seed=13)
    normals = unit_rows(cloud(4, seed=14, spread=1.0, offset=1.0))
    sources = cloud(3, seed=15, offset=0.25)

    base = compute_euler_gradn_green_TS(targets, normals, sources, KF, C, RHO)
    denser = compute_euler_gradn_green_TS(targets, normals, sources, KF, C, 2 * RHO)
    faster = compute_euler_gradn_green_TS(targets, normals, sources, KF, 2 * C, RHO)

    assert denser == pytest.approx(0.5 * base)
    assert faster == pytest.approx(0.5 * base)


def test_swapping_targets_and_sources_transposes_the_pressure_matrix():
    """Reciprocity: G depends on the distance and on nothing else.

    The pressure matrix cannot tell which cloud was called the target, so
    exchanging the two arguments has to return the transpose. This is the
    one property that fails if the subtraction inside separation is the
    right way round but the axes are assembled the wrong way, which the
    closed-form test above cannot see when M and N are both small.
    """
    first = cloud(5, seed=16)
    second = cloud(3, seed=17, offset=0.25)
    assert compute_green_TS(second, first, KF) == pytest.approx(
        compute_green_TS(first, second, KF).T
    )
