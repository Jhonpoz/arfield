"""Unit tests for arfield.field.

Step five of the method: strengths in, field out. The two functions are
short, so what is worth testing is not their arithmetic but the three
things that can go wrong in them without raising -- a strength paired with
the wrong source, an axis picked up in the wrong order, and the Euler
factor applied to the wrong side of the contraction.

`velocity` is checked against `pressure` by finite differences rather than
against the influence module, so a shared misreading of the kernel cannot
satisfy both.
"""

import numpy as np
import pytest

from arfield import mesh
from arfield.field import pressure, velocity

# Air at 40 kHz, the working point of every milestone in this project.
C = 343.0
RHO = 1.2
KF = 2.0 * np.pi * 40_000.0 / C
OMEGA = KF * C


def cloud(n, seed, spread=0.05, offset=0.0):
    """A reproducible scatter of points, away from the origin."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-spread, spread, size=(n, 3)) + offset


def strengths(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n) + 1j * rng.normal(size=n)


def green_by_hand(target, source, kf):
    """G(r) written out, with no package call in it."""
    r = np.linalg.norm(target - source)
    return np.exp(1j * kf * r) / (4.0 * np.pi * r)


def scene(n_targets=7, n_sources=5, seed=0):
    """Targets, strengths and sources, with M != N on purpose."""
    return (
        cloud(n_targets, seed=seed, offset=0.09),
        strengths(n_sources, seed=seed + 100),
        cloud(n_sources, seed=seed + 200, spread=0.005),
    )


# --------------------------------------------------------------------------
# pressure, against the formula rather than against the package
# --------------------------------------------------------------------------


def test_a_single_source_reproduces_the_greens_function():
    """One source of unit strength is the kernel itself.

    Pins the 1 / (4 pi) and the sign of the exponent in one assertion, and
    it is the only test here whose expected value is written by hand.
    """
    source = np.zeros((1, 3))
    target = np.array([[0.013, -0.021, 0.034]])

    value = pressure(target, np.array([1.0 + 0.0j]), source, KF)[0]
    assert value == pytest.approx(green_by_hand(target[0], source[0], KF))


def test_pressure_matches_a_direct_sum():
    """Every target, summed one source at a time with no package call.

    A double loop on purpose: it cannot be satisfied by a consistent
    misreading of the kernel, because it does not use the kernel.
    """
    targets, source_strengths, sources = scene()

    expected = np.array(
        [
            sum(
                a * green_by_hand(t, s, KF)
                for a, s in zip(source_strengths, sources, strict=True)
            )
            for t in targets
        ]
    )
    assert pressure(targets, source_strengths, sources, KF) == pytest.approx(expected)


def test_pressure_is_linear_in_the_strengths():
    """Superposition, which is the whole content of step five.

    Catches a normalisation by N and an abs() slipped in anywhere: both
    leave the shape and the order of magnitude untouched.
    """
    targets, first, sources = scene(seed=1)
    second = strengths(len(sources), seed=42)
    a, b = 2.0 - 1.0j, -0.5 + 3.0j

    combined = pressure(targets, a * first + b * second, sources, KF)
    apart = a * pressure(targets, first, sources, KF) + b * pressure(
        targets, second, sources, KF
    )
    assert combined == pytest.approx(apart)


def test_the_order_of_the_sources_does_not_matter():
    """A strength has to travel with its own source, not with its index.

    Permuting both together must leave the field alone. If the pairing were
    broken the field would still have the right shape and a plausible
    magnitude, and only a benchmark against a closed form would notice.
    """
    targets, source_strengths, sources = scene(seed=2)
    order = np.array([3, 0, 4, 1, 2])

    assert pressure(targets, source_strengths[order], sources[order], KF) == (
        pytest.approx(pressure(targets, source_strengths, sources, KF))
    )


def test_a_rigid_translation_leaves_the_pressure_alone():
    """Free space has no origin: only the differences matter.

    Catches a coordinate read from an absolute position instead of from the
    separation, which on a piston centred at the origin is invisible.
    """
    targets, source_strengths, sources = scene(seed=3)
    shift = np.array([0.7, -0.2, 1.3])

    moved = pressure(targets + shift, source_strengths, sources + shift, KF)
    assert moved == pytest.approx(pressure(targets, source_strengths, sources, KF))


def test_pressure_has_one_value_per_target_and_is_complex():
    """Including M = 1, which must not collapse to a scalar."""
    targets, source_strengths, sources = scene(seed=4)

    field = pressure(targets, source_strengths, sources, KF)
    assert field.shape == (len(targets),)
    assert field.dtype == np.complex128

    single = pressure(targets[:1], source_strengths, sources, KF)
    assert single.shape == (1,)


def test_real_strengths_still_give_a_complex_field():
    """The Rayleigh branch hands over real strengths on a uniform piston.

    A silent downcast would throw away the travelling part of the wave, and
    NumPy only warns about it.
    """
    targets, _, sources = scene(seed=5)
    field = pressure(targets, np.ones(len(sources)), sources, KF)
    assert field.dtype == np.complex128


# --------------------------------------------------------------------------
# velocity, against pressure
# --------------------------------------------------------------------------


def test_velocity_matches_finite_differences_of_pressure():
    """v = grad(p) / (i omega rho), checked without leaving this module.

    The reference is `pressure` itself, differenced. That is what makes this
    independent: a gradient kernel with the wrong sign or a factor astray
    disagrees with the pressure it is supposed to be the gradient of, even
    though both come from the same package.
    """
    targets, source_strengths, sources = scene(n_targets=6, seed=6)
    step = 1e-7

    expected = np.empty((len(targets), 3), dtype=np.complex128)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        ahead = pressure(targets + offset, source_strengths, sources, KF)
        behind = pressure(targets - offset, source_strengths, sources, KF)
        expected[:, axis] = (ahead - behind) / (2.0 * step)
    expected /= 1j * OMEGA * RHO

    field = velocity(targets, source_strengths, sources, KF, C, RHO)
    assert field == pytest.approx(expected, rel=1e-6)


def test_a_single_source_velocity_matches_the_closed_form():
    """v = p (i k - 1 / R) e_R / (i omega rho), written out by hand."""
    source = np.zeros((1, 3))
    target = np.array([[0.013, -0.021, 0.034]])
    distance = np.linalg.norm(target[0])

    field = velocity(target, np.array([1.0 + 0.0j]), source, KF, C, RHO)[0]
    expected = (
        green_by_hand(target[0], source[0], KF)
        * (1j * KF - 1.0 / distance)
        / (1j * OMEGA * RHO)
        * (target[0] / distance)
    )
    assert field == pytest.approx(expected)


def test_a_single_source_velocity_is_radial():
    """The gradient of a spherically symmetric field points along the ray.

    Isolates the direction from the magnitude: two swapped components give
    a vector of the same length pointing somewhere else, and no test that
    looks at abs(v) sees it.
    """
    source = np.zeros((1, 3))
    target = np.array([[0.013, -0.021, 0.034]])

    field = velocity(target, np.array([1.0 + 0.0j]), source, KF, C, RHO)[0]
    assert np.cross(field, target[0]) == pytest.approx(np.zeros(3), abs=1e-12)


def test_the_impedance_of_a_single_source_is_the_textbook_one():
    """p / v_R = rho c / (1 + i / (k R)), tending to rho c far away.

    The one check that ties the field back to a property of the fluid. A
    missing c, or an omega assembled from the wrong pair, breaks it by a
    factor large enough to see by eye.
    """
    source = np.zeros((1, 3))
    distance = 0.05
    target = np.array([[0.0, 0.0, distance]])
    strength = np.array([1.0 + 0.0j])

    p = pressure(target, strength, source, KF)[0]
    v = velocity(target, strength, source, KF, C, RHO)[0]
    radial = v @ (target[0] / distance)

    assert p / radial == pytest.approx(RHO * C / (1.0 + 1j / (KF * distance)))


def test_velocity_rotates_with_the_geometry():
    """A vector field, so a rotation of the scene rotates the answer.

    Deliberately not the invariance that holds for `pressure`: writing the
    same assertion for a vector would pass only if the components were being
    dropped or the magnitude returned instead.
    """
    targets, source_strengths, sources = scene(seed=7)
    matrix = mesh.rotation([0.3, -0.7, 0.2], 0.9)

    turned = velocity(
        targets @ matrix.T, source_strengths, sources @ matrix.T, KF, C, RHO
    )
    straight = velocity(targets, source_strengths, sources, KF, C, RHO)
    assert turned == pytest.approx(straight @ matrix.T)


def test_velocity_scales_inversely_with_c_and_rho():
    """Both enter only through the division by i * omega * rho.

    Neither appears anywhere else, so doubling either has to halve every
    component exactly. A c that leaked into the wavenumber breaks this and
    nothing else notices.
    """
    targets, source_strengths, sources = scene(seed=8)

    base = velocity(targets, source_strengths, sources, KF, C, RHO)
    denser = velocity(targets, source_strengths, sources, KF, C, 2 * RHO)
    faster = velocity(targets, source_strengths, sources, KF, 2 * C, RHO)

    assert denser == pytest.approx(0.5 * base)
    assert faster == pytest.approx(0.5 * base)


def test_velocity_is_linear_in_the_strengths():
    targets, first, sources = scene(seed=9)
    second = strengths(len(sources), seed=43)

    combined = velocity(targets, first + second, sources, KF, C, RHO)
    apart = velocity(targets, first, sources, KF, C, RHO) + velocity(
        targets, second, sources, KF, C, RHO
    )
    assert combined == pytest.approx(apart)


def test_velocity_has_three_components_per_target_and_is_complex():
    targets, source_strengths, sources = scene(seed=10)

    field = velocity(targets, source_strengths, sources, KF, C, RHO)
    assert field.shape == (len(targets), 3)
    assert field.dtype == np.complex128

    single = velocity(targets[:1], source_strengths, sources, KF, C, RHO)
    assert single.shape == (1, 3)


# --------------------------------------------------------------------------
# the contract with the caller
# --------------------------------------------------------------------------


def test_read_only_inputs_are_accepted():
    """Source freezes its arrays, and Source.positions is what gets passed.

    Nothing here needs to write into its inputs, but separation normalises
    in place, and an in-place step one array too early would fail only when
    fed a real Source -- never in a test built from fresh arrays.
    """
    targets, source_strengths, sources = scene(seed=11)
    for array in (targets, source_strengths, sources):
        array.flags.writeable = False

    pressure(targets, source_strengths, sources, KF)
    velocity(targets, source_strengths, sources, KF, C, RHO)


def test_the_inputs_come_back_unchanged():
    """Evaluating a field twice on the same scene must give the same answer.

    The strengths in particular are reused: the solved branch and the
    assigned branch are compared on one geometry, and a call that consumed
    its arguments would make the second curve depend on the first.
    """
    targets, source_strengths, sources = scene(seed=12)
    before = (targets.copy(), source_strengths.copy(), sources.copy())

    pressure(targets, source_strengths, sources, KF)
    velocity(targets, source_strengths, sources, KF, C, RHO)

    for after, original in zip(
        (targets, source_strengths, sources), before, strict=True
    ):
        assert np.array_equal(after, original)
