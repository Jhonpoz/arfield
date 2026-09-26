"""Unit tests for arfield.field.

Step five of the method: strengths in, field out. The calculations are
short, so what is worth testing is not their arithmetic but the things that
can go wrong in them without raising -- a strength paired with the wrong
source, an axis picked up in the wrong order, and the Euler factor applied
to the wrong side of a contraction.

Each quantity is checked against the one below it by finite differences
rather than against the influence module: `velocity` against `pressure`,
`velocity_gradient` against `velocity`. A shared misreading of the kernel
cannot satisfy both ends of a difference.

The three functions go through `Field`, so the physics tests exercise the
class as well. What the class adds on its own is caching, and that is
tested separately, by looking at ``__dict__``: a table that is built when
it should not be, or rebuilt when it is already there, changes no number
and is invisible to every assertion above.
"""

import numpy as np
import pytest

from arfield import mesh
from arfield.field import Field, pressure, velocity, velocity_gradient
from arfield.green_kernel import hessian_green
from arfield.influence import compute_green_TS
from arfield.medium import Medium
from arfield.pairwise import separation

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


# --------------------------------------------------------------------------
# the class, as a cache: nothing here checks a physical value
# --------------------------------------------------------------------------


def field_for(seed=20):
    """A `Field` and the strengths to spend on it."""
    targets, source_strengths, sources = scene(seed=seed)
    return Field(targets, sources, KF, Medium(C, RHO)), source_strengths


TABLES = ("_separation", "_green_TS", "_d_green_dr_TS", "_aux_h", "_aux_s")


def cached(field):
    """The tables built so far, read off the instance dictionary."""
    return {name for name in TABLES if name in field.__dict__}


def test_construction_computes_nothing():
    """The object is cheap to make; the tables cost M * N each.

    A constructor that eagerly filled them would make blocking over targets
    pointless, since the peak would be paid before the first method call.
    """
    field, _ = field_for()
    assert cached(field) == set()


def test_the_first_table_matches_the_influence_module():
    """The chain must not have changed the numbers Milestone 1 validated.

    `compute_green_TS` reaches the same kernel by its own route. Both go
    through the same difference and distance, so the agreement is exact,
    and anything less than exact is a change of arithmetic worth seeing.
    """
    field, _ = field_for()
    reference = compute_green_TS(field.targets, field.sources, KF)
    assert np.array_equal(field._green_TS, reference)


def test_a_table_is_computed_once():
    """Identity, not equality: two equal arrays would pass a value check."""
    field, _ = field_for()
    assert field._green_TS is field._green_TS
    assert field._separation is field._separation


def test_pressure_does_not_build_the_tables_it_does_not_need():
    """Pressure needs the distances and the kernel, and nothing else.

    Building the derivative tables here would cost 32 bytes per pair for an
    answer that never looks at them.
    """
    field, source_strengths = field_for()
    field.pressure(source_strengths)
    assert cached(field) == {"_separation", "_green_TS"}


def test_velocity_does_not_build_the_gradient_tables():
    """`h` and `s` belong to `velocity_gradient` alone."""
    field, source_strengths = field_for()
    field.velocity(source_strengths)
    assert cached(field) == {"_separation", "_green_TS", "_d_green_dr_TS"}


def test_a_second_quantity_reuses_the_tables_of_the_first():
    """The whole point of the class: one separation, one exponential.

    Asking for the velocity after the pressure must not rebuild what the
    pressure already paid for.
    """
    field, source_strengths = field_for()
    field.pressure(source_strengths)
    first = field._green_TS

    field.velocity(source_strengths)
    assert field._green_TS is first


def test_deleting_a_table_recomputes_it():
    """The escape valve for memory: drop a table, get it back on demand."""
    field, _ = field_for()
    original = field._green_TS

    del field._green_TS
    assert "_green_TS" not in field.__dict__

    rebuilt = field._green_TS
    assert rebuilt is not original
    assert np.array_equal(rebuilt, original)


def test_the_inputs_are_read_once_and_then_the_geometry_is_fixed():
    """The arrays are stored, not copied, and one chain hangs off them.

    Mutating before the first table takes effect; mutating after it does
    not, and cannot leave a mixed state, because every table descends from
    the single cached separation. That is the property worth pinning: a
    split chain would let one table describe the old geometry and the next
    the new, with nothing to signal it.
    """
    field, source_strengths = field_for()
    field.pressure(source_strengths)
    frozen = field.targets.copy()

    field.targets[0] += 0.01

    assert field.pressure(source_strengths) == pytest.approx(
        pressure(frozen, source_strengths, field.sources, KF)
    )
    assert field.velocity(source_strengths) == pytest.approx(
        velocity(frozen, source_strengths, field.sources, KF, C, RHO)
    )


def test_the_tables_have_the_shapes_and_dtypes_of_their_definitions():
    """A real table and a complex one, and the pair axis before the source."""
    field, _ = field_for()
    shape = (len(field.targets), len(field.sources))

    assert field._r_TS.shape == shape
    assert field._r_TS.dtype == np.float64
    assert field._e_TSj.shape == (*shape, 3)
    assert field._e_TSj.dtype == np.float64
    for table in (field._green_TS, field._d_green_dr_TS, field._aux_h, field._aux_s):
        assert table.shape == shape
        assert table.dtype == np.complex128


def test_a_field_without_a_medium_gives_pressure_and_refuses_the_rest():
    """There is no default fluid, and pressure does not need one.

    A `Field` that silently assumed air would return plausible numbers for a
    medium nobody chose.
    """
    targets, source_strengths, sources = scene(seed=21)
    field = Field(targets, sources, KF)

    field.pressure(source_strengths)
    with pytest.raises(AttributeError):
        field.velocity(source_strengths)
    with pytest.raises(AttributeError):
        field.velocity_gradient(source_strengths)


def test_the_functions_agree_with_the_class():
    """The wrappers must be wrappers, not a second implementation."""
    targets, source_strengths, sources = scene(seed=22)
    field = Field(targets, sources, KF, Medium(C, RHO))

    assert np.array_equal(
        pressure(targets, source_strengths, sources, KF),
        field.pressure(source_strengths),
    )
    assert np.array_equal(
        velocity(targets, source_strengths, sources, KF, C, RHO),
        field.velocity(source_strengths),
    )
    assert np.array_equal(
        velocity_gradient(targets, source_strengths, sources, KF, C, RHO),
        field.velocity_gradient(source_strengths),
    )


# --------------------------------------------------------------------------
# velocity_gradient, against velocity
# --------------------------------------------------------------------------


def test_velocity_gradient_matches_finite_differences_of_velocity():
    """Entry [m, i, j] is d v_j / d x_i, differenced along each axis.

    The independent check of the whole contraction: an index order swapped
    between the two unit vectors, or `s` used where `h` belongs, disagrees
    with the velocity it is supposed to be the gradient of.
    """
    targets, source_strengths, sources = scene(n_targets=6, seed=23)
    step = 1e-7

    expected = np.empty((len(targets), 3, 3), dtype=np.complex128)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        ahead = velocity(targets + offset, source_strengths, sources, KF, C, RHO)
        behind = velocity(targets - offset, source_strengths, sources, KF, C, RHO)
        expected[:, axis, :] = (ahead - behind) / (2.0 * step)

    gradient = velocity_gradient(targets, source_strengths, sources, KF, C, RHO)
    assert gradient == pytest.approx(expected, rel=1e-6)


def test_velocity_gradient_matches_the_tabulated_hessian():
    """The same sum with the (M, N, 3, 3) array built, which it never is.

    `hessian_green` has no other caller now, and this is why it stays: it is
    the reference the contracted route is checked against. Agreement is to
    rounding, not exact, because the factors are multiplied in a different
    order and floating-point addition is not associative.
    """
    targets, source_strengths, sources = scene(seed=24)
    r_TS, e_TSj = separation(targets, sources)

    expected = np.einsum(
        "mnij,n->mij", hessian_green(r_TS, e_TSj, KF), source_strengths
    ) / (1j * OMEGA * RHO)

    gradient = velocity_gradient(targets, source_strengths, sources, KF, C, RHO)
    assert gradient == pytest.approx(expected, rel=1e-12)


def test_the_velocity_gradient_is_symmetric():
    """It is the Hessian of a scalar, so d_i v_j = d_j v_i.

    Free, and it catches an asymmetric contraction: writing the two unit
    vector axes from the same operand twice by mistake would still produce a
    (M, 3, 3) that passes every shape check.
    """
    targets, source_strengths, sources = scene(seed=25)
    gradient = velocity_gradient(targets, source_strengths, sources, KF, C, RHO)
    assert gradient == pytest.approx(gradient.transpose(0, 2, 1))


def test_the_trace_of_the_velocity_gradient_is_helmholtz():
    """sum_i d_i v_i = -kf**2 p / (i omega rho), away from the sources.

    The diagonal of ``delta_ij h - e_i e_j s`` sums to ``3h - s``, which is
    ``-kf**2 G`` by the definition of `s`. That subtraction is badly
    conditioned when ``3h`` dwarfs ``kf**2 G``, so the identity is asserted
    where the package evaluates: the scene is checked to sit at kf * r of
    order one or more, and the tolerance would have to be loosened for a
    target pressed against a source.
    """
    targets, source_strengths, sources = scene(seed=26)
    r_TS, _ = separation(targets, sources)
    assert (KF * r_TS).min() > 1.0

    gradient = velocity_gradient(targets, source_strengths, sources, KF, C, RHO)
    field = pressure(targets, source_strengths, sources, KF)

    trace = np.trace(gradient, axis1=1, axis2=2)
    assert trace == pytest.approx(-(KF**2) * field / (1j * OMEGA * RHO))


def test_the_velocity_gradient_rotates_with_the_geometry():
    """A rank-two tensor: turning the scene gives Q G Q^T, not G.

    The assertion that holds for the pressure and the one that holds for the
    velocity would both pass on a wrongly transposed answer; this one does
    not.
    """
    targets, source_strengths, sources = scene(seed=27)
    matrix = mesh.rotation([0.3, -0.7, 0.2], 0.9)

    turned = velocity_gradient(
        targets @ matrix.T, source_strengths, sources @ matrix.T, KF, C, RHO
    )
    straight = velocity_gradient(targets, source_strengths, sources, KF, C, RHO)
    assert turned == pytest.approx(matrix @ straight @ matrix.T)


def test_the_velocity_gradient_is_linear_in_the_strengths():
    """Both terms contract A once, so superposition has to survive both."""
    targets, first, sources = scene(seed=28)
    second = strengths(len(sources), seed=44)

    combined = velocity_gradient(targets, first + second, sources, KF, C, RHO)
    one = velocity_gradient(targets, first, sources, KF, C, RHO)
    other = velocity_gradient(targets, second, sources, KF, C, RHO)
    assert combined == pytest.approx(one + other)


def test_the_velocity_gradient_scales_inversely_with_c_and_rho():
    """Same Euler factor as `velocity`, applied to the (M, 3, 3) result.

    A factor folded into `h` or `s` instead would scale the diagonal and the
    outer product differently and break this only in one of the two terms.
    """
    targets, source_strengths, sources = scene(seed=29)

    base = velocity_gradient(targets, source_strengths, sources, KF, C, RHO)
    denser = velocity_gradient(targets, source_strengths, sources, KF, C, 2 * RHO)
    faster = velocity_gradient(targets, source_strengths, sources, KF, 2 * C, RHO)

    assert denser == pytest.approx(0.5 * base)
    assert faster == pytest.approx(0.5 * base)


def test_the_velocity_gradient_has_a_matrix_per_target_and_is_complex():
    """One 3 by 3 per target, in target order, and never real."""
    targets, source_strengths, sources = scene(n_targets=9, seed=30)
    gradient = velocity_gradient(targets, source_strengths, sources, KF, C, RHO)

    assert gradient.shape == (9, 3, 3)
    assert gradient.dtype == np.complex128
