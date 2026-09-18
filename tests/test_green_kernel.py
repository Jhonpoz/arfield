"""Unit tests for arfield.green_kernel.

The module is eleven one-line functions, so what is worth testing is not
their arithmetic but three contracts: that each plain spelling is the
closed form written by hand; that each ``_from`` spelling is the plain one
bit for bit, so chaining from a single ``green`` loses nothing; and that
every function is elementwise, so ``(N,)``, ``(M, N)`` and a scalar all
work and give the same numbers.

The Hessian gets two identities that hold for any ``r``: symmetry in the
component axes and the trace, which is Helmholtz outside the origin. Both
are cheap and both fail for the most likely slips -- a dropped ``kf**2``
in ``s`` or a ``3`` gone missing.
"""

import numpy as np
import pytest

from arfield.green_kernel import (
    aux_fun_h,
    aux_fun_h_from,
    aux_fun_s,
    aux_fun_s_from,
    d_green_dr,
    d_green_dr_from,
    gradient_green,
    gradient_green_from,
    green,
    hessian_green,
    hessian_green_from,
)

# Air at 40 kHz, the working point of every milestone in this project.
KF = 2.0 * np.pi * 40_000.0 / 343.0


def distances(shape, seed=0, low=2e-3, high=5e-2):
    """Reproducible positive distances, away from the singularity."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=shape)


def unit_vectors(shape, seed=1):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size=(*shape, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# --------------------------------------------------------------------------
# closed forms, written out with no package call in them
# --------------------------------------------------------------------------


def green_by_hand(r, kf):
    return np.exp(1j * kf * r) / (4.0 * np.pi * r)


def d_green_dr_by_hand(r, kf):
    return np.exp(1j * kf * r) * (1j * kf * r - 1.0) / (4.0 * np.pi * r**2)


def h_by_hand(r, kf):
    return np.exp(1j * kf * r) * (1j * kf * r - 1.0) / (4.0 * np.pi * r**3)


def hessian_by_hand(r, e_r, kf):
    """The long form of gorkov_dpsm.pdf, Eq. (19), with R_i R_j / R**2 = e_i e_j."""
    diagonal = np.exp(1j * kf * r) * (1j * kf * r - 1.0) / r**3
    outer = np.exp(1j * kf * r) * (3.0 - 3j * kf * r - kf**2 * r**2) / r**3
    e_outer = e_r[..., :, None] * e_r[..., None, :]
    hess = np.eye(3) * diagonal[..., None, None] + e_outer * outer[..., None, None]
    return hess / (4.0 * np.pi)


# --------------------------------------------------------------------------
# the plain spellings, against the closed forms
# --------------------------------------------------------------------------


def test_green_matches_the_closed_form():
    r = distances((5, 3))
    assert green(r, KF) == pytest.approx(green_by_hand(r, KF))


def test_green_has_complex128_dtype_and_the_shape_of_r():
    for shape in ((), (4,), (5, 3)):
        r = distances(shape)
        g = green(r, KF)
        assert np.shape(g) == shape
        assert np.asarray(g).dtype == np.complex128


def test_d_green_dr_matches_the_closed_form():
    r = distances((5, 3), seed=2)
    assert d_green_dr(r, KF) == pytest.approx(d_green_dr_by_hand(r, KF))


def test_d_green_dr_matches_finite_differences_of_green():
    """The derivative is the derivative of the function next to it."""
    r = distances((6,), seed=3)
    step = 1e-8
    expected = (green(r + step, KF) - green(r - step, KF)) / (2.0 * step)
    assert d_green_dr(r, KF) == pytest.approx(expected, rel=1e-6)


def test_gradient_green_is_the_radial_derivative_along_e_r():
    r = distances((5, 3), seed=4)
    e_r = unit_vectors((5, 3), seed=5)
    expected = d_green_dr_by_hand(r, KF)[..., None] * e_r
    grad = gradient_green(r, e_r, KF)
    assert grad.shape == (5, 3, 3)
    assert grad == pytest.approx(expected)


def test_reversing_e_r_reverses_the_gradient():
    r = distances((4,), seed=6)
    e_r = unit_vectors((4,), seed=7)
    assert gradient_green(r, -e_r, KF) == pytest.approx(-gradient_green(r, e_r, KF))


def test_h_matches_the_closed_form():
    """h = (dG/dr) / r = exp(ikr) (ikr - 1) / (4 pi r**3), Eq. (10) of the notes."""
    r = distances((5, 3), seed=8)
    assert aux_fun_h(r, KF) == pytest.approx(h_by_hand(r, KF))


def test_h_times_the_separation_vector_is_the_gradient():
    """grad G = R * h with R = r * e_r: the identity h exists for."""
    r = distances((4,), seed=9)
    e_r = unit_vectors((4,), seed=10)
    R = r[..., None] * e_r
    assert R * aux_fun_h(r, KF)[..., None] == pytest.approx(gradient_green(r, e_r, KF))


def test_s_matches_its_definition():
    r = distances((5, 3), seed=11)
    expected = KF**2 * green_by_hand(r, KF) + 3.0 * h_by_hand(r, KF)
    assert aux_fun_s(r, KF) == pytest.approx(expected)


# --------------------------------------------------------------------------
# the chain: every _from spelling is the plain one, bit for bit
# --------------------------------------------------------------------------


def test_the_chain_from_one_green_reproduces_every_plain_spelling():
    """One exponential, five kernels, no tolerance.

    This is the contract the field evaluation will rely on: holding
    ``green(r, kf)`` and chaining is the same arithmetic as calling each
    plain function, because the plain ones call the ``_from`` ones. Exact
    equality, not approx: a ``_from`` that recomputed something on its own
    path would differ in the last bits and fail here.
    """
    r = distances((5, 3), seed=12)
    e_r = unit_vectors((5, 3), seed=13)

    g = green(r, KF)
    dgdr = d_green_dr_from(g, r, KF)
    grad = gradient_green_from(dgdr, e_r)
    h = aux_fun_h_from(dgdr, r)
    s = aux_fun_s_from(g, h, KF)
    hess = hessian_green_from(h, s, e_r)

    np.testing.assert_array_equal(dgdr, d_green_dr(r, KF))
    np.testing.assert_array_equal(grad, gradient_green(r, e_r, KF))
    np.testing.assert_array_equal(h, aux_fun_h(r, KF))
    np.testing.assert_array_equal(s, aux_fun_s(r, KF))
    np.testing.assert_array_equal(hess, hessian_green(r, e_r, KF))


def test_from_spellings_do_not_touch_their_inputs():
    """Chaining reuses arrays; a _from that wrote into one would corrupt
    every table computed after it from the same predecessor."""
    r = distances((3, 2), seed=14)
    e_r = unit_vectors((3, 2), seed=15)
    g = green(r, KF)
    before = (g.copy(), r.copy(), e_r.copy())

    dgdr = d_green_dr_from(g, r, KF)
    h = aux_fun_h_from(dgdr, r)
    s = aux_fun_s_from(g, h, KF)
    gradient_green_from(dgdr, e_r)
    hessian_green_from(h, s, e_r)

    for after, original in zip((g, r, e_r), before, strict=True):
        np.testing.assert_array_equal(after, original)


# --------------------------------------------------------------------------
# the Hessian
# --------------------------------------------------------------------------


def test_hessian_matches_the_long_closed_form():
    """Against Eq. (19) of gorkov_dpsm.pdf, the form with R_i R_j / R**5.

    The package uses the short form ``delta h - e e s``; this is the
    independent derivation, written out with neither h nor s in it.
    """
    r = distances((5, 3), seed=16)
    e_r = unit_vectors((5, 3), seed=17)
    hess = hessian_green(r, e_r, KF)
    assert hess.shape == (5, 3, 3, 3)
    assert hess.dtype == np.complex128
    assert hess == pytest.approx(hessian_by_hand(r, e_r, KF))


def test_hessian_is_symmetric_in_its_component_axes():
    r = distances((5, 3), seed=18)
    e_r = unit_vectors((5, 3), seed=19)
    hess = hessian_green(r, e_r, KF)
    assert hess == pytest.approx(np.swapaxes(hess, -1, -2))


def test_hessian_trace_is_helmholtz():
    """sum_i d2G/dx_i2 = -kf**2 G outside the origin.

    3 h - (kf**2 G + 3 h) = -kf**2 G, so the trace fails for a missing 3
    in ``s`` and for a ``kf`` that is not squared, which are the two slips
    the derivation actually produced along the way.
    """
    r = distances((5, 3), seed=20)
    e_r = unit_vectors((5, 3), seed=21)
    trace = np.trace(hessian_green(r, e_r, KF), axis1=-2, axis2=-1)
    assert trace == pytest.approx(-(KF**2) * green(r, KF))


def test_hessian_is_even_in_e_r():
    """Second derivatives do not care which way the separation points."""
    r = distances((4,), seed=22)
    e_r = unit_vectors((4,), seed=23)
    assert hessian_green(r, -e_r, KF) == pytest.approx(hessian_green(r, e_r, KF))


def test_hessian_matches_finite_differences_of_the_gradient():
    """Differentiate grad G with respect to the target coordinate.

    Built from a source at the origin and a handful of targets, so the
    dependence of ``r`` and ``e_r`` on the coordinate is real and not
    assumed. Central differences on each axis of the target; the answer
    has to be the column of the Hessian for that axis.
    """
    rng = np.random.default_rng(24)
    targets = rng.uniform(-3e-2, 3e-2, size=(6, 3)) + np.array([0.04, 0.0, 0.0])
    step = 1e-7

    def grad_at(points):
        r = np.linalg.norm(points, axis=-1)
        return gradient_green(r, points / r[..., None], KF)

    expected = np.empty((6, 3, 3), dtype=np.complex128)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        ahead = grad_at(targets + offset)
        behind = grad_at(targets - offset)
        expected[:, axis, :] = (ahead - behind) / (2.0 * step)

    r = np.linalg.norm(targets, axis=-1)
    hess = hessian_green(r, targets / r[..., None], KF)
    assert hess == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------
# elementwise: shape in, shape out
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(), (7,), (5, 3), (2, 3, 4)])
def test_every_kernel_is_elementwise_over_leading_dimensions(shape):
    """A scalar, a row, a table and a stack all give the same numbers.

    This is what lets one function serve the (N, N) system, an (M, N)
    field table and a single-point evaluation. The reference is the flat
    (K,) call on the raveled input, reshaped back.
    """
    r = distances(shape, seed=25)
    e_r = unit_vectors(shape, seed=26)
    flat_r = np.ravel(r)
    flat_e = np.reshape(e_r, (-1, 3))

    assert np.reshape(green(flat_r, KF), shape) == pytest.approx(green(r, KF))
    assert np.reshape(aux_fun_h(flat_r, KF), shape) == pytest.approx(aux_fun_h(r, KF))
    assert np.reshape(gradient_green(flat_r, flat_e, KF), (*shape, 3)) == pytest.approx(
        gradient_green(r, e_r, KF)
    )
    flat_hess = hessian_green(flat_r, flat_e, KF)
    assert np.reshape(flat_hess, (*shape, 3, 3)) == pytest.approx(
        hessian_green(r, e_r, KF)
    )
