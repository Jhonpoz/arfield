"""Unit tests for arfield.solver.

The module is one call to NumPy, so what is worth testing is not the
arithmetic but the contract: that it inverts the map the influence matrix
defines, and that it fails loudly in the two ways a source layer can be
built wrong.
"""

import numpy as np
import pytest

from arfield.solver import solve_strength


def known_system(n=6, seed=0):
    """A well-conditioned complex system with the answer known in advance.

    Built the other way round -- strengths first, velocities from them -- so
    the expected result owes nothing to the routine under test.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    matrix += n * np.eye(n)  # keep it away from singular
    expected = rng.normal(size=n) + 1j * rng.normal(size=n)
    return matrix, matrix @ expected, expected


def test_the_identity_returns_the_velocity_unchanged():
    """With no coupling between sources, each one answers for its own point.

    The smallest statement of what the routine means, and the one that
    localises a failure: if this passes and the next one does not, the
    problem is in the matrix, not here.
    """
    v0 = np.array([1.0 + 0.0j, 0.0 - 2.0j, 3.5 + 1.0j])
    assert solve_strength(np.eye(3, dtype=complex), v0) == pytest.approx(v0)


def test_a_known_system_is_recovered():
    matrix, v0, expected = known_system()
    assert solve_strength(matrix, v0) == pytest.approx(expected, rel=1e-12)


def test_the_solution_reproduces_the_velocity_it_was_given():
    """The residual, which is the property the milestone actually needs.

    Recovering the strengths is what the routine promises; reproducing v0 is
    what the physics asks for, and the two stop agreeing first when the
    matrix is badly conditioned rather than when the solve is wrong.
    """
    matrix, v0, _ = known_system(n=12, seed=1)
    strengths = solve_strength(matrix, v0)
    assert matrix @ strengths == pytest.approx(v0, rel=1e-10)


def test_the_result_is_complex_even_when_the_velocity_is_real():
    """A uniform piston has a real v0 and complex strengths.

    A silent downcast here would throw away the phase that the whole method
    exists to carry, and every field built afterwards would be a standing
    wave with no travelling part.
    """
    matrix, _, _ = known_system(n=4, seed=2)
    strengths = solve_strength(matrix, np.ones(4))
    assert strengths.dtype == np.complex128


def test_inputs_are_left_alone():
    """Callers keep the matrix: for cond, for a second right-hand side.

    ADR 0008 takes the matrix as an argument precisely so it can be reused,
    which only holds if solving does not consume it.
    """
    matrix, v0, _ = known_system(n=5, seed=3)
    before_matrix, before_v0 = matrix.copy(), v0.copy()
    solve_strength(matrix, v0)
    assert np.array_equal(matrix, before_matrix)
    assert np.array_equal(v0, before_v0)


def test_read_only_inputs_are_accepted():
    """The matrix may come from an array a caller froze, as Source does."""
    matrix, v0, expected = known_system(n=5, seed=4)
    matrix.flags.writeable = False
    v0.flags.writeable = False
    assert solve_strength(matrix, v0) == pytest.approx(expected, rel=1e-12)


def test_a_singular_matrix_is_refused():
    """The real way to reach this: alpha = 0, sources on the surface itself.

    Every source then radiates from its own collocation point and the rows
    stop being independent. Source allows that geometry on purpose, because
    it is usable for evaluating a field from known strengths; it is solving
    for them that has no answer, and the failure has to arrive here rather
    than as strengths of order 1e17 that go on to produce a field.
    """
    singular = np.ones((3, 3), dtype=complex)
    with pytest.raises(np.linalg.LinAlgError):
        solve_strength(singular, np.ones(3, dtype=complex))


def test_mismatched_shapes_are_refused():
    """A v0 built for one layer and a matrix built for another.

    Nothing checks that the two came from the same geometry, so the only
    mismatch that can be caught is the one that changes the length -- and it
    is worth knowing that this one, at least, does not go through.
    """
    matrix, _, _ = known_system(n=4, seed=5)
    with pytest.raises(ValueError):
        solve_strength(matrix, np.ones(3, dtype=complex))
