"""Tests for `arfield.pairwise`."""

import tracemalloc

import numpy as np
import pytest

from arfield.pairwise import separation


def two_clouds(n_targets=5, n_sources=4, seed=0):
    """Two clouds of different size, so a swapped axis cannot pass."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_targets, 3)), rng.standard_normal((n_sources, 3))


def test_distance_matches_a_direct_computation():
    targets, sources = two_clouds()
    expected = np.empty((len(targets), len(sources)))
    for m, target in enumerate(targets):
        for n, source in enumerate(sources):
            dx, dy, dz = target - source
            expected[m, n] = np.sqrt(dx * dx + dy * dy + dz * dz)

    r_TS, _ = separation(targets, sources)

    assert r_TS.shape == expected.shape
    np.testing.assert_allclose(r_TS, expected, rtol=1e-15)


def test_unit_vectors_have_unit_norm():
    targets, sources = two_clouds()

    _, e_TSj = separation(targets, sources)

    assert e_TSj.shape == (len(targets), len(sources), 3)
    np.testing.assert_allclose(np.linalg.norm(e_TSj, axis=-1), 1.0, rtol=1e-15)


def test_unit_vector_points_from_source_to_target():
    targets, sources = two_clouds()
    difference = targets[:, None, :] - sources[None, :, :]

    _, e_TSj = separation(targets, sources)

    # Parallel to the difference: the cross product vanishes.
    np.testing.assert_allclose(np.cross(e_TSj, difference), 0.0, atol=1e-15)
    # And in the same sense, not the opposite one: the projection is positive.
    assert np.all(np.einsum("mnj,mnj->mn", e_TSj, difference) > 0.0)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_exact_for_a_pythagorean_triple(dtype):
    targets = np.array([[3.0, 4.0, 0.0]], dtype=dtype)
    sources = np.zeros((1, 3), dtype=dtype)

    r_TS, e_TSj = separation(targets, sources)

    assert r_TS.dtype == dtype
    assert e_TSj.dtype == dtype
    assert r_TS[0, 0] == 5.0
    np.testing.assert_array_equal(e_TSj[0, 0], np.array([0.6, 0.8, 0.0], dtype=dtype))


def test_integer_input_raises():
    targets = np.array([[1, 2, 3]])
    sources = np.array([[0, 0, 0]])

    # The in-place division cannot write float results into an integer array.
    with pytest.raises(TypeError):
        separation(targets, sources)


def test_coincident_points_give_nan_with_a_warning():
    point = np.zeros((1, 3))

    with pytest.warns(RuntimeWarning):
        r_TS, e_TSj = separation(point, point)

    assert r_TS[0, 0] == 0.0
    assert np.all(np.isnan(e_TSj))


def test_separation_allocates_little_beyond_what_it_returns():
    """Guard on the two lines that keep the peak down; see the docstring.

    Measured ratios: 1.25 as written, 1.75 if the division stops being in
    place, 2.00 if the distances go back to ``np.linalg.norm``. The
    threshold sits between the first and the second.
    """
    targets, sources = two_clouds(200, 200)

    tracemalloc.start()
    try:
        r_TS, e_TSj = separation(targets, sources)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    returned = r_TS.nbytes + e_TSj.nbytes
    assert peak / returned < 1.5
