"""Tests for `arfield.pairwise`."""

import tracemalloc

import numpy as np
import pytest

from arfield.pairwise import separation, separation_distance


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


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_exact_for_a_pythagorean_triple(dtype):
    """3-4-5, and the output is float64 whatever the input dtype was.

    The dtype policy lives in ``_difference`` and nowhere else: integer and
    float32 coordinates are converted, never refused and never propagated.
    A float32 result here would mean the conversion was lost, and the
    in-place division of `separation` would be the next thing to break.
    """
    targets = np.array([[3, 4, 0]], dtype=dtype)
    sources = np.zeros((1, 3), dtype=dtype)

    r_TS, e_TSj = separation(targets, sources)

    assert r_TS.dtype == np.float64
    assert e_TSj.dtype == np.float64
    assert r_TS[0, 0] == 5.0
    np.testing.assert_array_equal(e_TSj[0, 0], np.array([0.6, 0.8, 0.0]))


def test_integer_input_is_converted_by_both_public_functions():
    """Same policy on both entry points, which is why it has one owner.

    Before ``_difference`` existed, `separation` refused integers by accident
    (NumPy could not divide in place into an int array) and a distances-only
    path accepted them; two functions of one module disagreed without saying
    so. Now both convert, and both must agree with the float64 call bit for
    bit, because the conversion happens before any arithmetic.
    """
    targets_int = np.array([[1, 2, 3], [4, 5, 6]])
    sources_int = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 2]])
    targets = targets_int.astype(np.float64)
    sources = sources_int.astype(np.float64)

    r_int, e_int = separation(targets_int, sources_int)
    r_flt, e_flt = separation(targets, sources)
    assert r_int.dtype == np.float64
    np.testing.assert_array_equal(r_int, r_flt)
    np.testing.assert_array_equal(e_int, e_flt)

    d_int = separation_distance(targets_int, sources_int)
    assert d_int.dtype == np.float64
    np.testing.assert_array_equal(d_int, r_flt)


def test_integer_input_is_not_modified():
    """Conversion makes a copy; the caller's integer array stays integer."""
    targets = np.array([[1, 2, 3]])
    sources = np.array([[0, 0, 0]])
    separation(targets, sources)
    separation_distance(targets, sources)
    assert targets.dtype.kind == "i"
    assert sources.dtype.kind == "i"


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


# --------------------------------------------------------------------------
# the distances-only path
# --------------------------------------------------------------------------


def test_separation_distance_is_the_first_output_of_separation_bit_for_bit():
    """The same expression on the same difference array: no tolerance.

    ``array_equal`` and not ``allclose`` on purpose. Both functions reach
    ``_distance`` with an array built by ``_difference``, so any difference
    in the last bit means one of them stopped going through the shared
    helpers, which is the regression this test exists for.
    """
    targets, sources = two_clouds(7, 4, seed=3)
    r_TS, _ = separation(targets, sources)
    np.testing.assert_array_equal(separation_distance(targets, sources), r_TS)


def test_separation_distance_has_one_row_per_target_and_one_column_per_source():
    targets, sources = two_clouds(7, 4, seed=4)
    d = separation_distance(targets, sources)
    assert d.shape == (7, 4)
    assert d.dtype == np.float64


def test_separation_distance_swaps_to_its_transpose():
    """A distance has no direction, so exchanging the clouds transposes it.

    The one property that separates this function from `separation`: the
    unit vectors change sign under the exchange, the distances do not.
    """
    targets, sources = two_clouds(5, 3, seed=5)
    np.testing.assert_array_equal(
        separation_distance(sources, targets), separation_distance(targets, sources).T
    )


def test_separation_distance_returns_no_unit_vectors():
    """One array back, not a tuple: the point of the function.

    A caller that unpacks two values from this must fail here rather than
    silently binding a row of distances to a name meant for directions.
    """
    targets, sources = two_clouds(3, 2, seed=6)
    result = separation_distance(targets, sources)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


def test_separation_distance_accepts_read_only_inputs():
    """`Source.positions` is frozen and is what `influence` passes here."""
    targets, sources = two_clouds(3, 2, seed=7)
    targets.flags.writeable = False
    sources.flags.writeable = False
    separation_distance(targets, sources)
    separation(targets, sources)


def test_separation_distance_peaks_at_the_difference_array():
    """Guard on the distances-only path: one (M, N, 3) temporary and no more.

    Measured ratios of peak to returned bytes, 200 by 200: 5.0 as written
    (the difference array is three matrices, the contraction one, the sqrt
    one more), 8.0 if ``np.linalg.norm`` replaces the self-contraction. The
    threshold sits between the two. Re-routing this through `separation` and
    dropping the vectors keeps the peak and doubles the time, which this
    cannot see; the docstring is what guards that.
    """
    targets, sources = two_clouds(200, 200, seed=8)

    tracemalloc.start()
    try:
        r_TS = separation_distance(targets, sources)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak / r_TS.nbytes < 6.5
