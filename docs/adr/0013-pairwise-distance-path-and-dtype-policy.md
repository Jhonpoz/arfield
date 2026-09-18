# 0013 - `pairwise` offers a distances-only path and owns the dtype policy

Status: accepted
Date: 2026-09-17

Related: 0002 (array shapes and naming), 0009 (signatures take point arrays),
0012 (kernels chain from one G)

## Context

`influence.compute_green_TS` needs distances and nothing else, but the only
function `pairwise` offered returned distances *and* unit vectors. The unit
vectors were built, normalised in place and discarded: an unused `(M, N, 3)`
float64 array of `24 * M * N` bytes, the largest single allocation of the
call. Measured at `M = 20000`, `N = 480`, 230 MB of a 614 MB peak. The
docstring of `compute_green_TS` recorded the waste and stated that the fix
would be a distances-only path in `pairwise`, not a change in `influence`.

Writing that path exposed a second problem. `separation` refused integer
coordinates by accident — NumPy will not divide float results into an integer
array in place — and the first draft of the distances-only function, which
does not divide, accepted them silently. Two public functions of one module
disagreed on their input contract without either saying so.

A first attempt to share code had `separation` call `separation_distance`.
That rebuilt the difference array a second time inside the callee and doubled
the temporaries `separation` had been written to avoid; the memory test that
guards its peak would have caught it. The structural reason is that
`separation` needs *its own* difference array for two things, the distance
and the in-place normalisation, and no function that receives `(targets,
sources)` rather than that array can reuse it.

## Decision

Two private helpers and two public functions:

    _difference(targets, sources)   -> (M, N, 3)   the only subtraction
    _distance(R_TSj)                -> (M, N)      the only norm
    separation_distance(targets, sources) = _distance(_difference(...))
    separation(targets, sources):   builds one difference, takes its
                                    distance, normalises it in place

**`_difference` is the only place coordinates are subtracted and the only
place their dtype is decided.** Integer and float32 input are converted to
`float64` there, so both public functions behave the same and the policy has
one owner. Changing it — to refuse rather than convert, say — is one edit.

**Neither public function calls the other.** Each builds one difference
array. `separation_distance` drops it on return; `separation` keeps it and
turns it into the unit vectors. The two share the subtraction and the norm
through the helpers, not through each other.

`influence.compute_green_TS` switches to `separation_distance`. Nothing else
in the package changes its call.

## Alternatives considered

**Leave the waste, one separation routine for the whole module.** What the
`compute_green_TS` docstring chose when it was written, and the right call
at the time: the fix was not needed yet. Superseded now that the field
evaluation (ADR 0012) makes the count of difference arrays matter.

**Refuse non-float64 input explicitly, in both functions.** The stricter
contract: a caller who wrote `np.array([[0, 0, 1]])` without a decimal point
learns about it. Rejected in favour of converting because arrays assembled by
hand — in tests and notebooks, the supported case — are exactly where an
integer sneaks in, and `Source` already normalises what comes through it.
Either policy is defensible; what was not defensible was two functions with
different policies. The test `test_integer_input_is_converted_by_both_public_
functions` fixes the chosen one and is the one to invert if it ever changes.

**Convert in each public function, no `_difference` helper.** Fewer
functions, but the policy written twice, which is how the inconsistency
arose in the first place. Rejected.

**`separation` calling `separation_distance`.** Tried; rejected for the
double difference array described above.

**Compute distances without the difference array,**
`sqrt(|t|**2 + |s|**2 - 2 t . s)` through a `(M, 3) @ (3, N)` matmul. No
`(M, N, 3)` temporary at all. Deferred, not rejected: it trades the temporary
for cancellation when `r` is small against `|t|` and `|s|`, about `1e-11`
relative at this project's geometry, and needs a test against the direct
subtraction before it is trusted. It is the next lever if the memory of the
difference array ever binds.

## Consequences

`separation_distance` returns one array, not a tuple. A caller that unpacks
two values fails immediately rather than binding distances to a name meant
for directions.

The `Notes` of `separation` that said "integer input raises" now say it is
converted. `tests/test_pairwise.py` replaces the test that asserted the
`TypeError` with one that asserts both functions convert and agree bit for
bit with the float64 call.

The docstrings of `compute_green_TS` and `field.pressure` that describe the
614 MB peak and the unused unit vectors are out of date and are rewritten in
the same change: the peak of `compute_green_TS` is now the difference array
plus the returned matrix, and the sentence "the fix, on the day a machine
runs out of room, is a distances-only path" has been carried out.

The memory test on `separation` (peak below 1.5 times the returned bytes)
is unchanged and passes, which is the evidence that the helpers introduced
no temporary. A second test guards `separation_distance` at its own ratio,
measured 5.0 against 8.0 for the `np.linalg.norm` spelling.
