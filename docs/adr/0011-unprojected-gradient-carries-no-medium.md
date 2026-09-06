# 0011 — The unprojected gradient carries no medium

Status: accepted
Date: 2026-09-05

## Context

`field.velocity` and, later, `gorkov` need the Green's gradient with all three
components. The projected matrix that `influence.compute_euler_gradn_green_TS`
returns cannot serve them: it collapses the spatial axis onto a normal, and a
target sitting in the fluid has no normal to collapse onto.

ADR 0008 already assumed this function would exist. One of the three reasons it
gave for moving the assembly out of the solver was that `gorkov` needs
`grad_green_TSj` unprojected and could not reach it if assembly lived inside
`solve`. The function was never written, so `field` was about to import
`separation` from `pairwise` and call `gradient_green` itself, which would have
made three copies of the same two-line preamble across the package.

The open question was not whether to add it but what its entries should hold:
the bare kernel gradient, or the gradient already divided by `i * omega * rho`.
The projected matrix applies that Euler factor internally and asks for `c` and
`rho`, so applying it here too would make the module symmetric.

## Decision

`influence.compute_grad_green_TSj(targets, sources, kf)` returns the bare
kernel gradient, shape `(M, N, 3)`, complex128. It takes no `normals`, no `c`
and no `rho`. Whoever needs a velocity divides by `i * omega * rho` once, after
contracting with the strengths.

The name follows ADR 0002 and 0010: `compute_` because a matrix is being built
from two point clouds, and the trailing `j` because the array carries a
cartesian component axis. `euler` is absent because the array never passes
through Euler's equation, which is the whole content of that prefix.

The function is deliberately not the building block of
`compute_euler_gradn_green_TS`. That one exists precisely because it projects
pair by pair and never materializes the `(M, N, 3)` array; rewriting it to call
this one and project afterwards would triple its memory and remove its reason
to exist. The two share a preamble of two lines, which is a second repetition,
and the rule of three says to leave it.

## Alternatives considered

**Apply the Euler factor here, for symmetry with the projected matrix.**
Rejected on the naming rule of ADR 0002: the name says what the array
*contains*, and the gradient of the Green's function is a property of the
kernel, not of the fluid. A secondary argument agrees: applying the factor
after contracting is `3 * M` multiplications instead of `3 * M * N`. That one
is small and was not decisive. The cost of the decision is a real asymmetry
inside `influence` — one function hands over the system matrix with the medium
already in it, the other hands over raw material — and both docstrings say so
rather than leaving it to look like an oversight.

**Let `compute_euler_gradn_green_TS` call this function and project its
result.** Rejected on memory: measured at `M = 20000` and `N = 480` the
unprojected array is 461 MB against a 154 MB projected matrix.

## Consequences

`influence` now has two entry points for the gradient and a caller has to know
which one it wants. The rule is short: a boundary condition on a surface takes
the projected one, a field in the fluid takes this one.

Nothing enforces the contraction being written the right way round. The
recommended spelling is `grad.transpose(0, 2, 1) @ A`, which needs no
temporary; `tensordot` and the broadcasting form both allocate a copy the size
of the input, measured at 440 MB on a 461 MB array. Applying the Euler factor
before the contraction instead of after produces the identical result and the
same copy, and no test on values or on peak memory detects it, because the peak
is already reached inside this function. The docstrings state the intent; a
review is what enforces it.

The tests that cover this live in `tests/test_influence.py`. The one that ties
the two routes together projects this array by hand and compares against
`compute_euler_gradn_green_TS`, with a tolerance rather than exact equality:
the two sum three terms in different orders and disagree by a couple of ulps.
