# 0014 - `field` holds its tables in a class and contracts before materializing

Status: accepted
Date: 2026-09-25

Related: 0003 (frozen geometry), 0008 (assembly lives in `influence`), 0009
(signatures take point arrays), 0011 (unprojected gradient carries no
medium), 0012 (kernels chain from one evaluation of G), 0013 (`pairwise`
distance path and dtype policy)

Supersedes one paragraph of 0012: `field` no longer depends on `influence`.

## Context

Milestone 5 needs three quantities at every target of one cloud: the
pressure, the particle velocity, and the gradient of the velocity that the
Gor'kov potential differentiates. ADR 0012 put the kernel chain in
`green_kernel` and named the caller it was written for — "a field object
performs one `pairwise.separation`, keeps `r_TS` and `e_TSj`, and calls the
`_from` chain on them". That object did not exist yet. This ADR is that
object, and it records the measurement 0012 deferred.

`field` as it stood held two functions, each calling its own
`influence.compute_*_TS`, each performing its own separation and its own
exponential. Asking for the pressure and the velocity over one cloud
therefore paid the `(M, N, 3)` subtraction twice and `exp(i * kf * r)`
twice, and the velocity materialized the `(M, N, 3)` complex gradient
array, the largest the package builds. With the gradient of the velocity
added, the same pattern would have cost three separations, three
exponentials, and an `(M, N, 3, 3)` complex Hessian.

The project's criterion for a class is data that has to stay consistent
with itself after the call. Tables derived from one geometry at one
wavenumber are exactly that, and unlike `Solver.A` — the counterexample of
ADR 0008 — they are not something a caller hands in and can hand in
differently next time.

## Decision

**`Field` holds the tables; the strengths stay arguments.** The object is
constructed from `targets`, `sources`, `kf` and a `medium`, and the line
between an attribute and a method is a single question: does the quantity
depend on the strengths `A`? Tables do not, and are `cached_property`,
chaining from one `pairwise.separation` through the `_from` spellings of
`green_kernel`:

    _separation -> _r_TS, _e_TSj
    _green_TS -> _d_green_dr_TS -> _aux_h -> _aux_s

`pressure`, `velocity` and `velocity_gradient` do depend on `A`, and take it
as an argument. Nothing in the object ever stores it, for the reason ADR
0008 gives for the influence matrices.

Constructing the object computes nothing. It is a plain class, not a
dataclass: there is no equality or representation worth generating, and the
four attributes are set once and read.

**The inputs are stored, not copied, and the single chain is what makes
that safe.** Mutating `targets` or `sources` before the first table is
built takes effect; mutating after it does not. There is no mixed state,
because every table descends from one cached `_separation`. A split chain
would lose that guarantee, and the test
`test_the_inputs_are_read_once_and_then_the_geometry_is_fixed` pins it.
Arrays that come from `Source.positions` are frozen already by ADR 0003.

**The three functions remain, as wrappers.** They build a `Field`, spend it
and drop it. They keep the signatures the three `validation/` scripts and
the sixteen existing tests of `tests/test_field.py` already use, so this
change moves the implementation without touching anything that verifies it.
The module docstring names the class as the way in and the functions as the
shortcut for one answer; there is one implementation of each quantity.

**`field` stops importing `influence`.** It reaches `green_kernel` through
`pairwise`, and nothing else. This supersedes the paragraph of ADR 0012 that
described `field` as depending on both. The rule stated there still holds
and now reads cleanly: `influence` is the route for "two clouds, one table",
which is the collocation system and nothing else; `green_kernel` is the
route for "geometry already separated, several tables".

**The medium enters the constructor, and is optional.** `Medium` had no
caller in the package before this; it does now. `pressure` never looks at
it, so a `Field` built without one reports pressure and raises on the other
two, which is the shape of the physics: the pressure of known strengths is
geometry and wavenumber only. ADR 0009 and 0011 keep loose scalars in
`influence`, where the core must stay agnostic of the fluid; `field` is the
layer that spends a solved field and is entitled to hold one.

**Contract before materializing, with `einsum`, without `optimize`.** No
array carrying `N` and a spatial axis is ever built:

- `v_j`: the strengths are folded into `dG/dr` first, giving an `(M, N)`
  complex intermediate, then contracted against `e_TSj`. The `(M, N, 3)`
  complex gradient is never formed.
- `d_i v_j`: `delta_ij` leaves the sum over sources because it does not
  depend on `n`, so the first term is a matrix-vector product laid on the
  diagonal; the second folds the strengths into `s` and contracts the unit
  vectors twice in one pass. The `(M, N, 3, 3)` Hessian is never formed.

The Euler factor multiplies the contracted result, `(M, 3)` or `(M, 3, 3)`,
never anything carrying `N`.

## Alternatives considered

**Splitting the chain so that pressure takes the cheap route.** `_r_TS`
from `pairwise.separation_distance`, `_e_TSj` from `separation`, so that a
pressure-only evaluation never keeps the unit vectors. Measured at
`M = 20000`, `N = 480`, peak over baseline:

| | pressure | velocity | both |
|---|---|---|---|
| one chain | 64 B/pair | 80 B/pair | 80 B/pair, 1127 ms |
| split | 40 B/pair | 96 B/pair | 96 B/pair, 1392 ms |

Rejected: it pays the primary path to relieve the secondary one, because
the `(M, N, 3)` subtraction is then performed twice whenever both
quantities are wanted, which is every Gor'kov call.

**The same split, with the chain choosing at run time.** `_r_TS` asks
whether `_separation` is already in `__dict__` and takes the cheap route
only when it is not, and `velocity` touches `_e_TSj` before the chain so
that one separation serves both. Measured: 40 and 80, the best of both
columns, all tests passing. Rejected anyway: it works because of the order
of two statements inside one method, and reordering them for readability
would slow the object down without breaking anything or raising. That is
invisible coupling, and 24 bytes per pair on the path this object is not
built for does not buy it.

**A batched `matmul` for the velocity contraction.** BLAS does not mix real
with complex, so it promotes `e_TSj` to `complex128` and allocates the very
array the route exists to avoid: 276 MB of peak against 1 MB for the
`einsum`, at `M = 4000`, `N = 1500`, and slower as well. Worse than not
optimizing at all. The measurement of 2026-09-16 that favors `matmul` over
`einsum` was taken with two large operands of the same dtype and does not
carry over.

**`optimize=True` on the three-operand contraction of the gradient.**
Fastest of everything measured, 570 ms against 709, and it allocates 736 MB
at `M = 20000`, `N = 480`, because it splits the contraction into pairwise
steps and materializes each one. Rejected: twenty percent of time is not
worth three quarters of a gigabyte when memory is the binding constraint,
and the ratio worsens with `N`.

**The medium as arguments of the methods.** `velocity(A, c, rho)`, keeping
the object purely geometric. Rejected because `kf = omega / c` already
carries a `c`: a later call with a different `c` produces a different
`omega` in silence, and the velocity comes back at a frequency the pressure
was not computed at. Of the three failure modes available here — an
incomplete object, a mismatched `c`, and a hidden default — the mismatch is
the only one that never announces itself.

**A default medium, air at 343 m/s.** Rejected on the same ground as the
package-wide rule that `kf` is never derived from a medium or kept in a
module constant: a default returns plausible numbers for a fluid nobody
chose.

**Deleting the loose functions in this change.** Rejected for now, not on
principle. Doing it means editing `__init__.py`, the sixteen tests of
`tests/test_field.py` and the three `validation/` scripts, and those
scripts produced the Milestone 1 tolerances and run under their own
workflow. Moving the implementation and the verification in one commit
leaves no unchanged witness if a number moves. The migration and the
deletion belong to a second change, where a moved number can only have come
from the migration.

**A `Wave` type carrying `k`, `omega` and `f`.** Proposed by the author and
deferred. Two of the three are determined by the other and by `c`, so the
type as proposed can hold impossible states; the version that cannot
carries `f` alone and derives the rest. Either way it contradicts the
package-wide convention that `kf` travels explicitly, which is a change to
`influence`, `solver`, the tests and the three scripts. It is a change of
its own, with its own ADR.

## Consequences

`compute_green_TS` and `compute_grad_green_TSj` lose every production
caller. They are kept, unlike `compute_h_TS` in ADR 0012, because they are
not merely unused: in `tests/test_influence.py` they are the oracles of the
matrix the solver actually uses. `compute_grad_green_TSj` is the reference
that `compute_euler_gradn_green_TS` is checked against, and
`compute_green_TS` is what the former is differenced against. Removing them
would remove the chain of checks under the only matrix Milestone 1 depends
on.

`hessian_green` gains the caller ADR 0012 anticipated: the test that checks
the contracted route against the tabulated form.

The saving 0012 left unmeasured, at `M = 20000`, `N = 480`: pressure and
velocity over one cloud drop from 96 to 80 bytes per pair and from 1826 ms
to 1139 ms, about a third of the time. The velocity alone drops from 96 to
80 bytes per pair.

The price is on the path this object is not for: pressure alone rises from
40 to 64 bytes per pair, because it reads its distances from a `separation`
that also keeps the unit vectors. The three `validation/` scripts are that
path, and go from 366 MB to 586 MB at their working shape, with no change
in time. The class docstring states the number rather than hiding it.

The object is the unit to block over when the cloud grows. What crosses the
boundary into `gorkov` carries no `N` at all: 208 bytes per target for the
pressure, the velocity and the velocity gradient together, against
gigabytes of tables. That is why `gorkov` can remain pure algebra over
evaluated fields and need not know what a Green's function is.

There are two public doors to each quantity for as long as the wrappers
live. The precedent is deliberate elsewhere — `sklearn.preprocessing` pairs
`scale` with `StandardScaler`, `scipy.interpolate` keeps a functional and an
object-oriented spelling of the same splines — and it is survivable only
while one of the two is documented as primary. The module docstring does
that.

Equality between the contracted route and the tabulated Hessian is to
rounding, not exact: the factors are multiplied in a different order and
floating-point addition is not associative. Agreement measured at 2e-16
relative, and the test asserts `rel=1e-12`.

The trace identity `sum_i d_i v_i = -kf**2 p / (i omega rho)` is a badly
conditioned difference in the near field, where `3h` exceeds `kf**2 G` by
`3 / (kf * r)**2`: relative error 2e-16 at `kf * r = 0.9`, 1e-12 at 0.01,
1e-8 at 1e-4. The test asserts the scene sits at `kf * r` above one, which
is where the package evaluates. It is a property of the subtraction, not of
the contraction.

`ARQUITECTURA.md` still describes `field` as two loose functions with a
dependency on `influence`, and needs the card and the dependency rule
updated.
