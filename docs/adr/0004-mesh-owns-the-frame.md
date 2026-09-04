# 0004 - The mesh owns the geometric frame, and `piston` is removed

Status: accepted

Related: 0002 (array shapes and naming), 0003 (frozen geometry)

## Context

The original module map had three stages between a shape and a linear system.
`mesh` produced a bare `(K, 3)` array of points. `Piston` wrapped those points
together with per-point normals and the area each point represents. `Source`
took those three arrays and placed the point sources.

    mesh  ->  Piston(points, normals, cell_area)  ->  Source(...)

Two things happened that made the middle stage empty.

**The normal moved into the mesh.** A path — `line`, `arc` — has no normal and
no cells; a surface has both, and they mean nothing apart from the points they
belong to. Once `mesh` returned a `Mesh` carrying `points`, `normal` and
`cell_area`, `Piston` had nothing left to hold that had not already been held
one stage earlier.

**The one-point quadrature needed a tangent as well.** Imposing the boundary
condition at a cell centre imposes the peak of the normal velocity the cell's
own source produces, not its average over the cell; the assembled system then
asks each source for too little strength, which is the 34 per cent deficit of
Milestone 1. The correction evaluates instead at a point displaced tangentially
by a distance `xi` with a closed form in `r_s` and the cell area
(HALLAZGOS_hito1.md, section 2). That is a third geometric vector, and the same
question arose for it: who owns it.

**Which tangent does not matter, and that is what makes it hard.** The tangent
plane is two-dimensional, so the valid directions form a circle. On a hexagonal
lattice the representative points form that same circle and any direction gives
the same answer, which is one reason the lattice was chosen; on a square lattice
the direction matters and choosing badly costs about 11 per cent. The
requirement is therefore not to find *the* tangent but to produce *a* tangent
for every normal, with no case that fails.

**Every closed-form recipe reading only the normal has such a case.** The usual
one is a cross product against a fixed auxiliary vector, which vanishes when the
normal aligns with it. A permutation of the normal's components fails the same
way in disguise; measured on the three supported planes, with the components
reversed:

    plane   normal       reversed     n . t
    xy      (0, 0, 1)    (1, 0, 0)    0
    yz      (1, 0, 0)    (0, 0, 1)    0
    xz      (0, 1, 0)    (0, 1, 0)    1

`xz` is the fixed point of the reversal: its normal is the middle axis and maps
to itself, so the recipe hands back the normal labelled as a tangent. That plane
is the observation plane of Milestone 1, so this is not a corner case here.

The information needed to avoid all of it already existed at construction.
`_PLANES` stores, per plane, the indices of its two in-plane axes and its
outward normal; the first index names a tangent directly. It was being
discarded, exactly as the normal used to be.

## Decision

**`mesh` owns the frame.** A `Mesh` carries `points`, `normal`, `tangent`,
`cell_area`, `exact_area` and `grid_shape`. The frame is built in `_embed`,
which every surface constructor goes through, so the three shapes cannot
disagree.

**The tangent is stored, not derived.** It is the basis vector of the plane's
first in-plane axis, built from the index `_PLANES` already holds. Nothing
downstream may depend on *which* tangent it is: the choice is one arbitrary pick
from a circle of equally valid directions, and any code whose result changes
under a different pick is wrong. Tests assert orthogonality and unit norm, never
a particular direction, except in the three `_PLANES` cases where the direction
is what the table says.

**Orthogonality is validated with an absolute tolerance, never against zero.**
Both vectors are unit, so the dot product is a direction cosine and `1e-9`
bounds the angle — the same scale as the check already in `rotated`. An exact
test passes today, when every tangent is an axis vector, and starts rejecting
valid meshes the first time one is rotated: measured, one rotation leaves a
residue of -1.3e-17 and two hundred chained rotations 6.8e-16.

**Per-node views, not per-node storage.** `Mesh.normals` and `Mesh.tangents`
return `np.broadcast_to(field, points.shape)`: a stride-zero view, no
allocation, no arithmetic, read-only. They exist as the articulation point for
curved surfaces, where `normal` and `tangent` widen from `(3,)` to `(K, 3)` and
`cell_area` from a scalar to `(K,)`. Consumers that read the properties rather
than the fields are unaffected by that change, which is the whole reason the
properties exist rather than the fields being read directly.

**`Source` takes plain arrays, not a `Mesh`.**

    Source(mesh.points, mesh.normals, mesh.tangents, mesh.cell_area)

`source` does not import `mesh`, so a caller with their own point cloud can use
the package without the mesh module, and the two can be tested in isolation.

**`piston.py` is deleted.** Its three fields are the mesh's output and `Source`
receives them explicitly, so the class would be a wrapper that unpacks what the
caller already had. `ARQUITECTURA.md` section 1 is updated: `Piston` leaves the
module map and the signature list, and the dependency line "`piston` depends on
`mesh`; `source` does not depend on `piston`" is removed.

## Alternatives considered

**Keep `piston.py` as an empty module** against a future transducer needing
something a mesh cannot express — per-element phase, apodisation. Rejected on
YAGNI: nothing needs it, an empty file is a promise the project has not made,
and adding it back later costs one commit. The counter-argument is real, which
is why this is recorded rather than assumed.

**Derive the tangent from the normal on demand,** by cross product against a
fixed auxiliary with a fallback branch for the near-aligned case. The standard
solution, and it works. Rejected for two independent reasons: the degenerate
branch needs its own test or nobody knows whether it is right, and
`m.rotated(R).tangent` would not equal `R @ m.tangent`, because the recipe
re-runs on the rotated normal and returns a different member of the same circle.
Physically irrelevant, since nothing may depend on which tangent it is, but it
breaks the expectation that rotating an object rotates its vectors and makes the
rotation test awkward to state.

**Derive it by choosing the axis in which the normal has its smallest
component.** No degenerate case and no branch, just an `argmin`. Rejected on the
second reason above alone, which is the one that survives.

**A fixed global direction, such as `(0, 1, 0)`.** Rejected: not tangent to a
curved surface at almost any point, and it fails to broadcast the moment
`cell_area` widens — measured, `np.array([0, xi, 0])` raises `ValueError:
setting an array element with a sequence` as soon as `xi` is per-node. It works
today only because every surface here is flat with uniform cells, which are the
two things most likely to change.

**Store per-node `(K, 3)` normals and tangents from the start,** rather than a
`(3,)` field plus a broadcast view. Rejected as premature: every surface this
module builds is flat, the arrays would be `K` identical rows, and the widening
is a non-breaking change for anyone reading the properties. The reverse — from
`(K, 3)` down to `(3,)` — would not be.

**Store the full frame, both in-plane tangents.** Rejected as redundant: with a
normal and one tangent the third is a cross product, and the quadrature needs
one
direction.

**Put the tangent vector in `_PLANES` as a third entry** rather than building it
from the index inside `_embed`. Defensible, and it would make the table the
whole
frame. Not adopted: the index is unpacked two lines above and one construction
from it is less to keep in sync than three more literals.

**`Source` takes a `Mesh`.** Rejected: it would make `source` depend on `mesh`
for no gain, and it would rule out a caller supplying their own point cloud —
which is the case a curved surface built elsewhere will be.

**Compute `xi` and its direction in `Solver` instead.** Rejected: once Milestone
4 couples two surfaces the collocation points are read from two places, the
diagonal block and the cross block. What is computed once per surface and read
many times belongs to the surface, which is the argument that already put
`positions` on `Source`.

## Consequences

The positional signature of `Mesh` changes: `Mesh(points, normal, cell_area,
exact_area)` becomes `Mesh(points, normal, tangent, cell_area, exact_area)`, and
`Source` gains a `tangents` argument. Calls through `circle`, `rectangle` and
`polygon` are unaffected, which is most of them: of 450 lines of mesh tests, two
lines construct a `Mesh` directly.

`Mesh.__post_init__` grows a third check on the frame and `rotated` carries a
third term. Forgetting the latter raises nothing — the mesh would keep the
tangent of its original plane alongside a rotated normal, and the collocation
points would be displaced in a direction no longer in the surface. A test
asserts
`turned.tangent == matrix @ m.tangent` for exactly that reason.

The parametrisation of a curved surface yields both tangents, the normal and the
area element from the same derivative, so a stored frame is what that
constructor
will naturally produce. A derived tangent would discard information the
parametrisation gives for free and then reinvent it with a recipe that can
degenerate.

`tangent_displacement(rs, cell_area)` is a module-level function in `source.py`,
not a method: its body does not touch `self`. That makes it testable on its own
against the closed form and against the equation it solves, which is the point —
it is the contribution the project makes to a known problem, and a formula
buried
in a `__post_init__` can be neither cited nor checked in isolation.

At the default `alpha = 0.25` the offset is `xi = 0.567 * r_c`, with `r_c` the
radius of the disc of equal area. It is not a fraction of the retreat distance:
at `alpha = 0.15` a source ends up 2.1 times `r_s` from its own surface point,
most of it tangential.

Left open, and not decided here: whether `xi` displaces the source positions or
the collocation points. `Source.positions` currently adds it, so
`positions = points - rs * normals + xi * tangents`. The frame this ADR settles
is needed either way; only the consumer of `xi` changes.
