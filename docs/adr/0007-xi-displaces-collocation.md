# 0007 - `xi` displaces the collocation points, not the sources

Status: accepted

Related: 0002 (array shapes and naming), 0003 (frozen geometry), 0004 (the
mesh owns the frame)

## Context

The one-point quadrature of `HALLAZGOS_hito1.md` section 2 is what takes the
solved branch of Milestone 1 from a 34 per cent deficit to a few per cent. A
point source at a retreat distance `rs` behind the surface does not produce the
same normal velocity everywhere on its cell: the velocity peaks directly above
the source and falls off across the cell. Imposing the boundary condition at the
cell centre therefore imposes the peak rather than the average, and the assembled
system asks every source for too little strength. The correction is to separate
the collocation point from its own source by a tangential distance `xi`, whose
closed form is in `tangent_displacement`.

**The correction fixes a separation, not a position.** What the derivation
requires is that the point where `v_n = v_0` is imposed be offset by `xi`,
tangentially, from the source that stands for that cell. Which of the two moves
is not determined by the derivation, and ADR 0004 recorded that explicitly in its
last paragraph: *"Left open, and not decided here: whether `xi` displaces the
source positions or the collocation points."*

`source.py` had to pick something to be able to run, and it picked moving the
sources:

    positions = points - rs * normals + xi * tangents

with the boundary condition still imposed on `points`. That is an
implementation, not a decision — the ADR that owns the question left it open,
and this one closes it.

**The two branches assemble the same matrix.** The separation vector between
collocation point `m` and source `n` is

    sources moved:      (p_m - p_n) + rs * n_hat - xi * t_hat
    collocation moved:  (p_m - p_n) + rs * n_hat + xi * t_hat

which differ only in the sign of `xi * t_hat`. ADR 0004 establishes that `-t_hat`
is as valid a member of the circle of tangents as `t_hat`, and that nothing
downstream may depend on which one was picked. So on a surface with a uniform
tangent — every flat surface this package builds, since `_PLANES` supplies one
tangent per plane — the choice is a re-election of tangent as far as steps 3 and
4 are concerned. Measured on the implemented version: the self separation is
`sqrt(rs**2 + xi**2)` in both branches, identical at every node.

**What does differ is where the radiating aperture is.** With a uniform tangent,
adding `xi` to `positions` translates the entire source cloud rigidly by `xi` in
the plane of the surface. The aperture is where the sources are, so the disc
being simulated is laterally offset from the disc the caller described: the axis
of an on-axis curve is not the axis of the emitting disc. At the default
`alpha = 0.25` the offset is `xi = 0.567 * r_c`, which for a lambda/7 mesh at
40 kHz is about 0.36 mm.

**And the derivation is stated about a source under the centre of its cell.**
Section 2 averages `v_n` over a cell produced by a source directly beneath it,
then asks where to evaluate in order to recover that average. Moving the source
means the quantity being averaged is no longer the quantity that was averaged.
That the matrix comes out numerically identical does not repair the mismatch; it
only says the error is not in steps 3 and 4.

## Decision

**`xi` displaces the collocation points.** `Source` stores three `(N, 3)` arrays:

    points       the physical surface, exactly as supplied, never modified
    positions    points - rs * normals            where the sources radiate from
    collocation  points + xi * tangents           where v_n = v_0 is imposed

`Solver` reads `collocation` to assemble the system and `positions` as the origin
of the field. `field` and `gorkov` read `positions` only. Nothing downstream
reads `points`.

**`points` is kept even though nothing downstream consumes it.** It is what makes
the invariant `norm(collocation - points) == xi` checkable, and that invariant is
the test of the contribution this project makes to a known problem. A formula
whose result cannot be compared against its input cannot be verified in
isolation. The array is `(N, 3)` floats — for the finest mesh used so far,
`N = 1945`, 46 kB against the 60 MB of the influence matrix for the same mesh, a
ratio of 1300.

**The collocation points stay in the surface.** `tangents` is in the tangent
plane by construction and `Mesh` validates it, so `collocation` never leaves the
surface. This is what a boundary condition on a physical surface should look
like: the velocity is imposed at points that are on the thing that has a
velocity.

## Alternatives considered

**Displace the sources, keep the boundary condition on `points`.** What the code
did until this ADR. Rejected for the rigid lateral translation of the aperture
described above. Its merit is real and worth recording: it needs no third array,
and for steps 3 and 4 it is numerically equivalent, so nothing measured so far
distinguishes the two.

**Overwrite `points` with the displaced collocation points, storing two arrays
instead of three.** Attractive because `points` is otherwise unread. Rejected on
one specific failure: `replace()` re-runs `__post_init__`, and `replace()` is the
sanctioned way to sweep `alpha` in this package (ADR 0003), so
`replace(src, alpha=0.30)` would displace an already displaced array. Nothing
raises. The same happens to anyone who feeds a `Source`'s own arrays back into
the constructor. Saving 46 kB is not worth a silent double displacement.

**Drop the `points` field entirely, keeping only `collocation` and
`positions`.** This was the author's first choice, and it removes the double
displacement risk by removing the name that invites it. Rejected on
verifiability: without `points` there is nothing to check `xi` against, and the
one-point quadrature is precisely the part of this package that needs an
independent check. Keeping the input costs one array of geometry.

**Compute the displacement in `Solver` instead of storing it.** Rejected for the
reason ADR 0004 already gives for `positions`: from Milestone 4 the collocation
points are read from two places, the diagonal block and the cross block. What is
computed once per surface and read many times belongs to the surface.

## Consequences

`Solver` reads two arrays from `Source`, `collocation` and `positions`, and does
not read `points`. This is one array more in the assembly call than the previous
design assumed.

**Milestone 1's curves do not move.** The self separation is unchanged and the
off-diagonal terms differ by a re-election of tangent, so any result already
obtained remains valid. This ADR is not a bug fix for a wrong number; it is a
decision that stops a wrong number from appearing later, on a surface where the
tangent is not uniform.

The aperture is now centred. Measured on the implemented version, with a lambda/7
disc: `positions.mean(0) - points.mean(0)` is `[0, 0, -rs]`, with no in-plane
component. Under the previous version it carried `xi` in the tangent direction.

`HALLAZGOS_hito1.md` section 2 needs no change. It says "displace the targets",
which is what this ADR adopts; it was the code that went the other way.

`alpha = 0` remains singular for solving, and for the same reason as before:
`xi = 0` too, so `collocation` coincides with `points` and the sources sit on the
surface. `tangent_displacement` warns about the division.

On a curved surface, where `tangents` varies from node to node, the two branches
stop being related by a global sign and stop assembling the same matrix. The
argument above for the aperture then applies locally rather than globally, and
this decision becomes the one that keeps each source under the centre of its own
cell. That is the case this ADR is really written for; the flat piston is where
it happens to be free.
