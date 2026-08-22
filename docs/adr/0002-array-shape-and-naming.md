# 0002 - Array shapes, dtype and influence-matrix naming

Status: accepted

Related: 0001 (time convention)

## Context

Every function in this package moves complex acoustic fields between observation
points and point sources. Three properties of those arrays are cheap to fix now
and expensive to change later: their axis order, their dtype, and their names.

**Axis order.** The core operations of the method are

    p     = <pressure influence>  @ A          step 5, field evaluation
    V     = <velocity influence>  @ A          step 3, boundary condition
    solve(<velocity influence>, V)             step 4, source strengths

so the influence arrays are indexed by (observation point, source). Which of the
two comes first determines whether `@` works directly and which axis the
contraction runs along in memory. NumPy is row-major: the last axis is the
contiguous one, and a matrix-vector product contracts over it. This is inverted
with respect to MATLAB, which is column-major, so the layout that is fast there
is the slow one here.

Measured on this machine for a (4000, 4000) complex128 array: summing along the
contiguous axis takes 109 ms, along the non-contiguous axis 324 ms. Storage is
identical either way; what a wrong layout costs is bandwidth and the silent
copies that `reshape` and `ascontiguousarray` make to repair it.

**dtype.** NumPy does not protect against losing the imaginary part. Assigning a
complex array into a float one raises no exception, only

    ComplexWarning: Casting complex values to real discards the imaginary part

A `np.zeros((M, N))` without an explicit dtype is float64, so a preallocate-then-
fill pattern silently discards phase. Promotion in the other direction is
automatic and harmless (`complex64 + float64 -> complex128`), so the only real
risk is the silent one.

**Naming.** The source text (Placko & Kundu) calls the pressure influence matrix
`Q` and the velocity influence matrix `M`, and never names either of them beyond
describing what they relate. Its `M` collides with its own symbol for the number
of target points, in the same paragraph. Its lowercase indices are crossed with
respect to the uppercase counts: sources are `y_m, m = 1..N` and targets are
`x_n, n = 1..M`. Its `Q` omits the `1/(4*pi)` that this project's Green's
function carries (ADR 0001), absorbing it into the source strengths instead.
Adopting the book's letters unchanged would import all four problems.

**Solver independence.** The classical DPSM obtains the source strengths by
direct inversion, which requires the number of surface target points to equal the
number of sources. The extended method (Cheng, Lin & Qin, *Ultrasonics* 51(5),
2011) solves a least-squares problem instead, does not require that equality, and
removes a scaling step the original method needs. Whether this package will need
the extended method is an open empirical question, decided by studying the
condition number as a function of the source recoil distance. The data
conventions must not prejudge it.

## Decision

**Shapes.** With `M` observation points and `N` sources:

    (M, N)      scalar influence: one complex value per (point, source) pair
    (M, N, 3)   vector influence: cartesian components on the last axis
    (M, 3)      observation points, as a flat list
    (N, 3)      source positions, as a flat list
    (N,)        source strengths A
    (M,)        boundary-condition values, evaluated fields

Observation index first, source index second, always. The component axis is
always last.

`M == N` is a property of one particular solver, not of the method. Nothing in
the package may assume the influence matrix is square outside the function that
solves the linear system.

**dtype.** Every acoustic field array is `complex128`, stated explicitly at
creation. Never inferred, never left to a default.

**Units.** SI everywhere inside the package. No constructor accepts millimetres,
not for convenience. Conversion happens in notebooks, visibly, or not at all.

**Names.** Three influence arrays, named for what they contain:

    green_TS               (M, N)      G(R) = exp(ikR) / (4*pi*R), tabulated
    grad_green_TSj         (M, N, 3)   grad G
    euler_gradn_green_TS   (M, N)      (n . grad G) / (i*omega*rho)

The rules that generate those names:

- A name states what the array **contains**, never what it produces when
  multiplied by something else.
- `T` = target, `S` = source, in that order, mirroring the `(M, N)` shape. The
  first letter is always where the field is evaluated, the second where the
  sources are. This holds for concrete surfaces too, not only the generic case.
- `j` is the cartesian component index; its presence marks the third axis. Same
  letter the book uses in `v_j^n`.
- Words in `snake_case`, subscripts appended in uppercase. Two separation
  conventions in one identifier, deliberately: the case change marks where words
  end and indices begin, and distinguishes surface indices (`T`, `S`) from the
  component index (`j`).
- `euler` marks that the array has passed through the linearised Euler equation,
  which is the only thing that introduces `1/(i*omega*rho)`. `gradn` marks the
  projection onto the surface normal.
- Names begin lowercase so that enabling Ruff's `pep8-naming` rules later does not
  invalidate the convention. Verified: `N` is not in Ruff's default rule set, and
  `N806` fires on `G_TS` but not on `green_TS`.

**Grid boundary.** Observation points travel as a flat `(M, 3)` list through the
entire pipeline. A flat list has no axis orientation, so there are no x-rows or
z-rows to confuse. All orientation ambiguity is confined to two lines:

    pts  = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    grid = p.reshape(nx, nz)

`np.meshgrid` is always called with `indexing='ij'`, and `reshape` never takes an
`order` argument.

## Alternatives considered

**`(N, M)`, source index first.** The natural choice coming from MATLAB, where
column-major layout makes it the fast one. Rejected: `p = green_TS @ A` works
without a transpose only with the observation index first, and in row-major the
contraction then runs along contiguous memory.

**`(M, 3, N)` for the vector influence.** This shape does support `@` directly,
whereas `(M, N, 3)` does not. Rejected: it breaks the invariant that the first two
axes are always (observation, source), and it misaligns with the component-last
convention that every point array in the ecosystem uses, which would force a
transpose of the normals at every projection. The cost is small and was measured:
of the three ways to contract `(M, N, 3)` with `(N,)`, `grad.transpose(0, 2, 1) @ A`
took 84 ms, `np.einsum('mnj,n->mj', ...)` 143 ms and `np.tensordot` 251 ms, on
`(4000, 3000, 3)`. Passing `optimize=True` to `einsum` was four times slower than
leaving it out, because with two operands the contraction-order analysis costs
more than the contraction.

**`complex64`, to halve memory.** Rejected: the conditioning of the linear system
is exactly the quantity under investigation in Milestone 1. Halving the mantissa
is saving precision at the one place where it might be needed.

**The book's letters, `Q_TS` and `M_TS`.** Maximum traceability when debugging
against Chapter 1. Rejected: `M` collides with the number of target points, `Q`
carries no meaning on its own, and neither name records whether the `1/(4*pi)` or
the `1/(i*omega*rho)` is inside.

**A `mat` prefix, as in `matG_TS`.** Rejected on grounds of consistency: `mat`
states what the object *is* while `grad` states what it *contains*, two different
axes of meaning in the same position of the name, and the shape suffix already
distinguishes rank.

**Folding `1/(i*omega*rho)` out of the matrix and into the right-hand side.**
Numerically identical — `cond(cA) == cond(A)` for any scalar, and
`(dGdn/(i*omega*rho)) A = v_n` is the same system as `dGdn A = i*omega*rho*v_n`.
Rejected because it puts a piece of physics on the line that calls the solver, and
repeats it once per surface assembled, which is where a sign eventually goes
wrong.

## Consequences

`p = green_TS @ A` and `A = solve(euler_gradn_green_TS, V)` are written without
transposes or reshapes. The gradient contraction is not a `@` and must be written
as `grad_green_TSj.transpose(0, 2, 1) @ A`, or equivalently with `einsum`; the
transpose is a view and copies nothing.

Memory cost is predictable before allocation: `16*M*N` bytes for a scalar
influence array and `48*M*N` for a vector one. The package can therefore report
the cost of a problem before attempting it.

Assembly does not need the full gradient. The boundary condition is scalar, so the
array that enters the solver is `(N, N)`, not `(N, N, 3)`. Building the full
tensor and projecting afterwards costs three times the memory for an intermediate
that is immediately discarded. The `(M, N, 3)` shape exists for field evaluation
(step 5, and Gorkov's `|v|^2` in Milestone 5), not for assembly (step 3).

Because `M == N` is not assumed, swapping `np.linalg.solve` for `np.linalg.lstsq`
is a change inside one function. The condition-number study can therefore decide
between them without a refactor.

Source strengths are **not** directly comparable with values printed in the book,
because the book's `Q` omits the `1/(4*pi)` that `green_TS` carries. Pressure
fields are comparable; strengths differ by that constant factor.

The following invariants hold and are asserted by unit tests:

    green_TS.shape             == (M, N)
    grad_green_TSj.shape       == (M, N, 3)
    euler_gradn_green_TS.shape == (M, N)
    every one of them has dtype complex128, checked explicitly
    (green_TS @ A).shape       == (M,)
    np.einsum('mnj,mj->mn', grad_green_TSj, normals).shape == (M, N)

A round-trip test guards the grid boundary. It requires three ingredients, each
catching a different failure: a **non-square** grid, so a swapped axis changes the
shape; **different spacings** along each axis, so a factor applied to the wrong
axis shows up; and a field with a **known asymmetric gradient**, so a permuted
component shows up. A square, equally spaced grid hides all three. Measured with
`nx=4, nz=6, dx=1.0, dz=0.2` and `F = 3x + 7z`, whose gradient is `(3, 7)`:
`indexing='ij'` recovers 3.0 and 7.0, while the default `'xy'` gives 1.4 and 15.0.

In three dimensions the default `'xy'` is worse than a transpose: it swaps only
the first two axes, producing `(ny, nx, nz)` where `(nx, ny, nz)` was intended —
a partial permutation, which is harder to spot by eye than a full one.
