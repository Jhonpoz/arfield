# ADR 0008 — Matrix assembly lives in `influence.py`, as functions

**Status**: accepted — 2026-09-05
**Context**: Milestone 1

## Context

`ARQUITECTURA.md` §1 planned a `Solver` class that would hold the `Source`, assemble the
influence matrix, solve, and keep `A` in `self.A`. Locating the module before writing it
surfaced three problems:

1. `A` already travels as an argument to `field.pressure(A, ...)`, so `self.A` would be a
   second copy of the same truth with nothing keeping the two in step. That is the
   problem classes exist to prevent, introduced by the class.
2. Building matrices is not the solver's alone. `field` needs `green_TS`, and `gorkov`
   needs `grad_green_TSj` **unprojected**, all three components. With assembly inside
   `Solver`, `gorkov` cannot reach it at all, because `Solver` projects onto the normal
   and discards the spatial axis. The duplication would not be an annoyance; it would be
   mandatory.
3. Rows and columns are not symmetric. Columns are always every source; rows are the
   collocation points of the surface in `solver`, and an arbitrary `(M, 3)` cloud with
   no normals and no structure in `field`. A module serving both must leave targets free.

## Decision

Create `influence.py` holding the functions that build matrices. No classes in
`solver.py` or `influence.py`.

```
influence.py   compute_green_TS, compute_euler_gradn_green_TS
solver.py      solve_strength          (linear algebra, no physics)
field.py       pressure                (the only module that knows A exists)
```

`solver` does not depend on `influence`: it takes the matrix already built. The matrix is
needed more than once — for `cond`, for a factorization reused across right-hand sides,
and to compare against the assigned-strength branch — so building it inside would make it
unreachable.

## Consequences

- `solver.py` imports nothing from the package and is tested with matrices that are not
  acoustic at all.
- `field` and `gorkov` reach the same kernel without duplicating geometry.
- Targets are a free `(M, 3)` array, so `M != N` is possible in both modules.
- `__init__.py` stops re-exporting `Solver` and re-exports functions instead.

## Alternatives rejected

- **A `Solver` class holding `self.A`**: rejected because `A` already travels as an
  argument to `field`, and storing it creates state that can fall out of step silently.
- **Assembly inside `solver`**: rejected because `gorkov` needs the unprojected gradient
  and could not reach it.
- **Merging `solver` into `field`**: rejected because one produces `A` and the other
  consumes it; a module doing both would invite `field` to stop taking `A` as an
  argument.
- **The name `dpsm.py`** (the sketch in `arfield_instrucciones.md` §6): rejected because
  it carries no information inside a package that implements DPSM, and because it
  included the solver, which is kept separate here.
