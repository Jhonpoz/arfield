# ADR 0010 — The `compute_` prefix on functions that build matrices

**Status**: accepted — 2026-09-05
**Extends**: ADR 0002 (does not supersede it)

## Context

ADR 0002 named the **arrays**: `green_TS`, `grad_green_TSj`, `euler_gradn_green_TS`, with
an index suffix. It said nothing about the **functions** producing them, and writing
`influence.py` produced two identical names for different things: `green_TS` the array
and `green_TS` the function.

## Decision

- No prefix when the name **is** the mathematical object and evaluating it means applying
  a formula pointwise: `green(r, kf)`, `gradient_green(r, e_r, kf)` in `green_kernel`.
- The `compute_` prefix when there is a **construction** to name — building a matrix from
  two point clouds, with a projection and a change of variable inside — and the result
  corresponds to no named mathematical function: `compute_green_TS`,
  `compute_euler_gradn_green_TS`.

A function is named after the array it returns, with the prefix in front. The index
suffix is kept: `_TS` for `(M, N)`, `_TSj` for `(M, N, 3)`.

`assemble_` was rejected because in numerical methods "assembly" denotes a specific
operation — placing local contributions element by element into a global matrix, summing
where they overlap — which these functions do not perform: every entry is computed once,
with no accumulation. The word is reserved for the function that will place blocks once
there are several layers.

## Consequences

- `green_kernel` is unchanged.
- A function name says whether it evaluates a formula or builds a structure.
- `assemble_` stays available with its technical meaning for Milestone 4.

## Alternatives rejected

- **`assemble_`**: reserved, see above.
- **No prefix, relying on the module to disambiguate**: rejected because
  `green_kernel.green` and `influence.green_TS` are easy to confuse when reading a bare
  call.
