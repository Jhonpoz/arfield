# ADR 0009 — `influence` functions take point arrays, not objects

**Status**: accepted — 2026-09-05
**Context**: Milestone 1

## Context

With 0008 settled, what the assembly functions receive was still open. Three forms were
considered: a `Source`, a list of `Source`, and bare arrays.

The case for the object was safety: `targets` and `normals` **must come from the same
surface**, and with bare arrays nothing enforces it. Passing the collocation points of
one surface and the normals of another yields a matrix, a strength vector and a field
that all look perfectly plausible, with no error anywhere. That is the same class of
silent failure that motivated ADR 0003.

What settled it the other way is the symmetry of the operation. In the solver the targets
**are** the collocation points of the source layer itself: the surface acting on itself.
An object on one side and an array on the other cannot express that. The operation is
between two point clouds, and the two must be allowed to be the same one.

The list of layers was rejected separately: with no coupling between surfaces in v1 (see
the revised `arfield_instrucciones.md` §5), a one-element list is machinery for a case
that does not exist. It comes back when the reflector does, wrapping these functions
rather than rewriting them.

## Decision

```python
def compute_green_TS(targets, sources, kf)                          # (M,3), (N,3)
def compute_euler_gradn_green_TS(targets, normals, sources, kf, c, rho)
```

`influence.py` imports no type from the package — not `source`, not `medium` — only
`green_kernel` and `pairwise`. Callers unpack their own `Source` and `Medium`.

The medium enters as bare `kf`, `c` and `rho`. Since `omega = kf * c`, the three are
redundant and inconsistent values are not detected; the docstring says so. This was
preferred over passing the product `omega * rho`, because `c` and `rho` are quantities
with names and units of their own and the product is not, and because a signature asking
for a product forces every call site to compute it.

## Consequences

- `M != N` is possible: targets are not tied to sources.
- The kernel is agnostic and re-exportable. A user with their own geometry can enter at
  `green`/`gradient_green`, at `separation`, or at these functions, without `Source`.
- The guarantee that `targets` and `normals` belong to the same surface is lost. It is
  documented in the docstring as unvalidated.
- The `typing.Protocol` considered for annotating the layer argument is dropped.

## Alternatives rejected

- **Taking a `Source`**: rejected because in the solver both sides are the same surface
  and the signature could not say so.
- **Taking a list of `Source`**: rejected as premature while there is no coupling.
- **Taking `medium: Medium`**: rejected because it would make `influence` depend on the
  package, against the agnostic-kernel criterion.
- **Taking `omega * rho` pre-multiplied**: rejected because the product is not a named
  quantity and it moves a multiplication into every call site.
