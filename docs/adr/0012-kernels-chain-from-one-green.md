# 0012 - Kernel derivatives chain from one evaluation of G

Status: accepted
Date: 2026-09-17

Related: 0008 (assembly lives in `influence`), 0009 (signatures take point
arrays), 0010 (`compute_` prefix), 0011 (unprojected gradient carries no
medium)

## Context

Milestone 5 needs three kernels at every target–source pair: `G` for the
pressure, `grad G` for the velocity, and the scalar `h = (dG/dr) / r` that the
Hessian is written with. The force adds `s = kf**2 G + 3 h`. All of them
contain the same `exp(i * kf * r)`.

As `green_kernel` stood, each function started from `r` and evaluated the
exponential on its own: `gradient_green` called `d_green_dr`, which called
`green`. Evaluating `G`, `grad G` and `h` on one `(M, N)` table therefore cost
three exponentials over `M * N` pairs. And each `influence.compute_*_TS`
function performed its own `pairwise.separation`, so a field evaluation that
wanted three tables paid three `(M, N, 3)` difference arrays for the same
pairs. The docstring of `separation` records that one such array is about a
quarter of the cost of a table on the machine measured, bandwidth-bound; two
spare ones are half a table of work for nothing.

Neither waste needed a measurement to decide, and this ADR records why the
project's "measure before optimising" rule did not apply. That rule is for
choices with a cost on both sides. Here one option gives the identical result
with strictly fewer operations, no extra memory and no new concept: it
dominates, and the measurement can only report how much was saved, not
whether to do it.

A Hessian for the force was derived in the notes (`gorkov_dpsm.pdf`, section
7.3) in the form `d2G/dx_i dx_j = delta_ij h - e_i e_j s`. The author's form,
with `h` and `s` reusing `G` and `dG/dr`, is the one that makes the chain
possible; the long form with `R_i R_j / R**5` does not expose it.

## Decision

**Every derived kernel in `green_kernel` has two spellings.** The plain one
takes `r` and evaluates what it needs, reads as the formula, and is the
reference tests compare against. The `_from` one receives its *immediate
predecessor in the chain* already evaluated and computes nothing twice. The
plain spelling always calls the `_from` one, so each formula is written once:

    g     = green(r, kf)
    dgdr  = d_green_dr_from(g, r, kf)
    grad  = gradient_green_from(dgdr, e_r)
    h     = aux_fun_h_from(dgdr, r)
    s     = aux_fun_s_from(g, h, kf)
    hess  = hessian_green_from(h, s, e_r)

A `_from` function receives exactly the link before it, not the origin.
`gradient_green_from` takes `dgdr`, not `g`, because a caller holding `g`
and wanting both `grad G` and `h` would otherwise evaluate `dG/dr` twice.

**The field evaluation obtains its tables from `green_kernel`, not from
`influence`.** A field object performs one `pairwise.separation`, keeps `r_TS`
and `e_TSj`, and calls the `_from` chain on them: one difference array, one
exponential, every table. `influence` keeps its role of "two clouds in, one
table out" for the collocation system, where one matrix is built once and
handed to the solver; it does not gain a `_from` variant because nothing
calls it twice on the same pairs.

**Two scalars are named by letter.** `h` and `s` have no name in the
literature. The module docstring defines them once, `notacion.md` carries the
entry, and the code uses the same letters as the derivation. A name that
spells the formula (`k2g_plus_3dgdror`) was tried and rejected as unreadable;
a name that says the role (`grad_scale`, `hess_scale`) is acceptable and was
not chosen. The prefix `aux_fun_` is the author's; the letter is what
matters.

**Plain spellings with no caller are still kept for `h`, `s` and the
Hessian.** They are the definitions the tests check against, and the
Hessian's plain form is what a single-point evaluation (Marzo-style phase
optimisation at one target, `M = 1`) will use. They are one line each.

**Function order in the module follows the chain, not the alphabet.** Each
function appears after the ones it calls; `__all__` stays alphabetical.

## Alternatives considered

**An optional argument, `d_green_dr(r, kf, g=None)`, computing `g` when not
given.** One function per formula and no new names. Rejected because it puts
a branch in every kernel and makes the signature say something a reader of the
formula does not expect; the two-spelling form keeps each function one
expression.

**A `Green` class in `green_kernel` holding `g`, `r`, `e_r` and offering the
derivatives as methods.** Proposed by the author, and it satisfies the
project's own criterion for a class: data that must stay consistent. Rejected
because that object already exists — the field-evaluation class of `field`
holds `r_TS`, `e_TSj` and the tables — and a second custodian of the same
exponential is the `Solver.A` problem of ADR 0008 again. State stays in
`field`; `green_kernel` stays a module of functions.

**`influence` functions taking `r_TS` instead of `(targets, sources)`.**
Rejected: it breaks the signature of ADR 0009 that the collocation system
relies on, for a caller that does not need it.

**Functions "from tables" in `influence`, e.g. `compute_h_TS_from_green`.**
Considered as the place for derived tables on the ground that `influence`
knows the Green's function and `field` does not. Superseded by the decision
above: the chain lives in `green_kernel`, which is the layer that already
knows the formulas, and `field` calls it directly. `field` still implements
no formula of its own; it only chooses which kernel goes with which table.

**Accepting the recomputation.** The option first recommended, on the ground
that the separation was a small fraction of the cost. Withdrawn once the
count was done: three separations for three tables, each about a quarter of a
table, on a bandwidth-bound machine.

## Consequences

`green_kernel` grows from three functions to eleven, all one line. The
number is the cost of the decision and is not hidden.

`field` gains a dependency on `green_kernel` next to the one it already has
on `influence`. The rule is stated in ARQUITECTURA.md: `influence` is the
route for "two clouds, one table"; `green_kernel` is the route for "geometry
already separated, several tables".

`compute_h_TS`, briefly written in `influence`, is removed: the collocation
system does not use `h`, so it had no caller.

The tests in `tests/test_green_kernel.py` assert exact equality, not
tolerance, between the chained values and the plain spellings, because both
paths execute the same expressions on the same inputs. A `_from` that quietly
recomputed something on a different path would break the last bits and fail.

The saving is not yet measured. The docstring of `separation` and the notes of
2026-09-08 give the estimate; the log book records the number when the field
class is written. The decision does not depend on it.
