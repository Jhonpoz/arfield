# 0003 - Geometry objects are frozen and their arrays read-only

Status: accepted

Related: 0002 (array shapes and naming)

## Context

Two classes in this package hold arrays that are only correct with respect to
each other. `Mesh` holds `points`, `normal` and `tangent`: a frame. `Source`
holds `points`, `normals`, `tangents` and `cell_area`, and from them derives
`rs`, `xi` and `positions`, the places the point sources radiate from.

In both cases a field can be rebound after construction without anything
noticing. Rebinding `Mesh.points` leaves a normal describing the surface the
points used to lie on. Rebinding `Source.points` leaves `positions` pointing at
where the surface used to be. Neither raises. The solver still assembles a
matrix, the matrix is still invertible, a field still comes out, and it is
wrong.

That failure mode is the one this package is least equipped to catch. The
outputs are complex fields with no independent reference: a pressure map that is
wrong by a few per cent looks exactly like a pressure map that is right. The
project's own history says the same thing from the other direction — the 34 per
cent deficit of Milestone 1 produced no warning, no exception and no visibly
implausible number, and took a dedicated study to find. A defect that changes
the geometry underneath a correct-looking field is the same class of problem
introduced deliberately.

Writing into an array is the same failure with a different syntax.
`m.points[0, 0] = 1.0` never rebinds anything, so any protection at the
attribute level alone would miss it, and `positions` is a fresh array that no
longer shares memory with `points`, so writing into either desynchronises them
silently.

## Decision

Both classes are `@dataclass(frozen=True, eq=False)`, and every array they hold
is marked `writeable = False` in `__post_init__`.

Two levels, because they stop two different things:

    m.points = other        FrozenInstanceError    rebinding the name
    m.points[0, 0] = 1.0    ValueError             writing through it

Construction goes through `object.__setattr__`, which is the documented way for
a frozen dataclass to assign during `__post_init__`; plain assignment raises
even there.

**To change anything, build another instance.** `Mesh.rotated` and
`Mesh.translated` return new meshes rather than mutating; a parameter sweep over
`alpha` is a loop over constructions, not a setter. This is affordable because
the geometry arrays are `(K, 3)` floats, three orders of magnitude smaller than
the `(M, N)` influence arrays that dominate memory (ADR 0002).

**What is coupled and what is not** is stated per class rather than assumed.
Under a rotation, `Mesh.points`, `normal` and `tangent` all change and must
change together; `cell_area`, `exact_area` and `grid_shape` do not change at
all, and `rotated` carries them over untouched. A translation moves the points
and nothing else.

**`eq=False` is not decoration.** A generated `__eq__` on a class holding arrays
raises on comparison, because `array == array` is an array and not a bool, and
`frozen=True` would additionally advertise a `__hash__` built from those same
fields. Both are removed by disabling the generated equality.

**Validation belongs in the same place.** An invariant that construction is the
only way to establish is worth checking at construction: `Mesh.__post_init__`
rejects a non-unit normal or tangent, a tangent not perpendicular to its normal,
and a `grid_shape` that does not match the number of points. `Source` does not
validate its inputs, deliberately and documented — it takes plain arrays that
may not have come from a `Mesh` — which makes the freeze the only guarantee it
offers, and therefore worth having.

## Alternatives considered

**Mutable, with a `move()` method that updates the coupled group.** The obvious
alternative and a real one: it keeps the coupling correct for the operation the
author remembers to route through it. Rejected because it is opt-in. The
failure being prevented is precisely the assignment written without thinking
about the coupling, and a method that has to be chosen does not prevent it. It
also invites partial states, where an object is briefly inconsistent between two
statements inside `move()`.

**Properties with setters that recompute the derived fields.** Rejected for the
same reason plus a worse one: it makes rebinding look supported, so a reader
concludes the class is designed to be mutated and starts relying on it. The cost
of recomputing `positions` on every write is real but secondary.

**Freeze the attributes only, without the read-only arrays.** Half the
protection for two-thirds of the code. Rejected: `m.points[0, 0] = 1.0` is not a
contrived way to break it, it is what anyone does when adjusting a mesh by hand
in a notebook, and it is the form that leaves no trace at all.

**Read-only arrays only, without `frozen`.** Rejected symmetrically:
`m.points = other` is the more common of the two.

**Defensive copies on read instead.** Rejected: it costs a copy per access on
the arrays that are read most, and it silently discards the caller's writes
rather than refusing them, which is a quieter failure than the one being fixed.

**Nothing, with a comment.** Worth stating because it is what most numerical
code does. Rejected on the specific ground that this package's outputs cannot be
eyeballed for correctness.

## Consequences

Both freezes are one line to remove if they get in the way, and the read-only
flags are one loop. They are stated in the class docstrings as removable on
purpose: a constraint that cannot be lifted becomes something people work around
instead of arguing with.

Derived arrays are frozen too, not just the inputs. `positions` no longer shares
memory with `points`, so writing into it desynchronises the two without touching
anything the attribute-level freeze protects. A parametrised test asserts the
flag on every stored array of `Source` — points, normals, tangents, cell_area,
rs, xi, positions — so a new derived field that forgets the loop fails
immediately.

`replace()` from `dataclasses` re-runs `__post_init__`, so `rotated` and
`translated` get the validation and the read-only marking for free rather than
having to repeat them.

Equality and hashing are unavailable on both classes. Nothing in the package
needs them; tests compare arrays with `pytest.approx` on the fields.

`Medium` and `Particle` are already frozen for a different reason — they are
value objects, and freezing them is conventional. This ADR does not restate that
case; it records why the two classes holding *coupled* arrays are frozen, which
is the argument that would otherwise have to be reconstructed.

Scripts that swept a parameter by mutating an object in a loop must build one
object per value instead. This is a real change in style, and the intended one:
a sweep is then a list of independent configurations rather than one object with
a history.
