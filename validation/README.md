# Validation benchmarks

Physical benchmarks against closed-form solutions and published references.
They answer a different question from `tests/`: a unit test asks whether the
code changed, a benchmark asks whether the method reproduces the physics on
the case that matters. The two are kept apart because they fail for
different reasons and are read by different people.

| | `tests/` | `validation/` |
|---|---|---|
| question | did the code change? | does the method reproduce the physics? |
| case | small, seconds | the real case, the real mesh, minutes |
| reference | a formula or a stored array | a closed form from the book or a paper |
| runs | every push, `pytest` | before a tag, by hand |
| marker | none | `@pytest.mark.slow` |

There is no numerical relation between the small case of a unit test and the
real case of a benchmark. A tolerance measured here does not transfer there.

## Running

Not run on push. `pytest` alone collects `tests/` only (`testpaths` in
`pyproject.toml`); the benchmarks are reached by naming the directory:

    pytest validation/ -m slow -s

The `-s` matters. Every benchmark prints the tables the log book records,
and pytest swallows `print` unless told not to. Figures are written to
`validation/figures/`; that directory is git-ignored except for this note,
and the CI workflow uploads it as an artifact.

The whole set takes about five minutes on one core. Memory peaks at the
finest piston mesh, 4106 sources, well under a gigabyte.

## Reading a benchmark

Each one plots three curves, not two: the closed form, the solved DPSM
branch and the discretised Rayleigh branch. The disagreement pattern says
where a fault is. A solved curve that leaves the other two blames the
linear system; two numerical curves that leave the closed form together
blame the kernel or the mesh.

Tolerances are provisional and marked `[C]` in each file, with the value
measured when the file was written in a comment next to them. The rule is
that they get tightened to the figure the log book records, never loosened;
a benchmark that starts failing is a finding, not a tolerance to move.

Two floors are expected and asserted as plateaus, not as convergence:

- The solved branch of the flat piston sits 2 to 3 per cent from Eq.
  (1.16b) at every mesh density. The closed form is for a piston in a rigid
  baffle and a free layer of monopoles has none; see `HALLAZGOS_hito1.md`,
  section 7.
- The Rayleigh branch on the pulsating sphere is twice the closed form,
  because `rayleigh_strength` carries the baffle's factor of two by design.
  The benchmark asserts that it fails.

## Benchmarks

| File | Milestone | Case | Reference | Asserts |
|---|---|---|---|---|
| `test_piston_axis.py` | 1 | flat piston, 5 mm², 1 MHz, water | Eq. (1.16b), Fig. 1.35 | Rayleigh identity on the source plane; solved plateau; error vs mesh table; cond vs alpha |
| `test_piston_directivity.py` | 1 | same piston, far-field arc at 100 πa²/λ | 2 J₁(ka sinθ) / (ka sinθ) | both branches; null position; error falls with mesh; alpha comparison |
| `test_pulsating_sphere.py` | C | sphere, a = 382 µm, 5 MHz, water, ka = 8 | radial closed form | modulus, phase, angular spread; Rayleigh at 2×; cond and fictitious resonances printed |

## Adding one

- Mark it `slow` at module level: `pytestmark = pytest.mark.slow`.
- State the reference in the module docstring: equation, figure or paper.
- Compute the closed form in the file, by hand. A benchmark that imports its
  reference from the package is testing the package against itself.
- Put the case in named constants at the top, tolerances last, each with the
  measured value that justified it and the date.
- Print the table the log book will need, and save a figure.
- Add a row to the table above.
