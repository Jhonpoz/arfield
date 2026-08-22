# Validation benchmarks

Physical benchmarks against analytical solutions and published references.
These are *not* unit tests: they are slow, memory-heavy, and assert convergence
against a reference rather than the behaviour of a single function.

Not run on push. Trigger manually from the Actions tab (`workflow_dispatch`),
or locally:

    pytest validation/ -m slow

Every benchmark must be marked `@pytest.mark.slow` and state its reference
(equation number, figure, or paper) in the docstring.

| Benchmark | Milestone | Reference |
|---|---|---|
| — | — | — |
