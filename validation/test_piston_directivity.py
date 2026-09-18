"""Milestone 1 benchmark: far-field directivity of a flat circular piston.

Same transducer as ``test_piston_axis``: the flat circular piston of Fig.
1.35 in Placko and Kundu (2007), 5 mm^2 at 1 MHz in water, ``ka = 5.28``.
The pressure is sampled on a half circle in the far field and normalised by
its maximum, and the pattern is compared with the closed form for a piston
in a rigid baffle,

    D(theta) = | 2 J1(ka sin theta) / (ka sin theta) |,

which at this ``ka`` has one null at 46.5 degrees and one side lobe beyond
it. A pattern with a null in it is a much better test than a pattern
without: the null position depends on the aperture the sources actually
form and the side-lobe level on the cell weights, and neither is visible on
the axis.

Where the axis benchmark shows a plateau, this one shows convergence. The
solved branch's directivity error falls with the mesh (about 1.4 to 0.9 per
cent in L2 from lambda/20 to lambda/40 at the default alpha), so this is the
benchmark that tells whether a mesh is fine enough, and the one on which the
retreat fraction ``alpha`` can be chosen: the axis cannot separate values of
alpha between 0.1 and 0.4, and this one can.

The arc radius matters. At ``10 pi a^2 / lambda`` the near-field residue at
the null is still 0.85 per cent, which would be read as a mesh error; at
``100 pi a^2 / lambda`` it is 0.13 per cent and the Rayleigh branch sits on
the closed form to 0.04 per cent in L2. The far-field condition is on the
arc, not on the mesh, and the constant below is where it is set.

Run with ``pytest validation/ -m slow -s``; the tables are the numbers the
log book records. Figures go to ``figures/``.

References
----------
Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*, Wiley
(2007), Fig. 1.35. HALLAZGOS_hito1.md, section 3, for the alpha sweep.
"""

from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from scipy.special import j1

import arfield as arf
from arfield import mesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  backend must be set first

pytestmark = pytest.mark.slow

FIGURES = Path(__file__).parent / "figures"

# --------------------------------------------------------------------------
# the case: the same piston as test_piston_axis, repeated rather than
# imported so that each benchmark reads as one file
# --------------------------------------------------------------------------

C = 1500.0
RHO = 1000.0
FREQUENCY = 1.0e6
WAVELENGTH = C / FREQUENCY
KF = 2.0 * np.pi / WAVELENGTH

AREA = 5.0e-6
RADIUS = np.sqrt(AREA / np.pi)
KA = KF * RADIUS
V0 = 1.0

ALPHA = 0.25
DENSITIES = (20, 40)

# Far-field arc. pi a^2 / lambda is the Rayleigh distance; see the module
# docstring for what the multiplier buys.
ARC_RADIUS = 100.0 * np.pi * RADIUS**2 / WAVELENGTH
ARC_STEP_DEGREES = 0.25

# Retreat fractions compared on this benchmark: the package default and the
# value at which xi* = rs, which the author proposed as a geometric rule.
ALPHAS_COMPARED = (0.25, 0.3417)

# --------------------------------------------------------------------------
# tolerances -- provisional, [C]. Measured 9 Sep 2026 on lambda/40, alpha
# 0.25: Rayleigh L2 = 0.04 per cent, solved L2 = 0.87 per cent, L_inf 1.25.
# --------------------------------------------------------------------------

RAYLEIGH_TOL = 2.0e-3  # L2
SOLVED_TOL = 0.03  # L2


# --------------------------------------------------------------------------
# closed form and metrics
# --------------------------------------------------------------------------


def directivity_baffled_piston(theta: np.ndarray, ka: float) -> np.ndarray:
    """|2 J1(x) / x| with x = ka sin(theta), and 1 on the axis."""
    x = ka * np.sin(theta)
    safe = np.where(x == 0.0, 1.0, x)
    return np.where(x == 0.0, 1.0, 2.0 * np.abs(j1(safe) / safe))


def pattern_metrics(d_num: np.ndarray, d_ref: np.ndarray, theta: np.ndarray) -> dict:
    """Errors of a normalised pattern, whole and by lobe.

    Both patterns are already normalised to one at their peak, so the
    differences are absolute. ``main`` is the lobe inside the first null and
    ``side`` everything beyond it; the two fail for different reasons, an
    aperture that is not the disc it should be and cell weights that are not
    the areas they should be.
    """
    first_null = np.arcsin(3.8317 / KA)
    inside = np.abs(theta) < first_null
    err = d_num - d_ref
    return {
        "L_2": float(np.linalg.norm(err) / np.linalg.norm(d_ref)),
        "L_inf": float(np.abs(err).max()),
        "main_L_inf": float(np.abs(err[inside]).max()),
        "side_L_inf": float(np.abs(err[~inside]).max()),
    }


# --------------------------------------------------------------------------
# the patterns
# --------------------------------------------------------------------------


def far_field_arc() -> tuple[np.ndarray, np.ndarray]:
    """Targets on the half circle in front of the piston, and their angle.

    ``mesh.arc`` measures its angle from +x towards +z, so the acoustic axis
    is at pi / 2; ``theta`` below is measured from the axis instead, which is
    what the closed form takes, and runs from -90 to +90 degrees.
    """
    targets = mesh.arc(
        [0.0, 0.0, 0.0],
        ARC_RADIUS,
        0.0,
        np.pi,
        ARC_RADIUS * np.radians(ARC_STEP_DEGREES),
    )
    theta = np.arctan2(targets[:, 0], targets[:, 2])
    return targets, theta


def patterns(density: int, alpha: float = ALPHA) -> dict:
    """Closed form, Rayleigh branch and solved branch on one mesh."""
    disc = mesh.vogel_circle(RADIUS, WAVELENGTH / density)
    layer = arf.Source(
        disc.points, disc.normals, disc.tangents, disc.cell_area, alpha=alpha
    )
    targets, theta = far_field_arc()

    matrix = arf.compute_euler_gradn_green_TS(
        layer.collocation, layer.normals, layer.positions, KF, C, RHO
    )
    a_solved = arf.solve_strength(matrix, np.full(len(disc), V0, dtype=np.complex128))
    a_rayleigh = arf.rayleigh_strength(V0, layer.cell_areas, KF, C, RHO)

    def normalised(strengths):
        p = np.abs(arf.pressure(targets, strengths, layer.positions, KF))
        return p / p.max()

    return {
        "n": len(disc),
        "cond": float(np.linalg.cond(matrix)),
        "theta": theta,
        "d_ref": directivity_baffled_piston(theta, KA),
        "d_rayleigh": normalised(a_rayleigh),
        "d_solved": normalised(a_solved),
    }


@pytest.fixture(scope="module")
def benchmark() -> dict:
    return patterns(DENSITIES[-1])


# --------------------------------------------------------------------------
# the benchmark
# --------------------------------------------------------------------------


def test_rayleigh_branch_matches_the_closed_form(benchmark):
    """The discretised baffled integral, seen from the far field.

    No system, so this is the kernel, the cell areas and the arc radius. A
    Vogel disc with equal cells gives the right aperture and the right
    weights; the residue is the finite arc.
    """
    metrics = pattern_metrics(
        benchmark["d_rayleigh"], benchmark["d_ref"], benchmark["theta"]
    )
    assert metrics["L_2"] < RAYLEIGH_TOL, metrics


def test_solved_branch_matches_the_closed_form(benchmark):
    metrics = pattern_metrics(
        benchmark["d_solved"], benchmark["d_ref"], benchmark["theta"]
    )
    assert metrics["L_2"] < SOLVED_TOL, metrics


def test_the_null_is_where_the_aperture_puts_it(benchmark):
    """The first null of the solved pattern within one sample of 46.5 deg.

    The null position is set by the aperture alone, ``sin(theta) = 3.83 /
    ka``, so a disc whose sources form a slightly larger or smaller disc
    than asked moves it. A half-degree tolerance is two arc samples.
    """
    theta = benchmark["theta"]
    positive = theta > 0
    null_ref = np.degrees(np.arcsin(3.8317 / KA))
    null_num = np.degrees(theta[positive][np.argmin(benchmark["d_solved"][positive])])
    assert abs(null_num - null_ref) < 0.5, (null_num, null_ref)


def test_directivity_figure(benchmark):
    FIGURES.mkdir(exist_ok=True)
    deg = np.degrees(benchmark["theta"])

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True, constrained_layout=True
    )
    top.plot(deg, benchmark["d_ref"], "-k", label="2 J1(x) / x")
    top.plot(deg, benchmark["d_rayleigh"], "--b", label="Rayleigh branch")
    top.plot(deg, benchmark["d_solved"], "-.m", label="solved branch")
    top.set_ylabel("|p| / max |p|")
    top.set_title(
        f"Flat piston, ka = {KA:.2f}, N = {benchmark['n']}, α = {ALPHA}, "
        f"arc at {ARC_RADIUS * 1e3:.0f} mm"
    )
    top.grid()
    top.legend()

    bottom.plot(deg, (benchmark["d_rayleigh"] - benchmark["d_ref"]) * 100, "--b")
    bottom.plot(deg, (benchmark["d_solved"] - benchmark["d_ref"]) * 100, "-.m")
    bottom.set_xlabel("θ (deg)")
    bottom.set_ylabel("error (points of 100)")
    bottom.grid()
    fig.savefig(FIGURES / "piston_directivity.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# mesh density and alpha, the two things this pattern can decide
# --------------------------------------------------------------------------


def test_error_falls_with_mesh_density_and_the_choice_of_alpha():
    """Solved-branch directivity error over mesh and retreat fraction.

    The table the choice of ``ALPHA`` rests on. On the axis every alpha in
    0.1-0.4 gives the same error because the floor is the baffle; here the
    error moves, and so far it moves the way HALLAZGOS section 3 recorded
    on a coarser square lattice: up with alpha. Whichever value the package
    settles on, this is the table to rerun.
    """
    print("\nsolved-branch directivity error, per cent of 100")
    print(
        f"{'mesh':>9} {'N':>6} {'alpha':>7} {'cond':>6} "
        f"{'L2':>6} {'Linf':>6} {'main':>6} {'side':>6}"
    )
    per_alpha: dict[float, list[float]] = {alpha: [] for alpha in ALPHAS_COMPARED}
    for density in DENSITIES:
        for alpha in ALPHAS_COMPARED:
            result = patterns(density, alpha=alpha)
            m = pattern_metrics(result["d_solved"], result["d_ref"], result["theta"])
            per_alpha[alpha].append(m["L_2"])
            print(
                f"lambda/{density:<2d} {result['n']:>6d} {alpha:>7.4f} "
                f"{result['cond']:>6.2f} "
                f"{m['L_2'] * 100:>6.2f} {m['L_inf'] * 100:>6.2f} "
                f"{m['main_L_inf'] * 100:>6.2f} {m['side_L_inf'] * 100:>6.2f}"
            )

    for alpha, errors in per_alpha.items():
        assert all(later < earlier for earlier, later in pairwise(errors)), (
            alpha,
            errors,
        )
        assert errors[-1] < SOLVED_TOL, (alpha, errors)
