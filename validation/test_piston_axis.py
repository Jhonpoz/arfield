"""Milestone 1 benchmark: pressure on the axis of a flat circular piston.

Reproduces the top-left panel of Fig. 1.35 in Placko and Kundu (2007): a flat
circular transducer of 5 mm^2 driven at 1 MHz in water, pressure along the
axis out to 6 mm, DPSM against the closed form of Eq. (1.16b). The book uses
259 point sources; here the mesh is refined until the numbers stop moving,
which is the point of the density table below.

Three curves, not two. The discretised Rayleigh integral, the solved DPSM
system and the closed form are plotted together because the disagreement
pattern says where a fault is: a solved curve that leaves the other two
blames the linear system; two numerical curves that leave the closed form
together blame the kernel or the mesh.

Each numerical branch is compared with the closed form on its own plane.
The Rayleigh branch radiates from the retreated layer at ``z = -rs`` and is
the one-point quadrature of the Rayleigh integral over that plane, so
against Eq. (1.16b) shifted by ``rs`` its error is the quadrature error and
nothing else: measured 0.12, 0.056 and 0.028 per cent in L2 from lambda/10
to lambda/40, falling as the cell count, not an identity. The solved branch
imposes ``v_n = v0`` on the surface at ``z = 0``, so it is compared with the
closed form there. Comparing both against the same plane
folds the retreat distance into one of the two errors and hides what the
other one is doing.

What the solved branch is expected to do is *not* converge to zero. Eq.
(1.16b) is the field of a piston in a rigid infinite baffle, and a free layer
of monopoles radiates without one. HALLAZGOS_hito1.md, section 7, records a
floor of a few per cent that does not move with the mesh and attributes it
to that difference of boundary condition; the measurements taken to set the
tolerances below reproduce it (2.3 to 2.6 per cent in L2 from lambda/10 to
lambda/40, flat). The tolerance is therefore a ceiling on a plateau, not a
convergence target, and the density table is what shows the plateau.

Run with ``pytest validation/ -m slow -s``. The ``-s`` is what lets the
tables reach the terminal; they are the numbers the log book records.
Figures are written next to this file under ``figures/``.

References
----------
Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*, Wiley
(2007), Eq. (1.16b), Fig. 1.35, Example 1.3.4.
"""

from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np
import pytest

import arfield as arf
from arfield import mesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  backend must be set first

pytestmark = pytest.mark.slow

FIGURES = Path(__file__).parent / "figures"

# --------------------------------------------------------------------------
# the case: Fig. 1.35, top left, and Example 1.3.4
# --------------------------------------------------------------------------

# Water, as in chapter 1. Example 1.3.4 works with lambda = 1.5 mm at 1 MHz,
# which fixes c; rho is not stated and only scales the amplitude, which every
# metric below divides out.
C = 1500.0
RHO = 1000.0
FREQUENCY = 1.0e6
WAVELENGTH = C / FREQUENCY
KF = 2.0 * np.pi / WAVELENGTH

AREA = 5.0e-6  # m^2, the flat transducers of Fig. 1.33
RADIUS = np.sqrt(AREA / np.pi)
V0 = 1.0

Z_MAX = 6.0e-3  # the book plots the axis out to 6 mm
LINE_SPACING = WAVELENGTH / 40

# Retreat distance as a fraction of the cell size. HALLAZGOS_hito1.md,
# section 3, puts the useful range at 0.2-0.3 with 0.25 as the starting
# value; the sweep below is what justifies it on this geometry.
ALPHA = 0.25

# Mesh densities for the table, as fractions of the wavelength. The finest
# one is the benchmark; the coarser two exist to show the trend.
DENSITIES = (10, 20, 40)

# Retreat fractions for the conditioning sweep, run on the middle mesh: cond
# does not move with the mesh once alpha fixes rs / pitch, and the SVD on the
# finest one would take most of the run time for nothing.
ALPHAS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80)
SWEEP_DENSITY = 20

# --------------------------------------------------------------------------
# tolerances -- provisional, [C], to be replaced by the figures the log book
# records for this case. Measured on 9 Sep 2026 while writing this file and
# reproduced on 18 Sep 2026: Rayleigh vs shifted closed form, L2 = 0.12 /
# 0.056 / 0.028 per cent from lambda/10 to lambda/40 (a quadrature, falling
# with the mesh); solved vs closed form, L2 = 2.6 / 2.4 / 2.3 per cent;
# cond = 2.0 / 2.1 / 2.1. RAYLEIGH_TOL is checked on the finest mesh.
# --------------------------------------------------------------------------

RAYLEIGH_TOL = 1.0e-3  # L2, against the closed form on the source plane
SOLVED_TOL = 0.05  # L2, against the closed form on the surface plane
COND_TOL = 5.0  # at ALPHA, on the benchmark mesh


# --------------------------------------------------------------------------
# closed form and metrics
# --------------------------------------------------------------------------


def axial_pressure_baffled_piston(
    z: np.ndarray, a: float, v0: complex, kf: float, c: float, rho: float
) -> np.ndarray:
    """On-axis pressure of a circular piston in a rigid baffle, Eq. (1.16b)."""
    return rho * c * v0 * (np.exp(1j * kf * z) - np.exp(1j * kf * np.hypot(z, a)))


def error_metrics(p_num: np.ndarray, p_ref: np.ndarray) -> dict[str, float]:
    """Relative errors of magnitude between two fields on the same points.

    Magnitudes only: a global phase is a shift of the time origin and is not
    observable. Every entry is normalised by the reference, so none depends
    on how many points the line was sampled with.

    ``L_inf`` is the worst point, ``L_2`` the whole curve, ``AE`` the mean
    absolute error over the peak, which is the figure Cheng et al. (2011)
    report and the one HALLAZGOS_hito1.md quotes as "error en el eje".
    """
    ref = np.abs(p_ref)
    err = np.abs(p_num) - ref
    return {
        "L_inf": float(np.abs(err).max() / ref.max()),
        "L_2": float(np.linalg.norm(err) / np.linalg.norm(ref)),
        "AE": float(np.abs(err).mean() / ref.max()),
    }


# --------------------------------------------------------------------------
# the three curves
# --------------------------------------------------------------------------


def axis_line() -> np.ndarray:
    """Targets on the axis, from the surface plane out to Z_MAX."""
    return mesh.line([0.0, 0.0, 0.0], [0.0, 0.0, Z_MAX], LINE_SPACING)


def three_curves(density: int, alpha: float = ALPHA) -> dict:
    """Closed form, Rayleigh branch and solved branch on one mesh.

    The disc is a Vogel spiral rather than the hexagonal lattice: its cells
    are exactly equal in area and the meshed area is exactly the disc area,
    which is what makes the error monotone in the mesh (HALLAZGOS_hito1.md,
    section 7). A lattice clipped to a circle keeps whole border cells and
    the effective radius jumps with the count.
    """
    disc = mesh.vogel_circle(RADIUS, WAVELENGTH / density)
    layer = arf.Source(
        disc.points, disc.normals, disc.tangents, disc.cell_area, alpha=alpha
    )
    targets = axis_line()
    z = targets[:, 2]

    matrix = arf.compute_euler_gradn_green_TS(
        layer.collocation, layer.normals, layer.positions, KF, C, RHO
    )
    v0 = np.full(len(disc), V0, dtype=np.complex128)
    a_solved = arf.solve_strength(matrix, v0)
    a_rayleigh = arf.rayleigh_strength(V0, layer.cell_areas, KF, C, RHO)

    rs = float(layer.rs)
    return {
        "n": len(disc),
        "rs": rs,
        "cond": float(np.linalg.cond(matrix)),
        "z": z,
        # Surface plane, where the boundary condition is imposed.
        "p_ref": axial_pressure_baffled_piston(z, RADIUS, V0, KF, C, RHO),
        # Source plane, where the Rayleigh sum actually integrates.
        "p_ref_shifted": axial_pressure_baffled_piston(z + rs, RADIUS, V0, KF, C, RHO),
        "p_rayleigh": arf.pressure(targets, a_rayleigh, layer.positions, KF),
        "p_solved": arf.pressure(targets, a_solved, layer.positions, KF),
        "strength_ratio": float(np.abs(a_solved).mean() / np.abs(a_rayleigh).mean()),
    }


@pytest.fixture(scope="module")
def benchmark() -> dict:
    """The finest mesh, computed once and shared by the tests below."""
    return three_curves(DENSITIES[-1])


@pytest.fixture(scope="module")
def density_table() -> list[dict]:
    """One row per density, the finest one being the benchmark itself."""
    rows = []
    for density in DENSITIES:
        result = three_curves(density)
        rows.append(
            {
                "density": density,
                "n": result["n"],
                "cond": result["cond"],
                "strength_ratio": result["strength_ratio"],
                "rayleigh_on_source_plane": error_metrics(
                    result["p_rayleigh"], result["p_ref_shifted"]
                ),
                "rayleigh_on_surface_plane": error_metrics(
                    result["p_rayleigh"], result["p_ref"]
                ),
                "solved": error_metrics(result["p_solved"], result["p_ref"]),
            }
        )
    return rows


# --------------------------------------------------------------------------
# the benchmark
# --------------------------------------------------------------------------


def test_rayleigh_branch_is_the_rayleigh_integral(benchmark):
    """Against the closed form on the plane of the sources: an identity.

    This is the curve that has to be exact. It contains no linear system, so
    the only things it can get wrong are the kernel, the cell area and the
    factor of two of the baffle -- and each of those is a factor, not a
    shape, so a tight L2 catches all three.
    """
    metrics = error_metrics(benchmark["p_rayleigh"], benchmark["p_ref_shifted"])
    assert metrics["L_2"] < RAYLEIGH_TOL, metrics


def test_solved_branch_stays_within_the_recorded_floor(benchmark):
    """Against the closed form on the surface plane, within the plateau.

    Not a convergence test: see the module docstring. What this asserts is
    that the solved curve has not left the other two, which is the failure
    that blames the linear system.
    """
    metrics = error_metrics(benchmark["p_solved"], benchmark["p_ref"])
    assert metrics["L_2"] < SOLVED_TOL, metrics


def test_the_two_branches_agree_in_mean_strength(benchmark):
    """Solved over assigned, mean |A|: about 1.05 on this geometry.

    HALLAZGOS_hito1.md, section 2, records 1.064 for the air piston. The
    ratio pins the conventions the field tests cannot see in isolation: a
    dropped 1 / (4 pi) would put it at 12.6 and a dropped cell area at
    orders of magnitude.
    """
    assert 0.9 < benchmark["strength_ratio"] < 1.2, benchmark["strength_ratio"]


def test_the_system_is_well_conditioned(benchmark):
    assert benchmark["cond"] < COND_TOL, benchmark["cond"]


def test_the_three_curves_figure(benchmark):
    """Fig. 1.35, top left, redrawn. Visual; the asserts are above."""
    FIGURES.mkdir(exist_ok=True)
    z_mm = benchmark["z"] * 1e3

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(z_mm, np.abs(benchmark["p_ref"]), "-k", label="Eq. (1.16b)")
    ax.plot(z_mm, np.abs(benchmark["p_rayleigh"]), "--b", label="Rayleigh branch")
    ax.plot(z_mm, np.abs(benchmark["p_solved"]), "-.m", label="solved branch")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("|p| (Pa)")
    ax.set_title(
        f"Flat piston, S = {AREA * 1e6:.0f} mm², {FREQUENCY / 1e6:.0f} MHz, "
        f"N = {benchmark['n']}, α = {ALPHA}"
    )
    ax.grid()
    ax.legend()
    fig.savefig(FIGURES / "piston_axis_three_curves.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# error against mesh density
# --------------------------------------------------------------------------


def test_error_against_mesh_density(density_table):
    """The table of section 5 of the project instructions.

    What it has to show: the Rayleigh error on the surface plane falls with
    the mesh, because it is the retreat distance and nothing else; the
    solved error does not, because it is the baffle. An error that stalls
    is qualitatively different from a large one that falls, and this is the
    table that tells them apart.
    """
    print("\nerror against mesh density, per cent")
    print(
        f"{'mesh':>8} {'N':>6} {'cond':>6} {'A_s/A_r':>8} "
        f"{'Ray L2*':>8} {'Ray L2':>8} {'Ray Linf':>9} "
        f"{'Sol L2':>8} {'Sol Linf':>9} {'Sol AE':>8}"
    )
    for row in density_table:
        print(
            f"lambda/{row['density']:<2d} {row['n']:>6d} {row['cond']:>6.2f} "
            f"{row['strength_ratio']:>8.3f} "
            f"{row['rayleigh_on_source_plane']['L_2'] * 100:>8.4f} "
            f"{row['rayleigh_on_surface_plane']['L_2'] * 100:>8.3f} "
            f"{row['rayleigh_on_surface_plane']['L_inf'] * 100:>9.3f} "
            f"{row['solved']['L_2'] * 100:>8.2f} "
            f"{row['solved']['L_inf'] * 100:>9.2f} "
            f"{row['solved']['AE'] * 100:>8.2f}"
        )
    print("  * against the closed form on the source plane, z + rs")

    rayleigh = [r["rayleigh_on_surface_plane"]["L_2"] for r in density_table]
    solved = [r["solved"]["L_2"] for r in density_table]

    assert all(later < earlier for earlier, later in pairwise(rayleigh))
    assert max(solved) < SOLVED_TOL, solved

    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    densities = [r["density"] for r in density_table]
    ax.loglog(densities, np.array(rayleigh) * 100, "o--b", label="Rayleigh branch")
    ax.loglog(densities, np.array(solved) * 100, "s-.m", label="solved branch")
    ax.set_xlabel("nodes per wavelength")
    ax.set_ylabel("L2 error on the axis (%)")
    ax.set_xticks(densities)
    ax.set_xticklabels([f"λ/{d}" for d in densities])
    ax.grid(which="both")
    ax.legend()
    fig.savefig(FIGURES / "piston_axis_mesh_density.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# conditioning against the retreat distance
# --------------------------------------------------------------------------


def test_conditioning_against_alpha():
    """cond(M) as a function of rs, the "fine point" of section 5.

    HALLAZGOS_hito1.md, section 3, records that both the conditioning and
    the field error grow monotonically with alpha, and that below about 0.1
    the mean velocity stays right while individual strengths run away. The
    sweep reproduces the first two columns of that table on the present
    case; the assertions are the monotonicity and the value at the default.
    """
    print(f"\nconditioning against alpha, lambda/{SWEEP_DENSITY} mesh")
    print(f"{'alpha':>6} {'cond':>8} {'Sol L2':>8} {'Sol Linf':>9}")
    conds = []
    for alpha in ALPHAS:
        result = three_curves(SWEEP_DENSITY, alpha=alpha)
        metrics = error_metrics(result["p_solved"], result["p_ref"])
        conds.append(result["cond"])
        print(
            f"{alpha:>6.2f} {result['cond']:>8.2f} "
            f"{metrics['L_2'] * 100:>8.2f} {metrics['L_inf'] * 100:>9.2f}"
        )

    assert all(later > earlier for earlier, later in pairwise(conds))
    assert conds[ALPHAS.index(ALPHA)] < COND_TOL
