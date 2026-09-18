"""Benchmark: the pulsating sphere against its closed form.

The first curved radiator, and the first closed surface. A sphere of radius
``a`` whose whole surface moves radially at ``v0`` radiates

    p(r) = rho c v0 a * (i k a) / (i k a - 1) * exp(i k (r - a)) / r,

which follows from ``v = grad(p) / (i omega rho)`` applied to ``A exp(ikr)
/ r`` at ``r = a`` under the ``exp(-i omega t)`` convention of ADR 0001.
The case is the one the author explored in ``sphere_spike``: water at 5 MHz,
``a = 382 um``, ``ka = 8.0``.

Three things the flat piston could not test, and one it could not show:

* **Per-node frames.** Every normal and tangent is different; a projection
  taken from the wrong end of the pair, or a retreat applied along ``z``
  instead of along the local normal, is invisible on the piston and wrong
  here.
* **Phase.** The modulus of ``p`` is a clean hyperbola and easy to hit by
  accident. All of ``ka`` lives in the phase: the factor ``ika / (ika - 1)``
  contributes exactly 45 degrees at ``ka = 1``. A reversed time convention
  gives an identical modulus curve and about 90 degrees of phase error.
* **Symmetry.** At fixed radius ``|p|`` is a constant in the continuum, so
  its spread over a shell is mesh error alone, with no closed form in the
  comparison at all.
* **The Rayleigh branch fails, and must.** ``rayleigh_strength`` carries the
  factor of two of a rigid infinite baffle, and a sphere radiating into free
  space has no baffle. The branch comes out at twice the closed form, which
  the docstring of that function predicts; asserting it here is what keeps
  the prediction honest.

Fictitious resonances. A closed source layer at radius ``a_s = a - rs`` has
interior eigenmodes at ``k a_s = x_n``, the zeros of the spherical Bessel
functions, where the influence matrix loses rank. ``ka = 8.0`` sits next to
the second zero of ``j_1`` at 7.725, and with ``alpha = 0.25`` on a lambda/6
mesh the layer lands at ``k a_s = 7.76``: ``cond(M)`` jumps to 52, against
about 7 away from resonance. The field does not notice -- the spurious mode
is dipolar, the boundary condition monopolar -- which is the docstring of
``sphere_spike`` cell C confirmed with the package. That is why this file
prints the resonance table and asserts on the field rather than on
``cond``: a conditioning threshold would fail for the wrong reason on a
mesh that is perfectly good, and pass on a ``v0`` that couples to the mode.

Run with ``pytest validation/ -m slow -s``. Figures go to ``figures/``.

References
----------
Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*, Wiley
(2007), section 1.3.3 for O'Neil's argument and the baffle hypothesis.
"""

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
# the case: sphere_spike, with the package default alpha
# --------------------------------------------------------------------------

C = 1500.0
RHO = 1000.0
FREQUENCY = 5.0e6
WAVELENGTH = C / FREQUENCY
KF = 2.0 * np.pi / WAVELENGTH

RADIUS = 382.0e-6
KA = KF * RADIUS
V0 = 1.0

ALPHA = 0.25
DENSITIES = (6, 10, 14)
BENCHMARK_DENSITY = 10

# Radial line, from just above the surface to 25 radii.
R_START = 1.001 * RADIUS
R_END = 25.0 * RADIUS
LINE_SPACING = WAVELENGTH / 41

# Shell radius for the angular spread, in units of a: far enough that no
# single node dominates, close enough that the mesh shows if it has to.
SHELL = 3.0

# Zeros of the spherical Bessel functions j_n, n = 0..4, up to about 10.
BESSEL_ZEROS = {
    "j0": (3.1416, 6.2832, 9.4248),
    "j1": (4.4934, 7.7253),
    "j2": (5.7635, 9.0950),
    "j3": (6.9879,),
    "j4": (8.1826,),
}

# --------------------------------------------------------------------------
# tolerances -- provisional, [C]. Measured 9 Sep 2026 on lambda/10, alpha
# 0.25: solved L2 1.70 per cent, phase 1.5 deg, spread 0.08 per cent,
# Rayleigh / closed form 2.05.
# --------------------------------------------------------------------------

SOLVED_TOL = 0.03  # L2 of |p| on the radial line
PHASE_TOL_DEG = 3.0  # max unwrapped phase error, referred to the first point
SPREAD_TOL = 2.0e-3  # std / mean of |p| on the shell
RAYLEIGH_RATIO = (1.9, 2.2)  # |p_rayleigh| / |p_ref|, mean over the line


# --------------------------------------------------------------------------
# closed form and metrics
# --------------------------------------------------------------------------


def pressure_pulsating_sphere(
    r: np.ndarray, a: float, v0: complex, kf: float, c: float, rho: float
) -> np.ndarray:
    ka = kf * a
    return (
        rho * c * v0 * a * (1j * ka / (1j * ka - 1.0)) * np.exp(1j * kf * (r - a)) / r
    )


def error_metrics(p_num: np.ndarray, p_ref: np.ndarray) -> dict[str, float]:
    """Relative errors of magnitude; see test_piston_axis for the names."""
    ref = np.abs(p_ref)
    err = np.abs(p_num) - ref
    return {
        "L_inf": float(np.abs(err).max() / ref.max()),
        "L_2": float(np.linalg.norm(err) / np.linalg.norm(ref)),
        "AE": float(np.abs(err).mean() / ref.max()),
    }


def phase_error_deg(p_num: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    """Unwrapped phase difference along the line, zeroed at the first point.

    Unwrap before subtracting: over 24 radii the phase advances several
    turns, and without it every 2 pi jump reads as 360 degrees of error.
    The origin of phase is arbitrary -- shifting the time origin shifts
    both curves alike -- so what is measured is the variation.
    """
    error = np.unwrap(np.angle(p_num)) - np.unwrap(np.angle(p_ref))
    return np.degrees(error - error[0])


# --------------------------------------------------------------------------
# the case, assembled
# --------------------------------------------------------------------------


def radial_line() -> tuple[np.ndarray, np.ndarray]:
    targets = mesh.line([0.0, 0.0, R_START], [0.0, 0.0, R_END], LINE_SPACING)
    return targets, np.linalg.norm(targets, axis=1)


def sphere_case(density: int, alpha: float = ALPHA) -> dict:
    surface = mesh.sphere(RADIUS, WAVELENGTH / density)
    layer = arf.Source(
        surface.points,
        surface.normals,
        surface.tangents,
        surface.cell_area,
        alpha=alpha,
    )
    targets, r = radial_line()

    matrix = arf.compute_euler_gradn_green_TS(
        layer.collocation, layer.normals, layer.positions, KF, C, RHO
    )
    a_solved = arf.solve_strength(
        matrix, np.full(len(surface), V0, dtype=np.complex128)
    )
    a_rayleigh = arf.rayleigh_strength(V0, layer.cell_areas, KF, C, RHO)

    # The mesh nodes scaled out to the shell radius: an equal-area cloud
    # that already exists.
    shell = surface.points * SHELL
    p_shell = np.abs(arf.pressure(shell, a_solved, layer.positions, KF))

    a_s = RADIUS - float(layer.rs.mean())
    return {
        "n": len(surface),
        "cond": float(np.linalg.cond(matrix)),
        "ka_s": KF * a_s,
        "r": r,
        "p_ref": pressure_pulsating_sphere(r, RADIUS, V0, KF, C, RHO),
        "p_rayleigh": arf.pressure(targets, a_rayleigh, layer.positions, KF),
        "p_solved": arf.pressure(targets, a_solved, layer.positions, KF),
        "shell_spread": float(p_shell.std() / p_shell.mean()),
        "shell_polar_deg": np.degrees(np.arccos(shell[:, 2] / (SHELL * RADIUS))),
        "shell_relative": p_shell / p_shell.mean(),
    }


@pytest.fixture(scope="module")
def benchmark() -> dict:
    return sphere_case(BENCHMARK_DENSITY)


# --------------------------------------------------------------------------
# the benchmark
# --------------------------------------------------------------------------


def test_solved_branch_matches_the_closed_form(benchmark):
    metrics = error_metrics(benchmark["p_solved"], benchmark["p_ref"])
    assert metrics["L_2"] < SOLVED_TOL, metrics


def test_the_phase_follows_the_closed_form(benchmark):
    """The one assertion here that a reversed time convention cannot pass."""
    error = phase_error_deg(benchmark["p_solved"], benchmark["p_ref"])
    assert np.abs(error).max() < PHASE_TOL_DEG, np.abs(error).max()


def test_the_field_is_spherically_symmetric(benchmark):
    """No closed form involved: the spread over a shell is mesh error alone.

    Checked against the polar angle in the figure below rather than against
    the node index, because a bias with latitude would say the z bands of
    the Fibonacci construction are not sharing the area evenly.
    """
    assert benchmark["shell_spread"] < SPREAD_TOL, benchmark["shell_spread"]


def test_the_rayleigh_branch_carries_the_baffle_it_should_not(benchmark):
    """Twice the closed form: the factor of two of the rigid baffle.

    This branch is documented as wrong for a surface radiating into free
    space, and this is the test that keeps that documentation true. The
    ratio is a little above two by ``sqrt(1 + 1 / (ka)^2)`` and by the
    sources sitting at ``a_s < a``.
    """
    ratio = float(np.mean(np.abs(benchmark["p_rayleigh"]) / np.abs(benchmark["p_ref"])))
    low, high = RAYLEIGH_RATIO
    assert low < ratio < high, ratio


def test_sphere_figures(benchmark):
    FIGURES.mkdir(exist_ok=True)
    r_over_a = benchmark["r"] / RADIUS

    fig, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(7, 9), constrained_layout=True
    )
    top.plot(r_over_a, np.abs(benchmark["p_ref"]), "-k", label="closed form")
    top.plot(r_over_a, np.abs(benchmark["p_rayleigh"]), "--b", label="Rayleigh branch")
    top.plot(r_over_a, np.abs(benchmark["p_solved"]), "-.m", label="solved branch")
    top.set_xlabel("r / a")
    top.set_ylabel("|p| (Pa)")
    top.set_title(f"Pulsating sphere, ka = {KA:.2f}, N = {benchmark['n']}, α = {ALPHA}")
    top.grid()
    top.legend()

    middle.plot(
        r_over_a, phase_error_deg(benchmark["p_solved"], benchmark["p_ref"]), "-m"
    )
    middle.set_xlabel("r / a")
    middle.set_ylabel("phase error (deg)")
    middle.grid()

    bottom.plot(benchmark["shell_polar_deg"], benchmark["shell_relative"], ".", ms=3)
    bottom.set_xlabel(f"polar angle on the shell at r = {SHELL:.0f} a (deg)")
    bottom.set_ylabel("|p| / mean")
    bottom.grid()
    fig.savefig(FIGURES / "pulsating_sphere.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# mesh density and the resonance table
# --------------------------------------------------------------------------


def test_error_against_mesh_density_and_conditioning():
    """Field error, phase, spread and cond over the mesh, with k a_s.

    The columns to read together are ``cond`` and ``k a_s``: as the mesh
    is refined ``rs`` shrinks, the source layer moves outward and ``k a_s``
    drifts past the zero of ``j_1`` at 7.725. ``cond`` falls by an order of
    magnitude while the field error barely moves. The assertion is on the
    field at every density, which is the statement that the resonance does
    not reach a monopolar boundary condition.
    """
    print("\npulsating sphere against mesh density")
    print(
        f"{'mesh':>9} {'N':>6} {'Sol L2':>7} {'Sol Linf':>9} {'phase':>7} "
        f"{'spread':>7} {'cond':>6} {'k a_s':>6}"
    )
    for density in DENSITIES:
        result = sphere_case(density)
        m = error_metrics(result["p_solved"], result["p_ref"])
        phase = np.abs(phase_error_deg(result["p_solved"], result["p_ref"])).max()
        print(
            f"lambda/{density:<2d} {result['n']:>6d} {m['L_2'] * 100:>7.2f} "
            f"{m['L_inf'] * 100:>9.2f} {phase:>6.2f}° "
            f"{result['shell_spread'] * 100:>6.3f}% {result['cond']:>6.1f} "
            f"{result['ka_s']:>6.3f}"
        )
        assert m["L_2"] < SOLVED_TOL, (density, m)

    print(f"\nfictitious resonances of the source layer, ka = {KA:.3f}:")
    for name, zeros in BESSEL_ZEROS.items():
        print(f"  {name}: {zeros}")
