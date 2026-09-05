"""Integration test: the piston reproduces the velocity it was asked for.

Everything else in the suite checks one function against a formula. This
checks the three together against the physics, and it is the only test that
would have caught the failure the tangential offset was introduced to fix:
before ADR 0007 each function was right on its own and the mean normal
velocity over the piston came out 34 per cent low.
"""

import numpy as np
import pytest

from arfield import mesh
from arfield.influence import compute_euler_gradn_green_TS
from arfield.solver import solve_strength
from arfield.source import Source

C = 343.0
RHO = 1.2
KF = 2.0 * np.pi * 40_000.0 / C
V0 = 1.0

RADIUS = 4.95e-3
PITCH = 8.575e-3 / 20.0


def surface_samples(layer, n_radial=32, n_angular=64):
    """Points spread over the piston face, equal area per sample.

    Not the mesh points. Evaluating on the mesh points would sample every
    cell exactly where its own source is strongest, which is the bias the
    tangential offset exists to correct -- the test would then be measuring
    the correction against itself. The plane and its in-plane directions are
    read off the layer rather than assumed, so the test does not depend on
    how `mesh.circle` happens to orient its output.

    Radii follow sqrt of a uniform sequence, which is what makes each
    sample stand for the same area, so a plain mean over them is the area
    average.
    """
    normal = layer.normals[0]
    first = layer.tangents[0]
    second = np.cross(normal, first)

    center = layer.points.mean(axis=0)
    radius = np.linalg.norm(layer.points - center, axis=1).max()

    fractions = (np.arange(n_radial) + 0.5) / n_radial
    radii = radius * np.sqrt(fractions)
    angles = 2.0 * np.pi * np.arange(n_angular) / n_angular

    grid_r, grid_a = np.meshgrid(radii, angles, indexing="ij")
    offsets = (
        grid_r[..., None] * np.cos(grid_a)[..., None] * first
        + grid_r[..., None] * np.sin(grid_a)[..., None] * second
    )
    return center + offsets.reshape(-1, 3)


def solved_layer():
    m = mesh.circle(RADIUS, PITCH)
    layer = Source(m.points, m.normals, m.tangents, m.cell_area)

    matrix = compute_euler_gradn_green_TS(
        layer.collocation, layer.normals, layer.positions, KF, C, RHO
    )
    strengths = solve_strength(
        matrix, np.full(len(layer.points), V0, dtype=np.complex128)
    )
    return layer, strengths


def test_the_boundary_condition_holds_where_it_was_imposed():
    """Exactly, at the collocation points. A restatement of the solve.

    It is here to separate two failures that look the same from outside: a
    solve that did not converge, and a solve that converged onto a condition
    imposed at the wrong points.
    """
    layer, strengths = solved_layer()
    matrix = compute_euler_gradn_green_TS(
        layer.collocation, layer.normals, layer.positions, KF, C, RHO
    )
    assert matrix @ strengths == pytest.approx(
        np.full(len(layer.points), V0, dtype=np.complex128), rel=1e-8
    )


def test_the_mean_normal_velocity_over_the_face_is_the_prescribed_one():
    """The criterion of Milestone 1, measured away from the collocation points.

    The boundary condition is imposed at one point per cell; this asks what
    the surface as a whole ends up doing. The tolerance below is provisional:
    it is set wide enough to state the claim that is actually on record --
    that the solved branch is no longer tens of per cent low -- and should be
    tightened to the measured figure once HALLAZGOS_hito1.md carries one for
    this mesh and this alpha. Loosening it instead of tightening it is the
    thing to notice.
    """
    layer, strengths = solved_layer()
    samples = surface_samples(layer)
    normals = np.broadcast_to(layer.normals[0], samples.shape)

    velocity = (
        compute_euler_gradn_green_TS(samples, normals, layer.positions, KF, C, RHO)
        @ strengths
    )
    ratio = velocity.mean() / V0
    assert abs(ratio - 1.0) < 0.03, f"mean vn / v0 = {ratio}"


def test_the_strengths_are_of_the_size_the_assignment_branch_predicts():
    """Against Ec. 1.25i, A_m = -2 i omega rho v_0 dS, order of magnitude only.

    The assigned branch ignores every interaction between sources, so the two are
    not meant to agree exactly — the point of solving is that they do not. They
    agree to about six per cent on this mesh, which is close enough to pin the
    conventions: a dropped 1 / (4 pi) would show up as a factor of twelve and a
    dropped cell area as orders of magnitude.
    """
    layer, strengths = solved_layer()
    assigned = abs(2.0 * KF * C * RHO * V0 * float(layer.cell_area))
    ratio = abs(strengths).mean() / assigned
    assert 0.1 < ratio < 1.20, f"solved / assigned = {ratio}"
