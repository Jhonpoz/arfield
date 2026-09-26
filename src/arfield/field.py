"""Field evaluation: pressure and velocity from known source strengths.

Step five of the DPSM, and the last one. Everything upstream builds
geometry, assembles matrices and solves for the strengths ``A``; this
module spends them. It is the only module in the package that knows ``A``
exists, which is why nothing here is reusable by the assembly code and
nothing there depends on this.

Targets are an arbitrary cloud of points ``(M, 3)``, never a grid. A flat
list has no axis order to get wrong, so the reshape between a grid and a
list lives in the script that plots, in one line each way, and never in
here. Gor'kov, array design and the spherical-harmonic bridge all hang off
this step.

`Field` is the way in. It holds the tables that depend on the geometry and
``kf`` but not on ``A``, so a second quantity over the same targets costs
only its own contraction. `pressure`, `velocity` and `velocity_gradient`
remain as functions for callers that want one answer and nothing else; they
build a `Field`, spend it and drop it.
"""

from functools import cached_property

import numpy as np

from .green_kernel import aux_fun_h_from, aux_fun_s_from, d_green_dr_from, green
from .medium import Medium
from .pairwise import separation

__all__ = ["Field", "pressure", "velocity", "velocity_gradient"]


class Field:
    """Tables of a target cloud against a source cloud, at one wavenumber.

    Holds everything that depends on the geometry and ``kf`` but not on the
    source strengths, so that a second evaluation over the same targets
    costs nothing extra. The tables are built on first use and chain from
    one another: one separation, one exponential, and every kernel derived
    from that (ADR 0012). Constructing the object computes nothing.

    The line between an attribute and a method is whether the quantity
    depends on the strengths ``A``. Tables do not, and are cached.
    `pressure`, `velocity` and `velocity_gradient` do, and take ``A`` as an
    argument: nothing here ever stores it, for the reason ADR 0008 gives for
    the influence matrices.

    Parameters
    ----------
    targets : ndarray, shape (M, 3)
        Points where the field is evaluated, in meters. Any cloud: a scan
        line, a flattened grid, a single point. ``M`` need not equal ``N``.
    sources : ndarray, shape (N, 3)
        Point-source positions, in meters. For a surface these are
        ``Source.positions``, the retreated layer, and they must be the same
        array the strengths were solved against. Passing the surface points
        instead changes the geometry that ``A`` was computed for, and the
        result is a plausible-looking field that means nothing.
    kf : float
        Wavenumber of the fluid, ``2 * pi / wavelength``, in rad/m.
    medium : Medium, optional
        Fluid the field propagates in. `velocity` and `velocity_gradient`
        need it; `pressure` never looks at it. Omitting it gives an object
        that reports pressure and raises on the other two, which is the
        honest shape of the physics: the pressure of known strengths is
        geometry and wavenumber only.

    Notes
    -----
    Measured peak over the baseline, ``VmHWM``, at ``M = 20000`` and
    ``N = 480``: 64 bytes per pair for pressure alone, 80 for velocity, and
    80 again for both from the same object. The loose functions cost about
    40 and 96 for the same two answers, 136 together, and share nothing
    between them. So the object is the worse choice for one pressure
    evaluation and the better one from the second quantity onwards, which is
    every case Gor'kov brings.

    The 64 against 40 is the price of one chain: `pressure` needs only the
    distances, but it reads them from `separation`, which also returns the
    unit vectors and keeps them. Splitting the chain so that pressure takes
    the cheap route was measured and rejected: it drops pressure to 40 but
    raises velocity from 80 to 88 and costs a third more time, because the
    ``(M, N, 3)`` subtraction is then done twice. Paying 24 bytes per pair on
    the secondary path is better than taxing the primary one.

    The tables live as long as the object, so the object is also the unit to
    block over when the cloud grows: build one per slab of targets, spend it,
    drop it. What leaves it carries no ``N`` and is small, 208 bytes per
    target for pressure, velocity and the velocity gradient together.

    The inputs are stored, not copied. Mutating ``targets`` or ``sources``
    after a table has been built leaves that table describing the old
    geometry and every later one describing the new, with nothing to signal
    it. Pass arrays you are done with, or `Source.positions`, which ADR 0003
    already froze.

    Examples
    --------
    >>> field = Field(targets, layer.positions, KF, Medium(C, RHO))
    >>> p = field.pressure(strengths)
    >>> v = field.velocity(strengths)
    >>> dv = field.velocity_gradient(strengths)

    See Also
    --------
    pressure, velocity, velocity_gradient : one-shot functions over the same
        machinery.
    """

    def __init__(
        self,
        targets: np.ndarray,
        sources: np.ndarray,
        kf: float,
        medium: Medium | None = None,
    ):
        self.targets = targets
        self.sources = sources
        self.kf = kf
        self.medium = medium

    @cached_property
    def _separation(self):
        """Distance and unit vector of every pair, computed once.

        The only place this module touches coordinates. Everything else
        works from what comes out of here.
        """
        return separation(self.targets, self.sources)

    @property
    def _r_TS(self):
        """``|x_T - y_S|``, ``(M, N)`` float64, in meters."""
        return self._separation[0]

    @property
    def _e_TSj(self):
        """Unit vector from source to target, ``(M, N, 3)`` float64.

        Source to target, not the reverse: that direction is what makes the
        gradient the one with respect to the target coordinate, which is the
        one the field evaluation needs.
        """
        return self._separation[1]

    @cached_property
    def _green_TS(self):
        """``G(r)`` for every pair, ``(M, N)`` complex128.

        The only exponential in the object. Every other kernel is derived
        from this one without calling ``exp`` again.
        """
        return green(self._r_TS, self.kf)

    @cached_property
    def _d_green_dr_TS(self):
        """``dG/dr`` for every pair, ``(M, N)`` complex128."""
        return d_green_dr_from(self._green_TS, self._r_TS, self.kf)

    @cached_property
    def _aux_h(self):
        """``(dG/dr) / r`` for every pair, ``(M, N)`` complex128.

        The scalar that makes ``grad G = R * h``, called by letter here and
        in the derivation notes because it has no name in the literature.
        """
        return aux_fun_h_from(self._d_green_dr_TS, self._r_TS)

    @cached_property
    def _aux_s(self):
        """``kf**2 * G + 3 * h`` for every pair, ``(M, N)`` complex128.

        The second scalar of the Hessian, ``d_i d_j G = delta_ij * h -
        e_i * e_j * s``.
        """
        return aux_fun_s_from(self._green_TS, self._aux_h, self.kf)

    def pressure(self, source_strengths: np.ndarray) -> np.ndarray:
        """Acoustic pressure at every target point.

        Sums the contribution of every point source: ``p = green_TS @ A``.

        Needs no medium. Pressure from known strengths is geometry and
        wavenumber only; ``c`` and ``rho`` enter the package when the
        boundary condition is imposed, and again in `velocity`, but not here.

        Parameters
        ----------
        source_strengths : ndarray, shape (N,)
            Complex source strengths in Pa*m, one per source, ordered as
            ``sources``. From `solver.solve_strength`, or assigned directly
            in the discretized Rayleigh branch.

        Returns
        -------
        ndarray, shape (M,)
            Complex128, in Pa. One value per target, in target order.

        Notes
        -----
        Phase follows the ``exp(-i * omega * t)`` convention, so the phase of
        ``p`` advances by ``kf * R`` with distance from a source. Comparing
        phases at two ranges through ``np.angle`` needs the values to be
        within half a wavelength of each other, or the comparison must
        unwrap; the wrap is not a sign error.

        A strength array that is a scalar rather than a length-one array is
        rejected here by matmul, which is the check `solver` relies on.

        References
        ----------
        Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
        Wiley (2007), chapter 1.
        """
        return self._green_TS @ source_strengths

    def velocity(self, source_strengths: np.ndarray) -> np.ndarray:
        """Acoustic particle velocity at every target point.

        The pressure gradient summed over sources, then Euler's equation:
        ``v = grad(p) / (i * omega * rho)``. All three components, not a
        projection: a target in the fluid carries no normal, and
        ``abs(v)**2`` in the Gor'kov potential needs the whole vector.

        The gradient array is never built. Since ``grad G = (dG/dr) * e``,
        the strengths are folded into ``dG/dr`` first, giving an ``(M, N)``
        complex intermediate, and only then contracted against the unit
        vectors. That is a third of the ``(M, N, 3)`` complex array the
        direct route materializes, and the Euler factor is applied to the
        ``(M, 3)`` result rather than to anything carrying ``N``.

        Parameters
        ----------
        source_strengths : ndarray, shape (N,)
            Complex source strengths in Pa*m, one per source, ordered as
            ``sources``.

        Returns
        -------
        ndarray, shape (M, 3)
            Complex128, in m/s. Cartesian components, in target order.

        Raises
        ------
        AttributeError
            If the object was built without a medium. There is no default
            fluid: a `Field` that assumed air would return plausible numbers
            for a medium nobody chose.

        Notes
        -----
        The final contraction is an ``einsum`` and not a ``matmul``, and the
        choice is load-bearing. BLAS does not mix real with complex, so a
        batched matmul promotes the unit vectors to ``complex128`` and
        allocates the very array this route exists to avoid: 276 MB of peak
        against 1 MB for the einsum, measured at ``M = 4000``, ``N = 1500``.
        The measurement that favours ``matmul`` over ``einsum`` elsewhere in
        the package was taken with two large operands of the same dtype and
        does not carry over here.

        For a single source the closed form is
        ``v = p * (i * kf - 1 / R) / (i * omega * rho)`` along the unit
        vector from source to target, and the ratio ``p / v_R`` is
        ``rho * c / (1 + i / (kf * R))``, which tends to the specific
        impedance of the fluid in the far field. A missing ``c`` or a
        mismatched ``omega`` breaks that ratio by a factor large enough to
        see by eye.

        ``c`` enters only through ``omega = kf * c``, so it is not
        independent of the ``kf`` the object was built with; values that
        disagree are not detected.

        The sign of the Euler factor follows ``exp(-i * omega * t)``. Under
        the opposite convention it is negated, together with every phase in
        the package.

        References
        ----------
        Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
        Wiley (2007), chapter 1.
        """
        w_TS = self._d_green_dr_TS * source_strengths
        factor = 1 / (1j * self.kf * self.medium.c * self.medium.rho)
        return factor * (np.einsum("mn,mnj->mj", w_TS, self._e_TSj))

    def velocity_gradient(self, source_strengths: np.ndarray) -> np.ndarray:
        """Cartesian gradient of the particle velocity at every target.

        Entry ``[m, i, j]`` is ``d v_j / d x_i`` at target ``m``. The Hessian
        of the pressure, summed over sources and divided by the Euler factor:

        ``d_i d_j G = delta_ij * h - e_i * e_j * s``

        with ``h`` and ``s`` the two scalars of `green_kernel`. It is the
        third quantity Gor'kov needs over the same cloud, after `pressure`
        and `velocity`, and the reason the tables are worth holding.

        The ``(M, N, 3, 3)`` array is never built. Each term is contracted as
        it stands: ``delta_ij`` leaves the sum over sources because it does
        not depend on ``n``, so the first term is a matrix-vector product
        laid on the diagonal; the second folds the strengths into ``s`` and
        contracts the unit vectors twice in one pass.

        Parameters
        ----------
        source_strengths : ndarray, shape (N,)
            Complex source strengths in Pa*m, one per source, ordered as
            ``sources``.

        Returns
        -------
        ndarray, shape (M, 3, 3)
            Complex128, in 1/s. Symmetric in the last two axes.

        Raises
        ------
        AttributeError
            If the object was built without a medium.

        Notes
        -----
        The three-operand ``einsum`` is written without ``optimize``, and the
        default is what is wanted. Measured at ``M = 20000``, ``N = 480``:
        the plain path allocates nothing and takes 709 ms, ``optimize=True``
        takes 570 ms but allocates 736 MB, because it splits the contraction
        into pairwise steps and materializes each one. Twenty percent of time
        is not worth three quarters of a gigabyte here, and the ratio only
        gets worse as ``N`` grows.

        Two identities hold and are worth testing. The matrix is symmetric,
        being the Hessian of a scalar. And its trace is Helmholtz: summing
        the diagonal gives ``3h - s = -kf**2 * G``, so

        ``sum_i d_i v_i = -kf**2 * p / (i * omega * rho)``

        That trace is a badly conditioned difference in the near field, where
        ``3h`` exceeds ``kf**2 * G`` by ``3 / (kf * r)**2``: the relative
        error of the identity is 2e-16 at ``kf * r = 0.9``, 1e-12 at 0.01 and
        1e-8 at 1e-4. It is a property of the subtraction, not of this
        contraction, so a test of it belongs at ``kf * r`` of order one,
        which is where the package evaluates anyway.

        Reordering the factors inside the sum is free -- they are complex
        scalars -- but it changes the last bits, since floating-point
        addition is not associative. Agreement with the tabulated Hessian is
        to 2e-16 relative, not exact.

        References
        ----------
        Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
        Wiley (2007), chapter 1.
        """
        hA = self._aux_h @ source_strengths
        diag_term = np.eye(3) * hA[:, None, None]
        w_TS = self._aux_s * source_strengths
        outer_term = np.einsum("mn,mni,mnj->mij", w_TS, self._e_TSj, self._e_TSj)
        factor = 1 / (1j * self.kf * self.medium.c * self.medium.rho)
        return factor * (diag_term - outer_term)


def pressure(
    targets: np.ndarray, source_strengths: np.ndarray, sources: np.ndarray, kf: float
) -> np.ndarray:
    """Acoustic pressure at every target point, in one call.

    Builds a `Field`, spends it and drops it. Two quantities over the same
    targets should build the object once instead of calling this and
    `velocity` in turn, which shares nothing between them and repeats the
    separation and the exponential.

    Parameters
    ----------
    targets : ndarray, shape (M, 3)
        Points where the pressure is evaluated, in meters.
    source_strengths : ndarray, shape (N,)
        Complex source strengths in Pa*m, one per source, ordered as
        ``sources``.
    sources : ndarray, shape (N, 3)
        Point-source positions, in meters, the same array the strengths were
        solved against.
    kf : float
        Wavenumber of the fluid, ``2 * pi / wavelength``, in rad/m.

    Returns
    -------
    ndarray, shape (M,)
        Complex128, in Pa. One value per target, in target order.

    See Also
    --------
    Field.pressure : the same calculation, with the tables kept.
    """
    return Field(targets, sources, kf).pressure(source_strengths)


def velocity(
    targets: np.ndarray,
    source_strengths: np.ndarray,
    sources: np.ndarray,
    kf: float,
    c: float,
    rho: float,
) -> np.ndarray:
    """Acoustic particle velocity at every target point, in one call.

    Builds a `Field`, spends it and drops it. Takes ``c`` and ``rho`` loose
    and packs them into a `Medium`, so that existing callers keep their
    signature.

    Parameters
    ----------
    targets : ndarray, shape (M, 3)
        Points where the velocity is evaluated, in meters.
    source_strengths : ndarray, shape (N,)
        Complex source strengths in Pa*m, one per source, ordered as
        ``sources``.
    sources : ndarray, shape (N, 3)
        Point-source positions, in meters, the same array the strengths were
        solved against.
    kf : float
        Wavenumber of the fluid, ``2 * pi / wavelength``, in rad/m.
    c : float
        Speed of sound in the fluid, in m/s. Enters only through
        ``omega = kf * c``, so it is not independent of ``kf``; values that
        disagree are not detected.
    rho : float
        Density of the fluid, in kg/m^3.

    Returns
    -------
    ndarray, shape (M, 3)
        Complex128, in m/s. Cartesian components, in target order.

    See Also
    --------
    Field.velocity : the same calculation, with the tables kept.
    """
    return Field(targets, sources, kf, Medium(c, rho)).velocity(source_strengths)


def velocity_gradient(
    targets: np.ndarray,
    source_strengths: np.ndarray,
    sources: np.ndarray,
    kf: float,
    c: float,
    rho: float,
) -> np.ndarray:
    """Cartesian gradient of the particle velocity, in one call.

    Builds a `Field`, spends it and drops it. Gor'kov needs this together
    with the pressure and the velocity over the same targets, and calling the
    three functions in turn repeats the separation and the exponential three
    times: build the object once instead.

    Parameters
    ----------
    targets : ndarray, shape (M, 3)
        Points where the gradient is evaluated, in meters.
    source_strengths : ndarray, shape (N,)
        Complex source strengths in Pa*m, one per source, ordered as
        ``sources``.
    sources : ndarray, shape (N, 3)
        Point-source positions, in meters, the same array the strengths were
        solved against.
    kf : float
        Wavenumber of the fluid, ``2 * pi / wavelength``, in rad/m.
    c : float
        Speed of sound in the fluid, in m/s. Enters only through
        ``omega = kf * c``, so it is not independent of ``kf``; values that
        disagree are not detected.
    rho : float
        Density of the fluid, in kg/m^3.

    Returns
    -------
    ndarray, shape (M, 3, 3)
        Complex128, in 1/s. Entry ``[m, i, j]`` is ``d v_j / d x_i``.

    See Also
    --------
    Field.velocity_gradient : the same calculation, with the tables kept.
    """
    return Field(targets, sources, kf, Medium(c, rho)).velocity_gradient(
        source_strengths
    )
