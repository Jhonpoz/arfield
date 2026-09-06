"""Acoustic field modeling by the Distributed Point Source Method.

DPSM replaces a radiating surface by a layer of point sources sitting a
short distance behind it. Their strengths are unknown: imposing the
prescribed normal velocity at collocation points on the surface gives a
linear system, and once that is solved the field follows anywhere in the
fluid. The sources are set back precisely so that no evaluation point ever
lands on one, which is what keeps the formulation free of the singular
integrals a boundary-element method has to handle.

Not a new method. This is an open, tested implementation of Placko and
Kundu (2007), aimed at airborne acoustic levitation.

The method in five steps, and the module that owns each:

1. field of a single point source ....... green_kernel
2. cover the surface with sources ....... mesh, source
3. assemble the influence matrix ........ pairwise, influence
4. obtain the source strengths .......... solver
5. evaluate the field anywhere .......... field

Step four has two routes and `solver` holds both: solving the linear
system, or assigning each element the strength the Rayleigh integral gives
it. They return the same quantity and step five cannot tell them apart.

Conventions, fixed package-wide and not negotiable per call:

- Time factor ``exp(-i * omega * t)``.
- Green's function ``G(R) = exp(i * kf * R) / (4 * pi * R)``, with the
  ``1 / (4 * pi)`` inside the kernel rather than absorbed into the
  strengths as the book does.
- Velocity from pressure through Euler, ``v = grad(p) / (i * omega * rho)``.
- SI units everywhere. No hidden millimeters.
- ``complex128``, explicit, in every acoustic quantity.
- Influence arrays shaped ``(target, source)``, in that order, with a third
  axis of length three when the entries are vectors.
- ``kf`` is passed explicitly wherever it is needed, never derived from a
  medium or kept in a module constant: frequency belongs to the experiment,
  not to the fluid or to the method.

What is bound here is what does not oblige a caller to adopt the geometry
of this package: functions that take arrays of points and scalars, plus
`Source` as a convenience rather than a required path. The objects that
describe something concrete are asked for explicitly, since importing them
is a statement that you want this package's idea of a mesh or a medium::

    from arfield.mesh import circle
    from arfield.medium import Medium

Scope. One homogeneous, lossless fluid, and surfaces that do not see each
other: there is no coupled block assembly, no reflector, and no layered or
heterogeneous medium. A boundary condition is imposed, not negotiated
between two surfaces.

References
----------
Placko, D. and Kundu, T., *DPSM for Modeling Engineering Problems*,
Wiley (2007).
"""

from .field import pressure, velocity
from .green_kernel import gradient_green, green
from .influence import (
    compute_euler_gradn_green_TS,
    compute_grad_green_TSj,
    compute_green_TS,
)
from .pairwise import separation
from .solver import rayleigh_strength, solve_strength
from .source import Source, tangent_displacement

__all__ = [
    "Source",
    "compute_euler_gradn_green_TS",
    "compute_grad_green_TSj",
    "compute_green_TS",
    "gradient_green",
    "green",
    "pressure",
    "rayleigh_strength",
    "separation",
    "solve_strength",
    "tangent_displacement",
    "velocity",
]
