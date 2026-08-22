# 0001 - Time convention, Green's function and velocity

Status: accepted

## Context

Every acoustic quantity in this package is a complex amplitude. Recovering the
physical field requires a time convention,

    p(r, t) = Re{ p_hat(r) * exp(-i*omega*t) }

and two conventions are in common use: exp(-i*omega*t), standard in physical
acoustics and in most scattering literature, and exp(+i*omega*t), standard in
engineering and in the electrical analogy. The two are related by complex
conjugation, so either is internally consistent. Mixing them is not: it inverts
the sign of every imaginary part. An outgoing wave becomes an incoming one,
near-field phase runs backwards, and the radiation force changes sign. Nothing
raises an error; the result is simply wrong.

The convention cannot be chosen locally. A single choice fixes, together:

- the sign in the exponent of the free-space Green's function, since an outgoing
  wave requires the phase to advance with distance;
- the sign relating particle velocity to the pressure gradient, since the
  linearised Euler equation turns d/dt into a factor of -i*omega;
- which Hankel function represents an outgoing spherical wave, this being the
  previous point again with angular structure added;
- the azimuthal phase sense of the spherical-harmonic expansion, whose
  exp(i*m*phi) factor conjugates along with everything else.

Harmonic normalization and the Condon-Shortley phase are separate conventions.
They are real positive factors and an overall sign respectively, and neither is
fixed by this decision; see Consequences.

The package must also interoperate with published work: Placko & Kundu, the DPSM
source; Baresch's MATLAB routines, the GLMT reference implementation; and Pazos
Ospina et al., Phys. Rev. Applied 18, 034026 (2022), the benchmark of Milestone 6.

## Decision

The package uses exp(-i*omega*t) throughout. Consequently

    G(R) = exp(i*k*R) / (4*pi*R)
    v    = grad(p) / (i*omega*rho)

These are one decision stated three ways, not three decisions. The velocity
relation is not independent: substituting the harmonic ansatz into

    rho * dv/dt = -grad(p)

gives rho * (-i*omega) * v_hat = -grad(p_hat), hence v_hat = grad(p_hat) /
(i*omega*rho). No module may assume otherwise, and no external result is used
without first checking the convention it was produced under.

## Alternatives considered

exp(+i*omega*t), with G(R) = exp(-i*k*R)/(4*pi*R) and v = -grad(p)/(i*omega*rho).
Equally consistent, and more familiar to readers arriving from electrical
engineering or electromagnetics. Rejected because all three sources this project
depends on use the opposite convention:

- Placko & Kundu, Eqs. 1.19-1.20;
- Baresch's code: plane_wave.m expands with i^n, focused_beam.m uses Hankel
  functions of the first kind for outgoing waves;
- the velocity potential phi = p / (i*omega*rho) of the Phys. Rev. Applied paper.

Adopting exp(+i*omega*t) would require conjugating every imported result, a
permanent source of sign errors for no benefit.

Leaving the convention implicit and fixing signs case by case. Rejected: that is
the failure mode described in Context.

## Consequences

Outgoing spherical waves are represented by Hankel functions of the first kind.
This is not an extra assumption. For n = 0 the asymptotic form is exact,

    h_0^(1)(x) = -i * exp(i*x) / x

so the Green's function above is that Hankel function up to a constant:

    G(R) = (i*k / (4*pi)) * h_0^(1)(k*R)

The second kind does not appear anywhere in this project: there are no sources at
infinity radiating inward. Incident fields expanded about a particle centre use
spherical Bessel functions j_n, which are finite at the origin; scattered fields
use h_n^(1).

Phase advances with distance, which gives a cheap invariant that any Green's
function implementation must satisfy:

    np.angle(G(R2)) - np.angle(G(R1)) == +k * (R2 - R1)

A unit test asserts it. It catches a conjugated Green's function immediately.

Any external field, dataset or reference implementation must have its convention
verified before its output is compared with ours. A mismatch appears as a sign
flip in the imaginary part, not as an error.

The symbol table in docs/notacion.md and the README state this convention
explicitly, so a reader can tell whether their result is comparable without
reading the source.

The normalization of spherical harmonics and the Condon-Shortley phase remain
open. They are settled in Milestone 6 by checking against Baresch's code, which
rectifies legendre(..., 'norm') to fully normalized harmonics, and not from
memory or from this document.
