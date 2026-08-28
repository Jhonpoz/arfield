from dataclasses import dataclass


@dataclass(frozen=True)
class Particle:
    radius: float
    rho: float
    kappa: float
