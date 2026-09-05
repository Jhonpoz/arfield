from dataclasses import dataclass

__all__ = ["Particle"]


@dataclass(frozen=True)
class Particle:
    radius: float
    rho: float
    kappa: float
