from dataclasses import dataclass


@dataclass(frozen=True)
class Medium:
    c: float
    rho: float
