from dataclasses import dataclass

__all__ = ["Medium"]


@dataclass(frozen=True)
class Medium:
    c: float
    rho: float
