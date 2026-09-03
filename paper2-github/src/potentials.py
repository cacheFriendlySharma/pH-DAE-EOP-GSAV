from dataclasses import dataclass,field
from typing import Protocol

import numpy as np


class SpringPotential(Protocol):
    name: str
    linear_stiffness: float

    def energy_density(self,r: np.ndarray) -> np.ndarray:
        ...

    def gradient(self,r: np.ndarray) -> np.ndarray:
        ...

    def curvature(self,r: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class FPUBetaPotential:
    beta: float
    linear_stiffness: float = 1.0
    name: str = field(default="FPU-beta",init=False)

    def energy_density(self,r):
        return 0.5*self.linear_stiffness*r**2+0.25*self.beta*r**4

    def gradient(self,r):
        return self.linear_stiffness*r+self.beta*r**3

    def curvature(self,r):
        return self.linear_stiffness+3.0*self.beta*r**2


@dataclass(frozen=True)
class TodaPotential:
    """
    Normalized Toda potential

        V(r) = k/b^2 * (exp(-b*r)+b*r-1),

    so that V''(0) = k.
    """
    b: float
    linear_stiffness: float = 1.0
    name: str = field(default="Toda",init=False)

    def __post_init__(self):
        if self.b <= 0.0:
            raise ValueError("Toda parameter b must be positive.")

        if self.linear_stiffness <= 0.0:
            raise ValueError("Toda linear stiffness must be positive.")

    def energy_density(self,r):
        b = self.b
        k = self.linear_stiffness

        return k*(np.expm1(-b*r)+b*r)/b**2

    def gradient(self,r):
        b = self.b
        k = self.linear_stiffness

        return -k*np.expm1(-b*r)/b

    def curvature(self,r):
        return self.linear_stiffness*np.exp(-self.b*r)


@dataclass(frozen=True)
class FENEPotential:
    """
    FENE potential

        V(r) = -k*R^2/2 * log(1-r^2/R^2),

    defined only for |r| < R and normalized by V''(0) = k.
    """
    extension_limit: float
    linear_stiffness: float = 1.0
    name: str = field(default="FENE",init=False)

    def __post_init__(self):
        if self.extension_limit <= 0.0:
            raise ValueError("FENE extension limit must be positive.")

        if self.linear_stiffness <= 0.0:
            raise ValueError("FENE linear stiffness must be positive.")

    def _normalized_extension(self,r):
        r = np.asarray(r,dtype=float)
        q = r/self.extension_limit

        if np.any(np.abs(q) >= 1.0):
            maximum_extension = np.max(np.abs(r))

            raise FloatingPointError(
                "FENE domain violation: "
                f"max |r| = {maximum_extension:.6e}, "
                f"extension limit = {self.extension_limit:.6e}."
            )

        return q

    def energy_density(self,r):
        q = self._normalized_extension(r)
        R = self.extension_limit
        k = self.linear_stiffness

        return -0.5*k*R**2*np.log1p(-q**2)

    def gradient(self,r):
        q = self._normalized_extension(r)

        return self.linear_stiffness*r/(1.0-q**2)

    def curvature(self,r):
        q = self._normalized_extension(r)

        return self.linear_stiffness*(1.0+q**2)/(1.0-q**2)**2


def nonlinear_energy_density(potential: SpringPotential,r):
    k = potential.linear_stiffness

    return potential.energy_density(r)-0.5*k*r**2


def nonlinear_gradient(potential: SpringPotential,r):
    k = potential.linear_stiffness

    return potential.gradient(r)-k*r