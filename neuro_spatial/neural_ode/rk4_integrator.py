"""
Continuous-Time Neural ODE Solver (Runge-Kutta 4th Order).
Integrates hidden neural state in continuous time dh/dt = f(h, t) for microsecond robot tracking.
"""
from typing import List, Callable
import math

class ContinuousNeuralODEIntegrator:
    @staticmethod
    def rk4_step(
        f: Callable[[float, List[float]], List[float]],
        t: float,
        h: List[float],
        dt: float
    ) -> List[float]:
        """
        Standard 4th Order Runge-Kutta step:
        k1 = f(t, h)
        k2 = f(t + dt/2, h + dt*k1/2)
        k3 = f(t + dt/2, h + dt*k2/2)
        k4 = f(t + dt, h + dt*k3)
        h_next = h + (dt/6)*(k1 + 2k2 + 2k3 + k4)
        """
        k1 = f(t, h)
        h_k1 = [hi + 0.5 * dt * ki for hi, ki in zip(h, k1)]

        k2 = f(t + 0.5 * dt, h_k1)
        h_k2 = [hi + 0.5 * dt * ki for hi, ki in zip(h, k2)]

        k3 = f(t + 0.5 * dt, h_k2)
        h_k3 = [hi + dt * ki for hi, ki in zip(h, k3)]

        k4 = f(t + dt, h_k3)

        h_next = [
            hi + (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
            for hi, k1i, k2i, k3i, k4i in zip(h, k1, k2, k3, k4)
        ]
        return h_next
