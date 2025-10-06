from jax import numpy as jnp

from .p2 import RK4
from .p3 import ELrhs
from .p4 import L

def solve(theta1, theta2, omega1, omega2, t, dt):

    t0  = 0
    xv0 = jnp.array([theta1, theta2, omega1, omega2], dtype=float)

    XV, T = RK4(ELrhs(L), xv0, t0, dt, int(t//dt))

    X = XV[:,:2]
    V = XV[:,2:]

    return X, V, T
