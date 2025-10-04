# Part 3: test Euler-Lagrange equation using autodiff on an SHO Lagrangian

import jax
jax.config.update('jax_enable_x64', True)

from jax import numpy as jnp
from hw3 import p3

def test_elrhs():

    def L(x, v):
        """SHO: L = (1/2) (v^2 - x^2)  (unit mass, unit stiffness)"""
        return 0.5 * jnp.sum(v * v - x * x)

    rhs = p3.ELrhs(L)

    # Check that dx/dt = v when x=0 ...
    for v in jnp.linspace(-1.0, 1.0, 11):
        out = rhs(jnp.array([0.0, v]))
        assert jnp.allclose(out, jnp.array([v,  0]), rtol=1e-6, atol=1e-6)

    # ... and dv/dt = -x when v=0
    for x in jnp.linspace(-1.0, 1.0, 11):
        out = rhs(jnp.array([x, 0.0]))
        assert jnp.allclose(out, jnp.array([0, -x]), rtol=1e-6, atol=1e-6)
