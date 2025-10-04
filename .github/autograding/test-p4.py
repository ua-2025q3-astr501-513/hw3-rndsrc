# Part 4: Double pendulum Lagrangian checks

import jax
jax.config.update('jax_enable_x64', True)

from jax import numpy as jnp
from hw3 import p3, p4


def test_L():

    L    = p4.L(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    Lref = 12.510860542404144
    assert jnp.isscalar(L)
    assert jnp.allclose(L, Lref, rtol=1e-6, atol=1e-6)

    L    = p4.L(jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0]))
    Lref = 28.959390699356288
    assert jnp.isscalar(L)
    assert jnp.allclose(L, Lref, rtol=1e-6, atol=1e-6)


def test_rhs():

    rhs   = p3.ELrhs(p4.L)

    VA    = rhs(jnp.array([1,2,3,4], dtype=float))
    VAref = jnp.array([3,4,-9.00440027114048,21.080178930488973])
    assert VA.shape == (4,)
    assert jnp.allclose(VA, VAref, rtol=1e-6, atol=1e-6)

    VA    = rhs(jnp.array([3,4,5,6], dtype=float))
    VAref = jnp.array([5,6,-20.97116005342414,55.99754725970956])
    assert VA.shape == (4,)
    assert jnp.allclose(VA, VAref, rtol=1e-6, atol=1e-6)
