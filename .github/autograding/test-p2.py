# Part 2: test RK4 integrator using a simple harmonic oscillator

import jax
jax.config.update('jax_enable_x64', True)

from jax import numpy as jnp
from hw3 import p2

def test_rk4():

    def rhs(xv):
        """SHO in first-order form: d/dt [x, v] = [v, -x]"""
        x, v = xv
        return jnp.array([v, -x])

    x0 = jnp.array([0.0, 0.1])  # x(0)=0, v(0)=0.1
    t0 = 0.0
    dt = 2 * jnp.pi / 200       # 100 steps per 2π period
    n  = 100

    X, T = p2.RK4(rhs, x0, t0, dt, n)

    # shape checks
    assert X.shape == (n + 1, 2)
    assert T.shape == (n + 1,)

    # Monotone increasing time, correct spacing (within FP tolerance)
    dT = jnp.diff(T)
    print(max(abs(dT-dt)))
    assert jnp.all(dT > 0)
    assert jnp.allclose(dT, float(dt), rtol=0, atol=1e-6)

    # Analytic reference solution: x(t) = 0.1 * sin(t)
    Xref = 0.1 * jnp.sin(T)
    X    = X[:,0]

    # RK4 should be very accurate for this smooth problem
    print(max(abs(X-Xref)))
    assert jnp.allclose(X, Xref, rtol=1e-6, atol=1e-6)
