from jax import numpy as jnp

def RK4(f, x, t, dt, n):

    X = [jnp.array(x)]
    T = [jnp.array(t)]

    for _ in range(n):
        k1 = f(X[-1]                )
        k2 = f(X[-1] + 0.5 * dt * k1)
        k3 = f(X[-1] + 0.5 * dt * k2)
        k4 = f(X[-1] +       dt * k3)
        X.append(X[-1] + dt * (k1/6 + k2/3 + k3/3 + k4/6))
        T.append(T[-1] + dt)

    return jnp.array(X), jnp.array(T)
