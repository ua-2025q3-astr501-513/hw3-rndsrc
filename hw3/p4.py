from jax import numpy as jnp

def L(x, v):
    x1, x2 = x
    v1, v2 = v
    return (
        (1/6) * (4 * v1*v1 + v2*v2 + 3 * v1*v2 * jnp.cos(x1 - x2)) +
        (1/2) * (3 * jnp.cos(x1) + jnp.cos(x2))
    )
