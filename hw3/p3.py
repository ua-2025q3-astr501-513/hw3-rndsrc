from jax import numpy as jnp
from jax import grad, jacfwd, jit
from jax.numpy.linalg import inv

def ELrhs(L):

    Lx = grad(L, argnums=0)
    Lv = grad(L, argnums=1)

    Lvp = jacfwd(Lv, argnums=(0,1))

    @jit
    def rhs(xv):
        n = len(xv)
        assert n % 2 == 0
        h = int(n//2)

        x = xv[:h]
        v = xv[h:]

        Lvx, Lvv = Lvp(x, v)

        a = inv(Lvv) @ (Lx(x, v) - v @ Lvx)

        return jnp.concatenate([v, a])

    return rhs
