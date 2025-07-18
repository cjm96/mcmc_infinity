import jax.numpy as jnp


class Rosenbrock:
    """
    A class representing the Rosenbrock banana test function.

    The log-posterior of the target distribution is defined as:
    .. math::
        f(x) = -\\sum_{i=1}^{n-1} ( 100 (x_{i+1}-x_{i}^2)^2 + (1-x_{i})^2 )
    where n is the dimensionality.
    """

    def __init__(self, dim=2):
        """
        INPUTS:
        -------
        dim : int
            The dimensionality of the Rosenbrock banana. Default is 2.
        """
        self.dim = int(dim)
        assert self.dim > 0, "dimension must be positive"

    def logP(self, x):
        """
        INPUTS:
        -------
        x : array-like, shape=(..., self.dim)
            An array of inputs to the Rosenbrock function.

        RETURNS
        -------
        logl : array-like, shape=(...,)
            The log-posterior of the target function.
        """
        x = jnp.asarray(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"

        x0 = x[..., :-1]
        x1 = x[..., 1:]
        logl = -jnp.sum(100 * (x1 - x0**2)**2 + (1 - x0)**2, axis=-1)
        return logl

    def __call__(self, x):
        return self.logP(x)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    R = Rosenbrock()

    x = jnp.linspace(-1, 3, 300)
    y = jnp.linspace(-1, 6, 300)
    X, Y = jnp.meshgrid(x, y)
    Z = R(jnp.stack((X,Y), axis=-1))
    
    fig, ax = plt.subplots(figsize=(8, 6))

    levels = jnp.linspace(-60, 0, 7)
    x = ax.contourf(X, Y, Z, levels=levels)
    cbar = plt.colorbar(x)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Rosenbrock Banana")
    ax.set_aspect('equal')

    plt.show()
