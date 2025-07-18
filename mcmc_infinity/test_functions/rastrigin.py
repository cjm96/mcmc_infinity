import jax.numpy as jnp


class Rastrigin:
    """
    A class representing the Rastrigin function, a well-known test function for 
    optimization and sampling problems.

    The log-posterior of the target distribution is defined as:
    .. math::
        f(x) = \\alpha ( -An - \\sum_{i=1}^{n} ( x_i^2 - A \\cos(2\\pi x_i) ) )
    where n is the dimensionality and A is a constant (commonly set to 10).
    """

    def __init__(self, alpha=1.0, A=10.0, dim=2):
        """
        INPUTS:
        -------
        alpha : float
            An overall scaling prefactor for the log-posterior. Default is 1.0.
        A : float
            The constant A in the Rastrigin function. Default is 10.0.
        dim : int
            The dimensionality of the Rastrigin function. Default is 2.
        """
        self.alpha = float(alpha)
        assert self.alpha > 0, "alpha must be positive"
        self.A = float(A)
        self.dim = int(dim)
        assert self.dim > 0, "dimension must be positive"

    def logP(self, x):
        """
        INPUTS:
        -------
        x : array-like, shape=(..., self.dim)
            An array of inputs to the Rastrigin function.

        RETURNS
        -------
        logl : array-like, shape=(...,)
            The log-posterior of the target function.
        """
        x = jnp.asarray(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"
        logl = -self.A*self.dim - jnp.sum(x**2-self.A*jnp.cos(2*jnp.pi*x), axis=-1)
        return self.alpha * logl

    def __call__(self, x):
        return self.logP(x)
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    R = Rastrigin()

    x = jnp.linspace(-5.12, 5.12, 300)
    y = jnp.linspace(-5.12, 5.12, 300)
    X, Y = jnp.meshgrid(x, y)
    Z = R(jnp.stack((X,Y), axis=-1))
    
    fig, ax = plt.subplots(figsize=(8, 6))

    x = ax.contourf(X, Y, Z)
    cbar = plt.colorbar(x)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Rastrigin Function")

    plt.show()
