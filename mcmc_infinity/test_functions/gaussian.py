import numpy as np


class Gaussian:
    """
    A class representing a multivariate normal, or Gaussian, test function.

    The log-posterior of the target distribution is defined as:
    .. math::
        f(x) = -(1/2) \sum_{i=1}^{n} x_i^2 / \sigma^2
    where n is the dimensionality.
    """

    def __init__(self, sigma=1.0, dim=2):
        """
        INPUTS:
        -------
        sigma : float 
            The width of the Gaussian distribution. Default is 1.0.
        dim : int
            The dimensionality of the Rastrigin function. Default is 2.
        """
        self.sigma = float(sigma)
        assert self.sigma > 0, "sigma must be positive"
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
        x = np.asarray(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"
        logl = -0.5 * np.sum(x**2 / self.sigma**2, axis=-1)
        return logl

    def __call__(self, x):
        return self.logP(x)
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    G = Gaussian()

    x = np.linspace(-5, 5, 300)
    y = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(x, y)
    Z = G(np.stack((X,Y), axis=-1))
    
    fig, ax = plt.subplots(figsize=(8, 6))

    x = ax.contourf(X, Y, Z)
    cbar = plt.colorbar(x)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Gaussian Function")

    plt.show()
