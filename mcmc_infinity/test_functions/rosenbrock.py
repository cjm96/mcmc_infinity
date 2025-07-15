import numpy as np


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
        x = np.asarray(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"
        mask = (np.arange(self.dim)<self.dim-1)
        logl = -np.sum(100*(np.roll(x,-1,axis=-1)[...,mask]-x[...,mask]**2)**2
                      +(1-x[...,mask])**2, axis=-1)
        return logl

    def __call__(self, x):
        return self.logP(x)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    R = Rosenbrock()

    x = np.linspace(-1, 3, 300)
    y = np.linspace(-1, 6, 300)
    X, Y = np.meshgrid(x, y)
    Z = R(np.stack((X,Y), axis=-1))
    
    fig, ax = plt.subplots(figsize=(8, 6))

    levels = np.linspace(-60, 0, 7)
    x = ax.contourf(X, Y, Z, levels=levels)
    cbar = plt.colorbar(x)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Rosenbrock Banana")
    ax.set_aspect('equal')

    plt.show()
