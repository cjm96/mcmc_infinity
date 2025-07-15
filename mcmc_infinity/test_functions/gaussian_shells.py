import numpy as np
from scipy.special import logsumexp


class GaussianShells:
    """
    A class representing a test function involving two Gaussian shells.

    The log-posterior of the target distribution is defined as:
    .. math::
        f(x) = \\log ( circ(x; c_1, r, w) + circ(x; c_2, r, w) )
    where
    .. math::
        circ(x; c, r, w) = \\exp( -(|x-c|-r)^2 / (2w^2) ) / \\sqrt{2\\pi w^1}
    and where c1 and c2 are the vectors of coordinates of the centres of the two 
    rings, r is the radius of the rings and w is their width.
    
    This function is defined for any dimensionality, n.
    """

    def __init__(self, s=7, r=2.0, w=0.1, dim=2):
        """
        INPUTS:
        -------
        s : float 
            The seperation of the centres of the Gaussian rings. Default is 7.0.
        r : float 
            The radius of the two of the Gaussian rings. Default is 2.0.
        w : float 
            The width of the two Gaussian rings. Default is 0.1.
        dim : int
            The dimensionality of the Rastrigin function. Default is 2.
        """
        self.s = float(s)
        assert self.s > 0, "seperation must be positive"
        self.r = float(r)
        assert self.r > 0, "radius must be positive"
        self.w = float(w)
        assert self.w > 0, "width must be positive"
        self.dim = int(dim)
        assert self.dim > 0, "dimension must be positive"

        # The centres of the two Gaussian shells
        self.c1 = np.zeros(self.dim)
        self.c2 = np.zeros(self.dim)
        self.c1[0] = -self.s / 2.0
        self.c2[0] = +self.s / 2.0

    def log_circ(self, x, c, r, w):
        """
        INPUTS:
        -------
        x : array-like, shape=(..., self.dim)
            An array of inputs to the Gaussian shells function.
        c : array-like, shape=(self.dim,)
            The centre of the Gaussian shell.
        r : float
            The radius of the Gaussian shell.
        w : float
            The width of the Gaussian shell.

        RETURNS
        -------
        ans : array-like, shape=(...,)
            The circ function value.
        """
        x = np.asarray(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"
        d = np.linalg.norm(x - c, axis=-1)
        return -0.5*(d-r)**2 / w**2 - 0.5 * np.log(2*np.pi*w**2)

    def logP(self, x):
        """
        INPUTS:
        -------
        x : array-like, shape=(..., self.dim)
            An array of inputs to the Gaussian shells function.

        RETURNS
        -------
        logl : array-like, shape=(...,)
            The log-posterior of the target function.
        """
        x = np.asarray(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"
        logl = logsumexp([self.log_circ(x, self.c1, self.r, self.w),
                          self.log_circ(x, self.c2, self.r, self.w)]
                        , axis=0)
        return logl

    def __call__(self, x):
        return self.logP(x)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    shells = GaussianShells()

    x = np.linspace(-8, 8, 300)
    y = np.linspace(-8, 8, 300)
    X, Y = np.meshgrid(x, y)
    Z = shells(np.stack((X,Y), axis=-1))
    
    fig, ax = plt.subplots(figsize=(8, 6))

    levels = np.linspace(-50, 2, 53)
    x = ax.contourf(X, Y, Z, levels=levels)
    cbar = plt.colorbar(x)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Gaussian Shells")
    ax.set_aspect('equal')

    plt.show()
    