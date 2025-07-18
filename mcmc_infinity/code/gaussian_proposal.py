import jax
import jax.numpy as jnp


class GaussianProposal:
    """
    A Gaussian proposal distribution.

    Parameters
    ----------
    dim : int
        The dimensionality of the proposal distribution.
    mu : float
        The mean of the Gaussian proposal.
    sigma : float
        The standard deviation of the Gaussian proposal.
    bounds : tuple, optional
        Optional bounds for the proposal distribution (not used in this implementation).
    """

    def __init__(self, dim, mu, sigma):
        """
        Initialize the Gaussian proposal distribution.

        Parameters
        ----------
        dim : int
            The dimensionality of the proposal distribution.
        mu : float
            The mean of the Gaussian proposal.
        sigma : float
            The standard deviation of the Gaussian proposal.
        """
        self.dim = int(dim)
        self.mu = jnp.asarray(mu)
        self.sigma = float(sigma)

    def sample(self, key, num_samples=None):
        """
        Generate samples from the Gaussian proposal distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey
            The random key for reproducibility.
        num_samples : int, optional
            The number of samples to generate. Default is None, which generates a single sample.

        Returns
        -------
        samples : jnp.ndarray
            Samples from the Gaussian proposal distribution.
            Shape=(num_samples, self.dim) or (self.dim,).
        """
        if num_samples is None:
            shape = (self.dim,)
        else:
            shape = (int(num_samples), self.dim)

        y = self.mu + self.sigma * jax.random.normal(key, shape=shape)

        return y

    def logP(self, x):
        """
        Compute the log-density of the Gaussian proposal distribution.

        Parameters
        ----------
        x : array-like, shape=(..., self.dim)
            An array of inputs to the log density function.

        Returns
        -------
        logl : array-like, shape=(...,)
            The log-density of the Gaussian proposal function.
        """
        x = jnp.asarray(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"

        diff = x - self.mu
        logl = -0.5 * jnp.sum(diff**2, axis=-1) / (self.sigma**2)
        logl -= 0.5 * self.dim * jnp.log(2 * jnp.pi * self.sigma**2)

        return logl

    def __call__(self, x):
        """
        Call the logP method for convenience.

        Parameters
        ----------
        x : array-like, shape=(..., self.dim)
            An array of inputs to the log density function.

        Returns
        -------
        logl : array-like, shape=(...,)
            The log-density of the Gaussian proposal function.
        """
        return self.logP(x)