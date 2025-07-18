import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal


class DiagonalGaussianProposal:
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
    

class GaussianProposal:
    """
    A Gaussian proposal distribution with a diagonal covariance matrix.

    Parameters
    ----------
    dim : int
        The dimensionality of the proposal distribution.
    mu : float or array-like
        The mean of the Gaussian proposal.
    sigma : float or array-like
        The standard deviation of the Gaussian proposal.
    bounds : tuple, optional
        Optional bounds for the proposal distribution (not used in this implementation).
    """

    def __init__(self, dim, bounds=None, inflation_scale=1.0):
        self.dim = int(dim)
        self.bounds = jnp.asarray(bounds) if bounds is not None else None
        self.mu = None
        self._cov = None
        self.inflation_scale = inflation_scale

    def set_inflation_scale(self, scale):
        """
        Set the inflation scale for the proposal distribution.

        Parameters
        ----------
        scale : float
            The inflation scale to apply to the covariance.
        """
        self.inflation_scale = scale

    @property
    def cov(self):
        """
        Get the covariance matrix of the proposal distribution.

        Returns
        -------
        cov : jnp.ndarray
            The covariance matrix of the proposal distribution.
        """
        return self._cov * self.inflation_scale

    def fit(self, samples):
        if self.bounds:
            # Apply bounds to samples
            samples = (samples - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
            samples = jnp.log(samples / (1 - samples))
        
        self.mu = jnp.mean(samples, axis=0)
        self._cov = jnp.cov(samples, rowvar=False)

    def sample(self, key, num_samples=None):
        if num_samples is None:
            num_samples = 1
        x = jax.random.multivariate_normal(
            key, self.mu, self.cov, shape=(num_samples,)
        )
        # Use logit 
        if self.bounds is not None:
            x = jnp.exp(x) / (1 + jnp.exp(x))
            x = x * (self.bounds[:, 1] - self.bounds[:, 0])
            x = x + self.bounds[:, 0]
        return x
    
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

        if self.bounds is not None:
            # Apply logit transformation
            x = (x - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
            x = jnp.log(x / (1 - x))

        return multivariate_normal.logpdf(
            x, mean=self.mu, cov=self.cov
        )

if __name__ == "__main__":
    # Example usage
    dim = 2
    mu = jnp.zeros(dim)
    sigma = 1.0
    proposal = GaussianProposal(dim)

    initial_samples = jax.random.normal(jax.random.PRNGKey(0), shape=(1000, dim))
    proposal.fit(initial_samples)

    key = jax.random.PRNGKey(0)
    samples = proposal.sample(key, num_samples=5)
    print("Samples:", samples)

    log_probs = proposal.logP(samples)
    print("Log probabilities:", log_probs)