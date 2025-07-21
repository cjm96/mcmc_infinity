import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from ..utils import logit, inv_logit

class GaussianProposal:
    """
    A Gaussian proposal distribution.

    The user can either provide a mean and covariance matrix, or these can
    be estimated from some samples that the user provides.
    """

    def __init__(self, dim, bounds=None, inflation_scale=1.0,
                 mean=None, cov=None):
        """
        INPUTS
        ------
        dim : int
            The dimensionality of the proposal distribution.
        bounds : array-like, optional
            Optional bounds for the proposal distribution. 
            If provided, the samples are transformed using a logit transformation.
        inflation_scale : float, optional
            A scale factor to inflate the covariance matrix. Default is 1.0.
        mean : array-like, shape=(dim,), optional
            The mean of the Gaussian proposal distribution. 
            If None, it will be estimated from samples.
        cov : array-like, shape=(dim, dim), optional
            The covariance matrix of the Gaussian proposal distribution. 
            If None, it will be estimated from samples.
        """
        self.dim = int(dim)
        self.bounds = jnp.asarray(bounds) if bounds is not None else None
        self.mu = jnp.array(mean) if mean is not None else None
        self._cov = jnp.array(cov) if cov is not None else None
        self.inflation_scale = float(inflation_scale)

    def set_inflation_scale(self, scale):
        """
        Change the inflation scale for the proposal distribution.

        PARAMETERS
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
        """
        Fit the mean and covariance of the proposal distribution from samples.

        PARAMETERS
        ----------
        samples : jnp.ndarray, shape=(num_samples, dim)
            The samples to fit the proposal distribution.
        """
        assert self.mu is None, "mean is already set"
        assert self._cov is None, "covariance is already set"

        if self.bounds is not None:
            samples = logit(samples, bounds=self.bounds)[0]

        self.mu = jnp.mean(samples, axis=0)
        self._cov = jnp.cov(samples, rowvar=False)

    def sample(self, key, num_samples=None):
        """
        Sample from the Gaussian proposal distribution.

        PARAMETERS
        ----------
        key : jax.random.PRNGKey
            The random key for reproducibility.
        num_samples : int, optional
            The number of samples to generate. 
            Default is None, which generates a single sample.

        RETURNS
        -------
        samples : jnp.ndarray, shape=(num_samples, self.dim) or (self.dim,)
            Samples from the Gaussian proposal
        """
        if num_samples is None:
            shape = ()
        else:
            shape = (int(num_samples),)
        x = jax.random.multivariate_normal(
            key, self.mu, self.cov, shape=shape
        )

        if self.bounds is not None:
            # Use logit, includes rescaling from [0, 1)
            x = inv_logit(x, bounds=self.bounds)[0]
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
            # Use logit, includes rescaling from [0, 1)
            x, log_j = logit(x, bounds=self.bounds)
        else:
            log_j = jnp.zeros(x.shape[:-1])

        return multivariate_normal.logpdf(
            x, mean=self.mu, cov=self.cov
        ) + log_j

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
