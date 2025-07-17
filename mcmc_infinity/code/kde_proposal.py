import jax
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde


class KernelDensityEstimateProposal:
    """
    Kernel Density Estimate (KDE) proposal distribution for perfect MCMC.
    """

    def __init__(self, 
                 dim, 
                 inflation_scale=1.0):
        """
        INPUTS
        ------
        dim : int
            The dimensionality of the input space.
        inflation_scale : float
            The scale factor for the covariance matrix.
        """
        self.dim = int(dim)
        self.inflation_scale = float(inflation_scale)


    def fit(self, x, **kwargs):
        """
        Fit the model to the training data.

        INPUTS
        ------
        x : jnp.ndarray, shape=(num_samples, dim)
            The training data to fit the model.
        kwargs : dict
            Additional keyword arguments passed to jax.scipy.stats.gaussian_kde.
        """
        x = jnp.asarray(x)

        self.kde = gaussian_kde(x.T, **kwargs)

        self.kde._setattr('covariance', self.kde.covariance * self.inflation_scale**2)
        self.kde._setattr('inv_cov', self.kde.inv_cov / self.inflation_scale**2)

    def sample(self, key, num_samples=None):
        """
        Generate samples from the kde distribution.

        INPUTS
        ------
        key : jax.random.PRNGKey
            The random key for reproducibility.
        num_samples : int, optional
            The number of samples to generate.
            Default is None, which generates a single sample.

        RETURNS
        -------
        samples : jnp.ndarray, shape=(num_samples, dim)
            Samples from the normalizing flow proposal distribution.
        """
        samples = self.kde.resample(key, shape=(num_samples,))

        return samples.T

    def logP(self, x):
        """
        INPUTS:
        -------
        x : array-like, shape=(..., self.dim)
            An array of inputs to the log density function.

        RETURNS
        -------
        logl : array-like, shape=(...,)
            The log-density of the normalizing flow proposal function.
        """
        x = jnp.asarray(x)
        return self.kde.logpdf(x.T)
    
    def __call__(self, x):
        """
        Call the logP method for convenience.
        """
        return self.logP(x)
    

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    dim = 2

    training_samples = np.random.normal(size=(1000, dim))

    Q = KernelDensityEstimateProposal(dim, inflation_scale=10.0)

    Q.fit(training_samples)

    proposal_samples = Q.sample(jax.random.PRNGKey(0), num_samples=1000)

    # evaluate the inflated kde logP method at the training samples
    print(f"Testing logP method: {Q(training_samples).shape}")

    fig, ax = plt.subplots()
    ax.scatter(training_samples[:, 0], 
               training_samples[:, 1], 
               label='Training Samples', 
               alpha=0.5, s=5,
               color='blue')
    ax.scatter(proposal_samples[:, 0], 
               proposal_samples[:, 1], 
               label='Proposal Samples (inflated)', 
               alpha=0.5, s=5,
               color='red')
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.legend()
    ax.set_aspect('equal')
    plt.show()
