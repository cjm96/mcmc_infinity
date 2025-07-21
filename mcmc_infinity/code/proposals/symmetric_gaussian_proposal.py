import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


class SymmetricGaussianProposal:
    """
    DO NOT USE THIS FOR PERFECT SAMPLING.
    This is for demo and dev purposes only.
    You have been warned!

    Generates samples from a multivariate Gaussian that is centred on the 
    current point.
    """

    def __init__(self, dim, bounds=None):
        """
        INPUTS
        ------
        dim : int
            The dimension of the proposal distribution.
        bounds : jnp.ndarray, shape=(dim, 2)
            The boundaries of the regon of support,
            of the form [(x0_min, x0_min), (x1_min, x1_max), ...].
            Each tuple specifies the lower and upper bounds for each dimension.
        """
        self.dim = int(dim)

        if bounds is None:
            self.bounds = None
        else:
            bounds = jnp.asarray(bounds)
            assert bounds.shape==(self.dim, 2), \
                f"Bounds must be of shape ({self.dim}, 2), got {bounds.shape}."

    def sample(self, key, mu, sigma, num_samples=None):
        """
        Generate samples from the uniform proposal distribution.

        INPUTS
        ------
        key : jax.random.PRNGKey
            The random key for reproducibility.
        mu : jnp.ndarray, shape=(self.dim,)
            The mean of the Gaussian distribution, which is the current point.
        sigma : float
            The standard deviation of the Gaussian distribution. 
            Assumed to be isotropic, the same along all dimensions.
        num_samples : int, optional
            The number of samples to generate. 
            Default is None, which generates a single sample.

        RETURNS
        -------
        samples : jnp.ndarray
            Samples from the uniform proposal distribution.
            Dhape=(num_samples, self.dim) or (self.dim,). 
        """
        if num_samples is None:
            shape = (self.dim,)
        else:
            shape = (int(num_samples), self.dim)

        y = mu + sigma * jax.random.normal(key, shape=shape)

        return y
    
    def logP(self, x, mu, sigma):
        """
        INPUTS:
        -------
        x : array-like, shape=(..., self.dim)
            An array of inputs to the log density function.
        mu : jnp.ndarray, shape=(self.dim,)
            The mean of the Gaussian distribution.
        sigma : float
            The standard deviation of the Gaussian distribution.

        RETURNS
        -------
        logl : array-like, shape=(...,)
            The log-density of the Gaussian proposal function.
        """
        x = jnp.asarray(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"
        logl = -0.5 * jnp.sum(((x - mu) / sigma) ** 2, axis=-1) 
        return logl
    
    def __call__(self, x, mu, sigma):
        return self.logP(x, mu, sigma)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    dim = 2
    bounds = jnp.array([[-5, 5], [-5, 5]])

    Q = SymmetricGaussianProposal(dim, bounds)

    key = jax.random.key(0)

    key, subkey = jax.random.split(key)
    samples = Q.sample(key, mu=jnp.zeros(dim), sigma=1.0, num_samples=1000)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    ax.set_xlim(1.1*bounds[0, 0], 1.1*bounds[0, 1])
    ax.set_ylim(1.1*bounds[1, 0], 1.1*bounds[1, 1])
    ax.set_title("Gaussian Proposal Samples")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect('equal')
    plt.show()