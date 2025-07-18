import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


class UniformProposal:
    """
    Generates samples from a distribution that is uniform in a d-dimensional
    hypercube.
    """

    def __init__(self, dim, bounds):
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
        self.bounds = jnp.array(bounds)

        assert self.bounds.shape == (self.dim, 2), \
            f"Bounds must have shape ({self.dim}, 2), got {self.bounds.shape}."
        
        self.norm = -jnp.sum(jnp.log(jnp.ptp(self.bounds, axis=1)))
        
    def sample(self, key, num_samples=None):
        """
        Generate samples from the uniform proposal distribution.

        INPUTS
        ------
        key : jax.random.PRNGKey
            The random key for reproducibility.
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
        samples = jax.random.uniform(key, shape=shape)
        samples = samples*(self.bounds[:,1]-self.bounds[:,0])+self.bounds[:,0]
        return samples
    
    def logP(self, x):
        """
        INPUTS:
        -------
        x : array-like, shape=(..., self.dim)
            An array of inputs to the log density function.

        RETURNS
        -------
        logl : array-like, shape=(...,)
            The log-density of the uniform proposal function.
        """
        x = jnp.asarray(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"
        logl = self.norm
        return logl * jnp.ones(x.shape[:-1], dtype=x.dtype)

    def __call__(self, x):
        return self.logP(x)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    dim = 2
    bounds = jnp.array([[-5, 5], [-5, 5]])

    U = UniformProposal(dim, bounds)

    key = jax.random.key(0)

    key, subkey = jax.random.split(key)
    samples = U.sample(key, num_samples=1000)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    ax.set_xlim(1.1*bounds[0, 0], 1.1*bounds[0, 1])
    ax.set_ylim(1.1*bounds[1, 0], 1.1*bounds[1, 1])
    ax.set_title("Uniform Proposal Samples")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect('equal')
    plt.show()
