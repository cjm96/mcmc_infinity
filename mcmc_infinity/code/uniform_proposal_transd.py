import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


class UniformProposalTransD:
    """
    Generates samples from a distribution that is uniform in a d-dimensional
    hypercube.
    """

    def __init__(self, dim, bounds, kbounds):
        """
        INPUTS
        ------
        dim : int
            The dimension of the proposal distribution.
        bounds : jnp.ndarray, shape=(dim, 2)
            The boundaries of the regon of support,
            of the form [(x0_min, x0_min), (x1_min, x1_max), ...].
            Each tuple specifies the lower and upper bounds for each dimension.
        kbounds : jnp.ndarray, shape=(dim, 1)
            The boundaries for the number of components for the nested model.
        """
        self.dim = int(dim)
        self.bounds = jnp.array(bounds)
        self.kbounds = jnp.array(kbounds)
        self.k = None

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
            shape=(num_samples, self.dim) or (self.dim,) if num_samples is None. 
        """
        if num_samples is None:
            shape = (self.dim,)
            num_samples = 1
        else:
            shape = (int(num_samples), self.dim)
        
        # Initialize with nans
        samples = jnp.full( (int(num_samples), self.kbounds[1], self.dim,), jnp.nan)

        # Here we should do it "birth-death"-like in order to be consistent,
        # but I believe it should work as well, just far more inefficiently 
        self.k = jax.random.randint(key, int(num_samples), self.kbounds[0], self.kbounds[1],)

        for i in range(num_samples):
            for ki in range(self.k[i]):
                key, subkey = jax.random.split(key)
                samples_ki = jax.random.uniform(key, shape=(self.dim, ))
                samples_ki = samples_ki*(self.bounds[:,1]-self.bounds[:,0])+self.bounds[:,0]
                samples = samples.at[i, ki, :].set(samples_ki)
        return samples
    
    def logP(self, x):
        """
        Compute the log-density of the uniform proposal distribution.

        INPUTS:
        -------
        x : array-like, shape=(num_points, self.dim) or (self.dim,)
            An array of inputs to the log density function.

        RETURNS
        -------
        logl : array-like, shape=(num_points,) or ()
            The log-density of the uniform proposal function.
        """
        x = jnp.asarray(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"
        logl = self.norm * self.k # Is this correct?
        if x.ndim == 1:
            return logl
        else:
            return logl * jnp.ones(x.shape[0], dtype=x.dtype) * self.k # Is this correct?

    def __call__(self, x):
        return self.logP(x)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    dim = 2
    bounds = jnp.array([[-5, 5], [-5, 5]])
    kbounds = jnp.array([1, 5])

    U = UniformProposalTransD(dim, bounds, kbounds)

    key = jax.random.key(0)

    key, subkey = jax.random.split(key)
    samples = U.sample(key, num_samples=1000)

    samples = samples.reshape(-1, dim) # reshape, put all components in rows (even empty slots)

    mask = ~jnp.any(jnp.isnan(samples), axis=1) # handle empty entries of components (nans)
    samples = samples[mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    ax.set_xlim(1.1*bounds[0, 0], 1.1*bounds[0, 1])
    ax.set_ylim(1.1*bounds[1, 0], 1.1*bounds[1, 1])
    ax.set_title("Uniform Proposal Samples")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect('equal')
    plt.show()

