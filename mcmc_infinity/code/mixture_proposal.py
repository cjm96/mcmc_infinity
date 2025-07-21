import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


class MixtureProposal:
    """
    A class for a mixture proposal distribution.

    This class is used to create a proposal distribution that is a mixture of multiple distributions.
    """

    def __init__(self, *proposals, weights=None):
        """
        INPUTS
        ------
        proposals : list of Proposal objects
            A list of proposal distributions to be mixed.
        weights : jnp.ndarray, optional
            The weights for each proposal distribution in the mixture.
            If None, equal weights are assigned to each proposal.
        """
        self.proposals = proposals
        self.n_proposals = len(proposals)
        self.dim = proposals[0].dim
        for proposal in proposals:
            assert proposal.dim == self.dim, \
                "All proposals must have the same dimension."
        if weights is None:
            self.weights = jnp.ones(len(proposals)) / len(proposals)
        else:
            self.weights = jnp.array(weights)
            assert len(self.weights) == len(proposals), \
                "Weights must match the number of proposals."
            self.weights /= jnp.sum(self.weights)

    def sample(self, key, num_samples=None):
        """
        Generate samples from the mixture proposal distribution.

        INPUTS
        ------
        key : jax.random.PRNGKey
            The random key for reproducibility.
        num_samples : int, optional
            The number of samples to generate. Default is None, which generates a single sample.

        RETURNS
        -------
        samples : jnp.ndarray
            Samples from the mixture proposal distribution.
            Shape=(num_samples, self.dim) or (self.dim,).
        """
        if num_samples is None:
            num_samples = 1

        # Sample from the mixture distribution
        n_samples_per_proposal = jax.random.multinomial(
            key, num_samples, self.weights, dtype=jnp.int32
        )
        keys = jax.random.split(key, self.n_proposals)
        samples = jnp.concatenate([
            proposal.sample(k, n) if n > 0 else jnp.empty((0, self.dim))
            for proposal, n, k in zip(self.proposals, n_samples_per_proposal, keys)
        ], axis=0)
        return samples

    def logP(self, x):
        """
        Compute the log density of the mixture proposal distribution.

        INPUTS:
        -------
        x : array-like, shape=(..., self.dim)
            An array of inputs to the log density function.

        RETURNS
        -------
        logl : array-like, shape=(...,)
            The log-density of the mixture proposal function.
        """
        x = jnp.asarray(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"
        # Compute the log density for each proposal
        log_probs = jnp.array([jnp.atleast_1d(proposal.logP(x)) for proposal in self.proposals])
        # Weight the log densities by the mixture weights
        log_prob = logsumexp(log_probs.T, b=self.weights, axis=1)
        return log_prob

    def __call__(self, x):
        """
        Call the logP method to compute the log density.

        INPUTS:
        -------
        x : array-like, shape=(..., self.dim)
            An array of inputs to the log density function.

        RETURNS
        -------
        logl : array-like, shape=(...,)
            The log-density of the mixture proposal function.
        """
        return self.logP(x)


if __name__ == "__main__":
    # Example usage with uniform and symmetric Gaussian proposals
    from mcmc_infinity.code.uniform_proposal import UniformProposal
    from mcmc_infinity.code.gaussian_proposal import DiagonalGaussianProposal, GaussianProposal
    from mcmc_infinity.code.normalizing_flow_proposal import NormalizingFlowProposal
    import matplotlib.pyplot as plt

    proposals = [
        UniformProposal(dim=2, bounds=jnp.array([[-5, 5], [-5, 5]])),
        DiagonalGaussianProposal(dim=2, mu=0, sigma=1),
        GaussianProposal(dim=2),
        NormalizingFlowProposal(dim=2, bounds=jnp.array([[-5, 5], [-5, 5]]))
    ]

    samples = jax.random.normal(jax.random.key(0), shape=(100, 2))

    proposals[2].fit(samples)
    proposals[3].fit(samples, key=jax.random.key(0))

    proposal = MixtureProposal(*proposals)

    x = proposal.sample(jax.random.key(0))
    proposal.logP(x)

    n = 1000
    samples = proposal.sample(jax.random.key(0), num_samples=n)
    assert samples.shape == (n, proposal.dim), f"Sample dimension mismatch: {samples.shape} != {(n, proposal.dim)}"
    log_probs = proposal.logP(samples)

    fig = plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1], c=log_probs, cmap='viridis', s=1)
    plt.colorbar(label='Log Probability Density')
    plt.title('Samples from Mixture Proposal Distribution')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
