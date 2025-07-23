import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp


class MixtureProposal:
    """
    A class for a mixture proposal distribution.

    This class is used to create a proposal distribution
    that is a mixture of multiple distributions.
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

        # Pre-compile individual proposal sampling functions to avoid
        # recompilation issues
        self._compiled_samplers = [
            jax.jit(lambda key, prop=prop: prop.sample(key))
            for prop in self.proposals
        ]
        
        # Create a compiled function for each proposal's logP
        self._compiled_logPs = [
            jax.jit(lambda x, prop=prop: prop.logP(x))
            for prop in self.proposals
        ]

    def sample(self, key, num_samples=None):
        """
        Generate samples from the mixture proposal distribution.

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
            Samples from the mixture proposal distribution.
            Shape=(num_samples, self.dim) or (self.dim,).
        """
        if num_samples is None:
            # For single sample - choose a proposal and sample from it directly
            key_choice, key_sample = jax.random.split(key)
            choice = jax.random.choice(key_choice, self.n_proposals,
                                       p=self.weights)
            
            # Sample from all proposals and select the chosen one
            # This approach ensures static shape compilation
            samples = jnp.stack([
                self._compiled_samplers[i](key_sample)
                for i in range(self.n_proposals)
            ])
            return samples[choice]
        else:
            n = int(num_samples)
            # For multiple samples
            key_choices, key_samples = jax.random.split(key)
            
            # Choose which proposal to use for each sample
            choices = jax.random.choice(key_choices, self.n_proposals,
                                        shape=(n,), p=self.weights)

            # Generate samples using vmap
            sample_keys = jax.random.split(key_samples, n)

            def sample_single(choice, sample_key):
                # Sample from all proposals and select the chosen one
                samples = jnp.stack([
                    self._compiled_samplers[i](sample_key)
                    for i in range(self.n_proposals)
                ])
                return samples[choice]

            return jax.vmap(sample_single)(choices, sample_keys)

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
        
        # Compute the log density for each proposal efficiently
        # using pre-compiled functions
        log_probs = jnp.stack([
            jnp.atleast_1d(self._compiled_logPs[i](x))
            for i in range(self.n_proposals)
        ])
        
        # Weight the log densities by the mixture weights
        log_prob = logsumexp(log_probs.T, b=self.weights, axis=1)
        
        if x.ndim == 1:
            return log_prob[0]
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
    from mcmc_infinity.code.proposals.uniform_proposal import (
        UniformProposal)
    from mcmc_infinity.code.proposals.gaussian_proposal import (
        GaussianProposal)
    from mcmc_infinity.code.proposals.normalizing_flow_proposal import (
        NormalizingFlowProposal)

    proposals = [
        UniformProposal(dim=2, bounds=jnp.array([[-5, 5], [-5, 5]])),
        GaussianProposal(dim=2),
        NormalizingFlowProposal(dim=2, bounds=jnp.array([[-5, 5], [-5, 5]]))
    ]

    # Fit proposals that need fitting
    samples = jax.random.normal(jax.random.key(0), shape=(100, 2))
    proposals[1].fit(samples)
    proposals[2].fit(samples, key=jax.random.key(0))

    proposal = MixtureProposal(*proposals)

    x = proposal.sample(jax.random.key(0))
    assert x.shape == (2,), (
        "Sample shape mismatch: expected (2,), got {}".format(x.shape))
    assert proposal.logP(x).shape == (), (
        "LogP shape mismatch: expected (), got {}".format(
            proposal.logP(x).shape))

    n = 1000
    samples = proposal.sample(jax.random.key(0), num_samples=n)
    assert samples.shape == (n, proposal.dim), (
        f"Sample dimension mismatch: "
        f"{samples.shape} != {(n, proposal.dim)}")
    log_probs = proposal.logP(samples)
    assert log_probs.shape == (n,), (
        f"LogP shape mismatch: {log_probs.shape} != {(n,)}")

    print("All tests passed!")
