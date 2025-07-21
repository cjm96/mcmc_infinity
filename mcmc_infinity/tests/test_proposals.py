import pytest
import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import mcmc_infinity as mc

def test_proposal_shapes():
    """
    Test that all the proposal distributions .logP and .sample methods accept 
    and return jax arrays of the correct shapes.
    """
    dim = 2

    key = jax.random.key(123)

    bounds = jnp.array([[0., 1.], [0., 1.]])

    proposals = [ mc.uniform_proposal.UniformProposal(dim, bounds) ]

    for proposal in proposals:
        # Test .logP method
        x = 0.5 * jnp.ones((1, dim))
        logP = proposal.logP(x)
        assert logP.shape == (1,), \
                f"Unexpected logP shape: expected (1,) got {logP.shape}"

        x = 0.5 * jnp.ones(dim)
        logP = proposal.logP(x)
        assert logP.shape == (), \
                f"Unexpected logP shape: expected () got {logP.shape}"

        x = 0.5 * jnp.ones((10, dim))
        logP = proposal.logP(x)
        assert logP.shape == (10,), \
            f"Unexpected logP shape: expected (10,) got {logP.shape}"

        # Test .sample method
        sample = proposal.sample(key)
        assert sample.shape == (dim,), \
            f"Unexpected sample shape: expected ({dim},) got {sample.shape}"
        
        sample = proposal.sample(key, num_samples=1)
        assert sample.shape == (1, dim), \
            f"Unexpected sample shape: expected (1, {dim}) got {sample.shape}"
        
        sample = proposal.sample(key, num_samples=10)
        assert sample.shape == (10, dim), \
            f"Unexpected sample shape: expected (10, {dim}) got {sample.shape}"