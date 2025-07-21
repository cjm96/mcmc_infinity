import pytest
import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import mcmc_infinity as mc

PROPOSALS = [
    (mc.uniform_proposal.UniformProposal, {}),
    (mc.gaussian_proposal.GaussianProposal, {"inflation_scale": 1.0}),
    (mc.kde_proposal.KernelDensityEstimateProposal, {"inflation_scale": 1.0}),
    (mc.normalizing_flow_proposal.NormalizingFlowProposal, {"key": jax.random.key(0)}),
]


@pytest.fixture()
def dim():
    """
    Fixture to provide the dimension for the proposal distributions.
    """
    return 2


@pytest.fixture()
def bounds():
    """
    Fixture to provide the bounds for the proposal distributions.
    """
    return jnp.array([[0., 1.], [0., 1.]])


@pytest.fixture()
def key():
    """
    Fixture to provide a JAX random key for reproducibility.
    """
    return jax.random.key(0)


@pytest.fixture(params=PROPOSALS)
def proposal(request, dim, bounds):
    """
    Fixture to provide a proposal class for testing.
    """
    return request.param[0](dim=dim, bounds=bounds, **request.param[1])


def test_proposal_shapes(dim, key, proposal):
    """
    Test that all the proposal distributions .logP and .sample methods accept 
    and return jax arrays of the correct shapes.
    """

    if hasattr(proposal, 'fit'):
        # Fit the proposal if it has a fit method
        initial_samples = jax.random.uniform(key, shape=(100, dim))
        try:
            proposal.fit(initial_samples)
        except TypeError:
            proposal.fit(initial_samples, key=key)

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
