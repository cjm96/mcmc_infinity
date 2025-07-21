from mcmc_infinity.code.utils import logit, inv_logit
import jax
import jax.numpy as jnp


def test_logit_inv_logit():
    """
    Test the logit and inv_logit functions for correctness.
    """
    x = jnp.array([0.1, 0.5, 0.9])

    # Test logit
    logit_x, logit_j = logit(x)
    assert jnp.allclose(logit_x, jax.scipy.special.logit(x))

    # Test inv_logit
    inv_x, inv_j = inv_logit(logit_x)
    assert jnp.allclose(inv_x, jax.scipy.special.expit(logit_x))
    assert jnp.allclose(inv_x, x)

    # Check the Jacobian determinants
    assert jnp.allclose(logit_j, -inv_j)

    grad_logit = jax.grad(jax.scipy.special.logit, argnums=0)
    log_j_true = jnp.log(jax.vmap(grad_logit)(x)).sum()
    assert jnp.allclose(logit_j, log_j_true)

    grad_expit = jax.grad(jax.scipy.special.expit, argnums=0)
    inv_j_true = jnp.log(jax.vmap(grad_expit)(logit_x)).sum()
    assert jnp.allclose(inv_j, inv_j_true)
