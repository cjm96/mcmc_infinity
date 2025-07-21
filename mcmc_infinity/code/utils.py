import jax.numpy as jnp


def logit(x, bounds=None):
    """Logit function.

    """
    log_j = jnp.zeros(x.shape[0]) if x.ndim > 1 else 0
    if bounds is not None:
        x = (x - bounds[..., 0]) / (bounds[..., 1] - bounds[..., 0])
        log_j = log_j - jnp.log(bounds[..., 1] - bounds[..., 0]).sum()
    log_j = log_j + (-jnp.log(x) - jnp.log1p(-x)).sum(axis=-1)
    x = jnp.log(x / (1 - x))
    return x, log_j


def inv_logit(x, bounds=None):
    """Inverse logit function.

    """
    x = 1 / (1 + jnp.exp(-x))
    log_j = (jnp.log(x) + jnp.log1p(-x)).sum(axis=-1)
    if bounds is not None:
        x = x * (bounds[..., 1] - bounds[..., 0]) + bounds[..., 0]
        log_j = log_j + jnp.log(bounds[..., 1] - bounds[..., 0]).sum()
    return x, log_j
