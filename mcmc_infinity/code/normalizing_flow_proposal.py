
import jax.numpy as jnp
import jax.random as jrandom
from flowjax.train import fit_to_data


from typing import Callable

import flowjax.bijections
import flowjax.distributions
import flowjax.flows
import jax
import jax.numpy as jnp
import jax.random as jrandom


def get_flow_function_class(name: str) -> Callable:
    try:
        return getattr(flowjax.flows, name)
    except AttributeError:
        raise ValueError(f"Unknown flow function: {name}")


def get_bijection_class(name: str) -> Callable:
    try:
        return getattr(flowjax.bijections, name)
    except AttributeError:
        raise ValueError(f"Unknown bijection: {name}")


def get_flow(
    *,
    key: jax.Array,
    dims: int,
    flow_type: str | Callable = "masked_autoregressive_flow",
    bijection_type: str | flowjax.bijections.AbstractBijection | None = None,
    bijection_kwargs: dict | None = None,
    **kwargs,
) -> flowjax.distributions.Transformed:
    if isinstance(flow_type, str):
        flow_type = get_flow_function_class(flow_type)

    if isinstance(bijection_type, str):
        bijection_type = get_bijection_class(bijection_type)
    if bijection_type is not None:
        transformer = bijection_type(**bijection_kwargs)
    else:
        transformer = None

    if bijection_kwargs is None:
        bijection_kwargs = {}

    base_dist = flowjax.distributions.Normal(jnp.zeros(dims))
    key, subkey = jrandom.split(key)
    return flow_type(
        subkey,
        base_dist=base_dist,
        transformer=transformer,
        **kwargs,
    )


def get_annealed_flow(flow, scale: float = 1.0) -> flowjax.distributions.Transformed:
    """
    Returns a new flow that is an annealed version of the input flow.
    The new flow scales the base distribution by the given scale factor.
    """
    if not isinstance(flow, flowjax.distributions.Transformed):
        raise ValueError("Input flow must be a Transformed distribution.")
    base_dist = flow.base_dist
    new_base_dist = flowjax.distributions.Normal(
        loc=base_dist.loc,
        scale=scale,
    )
    return flowjax.distributions.Transformed(
        base_dist=new_base_dist,
        bijection=flow.bijection,
    )


class NormalizingFlowProposal:

    def __init__(self, dim: int, bounds=None, key=None, **kwargs):
        if key is None:
            key = jrandom.key(0)
            print(
                "The key argument is None. "
                "A random key will be used for the flow. "
                "Results may not be reproducible."
            )
        self.loc = None
        self.scale = None
        self._flow = get_flow(
            key=key,
            dims=dim,
            **kwargs,
        )
        # (dims, 2) bounds
        self.dim = dim
        self.bounds = jnp.array(bounds) if bounds is not None else None
        self.flow_scale = 1.0  # Default scale for annealing

    def fit(self, x, key, **kwargs):
        x = jnp.asarray(x)
        if self.bounds is not None:
            # Rescale and apply logit
            assert x.shape[-1] == self.bounds.shape[0], "x and bounds must have the same dimensionality"
            x = jnp.clip(x, self.bounds[:, 0], self.bounds[:, 1])
            x = (x - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
            x = jnp.log(x / (1 - x))
        self._flow, losses = fit_to_data(key, self._flow, x, **kwargs)
        return losses

    def set_flow_annealing(self, scale: float = 1.0):
        """
        Set the flow to an annealed version with the given scale.
        """
        self.flow_scale = scale

    @property
    def annealed_flow(self):
        """
        Returns an annealed version of the flow with scale=1.0.
        """
        return get_annealed_flow(self._flow, scale=self.flow_scale)

    def sample(self, key, num_samples=None):
        """
        Generate samples from the normalizing flow proposal distribution.

        INPUTS
        ------
        subkey : jax.random.PRNGKey
            The random key for reproducibility.
        num_samples : int, optional
            The number of samples to generate.
            Default is None, which generates a single sample.

        RETURNS
        -------
        samples : jnp.ndarray
            Samples from the normalizing flow proposal distribution.
            Dhape=(num_samples, self.dim) or (self.dim,). 
        """
        if num_samples is None:
            num_samples = 1
        shape = (int(num_samples),)

        x = self.annealed_flow.sample(key, shape)
        if self.bounds is not None:
            # Apply inverse logit and rescale
            x = jnp.clip(x, -10, 10)
            x = jnp.exp(x) / (1 + jnp.exp(x))
            x = x * (self.bounds[:, 1] - self.bounds[:, 0])
            x = x + self.bounds[:, 0]
        return jnp.squeeze(x)

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
        x = jnp.atleast_2d(x)
        assert x.shape[-1] == self.dim, "wrong dimensionality"
        log_prob = self.annealed_flow.log_prob(x)
        if self.bounds is not None:
            # Apply logit transformation
            x = jnp.clip(x, self.bounds[:, 0], self.bounds[:, 1])
            x = (x - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
            x = jnp.log(x / (1 - x))
            log_prob += jnp.sum(jnp.log(self.bounds[:, 1] - self.bounds[:, 0]))
        return log_prob
    
    def __call__(self, x):
        """
        Call the logP method for convenience.
        """
        return self.logP(x)
    