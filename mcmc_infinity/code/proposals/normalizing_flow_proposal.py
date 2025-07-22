
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom
from flowjax.train import fit_to_data
from typing import Any, Callable, Dict, Optional, Union

import flowjax.bijections
import flowjax.distributions
import flowjax.flows

from ..utils import logit, inv_logit


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
    flow_type: Union[str, Callable] = "masked_autoregressive_flow",
    bijection_type: Union[
        str, flowjax.bijections.AbstractBijection, None
    ] = None,
    bijection_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
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


def get_annealed_flow(
    flow: flowjax.distributions.Transformed, scale: float = 1.0
) -> flowjax.distributions.Transformed:
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

    def __init__(
        self,
        dim: int,
        bounds: Optional[Union[jnp.ndarray, list]] = None,
        key: Optional[jax.Array] = None,
        rescale: bool = True,
        **kwargs: Any,
    ) -> None:
        if key is None:
            key = jrandom.key(0)
            print(
                "The key argument is None. "
                "A random key will be used for the flow. "
                "Results may not be reproducible."
            )
        self._flow = get_flow(
            key=key,
            dims=dim,
            **kwargs,
        )
        # (dims, 2) bounds
        self.dim: int = dim
        self.bounds: Optional[jnp.ndarray] = (
            jnp.array(bounds) if bounds is not None else None
        )
        self.inflation_scale: float = 1.0  # Default scale for annealing
        self.rescale: bool = rescale
        # These will be set during fitting if rescale=True
        self.mean: Optional[jnp.ndarray] = None
        self.std: Optional[jnp.ndarray] = None

    def fit(
        self,
        x: jnp.ndarray,
        key: jax.Array,
        **kwargs: Any
    ) -> jnp.ndarray:
        x = jnp.asarray(x)
        if self.bounds is not None:
            # Rescale and apply logit
            assert x.shape[-1] == self.bounds.shape[0], (
                "x and bounds must have the same dimensionality"
            )
            x = logit(x, bounds=self.bounds)[0]

        # Rescale to zero mean and unit variance
        if self.rescale:
            self.mean = jnp.mean(x, axis=0)
            self.std = jnp.std(x, axis=0)
            x = (x - self.mean) / self.std
        self._flow, losses = fit_to_data(key, self._flow, x, **kwargs)
        return losses

    def set_inflation_scale(self, scale: float = 1.0) -> None:
        """
        Set the inflation scale for the flow.
        """
        self.inflation_scale = scale

    @property
    def annealed_flow(self) -> flowjax.distributions.Transformed:
        """
        Returns an annealed version of the flow with scale=1.0.
        """
        return get_annealed_flow(self._flow, scale=self.inflation_scale)

    def sample(
        self,
        key: jax.Array,
        num_samples: Optional[int] = None
    ) -> jnp.ndarray:
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
            Shape=(num_samples, self.dim) or (self.dim,).
        """
        if num_samples is None:
            shape = ()
        else:
            shape = (int(num_samples),)

        x = self.annealed_flow.sample(key, shape)

        if self.rescale:
            # Rescale back to original scale
            x = x * self.std + self.mean

        if self.bounds is not None:
            # Apply inverse logit transformation
            x = inv_logit(x, bounds=self.bounds)[0]
        return x

    def logP(self, x: jnp.ndarray) -> jnp.ndarray:
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
        assert x.shape[-1] == self.dim, "wrong dimensionality"
        log_j = jnp.zeros(x.shape[0]) if x.ndim > 1 else 0.0
        if self.bounds is not None:
            x, log_j = logit(x, bounds=self.bounds)
        if self.rescale:
            # Rescale to zero mean and unit variance
            x = (x - self.mean) / self.std
            log_j += jnp.sum(jnp.log(self.std))
        log_prob = self.annealed_flow.log_prob(x) + log_j
        return log_prob

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Call the logP method for convenience.
        """
        return self.logP(x)


if __name__ == "__main__":
    dim = 2
    bounds = jnp.array([[-5, 5], [-5, 5]])

    key = jrandom.key(0)
    proposal = NormalizingFlowProposal(dim, bounds=bounds, key=key)

    # Generate some random data for fitting
    data = jrandom.uniform(key, (1000, dim), minval=-5, maxval=5)
    proposal.fit(data, key)

    sample = proposal.sample(key)

    assert sample.shape == (dim,), "Sample shape mismatch"

    assert proposal.logP(sample).shape == (), "LogP shape mismatch"

    # Sample from the proposal
    samples = proposal.sample(key, num_samples=1000)

    assert samples.shape == (1000, dim), "Sampled shape mismatch"
    assert proposal.logP(samples).shape == (1000,), "LogP shape mismatch"
