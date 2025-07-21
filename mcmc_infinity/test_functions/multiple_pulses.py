import jax.numpy as jnp 
import matplotlib.pyplot as plt
import numpy as np 

def gaussian_pulse(x, a, b, c):
    f_x = a * jnp.exp(-((x - b) ** 2) / (2 * c ** 2))
    return f_x

def combine_gaussians(t, params):
    template = jnp.zeros_like(t)
    for param in params:
        template += gaussian_pulse(t, *param)  # *params -> a, b, c
    return template

def log_like_fn_gauss_pulse(params, t, data, sigma):
    template = combine_gaussians(t, params)
    ll = -0.5 * jnp.sum(((template - data) / sigma) ** 2, axis=-1)
    return ll

# define time stream
num = 500
t = jnp.linspace(-1, 1, num)

gauss_inj_params = [
    [3.3, -0.2, 0.1],
    [2.6, -0.1, 0.1],
    [3.4, 0.0, 0.1],
    [2.9, 0.3, 0.1],
]

# combine gaussians
injection = combine_gaussians(t, jnp.asarray(gauss_inj_params))

# set noise level
sigma = 2.0

# produce full data
y = injection + sigma * np.random.randn(len(injection))
y = jnp.array(y)

# plt.plot(t, y, label="data", color="lightskyblue")
# plt.plot(t, injection, label="injection", color="crimson")
# plt.legend();
# plt.show()


class pulses:
    """
    A class representing the likelihood for the Gaussian pulses problem.
    It accepts arrays of pulse parameters and computes the likelihood 
    after summing the resulting Gaussian pulses.
    """

    def __init__(self, data, sigma=2.0):
        """
        INPUTS:
        -------
        data : array-like, shape=(...,)
            The time series data set.
        sigma : float
            The noise variance to be used for the likelihood calculation.
        """
        self.data = data
        self.sigma = sigma
        assert self.sigma > 0, "sigma must be positive"

    def logP(self, x):
        """
        INPUTS:
        -------
        x : array-like, shape=(..., dim)
            An array of inputs to the likelihood function.

        RETURNS
        -------
        logl : array-like, shape=(...,)
            The log-posterior of the target function.
        """
        x = jnp.asarray(x)
        template = combine_gaussians(t, x)
        logl = -0.5 * jnp.sum(((template - self.data) / self.sigma) ** 2, axis=-1)
        return logl

    def __call__(self, x):
        return self.logP(x)

