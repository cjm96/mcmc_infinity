import jax.numpy as jnp 
import numpy as np 

def gaussian_pulse(x, a, b, c):
    """
    A single Gaussian pulse

    Args:
        x (array-like): The abscissa pf the data
        a (float): The amplitude of the pules
        b (float): the position of the pulse
        c (float): The spread of the pulse

    Returns:
        array-like: The resulting y-values for the given x-vector 
    """
    f_x = a * jnp.exp(-((x - b) ** 2) / (2 * c ** 2))
    return f_x

def combine_gaussians(t, params):
    """
    Utility function to combine pulses given the number of parameter 
    sets that are given as input.

    Args:
        t (array-like): The time vector
        params (array-like): The parameter vector

    Returns:
        array-like: The resulting y-values for the given x-vector
    """
    template = jnp.zeros_like(t)
    for param in params:
        template += gaussian_pulse(t, *param)  # *params -> a, b, c
    return template

def log_like_fn_gauss_pulse(params, t, data, sigma):
    """
    A Gaussian log-likelihood function that is computed given
    an arbitrary number of parameter sets for Gaussian pulses.

    Args:
        params (array-like): The parameter vector
        t (array-like): The time vector
        data (array-like): The data vector
        sigma (float): The noise variance, which is considered known

    Returns:
        float: The log-likelihood value 
    """
    template = combine_gaussians(t, params)
    ll = -0.5 * jnp.sum(((template - data) / sigma) ** 2, axis=-1)
    return ll

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


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # define time stream
    num = 500
    t = jnp.linspace(-1, 1, num)

    # define some injection parameters
    gauss_inj_params = [
        [3.3, -0.2, 0.1],
        [2.6, -0.1, 0.1],
        [3.4, 0.0, 0.1],
        [2.9, 0.3, 0.1],
    ]

    # combine gaussian pulses
    injection = combine_gaussians(t, jnp.asarray(gauss_inj_params))

    # set noise level
    sigma = 2.0

    # produce full data
    y = injection + sigma * np.random.randn(len(injection))
    y = jnp.array(y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, y, label="data", color="lightskyblue")
    plt.plot(t, injection, label="injection", color="crimson")
    ax.legend(loc="upper right")
    ax.set_xlabel("t")
    ax.set_ylabel("A")
    ax.set_title("1D Gaussian pulses in white noise")
    plt.show()

# END