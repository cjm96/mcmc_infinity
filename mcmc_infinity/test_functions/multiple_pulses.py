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

class llh_pulses:
    """
    A class representing the likelihood for the Gaussian pulses problem.
    It accepts arrays of pulse parameters and computes the likelihood 
    after summing the resulting Gaussian pulses.
    """

    def __init__(self, t, data, sigma=2.0, dim=0):
        """
        INPUTS:
        -------
        data : array-like, shape=(...,)
            The time series data set.
        sigma : float
            The noise variance to be used for the likelihood calculation.
        """
        self.data = data
        self.t = t
        self.sigma = sigma
        self.dim = dim
        assert self.sigma > 0, "sigma must be positive"
        assert self.dim > 0, "Please provide the dim of the uni-component model (int)"

    def logP(self, x):
        """
        A Gaussian log-likelihood function that is computed given
        an arbitrary number of parameter sets for Gaussian pulses.

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
        x = x.reshape(-1, self.dim) # reshape, put all components in rows (even empty slots)
        mask = ~jnp.any(jnp.isnan(x), axis=1) # handle empty entries of components (nans)
        x = x[mask]
        template = combine_gaussians(self.t, x)
        logl = -0.5 * jnp.sum(((template - self.data) / self.sigma) ** 2, axis=-1)
        return logl

    def __call__(self, x):
        return self.logP(x)


def gen_data(num=100, make_plot=True):
    """
    A function that generates Gaussian pulses in white noise.

    INPUTS:
    -------
    num : int, 
        Number of data points in the time series.
    make_plot : bool, optional, 
        Flag to plot the data.

    RETURNS t, y, sigma, p_inj
    -------
    t : array-like, shape=(...,)
        The time vector of the data.
    y : array-like, shape=(...,)
        The y vector of the data.
    sigma : float, 
        The variance of the white noise.
    p_inj : array-like, shape=(...,)
        The injection values of the Gaussian pulses.
    """

    import matplotlib.pyplot as plt

    # define time stream
    t = jnp.linspace(-1, 1, num)

    # define some injection parameters
    p_inj = [
        [2.6, -0.1, 0.1],
        [3.4, 0.3, 0.15],
    ]

    # combine gaussian pulses
    injection = combine_gaussians(t, jnp.asarray(p_inj))

    # set noise level
    sigma = 2.0

    # produce full data
    y = injection + sigma * np.random.randn(len(injection))
    y = jnp.array(y)

    if make_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(t, y, label="data", color="lightskyblue")
        for individual_pulse_params in p_inj:
            plt.plot(t, combine_gaussians(t, jnp.asarray([individual_pulse_params])), color="gray", alpha=.5, linestyle="--")
        plt.plot(t, injection, label="injection", color="crimson")
        ax.legend(loc="upper right")
        ax.set_xlabel("t")
        ax.set_ylabel("A")
        ax.set_title("1D Gaussian pulses in white noise")
        plt.show()

    return t, y, sigma, p_inj

if __name__ == '__main__':
    t, y, s, i = gen_data()
    

# END