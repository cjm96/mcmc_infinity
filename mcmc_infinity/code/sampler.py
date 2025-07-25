import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import time
import tqdm

from .proposals.uniform_proposal import UniformProposal as Quniform
from .proposals.symmetric_gaussian_proposal import SymmetricGaussianProposal as Qsymgauss
from .proposals.normalizing_flow_proposal import NormalizingFlowProposal as Qflow
from .proposals.kde_proposal import KernelDensityEstimateProposal as Qkde
from .proposals.gaussian_proposal import GaussianProposal as Qgauss
from .proposals.mixture_proposal import MixtureProposal as Qmixture


class PerfectSampler:
    """
    A class for performing perfect sampling using Metropolis-Hastings MCMC 
    with a common proposal distribution.

    This class implements the SRS (Sequential Random Sampling) rule phi for 
    evolving the Markov chain. 

    Use the method self.get_samples(T, num_samples) to generate perfect samples 
    from the target distribution.
    """

    def __init__(self, target, proposal, initial_positions, 
                 seed=None, proposal_kwargs={}):
        """
        INPUTS:
        -------
        target : callable
            The log-posterior PDF of the target distribution.
        proposal : class
            The common proposal distribution.
            This must have a method `proposal.sample(subkey)` that returns a 
            sample from the proposal, jax.ndarray shape=(dim,).
        initial_positions : jnp.ndarray, shape=(num_chains, dim)
            Usually just a single MCMC is needed (num_chains=1) provided it is
            initialised at the peak of the function P/Q. It is the user's 
            responsibility to provide this correct starting location, 
            otherwise this will NOT give perfect samples! Optionally, multiple
            chains can be run from multiple initial positions (num_chains>=2).
        seed : int, optional
            The random seed for reproducibility. 
            Default is None, which uses a seed set using the clock.
        proposal_kwargs : dict, optional
            Additional arguments for the proposal distribution, if needed.
            E.g., for the SymmetricGaussianProposal {'sigma': 0.1}.
        """
        self.target = target

        self.proposal = proposal

        assert self.target.dim == self.proposal.dim, \
            "Target and proposal must have the same dimensionality."
        self.dim = self.target.dim

        self.initial_positions = jnp.asarray(initial_positions)
        self.num_chains, d = self.initial_positions.shape
        assert d == self.dim, \
            f"Initial position must have dimension ({self.dim},), got {d}."

        self.key = jax.random.key(int(time.time()) if seed is None else seed)

        self.proposal_kwargs = dict(proposal_kwargs)

    def phi(self, x, xi):
        """
        The Metropolis-Hastings SRS rule for evolving the Markov chain.

        ..math::
            x_{i+1} = \\phi(x_i, \\xi_{i+1})

        INPUTS:
        -------
        x : array-like, shape=(self.target.dim)
            The current position of the chain.
        xi : jax.numpy.uint64
            The random number used to seed the generator.

        RETURNS
        -------
        y : array-like, shape=(self.target.dim)
            The next position of the chain.
        accept : bool
            Whether the proposed move was accepted.
        """
        if isinstance(self.proposal, Quniform):
            Qargs = ()
        elif isinstance(self.proposal, Qsymgauss):
            Qargs = (x, self.proposal_kwargs['sigma'])
        elif isinstance(self.proposal, Qflow):
            Qargs = ()
        elif isinstance(self.proposal, Qkde):
            Qargs = ()
        elif isinstance(self.proposal, Qgauss):
            Qargs = ()
        elif isinstance(self.proposal, Qmixture):
            Qargs = ()
        else:
            raise ValueError(f"Unrecognised proposal type {self.proposal_type}")

        key = jax.random.key(xi)
        key, subkey = jax.random.split(key)
        y = jnp.squeeze(self.proposal.sample(subkey, *Qargs))

        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey)

        a = ( self.target(y) - self.proposal(y, *Qargs) ) - \
                ( self.target(x) - self.proposal(x, *Qargs) )

        if jnp.log(u) < a:
            return y, True
        else:
            return x, False
        
    def evolve_chain(self, start, num_steps):
        """
        Run the sampler for a specified number of iterations from start.

        THIS DOESN'T DO PERFECT SAMPLING - JUST A NORMAL MCMC FOR TESTING.

        INPUTS:
        -------
        start : array-like, shape=(self.target.dim,)
            The initial position of the chain.
        num_steps : int
            The number of steps to run the sampler.

        RETURNS
        -------
        chain: jnp.ndarray, shape=(num_steps, self.target.dim)
            The Markov chain of samples.
        acceptance: float
            The acceptance fraction.
        """
        start = jnp.asarray(start)
        assert start.shape == (self.target.dim,), \
            f"Start must have shape ({self.target.dim},), got {start.shape}."

        self.key, subkey = jax.random.split(self.key)
        seeds = jax.random.randint(subkey, 
                                   (num_steps,), 
                                   0, jnp.iinfo(jnp.int64).max, 
                                   dtype=jnp.uint64)

        chain = jnp.zeros((num_steps+1, self.target.dim))

        chain = chain.at[0].set(start)

        acceptance = 0

        for i in range(num_steps):
            y, a = self.phi(chain[i], seeds[i])
            chain = chain.at[i+1].set(y)
            acceptance += int(a)

        acceptance = acceptance/num_steps
        
        return chain, acceptance
    
    def try_mcmc(self, seeds):
        """
        Evolve all the MCMC chains using random seeds provided. 

        INPUTS
        ------
        seeds : jnp.ndarray, shape=(T,), dtype=jnp.uint64
            Random seeds for each step of the MCMC chains.
            The length of this array determines the number of steps T.

        RETURNS
        -------
        chains : jnp.ndarray, shape=(self.num_chains, T+1, self.dim)
            The Markov chains evolved from the initial positions.
        """
        T = len(seeds)

        chains = jnp.zeros((self.num_chains, T+1, self.dim))
        chains = chains.at[:,0,:].set(self.initial_positions)

        for i in range(T):
            for c, chain in enumerate(chains):
                pos, a = self.phi(chains[c, i, :], seeds[i])
                chains = chains.at[c, i+1, ...].set(pos)

        return chains

    def get_perfect_sample(self, T, show_all_output=False, verbose=False, return_T=False):
        """
        Generate a single perfect sample from the target distribution.

        INPUTS:
        -------
        T : int
            The initial number of steps to try.
        show_all_output : bool, optional
            If True, return all output chains for diagnosing problems.
            Default is False, which returns only the final sample.
        verbose : bool, optional
            If True, print the current number of steps being tried.
            Default is False.
        return_T : bool, optional
            If True, return the final number of steps T used.
            Default is False, which returns only the sample.

        RETURNS
        -------
        sample : jnp.ndarray, shape=(self.dim,)
            A single sample from the target distribution.
        T : int, optional
            Only returned if return_T is True. The final number of steps used.
        all_output : list of jnp.ndarray
            Only returned if show_all_output is True.
        """
        coupled = False

        seeds = jnp.array([], dtype=jnp.uint64)

        T = int(T)

        all_output = []

        while not coupled:
            if verbose:
                print(f"Trying T={T} steps...")
            self.key, subkey = jax.random.split(self.key)
            new_seeds = jax.random.randint(subkey, 
                                   (T//2,), 
                                   0, jnp.iinfo(jnp.int64).max, 
                                   dtype=jnp.uint64)

            seeds = jnp.concatenate((new_seeds, seeds))

            chains = self.try_mcmc(seeds)

            if show_all_output:
                all_output.append(chains)

            if self.num_chains > 1:
                coupled = jnp.all(jnp.all(chains[:,-1,:]==chains[0,-1,:], 
                                          axis=0))
            else:
                coupled = jnp.any(jnp.not_equal(chains[0,-1,:], chains[0,0,:]))

            T *= 2

        sample = chains[0,-1,:]

        outputs = (sample,)
        if return_T:
            outputs += (T/2,)
        if show_all_output:
            outputs += (all_output,)
        return outputs

    def get_perfect_samples(self, T, num_samples, verbose=False, return_T=False):
        """
        Generate perfect samples from the target distribution.

        INPUTS:
        -------
        T : int
            The initial number of steps to try for each sample.
        num_samples : int
            The number of samples to generate. 
        verbose : bool, optional
            If True, print the current number of steps being tried.
            Default is False.
        return_T : bool, optional
            If True, return the final number of steps T used.
            Default is False, which returns only the sample.

        RETURNS
        -------
        samples : jnp.ndarray, shape=(num_samples, self.dim)
            I.i.d. perfect samples from the target distribution.
        t_vals : jnp.ndarray, shape=(num_samples,), dtype=jnp.int32
            Only returned if return_T is True. 
            The number of steps T used for each sample.
        """
        samples = jnp.zeros((num_samples, self.dim))
        t_vals = jnp.zeros((num_samples,), dtype=jnp.int32)

        for i in tqdm.trange(num_samples):
            if return_T:
                s, t = self.get_perfect_sample(T, verbose=verbose, return_T=return_T)
                samples = samples.at[i].set(s)
                t_vals = t_vals.at[i].set(t)
            else:
                s, = self.get_perfect_sample(T, verbose=verbose, return_T=return_T)
                samples = samples.at[i].set(s)
        
        if return_T:
            return samples, t_vals
        else:
            return samples
