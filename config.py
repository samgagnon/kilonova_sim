import swyft

from events import *

# set simulator parameters
device = 'cpu'  # TODO: install CUDA toolkit for Win10
k = 1 # number of free observed event parameters

n_training_samples = 100
observation_key = "x"

n_weighted_samples = 10_000

n_posterior_samples_for_truncation = 10_000


if obs_list is not None:
    n_obs = len(obs_list) # number of observed events
else:
    n_obs = 0
if nobs_list is not None:
    n_nobs = len(nobs_list) # number of unobserved events
else:
    n_nobs = 0

n_parameters = 1 + (k * n_obs) + (k + 1) * n_nobs # number of free parameters (event parameters plus H0)

# marginal_indices_1d, marginal_indices_2d = swyft.utils.get_corner_marginal_indices(n_parameters)

marginal_indices_1d = (0,)