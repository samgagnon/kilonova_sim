import swyft

from events import *

# set simulator parameters
device = 'cpu'  # TODO: install CUDA toolkit for Win10
k = 1 # number of free observed event parameters
# theshold for rejecting samples pre-light curve generation
threshold = 1e-7

n_training_samples = int(1000)
observation_key = "x"

n_weighted_samples = 10_000

n_posterior_samples_for_truncation = 10_000

n_parameters = 1 + (k * n_obs) + (k + 1) * n_nobs # number of free parameters (event parameters plus H0)

marginal_indices_1d, marginal_indices_2d = swyft.utils.get_corner_marginal_indices(n_parameters)
# marginal_indices_1d, marginal_indices_2d = swyft.utils.get_corner_marginal_indices(2)

# H0_index_1d = (0,)
