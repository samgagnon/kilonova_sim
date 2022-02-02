import matplotlib.pyplot as plt
import torch
import swyft

from forward import *

# a TMNRE prior function takes a random variable between 0 and 1 and produces an input vector from a prior of your choice

# by convention, nobs parameters preceed obs parameters

# define prior chunks
global_low_list = [50.0]
global_high_list = [100.0]
nobs_low_list = [0.0]*((k+1)*n_nobs)
obs_low_list = [0.0]*(k*n_obs)
nobs_high_list = [1.0]*((k+1)*n_nobs)
obs_high_list = [1.0]*(k*n_obs)

# append prior chunks
low_list = global_low_list + nobs_low_list + obs_low_list
high_list = global_high_list + nobs_high_list + obs_high_list

# instantiate prior
low = np.array(low_list)
high = np.array(high_list)
prior = swyft.get_uniform_prior(low, high)

observation_o = {'x': np.array([1.0])}

n_observation_features = observation_o[observation_key].shape[0]
observation_shapes = {key: value.shape for key, value in observation_o.items()}

simulator = swyft.Simulator(
    forward,
    n_parameters,
    sim_shapes=observation_shapes
)

# set up storage

store = swyft.Store.memory_store(simulator)
store.add(n_training_samples, prior)
store.simulate()

dataset = swyft.Dataset(n_training_samples, prior, store)


def do_round_2d(bound, observation_focus):
    store.add(n_training_samples, prior, bound=bound)
    store.simulate()

    dataset = swyft.Dataset(n_training_samples, prior, store, bound = bound)

    network_2d = swyft.get_marginal_classifier(
        observation_key=observation_key,
        marginal_indices=marginal_indices_2d,
        observation_shapes=observation_shapes,
        n_parameters=n_parameters,
        hidden_features=32,
        num_blocks=2,
    )
    mre_2d = swyft.MarginalRatioEstimator(
        marginal_indices=marginal_indices_2d,
        network=network_2d,
        device=device,
    )
    mre_2d.train(dataset)

    posterior_2d = swyft.MarginalPosterior(mre_2d, prior, bound)
    new_bound = posterior_2d.truncate(n_posterior_samples_for_truncation, observation_focus)

    return posterior_2d, new_bound

bound = None
for i in range(3):
    posterior_2d, bound = do_round_2d(bound, observation_o)

network_1d = swyft.get_marginal_classifier(
    observation_key=observation_key,
    marginal_indices=marginal_indices_1d,
    observation_shapes=observation_shapes,
    n_parameters=n_parameters,
    hidden_features=32,
    num_blocks=2,
)
mre_1d = swyft.MarginalRatioEstimator(
    marginal_indices=marginal_indices_1d,
    network=network_1d,
    device=device,
)
mre_1d.train(dataset)

store.add(n_training_samples + 100, prior, bound=bound)
store.simulate()
dataset = swyft.Dataset(n_training_samples, prior, store, bound = bound)

mre_1d.train(dataset)

# SAVING

prior_filename = "example3.prior.pt"
dataset_filename = "examples3.dataset.pt"
mre_1d_filename = "examples3.mre_1d.pt"
bound_filename = "example3.bound.pt"

prior.save(prior_filename)
dataset.save(dataset_filename)
mre_1d.save(mre_1d_filename)
bound.save(bound_filename)