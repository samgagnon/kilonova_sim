import matplotlib.pyplot as plt
import torch
import swyft
import time

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

# save prior
prior.save(prior_filename)

observation_o = {'x': np.array([1.0])}

n_observation_features = observation_o[observation_key].shape[0]
observation_shapes = {key: value.shape for key, value in observation_o.items()}

simulator = swyft.Simulator(
    forward,
    n_parameters,
    sim_shapes=observation_shapes
)

# set up storage

tic = time.time()
tlist = []

store = swyft.Store.memory_store(simulator)
store.add(n_training_samples, prior)
store.simulate()

toc = time.time()

elapsed = toc-tic
tlist += [elapsed]
np.savetxt('time.txt', np.array(tlist))

dataset = swyft.Dataset(n_training_samples, prior, store)


def do_round_1d(bound, observation_focus):
    store.add(n_training_samples, prior, bound=bound)
    store.simulate()

    dataset = swyft.Dataset(n_training_samples, prior, store, bound = bound)

    # save dataset
    dataset.save(dataset_filename)

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

    mre_1d.save(mre_1d_filename)

    posterior_1d = swyft.MarginalPosterior(mre_1d, prior, bound)
    new_bound = posterior_1d.truncate(n_posterior_samples_for_truncation, observation_focus)
    
    # save new bound
    new_bound.save(bound_filename)

    return posterior_1d, new_bound

bound = None
for i in range(2):
    tic = time.time()
    posterior_1d, bound = do_round_1d(bound, observation_o)
    toc = time.time()

    elapsed = toc-tic
    tlist += [elapsed]
    np.savetxt('time.txt', np.array(tlist))

    dataset = swyft.Dataset(n_training_samples, prior, store)


tic = time.time()

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

toc = time.time()
elapsed = toc-tic
tlist += [elapsed]
np.savetxt('time.txt', np.array(tlist))

dataset = swyft.Dataset(n_training_samples, prior, store)

# SAVING

prior.save(prior_filename)
dataset.save(dataset_filename)
mre_1d.save(mre_1d_filename)
bound.save(bound_filename)


n_rejection_samples = 100000

print("producing posterior")

posterior_1d = swyft.MarginalPosterior(mre_1d, prior, bound=bound)

print("sampling")

samples_1d = posterior_1d.sample(n_rejection_samples, observation_o)
key = marginal_indices_1d[0]

print(samples_1d[key])

plt.hist(samples_1d[key], 100)
plt.show()

np.savetxt("H0_samples.txt", samples_1d[key])