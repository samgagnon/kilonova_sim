import matplotlib.pyplot as plt
import torch
import swyft

from forward import *


# a TMNRE prior function takes a random variable between 0 and 1 and produces an input vector from a prior of your choice

# nobs
# low = np.array([50.0, 0.0, 0.0])
# high = np.array([100.0, 1.0, 1.0])
# obs
low = np.array([50.0, 0.0])
high = np.array([100.0, 1.0])
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

# def do_round_1d(bound, observation_focus):

#     store.add(n_training_samples, prior, bound=bound)
#     store.simulate()

#     dataset = swyft.Dataset(n_training_samples, prior, store, bound = bound)

#     network_1d = swyft.get_marginal_classifier(
#         observation_key=observation_key,
#         marginal_indices=marginal_indices_1d,
#         observation_shapes=observation_shapes,
#         n_parameters=n_parameters,
#         hidden_features=32,
#         num_blocks=2,
#     )
#     mre_1d = swyft.MarginalRatioEstimator(
#         marginal_indices=marginal_indices_1d,
#         network=network_1d,
#         device=device,
#     )
#     mre_1d.train(dataset)
#     posterior_1d = swyft.MarginalPosterior(mre_1d, prior, bound)
#     new_bound = posterior_1d.truncate(n_posterior_samples_for_truncation, observation_focus)
#     return posterior_1d, new_bound


# bound = None
# for i in range(3):
#     posterior_1d, bound = do_round_1d(bound, {'x': np.array([1.0])})

# # create a simple violin plot

# n_rejection_samples = 5_000

# samples_1d = posterior_1d.sample(n_rejection_samples, observation_o)

# _ = swyft.violin(samples_1d)

# # create row of histograms

# _, _ = swyft.hist1d(samples_1d, kde=True)

# plt.show()

# # NO TRUNCATION FOLLOWS

# train a 1d marginal estimator

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

print("training")

dlen = len(dataset)

for i in range(dlen):
    print(dataset.__getitem__(i))

mre_1d.train(dataset)


# SAVING

prior_filename = "example3.prior.pt"
dataset_filename = "examples3.dataset.pt"
mre_1d_filename = "examples3.mre_1d.pt"

prior.save(prior_filename)
dataset.save(dataset_filename)
mre_1d.save(mre_1d_filename)