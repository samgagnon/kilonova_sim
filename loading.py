import matplotlib.pyplot as plt

from forward import *

# prior_filename = "example3.prior.pt"
dataset_filename = "d1.dat"
# mre_1d_filename = "examples3.mre_1d.pt"
# bound_filename = "example3.bound.pt"

# define prior chunks
global_low_list = [65.0]
global_high_list = [75.0]
# nobs_low_list = [0.0]*((k+1)*n_nobs)
# obs_low_list = [0.0]*(k*n_obs)
# nobs_high_list = [1.0]*((k+1)*n_nobs)
# obs_high_list = [1.0]*(k*n_obs)
low_list = [0.0]*n_events
high_list = [1.0]*n_events

# append prior chunks
# low_list = global_low_list + nobs_low_list + obs_low_list
# high_list = global_high_list + nobs_high_list + obs_high_list
low_list = global_low_list + low_list
high_list = global_high_list + high_list

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

# LOADING

print("loading bound")

# bound_loaded = swyft.Bound.load(bound_filename)

bound_loaded=None

print("loading prior")

# prior_loaded = swyft.Prior.load(prior_filename)
dataset_loaded = swyft.Dataset.load(
    filename=dataset_filename,
    store=store
)


print(list(dataset_loaded))
print(dataset_loaded.__len__())
# print(dataset_loaded.__getitem__(0))

network_new = swyft.get_marginal_classifier(
    observation_key=observation_key,
    marginal_indices=marginal_indices_1d,
    observation_shapes=observation_shapes,
    n_parameters=n_parameters,
    hidden_features=32,
    num_blocks=2,
)

print("loading network")

# mre_1d_loaded = swyft.MarginalRatioEstimator.load(
#     network=network_new,
#     device=device,
#     filename=mre_1d_filename,
# )

mre_1d_loaded = swyft.MarginalRatioEstimator(
    marginal_indices=marginal_indices_1d,
    network=network_new,
    device=device,
)

mre_1d_loaded.train(dataset_loaded)

# create a simple violin plot

n_rejection_samples = 10

print("producing posterior")

posterior_1d = swyft.MarginalPosterior(mre_1d_loaded, prior, bound=bound_loaded)

print("sampling")

samples_1d = posterior_1d.sample(n_rejection_samples, observation_o)
key = marginal_indices_1d[0]

print(samples_1d[key])

plt.hist(samples_1d[key])
plt.show()

np.savetxt("H0_samples.txt", samples_1d[key])