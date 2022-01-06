import matplotlib.pyplot as plt

from forward import *


# prior_filename = "1obs.prior.pt"
# dataset_filename = "1obs.dataset.pt"
# mre_1d_filename = "1obs.mre_1d.pt"

prior_filename = "example3.prior.pt"
dataset_filename = "examples3.dataset.pt"
mre_1d_filename = "examples3.mre_1d.pt"
bound_filename = "example3.bound.pt"

# prior_filename = "1obs.prior.pt"
# dataset_filename = "1obs.dataset.pt"
# mre_2d_filename = "1obs.mre_2d.pt"

# low = np.array([50.0, 0.0])
# high = np.array([100.0, 1.0])

# 1obs 1nobs
low = np.array([50.0, 0.0, 0.0, 0.0])
high = np.array([100.0, 1.0, 1.0, 1.0])

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

bound_loaded = swyft.Bound.load(bound_filename)


print("loading prior")

prior_loaded = swyft.Prior.load(prior_filename)
dataset_loaded = swyft.Dataset.load(
    filename=dataset_filename,
    store=store
)


network_new = swyft.get_marginal_classifier(
    observation_key=observation_key,
    marginal_indices=marginal_indices_1d,
    observation_shapes=observation_shapes,
    n_parameters=n_parameters,
    hidden_features=32,
    num_blocks=2,
)

print("loading network")

mre_1d_loaded = swyft.MarginalRatioEstimator.load(
    network=network_new,
    device=device,
    filename=mre_1d_filename,
)

# create a simple violin plot

n_rejection_samples = 10000

print("producing posterior")

posterior_1d = swyft.MarginalPosterior(mre_1d_loaded, prior_loaded, bound=bound_loaded)

print("sampling")

samples_1d = posterior_1d.sample(n_rejection_samples, observation_o)
key = list(samples_1d.keys())[0]

_ = swyft.violin(samples_1d)
plt.show()

# create row of histograms

_, _ = swyft.hist1d(samples_1d, kde=True, ylim=(0,0.22))
plt.show()

print(samples_1d[key])

np.savetxt("1obs1nobs.H0_samples.txt", samples_1d[key])

# print(key)
# print(samples_1d[key])

# dv_arr = np.array(samples_1d[key])[:,1]

# pdist = pDV_dist(d_true, 0.9, m1_true, m2_true)
# vdist = np.sum(pdist[0], 0)
# vdist /= vdist.sum()
# vcdf = np.cumsum(vdist)
# vloc_arr = [np.abs(vcdf - dv).argmin() for dv in dv_arr]
# v_arr = [pdist[3][vloc] for vloc in vloc_arr]

# plt.hist(np.array(samples_1d[key])[:,0], 1000)
# plt.xlim([50, 100])
# plt.show()

# plt.hist(v_arr, 1000)
# plt.xlim([0.0,1.0])
# plt.show()