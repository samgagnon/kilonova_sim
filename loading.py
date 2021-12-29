import matplotlib.pyplot as plt

from forward import *

prior_filename = "example3.prior.pt"
dataset_filename = "examples3.dataset.pt"
mre_1d_filename = "examples3.mre_1d.pt"

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

# LOADING

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

mre_1d_loaded = swyft.MarginalRatioEstimator.load(
    network=network_new,
    device=device,
    filename=mre_1d_filename,
)

# create a simple violin plot

n_rejection_samples = 5000

print("producing posterior")

posterior_1d = swyft.MarginalPosterior(mre_1d_loaded, prior_loaded)

print("sampling")

samples_1d = posterior_1d.sample(n_rejection_samples, observation_o)

# _ = swyft.violin(samples_1d)

# create row of histograms

# _, _ = swyft.hist1d(samples_1d, kde=True)

key = list(samples_1d.keys())[0]
print(key)

plt.hist(samples_1d[key], 100)
plt.xlim(50, 100)

plt.show()