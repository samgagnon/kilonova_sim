import swyft
import sys
import torch
from torch import tensor

from forward import *

affix = str(sys.argv[1])

# a TMNRE prior function takes a random variable between 0 and 1 and produces an input vector from a prior of your choice

# by convention, nobs parameters preceed obs parameters

# define prior chunks
global_low_list = [65.0]
global_high_list = [95.0]
low_list = [0.0]*n_events
high_list = [1.0]*n_events

# append prior chunks
low_list = global_low_list + low_list
high_list = global_high_list + high_list

# instantiate prior
low = np.array(low_list)
high = np.array(high_list)
prior = swyft.get_uniform_prior(low, high)

# save prior
prior.save(prior_filename)

observation_o = {'x': np.array(good_reference)}

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

print(len(dataset))
print(list(dataset))

dlist = list(dataset)
torch.save(dlist, 'el' + affix + '.dataset.pt')

dataset.save('e' + affix + '.dataset.pt')

dl = swyft.Dataset.load(
    filename='e' + affix + '.dataset.pt',
    store=store
)