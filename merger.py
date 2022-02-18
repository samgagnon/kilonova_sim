import matplotlib.pyplot as plt
import torch
import swyft
import time
import sys

from forward import *

# a TMNRE prior function takes a random variable between 0 and 1 and produces an input vector from a prior of your choice

# by convention, nobs parameters preceed obs parameters

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

dl1 = swyft.Dataset.load(
    filename='dir1' + '/' + dataset_filename,
    store=store
)

dl2 = swyft.Dataset.load(
    filename='dir2' + '/' + dataset_filename,
    store=store
)

print(dl1.__getitem__(200))
print(dl2.__getitem__(200))

dl3 = dl1 + dl2

print(dl3.__getitem__(600))