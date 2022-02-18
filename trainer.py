import matplotlib.pyplot as plt
import torch
import swyft
import time

from forward import *

# load prior
print("loading prior")

prior_loaded = swyft.Prior.load(prior_filename)

# load bound
print("loading bound")

# bound_loaded = swyft.Bound.load(bound_filename)

bound_loaded=None

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

dataset_loaded = swyft.Dataset.load(
    filename=dataset_filename,
    store=store
)