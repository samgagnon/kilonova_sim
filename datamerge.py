import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor
import swyft
import sys
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 15

from forward import *

# define prior chunks
global_low_list = [65.0]
global_high_list = [90.0]
# nobs_low_list = [0.0]*((k+1)*n_nobs)
# obs_low_list = [0.0]*(k*n_obs)
# nobs_high_list = [1.0]*((k+1)*n_nobs)
# obs_high_list = [1.0]*(k*n_obs)
# low_list = [0.0]*10
# high_list = [1.0]*10

# append prior chunks
# low_list = global_low_list + nobs_low_list + obs_low_list
# high_list = global_high_list + nobs_high_list + obs_high_list
low_list = global_low_list# + low_list
high_list = global_high_list# + high_list

# instantiate prior
low = np.array(low_list)
high = np.array(high_list)
prior = swyft.get_uniform_prior(low, high)

observation_o = {'x': np.array([1.0])}

n_observation_features = observation_o[observation_key].shape[0]
observation_shapes = {key: value.shape for key, value in observation_o.items()}

simulator = swyft.Simulator(
    forward,
    1,
    sim_shapes=observation_shapes
)

store = swyft.Store.memory_store(simulator)

subdir = str(sys.argv[1])

store.add(5000*5, prior)
d1 = torch.load(subdir + '/el1.dataset.pt')
d2 = torch.load(subdir + '/el2.dataset.pt')
d3 = torch.load(subdir + '/el3.dataset.pt')
d4 = torch.load(subdir + '/el4.dataset.pt')
d5 = torch.load(subdir + '/el5.dataset.pt')

dtl = torch.load('data archive/' + subdir + '.dataset.pt')
dT = d1 + d2 + d3 + d4 + d5 + dtl

torch.save(dT, 'data archive/' + subdir + '.dataset.pt')