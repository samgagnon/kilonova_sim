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
global_high_list = [80.0]
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

subdir1 = str(sys.argv[1])
subdir2 = str(sys.argv[2])

subdirs = [subdir1, subdir2]

if (subdir1 == 'e100o') or (subdir1 == 'o4') or (subdir1 == 'o1') or (subdir1 == 'o4'):
    labels = ['Standard sirens only', 'GW-only incl.']

elif subdir1 == 'm100o':
    labels = ['Malmquist-corrected', 'Original']

elif subdir1 == 'o100o':
    labels = ['Original', 'Malmquist-corrected']

if subdir1 == 'o1':
    max_arr = [-4.7891728728932765, -9.88041224019397]

elif subdir1 == 'o4':
    max_arr = [-14.315450220814723, -48.002909472889634]

def find_min(dataset):
    """
    Finds the minimum value of S in a dataset.
    """
    val = []
    for i in range(len(dataset)):
        val += [dataset[i][0]['x'].numpy()]
    return min(val)

def clean_dataset(dataset, min_val):
    """
    dataset: dataset object
    min_val: the default minimum value of S
    """
    nd = len(dataset)
    new_dataset = []
    for i in range(nd):
        if np.isnan(dataset[i][0]['x'].numpy()):
            dataset[i][0]['x'] = tensor(min_val)
        else:
            new_dataset += [dataset[i]]
    return new_dataset

def truncated_dataset(dataset, min_val):
    """
    dataset: dataset object
    min_val: the default minimum value of S
    """
    nd = len(dataset)
    new_dataset = []
    trash_dataset = []
    for i in range(nd):
        if np.isnan(dataset[i][0]['x'].numpy()):
            dataset[i][0]['x'] = tensor([0.0])
            trash_dataset += [dataset[i]]
        elif (dataset[i][0]['x'].numpy() == min_val[0]):
            dataset[i][0]['x'] = tensor([0.0])
            trash_dataset += [dataset[i]]
        else:
            new_dataset += [dataset[i]]
    new_dataset = normalize_dataset(new_dataset)
    r_dataset = new_dataset + trash_dataset
    return r_dataset

def transform_dataset(dataset):
    """
    Returns a transformed dataset
    """
    d_trans = []
    St = []
    for i in range(len(dataset)):
        St += [10**(dataset[i][0]['x'].numpy()[0])]
    Sm = max(St)
    for i in range(len(dataset)):
        d2 = dataset[i][1].numpy()[0]
        H0 = dataset[i][2].numpy()[0]
        d_trans += [({'x': tensor([(St[i]/Sm).astype(np.float32)])},\
                         tensor([d2.astype(np.float32)]), tensor([H0.astype(np.float32)]))]
    return d_trans

def S_filter(H0, H0min, S, smin, smax):
    """
    Maps summary statistics above a certain H0 (H0min) to 
    match the zero fraction of "GW-only" inclusive samplings.
    """
    if H0>H0min:
        f = (H0-H0min)*(1-0.6)/(H0min-90)+1
    else:
        f = 1
    return ((S-smin)/(smax-smin)*f)*(smax-smin)+smin

def filter_dataset(dataset, smax):
    """
    Returns a filtered dataset
    """
    d_trans = []
    smin = find_min(dataset)
    for i in range(len(dataset)):
        S = dataset[i][0]['x'].numpy()[0]
        d2 = dataset[i][1].numpy()[0]
        H0 = dataset[i][2].numpy()[0]
        Smap = S_filter(H0, 70, S, smin, smax)
        d_trans += [({'x': tensor([(Smap[0]).astype(np.float32)])},\
                         tensor([d2.astype(np.float32)]), tensor([H0.astype(np.float32)]))]
    return d_trans

def normalize_dataset(dataset):
    """
    Returns a normalized dataset
    """
    d_trans = []
    St = []
    for i in range(len(dataset)):
        St += [dataset[i][0]['x'].numpy()[0]]
    Sm = max(St)
    Smin = min(St)
    for i in range(len(dataset)):
        d2 = dataset[i][1].numpy()[0]
        H0 = dataset[i][2].numpy()[0]
        d_trans += [({'x': tensor([((-St[i]+Smin)/(-Sm+Smin)).astype(np.float32)])},\
                         tensor([d2.astype(np.float32)]), tensor([H0.astype(np.float32)]))]
    return d_trans

def plot_dataset(dataset):
    S = [dataset[i][0]['x'].numpy()[0] for i in range(len(dataset))]
    H0 = [dataset[i][2].numpy()[0] for i in range(len(dataset))]
    plt.plot(H0, S, '.')
    plt.show()

datasets = []
datasets2 = []
posteriors = []

for i in [0, 1]:
    dtl = torch.load('data archive/' + subdirs[0] + '.dataset.pt')

    if i == 0:
        min_val = find_min(dtl)
        dtl = clean_dataset(dtl, min_val)
        datasets2 += [dtl]
    else:
        min_val = find_min(dtl)
        # dtl = truncated_dataset(dtl, min_val)
        # d_t = normalize_dataset(dT)
        # min_val = find_min(dtl)
        dtl = clean_dataset(dtl, min_val)
        dtl = filter_dataset(dtl, max_arr[0])
        # datasets += [d_t]
        datasets2 += [dtl]
    # d_tt = transform_dataset(dT)
    # d_t = normalize_dataset(dT)
    # plot_dataset(d_t)
    # plt.show()

    marginal_indices_1d, marginal_indices_2d = swyft.utils.get_corner_marginal_indices(1)


    network_1d = swyft.get_marginal_classifier(
        observation_key='x',
        marginal_indices=marginal_indices_1d,
        observation_shapes=observation_shapes,
        n_parameters=1,
        hidden_features=32,
        num_blocks=2,
    )
    mre_1d = swyft.MarginalRatioEstimator(
        marginal_indices=marginal_indices_1d,
        network=network_1d,
        device=device,
    )

    mre_1d.train(dtl, learning_rate = 5e-5, early_stopping_patience=25, scheduler_kwargs = {"factor": 0.1, "patience": 5})

    n_rejection_samples = 10000

    print("producing posterior")

    posterior_1d = swyft.MarginalPosterior(mre_1d, prior)

    print("sampling")

    samples_1d = posterior_1d.sample(n_rejection_samples, {'x': np.array([max_arr[0]])})
    key = marginal_indices_1d[0]

    posteriors += [samples_1d[key]]


plt.figure(figsize=(10,8))

plt.hist(posteriors[0], 100, alpha=0.5, color='g', density=True, label=labels[i])
plt.hist(posteriors[1], 100, alpha=0.5, color='b', density=True, label=labels[i])
plt.axvline(np.mean(posteriors[0]), linestyle='--', color='g')
plt.axvline(np.mean(posteriors[1]), linestyle='--', color='b')
plt.xlabel("$H_0$ [km/s/Mpc]", fontsize=20)
plt.legend(fontsize=20)
plt.show()

fig, axs = plt.subplots(1,2)
fig.set_figheight(8)
fig.set_figwidth(6)

i = 0
for dataset in datasets2:
    S = [dataset[i][0]['x'].numpy()[0] for i in range(len(dataset))]
    H0 = [dataset[i][2].numpy()[0] for i in range(len(dataset))]
    axs[i].plot(H0, S, '.')
    axs[i].set_title(labels[i], fontsize=25)
    axs[i].axhline(max_arr[0], linestyle='--', color='r')
    i += 1
plt.show()