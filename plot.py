import numpy as np
import matplotlib.pyplot as plt
import torch
import swyft
import sys
from torch import tensor
from scipy.stats import linregress
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

if (subdir1 == 'eo10o'):
    labels = ['Standard sirens only', 'GW-only incl.']
    max_arr = [-24.1, -48.002909472889634]


if (subdir1 == 'eo100o'):
    labels = ['Standard sirens only', 'GW-only incl.']
    max_arr = [-24.1, -48.725]

if (subdir1 == 'o4') or (subdir1 == 'o1') or (subdir1 == 'o4'):
    labels = ['Standard sirens only', 'GW-only incl.']

elif subdir1 == 'm100o':
    labels = ['Original', 'Malmquist-corrected']
    max_arr = [-470.7763166268362, -470.7763166268362]

elif (subdir1 == 'e25100m'):
    labels = ['Original', 'Malmquist-corrected']
    max_arr = [-352.0452905522127, -352.0452905522127]

elif (subdir1 == 'e25100o'):
    labels = ['Original', 'GW-only incl.']
    max_arr = [-352.0452905522127]

elif (subdir1 == 'es25100m'):
    labels = ['Original', 'Malmquist-corrected']
    max_arr = [-84.77575904933398, -84.77575904933398]

elif (subdir1 == 'es25100o'):
    labels = ['Original', 'GW-only incl.']
    max_arr = [-84.77575904933398]

elif (subdir1 == 'esg100m'):
    labels = ['Original', 'Malmquist-corrected']
    max_arr = [-104.69960725284007, -104.69960725284007]

elif (subdir1 == 'esg100o'):
    labels = ['Original', 'GW-only incl.']
    max_arr = [-104.69960725284007]

elif (subdir1 == 'e100m') or (subdir1 == 'e100o'):
    labels = ['Original', 'Malmquist-corrected']
    max_arr = [-116.60953916980247, -116.60953916980247]

elif subdir1 == 'o100o':
    labels = ['Original', 'Malmquist-corrected']
    max_arr = [-483.3006904738374, ]

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
    
def find_max_H0(dataset):
    """
    Finds the H0 associated with the maximum S in a dataset.
    """
    val = []
    H0 = []
    for i in range(len(dataset)):
        val += [dataset[i][0]['x'].numpy()]
        H0 += [dataset[i][2].numpy()[0]]
    return H0[np.argmax(np.asarray(val))]

def find_end_H0(dataset, H0_max, H0_min):
    """
    Finds the last informative H0 in a dataset.
    """
    H_list = []
    for i in range(len(dataset)):
        if dataset[i][2].numpy()[0] > H0_max:
            if dataset[i][0]['x'].numpy()[0] == H0_min:
                H_list += [dataset[i][2].numpy()[0]]
    if len(H_list) != 0:
        return min(H_list)
    else:
        return dataset[-1][2].numpy()[0]

def find_start_H0(dataset, H0_max, H0_min):
    """
    Finds the last informative H0 in a dataset.
    """
    H_list = []
    for i in range(len(dataset)):
        if dataset[i][0]['x'].numpy()[0] < H0_max:
            if dataset[i][0]['x'].numpy()[0] == H0_min:
                H_list += [dataset[i-1][2].numpy()[0]]
    return max(H_list)

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

def clean_m_dataset(dataset, H0_min, H0_start, H0_end, printy=False):
    """
    dataset: dataset object
    min_val: the default minimum value of S
    """
    nd = len(dataset)
    new_dataset = []
    for i in range(nd):
        if dataset[i][2].numpy()[0] < H0_start:
            if printy is True:
                print(dataset[i][0]['x'].numpy()[0], H0_min)
            if dataset[i][0]['x'].numpy()[0] == H0_min:
                if printy is True:
                    print('yeah')
                new_dataset += [dataset[i]]
        elif dataset[i][2].numpy() > H0_end:
            if dataset[i][0]['x'].numpy()[0] == H0_min:
                new_dataset += [dataset[i]]
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

def S_filter(H0, H0min, S, m, smin, smax, truemin):
    """
    Maps summary statistics above a certain H0 (H0min) to 
    match the zero fraction of "GW-only" inclusive samplings.
    """
    if H0>H0min:
        f = (H0-H0min)*m+1
    else:
        f = 1
    if S == truemin[0]:
        return truemin
    else:
        return ((S-smin)/(smax-smin)*f)*(smax-smin)+smin

def S_filter2(H0, H0min, S, m, smin, smax, truemin):
    """
    Maps summary statistics above a certain H0 (H0min) to 
    match the zero fraction of "GW-only" inclusive samplings.
    """
    if H0>H0min:
        f = (H0-H0min)*m+1
        if np.random.uniform() > f:
            Sm = truemin[0]
        else:
            Sm = S
    else:
        Sm = S
    return [Sm]

def filter_dataset(dataset, m, H0_max, smax, smin):
    """
    Returns a filtered dataset
    """
    d_trans = []
    truemin = find_min(dataset)
    for i in range(len(dataset)):
        S = dataset[i][0]['x'].numpy()[0]
        d2 = dataset[i][1].numpy()[0]
        H0 = dataset[i][2].numpy()[0]
        Smap = S_filter(H0, H0_max, S, m, smin, smax, truemin)
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

def zero_percentage(dataset, n):
    mdtl = find_min(dataset)[0]
    H0_zeros = []
    H0 = []
    for dat in dataset:
        if dat[0]['x'].numpy() == mdtl:
            H0_zeros += [dat[2].numpy()[0]]
        if not np.isnan(dat[0]['x'].numpy()):
            H0 += [dat[2].numpy()[0]]
    bins = plt.hist(H0, n)
    # plt.show()
    bins0 = plt.hist(H0_zeros, n)
    H0 = (bins[1][:-1] + bins[1][1:])/2
    # plt.show()
    # plt.clf()
    # plt.plot(H0, 1 - bins0[0]/bins[0])
    # plt.show()
    return H0, 1 - bins0[0]/bins[0]

def plot_dataset(dataset):
    S = [dataset[i][0]['x'].numpy()[0] for i in range(len(dataset))]
    H0 = [dataset[i][2].numpy()[0] for i in range(len(dataset))]
    # plt.plot(H0, S, '.')
    # plt.show()

datasets = []
datasets2 = []
posteriors = []

for i in [0, 1]:
    dtl = torch.load('data archive/' + subdirs[i] + '.dataset.pt')

    # if i == 0:
    #     min_val = find_min(dtl)
    #     dtl = clean_dataset(dtl, min_val)
    #     H0_max = find_max_H0(dtl)
    #     H0_end = find_end_H0(dtl, H0_max, min_val)
    #     # print("end", H0_end)
    #     datasets2 += [dtl]
    # else:
    #     min_val = find_min(dtl)
    #     dtl2 = torch.load('data archive/' + subdirs[i] + '.dataset.pt')
    #     # dtl = clean_dataset(dtl, min_val)
    #     # dtl2 = clean_dataset(dtl2, min_val[0])
    #     # dtl = clean_m_dataset(dtl, min_val, 65, 90)
    #     # dtl2 = clean_m_dataset(dtl2, min_val[0], 65, 90)
    #     S = [dtl[i][0]['x'].numpy()[0] for i in range(len(dtl))]
    #     H0 = [dtl[i][2].numpy()[0] for i in range(len(dtl))]
    #     plt.plot(H0, S, '.')
    #     plt.show()
    #     S = [dtl2[i][0]['x'].numpy()[0] for i in range(len(dtl2))]
    #     H0 = [dtl2[i][2].numpy()[0] for i in range(len(dtl2))]
    #     plt.plot(H0, S, '.')
    #     plt.show()
    #     H01, percent1 = zero_percentage(dtl, 100)
    #     H02, percent2 = zero_percentage(dtl2, 100)
    #     # plt.plot()
    #     # plt.show()
    #     # plt.plot(H01, percent1)
    #     # plt.show()
    #     # plt.plot(H02, percent2)
    #     # plt.show()
    #     lr1 = linregress(H02[(H02>H0_max) + (percent2>0.5)], percent2[(H02>H0_max) + (percent2>0.5)])
    #     # print(lr1)
    #     percent = percent2*percent1/(lr1[1] + lr1[0]*H0_max)
    #     print(H01[:2], H02[:2])
    #     lr2 = linregress(H02[(H02>H0_max) + (percent>0.05)], percent[(H02>H0_max) + (percent>0.05)])
    #     print(lr2)
    #     print("newmin", (H0_end*lr2[0]+lr2[1])*(max_arr[0]-min_val)+min_val)
    #     newmin = (H0_end*lr2[0]+lr2[1])*(max_arr[0]-min_val)+min_val
    #     dtl = filter_dataset(dtl, lr2[0], H0_max, max_arr[0], newmin)
    #     datasets2 += [dtl]

    min_val = find_min(dtl)
    dtl = clean_dataset(dtl, min_val)
    if i == 0:
        H0_max = find_max_H0(dtl)
        H0_start = find_start_H0(dtl, H0_max, min_val[0])
        H0_end = find_end_H0(dtl, H0_max, min_val[0])
        print("H0 start", H0_start)
        print("H0 end", H0_end)
    if i == 1:
        print(len(dtl))
        dtl = clean_m_dataset(dtl, min_val[0], 60, 90)
        print(len(dtl))
    datasets2 += [dtl]
    
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

    # lpar = [(5, 20, 5e-4),(3, 20, 5e-4)] # optimal for m100o
    # lpar = [(10, 25, 5e-5),(5, 30, 5e-4)] # optimal for em100o
    lpar = [(10, 1, 5e-5),(10, 1, 5e-5)]
    print(lpar)
    mre_1d.train(dtl, learning_rate = lpar[i][2], validation_percentage=0.5, \
        early_stopping_patience=lpar[i][1], scheduler_kwargs = {"factor": 0.1, "patience": lpar[i][0]})

    n_rejection_samples = 1000000

    print("producing posterior")

    posterior_1d = swyft.MarginalPosterior(mre_1d, prior)

    print("sampling")

    samples_1d = posterior_1d.sample(n_rejection_samples, {'x': np.array([max_arr[0]])})
    key = marginal_indices_1d[0]

    posteriors += [samples_1d[key]]


# np.savetxt(subdir1 + '_samples.txt', posteriors[0])
# np.savetxt(subdir2 + '_samples.txt', posteriors[1])

plt.figure(figsize=(10,8))

plt.hist(posteriors[0], 100, alpha=0.5, color='g', density=True, label=labels[0])
plt.axvline(np.mean(posteriors[0]), linestyle='--', color='g')
plt.xlabel("$H_0$ [km/s/Mpc]", fontsize=20)
plt.ylabel("$P(H_0)$", fontsize=20)
plt.legend(fontsize=20)
plt.show()

plt.figure(figsize=(10,8))

plt.hist(posteriors[1], 100, alpha=0.5, color='b', density=True, label=labels[1])
plt.axvline(np.mean(posteriors[1]), linestyle='--', color='b')
plt.xlabel("$H_0$ [km/s/Mpc]", fontsize=20)
plt.ylabel("$P(H_0)$", fontsize=20)
plt.legend(fontsize=20)
plt.show()

plt.figure(figsize=(10,8))

plt.hist(posteriors[0], 100, alpha=0.5, color='g', density=True, label=labels[0])
plt.axvline(np.mean(posteriors[0]), linestyle='--', color='g')
plt.hist(posteriors[1], 100, alpha=0.5, color='b', density=True, label=labels[1])
plt.axvline(np.mean(posteriors[1]), linestyle='--', color='b')
plt.xlabel("$H_0$ [km/s/Mpc]", fontsize=20)
plt.ylabel("$P(H_0)$", fontsize=20)
plt.legend(fontsize=20)
plt.show()

print(np.mean(posteriors[0]))
print(np.mean(posteriors[1]))
print("Correction: ", np.mean(posteriors[0]) - np.mean(posteriors[1]))

fig, axs = plt.subplots(1,2)
fig.set_figheight(8)
fig.set_figwidth(6)

i = 0
for dataset in datasets2:
    S = [dataset[i][0]['x'].numpy()[0] for i in range(len(dataset))]
    H0 = [dataset[i][2].numpy()[0] for i in range(len(dataset))]
    axs[i].plot(H0, S, '.')
    axs[i].set_title(labels[i], fontsize=25)
    # axs[i].axhline(max_arr[0], linestyle='--', color='r')
    axs[i].set_xlabel("$H_0$ [km/s/Mpc]", fontsize=20)
    axs[i].set_ylabel("$\mathcal{S}$", fontsize=20)
    i += 1
plt.show()