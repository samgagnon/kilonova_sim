import swyft

import matplotlib.pyplot as plt

from forward import *


# a TMNRE prior function take a random variable between 0 and 1 and produces an input vector from a prior of your choice

def pfunc(u):
    H0 = 50 * u[0] + 50
    theta = np.pi / 2 * u[1]
    du = u[2]
#     return np.array([H0, theta, u[2]]) # nobs
    return np.array([H0, theta, du]) # obs


simulator = swyft.Simulator(forward, ["H_0", "\theta", "du"], sim_shapes = {"x": [Npar]})
store = swyft.MemoryStore(simulator)

prior = swyft.Prior(pfunc, Npar)
store.add(Ntrain, prior)
store.simulate()

print("are we adding?")
store.add(Ntrain, prior)
store.simulate()

dataset = swyft.Dataset(Ntrain, prior, store)
post = swyft.Posteriors(dataset)


post.add([(0, 1, 2)], device=DEVICE)
post.train([(0, 1, 2)], max_epochs = 20, nworkers=0)


v0 = np.array([70, np.arccos(0.1), 0.5]) # nobs
obs0 = forward(v0)

samples = post.sample(100000, obs0)
swyft.plot_corner(samples, [0, 1, 2], color='k', figsize = (8,8), truth=v0, bins = 40)

plt.show()