from events import *

# set simulator parameters
DEVICE = 'cpu'  # replace with 'cpu' if no GPU is available
Ntrain = 100
k = 1 # number of free observed event parameters

if obs_list is not None:
    Nobs = len(obs_list) # number of observed events
else:
    Nobs = 0
if nobs_list is not None:
    Nnobs = len(nobs_list) # number of unobserved events
else:
    Nnobs = 0

Npar = 1 + (k * Nobs) + (k + 1) * Nnobs # number of free parameters (event parameters plus H0)