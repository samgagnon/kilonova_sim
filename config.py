import swyft

from events import *

# set simulator parameters
device = 'cpu'  # TODO: install CUDA toolkit for Win10
k = 1 # number of free observed event parameters
malm = False # do we want to consider malmquist bias?
# theshold for rejecting samples pre-light curve generation
threshold = 1e-3
# SNR threshold
rho_cri = 8
# R0, tau, m_mean, m_sclae, m_low, m_high, chi_sigma
# R0: merger rate [/yr/Gpc3] 
# tau: Delay time from formation to merger [Gyr]
# mass mean, standard deviation, low, high [solar masses]
# dispersion of effective spin
# be sure to keep these consistent with MOSFIT!
# find sauce to motivate the selected values
BNS_par = [300,3,1.4,0.5,1.1,2.5,0.1]

if malm:
    bad_result = [-3*n_events]
else:
    bad_result = [-3]
good_reference = [0.0]

n_training_samples = int(5000)
observation_key = "x"

n_weighted_samples = 10_000

n_posterior_samples_for_truncation = 10_000

# n_parameters = 1 + (k * n_obs) + (k + 1) * n_nobs # number of free parameters (event parameters plus H0)
n_parameters = 1 + n_events

marginal_indices_1d, marginal_indices_2d = swyft.utils.get_corner_marginal_indices(n_parameters)

prior_filename = "e.prior.pt"
dataset_filename = "e.dataset.pt"
mre_1d_filename = "e.mre_1d.pt"
bound_filename = "e.bound.pt"