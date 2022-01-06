import astropy.cosmology as cosmo

# assume a true value for the Hubble constant
u_true = cosmo.FlatLambdaCDM(70, 0.3)
# set the true redshift of the events
z_true = 0.031
# calculate the true distance of the events
d_true = u_true.luminosity_distance(z_true).value

# true masses for each neutron star
m1_true = 1.4
m2_true = 1.4

# z, dl, m1, m2, v
# event_list = [[z_true, d_true, m1_true, m2_true, 0.1, False],
# [z_true, d_true, m1_true, m2_true, 0.9, True]]
# let's leave two events for later due to the complications 
# of observed and unobserved events being put together
event_list = [[z_true, d_true, m1_true, m2_true, 0.9, True],\
    [z_true, d_true, m1_true, m2_true, 0.1, False]]
# event_list = [[z_true, d_true, m1_true, m2_true, 0.1, False]]

# split the event list into two lists, one for obs (true) and one for nobs (false)
# obs_list = None
nobs_list = [[z_true, d_true, m1_true, m2_true, 0.1, False]]

obs_list = [[z_true, d_true, m1_true, m2_true, 0.9, True]]
# nobs_list = None
det_obs = [event[-1] for event in event_list]