import astropy.cosmology as cosmo

# how many observations?
o1 = False
o5 = True

# are non-observations included?
n = True

# assume a true value for the Hubble constant
u_true = cosmo.FlatLambdaCDM(70, 0.3)
# set the true redshift of the events
ZT1 = 0.031
ZT2 = 2.49368266e-2
ZT3 = 1.69988646e-02
ZT4 = 4.22995849e-02
ZT5 = 1.54850565e-02
ZT6 = 2.35409134e-02
ZT7 = 6.60280705e-02
ZT8 = 6.11473418e-02
ZT9 = 4.03183165e-02
ZT10 = 3.08064095e-02
ZT11 = 8.85648358e-03
# calculate the true distance of the events
DT1 = u_true.luminosity_distance(ZT1).value
DT2 = u_true.luminosity_distance(ZT2).value
DT3 = u_true.luminosity_distance(ZT3).value
DT4 = u_true.luminosity_distance(ZT4).value
DT5 = u_true.luminosity_distance(ZT5).value
DT6 = u_true.luminosity_distance(ZT6).value
DT7 = u_true.luminosity_distance(ZT7).value
DT8 = u_true.luminosity_distance(ZT8).value
DT9 = u_true.luminosity_distance(ZT9).value
DT10 = u_true.luminosity_distance(ZT10).value
DT11 = u_true.luminosity_distance(ZT11).value

# true masses for each neutron star
m1 = (1.4, 1.4)
m2 = (1.19124606, 1.68521165)
m3 = (1.82530210, 1.36362431)
m4 = (2.05529767e+00, 1.25570351e+00)
m5 = (1.75293951e+00, 1.29532747e+00)
m6 = (1.70298094e+00, 1.61503774e+00)
m7 = (1.39189677e+00, 2.44430152e+00)
m8 = (1.47323485e+00, 1.36881131e+00)
m9 = (1.79676445e+00, 1.41040578e+00)
m10 = (1.29790736e+00, 1.64149555e+00)
m11 = (1.75336110e+00,  1.34787474e+00)

# mass uncertainties
dm1 = (1e-1, 1e-1)
dm2 = (5.95623030e-02, 8.42605827e-02)
dm3 = (9.12651050e-02, 6.81812157e-02)
dm4 = (1.02764883e-01, 6.27851754e-02)
dm5 = (8.76469757e-02, 6.47663735e-02)
dm6 = (8.51490469e-02, 8.07518870e-02)
dm7 = (6.95948385e-02, 1.22215076e-01)
dm8 = (7.36617423e-02, 6.84405653e-02)
dm9 = (8.98382225e-02, 7.05202889e-02)
dm10 = (6.48953678e-02, 8.20747775e-02)
dm11 = (8.76680552e-02,  6.73937371e-02)

# cosine of observation angles
v2, v3, v4, v5, v6, v7, v8, v9, v10, v11 = 0.13039641, 0.63846131, 0.9070467, 0.34523033, 0.8636566, \
       0.61501855, 0.5952604 , 0.41010708, 0.10329073, 0.46383165

# observation bool
o2, o3, o4, o5, o6, o7, o8, o9, o10, o11 = True, True, False, True, True, \
    False, False, False, False, True

# split the event list into two lists, one for obs (true) and one for nobs (false)

if o5:
    if n:
        obs_list = [[ZT10, DT10, m10, dm10, v10, o10],\
            [ZT2, DT2, m2, dm2, v2, o2],\
            [ZT3, DT3, m3, dm3, v3, o3],\
            [ZT5, DT5, m5, dm5, v5, o5],\
            [ZT6, DT6, m6, dm6, v6, o6],\
            [ZT11, DT11, m11, dm11, v11, o11]]

        nobs_list = [[ZT4, DT4, m4, dm4, v4, o4],\
            [ZT7, DT7, m7, dm7, v7, o7],\
            [ZT8, DT8, m8, dm8, v8, o8],\
            [ZT9, DT9, m9, dm9, v9, o9]]

        event_list = [[ZT4, DT4, m4, dm4, v4, o4],\
            [ZT7, DT7, m7, dm7, v7, o7],\
            [ZT8, DT8, m8, dm8, v8, o8],\
            [ZT9, DT9, m9, dm9, v9, o9],\
            [ZT10, DT10, m10, dm10, v10, o10],\
            [ZT2, DT2, m2, dm2, v2, o2],\
            [ZT3, DT3, m3, dm3, v3, o3],\
            [ZT5, DT5, m5, dm5, v5, o5],\
            [ZT6, DT6, m6, dm6, v6, o6],\
            [ZT11, DT11, m11, dm11, v11, o11]]
    else:
        obs_list = [[ZT10, DT10, m10, dm10, v10, o10],\
            [ZT2, DT2, m2, dm2, v2, o2],\
            [ZT3, DT3, m3, dm3, v3, o3],\
            [ZT5, DT5, m5, dm5, v5, o5],\
            [ZT6, DT6, m6, dm6, v6, o6],\
            [ZT11, DT11, m11, dm11, v11, o11]]

        nobs_list =  None

        event_list = [[ZT10, DT10, m10, dm10, v10, o10],\
            [ZT2, DT2, m2, dm2, v2, o2],\
            [ZT3, DT3, m3, dm3, v3, o3],\
            [ZT5, DT5, m5, dm5, v5, o5],\
            [ZT6, DT6, m6, dm6, v6, o6],\
            [ZT11, DT11, m11, dm11, v11, o11]]


if o1:
    if n:
        obs_list = [[ZT1, DT1, m1, dm1, 0.1, True]]
        nobs_list = [[ZT1, DT1, m1, dm1, 0.9, False]]

        event_list = [[ZT1, DT1, m1, dm1, 0.9, False],\
            [ZT1, DT1, m1, dm1, 0.1, True]]
    else:
        obs_list = [[ZT1, DT1, m1, dm1, 0.1, True]]
        nobs_list = None
        event_list = [[ZT1, DT1, m1, dm1, 0.1, True]]


n_events = len(event_list)
if nobs_list is not None:
    n_nobs = len(nobs_list)
else:
    n_nobs = 0
if obs_list is not None:
    n_obs = len(obs_list)
else:
    n_obs = 0

det_obs = [event[-1] for event in event_list]

if __name__ == "__main__":
    print(v7)