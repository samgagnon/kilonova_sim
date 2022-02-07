import mosfit

import astropy.units as u

from scipy.stats import truncnorm
from astropy.constants import c
from astropy.cosmology import z_at_value

from events import *
from config import *
from distance import *

# instantiate pmesh dictionary
pdict = gen_pDV_dists(event_list, plot=False)

# instantiate fitter
my_fitter = mosfit.fitter.Fitter(quiet=False, test=True, offline=False)


def light_curve(fitfunc, fixed_params):
    """
    Given a fit function and a list of fixed parameters, generates a light curve using mosfit 
    and returns whether a detection is made.
    args:
    - fixed_params: dictionary of fixed parameters
    returns:
    - detection: bool
    
    To do: this should also take the observing time window in units of days after the explosion,
    the magnitude cutoff of each observing band, and probably something else.
    """
    # convert dictionary to list
    # consider turning this into a utility function
    fpar = ["ebv", fixed_params['ebv'], "rvhost", fixed_params['rvhost'],\
            "frad", fixed_params['frad'], "nnhost", fixed_params['nnhost'],\
            "texplosion", fixed_params['texplosion'],\
            "temperature", fixed_params['temperature'],\
            "kappa_red", fixed_params['kappa_red'],\
            "kappa_blue", fixed_params['kappa_blue'],\
            "kappagamma", fixed_params['kappagamma'],\
            "Mchirp", fixed_params['Mchirp'],\
            "q", fixed_params['q'], "cos_theta", fixed_params['cos_theta'],\
            "cos_theta_open", fixed_params['cos_theta_open'],\
            "disk_frac", fixed_params['disk_frac'],\
            "radius_ns", fixed_params['radius_ns'],\
            "alpha", fixed_params['alpha'], "Mtov", fixed_params['Mtov'],\
            "cos_theta_cocoon", fixed_params['cos_theta_cocoon'],\
            "tshock", fixed_params['tshock'], "temperature_shock",\
            fixed_params['temperature_shock'],\
            "lumdist", fixed_params['lumdist'], "redshift", fixed_params['redshift']]
    # create kwargs dict using fixed_params
    kwargs = dict(events=[], models=['bns_generative'],\
              max_time=4, band_list="z", band_systems="AB", iterations=0, num_walkers=1,\
              smooth_times=4, suffix="jupyter_test", user_fixed_parameters=fpar,\
             quiet=True)
    # run mosfit
    entries, ps, lnprobs = my_fitter.fit_events(**kwargs)
    # get observed magnitudes
    # see if there is any way to do this without a for loop
    obs_mags = []
    for entry in entries[0][0]['photometry']:
        if entry['system'] == 'AB':
            obs_mags += [float(entry['magnitude'])]
#     print(obs_mags)

    # probabilistic magnitude shouuld be a utility function

    # magnitude probability of detection
    sig10 = 23.3
#     mclip_a = 0
#     mclip_b = 1
#     m_std = abs((jansky(sig10)) / 10)
#     a, b = mclip_a / m_std, mclip_b / m_std
#     v = truncnorm(a,b)
    # see if the lowest magnitude beats our probabilistic detection threshold
#     flux_obs = jansky(np.array(obs_mags))
#     print(flux_obs)
    # test plot
#     x = np.linspace(20, 30)
#     plt.plot(x, v.cdf(jansky(x)/m_std))
#     plt.show()
    
#     obs_probs = v.cdf(flux_obs/m_std)
#     mprob = np.random.uniform(0,1)
#     detect = mprob < obs_probs
#     pobs = max(obs_probs)
#     print(obs_mags)
#     print(obs_probs)
#     print(detect)
    detect = min(obs_mags) <= sig10
#     print(True in detect)
#     return (True in detect), pobs
    return detect, 1


def nobs_forward(H0, ve, DL, pDL, event):
    # get free event parameters
    v0 = v_from_CDF(event[1], event[4], event[2][0], event[2][1], ve[0])
    # du = ve[1] # distance variable
    if k>2:
        M1, M2 = m_from_dm((ve[2], ve[3]), event)
    else:
        M1 = event[2][0]
        M2 = event[2][1]
    # use results from GWToolbox to get the true DL, M1, M2, v
    TDL = event[1]
    # derived mass quantities
    if M1 < M2:
        Q = M1/M2
    else:
        Q = M2/M1
    Mchirp = ((M1*M2)**3/(M1+M2))**(1/5)
    # create a cosmology from H0 and generate the true redshift
    # in this sample cosmology
    # assume omega_m=0.31 (make sure this is consistent with GWTB)
    universe = cosmo.FlatLambdaCDM(H0, 0.3)
    # "z in cosmology"
    ZIC = z_at_value(universe.luminosity_distance, DL*u.Mpc)
    fixed_params = {"ebv": 2.2, "rvhost": 3.1, "frad": 0.999, "nnhost": 1e18,\
              "texplosion": -0.01, "temperature": 2500, "kappa_red": 10,\
              "kappa_blue": 0.5, "kappagamma": 10000.0, "Mchirp": Mchirp,\
              "q": Q, "cos_theta": v0, "cos_theta_open": 0.707107,\
              "disk_frac": 0.15, "radius_ns": 11.0, "alpha": 1.0,\
              "Mtov": 2.2, "cos_theta_cocoon": 0.5, "tshock": 1.7,\
              "temperature_shock": 100, "lumdist": DL, "redshift": ZIC}
    det, pobs = light_curve(my_fitter, fixed_params)
#     return det, 1-pobs
    return det, pDL # if probabilistic magnitude is off


def obs_forward(H0, ve, event):
    # get free event parameters
    v0 = v_from_CDF(event[1], event[4], event[2][0], event[2][1], ve[0])
    if k>2:
        M1, M2 = m_from_dm((ve[1], ve[2]), event)
    else:
        M1 = event[2][0]
        M2 = event[2][1]
    # use results from GWToolbox to get the true DL, M1, M2
    TZ = event[0]
    TDL = event[1]
    # derived mass quantities
    if M1 < M2:
        Q = M1/M2
    else:
        Q = M2/M1
    Mchirp = ((M1*M2)**3/(M1+M2))**(1/5)
    # create a cosmology from H0 and generate the true redshift
    # in this sample cosmology
    # assume omega_m=0.31 (make sure this is consistent with GWTB)
    universe = cosmo.FlatLambdaCDM(H0, 0.3)
    # "distance in cosmology"
    DIC = universe.luminosity_distance(TZ).value
    # probability density of given cosmology-assigned distance at sampled angle v0
    dprob, dpm, dps = p_DV(event[1], event[4], event[2][0], event[2][1], v0, DIC)
    dprob /= dpm
    # print("dprob:", dprob)
#     dprob = 1.0
    # I also need to calculate the probability of this distance being allowed
    # and return that probability to be used in the summary statistic
    fixed_params = {"ebv": 2.2, "rvhost": 3.1, "frad": 0.999, "nnhost": 1e18,\
              "texplosion": -0.01, "temperature": 2500, "kappa_red": 10,\
              "kappa_blue": 0.5, "kappagamma": 10000.0, "Mchirp": Mchirp,\
              "q": Q, "cos_theta": v0, "cos_theta_open": 0.707107,\
              "disk_frac": 0.15, "radius_ns": 11.0, "alpha": 1.0,\
              "Mtov": 2.2, "cos_theta_cocoon": 0.5, "tshock": 1.7,\
              "temperature_shock": 100, "lumdist": DIC, "redshift": TZ}
    det, pobs = light_curve(my_fitter, fixed_params)
    return det, dprob #*dobs


def forward(v):
    # get terms from input vector
    H0 = v[0]
    print("Input Vector:", v)
    det_list = []
    # first, loop through observations to see if observed distances make H0 illegal
    if obs_list is not None:
        i = 0
        ve_list = list(chunks(v[n_nobs*(k+1)+1:], k))
        for event in obs_list:
            ve = ve_list[i]
            i += 1
            universe = cosmo.FlatLambdaCDM(H0, 0.3)
            # "distance in cosmology"
            DIC = universe.luminosity_distance(event[0]).value
            # probability density of given cosmology-assigned distance at sampled angle v0
            v0 = v_from_CDF(event[1], event[4], event[2][0], event[2][1], ve[0])
            dprob, dpm, dps = p_DV(event[1], event[4], event[2][0], event[2][1], v0, DIC)
            if dprob/dps < threshold:
                return dict(x=np.array([0.0]))
    nobs_dl = []
    if nobs_list is not None:
        i = 0
        ve_list = list(chunks(v[1:n_nobs*(k+1)+1], k+1))
        for event in nobs_list:
            ve = ve_list[i]
            i += 1
            # generate a possible distance from the random variables du, v0
            # and the prior distribution (depending on true event parameters)
            pdist = pdict[str(event)]
            v0 = v_from_CDF(event[1], event[4], event[2][0], event[2][1], ve[0])
            DL, pDL, dpm, dps = D_vdu(event[1], event[4], event[2][0], event[2][1], v0, ve[1], pdist)

            # TODO: produce similar list for v in both this and obs_list

            nobs_dl += [[DL, pDL/dpm]]
            print(pDL/dps)
            if pDL/dps < threshold:
                return dict(x=np.array([0.0]))
    j = 0
    p = 1
    if obs_list is not None:
        i = 0
        ve_list = list(chunks(v[n_nobs*(k+1)+1:], k))
        for event in obs_list:
            ve = ve_list[i]
            i += 1
            det, pobs = obs_forward(H0, ve, event)
            if det is False:
                print(det, 'bad')
                return dict(x=np.array([0.0]))
            det_list += [det]
            p += pobs
    if nobs_list is not None:
        i = 0
        ve_list = list(chunks(v[1:n_nobs*(k+1)+1], k+1))
        for event in nobs_list:
            ve = ve_list[i]
            i += 1
#             print(nobs_dl[j])
            DL, pDL = nobs_dl[j]
            j += 1
            det, pobs = nobs_forward(H0, ve, DL, pDL, event)
            if det is True:
                print(det, 'bad')
                return dict(x=np.array([0.0]))
            det_list += [det]
            # possibly remove pobs for nobs?
            p += pobs
    print((det_list == det_obs))
    # if det_list == det_obs:
        # print(p)
        # x = [p]
    # else:
        # x = [0.0]
    x = [p]
    return dict(x=np.array(x))


if __name__ == "__main__":
    M1 = m11[0]
    M2 = m11[1]
    Q = M1/M2
    Mchirp = ((M1*M2)**3/(M1+M2))**(1/5)
    fixed_params = {"ebv": 2.2, "rvhost": 3.1, "frad": 0.999, "nnhost": 1e18,\
              "texplosion": -0.01, "temperature": 2500, "kappa_red": 10,\
              "kappa_blue": 0.5, "kappagamma": 10000.0, "Mchirp": Mchirp,\
              "q": Q, "cos_theta": v11, "cos_theta_open": 0.707107,\
              "disk_frac": 0.15, "radius_ns": 11.0, "alpha": 1.0,\
              "Mtov": 2.2, "cos_theta_cocoon": 0.5, "tshock": 1.7,\
              "temperature_shock": 100, "lumdist": DT11, "redshift": ZT11}
    p = light_curve(my_fitter, fixed_params)
    print(p)