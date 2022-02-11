import mosfit

import astropy.units as u

from scipy.stats import truncnorm
from astropy.constants import c
from astropy.cosmology import z_at_value
from scipy.special import erfc

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
#     return (True in detect), pobs
    return detect, 1


def nobs_forward(H0, ve, DL, pDL, event):
    # get free event parameters
    pdist = pdict[str(event)]
    v0 = v_from_CDF(pdist, ve[0])
    if k>2:
        M1, M2 = m_from_dm((ve[2], ve[3]), event)
    else:
        M1 = event[2][0]
        M2 = event[2][1]
    # use results from GWToolbox to get the true DL, M1, M2, v
    # TDL = event[1]
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
              "q": Q, "cos_theta": 1 - v0, "cos_theta_open": 0.707107,\
              "disk_frac": 0.15, "radius_ns": 11.0, "alpha": 1.0,\
              "Mtov": 2.2, "cos_theta_cocoon": 0.5, "tshock": 1.7,\
              "temperature_shock": 100, "lumdist": DL, "redshift": ZIC}
    det, pobs = light_curve(my_fitter, fixed_params)
#     return det, 1-pobs
    return det, pDL # if probabilistic magnitude is off


def obs_forward(H0, ve, event):
    # get free event parameters
    pdist = pdict[str(event)]
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
    v0 = v_from_DCDF(pdist, DIC, ve[0])
    # probability density of given cosmology-assigned distance at sampled angle v0
    # dprob, dpm, dps = p_DV(event[1], event[4], event[2][0], event[2][1], v0, DIC)
    # dprob, dpm, dps = p_DV(pdist, v0, DIC)
    # dprob /= dpm
    # print("dprob:", dprob)
#     dprob = 1.0
    fixed_params = {"ebv": 2.2, "rvhost": 3.1, "frad": 0.999, "nnhost": 1e18,\
              "texplosion": -0.01, "temperature": 2500, "kappa_red": 10,\
              "kappa_blue": 0.5, "kappagamma": 10000.0, "Mchirp": Mchirp,\
              "q": Q, "cos_theta": 1 - v0, "cos_theta_open": 0.707107,\
              "disk_frac": 0.15, "radius_ns": 11.0, "alpha": 1.0,\
              "Mtov": 2.2, "cos_theta_cocoon": 0.5, "tshock": 1.7,\
              "temperature_shock": 100, "lumdist": DIC, "redshift": TZ}
    det, pobs = light_curve(my_fitter, fixed_params)
    return det, pobs


def forward(v):
    # get terms from input vector
    H0 = v[0]
    print("Input H0:", H0)
    p = 0
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
            pdist = pdict[str(event)]

            # compute summary statistic
            v0 = v_from_DCDF(pdist, DIC, ve[0])
            vline = np.linspace(0, 1, 1000)
            dline = np.linspace(DIC/3, DIC*5, 1000)
            gi = np.abs(v0 - vline).argmin()
            pslice = pdist[0][:, gi]
            pslice /= pslice.sum()
            mean = np.average(dline, weights=pslice)
            var = np.average((dline-mean)**2, weights=pslice)
            # print(DIC, mean, var)
            p += (DIC-mean)**2/(2*var)
            # p *= erfc(np.abs(DIC-mean)/np.sqrt(var))

            dprob, dpm, dps = p_DV(pdist, v0, DIC)
            # if dprob/dps < threshold:
            #     return dict(x=np.array([0.0]))
    nobs_dl = []
    if nobs_list is not None:
        # is this bit really necessary? I think not.
        i = 0
        ve_list = list(chunks(v[1:n_nobs*(k+1)+1], k+1))
        for event in nobs_list:
            ve = ve_list[i]
            i += 1
            # generate a possible distance from the random variables du, v0
            # and the prior distribution (depending on true event parameters)
            pdist = pdict[str(event)]
            # v0 = v_from_CDF(event[1], event[4], event[2][0], event[2][1], ve[0])
            v0 = v_from_CDF(pdist, ve[0])

            DL, pDL, dpm, dps = D_vdu(event[1], event[4], event[2][0], event[2][1], v0, ve[1], pdist)
            # TODO: produce similar list for v in both this and obs_list
            nobs_dl += [[DL, pDL/dpm]]
            if pDL/dps < threshold:
                return dict(x=np.array([0.0]))
    j = 0
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
            # pdist = pdict[str(event)]
            # dprob, dpm, dps = p_DV(pdist, v0, DIC)
            # p += [pobs]
    if nobs_list is not None:
        i = 0
        ve_list = list(chunks(v[1:n_nobs*(k+1)+1], k+1))
        for event in nobs_list:
            ve = ve_list[i]
            i += 1
            DL, pDL = nobs_dl[j]
            j += 1
            det, pobs = nobs_forward(H0, ve, DL, pDL, event)
            if det is True:
                print(det, 'bad')
                return dict(x=np.array([0.0]))
            # possibly remove pobs for nobs?
            # p *= pobs
    x = [erfc(np.sqrt(p/(n_obs-1)))]
    return dict(x=np.array(x))


if __name__ == "__main__":
    # event = event_list[0]
    # M1 = event[2][0]
    # M2 = event[2][1]
    # Q = M1/M2
    # Mchirp = ((M1*M2)**3/(M1+M2))**(1/5)
    # print(Q, Mchirp)
    # print(event[0], event[1])
    # fixed_params = {"ebv": 2.2, "rvhost": 3.1, "frad": 0.999, "nnhost": 1e18,\
    #           "texplosion": -0.01, "temperature": 2500, "kappa_red": 10,\
    #           "kappa_blue": 0.5, "kappagamma": 10000.0, "Mchirp": Mchirp,\
    #           "q": Q, "cos_theta": 0.89, "cos_theta_open": 0.707107,\
    #           "disk_frac": 0.15, "radius_ns": 11.0, "alpha": 1.0,\
    #           "Mtov": 2.2, "cos_theta_cocoon": 0.5, "tshock": 1.7,\
    #           "temperature_shock": 100, "lumdist": 141, "redshift": event[0]}
    # p = light_curve(my_fitter, fixed_params)
    # print(p)
    a = 0
    outlist = []
    while a<1:
        # h0 = np.random.uniform(65, 75)
        h0 = 70
        vu = np.random.uniform(0, 1, k*5+(k+1)*5)
        v = np.asarray([h0]+list(vu))
        pingas = forward(v)
        print(v)
        print(pingas)
        a += 1