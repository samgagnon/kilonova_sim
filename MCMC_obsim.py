import numpy as np
import requests
import urllib.parse
import os, sys, json
import mosfit
import subprocess as sub
import emcee as mc
import astropy.cosmology as cosmo
import astropy.units as u
import json as j
from astropy.constants import c
from astropy.cosmology import z_at_value
from astropy.io import fits
from random import choices
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from pca import *
from bns import *
# from gwsim import *

"""
A scientist wants H0. They have a list of events. 

INPUTS: For each event they have the localization,
the coverage footprint of the survey, GW parameters (mass ratio etc...).

For events w/ a kilonova counterpart, we just compute H0 as normal.

For events w/o a kilonova counterpart, is the lack of a counterpart due 
to inclination angle effects or H0 being different?

-> Input the sky footprint from LIGO, the footprint of all of the 
detector CAMS, the limiting depth of each exposure, and the mass ratio
of the merger itself.

----------------------------------------------------------------------

Outline a plan for how to infer H0 when you have some mergers with
counterparts and others without counterparts.
"""

def dpc_from_H0(H0_arr, z):
    dpc_arr = []
    for H0 in H0_arr:
        universe = cosmo.FlatLambdaCDM(H0, 0.27)
        dpc_arr.append(universe.luminosity_distance(z).value * 1e6)
    dpc_arr = np.asarray(dpc_arr)
    return dpc_arr

def H(x):
    """
    Heaviside step function
    """
    return 1/(1+np.exp(-50*x))

def delta1(D_0):
    """
    Uncertainty function one
    """
    m = 1.4
    M = ((m**2)**(3/5)/(2*m)**(1/5))**(5/6)
    r_0 = 6.5e3*M
    eps = 0.74
    sig = 1.03
    rad = np.sqrt((1+eps*np.cos(4*56*np.pi/180))/(sig*(1-eps**2)))
    return D_0*rad/r_0


def delta2(D_0):
    """
    Uncertainty function two
    """
    m = 1.4
    M = ((m**2)**(3/5)/(2*m)**(1/5))**(5/6)
    r_0 = 6.5e3*M
    eps = 0.74
    sig = 1.03
    rad = np.sqrt((1-eps*np.cos(4*56*np.pi/180))/(sig*(1-eps**2)))
    return D_0*rad/r_0

def dp(D_0, D, Dmax, v_0, v):
    # d1 = delta1(D_0)
    # d2 = delta2(D_0)
    d1 = 0.1
    d2 = 0.057
    e = np.exp(-1/(2*d1**2) * (v*D_0/D - v_0)**2 - 
              1/(2*d2**2)*((1+v**2)*D_0/(2*D) - (1+v_0**2)/2)**2)
    return D**2*e*H(D/D_0)*H(Dmax/D_0 - D/D_0)*H(1-v**2)/D_0**2

def pdf(D0, v0, dspace):
    # dspace = np.linspace(D0/3, 3*D0, int(1e3))
    vspace = np.linspace(-1, 1)
    pdf = 0
    for v in vspace:
        pdf += dp(D0, dspace, dspace[-1], v0, v)
    # plt.plot(dspace, pdf)
    # plt.show()
    if pdf.sum() > 0:
        pdf /= pdf.sum()
    return pdf

def d_est(D0, v0):
    dspace = np.linspace(D0/3, 3*D0, int(1e3))
    d_est = 0
    p = pdf(D0, v0, dspace)
    if p.sum() is not 1.0:
        p /= p.sum()
    # for j in range(len(p)):
        # d_est += p[j] * dspace[j]
    d_est = np.random.choice(dspace, p=p)
    # plt.plot(dspace, p, 'b-', label="Sim. Dist.")
    # plt.axvline(d_est, c='r', ls='--', label="Dist. Sample")
    # plt.title("Simulated Distance Distribution", fontsize=25)
    # plt.legend(fontsize=20)
    # plt.xlabel("Distance [Mpc]", fontsize=20)
    # plt.show()
    return d_est


def z_from_dpc_H0(H0_arr, dpc_arr):
    z_arr = []
    for i in range(len(H0_arr)):
        universe = cosmo.FlatLambdaCDM(H0_arr[i], 0.27)
        z_arr.append(z_at_value(universe.luminosity_distance, dpc_arr[i] * u.pc))
    z_arr = np.asarray(z_arr)
    return z_arr

def Y2(v):
    # equation 97 in the 1994 paper
    eps = 0.25
    sig = 1.0
    n_d = 3
    if v == 1.0:
        v = 0.99
    pol = np.random.uniform(0, 2*np.pi) # polarization angle
    num = n_d*((1+v**2) - eps*np.cos(4*pol)*(1-v**2))
    den = 2*sig*(1-eps**2)*(1-v**2)
    return num/den

def inferred_distance(dpc, angle):
    m = 1.1792386288325938
    r_0 = 6.5e9 * m
    n_d = 3
    delta_D = np.sqrt((8*dpc**4)*Y2(angle)/(n_d*r_0**2))
    print("uncertainty (Mpc):", delta_D/1e6)
    D = np.random.normal(loc=dpc, scale=delta_D)
    return D

def i_weight(w):
    sigma = 1470/np.sqrt(8*np.log(2))
    mu = 7835
    return np.exp(-1*((w - mu)**2)/(2*sigma**2))

def z_weight(w):
    sigma = 1520/np.sqrt(8*np.log(2))
    mu = 9260
    return np.exp(-1*((w - mu)**2)/(2*sigma**2))

def convert_flux(flux, wavelength, dpc):
    # adjust flux for viewing distance
    flux *= (10/dpc)**2
    # c = 299792458e7 # speed of light in angstroms per second
    flux_density = 3.34e4 * flux * (wavelength ** 2)
    return flux_density

def DEC_response():
    hdul = fits.open("STD_BANDPASSES_DR1.fits")
    return hdul[1].data

def interpolate_response(response, lams):
    extended_range = response['lambda'][-1]*np.linspace(1,1000, 50)
    lambda_range = np.concatenate((response['lambda'], extended_range))
    z_response = np.concatenate((response['z'], np.zeros(50)))
    i_response = np.concatenate((response['i'], np.zeros(50)))
    fz = interp1d(lambda_range, z_response)
    fi = interp1d(lambda_range, i_response)
    z_response = fz(lams[32:43])
    i_response = fi(lams[32:43])
    return {"lambda": lams, "z": z_response, "i": i_response}


def from_gwsim():
    bns = get_bns()
    time = bns['time']
    api_token = "zd6DGajyV66nlxuGhWt4CzIbdJ3wwr2UzlbkEg"
    graceid = ['S190425z', 'S200224ca']
    mapfile = ['LALInference.fits.gz,0', 'S200224ca.fits']
    # e1: [instrument name, band, depth, t1, t2, index of time]
    # this should also include the distance (profile?) of the event
    # for every instrument, I need to know the frequency range
    # of its observation band
    e1 = sim_event(api_token, graceid[0], mapfile[0], [int(time[0]), \
                   int(time[1]), int(time[2])])
    e2 = sim_event(api_token, graceid[1], mapfile[1], [int(time[0]), \
                   int(time[1]), int(time[2])])


def obsim(H0, mej, phi, angle, event_list):
    # n = np.random.uniform(0, 1)
    # cutoff = 0.95
    # if n < cutoff:
    #     detection = True
    # else:
    #     detection = False
    # if detection is False:
    #     print("no detection made")
    #     return [0]
    # else:
    # i_depth = 22.0 # 22.0
    z_depth = 21.3 # 21.3
    # angler = np.random.uniform(0, np.pi/2, (2, 10))
    # angle = np.sin(angler[0]) * np.cos(angler[1])
    ################################################################
    # determine if inferred distance is consistent with observations
    dpc_list = [dpc_from_H0(H0, event[1])[0] for event in event_list]
    d_inf_list = [d_est(dpc_list[i]/1e6, angle[i]) for i in range(len(event_list))]
    # pdl checker
    range_list = []
    for event in event_list:
        dist_range, ref_dist = event[0]
        ref_dist /= ref_dist.max()
        range_list += [[dist_range, ref_dist]]
    bounds = []
    dprob_list = []
    for i in range(len(range_list)):
        bounds += [range_list[i][0].min() <= d_inf_list[i] <= range_list[i][0].max()]
    if False in bounds:
        print("distance rejected")
        return [0]
    else:
        for i in range(len(event_list)):
            dprob_list += [interp1d(range_list[i][0], range_list[i][1])(d_inf_list[i])]
            # plt.plot(range_list[i][0], range_list[i][1], 'b-', label="Reference Dist.")
            # plt.plot(d_inf_list[i], interp1d(range_list[i][0], range_list[i][1])(d_inf_list[i]), 'r+', label="Dist. Sample")
            # plt.legend(fontsize=20)
            # plt.xlabel("Distance [Mpc]", fontsize=20)
            # plt.title("Reference Distance Distribution", fontsize=25)
            # plt.show()
            # let's just hold on to and save the dprobs as weights for the time being
        # rnum = np.random.uniform(0, 1)
        # if rnum > dprob_list.any():
            # print("distance rejected")
            # return [0]
    ################################################################
    # generate magnitude here and compare with cutoff
    detections = []
    mags = []
    dprob_global = min(dprob_list)
    for i in range(len(event_list)):
        runstr = "mosfit -m bns_generative -N 1 -S 4 --max-time 4 --band-systems AB --extra-outputs times model_observations all_bands -F texplosion -0.01 lumdist " + \
                str(event_list[i][2]) + " redshift " + str(event_list[i][1]) + " Mchirp " + str(1.188) + " q " + str(0.92) + " disk_frac " + \
                str(0.15) + " cos_theta " + str(event_list[0][3]) + " cos_theta_cocoon " + str(phi[i]) + " --band-list z -s my_name" 
        sub.run(runstr, cwd='./models/', input='22'.encode())
        with open("./models/products/bns_generative_extras_my_name.json") as f:
            d = j.load(f)
        observations = d['model_observations'][0] # this is a list of AB magnitudes
        print("event weight (Mpc):", dprob_list[i])
        print("inferred distance (Mpc)", d_inf_list[i])
        print("distance (Mpc):", event_list[i][2])
        print("observation angle (deg):", np.arccos(angle[i]) *180/np.pi)
        print("z band:", min(observations))
        mags += [min(observations)]
        detections += [any(observations) < z_depth]
    return detections, angle, [dprob_global]*len(d_inf_list), d_inf_list, mags


def gen_event(H0):
    # generate true distance
    O4range = np.linspace(1, 190, 1000)
    O4prob = O4range**2
    O4prob /= O4prob.sum()
    distance = np.random.choice(O4range, p=O4prob)
    # generate observed redshift
    z = z_from_dpc_H0([H0], [distance*1e6])[0]
    # generate distance posterior
    angler = np.random.uniform(0, np.pi/2, 2)
    angle = np.sin(angler[0]) * np.cos(angler[1])
    dspace = np.linspace(distance/3, 3*distance, int(1e3))
    pdl = pdf(distance, angle, dspace)
    if pdl.sum() is not 1.0:
        pdl /= pdl.sum()
    return [dspace, pdl], z, distance, angle


def MCMC_O4sim():
    """
    Creates 10 randomized events representative of the O4 observing run.
    Each is generated with a true seed H0, seed distances, and seed observing angles
    """
    H0_true = 72
    a = 0
    n = int(1)
    params = int(9)
    H0 = np.random.uniform(1, 150, 1)
    event_list = []
    for i in range(10):
        pdl, z, dl, v = gen_event(H0_true)
        event_list += [[pdl, z, dl, v]]
    results = walker(event_list, H0, params, n)
    np.savetxt('MCMC_O4_100k_772021_H072_interp_1-150', results)
    


def walker(event_list, H0, params, n):
    phi = np.random.uniform(15, 75, 10)
    mej = np.random.uniform(0.01, 0.1, 10)
    angler = np.random.uniform(0, np.pi/2, (2, 10))
    angle = np.sin(angler[0]) * np.cos(angler[1])
    # first step
    result = obsim(H0, mej, phi, angle, event_list)
    results = []
    if result != [0]:
        for i in range(len(event_list)):
            print(H0, event_list[i][1], event_list[i][2], \
                    phi[i], result[1][i], result[2][i], result[3][i], result[4][i], event_list[i][3], mej[i])
            results += [np.array([H0[0], event_list[i][1], event_list[i][2], \
                phi[i], result[1][i], result[2][i], result[3][i], result[4][i], event_list[i][3], mej[i]])]
    # else:
    #     results[i] = np.zeros((params))
    for j in range(1, n):
        a_H0, b_H0 = (1 - H0) / 20, (150 - H0) / 20
        H0 = truncnorm.rvs(a_H0, b_H0, loc=H0, scale=20, size=1)
        # dpc = dpc_from_H0(H0, z)[0]
        a_phi, b_phi = (15 - phi) / 15, (75 - phi) / 15
        phi = truncnorm.rvs(a_phi, b_phi, loc=phi, scale=15, size=10)
        a_mej, b_mej = (0.01 - mej) / 0.02, (0.1 - mej) / 0.02
        mej = truncnorm.rvs(a_mej, b_mej, loc=mej, scale=0.02, size=10)
        a_angle, b_angle = (0 - angle) / 0.2, (1 - angle) / 0.2
        angle = truncnorm.rvs(a_angle, b_angle, loc=mej, scale=0.2, size=10)
        result = obsim(H0, mej, phi, angle, event_list)
        if result != [0]:
            r = np.random.uniform(0, 1)
            if r > result[2][0]:
                H0 = results[-1][0]
                phi = results[-1][3]
                mej = results[-1][-1]
                angle = results[-1][-3]
            for i in range(len(event_list)):
                results += [np.array([H0[0], event_list[i][1], event_list[i][2], \
                    phi[i], result[1][i], result[2][i], result[3][i], result[4][i], event_list[i][3], mej[i]])]
    return results



if __name__ == "__main__":
    # O4sim()
    MCMC_O4sim()