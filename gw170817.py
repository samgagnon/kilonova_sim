import numpy as np
# import healpy as hp
import requests
import urllib.parse
import os, sys, json
import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import z_at_value
from random import choices
from datetime import datetime
from bns import *


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

def pdf(D0, v0):
    dspace = np.linspace(D0/3, 3*D0, int(1e3))
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
    d_est = 0
    dspace = np.linspace(D0/3, 3*D0, int(1e3))
    p = pdf(D0, v0)
    if p.sum() is not 1.0:
        p /= p.sum()
    for j in range(len(p)):
        d_est += p[j] * dspace[j]
    # d_est = np.random.choice(dspace, p=p)
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

def main(bns, dpc, z):
    n = np.random.uniform(0, 1)
    if dpc < 8:
        cutoff = 0.9
    elif 25 < dpc < 50:
        cutoff = 0.8
    else:
        cutoff = 0.5
    if n < cutoff:
        detection = True
    else:
        detection = False
    if detection is False:
        print("no detection made")
        return [0]
    else:
        pos_angles = []
        i_depth = 20.0 # 22.0
        z_depth = 20.0 # 21.3
        # convert depth to correct flux density
        wavecollection = [bns['oa0.1'], bns['oa0.2'], bns['oa0.3'], 
            bns['oa0.4'], bns['oa0.5'], bns['oa0.6'], bns['oa0.7'],
            bns['oa0.8'], bns['oa0.9'], bns['oa1.0']]
        angler = np.random.uniform(0, np.pi/2, 2)
        angle = np.sin(angler[0]) * np.cos(angler[1])
        if angle <= 0.1:
            wave = bns['oa0.1']
            # angle = 0.1
        elif 0.1 < angle <= 0.2:
            wave = bns['oa0.2']
            # angle = 0.2
        elif 0.2 < angle <= 0.3:
            wave = bns['oa0.3']
            # angle = 0.3
        elif 0.3 < angle <= 0.4:
            wave = bns['oa0.4']
            # angle = 0.4
        elif 0.4 < angle <= 0.5:
            wave = bns['oa0.5']
            # angle = 0.5
        elif 0.5 < angle <= 0.6:
            wave = bns['oa0.6']
            # angle = 0.6
        elif 0.6 < angle <= 0.7:
            wave = bns['oa0.7']
            # angle = 0.7
        elif 0.7 < angle <= 0.8:
            wave = bns['oa0.8']
            # angle = 0.8
        elif 0.8 < angle <= 0.9:
            wave = bns['oa0.9']
            # angle = 0.9
        else:
            wave = bns['oa1.0']
            # angle = 1.0
        wti, wtz = 0, 0
        flux_i, flux_z = 0, 0
        flux_i_list = []
        flux_z_list = []
        for waveform in wave:
            wl_obs = (z + 1) * waveform[0]
            # wl_obs = waveform[0]
            if 6365 < wl_obs < 9305:
                flux_i_list.append(convert_flux(waveform[3], wl_obs, dpc))
                # flux_i += convert_flux(waveform[3], wl_obs, dpc) * i_weight(wl_obs)
                # wi = i_weight(wl_obs)
                # if wi > wti:
                #     wti = wi 
            elif 7740 < wl_obs < 10780:
                flux_z_list.append(convert_flux(waveform[3], wl_obs, dpc))
                # flux_z += convert_flux(waveform[3], wl_obs, dpc) * z_weight(wl_obs)
                # wz = z_weight(wl_obs)
                # if wz > wtz:
                #     wtz = wz
        filen, fzlen = len(flux_i_list), len(flux_z_list)
        if filen > 0:
            mid = int(filen/2)
            flux_i = flux_i_list[mid]
            mAB_i = 8.9 - 2.5 * np.log10(flux_i)
        else:
            mAB_i = 100
        if fzlen > 0:
            mid = int(fzlen/2)
            flux_z = flux_z_list[mid]
            mAB_z = 8.9 - 2.5 * np.log10(flux_z)
        else:
            mAB_z = 100
        # d_inf = dpc / 1e6
        d_inf = d_est(dpc/1e6, angle)
        print("inferred distance (Mpc):", d_inf)
        print("distance (Mpc):", dpc/1e6)
        print("observation angle (deg):", np.arccos(angle) *180/np.pi)
        print("i band:", mAB_i, i_depth)
        print("z band:", mAB_z, z_depth)
        if mAB_i < i_depth or mAB_z < z_depth:
            detection = True
            print("detection made")
            return detection, angle, d_inf
        else:
            detection = False
            print("no detection made")
            return detection, angle, d_inf

        


if __name__ == "__main__":
    n = int(1e5)
    params = int(9)
    results = np.zeros((n, params))
    # a = np.asarray(list(range(1,151)))
    # b = a**2
    # p = b/b.sum()
    # dpc_array = np.random.choice(a*1e6, n, p=p)
    H0_array = np.random.uniform(60, 90, n)
    # z_array = z_from_dpc_H0(H0_array, dpc_array)
    z_array = np.array([0.0099]*n)
    # z_array = np.array([0.1]*n)
    dpc_array = dpc_from_H0(H0_array, z_array[0])
    print("Distance Max, Min: ", dpc_array.max()/1e6, dpc_array.min()/1e6)
    for i in range(n):
        bns = get_bns()
        result = main(bns, dpc_array[i], z_array[i])
        if result != [0]:
            results[i] = np.array([result[0], H0_array[i], z_array[i], dpc_array[i], bns['mc_packets'], \
            bns['total_ejecta_mass'], bns['half_angle'], result[1], result[2]])
        else:
            results[i] = np.zeros((params))
    np.savetxt('100k_3172021_sampled_60-90', results)