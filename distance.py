import numpy as np
from gwtoolbox.functions_earth import rho_sq_core

random_angles_more=np.loadtxt("angles_more.dat")

def tel_fun(dns, z, m1, m2, iota, rho_cri, ant_fun, noise_tab):
    """
    The telescope function of Laser Interferometers and kHz sources.

    Parameters:
      z (float): The redshift of the GW source
      m1 (float): Red-shifted masses of the BHB
      m2 (float): Red-shifted masses of the BHB
      #chi (float): spin
      iota (float): inclination angle in radians
      rho_cri (float): The detection SNR threshold
      ant_fun (function): antenna pattern
      noise_tab (array of dtype float): noise function for detector

    Returns:
      (float): The probability of detection
    """
    # both masses should be intrinsic here.
    Mch = (m1*m2)**(5/6)/(m1+m2)**(1/5)
    
    theta_array_more = random_angles_more[:,0]
    varphi_array_more = random_angles_more[:,1]
    iota_array_more = np.ones(len(random_angles_more[:,1]))*iota
    psi_array_more = random_angles_more[:,3]
    F = ant_fun(theta_array_more, varphi_array_more, psi_array_more)
    A_array = dns.mod_norm(Mch*(1+z), F, iota_array_more, z)
    
    f_up = dns.freq_limit(m1=m1*(1+z), m2=m2*(1+z), chi=0)
    f2=dns.freq_limit_merger(m1=m1*(1+z), m2=m2*(1+z), chi=0)
    f3=dns.freq_limit_ringdown(m1=m1*(1+z), m2=m2*(1+z), chi=0)
    freq_sig=dns.freq_sigma(m1=m1*(1+z), m2=m2*(1+z), chi=0)
    f1=f_up
    rho_sq_core_value = rho_sq_core(noise_tab, dns.mod_shape, f_up=f_up)
    
    if len(A_array.shape)==2:
        rho_sq_array=4.*np.einsum('i...,i->i...',A_array**2,rho_sq_core_value)
        heav_array = np.heaviside(rho_sq_array-rho_cri**2,0)

        return np.mean(heav_array,axis=1)
    else: 
        rho_sq_array = np.array(4.*A_array**2*rho_sq_core_value)
        heav_array = np.heaviside(rho_sq_array-rho_cri**2,0)
        return np.mean(heav_array)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def H(x):
    """
    Heaviside step function
    """
    return 1/(1+np.exp(-50*x))


def delta1(D_0, m1, m2):
    """
    Uncertainty function one
    """
    M = ((m1*m2)**(3/5)/(m1+m2)**(1/5))**(5/6)
    r_0 = 6.5e3*M
    eps = 0.74
    sig = 1.03
    rad = np.sqrt((1+eps*np.cos(4*56*np.pi/180))/(sig*(1-eps**2)))
    return D_0*rad/r_0


def delta2(D_0, m1, m2):
    """
    Uncertainty function two
    """
#     m = 1.4
    M = ((m1*m2)**(3/5)/(m1+m2)**(1/5))**(5/6)
    r_0 = 6.5e3*M
    eps = 0.74
    sig = 1.03
    rad = np.sqrt((1-eps*np.cos(4*56*np.pi/180))/(sig*(1-eps**2)))
    return D_0*rad/r_0


def dp(D_0, D, v_0, v, m1, m2):
    d1 = delta1(D_0, m1, m2) # 0.1 values in 90s paper
    d2 = delta2(D_0, m1, m2) # 0.057
    M = ((m1*m2)**(3/5)/(m1+m2)**(1/5))**(5/6)
    Dmax = 6.5e3*M
    Dd = D/D_0
    e = np.exp(-1/(2*d1**2) * (v/Dd - v_0)**2 - 
              1/(2*d2**2)*((1+v**2)/(2*Dd) - (1+v_0**2)/2)**2)
    return e*H(D/D_0)*H(Dd)*H(Dmax/D_0 - Dd)*H(1-v**2) # Dd**2* prefactor removed to undo low H0 bias


def pDV_dist(D0, v0, m1, m2, plot=False):
    # generate generic range
    vline = np.linspace(0, 1, 1000)
    dline = np.linspace(D0/3, D0*5, 1000)
    vv, dd = np.meshgrid(vline, dline)
    positions = np.vstack([vv.ravel(), dd.ravel()])
    values = np.vstack([vline, dline])
    pmesh = dp(D0, dd, v0, vv, m1, m2)
    # search for non-informative regions
    vpdf = np.sum(pmesh, 0)
    vpdf /= vpdf.sum()
    dpdf = np.sum(pmesh, 1)
    dpdf /= dpdf.sum()
    # get indices
    v1i = (vpdf>1e-4).argmax()
    vli = (vpdf[::-1]>1e-4).argmax()
    d1i = (dpdf>1e-4).argmax()
    dli = (dpdf[::-1]>1e-4).argmax()
    if vli == 0:
        vli = 1
    if dli == 0:
        dli = 1
    # get values
    v1 = vline[v1i]
    vl = vline[-vli]
    d1 = dline[d1i]
    dl = dline[-dli]
    # re-create pmesh on more specific domain
    vline = np.linspace(v1, vl, 1000)
    dline = np.linspace(d1, dl, 1000)
    vv, dd = np.meshgrid(vline, dline)
    positions = np.vstack([vv.ravel(), dd.ravel()])
    values = np.vstack([vline, dline])
    pmesh = dp(D0, dd, v0, vv, m1, m2)
    return pmesh, values, positions, vline, dline


def gen_pDV_dists(event_list, plot=False):
    """
    Given the event list, return a dictionary with all
    relevant DL-v meshgrids using the event strings as keys.
    """
    pdict = {}
    for event in event_list:
        # generate pdist
        pdist = pDV_dist(event[1], event[4], event[2][0], event[2][1], plot)
        # update dictionary
        pdict.update({str(event):pdist})
    return pdict


def D_vdu(v_guess, du, pdist):
    """
    -d_true: true event luminosity distance
    -v_true: true event angle variable
    -m1: true event m1
    -m2: true event m2
    -v_guess: guess for angle variable
    -du: random variable to determine distance from inverse CDF
    
    returns: (tuple) luminosity distance, probability of selection
    """
    vloc = np.abs(pdist[3] - v_guess).argmin()
    p_at_v = np.copy(pdist[0][...,vloc])
    cdf = np.cumsum(p_at_v)
    cdf /= cdf.max()
    dloc = np.abs(cdf - du).argmin()
    return pdist[-1][dloc], pdist[0][dloc, vloc], pdist[0].max(), pdist[0].sum()


def p_DV(pdist, v_guess, d_guess):
    """
    -d_true: true event luminosity distance
    -v_true: true event angle variable
    -m1: true event m1
    -m2: true event m2
    -v_guess: guess for angle variable
    -d_guess: guess for luminosity distance
    
    returns: probability of guess given prior
    """
    if (d_guess < pdist[-1].max()) and (d_guess > pdist[-1].min()):
        vloc = np.abs(pdist[3] - v_guess).argmin()
        dloc = np.abs(pdist[-1] - d_guess).argmin()
        return pdist[0][dloc, vloc], pdist[0].max(), pdist[0].sum()
    else:
        return 0, 1, 1


def v_from_CDF(pdist, dv):
    """
    Generates v from GW cdf
    -d_true: true event luminosity distance
    -v_true: true event angle variable
    -m1: true event m1
    -m2: true event m2
    -dv: random variable to determine v from inverse CDF

    returns: 
    -v: cosine of observation angle
    """
    vdist = np.copy(np.sum(pdist[0], 0))
    vdist /= vdist.sum()
    vcdf = np.cumsum(vdist)
    vloc = np.abs(vcdf - dv).argmin()
    v = pdist[3][vloc]
    return v

def v_from_DCDF(pdist, DL, dv):
    """
    Generates v from GW cdf
    -d_true: true event luminosity distance
    -v_true: true event angle variable
    -m1: true event m1
    -m2: true event m2
    -dv: random variable to determine v from inverse CDF

    returns: 
    -v: cosine of observation angle
    """
    dloc = np.abs(pdist[-1] - DL).argmin()
    vdist = np.copy(pdist[0][dloc])
    vdist /= vdist.sum()
    vcdf = np.cumsum(vdist)
    vloc = np.abs(vcdf - dv).argmin()
    v = pdist[3][vloc]
    # import matplotlib.pyplot as plt
    # plt.plot(pdist[3], vcdf)
    # plt.axvline(0.1)
    # plt.axvline(v)
    # plt.show()
    return v

def m_from_dm(dm_tuple, event):
    """
    Calculates NS mass from iCDF sample.
    arguments:
    - dm_tuple: iCDF samples from uniform prior
    - event: relevant event
    """
    mean_tuple = event[2]
    std_tuple = event[3]
    mx = np.linspace(1.1, 2.5, 1000)
    gaus1 = np.exp(-1*(mx - mean_tuple[0])**2/(2*std_tuple[0]**2))
    gcdf1 = np.cumsum(gaus1)
    i1 = np.abs(dm_tuple[0] - gcdf1).argmin()
    m1 = mx[i1]
    gaus2 = np.exp(-1*(mx - mean_tuple[1])**2/(2*std_tuple[1]**2))
    gcdf2 = np.cumsum(gaus2)
    i2 = np.abs(dm_tuple[1] - gcdf2).argmin()
    m2 = mx[i2]
    return m1, m2


def plot_all(pdist, v, d, event, cmap='coolwarm'):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    iline = np.arccos(pdist[-2])*180/np.pi
    i0 = np.arccos(v)*180/np.pi
    # print(v, i0)
    plt.pcolor(iline, pdist[-1], pdist[0]/np.sum(pdist[0]), shading='auto', cmap=cmap)
    # plt.plot(i0, d,'+r')
    # plt.plot(np.arccos(event[4])*180/np.pi, event[1],'+k')
    plt.xlabel("$\iota$ [$^\circ$]", fontsize=20)
    plt.ylabel("$D_L$ [Mpc]", fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('$P(D_L, \iota)$', fontsize=20)
    plt.show()


if __name__ == "__main__":
    from config import *
    pdict = gen_pDV_dists(event_list[0:1], plot=False)
    for event in event_list[0:1]:
        pdist = pdict[str(event)]
        max = pdist[0].max()/pdist[0].sum()
        a = 0
        while a<100:
            max *= max
            a += 1
        print(max)