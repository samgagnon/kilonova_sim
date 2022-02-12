import numpy as np

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
    # get values
    v1 = vline[v1i]
    vl = vline[-vli]
    d1 = dline[d1i]
    dl = dline[-dli]
    if vl == 0:
        vl = 1
    if dl == 0:
        dl = 1
    # re-create pmesh on more specific domain
    vline = np.linspace(v1, vl, 1000)
    dline = np.linspace(d1, dl, 1000)
    vv, dd = np.meshgrid(vline, dline)
    positions = np.vstack([vv.ravel(), dd.ravel()])
    values = np.vstack([vline, dline])
    pmesh = dp(D0, dd, v0, vv, m1, m2)
    if plot:
        vloc = np.abs(vline - v0).argmin()
        dloc = np.abs(dline - D0).argmin()
        ix = np.arccos(1 - vline[vloc])*180/np.pi
        Dx = dline[dloc]
        dsamp = 135.9200303446712
        # vsamp = 0.10717223730236744
        vsamp = 0.09948687426164904
        print(v0, D0)
        isamp = np.arccos(1 - vsamp)*180/np.pi
        vsloc = np.abs(vline - vsamp).argmin()
        dsloc = np.abs(dline - dsamp).argmin()
        print(dsloc, vsloc)
        print('Sample height: ', pmesh[dsloc, vsloc])
        print('True height: ', pmesh[dloc, vloc])
        iDM, ivM = np.unravel_index(pmesh.argmax(), pmesh.shape)
        iM = np.arccos(1 - vline[ivM])*180/np.pi
        print('Max height: ', pmesh.max())
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.set_figwidth(10)
        fig.set_figheight(10)
        iline = np.arccos(1 - vline)*180/np.pi
        i0 = np.arccos(1 - v0)*180/np.pi
        plt.pcolor(iline, dline, pmesh, shading='auto')
        plt.plot(i0, D0,'+r')
        plt.plot(ix, Dx,'+k')
        plt.plot(iM, dline[iDM],'+m')
        plt.plot(isamp, dsamp,'+b')
        plt.xlabel("$\iota$ [$^\circ$]", fontsize=20)
        plt.ylabel("$D_L$ [Mpc]", fontsize=20)
        plt.colorbar()
        plt.show()
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


def D_vdu(d_true, v_true, m1, m2, v_guess, du, pdist):
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
    vloc = np.abs(pdist[3] - v_guess).argmin()
    dloc = np.abs(pdist[-1] - d_guess).argmin()
    return pdist[0][dloc, vloc], pdist[0].max(), pdist[0].sum()


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


def plot_all(pdist, v, d, event):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    iline = np.arccos(1 - pdist[-2])*180/np.pi
    i0 = np.arccos(1 - v)*180/np.pi
    plt.pcolor(iline, pdist[-1], pdist[0], shading='auto')
    plt.plot(i0, d,'+r')
    plt.plot(np.arccos(1 - event[4])*180/np.pi, event[1],'+k')
    plt.xlabel("$\iota$ [$^\circ$]", fontsize=20)
    plt.ylabel("$D_L$ [Mpc]", fontsize=20)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    from config import *
    # xr = np.linspace(0.7, 0.95, int(1e2))
    # vr = [v_from_CDF(DT1, 0.9, 1.4, 1.4, i) for i in xr]
    # p = [p_DV(DT1, 0.9, m1[0], m1[1], v, DT1) for v in vr]
    # e = np.array(p).argmax()
    # print(xr[e])
    # print(vr[e])
    gen_pDV_dists(event_list, plot=True)
    # import matplotlib.pyplot as plt
    # from astropy import units as u
    # from astropy import constants as c
    # universe = cosmo.FlatLambdaCDM(70, 0.3)
    # D0 = 40
    # z = cosmo.z_at_value(universe.luminosity_distance, D0*u.Mpc)
    # v0 = np.cos(np.linspace(0, np.pi/2, 5))
    # # thetaline = np.linspace(0, np.pi, 1000)
    # # vline = np.cos(thetaline)
    # vline = np.linspace(-1, 1, 1000)
    # dline = np.linspace(D0/3, D0*5, 1000)
    # vv, dd = np.meshgrid(vline, dline)
    # positions = np.vstack([vv.ravel(), dd.ravel()])
    # values = np.vstack([vline, dline])
    # for v in v0:
    #     pmesh = dp(D0, dd, v, vv, 1.4, 1.4)
    #     plt.plot(z*c.c.value/(dline*1e3), np.sum(pmesh, 1), label=str(np.arccos(v)*180/np.pi)+'$^\circ$')
    # plt.axvline(70, color='r')
    # plt.legend()
    # plt.show()