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
#     m = 1.4
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
    return Dd**2*e*H(D/D_0)*H(Dd)*H(Dmax/D_0 - Dd)*H(1-v**2)


def pDV_dist(D0, v0, m1, m2, plot=False):
    thetaline = np.linspace(0, np.pi/2, 100)
    vline = np.cos(thetaline)
    dline = np.linspace(D0/3, D0*5, 100)
    vv, dd = np.meshgrid(vline, dline)
    positions = np.vstack([vv.ravel(), dd.ravel()])
    values = np.vstack([vline, dline])
    pmesh = dp(D0, dd, v0, vv, m1, m2)
    if plot:
        import matplotlib.pyplot as plt
        dps = pmesh.sum()
        fig = plt.figure()
        fig.set_figwidth(10)
        fig.set_figheight(10)
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(vv, dd, pmesh/dps, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none', zorder=1)
        ax.plot_surface(vv, dd, np.ones((100, 100))*1e-5, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none', zorder=2)
        # ax.plot(0.3, 400, pm, 'k.', zorder=20)
        # ax.plot(0.34, 453, pm, 'kx', zorder=20)
        # ax.plot(0.3003308776117506, 400.5338672005339, pm, 'r.', zorder=20)
        ax.set_xlabel("$v$", fontsize=20)
        ax.set_ylabel("$D_L$ [Mpc]", fontsize=20)
        ax.set_zlabel("$P(v,D_L)$", fontsize=20)
        ax.view_init(azim=30, elev=30)
        plt.show()
    return pmesh, values, positions, vline, dline


def gen_pDV_dists(event_list, plot=False):
    """
    Given the event list, return a dictionary with all
    relevant DL-v meshgrids using the event string as keys.
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

    # TODO: Re-engineer this and p_DV to take dv and calculate v using the same pdist

    # pdist = pDV_dist(d_true, v_true, m1, m2)
    vloc = np.abs(pdist[3] - v_guess).argmin()
    p_at_v = pdist[0][...,vloc]
    cdf = np.cumsum(p_at_v)
    cdf /= cdf.max()
    dloc = np.abs(cdf - du).argmin()
    return pdist[-1][dloc], pdist[0][dloc, vloc], pdist[0].max(), pdist[0].sum()


def p_DV(d_true, v_true, m1, m2, v_guess, d_guess):
    """
    -d_true: true event luminosity distance
    -v_true: true event angle variable
    -m1: true event m1
    -m2: true event m2
    -v_guess: guess for angle variable
    -d_guess: guess for luminosity distance
    
    returns: probability of guess given prior
    """
    pdist = pDV_dist(d_true, v_true, m1, m2)
    vloc = np.abs(pdist[3] - v_guess).argmin()
    dloc = np.abs(pdist[-1] - d_guess).argmin()
    am = pdist[0].argmax()
    
    ovloc = np.abs(pdist[3] - pdist[2][0][am]).argmin()
    odloc = np.abs(pdist[-1] - pdist[2][1][am]).argmin()
    return pdist[0][dloc, vloc], pdist[0].max(), pdist[0].sum()


def v_from_CDF(d_true, v_true, m1, m2, dv):
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
    pdist = pDV_dist(d_true, v_true, m1, m2)
    vdist = np.sum(pdist[0], 0)
    vdist /= vdist.sum()
    vcdf = np.cumsum(vdist)
    vloc = np.abs(vcdf - dv).argmin()
    v = pdist[3][vloc]
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
    gaus1 = np.exp((mx - mean_tuple[0])**2/(2*std_tuple[0]**2))
    gcdf1 = np.cumsum(gaus1)
    i1 = np.abs(dm_tuple[0] - gcdf1).argmin()
    m1 = mx[i1]
    gaus2 = np.exp((mx - mean_tuple[1])**2/(2*std_tuple[1]**2))
    gcdf2 = np.cumsum(gaus2)
    i2 = np.abs(dm_tuple[1] - gcdf2).argmin()
    m2 = mx[i2]
    return m1, m2

if __name__ == "__main__":
    from config import *
    xr = np.linspace(0.7, 0.95, int(1e2))
    vr = [v_from_CDF(DT1, 0.9, 1.4, 1.4, i) for i in xr]
    p = [p_DV(DT1, 0.9, m1[0], m1[1], v, DT1) for v in vr]
    e = np.array(p).argmax()
    print(xr[e])
    print(vr[e])