import numpy as np
from scipy.interpolate import LinearNDInterpolator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bns import *

def get_all_bns():
    path = "../kilonova_models/bns_m1_2comp/"
    fn_list = os.listdir(path)
    fn_list.remove('citations.bib')
    fn_list.remove('README')
    bns_list = []
    for fn in fn_list:
        bns = file_to_list(path + fn)
        bns_list.append(bns)
    return bns_list

def bns_dict2list(bns_list):
    bns_table = []
    for bns in bns_list:
        mej = bns['total_ejecta_mass']
        phi = bns['half_angle']
        t0 = bns['T0']
        thetal = [bns['oa0.0'], bns['oa0.1'], bns['oa0.2'], bns['oa0.3'],\
                 bns['oa0.4'], bns['oa0.5'], bns['oa0.6'], \
                 bns['oa0.7'], bns['oa0.8'], bns['oa0.9'], \
                 bns['oa1.0']]
        vl = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        tl = list(np.linspace(bns['time'][1], bns['time'][2], int(bns['time'][0])))
        i = 0
        for theta in thetal:
            j = 0
            for wave in theta.T[1:]:
                bns_table += [[mej, phi, t0, vl[i], tl[j]] + list(wave)]
                j += 1
            i += 1
    return np.asarray(bns_table)

def pcomp_matrix(bns_table):
    X = bns_table[:,5:]
    sc = StandardScaler()
    Y = sc.fit_transform(X)
    pca = PCA(0.999)
    pca.fit(Y)
    comp = pca.transform(Y)
    mean = pca.mean_
    components = pca.components_
    var = pca.explained_variance_
    ds0 = np.shape(X)[0]
    ds1 = 13
    pcm = np.zeros((ds0, ds1))
    pcm[:,:5] = bns_table[:,:5]
    pcm[:,5:] = comp
    return pcm, components, sc

def create_interpolators(pcm):
    points = (pcm[:,1], pcm[:,3], pcm[:,4])
    alpha = pcm[:,5]
    beta = pcm[:,6]
    gamma = pcm[:,7]
    delta = pcm[:,8]
    epsilon = pcm[:,9]
    eta = pcm[:,10]
    iota = pcm[:,11]
    omicron = pcm[:,12]
    global interp_alpha
    global interp_beta
    global interp_gamma
    global interp_delta
    global interp_epsilon
    global interp_eta
    global interp_iota
    global interp_omicron
    interp_alpha = LinearNDInterpolator(points, alpha)
    interp_beta = LinearNDInterpolator(points, beta)
    interp_gamma = LinearNDInterpolator(points, gamma)
    interp_delta = LinearNDInterpolator(points, delta)
    interp_epsilon = LinearNDInterpolator(points, epsilon)
    interp_eta = LinearNDInterpolator(points, eta)
    interp_iota = LinearNDInterpolator(points, iota)
    interp_omicron = LinearNDInterpolator(points, omicron)

def interpolate(pcm, components, sc, point):
    pc_coeff = np.array([interp_alpha(point), interp_beta(point), interp_gamma(point), \
            interp_delta(point), interp_epsilon(point), interp_eta(point), \
            interp_iota(point), interp_omicron(point)])
    interp_curve = sc.inverse_transform(pc_coeff@components)
    return interp_curve