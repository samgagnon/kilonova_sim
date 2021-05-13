import numpy as np
import healpy as hp
import requests
import urllib.parse
import os, sys, json
from random import choices
from datetime import datetime
from bns import *


def get_time_and_band(instrument='MMT'):
    """
    Returns times, bands, and depths related to an instrument.
    """
    BASE = 'http://treasuremap.space/api/v0'
    api_token = "zd6DGajyV66nlxuGhWt4CzIbdJ3wwr2UzlbkEg"
    
    TARGET = 'pointings'

    #define the filtering parameters 
    #grab all of the completed pointings taken with a specific instrument for a given event

    graceid = 'S190425z'

    params = {
        "api_token":api_token,
        "instrument":instrument,
        "graceid":graceid,
        "status":"completed"
    }

    url = "{}/{}?{}".format(BASE, TARGET, urllib.parse.urlencode(params))
    r = requests.get(url = url)
    print("There are %s pointings" % len(json.loads(r.text)))

    #print the first
    # print(json.loads(r.text))

    pointings = []
    positions = []
    depths = []
    depth_units = []
    depth_errs = []
    time = []
    pos_angle = []
    band = []
    for p in range(len(json.loads(r.text))):
        pointing = json.loads(json.loads(r.text)[p])
        pointings.append(json.loads(json.loads(r.text)[p]))
        positions.append(pointing["position"])
        depths.append(pointing["depth"])
        depth_units.append(pointing["depth_unit"])
        depth_errs.append(pointing["depth_err"])
        time.append(pointing["time"])
        pos_angle.append(pointing["pos_angle"])
        band.append(pointing["band"])
    
    return time, band, [depths, depth_units, depth_errs]


def get_pointings():

    BASE = 'http://treasuremap.space/api/v0/'
    TARGET = 'pointings'
    api_token = "zd6DGajyV66nlxuGhWt4CzIbdJ3wwr2UzlbkEg"

    payload = {'graceid':'S190425z', 'api_token':api_token}
    r = requests.get(url = BASE+TARGET, params = payload)
    results = r.json()

    pointings = []
    for tile in results:
        tile = eval(tile)
        position = tile["position"]
        ra = float(position.split()[1].split('(')[1])
        dec = float(position.split()[2].split(')')[0])
        pos_angle = tile['pos_angle']
        instrument_id = tile['instrumentid']
        pointings.append([instrument_id, ra,dec, pos_angle])
    return np.asarray(pointings)


def get_footprints():
    """
    Gets the footprints for each instrument
    """
    BASE = 'http://treasuremap.space/api/v0/'
    TARGET='footprints'
    api_token = "zd6DGajyV66nlxuGhWt4CzIbdJ3wwr2UzlbkEg"

    payload = {'graceid':'S190425z', 'api_token':api_token}
    r = requests.get(url = BASE+TARGET, params = payload)
    results = r.json()

    footprints_dict={}
    for footprint_obj in results:
        footprint_obj = eval(footprint_obj)
        inst_id = int(footprint_obj['instrumentid'])
        footprint = footprint_obj['footprint']
        sanitized = footprint.strip('POLYGON ').strip(')(').split(',')
        footprint = []
        for vertex in sanitized:
            obj = vertex.split()
            ra = float(obj[0])
            dec = float(obj[1])
            footprint.append([ra,dec])
        new_entry = {inst_id:footprint}
        footprints_dict.update(new_entry)
    return footprints_dict


def get_instruments():
    BASE = 'http://treasuremap.space/api/v0/'
    TARGET = 'instruments'
    api_token = "zd6DGajyV66nlxuGhWt4CzIbdJ3wwr2UzlbkEg"

    payload = {'graceid':'S190425z', 'api_token':api_token}
    r = requests.get(url = BASE+TARGET, params = payload)
    results = r.json()

    instrument_dict={}
    for instrument in results:
        instrument=eval(instrument)
        inst_id = instrument['id']
        inst_name = instrument['instrument_name']
        new_entry = {inst_id: inst_name}
        instrument_dict.update(new_entry)
    return instrument_dict


def get_inst_pointings(pointings):
    """
    deprecated
    """
    MMT_pointings = []
    CSS_pointings = []
    GOTO_pointings = []
    print(np.asarray(pointings))
    for p in range(len(pointings)):
        if pointings[p][0] == 22:
            MMT_pointings.append(pointings[p])
        if pointings[p][0] == 71:
            GOTO_pointings.append(pointings[p])
        if pointings[p][0] == 11:
            CSS_pointings.append(pointings[p])
    MMT_pointings = np.array(MMT_pointings)[:,1:3]
    print(MMT_pointings)
    CSS_pointings = np.array(CSS_pointings)[:,1:3]
    GOTO_pointings = np.array(GOTO_pointings)[:,1:3]
    return MMT_pointings, CSS_pointings, GOTO_pointings


def get_important_things(instrument):
    # for a given instrument, returns the median time of the pointing
    # since the event as well as the band
    time, band, depth_list = get_time_and_band(instrument)
    event = '2019-04-25T08:18:05'
    FMT = '%Y-%m-%dT%H:%M:%S'
    time = [datetime.strptime(t[:19], FMT) - datetime.strptime(event, FMT) for t in time]
    time = [t.days + t.seconds/86400 for t in time]
    return time, band, depth_list


def deg2rad(point):
    return np.deg2rad(point[0]), np.deg2rad(point[1])


def astro2sky(point):
    colat = np.pi/2 - point[1]
    if point[0] < 0:
        colong = 2*np.pi + point[0]
    else:
        colong = point[0]
    return [colat, colong]


def sky2vec(point):
    return hp.ang2vec(point[0], point[1])


def point_projector(point):
    point = deg2rad(point)
    point = astro2sky(point)
    point = sky2vec(point)
    return point


def query_footprint(footprint, pointing):
    polygon = []
    for p in footprint:
        point = np.array(p) + np.array(pointing)
        point = point_projector(point)
        polygon.append(point)
    return hp.query_polygon(256, polygon)


def coverage_in_time(ti, tf, instrument='css'):
    """
    Returns total probability coverage during time period
    ti: initial time in days
    tf: final time in days
    instrument: either goto or css, the instrument dictionary
    """
    if instrument == 'css':
        footprint = footprints_dict.get(11)[:-1]
        instrument = css
    elif instrument == 'goto':
        footprint = footprints_dict.get(71)[:-1]
        instrument = goto
    pointings = instrument['pointings']
    times = instrument['time']
    depths = instrument['depth']
    indices = []
    for i in range(len(times)):
        if ti <= times[i] <= tf:
            indices.append(i)
    pointings = pointings[indices]
    idx = []
    index = 0
    point_n = []
    for pointing in pointings:
        new_indices = list(query_footprint(footprint, pointing))
        idx = idx + new_indices
        point_n = point_n + [index] * len(new_indices)
        index += 1
    ids = np.array([np.asarray(idx), np.asarray(point_n)]) 
    a, x = len(ids[0]), 0
    while x < a:
        list_trunc = list(ids[0])[:x]
        if ids[0][x] in list_trunc:
            ids = np.delete(ids, x, 1)
        x += 1
        a = len(ids[0])
    #     idx = list(dict.fromkeys(idx)) # kills duplicates
    pdist = hpx[ids[0].astype(np.int)]
    p = pdist.sum()
    if ids[0].size:
        ptgid = choices(ids[1], pdist)
        depth = depths[ptgid]
    else:
        depth = 0.0
    return p, idx, depth



def coverage_iterator(time, instrument):
    """
    Given a list describing the number of time windows,
    the beginning of the first window and the end of the last,
    iterates through an instrument's sky coverage to determine whether
    a detection was made, and if so, in which epoch.
    - time: time list [dt, ti, tf]
    - instrument: instrument string
    """
    global hpx
    hpx, header = hp.read_map('LALInference.fits.gz,0', h=True, verbose=False)
    ti = time[1]
    tf = time[2]
    nt = time[0]
    dt = (tf - ti) / nt
    t2 = ti + dt
    for i in range(int(nt)):
        ti += dt * i
        t2 += dt * i
        p, idx, depth = coverage_in_time(ti, t2, instrument)
        r = np.random.uniform()
        if r <= p:
            return [ti, t2], i, depth
        else:
            hpx[idx] = 0
            hpx /= hpx.sum()
    return None


def convert_flux(flux, wavelength):
    # adjust flux for viewing distance
    dpc = 157.965e6
    flux *= (10/dpc)**2
    # c = 299792458e7 # speed of light in angstroms per second
    flux_density = 3.34e4 * flux * (wavelength ** 2)
    return flux_density


def main(time, instrument, bns):
    global pointings, footprints_dict, instrument_dict, mmt, css, goto
    pointings = get_pointings()
    footprints_dict = get_footprints()
    instrument_dict = get_instruments()
    pointings = np.delete(pointings, 1, 1)
    # get the ids for each pointing class
    id_set = set(list(pointings[:, 0]))
    # instrument_dict gives the name associated with each instrument ID
    # MMT_pointings, CSS_pointings, GOTO_pointings = get_inst_pointings(pointings) deprecated
    # mmt_time, mmt_band, mmt_depth = get_important_things('MMT') dep
    # goto_time, goto_band, goto_depth = get_important_things('GOTO-4') dep
    # css_time, css_band, css_depth = get_important_things('CSS') dep
    info_list = []
    for id_ in id_set:
        time, band, depth = get_important_things(instrument_dict[id_])
        info = {}
        info['pointings'] = pointings[pointings[:,0] == id_]
        info['time'] = np.array(time)
        info['band'] = np.array(band)
        info['depth'] = np.array(depth[0])
        info_list += info
    print(info_list)
    # mmt = {}
    # mmt['pointings'] = MMT_pointings
    # mmt['time'] = np.array(mmt_time)
    # mmt['band'] = np.array(mmt_band)
    # mmt['depth'] = np.array(mmt_depth[0])
    # css = {}
    # css['pointings'] = CSS_pointings
    # css['time'] = np.array(css_time)
    # css['band'] = np.array(css_band)
    # css['depth'] = np.array(css_depth[0])
    # goto = {}
    # goto['pointings'] = GOTO_pointings
    # goto['time'] = np.array(goto_time)
    # goto['band'] = np.array(goto_band)
    # goto['depth'] = np.array(goto_depth[0])
    detection = coverage_iterator(time, instrument)
    pos_angles = []
    if detection is None:
        print("No detection made")
        return [0]
    else:
        [t1, t2], i, depth = detection
        # convert depth to correct flux density
        if instrument == 'goto':
            r1, r2 = 3360, 5420 # units are angstroms
        [t1, t2], i, depth = detection
        wavecollection = [bns['oa0.1'], bns['oa0.2'], bns['oa0.3'], 
            bns['oa0.4'], bns['oa0.5'], bns['oa0.6'], bns['oa0.7'],
            bns['oa0.8'], bns['oa0.9'], bns['oa1.0']]
        angle = 0.1
        for wave in wavecollection:
            flux = 0
            for waveform in wave:
                if r1 < waveform[0] < r2:
                    flux += convert_flux(waveform[i], waveform[0])
            mAB = 8.9 - 2.5 * np.log10(flux)
            print(mAB, depth)
            if mAB < depth:
                pos_angles.append(angle)
            angle += 0.1
        return pos_angles

        


if __name__ == "__main__":
    bns = get_bns()
    time = bns['time']
    angles = []
    for i in range(100):
        a = main([int(time[0]), int(time[1]), int(time[2])], 'goto', bns)
        if a != [0]:
            angles = angles + a
    np.savetxt('angles_histogram', np.asarray(angles))