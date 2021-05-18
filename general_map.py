import numpy as np
import healpy as hp
import requests
import urllib.parse
import os, sys, json
from random import choices
from datetime import datetime
from bns import *


def get_time_and_band(api_token, graceid, instrument='MMT'):
    """
    Returns times, bands, and depths related to an instrument.
    """
    BASE = 'http://treasuremap.space/api/v0'
    # api_token = "zd6DGajyV66nlxuGhWt4CzIbdJ3wwr2UzlbkEg"
    
    TARGET = 'pointings'

    #define the filtering parameters 
    #grab all of the completed pointings taken with a specific instrument for a given event

    # graceid = 'S190425z'

    params = {
        "api_token":api_token,
        "instrument":instrument,
        "graceid":graceid,
        "status":"completed"
    }

    url = "{}/{}?{}".format(BASE, TARGET, urllib.parse.urlencode(params))
    r = requests.get(url = url)
    # print("There are %s pointings" % len(json.loads(r.text)))

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


def get_pointings(api_token, graceid):
    """
    Gets all EM pointings for a particular event
    """
    BASE = 'http://treasuremap.space/api/v0/'
    TARGET = 'pointings'
    # api_token = "zd6DGajyV66nlxuGhWt4CzIbdJ3wwr2UzlbkEg"

    payload = {'graceid':graceid, 'api_token':api_token}
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


def get_footprints(api_token, graceid):
    """
    Gets the footprints for each instrumen
    """
    BASE = 'http://treasuremap.space/api/v0/'
    TARGET='footprints'
    # api_token = "zd6DGajyV66nlxuGhWt4CzIbdJ3wwr2UzlbkEg"

    payload = {'graceid':graceid, 'api_token':api_token}
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


def get_instruments(api_token, graceid):
    BASE = 'http://treasuremap.space/api/v0/'
    TARGET = 'instruments'
    # api_token = "zd6DGajyV66nlxuGhWt4CzIbdJ3wwr2UzlbkEg"

    payload = {'graceid':graceid, 'api_token':api_token}
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
    for p in range(len(pointings)):
        if pointings[p][0] == 22:
            MMT_pointings.append(pointings[p])
        if pointings[p][0] == 71:
            GOTO_pointings.append(pointings[p])
        if pointings[p][0] == 11:
            CSS_pointings.append(pointings[p])
    MMT_pointings = np.array(MMT_pointings)[:,1:3]
    CSS_pointings = np.array(CSS_pointings)[:,1:3]
    GOTO_pointings = np.array(GOTO_pointings)[:,1:3]
    return MMT_pointings, CSS_pointings, GOTO_pointings


def get_important_things(api_token, graceid, instrument):
    # for a given instrument, returns the median time of the pointing
    # since the event as well as the band
    time, band, depth_list = get_time_and_band(api_token, graceid, instrument)
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


def coverage_in_time(ti, tf, info_list, event_loc):
    """
    Returns total probability coverage during time period
    ti: initial time in days
    tf: final time in days
    info_list: the information list
    """
    results = []
    for info in info_list:
        footprint = footprints_dict.get(info['ID'])[:-1]
        pointings = info['pointings']
        print(pointings)
        times = info['time']
        depths = info['depth']
        band = info['band']
        indices = []
        for i in range(len(times)):
            if ti <= times[i] <= tf:
                indices.append(i)
        # print(tf, indices)
        if indices == []:
            # if there are no pointings for this instrument,
            # try next instrument
            continue
        pointings = pointings[indices]
        idx = []
        index = 0
        point_n = []
        print(pointings)
        for pointing in pointings:
            # loops through pointings and appends indices covered by
            # the instrument footprint
            # print(pointing)
            new_indices = list(query_footprint(footprint, pointing))
            # dpx = np.log10(hpx)
            # dpx[new_indices] *= 5e5
            # hp.visufunc.mollview(dpx)
            # plt.show()
            if event_loc in new_indices:
                print(True)
                depth = depths[event_loc]
                band = bands[event_loc]
                inst_name = info["inst_name"]
                results += [inst_name, band, depth]
    return results



def coverage_iterator(hpx, time, info_list, event_loc):
    """
    Given a list describing the number of time windows,
    the beginning of the first window and the end of the last,
    iterates through an instrument's sky coverage to determine whether
    a detection was made, and if so, in which epoch.
    - time: time list [dt, ti, tf]
    - instrument: instrument string
    """
    ti = time[1]
    tf = time[2]
    nt = time[0]
    dt = (tf - ti) / nt
    t2 = ti + dt
    detections = []
    for i in range(int(nt)):
        ti += dt * i
        t2 += dt * i
        # print("Calling coverage in time", i)
        results = coverage_in_time(ti, t2, info_list, event_loc)
        if results == []:
            continue
        else:
            for result in result:
                result += [ti, t2, i,]
            detections += results
    return detections


def sim_event(api_token, graceid, mapfile, time):
    """
    Runs multiple simulations for a particular event 
    """
    global pointings, footprints_dict, instrument_dict, mmt, css, goto
    global hpx
    # load in GW localization
    hpx = hp.read_map(mapfile, verbose=False)
    print(hp.pixelfunc.get_nside(hpx))
    hpx = hp.pixelfunc.ud_grade(hpx, 256)
    # hp.visufunc.cartview(hpx)
    # plt.show()
    # print(hpx)
    event_loc = choices(list(range(len(hpx))), hpx)
    print(event_loc, hpx[event_loc])
    print(hp.pixelfunc.pix2ang(256, event_loc, lonlat=True))
    pointings = get_pointings(api_token, graceid)
    footprints_dict = get_footprints(api_token, graceid)
    # instrument_dict gives the name associated with each instrument ID
    instrument_dict = get_instruments(api_token, graceid)
    pointings = np.delete(pointings, 1, 1)
    # get the ids for each pointing class
    print(instrument_dict)
    id_set = set(list(pointings[:, 0]))
    legal_inst = [38, 22, 23, 24, 25, 26, 11, 65, 52, 47, 70, 71, 12]
    # I removed the Swift Burst Alert Telescope due to a polygon issue with 
    # its footprint
    info_list = []
    for id_ in id_set:
        if id_ not in legal_inst:
            continue
        # construct list of dictionaries with info for each instrument
        ob_time, band, depth = get_important_things(api_token, \
                               graceid, instrument_dict[id_])
        info = {}
        info['ID'] = id_
        info['inst_name'] = instrument_dict[id_]
        info['pointings'] = pointings[pointings[:,0] == id_][:,1:]
        info['time'] = np.array(ob_time)
        info['band'] = np.array(band)
        info['depth'] = np.array(depth[0])
        info_list += [info]
    # given a time window and the pointing info, determine if a detection is made
    detections = coverage_iterator(hpx, time, info_list, event_loc)
    return detections

        


if __name__ == "__main__":
    bns = get_bns()
    time = bns['time']
    api_token = "zd6DGajyV66nlxuGhWt4CzIbdJ3wwr2UzlbkEg"
    graceid = ['S190425z', 'S200224ca']
    mapfile = ['LALInference.fits.gz,0', 'S200224ca.fits']
    e1 = sim_event(api_token, graceid[0], mapfile[0], [int(time[0]), \
                      int(time[1]), int(time[2])])
    e2 = sim_event(api_token, graceid[1], mapfile[1], [int(time[0]), \
                      int(time[1]), int(time[2])])
    print("S190425z:", e1)
    print("S200224ca:", e2)