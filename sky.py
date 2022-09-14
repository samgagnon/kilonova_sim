from this import d
import healpy as hp
import numpy as np
import requests
import urllib.parse
import os, sys, json
from datetime import datetime
from numpy.random import choice

from dustmaps.config import config
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

hpx, reader = hp.read_map('GW190814_skymap.fits.gz', h=True, nest=False)

# hp.mollview(hpx)
# plt.show()

# as listed in treasuremap, wrong for some reason
# footprint190814 = [[0.43413, -0.4737], [0.43413, 0.23685], \
#                 [-0.3256, 0.23685], [-0.3256, -0.4737]]
# my footprint for DECAM, which is correct.
footprint190814 = [[0.6512, -0.4737], [0.6512, 0.4737], \
                [-0.54267, 0.4737], [-0.54267, -0.4737]]


def random_ipix(n=1):
    npix = len(hpx)
    ipix = choice(range(npix), size=n, p=hpx)
    if n == 1:
        return ipix[0]
    else:
        return ipix

def ebv_from_loc(loc):
    npix = len(hpx)
    nside = hp.npix2nside(npix)

    theta, phi = hp.pix2ang(nside, loc)

    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5 * np.pi - theta)

    # print("RA, DEC: ", ra, dec)

    config['data_dir'] = '.'

    coords = SkyCoord(phi, theta, unit='deg', frame='icrs')
    sfd = SFDQuery()
    ebv = sfd(coords)
    return ebv

BASE = 'http://treasuremap.space/api/v0/'
api_token = "zd6DGajyV66nlxuGhWt4CzIbdJ3wwr2UzlbkEg"
TARGET = 'pointings'

payload = {'graceid':'S190814bv', 'api_token':api_token}
r = requests.get(url = BASE+TARGET, params = payload)
print('r', r)
results = r.json()

print('results', results)

# event = '2019-07-14T21:10:39'
event = '2019-08-14T21:10:39'

pointings = []
for tile in results:
    tile = eval(tile)
    position = tile["position"]
    ra = float(position.split()[1].split('(')[1])
    dec = float(position.split()[2].split(')')[0])
    pos_angle = tile['pos_angle']
    instrument_id = tile['instrumentid']
#     print(instrument_id)
    band = tile['band']
    depth = tile['depth']
    time = tile['time']
    FMT = '%Y-%m-%dT%H:%M:%S'
    time = datetime.strptime(time[:19], FMT) - datetime.strptime(event, FMT)
    time = time.days + time.seconds/86400
    if instrument_id == 52:
        pointings.append([ra, dec, pos_angle, band, time])

TARGET='footprints'
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

def deg2rad(point):
    return np.deg2rad(point[0]), np.deg2rad(point[1])

def astro2sky(point):
    colat = np.pi/2 - point[1]
    if ra < 0:
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
    return hp.query_polygon(1024, polygon) # 256 for GW190425 1064 for GW190814

id_list = []
for pointing in pointings:
    id_list += [query_footprint(footprint190814, [pointing[0], pointing[1]])]

def get_pointing_ids(loc_id):
    #275221
    a = 0
    good_ids = []
    for ids in id_list:
        b = ids==loc_id
        if True in b:
            good_ids += [a]
        a += 1
    return good_ids

def time_from_loc(loc):
    ids = get_pointing_ids(loc)
    time = []
    for idi in ids:
        time += [pointings[idi][4]]
    return time

def band_from_loc(loc):
    ids = get_pointing_ids(loc)
    band = []
    for idi in ids:
        band += [pointings[idi][3]]
    return band

# hp.mollview(hpx)
# plt.show()

# locl = [random_ipix() for i in range(10)]
# timel = [time_from_loc(loc) for loc in locl]
# print(timel)
# print(loc_list)
# go = 0
# for loc in loc_list:
#     for id_ in id_list:
#         if loc in id_:
#             go += 1
#             print(go)
# ebv = [ebv_from_loc(loc) for loc in loc_list]
# # print(loc_list)
# # print(ebv)
# print(hp.npix2nside(len(hpx)))
# # print(np.sum(hpx[id_list]))
# print(np.argmax(hpx))

# prob = 0
# jpx = np.copy(hpx)
# id_coverage = []
# for id_ in id_list:
#     jpx[id_]+=1
#     id_coverage += list(id_)
#     # prob+=np.sum(hpx[id_])
# # print(prob, np.sum(hpx))
# id_coverage = list(dict.fromkeys(id_coverage))
# print(np.sum(hpx[id_coverage]))
# # hp.mollview(jpx)
# # plt.show()

# for pointing in pointings:
#     print(pointing[3])