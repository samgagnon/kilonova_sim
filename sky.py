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

hpx, reader = hp.read_map('LALInference.fits.gz,0', h=True)

def random_ipix():
    npix = len(hpx)
    ipix = choice(range(npix), 1, p=hpx)
    return ipix[0]

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

payload = {'graceid':'S190425z', 'api_token':api_token}
r = requests.get(url = BASE+TARGET, params = payload)
print(r)
results = r.json()

print(results)

event = '2019-04-25T08:18:05'

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
#     print(time)
    FMT = '%Y-%m-%dT%H:%M:%S'
    time = datetime.strptime(time[:19], FMT) - datetime.strptime(event, FMT)
    time = time.days + time.seconds/86400
    if instrument_id == 71:
        if band == 'g':
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
    return hp.query_polygon(256, polygon)

id_list = []
for pointing in pointings:
    id_list += [query_footprint(footprints_dict[71][:-1], [pointing[0], pointing[1]])]

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

# print(get_pointing_ids(ipix[0]))
# loc_list = [random_ipix() for i in range(100)]
# ebv = [ebv_from_loc(loc) for loc in loc_list]
# print(loc_list)
# print(ebv)
