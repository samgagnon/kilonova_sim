import mosfit

import astropy.units as u

from scipy.stats import truncnorm
from astropy.constants import c
from astropy.cosmology import z_at_value
from scipy.special import erfc # note necessary afaik

from gwtoolbox import tools_earth
from gwtoolbox.sources_kHz import DNS

from events import *
from config import *
from distance import *

# instantiate pmesh dictionary
pdict = gen_pDV_dists(event_list, plot=False)

max = 0
for event in event_list:
    pdist = pdict[str(event)]
    max += np.log10(pdist[0].max()/pdist[0].sum())

print(max)