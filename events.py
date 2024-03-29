import astropy.cosmology as cosmo

# how many observations?
o1 = False
o5 = False
o100 = False
o100o = False
o100m = False
m100 = False
m100m = False # malmquist bias
e100 = False
es100 = False
es25100 = False # GW190425
esg25100 = True # GW190814
esg23100 = False # GW190814
em100 = False # malmquist bias different masses
e10 = False
one = False

# are non-observations included?
n = False
# is malmquist bias considered?
m = True
# is em isotropy assumed?
monly = False

# assume a true value for the Hubble constant
u_true = cosmo.FlatLambdaCDM(70, 0.3)
# set the true redshift of the events
ZT1 = 0.031
ZT2 = 2.49368266e-2
ZT3 = 1.69988646e-02
ZT4 = 4.22995849e-02
ZT5 = 1.54850565e-02
ZT6 = 2.35409134e-02
ZT7 = 6.60280705e-02
ZT8 = 6.11473418e-02
ZT9 = 4.03183165e-02
ZT10 = 3.08064095e-02
ZT11 = 8.85648358e-03
# calculate the true distance of the events
DT1 = u_true.luminosity_distance(ZT1).value
DT2 = u_true.luminosity_distance(ZT2).value
DT3 = u_true.luminosity_distance(ZT3).value
DT4 = u_true.luminosity_distance(ZT4).value
DT5 = u_true.luminosity_distance(ZT5).value
DT6 = u_true.luminosity_distance(ZT6).value
DT7 = u_true.luminosity_distance(ZT7).value
DT8 = u_true.luminosity_distance(ZT8).value
DT9 = u_true.luminosity_distance(ZT9).value
DT10 = u_true.luminosity_distance(ZT10).value
DT11 = u_true.luminosity_distance(ZT11).value

# true masses for each neutron star
m1 = (1.4, 1.4)
m2 = (1.19124606, 1.68521165)
m3 = (1.82530210, 1.36362431)
m4 = (2.05529767e+00, 1.25570351e+00)
m5 = (1.75293951e+00, 1.29532747e+00)
m6 = (1.70298094e+00, 1.61503774e+00)
m7 = (1.39189677e+00, 2.44430152e+00)
m8 = (1.47323485e+00, 1.36881131e+00)
m9 = (1.79676445e+00, 1.41040578e+00)
m10 = (1.29790736e+00, 1.64149555e+00)
m11 = (1.75336110e+00,  1.34787474e+00)

# mass uncertainties
dm1 = (1e-1, 1e-1)
dm2 = (5.95623030e-02, 8.42605827e-02)
dm3 = (9.12651050e-02, 6.81812157e-02)
dm4 = (1.02764883e-01, 6.27851754e-02)
dm5 = (8.76469757e-02, 6.47663735e-02)
dm6 = (8.51490469e-02, 8.07518870e-02)
dm7 = (6.95948385e-02, 1.22215076e-01)
dm8 = (7.36617423e-02, 6.84405653e-02)
dm9 = (8.98382225e-02, 7.05202889e-02)
dm10 = (6.48953678e-02, 8.20747775e-02)
dm11 = (8.76680552e-02,  6.73937371e-02)

# cosine of observation angles
v2, v3, v4, v5, v6, v7, v8, v9, v10, v11 = 0.13039641, 0.63846131, 0.9070467, 0.34523033, 0.8636566, \
       0.61501855, 0.5952604 , 0.41010708, 0.10329073, 0.46383165

v100 = [4.77906956e-05, 2.68475939e-04, 1.47212644e-03, 4.52025802e-03,\
       6.51430168e-03, 2.03418616e-02, 2.52239406e-02, 3.07185673e-02,\
       3.32863463e-02, 3.39670384e-02, 3.94116842e-02, 4.60509004e-02,\
       5.34217880e-02, 5.99894314e-02, 6.25462253e-02, 6.96281868e-02,\
       7.27006583e-02, 8.19359825e-02, 9.25879395e-02, 9.82132682e-02,\
       1.01924744e-01, 1.06709737e-01, 1.11540999e-01, 1.22692806e-01,\
       1.28521450e-01, 1.29279428e-01, 1.35187764e-01, 1.37893039e-01,\
       1.66170873e-01, 1.74022986e-01, 1.75345235e-01, 1.99686613e-01,\
       2.00398401e-01, 2.04280203e-01, 2.06207961e-01, 2.09157200e-01,\
       2.09576426e-01, 2.30857056e-01, 2.58518653e-01, 2.59074762e-01,\
       2.76521974e-01, 2.77147958e-01, 2.82890493e-01, 2.83578238e-01,\
       2.94354762e-01, 3.00404442e-01, 3.16698616e-01, 3.47391796e-01,\
       3.72540661e-01, 3.87268100e-01, 4.00112491e-01, 4.17171731e-01,\
       4.19240317e-01, 4.19839283e-01, 4.26947246e-01, 4.27156848e-01,\
       4.31519403e-01, 4.77895401e-01, 4.81756345e-01, 4.84891688e-01,\
       5.15651495e-01, 5.20361984e-01, 5.36113819e-01, 5.55492821e-01,\
       5.56677797e-01, 5.59475514e-01, 5.74814962e-01, 5.81560078e-01,\
       5.94295501e-01, 5.98249842e-01, 6.06021896e-01, 6.21729552e-01,\
       6.25455040e-01, 6.25513414e-01, 6.39663703e-01, 6.42805563e-01,\
       6.49028011e-01, 6.54673548e-01, 6.57032171e-01, 6.57676870e-01,\
       6.61109085e-01, 6.61322693e-01, 7.04327437e-01, 7.07437600e-01,\
       7.92480293e-01, 8.01381934e-01, 8.24436151e-01, 8.47735938e-01,\
       8.60156868e-01, 8.72748653e-01, 8.90845647e-01, 8.96374606e-01,\
       9.24408414e-01, 9.42610483e-01, 9.51916090e-01, 9.52637215e-01,\
       9.67009787e-01, 9.78018016e-01, 9.80708483e-01, 9.96934646e-01]

ob100 = [False, False, False, False, False, False, False, False, False, False, False,\
     False, False, False, False, False, False, False, False, False, False, False, False,\
    False, False, False, False, False, False, False, False, False, False, False,\
    False, False, False, False, False, False, False, False, False, False, False, False, \
    False, False, False, False, False, False, False, False, False, False, False, False, False, \
    False, False, False, False, False, False, False, False, False, False, False, False, False, \
    False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, \
    True, True, True, True, True, True, True, True, True, True, True, True]

mz100 = [0.0293858 , 0.02172356, 0.02616004, 0.07002505, 0.00257538,\
       0.08013533, 0.05570062, 0.04437526, 0.06610401, 0.00610015,\
       0.02840253, 0.0219356 , 0.06422882, 0.02936137, 0.02514904,\
       0.05782984, 0.06812726, 0.03413788, 0.02488582, 0.0515162,\
       0.06521696, 0.05516568, 0.02615707, 0.04380263, 0.02985176,\
       0.03695501, 0.04418034, 0.02806767, 0.03854945, 0.03281109,\
       0.03188937, 0.09912395, 0.02301441, 0.06541106, 0.04993108,\
       0.01025563, 0.05207773, 0.0239487 , 0.03077531, 0.06829539,\
       0.04546793, 0.02296364, 0.04990143, 0.01254792, 0.03764861,\
       0.03008149, 0.06724979, 0.01034294, 0.02749771, 0.06936323,\
       0.07560435, 0.03528306, 0.01849835, 0.0215221 , 0.04281204,\
       0.05829047, 0.03113406, 0.01899822, 0.05258527, 0.02818597,\
       0.02631603, 0.04206268, 0.04126473, 0.05036618, 0.04087791,\
       0.04303329, 0.06363399, 0.04758812, 0.02620117, 0.06297139,\
       0.03221821, 0.09633496, 0.05528829, 0.04823364, 0.07057507,\
       0.04703309, 0.02846339, 0.07756464, 0.06417388, 0.05103614,\
       0.05338664, 0.06502364, 0.03303712, 0.05099171, 0.0622192,\
       0.03706093, 0.06242086, 0.02902463, 0.02709266, 0.02137966,\
       0.0208635 , 0.01888762, 0.01004494, 0.07313856, 0.037274,\
       0.0292486 , 0.05631432, 0.0163161]

md100 = [128.68810991,  94.59069868, 114.28710353, 315.77191459,\
        11.05169668, 363.91034097, 248.64625276, 196.48101631,\
       297.27127484,  26.24866257, 124.29132596,  95.52919161,\
       288.45713202, 128.57881825, 109.78745036, 258.54336498,\
       306.80576875, 150.02506983, 108.61700755, 229.27871072,\
       293.09908544, 246.16419306, 114.2738615 , 193.86491976,\
       130.7738684, 162.74231228, 195.59028469, 122.79540618,\
       169.96242054, 144.05318132, 139.91115391, 455.99538149,\
       100.3085151, 294.01162833, 221.97066347,  44.27008102,\
       231.87136387, 104.45364464, 134.91205917, 307.59919393,\
       201.47872021, 100.08342816, 221.83412976,  54.2598241,\
       165.88114909, 131.80273826, 302.66760658,  44.64995574,\
       120.25082957, 312.64264027, 342.25958334, 155.1885863,\
        80.35200801,  93.69932666, 189.34423488, 260.68823163,\
       136.52102511,  82.55444689, 234.21645668, 123.32378834,\
       114.98197388, 185.92856988, 182.29533367, 223.97506539,\
       180.5355419 , 190.35340276, 285.66572791, 211.19763616,\
       114.47030399, 282.55888986, 141.38825686, 442.33471351,\
       246.73294728, 214.16230035, 318.37467655, 208.65063624,\
       124.56327479, 351.61126564, 288.19921895, 227.06378446,\
       237.92248645, 292.19045277, 145.06976339, 226.85884846,\
       279.03522973, 163.22143787, 279.97956517, 127.07239391,\
       118.4437801,  93.06926051,  90.78711199,  82.06700988,\
        43.35363238, 330.52974054, 164.1854731 , 128.07426109,\
       251.49597308,  70.75620238]

mv100 = [0.31389834, 0.17561909, 0.63693177, 0.91257378, 0.40155497,\
       0.92806967, 0.89898251, 0.85018345, 0.87379803, 0.95748116,\
       0.91301275, 0.96698114, 0.97641994, 0.59728696, 0.41614088,\
       0.76068992, 0.97751109, 0.93026727, 0.90863421, 0.80718821,\
       0.98626076, 0.93107427, 0.7061191, 0.4826587 , 0.26285508,\
       0.94154624, 0.97192396, 0.53963649, 0.73944896, 0.74845313,\
       0.36747834, 0.93291722, 0.70583983, 0.88745394, 0.8853057,\
       0.29353942, 0.70026241, 0.25720693, 0.64536569, 0.93816323,\
       0.72701977, 0.9648573 , 0.97340886, 0.88353418, 0.83308296,\
       0.59065243, 0.94042811, 0.94235443, 0.67471421, 0.97426814,\
       0.90403291, 0.72472942, 0.7637999 , 0.74235089, 0.94054623,\
       0.5948894 , 0.86690974, 0.88045721, 0.63177281, 0.93544336,\
       0.55911585, 0.62530218, 0.58796751, 0.87568253, 0.77572749,\
       0.86111507, 0.73704382, 0.86243067, 0.46556752, 0.76659221,\
       0.56413601, 0.85394033, 0.87936983, 0.88614796, 0.88763289,\
       0.57271049, 0.48722432, 0.940598, 0.73843183, 0.72851537,\
       0.81467583, 0.98942429, 0.71584448, 0.85663254, 0.98785906,\
       0.80383855, 0.90442023, 0.62479538, 0.45122242, 0.41406026,\
       0.60415017, 0.24200349, 0.26356791, 0.82213979, 0.60406624,\
       0.62631485, 0.87563025, 0.09010542]

mo100old = [True, True, True, False, True, False, False, True, False, \
    True, True, True, False, True, True, False, False, True, True, \
    False, False, False, True, False, True, True, True, True, True, \
    True, True, False, True, False, True, True, False, True, True, \
    False, True, True, True, True, True, True, False, True, True, False, \
    False, True, True, True, True, False, True, True, False, True, True, \
    True, True, False, True, True, False, True, True, False, True, False, \
    False, True, False, False, True, False, False, False, False, False, True, \
    False, False, True, False, True, True, True, True, True, True, False, \
    True, True, False, True]

mo100 = [True, True, True, False, True, False, False, False, False,\
    True, True, True, False, True, True, False, False, False, True, \
    False, False, False, True, False, True, False, False, True, False, \
    True, True, False, True, False, False, True, False, True, True, False, \
    False, True, False, True, False, True, False, True, True, False, False, \
    False, True, True, False, False, True, True, False, True, True, False, \
    False, False, False, False, False, False, True, False, True, False, False, \
    False, False, False, True, False, False, False, False, False, False, False, \
    False, False, False, True, True, True, True, True, True, False, False, True, \
    False, True]

ez100 = [0.02744049, 0.02457503, 0.07981769, 0.0675472 , 0.03378058,\
       0.02152214, 0.0862362 , 0.09026349, 0.03245951, 0.02602468,\
       0.07116436, 0.03663973, 0.03214562, 0.02131412, 0.03100677,\
       0.12455784, 0.0572627 , 0.05479248, 0.05878143, 0.02463472,\
       0.05195538, 0.03604569, 0.09859508, 0.0465312 , 0.05622843,\
       0.05012148, 0.06297815, 0.07793525, 0.08692068, 0.06101318,\
       0.068774  , 0.04093243, 0.03082357, 0.04118058, 0.06410164,\
       0.05221065, 0.07367782, 0.1158446 , 0.02277494, 0.02543437,\
       0.00900018, 0.04354585, 0.09013565, 0.04204599, 0.04598776,\
       0.03284299, 0.07910262, 0.05857474, 0.01010622, 0.04890589,\
       0.08556459, 0.0182169 , 0.09022002, 0.05966181, 0.02898755,\
       0.05078501, 0.0306114 , 0.03497173, 0.07755618, 0.05252812,\
       0.0709885 , 0.04746052, 0.07975907, 0.05226923, 0.04849469,\
       0.07191483, 0.12924101, 0.06359174, 0.02947247, 0.13995157,\
       0.0986103 , 0.06029928, 0.04447743, 0.03323659, 0.05434456,\
       0.03662264, 0.03242957, 0.05028558, 0.03061894, 0.04754943,\
       0.07373878, 0.03930278, 0.03115537, 0.02846339, 0.06057552,\
       0.02775782, 0.02352933, 0.06089618, 0.04792738, 0.0358918,\
       0.0340004 , 0.10517011, 0.13069704, 0.05649093, 0.03310187,\
       0.05471018, 0.00905742, 0.07345756, 0.02231454]

ed100 = [119.99551605, 107.23561303, 362.38847791, 304.06968676,\
       148.41572459,  93.69951181, 393.25972011, 412.75694189,\
       142.47262255, 113.68426628, 321.16526528, 161.31653357,\
       141.06214827,  92.7793737 , 135.95001248, 582.68912903,\
       255.90441587, 244.43362456, 262.97574077, 107.50085894,\
       231.30632011, 158.63183723, 453.40136513, 206.34916519,\
       251.09701267, 222.84766253, 282.59056142, 353.3819034,\
       396.56659671, 273.39295009, 309.85883378, 180.78353119,\
       135.1284413 , 181.91243874, 287.86010714, 232.48537364,\
       333.09186839, 538.85950854,  99.2469349 , 111.05668212,\
        38.81349634, 192.69246357, 412.1365266 , 185.85255547,\
       203.85895799, 144.19663593, 358.96465898, 262.01254246,\
        43.62017864, 217.25252593, 390.01777714,  79.11267854,\
       412.54596819, 267.08144114, 126.90658273, 225.90565401,\
       134.17721277, 153.78396623, 351.57086303, 233.95230663,\
       320.33225487, 210.61191023, 362.10769969, 232.75602233,\
       215.36199609, 324.72224145, 606.42808442, 285.46751723,\
       129.07596825, 661.19230824, 453.47598933, 270.05728551,\
       196.94801241, 145.96711828, 242.3577379 , 161.23926023,\
       142.33804742, 223.60370271, 134.21101164, 211.02000618,\
       333.38158502, 173.3793742 , 136.61660475, 124.56327479,\
       271.34766338, 121.41180937, 102.59233971, 272.84608136,\
       212.75539906, 157.93674919, 149.40573882, 485.76859955,\
       613.83436262, 252.31653913, 145.36099304, 244.05214777,\
        39.06203163, 332.04517352,  97.20712526]

em1100 = [1.78804734, 1.37276329, 1.75862666, 1.70197778, 1.42951244,\
       1.42937209, 1.56659591, 1.65570321, 2.3042711, 2.03558617,\
       2.04300576, 1.42699956, 2.17233682, 1.1552981, 1.22013732,\
       2.36432929, 1.7553702, 1.90374108, 2.23482057, 1.87495409,\
       1.68935489, 1.53309182, 1.53854945, 2.01399975, 1.567554,\
       1.55241195, 1.4321211, 1.88542874, 2.06580259, 1.3945316,\
       1.57080524, 1.61913384, 1.84849895, 1.73276194, 2.03428426,\
       1.69954329, 1.37342217, 2.48775257, 2.09457425, 1.47622775,\
       2.26902775, 1.77873378, 1.19921404, 1.94786734, 2.19874536,\
       1.60833019, 1.70084888, 1.16881708, 1.49579713, 1.28511594,\
       1.25508093, 1.1434775, 1.21707772, 1.22804718, 1.64778131,\
       1.62765348, 2.21312101, 1.48596972, 1.51877312, 1.34375998,\
       1.81732795, 1.97873293, 1.47601641, 1.76079046, 1.20979867,\
       2.00157808, 1.83038713, 1.47133269, 2.16568342, 1.77586528,\
       1.40976843, 1.84623083, 2.09151186, 1.6530635 , 2.08085307,\
       1.26118666, 2.25153739, 2.09560621, 1.16498118, 1.69507774,\
       1.99365648, 1.39064828, 2.4026408, 2.15552772, 1.94108805,\
       1.51101681, 1.75276617, 2.04871699, 1.62948473, 1.87758846,\
       1.35078669, 1.7585702, 2.02866198, 2.2892655, 1.67066087,\
       1.31436691, 1.73701528, 1.55927427, 1.41699667]

em2100 = [1.5085648, 1.46831125, 1.52325121, 1.35138806, 2.1014966,\
       2.46350786, 1.90042998, 1.45086005, 1.12107758, 2.14967273,\
       2.17299094, 1.6870993, 1.12221363, 1.33424106, 1.45066019,\
       1.37346111, 2.12064039, 1.9045975, 1.470304, 1.57816126,\
       1.47154772, 1.48624063, 1.98571823, 1.90123735, 1.51148357,\
       1.5453361, 1.52689579, 1.11134608, 1.4635485, 1.64132335,\
       1.50785612, 1.86660143, 1.52296786, 1.77563628, 1.1282877,\
       1.34860363, 1.43327638, 1.49670004, 1.612129, 2.472681,\
       1.28120848, 1.49492251, 1.64442907, 1.76052288, 1.47440702,\
       1.6718918, 1.48392039, 1.93649245, 1.74539151, 1.76378542,\
       1.70618766, 1.43849545, 1.81354939, 1.79381084, 1.59115282,\
       1.64018639, 1.36585375, 1.63374008, 1.27399672, 1.25671264,\
       1.43280587, 1.50348524, 2.13173017, 1.69822857, 1.69822848,\
       2.40731651, 1.61163931, 2.18650153, 1.62011471, 1.82047625,\
       2.25869867, 1.35080992, 1.47066964, 1.68079656, 1.10424269,\
       1.46762204, 1.95212369, 1.90301196, 2.27848906, 1.61290314,\
       2.08740708, 2.20775645, 1.40367725, 1.62078297, 1.14216789,\
       2.26079199, 2.07256846, 1.44458391, 1.19682359, 1.14548122,\
       1.11234904, 1.52519578, 1.79005848, 1.17064028, 1.3735233,\
       1.6891996, 1.8768255 , 1.51576643, 1.66938486]

edm1100 = [0.08940237, 0.06863816, 0.08793133, 0.08509889, 0.07147562,\
       0.0714686, 0.0783298, 0.08278516, 0.11521355, 0.10177931,\
       0.11240857, 0.07134998, 0.10861684, 0.05776491, 0.06100687,\
       0.11821646, 0.08776851, 0.95187054, 0.11174103, 0.0937477,\
       0.08446774, 0.08361238, 0.07692747, 0.10069999, 0.10286397,\
       0.77620597, 0.07160606, 0.09427144, 0.10329013, 0.06972658,\
       0.11487962, 0.08095669, 0.09242495, 0.15709837, 0.10171421,\
       0.08497716, 0.09543206, 0.12438763, 0.10472871, 0.07381139,\
       0.11345139, 0.08893669, 0.0599607 , 0.09739337, 0.10993727,\
       0.08041651, 0.08504244, 0.05844085, 0.07478986, 0.0642558 ,\
       0.06275405, 0.05717387, 0.06085389, 0.06140236, 0.08238907,\
       0.38003281, 0.11065605, 0.07429849, 0.07593866, 0.067188,\
       0.0908664, 0.09893665, 0.07380082, 0.10291038, 0.06048993,\
       0.1000789, 0.09151936, 0.07356663, 0.10828417, 0.46490363,\
       0.07048842, 0.09231154, 0.10457559, 0.15656675, 0.10404265,\
       0.06305933, 0.11257687, 0.10478031, 0.05824906, 0.08475389,\
       0.14378691, 0.06953241, 0.12013204, 0.10777639, 0.0970544,\
       0.07555084, 0.08763831, 0.10243585, 0.08147424, 0.09387942,\
       0.06753933, 0.08792851, 0.11337192, 0.11446327, 0.08353304,\
       0.06571835, 0.08685076, 0.1773818, 0.07084983]

edm2100 = [0.07542824, 0.07341556, 0.07616256, 0.0675694, 0.10507483,\
       0.12317539, 0.0950215, 0.072543, 0.05605388, 0.10748364,\
       0.12132243, 0.08435496, 0.05611068, 0.06671205, 0.07253301,\
       0.06867306, 0.10603202, 0.95229875, 0.0735152 , 0.07890806,\
       0.07357739, 0.08054595, 0.09928591, 0.09506187, 0.09820522,\
       0.77266805, 0.07634479, 0.0555673, 0.07317742, 0.08206617,\
       0.10899846, 0.09333007, 0.07614839, 0.16214966, 0.05641439,\
       0.06743018, 0.10089854, 0.074835, 0.08060645, 0.12363405,\
       0.06406042, 0.07474613, 0.08222145, 0.08802614, 0.07372035,\
       0.08359459, 0.07419602, 0.09682462, 0.08726958, 0.08818927,\
       0.08530938, 0.07192477, 0.09067747, 0.08969054, 0.07955764,\
       0.38343866, 0.06829269, 0.081687, 0.06369984, 0.06283563,\
       0.07164029, 0.07517426, 0.10658651, 0.0984088 , 0.08491142,\
       0.12036583, 0.08058197, 0.10932508, 0.08100574, 0.47945963,\
       0.11293493, 0.0675405, 0.07353348, 0.1599064 , 0.05521213,\
       0.0733811, 0.09760618, 0.0951506, 0.11392445, 0.08064516,\
       0.15233163, 0.11038782, 0.07018386, 0.08103915, 0.05710839,\
       0.1130396, 0.10362842, 0.0722292, 0.05984118, 0.05727406,\
       0.05561745, 0.07625979, 0.09709772, 0.05853201, 0.06867616,\
       0.08445998, 0.09384128, 0.17105377, 0.08346924]

ev100 = [0.1588633 , 0.22194531, 0.91346988, 0.85653405, 0.62906009,\
       0.4337486 , 0.78404308, 0.90638632, 0.98966223, 0.67541597,\
       0.64288604, 0.28957533, 0.27355106, 0.55686484, 0.58878085,\
       0.97888162, 0.96456746, 0.86055917, 0.5568404, 0.89949937,\
       0.94956283, 0.42260482, 0.93239676, 0.89964181, 0.81453726,\
       0.9225915, 0.81348604, 0.69436784, 0.87719617, 0.8208483,\
       0.82215903, 0.80420646, 0.82493961, 0.16023754, 0.80891926,\
       0.81658535, 0.80787066, 0.82056624, 0.6530038, 0.69427156,\
       0.76341884, 0.85364756, 0.96972242, 0.65181492, 0.24130105,\
       0.66219276, 0.93934772, 0.8005675, 0.43918537, 0.41872456,\
       0.92819202, 0.82379763, 0.89116001, 0.7295037 , 0.73793374,\
       0.92920523, 0.9072916 , 0.39370373, 0.98884122, 0.45605701,\
       0.93597618, 0.99482216, 0.90985728, 0.80082792, 0.66092213,\
       0.85874382, 0.92388974, 0.93895286, 0.72684629, 0.90311381,\
       0.95334409, 0.81091021, 0.6483218, 0.51376076, 0.69141727,\
       0.88825337, 0.63053597, 0.78972597, 0.52668327, 0.95849793,\
       0.95692547, 0.57739743, 0.38851356, 0.05706949, 0.5036213,\
       0.07007895, 0.09457709, 0.86787843, 0.98281888, 0.68856288,\
       0.59791779, 0.94709464, 0.97276485, 0.58134105, 0.39827457,\
       0.5267651, 0.6732626, 0.83047453, 0.86594258]

# eo100 = [True, True, False, False, True, True, False, False, True, True,\
#     False, False, True, True, True, False, False, False, False, True, \
#     False, True, False, False, False, False, False, False, False, False, \
#     False, True, True, False, False, False, False, False, True, True, \
#     True, True, False, False, False, True, False, False, True, False, \
#     False, True, False, False, True, False, True, True, False, False, \
#     False, True, False, False, False, False, False, False, True, False, \
#     False, False, True, True, False, True, True, False, True, True, False, \
#     True, True, True, False, True, True, False, True, True, True, False, \
#     False, True, True, False, True, False]

eso100 = [True, True, False, False, True, False, False, False, True, False, False, False, True,\
    False, False, False, False, False, False, False, False, False, False, False, False, False, False, \
    False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, \
    False, False, False, False, False, False, True, False, False, False, False, False, False, False, \
    False, False, False, False, False, False, False, False, False, False, False, False, False, False, \
    False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, \
    False, False, False, False, True, False, False, False, False, False, False, True, False, True]

eso25100 = [True, True, False, False, True, False, False, False, True, True, False, True, True, False, False, \
    False, False, False, True, False, False, False, False, False, False, True, False, False, False, True, False, \
    False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, \
    False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, \
    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, \
    False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, \
    False, False, True, True, True]

eo100 = [True, True, False, False, True, True, False, False, True, False, False, \
    False, True, True, False, False, False, False, False, True, False, False, False, \
    False, False, False, False, False, False, False, False, False, True, False, False, \
    False, False, False, True, True, True, False, False, False, False, False, False, \
    False, True, False, False, True, False, False, True, False, True, False, False, False, \
    False, False, False, False, False, False, False, False, True, False, False, False, False, \
    False, False, False, False, False, True, False, False, True, True, True, False, True, \
    True, False, False, False, False, False, False, False, False, False, True, False, True]

# esgo25100 = [True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, \
#     False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, \
#     False, True, False, False, False, False, False, False, True, False, False, False, False, False, True, False, \
#     False, False, False, False, True, False, False, True, False, True, False, False, False, True, False, False, False, \
#     False, False, False, False, False, False, True, False, True, False, True, False, False, False, False, False, False, \
#     True, True, False, False, False, False, False, False, False, True, False, False, True, True, False, False, False, True]

esgo23100 = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, \
    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, \
    False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, \
    False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, \
    False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, \
    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]

esgloc25100 = [8915091, 8874135, 8972412, 8882313, 9660663, 9615610, 9668858, 8796287, 9734391, 8919185, 9042045, 9050255, \
    8960150, 8837235, 9668855, 8898694, 8947855, 8968330, 8898706, 8890509, 8898693, 9001108, 8910999, 8960157, 9636088, \
    8878212, 9062536, 8849539, 9652478, 8984724, 8874137, 9074835, 8816776, 9029787, 9742613, 8951971, 8997015, 9201826, \
    9070756, 8845447, 8992912, 9042078, 9574637, 8964247, 9005225, 8939667, 9742599, 9664779, 9017501, 9885984, 8923255, \
    8927372, 9037976, 9013412, 8824963, 8951954, 9042070, 8960153, 9349295, 9177243, 8943758, 8976536, 9013380, 9005208, \
    9722101, 9586920, 8960137, 8841361, 8906881, 8943762, 8939658, 9619699, 9029782, 9078922, 8804497, 8956064, 8673408, \
    8968336, 9025684, 9107598, 9169072, 8997005, 8829069, 8747153, 9029765, 9726232, 8898703, 8874147, 9017508, 8738936, \
    8771707, 8640629, 9013386, 9037964, 8935558, 9070750, 9021574, 8833166, 8853646]

esgebv25100 = [0.028844845, 0.028847808, 0.028348168, 0.028700763, 0.028843166, 0.028718002, 0.02878778, 0.028523818, \
    0.0287632, 0.02883155, 0.028385356, 0.028979152, 0.02892758, 0.028040541, 0.028831469, 0.028640268, 0.0288199, \
    0.028755674, 0.028824966, 0.02876265, 0.028618986, 0.028970744, 0.028876683, 0.028939754, 0.028866785, 0.02861639, \
    0.028810557, 0.028586617, 0.028722214, 0.028942904, 0.02885002, 0.029111015, 0.02872505, 0.029037599, 0.027865086, \
    0.028875869, 0.028994108, 0.029327722, 0.029055573, 0.028692786, 0.028896622, 0.029039513, 0.029144818, 0.02893829, \
    0.028888952, 0.028866095, 0.02845084, 0.028008778, 0.029008571, 0.027789498, 0.028122598, 0.028765904, 0.029057968, \
    0.028954687, 0.028598934, 0.028873114, 0.029065998, 0.028951773, 0.029121635, 0.029224306, 0.0288087, 0.028971639, \
    0.028633751, 0.02901756, 0.028787315, 0.029160358, 0.028728439, 0.028808504, 0.02851649, 0.028864365, 0.028726667, \
    0.028969338, 0.029048834, 0.02888144, 0.028742427, 0.028914962, 0.028463315, 0.028864035, 0.029018316, 0.029047769, \
    0.029144118, 0.028837653, 0.028782427, 0.02862542, 0.028681058, 0.027615702, 0.028791029, 0.0287437, 0.02895664, \
    0.028401135, 0.028435549, 0.028334184, 0.028793186, 0.028877972, \
    0.02865156, 0.029089633, 0.028699849, 0.028785283, 0.028783228]

esgo25100 = [True, True, 'skip', False, 'skip', 'skip', False, False, 'skip', False, False, False, True, True, 'skip', \
    False, True, True, 'skip', 'skip', 'skip', False, False, False, 'skip', False, False, False, 'skip', False, False, \
    False, True, False, 'skip', 'skip', False, False, 'skip', True, 'skip', False, False, False, 'skip', True, 'skip', 'skip', \
    'skip', 'skip', False, True, False, 'skip', True, False, True, False, False, 'skip', True, False, False, 'skip', 'skip', \
    'skip', False, False, 'skip', False, True, 'skip', True, False, True, 'skip', 'skip', 'skip', 'skip', False, False, True, \
    True, 'skip', False, 'skip', 'skip', False, 'skip', 'skip', True, False, False, True, True, False, False, False, True]

loc100 = [78454, 272152, 116293, 631460, 242861, 535254, 519389, 647905, 497881, \
    369866, 553668, 234665, 267454, 575218, 580803, 639712, 565968, 729006, 279746, \
    642690, 130291, 316150, 639640, 482913, 518736, 229550, 644748, 432903, 600257, \
    194696, 729053, 618143, 615092, 537210, 486104, 558798, 377033, 742379, 652875, \
    624246, 143503, 628380, 73092, 505608, 478854, 747548, 600731, 776816, 221351, \
    718456, 539395, 734282, 468720, 314569, 94171, 682136, 577239, 625304, 186484, \
    426733, 725591, 416503, 473828, 110448, 721359, 640741, 663018, 603325, 673775, \
    415469, 600616, 632488, 679413, 637557, 583360, 705363, 758824, 591605, 183436, \
    650984, 548614, 589563, 414927, 592464, 483047, 552700, 138275, 723518, 621245, \
    264380, 597670, 717555, 188575, 716243, 500452, 728386, 247973, 302300, 167041, 181526]

ebv100 = [0.025876775, 0.025381923, 0.02545115, 0.029239096, 0.027769957, \
    0.022483557, 0.027306523, 0.019454915, 0.027479878, 0.025833605, 0.023340702, \
    0.027686868, 0.028030062, 0.022416193, 0.028364722, 0.019137364, 0.021638894, \
    0.024110856, 0.027289705, 0.03657897, 0.026405858, 0.028910855, 0.033827726, \
    0.023511775, 0.023106012, 0.02743964, 0.03706268, 0.030099545, 0.027191944, 0.02644228, \
    0.02319801, 0.032319535, 0.026870092, 0.025152072, 0.02979644, 0.021584136, \
    0.025253946, 0.022835098, 0.028096491, 0.035981294, 0.026567498, 0.03286307, \
    0.024157215, 0.02724219, 0.027747933, 0.020931374, 0.035107974, 0.0206298, \
    0.02759651, 0.021936491, 0.024791095, 0.024062393, 0.030018926, 0.025877524, \
    0.028594714, 0.028363746, 0.022205655, 0.03570793, 0.026070293, 0.029912295, \
    0.023829887, 0.03027212, 0.032056548, 0.028645368, 0.023324989, 0.019661417, \
    0.03256602, 0.026828483, 0.027702205, 0.029651044, 0.027708959, 0.027684407, \
    0.023480184, 0.034941025, 0.025229638, 0.021555161, 0.022973388, 0.021126183, \
    0.026750235, 0.020574965, 0.024479756, 0.020967092, 0.024577294, 0.024709476, \
    0.031490784, 0.023758978, 0.029028816, 0.021863073, 0.024401724, 0.028244575, \
    0.03190075, 0.026354993, 0.02828477, 0.021243712, 0.029832006, 0.02156803, \
    0.028065132, 0.027602375, 0.026776005, 0.027300343]

# observation bool
o2, o3, o4, o5, o6, o7, o8, o9, o10, o11 = True, True, False, True, True, \
    False, False, False, False, True

# split the event list into two lists, one for obs (true) and one for nobs (false)

# 4 observations, 7 non-observations, all at the same exact luminosity distance
if o5:
    if n:
        obs_list = [[ZT1, DT1, m1, dm1, 0.7, True],\
            [ZT1, DT1, m1, dm1, 0.8, True],\
            [ZT1, DT1, m1, dm1, 0.9, True]]#,\
            # [ZT1, DT1, m1, dm1, 1.0, True]]

        nobs_list = [[ZT1, DT1, m1, dm1, 0.0, False],\
            [ZT1, DT1, m1, dm1, 0.1, False],\
            [ZT1, DT1, m1, dm1, 0.2, False],\
            [ZT1, DT1, m1, dm1, 0.3, False],\
            [ZT1, DT1, m1, dm1, 0.4, False],\
            [ZT1, DT1, m1, dm1, 0.5, False],\
            [ZT1, DT1, m1, dm1, 0.6, False]]

        event_list = [[ZT1, DT1, m1, dm1, 0.0, False],\
            [ZT1, DT1, m1, dm1, 0.1, False],\
            [ZT1, DT1, m1, dm1, 0.2, False],\
            [ZT1, DT1, m1, dm1, 0.3, False],\
            [ZT1, DT1, m1, dm1, 0.4, False],\
            [ZT1, DT1, m1, dm1, 0.5, False],\
            [ZT1, DT1, m1, dm1, 0.6, False],\
            [ZT1, DT1, m1, dm1, 0.7, True],\
            [ZT1, DT1, m1, dm1, 0.8, True],\
            [ZT1, DT1, m1, dm1, 0.9, True]]
    elif monly:
        obs_list = [[ZT1, DT1, m1, dm1, 0.0, False],\
            [ZT1, DT1, m1, dm1, 0.1, False],\
            [ZT1, DT1, m1, dm1, 0.2, False],\
            [ZT1, DT1, m1, dm1, 0.3, False],\
            [ZT1, DT1, m1, dm1, 0.4, False],\
            [ZT1, DT1, m1, dm1, 0.5, False],\
            [ZT1, DT1, m1, dm1, 0.6, False],\
            [ZT1, DT1, m1, dm1, 0.7, True],\
            [ZT1, DT1, m1, dm1, 0.8, True],\
            [ZT1, DT1, m1, dm1, 0.9, True]]
        
        nobs_list =  None
        
        event_list = [[ZT1, DT1, m1, dm1, 0.0, False],\
            [ZT1, DT1, m1, dm1, 0.1, False],\
            [ZT1, DT1, m1, dm1, 0.2, False],\
            [ZT1, DT1, m1, dm1, 0.3, False],\
            [ZT1, DT1, m1, dm1, 0.4, False],\
            [ZT1, DT1, m1, dm1, 0.5, False],\
            [ZT1, DT1, m1, dm1, 0.6, False],\
            [ZT1, DT1, m1, dm1, 0.7, True],\
            [ZT1, DT1, m1, dm1, 0.8, True],\
            [ZT1, DT1, m1, dm1, 0.9, True]]
    else:
        obs_list = [[ZT1, DT1, m1, dm1, 0.7, True],\
            [ZT1, DT1, m1, dm1, 0.8, True],\
            [ZT1, DT1, m1, dm1, 0.9, True]]

        nobs_list =  None

        event_list = [[ZT1, DT1, m1, dm1, 0.7, True],\
            [ZT1, DT1, m1, dm1, 0.8, True],\
            [ZT1, DT1, m1, dm1, 0.9, True]]

if o100:
    obs_list = [[ZT1, DT1, m1, dm1, v100[i], ob100[i]] for i, x in enumerate(ob100) if x==True]
    if n:
        nobs_list = [[ZT1, DT1, m1, dm1, v100[i], ob100[i]] for i, x in enumerate(ob100) if x==False]
        event_list = nobs_list + obs_list
    else:
        nobs_list = None
        event_list = obs_list

if o100m or o100o:
    obs_list = [[ZT1, DT1, m1, dm1, v, True] for v in v100]
    nobs_list = None
    event_list = obs_list


if m100m:
    obs_list = [[mz100[i], md100[i], (1.4, 1.4), (0.0, 0.0), mv100[i], True] for i, x in enumerate(mo100)]
    nobs_list = None
    event_list = obs_list

if m100:
    obs_list = [[mz100[i], md100[i], (1.4, 1.4), (0.0, 0.0), mv100[i], mo100[i]] for i, x in enumerate(mo100) if x==True]
    if n:
        nobs_list = [[mz100[i], md100[i], (1.4, 1.4), (0.0, 0.0), mv100[i], mo100[i]] for i, x in enumerate(mo100) if x==False]
        event_list = nobs_list + obs_list
    else:
        nobs_list = None
        event_list = obs_list

if e100:
    obs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], eo100[i]] for i, x in enumerate(eo100) if x==True]
    if n:
        nobs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], eo100[i]] for i, x in enumerate(eo100) if x==False]
        event_list = nobs_list + obs_list
    else:
        nobs_list = None
        event_list = obs_list

if es100:
    obs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], eso100[i], loc100[i], ebv100[i]] for i, x in enumerate(eso100) if x==True]
    if n:
        nobs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], eso100[i], loc100[i], ebv100[i]] for i, x in enumerate(eso100) if x==False]
        event_list = nobs_list + obs_list
    else:
        nobs_list = None
        event_list = obs_list


if es25100:
    obs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], eso100[i], loc100[i], ebv100[i]] for i, x in enumerate(eso25100) if x==True]
    if n:
        nobs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], eso100[i], loc100[i], ebv100[i]] for i, x in enumerate(eso25100) if x==False]
        event_list = nobs_list + obs_list
    else:
        nobs_list = None
        event_list = obs_list


if esg25100:
    obs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], esgo25100[i], esgloc25100[i], esgebv25100[i]] for i, x in enumerate(esgo25100) if x==True]
    if n:
        nobs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], esgo25100[i], esgloc25100[i], esgebv25100[i]] for i, x in enumerate(esgo25100) if x==False]
        event_list = nobs_list + obs_list
    else:
        nobs_list = None
        event_list = obs_list


if esg23100:
    obs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], esgo23100[i], esgloc25100[i], esgebv25100[i]] for i, x in enumerate(esgo23100) if x==True]
    if n:
        nobs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], esgo23100[i], esgloc25100[i], esgebv25100[i]] for i, x in enumerate(esgo23100) if x==False]
        event_list = nobs_list + obs_list
    else:
        nobs_list = None
        event_list = obs_list


if em100:
    obs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], True] for i, x in enumerate(eo100)]
    nobs_list = None
    event_list = obs_list

if e10:
    obs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], eo100[i]] for i, x in enumerate(eo100[:10]) if x==True]
    if n:
        nobs_list = [[ez100[i], ed100[i], (em1100[i], em2100[i]),\
         (edm1100[i], edm2100[i]), ev100[i], eo100[i]] for i, x in enumerate(eo100[:10]) if x==False]
        event_list = nobs_list + obs_list
    else:
        nobs_list = None
        event_list = obs_list

if one:
    idx = 0
    obs_list = None
    nobs_list = [[mz100[idx], md100[idx], (1.4, 1.4), (0.0, 0.0), mv100[idx], mo100[idx]]]
    event_list = nobs_list

# if o5:
#     if n:
#         obs_list = [[ZT2, DT2, m2, dm2, v2, o2],\
#             [ZT3, DT3, m3, dm3, v3, o3],\
#             [ZT5, DT5, m5, dm5, v5, o5],\
#             [ZT6, DT6, m6, dm6, v6, o6],\
#             [ZT11, DT11, m11, dm11, v11, o11]]

#         nobs_list = [[ZT10, DT10, m10, dm10, v10, o10],\
#             [ZT4, DT4, m4, dm4, v4, o4],\
#             [ZT7, DT7, m7, dm7, v7, o7],\
#             [ZT8, DT8, m8, dm8, v8, o8],\
#             [ZT9, DT9, m9, dm9, v9, o9]]

#         event_list = [[ZT4, DT4, m4, dm4, v4, o4],\
#             [ZT7, DT7, m7, dm7, v7, o7],\
#             [ZT8, DT8, m8, dm8, v8, o8],\
#             [ZT9, DT9, m9, dm9, v9, o9],\
#             [ZT10, DT10, m10, dm10, v10, o10],\
#             [ZT2, DT2, m2, dm2, v2, o2],\
#             [ZT3, DT3, m3, dm3, v3, o3],\
#             [ZT5, DT5, m5, dm5, v5, o5],\
#             [ZT6, DT6, m6, dm6, v6, o6],\
#             [ZT11, DT11, m11, dm11, v11, o11]]
#     else:
#         obs_list = [[ZT2, DT2, m2, dm2, v2, o2],\
#             [ZT3, DT3, m3, dm3, v3, o3],\
#             [ZT5, DT5, m5, dm5, v5, o5],\
#             [ZT6, DT6, m6, dm6, v6, o6],\
#             [ZT11, DT11, m11, dm11, v11, o11]]

#         nobs_list =  None

#         event_list = [[ZT2, DT2, m2, dm2, v2, o2],\
#             [ZT3, DT3, m3, dm3, v3, o3],\
#             [ZT5, DT5, m5, dm5, v5, o5],\
#             [ZT6, DT6, m6, dm6, v6, o6],\
#             [ZT11, DT11, m11, dm11, v11, o11]]


if o1:
    if n:
        obs_list = [[ZT1, DT1, m1, dm1, 0.9, True]]
        nobs_list = [[ZT1, DT1, m1, dm1, 0.1, False]]
        # nobs_list = [[0.5, u_true.luminosity_distance(0.5).value, m1, dm1, 0.5, False]]

        event_list = [[ZT1, DT1, m1, dm1, 0.1, False],\
            [ZT1, DT1, m1, dm1, 0.9, True]]
    else:
        obs_list = [[ZT1, DT1, m1, dm1, 0.9, True]]
        nobs_list = None
        event_list = [[ZT1, DT1, m1, dm1, 0.9, True]]


n_events = len(event_list)
if nobs_list is not None:
    n_nobs = len(nobs_list)
else:
    n_nobs = 0
if obs_list is not None:
    n_obs = len(obs_list)
else:
    n_obs = 0

det_obs = [event[-1] for event in event_list]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # ed1001 = [ed100[i] for i, x in enumerate(eso25100) if x==True]
    # ev1001 = [ev100[i] for i, x in enumerate(eso25100) if x==True]
    # ed1002 = [ed100[i] for i, x in enumerate(esgo25100) if x==True]
    # ev1002 = [ev100[i] for i, x in enumerate(esgo25100) if x==True]
    # plt.plot(ed100, ev100, 'k.')
    # plt.plot(ed1001, ev1001, 'or')
    # plt.plot(ed1002, ev1002, 'k+')
    # plt.show()
    import numpy as np
    # trustworthy, made by chccking every pointing
    withskips = [True, True, False, False, 'skip', 'skip', False, False, 'skip', False, False, False, True, True, False, False, True, True, False, 'skip', False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, 'skip', True, 'skip', False, False, False, False, True, False, False, 'skip', False, False, True, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, 'skip', False, True, False, True, False, True, False, False, False, 'skip', False, False, True, True, 'skip', False, 'skip', 'skip', False, False, 'skip', True, False, False, True, True, False, False, False, True]
    ws2 = [True, True, False, False, 'skip', 'skip', False, False, 'skip', False, False, False, True, True, False, False, True, True, False, 'skip', False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, 'skip', True, 'skip', False, False, False, False, True, False, False, 'skip', False, False, True, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, 'skip', False, True, False, True, False, True, False, False, False, 'skip', False, False, True, True, 'skip', False, 'skip', 'skip', False, False, 'skip', True, False, False, True, True, False, False, False, True]
    print(len(list(filter(('skip').__ne__, withskips))))
    print(len(list(filter(('skip').__ne__, ws2))))
    print(withskips==ws2)
    print(ws2)
