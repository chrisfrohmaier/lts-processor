import healpy as hp
import pandas as pd
from astropy.table import Table, QTable
import numpy as np
from shapely.geometry import Point
import lts_areas as lts
import glob
import os
import json
import astropy.units as u
import time

NSIDE = 64


def merge_JsonFiles(filename):
    result = list()
    for f1 in filename:
        with open(f1, 'r') as infile:
            result.append(json.load(infile))
    # print(result)
    # with open('testDump.json', 'w') as output_file:
    #     json.dump(result, output_file, indent=4)
    return result


def convertUserWeightToLTSWeight(userWeight):
    """
    Converts a user-provided weight to an LTS weight.
    The LTS is defined as a weight of 1 == 20% of the avaliable time. This scales such that 
    100% of the time is infinty.

    The function ensures that the input user weight does not exceed 0.9. 
    It then calculates the LTS weight using the formula:
        weightLTS = (userWeight * 4.0) / (1 - userWeight)

    Args:
        userWeight (float): The user-provided weight, expected to be in the range [0, 1].

    Returns:
        float: The calculated LTS weight.

    Notes:
        - If the input user weight is greater than 0.9, it is capped at 0.9 
          to avoid division by zero or excessively large values.
    """
    '''
    This function takes the user weights
    '''
    if userWeight > 0.9:
        userWeight = 0.9
    weightLTS = (userWeight*4.0)/(1-userWeight)
    return weightLTS


ltsInputsJSONFiles = glob.glob('surveyInput/*.json')


x2 = merge_JsonFiles(ltsInputsJSONFiles)

# Make an array of LTS objects
ltsObjs = []
for i in range(len(x2)):
    for j in range(len(x2[i]['year1Areas'])):
        if x2[i]['year1Areas'][j]['type']=='box':
            ltsObjs.append(lts.LTS_Areas(survey=x2[i]['survey'],
                                            area_name=x2[i]['year1Areas'][j]['name'],
                                            ra=list(x2[i]['year1Areas'][j]['RA']),
                                            dec=list(x2[i]['year1Areas'][j]['Dec']),
                                            poly_type=x2[i]['year1Areas'][j]['type'],
                                            t_frac=x2[i]['year1Areas'][j]['t_frac']
                                        )
                            )
        elif x2[i]['year1Areas'][j]['type']=='ellipse':
            ltsObjs.append(lts.LTS_Areas(survey=x2[i]['survey'],
                                            area_name=x2[i]['year1Areas'][j]['name'],
                                            ra = x2[i]['year1Areas'][j]['RA'],
                                            dec = x2[i]['year1Areas'][j]['Dec'],
                                            poly_type= x2[i]['year1Areas'][j]['type'],
                                            t_frac= x2[i]['year1Areas'][j]['t_frac'],
                                            a = x2[i]['year1Areas'][j]['a'],
                                            b = x2[i]['year1Areas'][j]['b'],
                                            theta =x2[i]['year1Areas'][j]['theta']
                                        )
                            )
        elif x2[i]['year1Areas'][j]['type']=='circle':
            ltsObjs.append(lts.LTS_Areas(survey=x2[i]['survey'],
                                            area_name=x2[i]['year1Areas'][j]['name'],
                                            ra = float(x2[i]['year1Areas'][j]['RA_center']),
                                            dec = float(x2[i]['year1Areas'][j]['Dec_center']),
                                            poly_type= x2[i]['year1Areas'][j]['type'],
                                            t_frac= x2[i]['year1Areas'][j]['t_frac'],
                                            radius = x2[i]['year1Areas'][j]['radius'],
                                            
                                        )
                            )
                                            
        else:
            continue

t_frac_array = np.array([x.t_frac for x in ltsObjs])

t_frac_argsort = np.argsort(t_frac_array)

npix = hp.nside2npix(NSIDE)
hpCentres = np.array(list(map(lambda x: [x]+list(hp.pix2ang(NSIDE, x, lonlat=True, nest=True)), range(npix))))

allHpxPoint = list(map(Point, hpCentres[:,1], hpCentres[:,2]))

hpx_map = np.zeros(npix, dtype=np.float64)
hpx_map = hpx_map
for i in t_frac_argsort:
    # print(ltsObjs[i].area_name)
    tfrac = convertUserWeightToLTSWeight(ltsObjs[i].t_frac)
    survey = ltsObjs[i].survey
    area_name = ltsObjs[i].area_name
    # print(tfrac)
    ltsArea = ltsObjs[i].makeShapelyPolygon()
    tileInArea = np.array(list(map(ltsArea.contains, allHpxPoint)))
    hpx_map[tileInArea] = tfrac

healpixPD = pd.DataFrame(hpCentres, columns=['hpxIdx', 'ra', 'dec'])
pixWeights = pd.DataFrame(
    np.column_stack((range(npix), hpx_map)),
    columns=['hpxIdx', 'weight']
)

year1Weights = pd.merge(healpixPD, pixWeights)
year1Weights = year1Weights[(year1Weights['dec'] < 40) & (year1Weights['weight'] > 0)].copy()

year1Weights['hpxIdx'] = year1Weights['hpxIdx'].astype(int)
year1Weights['JD'] = 2460828.17
year1Weights['weight_timescale'] = 365.0

year1Weights.rename(columns={'hpxIdx':'field_id'}, inplace=True)
year1Weights.rename(columns={'weight':'LTS_user_weight'}, inplace=True)

ltsY1 = QTable.from_pandas(
    year1Weights[
        ['field_id', 'ra', 'dec', 'JD', 'LTS_user_weight', 'weight_timescale']
    ]
)

col_units = {
    'field_id': '',
    'ra': 'deg',
    'dec': 'deg',
    'JD': 'd',
    'LTS_user_weight': '',
    'weight_timescale': 'd'
}

for k, v in col_units.items():
    ltsY1[k].unit = u.Unit(v, format='fits')

ltsY1.meta = {
    'nside': 64,
    'ordering': 'NESTED',
    'COORDSYS': 'C',
    'MAX_DEC': 40,
    'JD_zp': 2460828.17,
    'author':'Chris Frohmaier',
    'Date': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    
}

# ltsY1.description = {
#     'Description':'This file contains the weights for the LTS year 1 fields.',
#     'field_id':'The HEALPIX ID of the field',
#     'ra':'The RA of the field in degrees',
#     'dec':'The Dec of the field in degrees',
#     'JD':'The Julian Date on which the weight will become active',
#     'LTS_user_weight':'The LTS weight converted from the user weight. A weight of 1.0 corresponds to 20 per cent of the available time. Weights are capped at 0.9 to avoid division by zero or excessively large values.',
#     'weight_timescale': 'The timescale (in days) over which the weight is '
#                         'applied.'
# }

ltsY1['ra'].info.description = 'The RA of the field in degrees'
ltsY1['dec'].info.description = 'The Dec of the field in degrees'
ltsY1['JD'].info.description = 'The Julian Date on which the weight will become active'
ltsY1['LTS_user_weight'].info.description = (
    'The LTS weight converted from the user weight. '
    'A weight of 1.0 corresponds to 20 per cent of the available time. '
    'Weights are capped at 0.9 to avoid division by zero or excessively large values.'
)
ltsY1['weight_timescale'].info.description = 'The timescale (in days) over which the weight is applied.'
ltsY1['field_id'].info.description = 'The HEALPIX ID of the field'

ltsY1.write('lts_year1_weights.fits', overwrite=True)
