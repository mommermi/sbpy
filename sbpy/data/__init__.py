# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
sbpy.data
---------

:author: Michael Mommert (mommermiscience@gmail.com)
"""


class Conf():

    # alternative field names for DataClass
    # append field description as final item in the list (for documentation)
    fieldnames = [
        ['targetname', 'id', 'Object', 'Target Identifier'],

        # orbital elements
        ['i', 'inc', 'incl', 'Inclination'],
        ['epoch', 'datetime_jd', 'JD', 'Date', 'date', 'Epoch'],
        ['Omega', 'longnode', 'Longitude of the Ascending Node'],
        ['w', 'argper', 'Argument of the Periapsis'],

        # ephemerides
        ['r', 'rh', 'r_hel', 'heldist', 'Heliocentric Distance'],
        ['delta', 'Delta', 'obsdist', 'Distance to the Observer'],
        ['ra', 'RA', 'Right Ascension'],
        ['dec', 'DEC', 'Dec', 'Declination'],
        ['ra_rate', 'RA_rate', 'ra_rates', 'RA_rates', 'dRA',
         'dra', 'RA Rate'],
        ['RA*cos(Dec)_rate', 'dra cos(dec)', 'dRA cos(Dec)',
         'dra*cos(dec)', 'dRA*cos(Dec)', 'RA*cos(Dec) Rate'],
        ['dec_rate', 'DEC_rate', 'Dec_rate', 'dec_rates', 'DEC_rates',
         'Dec_rates', 'dDec', 'dDEC', 'ddec', 'Dec Rate'],
        ['mu', 'Proper motion', 'Proper Motion'],
        ['Direction', 'directin', 'Proper Motion Direction'],
        ['alpha', 'phaseangle', 'Phase', 'Solar Phase Angle'],
        ['elong', 'solarelong', 'solarelongation', 'elongation',
         'Elongation', 'Solar Elongation'],
        ['V', 'Vmag', 'V-band Magnitude'],
        ['hlon', 'EclLon', 'ecllon', 'HelEclLon',
         'helecllon', 'Heliocentric Ecliptic Longitude'],
        ['hlat', 'EclLat', 'ecllat', 'HelEclLat', 'helecllat',
         'Heliocentric Ecliptic Latitude'],
        ['el', 'EL', 'elevation', 'alt', 'altitude', 'Altitude',
         'Elevation'],
        ['az', 'AZ', 'azimuth', 'Azimuth'],
        ['lunar_elong', 'elong_moon', 'elongation_moon',
         'lunar_elongation', 'lunarelong', 'Lunar Elongation'],
        ['vx', 'dx', 'dx/dt', 'x Velocity Component'],
        ['vy', 'dy', 'dy/dt', 'y Velocity Component'],
        ['vz', 'dz', 'dz/dt', 'z Velocity Component'],

        # physical properties
        ['d', 'D', 'diam', 'diameter', 'Diameter'],
        ['R', 'radius', 'Radius'],
        ['pv', 'pV', 'p_v', 'p_V', 'albedo', 'geoalb',
         'V-band Geometric Albedo'],
    ]

    fieldname_idx = {}
    for idx, field in enumerate(fieldnames):
        for alt in field:
            fieldname_idx[alt] = idx

    # field equivalencies defining conversions
    # key defines target quantity; dict with source quantity and function
    # for conversion
    # conversions considered as part of DataClass._translate_columns
    field_eq = {'R': {'d': lambda r: r/2},
                # diameter to radius}
                'd': {'R': lambda d: d*2}
                }

    # definitions for use of pyoorb in Orbits
    oorb_timeScales = {'UTC': 1, 'UT1': 2, 'TT': 3, 'TAI': 4}
    oorb_elemType = {'CART': 1, 'COM': 2, 'KEP': 3, 'DEL': 4, 'EQX': 5}

    oorb_orbit_fields = {'COM': ['id', 'q', 'e', 'incl', 'Omega',
                                 'w', 'Tp_jd', 'orbtype', 'epoch',
                                 'epoch_scale', 'H', 'G'],
                         'KEP': ['id', 'a', 'e', 'incl', 'Omega', 'w', 'M',
                                 'orbtype', 'epoch', 'epoch_scale', 'H',
                                 'G'],
                         'CART': ['id', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                                  'orbtype', 'epoch', 'epoch_scale', 'H',
                                  'G']}
    oorb_orbit_units = {'COM': [None, 'au', None, 'deg', 'deg',
                                'deg', 'd', None, 'd',
                                None, 'mag', None],
                        'KEP': [None, 'au', None, 'deg', 'deg', 'deg', 'deg',
                                None, 'd', None, 'mag', None],
                        'CART': [None, 'au', 'au', 'au', 'au/d', 'au/d',
                                 'au/d', None, 'd', None, 'mag', None]}

    oorb_ephem_fields = ['MJD', 'RA', 'DEC', 'RA*cos(Dec)_rate', 'DEC_rate',
                         'alpha', 'elong', 'r', 'Delta', 'V', 'pa', 'TopEclLon',
                         'TopEclLat', 'OppTopEclLon', 'OppTopEclLat',
                         'HelEclLon', 'HelEclLat', 'OppHelEclLon',
                         'OppHelEclLat', 'EL', 'ELsun', 'ELmoon',
                         'lunarphase', 'lunarelong', 'x', 'y', 'z',
                         'vx', 'vy', 'vz', 'obsx', 'obsy', 'obsz',
                         'trueanom']
    oorb_ephem_units = ['d', 'deg', 'deg', 'deg/d', 'deg/d', 'deg',
                        'deg', 'au', 'au', 'mag', 'deg', 'deg',
                        'deg', 'deg', 'deg',
                        'deg', 'deg', 'deg',
                        'deg', 'deg', 'deg', 'deg',
                        None, 'deg', 'au', 'au', 'au',
                        'au/d', 'au/d', 'au/d', 'au', 'au', 'au', 'deg']


conf = Conf()

from .core import (DataClass, mpc_observations, sb_search,
                   image_search, pds_ferret)
from .ephem import Ephem
from .orbit import Orbit
from .phys import Phys
from .names import Names, natural_sort_key

__all__ = ['DataClass', 'Ephem', 'Orbit', 'Phys', 'Names', 'conf', 'Conf',
           'mpc_observations', 'sb_search', 'image_search',
           'pds_ferret']
