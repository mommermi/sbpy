# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
=============================================
sbpy Asteroid Reflectance Spectroscopy Module
=============================================
created on June 4, 2019
"""

# Parameters passed to reflectance.from_file; 'filename' is a URL or name of a
# file distributed with sbpy and located in sbpy/spectroscopy/reflectance/data
# (see also spectroscopy/reflectance/setup_package.py). After adding taxonomy
# schemas here, update __init__.py docstring and docs/sbpy/spectroscopy.rst

available = [
    'BusDeMeo'
]

BusDeMeo = {
    'filename': 'busdemeo-meanspectra.csv',
    'wave_unit': 'um',
    'flux_unit': 'dimensionless_unscaled',
    'description': 'Bus-DeMeo asteroid taxonomy schema',
    'normalized_to': 0.9,
    'types': ['A', 'B', 'C', 'Cb', 'Cg', 'Cgh', 'Ch', 'D', 'K', 'L',
              'O', 'Q', 'R', 'S', 'Sa', 'Sq', 'Sr', 'Sv', 'T', 'V',
              'X', 'Xc', 'Xe', 'Xk', 'Xn'],
    'bibcode': '2009Icar..202..160D'
}
