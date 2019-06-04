# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
=============================================
sbpy Asteroid Reflectance Spectroscopy Module
=============================================
created on June 4, 2019
"""

__all__ = ['Taxon', 'classify']

import os
import numpy as np
from collections import OrderedDict

from scipy.interpolate import InterpolatedUnivariateSpline

import astropy.units as u
from astropy.modeling import Fittable1DModel, Parameter
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table

from ...data import Phys
from . import schemas


class Taxon(Fittable1DModel):

    # astropy.modeling parameter: spectral slope
    slope = Parameter(unit=u.percent/u.um)
    _input_units_allow_dimensionless = True
    input_units_equivalencies = {'x': u.spectral()}

    @property
    def input_units(self):
        """Define units of input data."""
        return {'x': u.micron}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        """Define units of model parameters."""
        return OrderedDict([('slope', u.percent/u.um)])

    def __init__(self, taxon, schema='BusDeMeo', slope=0):
        self.schema = schema
        self.taxon = taxon
        self.slope = slope

        self.raw_spec = None  # raw discrete spectrum
        self.spec = None  # interpolated spectrum
        self.sigma = None  # interpolated (nearest-neighbor) uncertainties

        self.wavelength_range = self.from_file(schema, taxon)

    def from_file(self, schema, taxon, spline_order=3):

        from astropy.utils.data import _is_url

        try:
            parameters = getattr(schemas, schema).copy()

            # locate data file
            if not _is_url(parameters['filename']):
                # find in the module's location
                parameters['filename'] = get_pkg_data_filename(
                    os.path.join('data', parameters['filename']))

            # read data file
            spec = Table.read(parameters['filename'],
                              format='ascii.commented_header',
                              header_start=2,
                              delimiter=' ',
                              fill_values=[('-0.999', '0')])

            # extract spectrum for this taxon
            spec = spec['Wavelength', taxon+'_Mean', taxon+'_Sigma']
            spec.rename_column(taxon+'_Mean', 'Spec')
            spec.rename_column(taxon+'_Sigma', 'Sigma')
            spec['Wavelength'].unit = parameters['wave_unit']
            spec['Spec'].unit = parameters['flux_unit']
            spec['Sigma'].unit = parameters['flux_unit']
            self.raw_spec = Phys.from_table(spec)

            # # apply reddening
            # if reddening_slope > 0:
            #     self.raw_spec = self.redden(
            #         self.raw_spec, reddening_slope)

            # derive weights limiting minimum uncertainties
            weights = np.array([1/max(val, 0.001)**2 for val in
                                self.raw_spec['Sigma'].value])

            # interpolate spectrum using splines
            self.spec = InterpolatedUnivariateSpline(
                self.raw_spec['Wavelength'].to('um').value,
                self.raw_spec['Spec'].value,
                w=weights,
                k=spline_order)

        except AttributeError:
            msg = 'Unknown taxonomy schema "{}".  Valid schemas:\n{}'.format(
                schema, schemas.available)
            raise ValueError(msg)
        except KeyError:
            msg = 'Unknown taxonomic type "{}".  Valid types:\n{}'.format(
                taxon, parameters['types'])
            raise ValueError(msg)

        return (np.min(self.raw_spec['Wavelength']),
                np.max(self.raw_spec['Wavelength']))

    @staticmethod
    def redden(spec, slope, normalized_at=0.9*u.um):

        offset = ((spec['Wavelength']-normalized_at).to('um').value *
                  slope.to('percent/um').value/100)
        spec['Spec'] = spec['Spec'].value+offset*u.dimensionless_unscaled

        return spec

    def evaluate(self, x, *pargs):

        if len(pargs) > 0:
            if (isinstance(pargs[0], u.Quantity)):
                slope = pargs[0].to(u.percent/u.um)
            else:

                slope = u.Quantity(pargs[0], u.percent/u.um)

        return self.spec(x)


def classify(eph, schema='BusDemeo'):
    pass
