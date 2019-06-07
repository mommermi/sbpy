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
from astropy.modeling import Fittable1DModel, Parameter, fitting
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table

from ...data import Phys
from . import schemas


class Taxon(Fittable1DModel):

    # astropy.modeling parameters: spectral slope and reflectance offset
    slope = Parameter(unit=u.percent/u.um)
    offset = Parameter(unit=u.dimensionless_unscaled)
    _input_units_allow_dimensionless = True
    input_units_equivalencies = {'x': u.spectral()}

    @property
    def input_units(self):
        """Define units of input data."""
        return {'x': u.um}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        """Define units of model parameters."""
        return OrderedDict([('slope', u.percent/u.um),
                            ('offset', u.dimensionless_unscaled)])

    def __init__(self, taxon, schema='BusDeMeo', slope=0):
        self.schema = schema
        self.taxon = taxon

        self.raw_spec = None  # raw discrete spectrum
        self.interp_spec = None  # interpolated spectrum
        self.sigma = None  # interpolated (nearest-neighbor) uncertainties

        self.spline_order = None  # polynomial order used for splines
        self.wavelength_range = self.from_file(schema, taxon)

    def from_file(self, schema, taxon, reddening_slope=0*u.percent/u.um,
                  spline_order=3):

        from astropy.utils.data import _is_url

        try:
            parameters = getattr(schemas, schema).copy()
        except AttributeError:
            msg = 'Unknown taxonomy schema "{}".  Valid schemas:\n{}'.format(
                schema, schemas.available)
            raise ValueError(msg)

        # locate data file
        try:
            if not _is_url(parameters['filename']):
                # find in the module's location
                parameters['filename'] = get_pkg_data_filename(
                    os.path.join('data', parameters['filename']))
        except KeyError:
            msg = 'Unknown taxonomic type "{}".  Valid types:\n{}'.format(
                taxon, parameters['types'])
            raise ValueError(msg)

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

        # apply reddening and update interpolated spectrum
        self.spline_order = spline_order
        spec = self.redden(
            self.raw_spec, reddening_slope)
        self._update(spec=spec)

        return (np.min(self.raw_spec['Wavelength']),
                np.max(self.raw_spec['Wavelength']))

    @staticmethod
    def redden(spec, slope, normalized_at=0.9*u.um):

        spec = spec.copy()
        offset = ((spec['Wavelength']-normalized_at).to('um').value *
                  slope.to('percent/um').value/100)
        spec['Spec'] = spec['Spec'].value+offset*u.dimensionless_unscaled

        return spec

    def _update(self, spec=None):

        if spec is None:
            spec = self.raw_spec

        # derive weights limiting minimum uncertainties
        weights = np.array([1/max(val, 0.001)**2 for val in
                            spec['Sigma'].value])

        # interpolate spectrum using splines
        self.interp_spec = InterpolatedUnivariateSpline(
            spec['Wavelength'].to('um').value,
            spec['Spec'].value,
            w=weights,
            k=self.spline_order)

    def evaluate(self, x, *pargs):

        if len(pargs) > 0:
            if (isinstance(pargs[0], u.Quantity)):
                slope = pargs[0].to(u.percent/u.um)
            else:
                slope = u.Quantity(pargs[0], u.percent/u.um)
            if (isinstance(pargs[1], u.Quantity)):
                offset = pargs[1].to(u.dimensionless_unscaled)
            else:
                offset = u.Quantity(pargs[1], u.dimensionless_unscaled)
        else:
            offset = 0*u.dimensionless_unscaled

        # apply reddening and offset, update interpolated spectrum
        spec = self.redden(self.raw_spec, slope)
        spec['Spec'] = spec['Spec'] - offset
        self._update(spec=spec)

        return self.interp_spec(x)

    def fit(self, spec, init_slope=0*u.percent/u.um,
            init_offset=0*u.dimensionless_unscaled,
            fitter=fitting.SLSQPLSQFitter()):
        slope = init_slope
        offset = init_offset
        Fittable1DModel.__init__(self, slope, offset)

        fit = fitter(self, spec['Wavelength'], spec['Spec'])

        # apply best-fit slope
        spec = self.redden(self.raw_spec, fit.slope.value*fit.slope.unit)
        spec['Spec'] = spec['Spec'] - fit.offset
        self._update(spec=spec)

        return fit


def normalize(spec, normalize_at=0.9*u.um):
    pass


def classify(spec, schema='BusDemeo'):
    pass
