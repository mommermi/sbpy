# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy Thermal Module

created on June 27, 2017
"""

import numpy as np

import astropy.units as u
from astropy import constants as const
from ._thermal import integrate_planck
from astropy.modeling import Fittable1DModel, Parameter

from collections import OrderedDict

__all__ = ['ThermalClass', 'STM', 'FRM', 'NEATM']


class ThermalClass(Fittable1DModel):

    def __init__(self, phys, eph):
        """phys should only refer to a single object, while eph can have
        multiple epochs"""
        self.phys = phys.table
        self.eph = eph.table

    def get_subsolartemp(self):
        return (const.L_sun/(4*np.pi*const.au.to(u.m)**2) *
                (1.-self.phys['bondalbedo']) /
                (self.eph['r'].to('au').value**2*self.phys['eta'] *
                 const.sigma_sb*self.phys['emissivity']))**0.25

    def _flux(self, model, wavelengths, jy=True):
        """Evaluate model at given wavelengths and return spectral flux
        densities.

        Parameters
        ----------
        model : integer
            Model identifier code for C code (1: STM, 2: FRM, 3: NEATM)
        wavelengths :`astropy.units` quantity or list
            Wavelengths at which to evaluate model. If a list of floats is
            provided, wavelengths must be in units of micron.
        jy : bool, optional
            If `True`, resulting flux densities will be returned in units of
            Janskys. If `False`, flux densities will be returned in SI units
            (W / (micron m2)). Default: `True`

        """

        if jy:
            flux_unit = u.astrophys.Jy
        else:
            flux_unit = u.W/(u.micron*u.m**2)

        return (self.phys['emissivity'] * self.phys['diam']**2 /
                self.eph['delta']**2 *
                np.pi * const.h * const.c**2 / wavelengths**5 *
                integrate_planck(
                    model, 0, np.pi/2, np.array(wavelengths),
                    self.phys['Tss'][0].to('K').value,
                    self.eph['alpha'].to('deg').value)).to(
                        flux_unit,
                        equivalencies=u.spectral_density(wavelengths))

    @staticmethod
    def _apply_unit(value, unit, equiv=None):
        """utility function to enforce Quantity nature for object"""
        if isinstance(value, u.Quantity):
            return value.to(unit, equivalencies=equiv)
        else:
            return u.Quantity(value, unit)


class STM(ThermalClass):

    # define astropy.modeling.Fittable1DModel model parameters
    diam = Parameter(default=1, min=0, unit=u.km)
    subsolartemp = Parameter(default=200, min=0, unit=u.K)

    # enable dimensionless input parameter (wavelength)
    _input_units_allow_dimensionless = True

    # input unit is a wavelength, assign unit equivalency
    input_units_equivalencies = {'x': u.spectral()}

    @property
    def input_units(self):
        return {'x': u.micron}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return OrderedDict([('subsolartemp', u.K),
                            ('diam', u.km)])

    def __init__(self, phys, eph):
        """phys should only refer to a single object, while eph can have
        multiple epochs"""
        ThermalClass.__init__(self, phys, eph)

        # assign model parameters from phys
        diam = self._apply_unit(self.phys['diam'], u.km)
        subsolartemp = self._apply_unit(self.phys['Tss'], u.K)

        Fittable1DModel.__init__(self, diam, subsolartemp)

    def evaluate(self, x, *args, jy=True):
        """Evaluate model at given wavelengths and return spectral flux densities.

        Parameters
        ----------
        model : integer
            Model identifier code for C code (1: STM, 2: FRM, 3: NEATM)
        wavelengths :`astropy.units` quantity or list
            Wavelengths at which to evaluate model. If a list of floats is
            provided, wavelengths must be in units of micron.
        jy : bool, optional
            If `True`, resulting flux densities will be returned in units of
            Janskys. If `False`, flux densities will be returned in SI units
            (W / (micron m2)). Default: `True`
        """

        _no_unit = False
        if not hasattr(x, 'unit'):
            _no_unit = True

        # set parameters and input as quantities
        x = self._apply_unit(x, u.micron)

        # use *args, if provided...
        if len(args) == 2:
            diam = self._apply_unit(args[0], u.km)
            subsolartemp = self._apply_unit(args[1], u.K,
                                            equiv=u.temperature())
        # or pull the parameters from self.phys
        else:
            diam = self._apply_unit(self.phys['diam'][0], u.km)
            subsolartemp = self._apply_unit(self.phys['Tss'][0], u.K)

        self.phys['diam'] = diam
        self.phys['Tss'] = subsolartemp

        flux = self._flux(1, x, jy)

        if _no_unit:
            return flux.value
        else:
            return flux


class FRM(ThermalClass):

    # define astropy.modeling.Fittable1DModel model parameters
    diam = Parameter(default=1, min=0, unit=u.km)
    subsolartemp = Parameter(default=200, min=0, unit=u.K)

    # enable dimensionless input parameter (wavelength)
    _input_units_allow_dimensionless = True

    # input unit is a wavelength, assign unit equivalency
    input_units_equivalencies = {'x': u.spectral()}

    @property
    def input_units(self):
        return {'x': u.micron}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return OrderedDict([('subsolartemp', u.K),
                            ('diam', u.km)])

    def __init__(self, phys, eph):
        """phys should only refer to a single object, while eph can have
        multiple epochs"""
        ThermalClass.__init__(self, phys, eph)

        # assign model parameters from phys
        diam = self._apply_unit(self.phys['diam'], u.km)
        subsolartemp = self._apply_unit(self.phys['Tss'], u.K)

        Fittable1DModel.__init__(self, diam, subsolartemp)

    def evaluate(self, x, *args, jy=True):
        """Evaluate model at given wavelengths and return spectral flux densities.

        Parameters
        ----------
        model : integer
            Model identifier code for C code (1: STM, 2: FRM, 3: NEATM)
        wavelengths :`astropy.units` quantity or list
            Wavelengths at which to evaluate model. If a list of floats is
            provided, wavelengths must be in units of micron.
        jy : bool, optional
            If `True`, resulting flux densities will be returned in units of
            Janskys. If `False`, flux densities will be returned in SI units
            (W / (micron m2)). Default: `True`
        """

        _no_unit = False
        if not hasattr(x, 'unit'):
            _no_unit = True

        # set parameters and input as quantities
        x = self._apply_unit(x, u.micron)

        # use *args, if provided...
        if len(args) == 2:
            diam = self._apply_unit(args[0], u.km)
            subsolartemp = self._apply_unit(args[1], u.K,
                                            equiv=u.temperature())
        # or pull the parameters from self.phys
        else:
            diam = self._apply_unit(self.phys['diam'][0], u.km)
            subsolartemp = self._apply_unit(self.phys['Tss'][0], u.K)

        self.phys['diam'] = diam
        self.phys['Tss'] = subsolartemp

        flux = self._flux(2, x, jy)/np.pi

        if _no_unit:
            return flux.value
        else:
            return flux


class NEATM(ThermalClass):

    # define astropy.modeling.Fittable1DModel model parameters
    diam = Parameter(default=1, min=0, unit=u.km)
    bondalbedo = Parameter(default=0.05, min=1e-5, max=1,
                           unit=u.dimensionless_unscaled)
    eta = Parameter(default=1, min=0, max=10,
                    unit=u.dimensionless_unscaled)

    # enable dimensionless input parameter (wavelength)
    _input_units_allow_dimensionless = True

    # input unit is a wavelength, assign unit equivalency
    input_units_equivalencies = {'x': u.spectral()}

    @property
    def input_units(self):
        return {'x': u.micron}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return OrderedDict([('bondalbedo', u.dimensionless_unscaled),
                            ('diam', u.km),
                            ('eta', u.dimensionless_unscaled)])

    def __init__(self, phys, eph):
        """phys should only refer to a single object, while eph can have
        multiple epochs"""
        ThermalClass.__init__(self, phys, eph)

        # assign model parameters from phys
        diam = self._apply_unit(self.phys['diam'], u.km)
        bondalbedo = self._apply_unit(self.phys['bondalbedo'],
                                      u.dimensionless_unscaled)
        eta = self._apply_unit(self.phys['eta'], u.dimensionless_unscaled)

        Fittable1DModel.__init__(self, diam, bondalbedo, eta)

    def evaluate(self, x, *args, jy=True):
        """Evaluate model at given wavelengths and return spectral flux densities.

        Parameters
        ----------
        model : integer
            Model identifier code for C code (1: STM, 2: FRM, 3: NEATM)
        wavelengths :`astropy.units` quantity or list
            Wavelengths at which to evaluate model. If a list of floats is
            provided, wavelengths must be in units of micron.
        jy : bool, optional
            If `True`, resulting flux densities will be returned in units of
            Janskys. If `False`, flux densities will be returned in SI units
            (W / (micron m2)). Default: `True`
        """

        _no_unit = False
        if not hasattr(x, 'unit'):
            _no_unit = True

        # set parameters and input as quantities
        x = self._apply_unit(x, u.micron)

        # use *args, if provided...
        if len(args) == 3:
            diam = self._apply_unit(args[0], u.km)
            bondalbedo = self._apply_unit(args[1], u.dimensionless_unscaled)
            eta = self._apply_unit(args[2], u.dimensionless_unscaled)
        # or pull the parameters from self.phys
        else:
            diam = self._apply_unit(self.phys['diam'][0], u.km)
            bondalbedo = self._apply_unit(self.phys['bondalbedo'][0],
                                          u.dimensionless_unscaled)
            eta = self._apply_unit(self.phys['eta'][0],
                                   u.dimensionless_unscaled)

        self.phys['diam'] = diam
        self.phys['bondalbedo'] = bondalbedo
        self.phys['eta'] = eta
        self.phys['Tss'] = self.get_subsolartemp()

        print(diam, bondalbedo, eta)

        flux = self._flux(3, x, jy)/np.pi

        if _no_unit:
            return flux.value
        else:
            return flux


# TBD:
# def colorcorrection(bandpass, temp):
#    """Derives color correction function based on bandpass and
#    black body temperature; returns an interpolated function which
#    can be passed to `evaluate` function.
#    """

# TBD: reflected solar light treatment; either based on sbpy.photometry
# model or using a sub-class of ThermalClass
