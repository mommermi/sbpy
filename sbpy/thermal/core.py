# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy Thermal Module

created on June 27, 2017
"""

import os
import numpy as np

import astropy.units as u
from astropy import constants as const
from ._thermal import integrate_planck
from ..data import Ephem, Phys
from astropy.modeling import fitting, Fittable1DModel, Parameter
from astropy.table import vstack

from collections import OrderedDict

__all__ = ['ThermalClass', 'STM', 'FRM', 'NEATM']


class ThermalClass():
    """Base class for thermal modeling. Objects instantiated from this
    class can be used to estimate thermal flux densities or fit thermal
    flux observations to obtain target diameter and albedo estimates."""

    def __init__(self, *pargs):
        """A `ThermalClass` object can be instantiated with or without
        keyword arguments as follows. If no arguments are provided, the
        `from_data` method can be used to instantiate an object.

        Parameters
        ----------
        phys : `~sbpy.data.Phys` object, optional
            This object describes the known physical properties of the
            target body. Note that currently ``phys`` is only allowed to
            describe a single target object, hence, the underlying data
            table can only have a single row. `~sbpy.data.Phys` properties
            that are relevant to thermal modeling are: ``diam`` (target
            diameter), ``pv`` (geometric albedo), ``bolomalbedo``
            (Bolometric albedo), ``emissivity``, ``absmag`` (visual
            absolute magnitude), and ``ir_v_reflectance``
            (IR/V reflectance). Refer to
            `evaluate` and `from_data` for a list of required properties
            for flux estimation and thermal model fitting, respectively.
        ephem : `~sbpy.data.Ephem` object, optional
            This object describes the observational circumstances for the
            target body. `~sbpy.data.Ephem` properties of the target that
            are relevant to thermal modeling are: ``heliodist``
            (heliocentric distance), ``obsdist`` (distance to the observer),
            ``phaseangle`` (solar phase angle), ``wavelength``
            (wavelength of observation), ``flux`` (measured flux
            density at a given wavelength), ``fluxerr`` (error of the
            flux density measurement), and ``subsolartemp`` (subsolar
            temperature). Refer to `evaluate` and
            `from_data` for a list of relevant and required properties
            for flux estimation and thermal model fitting, respectively.

        Notes
        -----
        * If no emissivity is provided as part of ``phys``, an emissivity
          of 0.9 is assumed.

        """

        # create empty Phys and Ephem objects if none are provided
        self.phys = Phys.from_dict({})
        self.ephem = Phys.from_dict({})

        for parg in pargs:
            if isinstance(parg, Phys):
                self.phys = Phys.from_table(parg.table)
            if isinstance(parg, Ephem):
                self.ephem = parg

        # assume emissivity=0.9 (Harris and Lagerros XXX) if none is provided
        try:
            self.phys['emissivity']
        except KeyError:
            self.phys.add_column([0.9]*max([len(self.phys), 1]),
                                 'emissivity')

        # # # assume ir_v_reflectance=1.4 (XXX) if not provided
        # # try:
        # #     phys['ir_v_reflectance']
        # # except KeyError:
        # #     phys.add_column([1.4]*max([len(phys), 1]), 'ir_v_reflectance')

        # # fill other properties with nan if not provided
        # for field in ['diam', 'pv', 'bondalbedo', 'subsolartemp']:
        #     try:
        #         phys[field]
        #     except KeyError:
        #         phys.add_column([np.nan]*max([len(phys), 1]), field)

        # self.phys = phys.table
        # self.eph = eph.table

    @property
    def get_phys(self):
        return self.phys

    @property
    def get_ephem(self):
        return self.ephem

    def calculate_subsolartemp(self):
        """Calculate the subsolar temperature based on the physical
        properties and observational circumstances."""

        subsolartemp = (const.L_sun/(4*np.pi*const.au.to(u.m)**2) *
                        (1.-self.phys['bondalbedo']) /
                        (self.ephem['heliodist'].to('au').value**2 *
                         self.phys['eta'] *
                         const.sigma_sb*self.phys['emissivity']))**0.25

        self.ephem.add_column(subsolartemp, 'subsolartemp')

    def flux(self, lam, jy=True):
        """Wrapper for flux density estimation."""

        # calculate subsolar temperature if not provided
        try:
            self.ephem['subsolartemp']
        except KeyError:
            self.calculate_subsolartemp()

        if not isinstance(lam, (list, tuple, np.ndarray)):
            lam = [lam]

        if len(lam) != len(self.ephem):
            # apply sequence of wavelengths to every single epoch
            lam = [lam]*len(self.ephem)

        # reshape self.ephem and lam for self.evaluate
        self.ephem.expand(lam, 'thermal_lam')
        lam = self.ephem['thermal_lam'].data

        flux = self.evaluate(lam, jy=jy)

        try:
            self.ephem.add_column(lam, 'thermal_lam')
        except ValueError:
            pass
        self.ephem.add_column(flux, 'thermal_flux')

    def _flux(self, model, wavelengths, jy=True):
        """Internal lowe-level method to evaluate model at given
        wavelengths and return spectral flux densities. Use higher-level
        function ``self.flux`` instead.

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

        # flux.shape: [len(wavelengths), len(self.ephem)]
        flux = (self.phys['emissivity'][0] * self.phys['diam'][0]**2 /
                self.ephem['delta']**2 *
                np.pi * const.h * const.c**2 /
                wavelengths**5)*integrate_planck(
                    model, 0, np.pi/2, np.array(wavelengths),
                    self.ephem['subsolartemp'].to('K').value,
                    self.ephem['alpha'].to('deg').value)

        # convert flux densities to target unit
        # flux_converted.shape: [len(self.ephem), len(wavelengths)]
        flux_converted = self._apply_unit(
            flux.to(flux_unit,
                    equivalencies=u.spectral_density(wavelengths)),
            flux_unit)

        return flux_converted

    @staticmethod
    def _apply_unit(value, unit, equiv=None):
        """utility function to enforce Quantity nature for object"""
        if isinstance(value, u.Quantity):
            return value.to(unit, equivalencies=equiv)
        else:
            return u.Quantity(value, unit)


class STM(ThermalClass, Fittable1DModel):

    # define astropy.modeling.Fittable1DModel model parameters:
    # - target diameter:
    diam = Parameter(default=1, min=0, unit=u.km)
    # - subsolar temperature at 1 au (to make it a scalar value)
    subsolartemp_au = Parameter(default=150, min=0, unit=u.K)

    # enable dimensionless input parameter (wavelength)
    _input_units_allow_dimensionless = True

    # input unit is a wavelength, assign unit equivalency
    input_units_equivalencies = {'x': u.spectral()}

    @property
    def input_units(self):
        return {'x': u.micron}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return OrderedDict([('subsolartemp_au', u.K),
                            ('diam', u.km)])

    def __init__(self, *pargs):

        ThermalClass.__init__(self, *pargs)

        # assume eta=1.0 if none is provided
        try:
            self.phys['eta']
        except KeyError:
            self.phys.add_column([1.0]*max([len(self.phys), 1]),
                                 'eta')

    def from_data(self):

        # initialize parameters if not yet done
        try:
            self.phys['diam']
        except KeyError:
            self.phys.table['diam'] = [1]*u.km
        try:
            self.ephem['subsolartemp']
        except KeyError:
            self.calculate_subsolartemp()

        # initialize Fittable1DModel
        diam = self._apply_unit(self.phys['diam'][0], u.km)
        subsolartemp_au = self._apply_unit(
            np.mean(self.ephem['subsolartemp'] *
                    np.sqrt(self.ephem['heliodist'].to('au').data)), u.K)
        Fittable1DModel.__init__(self, diam, subsolartemp_au)

        # extract wavelengths and fluxes from self.ephem
        lam = self.ephem['thermal_lam']
        flux = self.ephem['thermal_flux']

        fitter = fitting.LevMarLSQFitter()

        fit = fitter(self, lam, flux)

        self.phys._table['diam'] = fit.diam
        self.ephem._table['subsolartemp'] = (
            fit.subsolartemp_au /
            np.sqrt(self.ephem['heliodist'].to('au').data))

    def evaluate(self, x, *args, jy=True):
        """Use this method only internally!!!!!
        Evaluate model at given wavelengths and return spectral flux
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

        Notes
        -----
        In order for this method to run, ``self.ephem`` and ``x`` have to
        have the same shape and ``x`` has to be flat.
        """

        _no_unit = False
        if not hasattr(x, 'unit'):
            _no_unit = True

        # set parameters and input as quantities
        x = self._apply_unit(x, u.micron)

        # use *args, if provided...
        if len(args) == 2:
            diam = self._apply_unit(args[0][0], u.km)
            subsolartemp_au = self._apply_unit(args[1][0], u.K,
                                               equiv=u.temperature())

            self.phys._table['diam'] = diam
            self.ephem._table['subsolartemp'] = (
                subsolartemp_au /
                np.sqrt(self.ephem['heliodist'].to('au').data))

        # or pull the parameters from self.phys
        else:
            diam = self._apply_unit(self.phys['diam'], u.km)
            subsolartemp_au = self._apply_unit(
                np.mean(self.ephem['subsolartemp'] *
                        np.sqrt(self.ephem['heliodist'].to('au').data)),
                u.K)

        # calculate flux densities
        flux = self._flux(1, x, jy=jy)

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

        # assign model parameters
        diam = self._apply_unit(self.phys['diam'], u.km)
        subsolartemp = self._apply_unit(self.phys['subsolartemp'], u.K)

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
            subsolartemp = self._apply_unit(self.phys['subsolartemp'][0],
                                            u.K)

        self.phys['diam'] = diam
        self.phys['subsolartemp'] = subsolartemp

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
        self.phys['subsolartemp'] = self.get_subsolartemp()

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
