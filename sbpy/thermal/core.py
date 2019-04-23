# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy Thermal Module

created on June 27, 2017
"""

import os
import numpy as np
from collections import OrderedDict
import warnings

import astropy.units as u
from astropy import constants as const
from astropy.modeling import fitting, Fittable1DModel, Parameter
from astropy.table import vstack
from astropy.utils.exceptions import AstropyUserWarning

from .. import bib
from ..data import Ephem, Phys
from ._thermal import integrate_planck


__all__ = ['ThermalClass', 'STM', 'FRM', 'NEATM']


class ThermalWarning(AstropyUserWarning):
    """Warning related to `~sbpy.thermal` functionality."""


class ThermalError(Exception):
    """Exception related to `~sbpy.thermal` functionality."""


class ThermalClass(Fittable1DModel):
    """Base class for thermal modeling. Objects instantiated from this
    class can be used to estimate thermal flux densities or fit thermal
    flux observations based on the target diameter and subsolar temperatures.
    Other properties like geometric albedo and the beaming parameter
    :math:`\eta` can be derived from the latter.

    All thermal models are derived from this class: `~sbpy.thermal.STM`,
    `~sbpy.thermal.FRM`, and `~sbpy.thermal.NEATM`. The actual model
    model implementations can be found as C extensions in
    `~sbpy.thermal._thermal`; the implementations are accessible through
    `~sbpy.thermal._thermal.integrate_planck`.
    """

    # define astropy.modeling.Fittable1DModel model parameters:
    # target diameter
    diam = Parameter(default=1, min=0, unit=u.km)
    # subsolar temperature normalized to heliocentric distance of 1 au
    subsolartemp_au = Parameter(default=150, min=0, unit=u.K)

    # enable dimensionless input parameter (wavelength)
    _input_units_allow_dimensionless = True

    # input unit is a wavelength, assign unit equivalency
    input_units_equivalencies = {'x': u.spectral()}

    @property
    def input_units(self):
        """Define units of input data."""
        return {'x': u.micron}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        """Defines units of model parameters."""
        return OrderedDict([('subsolartemp_au', u.K),
                            ('diam', u.km)])

    def __init__(self, *pargs):
        """A `ThermalClass` object can be instantiated with or without
        keyword arguments as follows. If no arguments are provided, the
        ``from_data`` method can be used to instantiate an object. If
        ``phys`` and ``ephem`` are passed to the constructor, they are
        stored internally as ``self.phys`` and ``self.ephem``,
        respectively.

        Parameters
        ----------
        phys : `~sbpy.data.Phys` object, optional
            Defines the known physical properties of the
            target body. ``phys`` is only allowed to
            describe a single target object, hence, the underlying data
            table can only have a single row. `~sbpy.data.Phys` properties
            that are acceptable are: ``diam`` (target
            diameter), ``pv`` (geometric albedo), ``bondalbedo``
            (Bond albedo), ``emissivity``, ``absmag`` (visual
            absolute magnitude), and ``ir_v_reflectance``
            (IR/V reflectance). Refer to the documentation of methods for
            flux estimation and thermal model fitting for lists of
            required fields.
        ephem : `~sbpy.data.Ephem` object, optional
            Defines the observational circumstances for the
            target body. `~sbpy.data.Ephem` properties of the target that
            are relevant to thermal modeling are: ``heliodist``
            (heliocentric distance), ``obsdist`` (distance to the observer),
            ``phaseangle`` (solar phase angle), ``wavelength``
            (wavelength of observation), ``flux`` (measured flux
            density at a given wavelength), ``fluxerr`` (error of the
            flux density measurement), and ``subsolartemp`` (subsolar
            temperature). Refer to the documentation of methods for
            flux estimation and thermal model fitting for lists of
            required fields.

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
                if len(parg) > 1:
                    parg._table = parg.table[0]
                    warnings.warn(
                        ("'phys' object has a length of {}, but currently "
                         "only single-row objects are supported. All rows "
                         "following the first row will be ignored.").format(
                             len(self.phys)), ThermalWarning)
                # create copies of input objects to not affect them outside
                # this namespace
                self.phys = Phys.from_table(parg.table)
            if isinstance(parg, Ephem):
                self.ephem = Ephem.from_table(parg.table)

        # assume emissivity=0.9 (Harris and Lagerros 2002) if none provided
        try:
            self.phys['emissivity']
        except KeyError:
            self.phys.add_column([0.9]*max([len(self.phys), 1]),
                                 'emissivity')

    @property
    def get_phys(self):
        """Returns full physical properties object ``self.phys``."""
        return self.phys

    @property
    def get_ephem(self):
        """Returns full ephemerides object ``self.ephem``."""
        return self.ephem

    def calculate_subsolartemp(self, append_results=False):
        """Calculate the subsolar temperature based on the target's physical
        properties and observational circumstances.

        Parameters
        ----------
        append_results : bool, optional
            If `append_results=True`, results from this function are
            appended to `self.ephem` (as the subsolar temperature depends
            upon the heliocentric distance and is hence variable) and a
            copy of `self.ephem` is returned;
            if `append_results=False`,
            subsolar temperatures for each epoch in `self.ephem` are simply
            returned as a `~astropy.units.Quantity` object. Default: `False`

        Notes
        -----
        This method requires the following fields in ``self.phys``:
           * ``bondalbedo``: target's Bond albedo
           * ``eta``: infrared beaming parameter (see Harris 1998)
           * ``emissivity``: target emissivity

        This method requires the following fields in ``self.ephem``:
           * ``heliodist``: heliocentric distance of the target

        Returns
        -------
        `~astropy.units.Quantity` or `~sbpy.data.Ephem` object depending on
        `append_results`.
        """

        subsolartemp = (const.L_sun/(4*np.pi*const.au.to(u.m)**2) *
                        (1.-self.phys['bondalbedo']) /
                        (self.ephem['heliodist'].to('au').value**2 *
                         self.phys['eta'] *
                         const.sigma_sb*self.phys['emissivity']))**0.25

        if append_results:
            self.ephem._table['subsolartemp'] = subsolartemp
            return Ephem.from_table(self.ephem.table)
        else:
            return subsolartemp

    def calculate_albedo(self, append_results=False):
        """Calculate the geometric albedo and the Bond albedo based on
        the target's physical properties.

        Parameters
        ----------
        append_results : bool, optional
            If `append_results=True`, results from this function are
            appended to `self.phys` and a copy of `self.phys` is returned;
            if `append_results=False`,
            the geometric albedo and the Bond albedo are returned as as
            a separate `~sbpy.data.Phys` object. Default: `False`

        Notes
        -----
        This method requires the following fields in ``self.phys``:
           * ``diam``: target's diameter
           * ``absmag``: target's absolute magnitude (V-band)
           * ``slopepar``: target's photometric slope parameter TBD


        Returns
        -------
        `~sbpy.data.Phys` object.

        """

        pv = ((1329/self.phys['diam'].to('km').value *
               10**(-self.phys['absmag'].to('mag').value/5))**2)

        # replace this in the future with functionality from
        # sbpy.photometry
        bondalbedo = ((0.29+0.684*self.phys['slopepar'])*pv)

        if append_results:
            self.phys['pv'] = pv
            self.phys['bondalbedo'] = bondalbedo
            return Phys.from_table(self.phys.table)
        else:
            return Phys.from_dict({'pv': pv,
                                   'bondalbedo': bondalbedo})

    def eta_from_subsolartemp(self, append_results=False):
        """Calculate the beaming parameter :math:`\eta` from the target's
        subsolar temperature, its physical properties, and observational
        circumstances.

        Parameters
        ----------
        append_results : bool, optional
            If `append_results=True`, results from this function are
            appended to `self.phys` and a copy of `self.phys` is returned;
            if `append_results=False`, the beaming parameter :math:`\eta`
            is returned as as a `~astropy.units.Quantity` object. Default:
            `False`

        Notes
        -----
        This method requires the following fields in ``self.phys``:
           * ``bondalbedo``: target's Bond albedo
           * ``emissivity``: target's emissivity

        This method requires the following fields in ``self.ephem``:
           * ``subsolartemp``: target's subsolar temperature per epoch
           * ``heliodist``: target's heliocentric distance per epoch

        Returns
        -------
        `~astropy.units.Quantity` or `~sbpy.data.Phys` object.

        """

        self.calculate_albedo(append_results=True)

        eta = (const.L_sun/(4*np.pi*const.au.to(u.m)**2) *
               (1.-self.phys['bondalbedo']) /
               (self.ephem['heliodist'].to('au').value**2 *
                self.ephem['subsolartemp']**4 *
                const.sigma_sb*self.phys['emissivity']))
        eta = np.mean(eta)

        if append_results:
            self.phys['eta'] = eta
            return Phys.from_table(self.phys.table)
        else:
            return eta

    def calculate_flux(self, lam, jy=True):
        """Wrapper method to calculate flux density estimates.

        Parameters
        ----------
        lam : `~astropy.units.Quantity` or sequence of floats
           The wavelengths at which to evaluate the thermal model and
           estimate the thermal flux densities. The length of ``lam`` has
           to be the same as the length of ``self.ephem``. If a
           `~astropy.units.Quantity` object is provided, each element of
           ``lam`` is applied to the corresponding element of
           ``self.ephem``. If a sequence is provided (a nested list),
           the sequence is used to expand ``self.ephem`` (see
           `~sbpy.data.DataClass.expand` for a discussion).
           jy : bool, optional
           Flux density units to be used: if ``True``,
           Janskys are used; if ``False``, units of
           :math:`W m^{-1} {\mu}m^{-1}` are used. Default: ``True``

        Notes
        -----
        This method requires the following fields in ``self.phys``:
           * ``diam``: target diameter
           * ``emissivity``: target emissivity

        If ``subsolartemp`` is not present in ``self.phys`` it is calculated
        using `~sbpy.thermal.ThermalClass.calculate_subsolartemp`.

        This method requires the following fields in ``self.ephem``:
           * ``heliodist``: heliocentric distance of the target
           * ``obsdist``: distance of the target from the observer
           * ``phaseangle``: solar phase angle (only required for
             `~sbpy.thermal.NEATM`)

        Resulting flux densities are added as columns
        ``thermal_lam`` and ``thermal_flux`` to ``self.ephem``.

        Returns
        -------
        `~sbpy.data.Ephem` object that is a copy of ``self.ephem``,
        containing ``thermal_lam`` and ``therml_flux``.
        """

        # calculate subsolar temperature if not provided
        try:
            self.ephem['subsolartemp']
        except KeyError:
            self.calculate_subsolartemp(append_results=True)

        if not isinstance(lam, (list, tuple, np.ndarray)):
            lam = [lam]

        if len(lam) != len(self.ephem):
            # apply sequence of wavelengths to every single epoch
            lam = [lam]*len(self.ephem)

        # reshape self.ephem and lam for self.evaluate
        self.ephem.expand(lam, 'thermal_lam')
        lam = self.ephem['thermal_lam']

        # call abstract `evaluate` method
        flux = self.evaluate(lam, jy=jy)

        try:
            self.ephem.add_column(lam, 'thermal_lam')
        except ValueError:
            pass
        self.ephem.add_column(flux, 'thermal_flux')

        return self.ephem

    def _flux(self, model, wavelengths, jy=True):
        """Internal low-level method to evaluate model at given
         wavelengths and return spectral flux densities. Users are advised to
         use the high-level
         function `~sbpy.thermal.ThermalClass.calculate_flux` instead.

         Parameters
         ----------
         model : integer
            Model identifier code for C code (1: STM, 2: FRM, 3: NEATM)
            wavelengths : `~astropy.units.Quantity` object or list
            Wavelengths at which to evaluate model. If a list of floats is
            provided, wavelengths must be in units of micron. Must have the
            same length as ``self.ephem``.
         jy : bool, optional
            Flux density units to be used: if ``True``,
            Janskys are used; if ``False``, units of
            :math:`W m^{-1} {\mu}m^{-1}` are used. Default: ``True``
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
        """Utility function to enforce Quantity nature for object"""
        if isinstance(value, u.Quantity):
            return value.to(unit, equivalencies=equiv)
        else:
            return u.Quantity(value, unit)

    def fit(self, init_diam=1*u.km, init_subsolartemp=150*u.K,
            fitter=fitting.SLSQPLSQFitter()):
        """Fit this `~sbpy.thermal.ThermalClass` object to observational data
        via the target diameter and its subsolar temperature.

        Parameters
        ----------
        init_diam: `~astropy.units.Quantity`, optional
           Initial guess for the target's diameter. Default: ``1*u.km``
        init_subsolartemp: `~astropy.units.Quantity`, optional
           Initial guess for the target's mean subsolar temperature across
           all epochs in ``self.ephem``. Default: ``150*u.K``
        fitter: `~astropy.modeling.fitting` method, optional
           Fitting method to be utilized. Default:
           `~astropy.modeling.fitting.SLSQPLSQFitter()`

        Notes
        -----
        If ``self.phys`` does not contain ``diam``, this field is
        created using the value of ``init_diam``. If ``self.ephem``
        does not contain ``subsolartemp``, this field is created using the
        value of ``init_subsolartemp``.

        In order to account for the target's potentially different
        heliocentric distances as provided by ``self.ephem``, the subsolar
        temperature divided by the square-root of the respective
        heliocentric distance is actually used as a fitting parameter.
        """

        # initialize parameters if not yet done
        try:
            self.phys['diam']
        except KeyError:
            self.phys.table['diam'] = self._apply_unit(init_diam, u.km)
        try:
            self.ephem['subsolartemp']
        except KeyError:
            self.ephem._table['subsolartemp'] = self._apply_unit(
                init_subsolartemp, u.K)

        # initialize Fittable1DModel
        diam = self._apply_unit(self.phys['diam'][0], u.km)
        subsolartemp_au = self._apply_unit(
            np.mean(self.ephem['subsolartemp'] *
                    np.sqrt(self.ephem['heliodist'].to('au').data)), u.K)
        Fittable1DModel.__init__(self, diam, subsolartemp_au)

        # extract wavelengths and fluxes from self.ephem
        lam = self.ephem['thermal_lam']
        flux = self.ephem['thermal_flux']

        fit = fitter(self, lam, flux)

        self.phys._table['diam'] = fit.diam
        self.ephem._table['subsolartemp'] = (
            fit.subsolartemp_au /
            np.sqrt(self.ephem['heliodist'].to('au').data))

        self.ephem._table['thermal_flux_fit'] = self.evaluate(lam)

    @classmethod
    def from_data(cls, phys, ephem, *pargs, **kwargs):
        """Create a `~sbpy.thermal.ThermalClass` object by fitting
        the corresponding thermal model to observations provided
        through ``ephem``."""

        self = cls(phys, ephem)
        self.fit(*pargs, **kwargs)

        return self


class STM(ThermalClass):
    """Implementation of the Standard Thermal Model(STM) as defined by
    `Morrison and Lebofsky(1979)
    <https://ui.adsabs.harvard.edu/abs/1979aste.book..184M/abstract>`_
    and `Lebofsky et al. (1986)
    <https://ui.adsabs.harvard.edu/abs/1986Icar...68..239L/abstract>`_.

    This class derives from both `~sbpy.thermal.ThermalClass` and
    `~astropy.modeling.Fittable1DModel`. Fitting parameters are the target
    diameter(``self.phys['diam']``) and the target subsolar temperature
    (``self.ephem['subsolartemp']``).
    """

    def __init__(self, *pargs):
        """If a `~sbpy.data.Phys` object is provided for initiation but
        does not contain a beaming parameter ``eta``,
        : math: `\eta = 1` is assumed."""

        ThermalClass.__init__(self, *pargs)

        # assume eta=1.0 if none is provided
        try:
            self.phys['eta']
        except KeyError:
            self.phys.add_column([1.0]*max([len(self.phys), 1]),
                                 'eta')

        # register STM references
        bib.register('sbpy.thermal.STM',
                     {'method': ['1979aste.book..184M',
                                 '1986Icar...68..239L']})

    def evaluate(self, x, *args, jy=True):
        """Internal method to evaluate the underlying model. Users should
        not call this method direclty, but instead call
        `~sbpy.thermal.ThermalClass.calculate_flux` to
        obtain a flux density estimate.

        Parameters
        ----------
        x: `astropy.units` quantity, `~numpy.ndarray`, or list
           Wavelengths at which to evaluate model. If a list of floats is
           provided, wavelengths must be in units of micron.
        jy: bool, optional
           Flux density units to be used: if ``True``,
           Janskys are used; if ``False``, units of
           :math: `W m ^ {-1} {\mu}m ^ {-1}` are used. Default: ``True``

        Notes
        -----
        This method requires the following fields in ``self.phys``:
           * ``diam``: target diameter
           * ``emissivity``: target emissivity

        This method requires the following fields in ``self.ephem``:
           * ``heliodist``: heliocentric distance of the target
           * ``obsdist``: distance of the target from the observer
           * ``subsolartemp``: subsolar temperature for each epoch
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

    @classmethod
    def from_data(cls, phys, ephem, *pargs, **kwargs):
        """Create a `~sbpy.thermal.ThermalClass` object by fitting
        the corresponding thermal model to observations provided
        through ``ephem``."""

        self = ThermalClass.from_data(*pargs, **kwargs)

        # derive albedos based on subsolartemp and eta

        return self


class FRM(ThermalClass):
    """Implementation of the Fast Rotating Model(FRM) as defined by
    `Lebofsky and Spencer(1989)
    <https://ui.adsabs.harvard.edu/abs/1989aste.conf..128L/abstract>`_.

    This class derives from both `~sbpy.thermal.ThermalClass` and
    `~astropy.modeling.Fittable1DModel`. Fitting parameters are the target
    diameter(``self.phys['diam']``) and the target subsolar temperature
    (``self.ephem['subsolartemp']``).
    """

    def __init__(self, *pargs):
        """If a `~sbpy.data.Phys` object is provided for initiation but
         does not contain a beaming parameter ``eta``,
         :math:`\eta =\pi` is assumed."""

        ThermalClass.__init__(self, *pargs)

        # assume eta=pi if none is provided
        try:
            self.phys['eta']
        except KeyError:
            self.phys.add_column([np.pi]*max([len(self.phys), 1]),
                                 'eta')

        # register FRM references
        bib.register('sbpy.thermal.FRM', {'method': '1989aste.conf..128L'})

    def evaluate(self, x, *args, jy=True):
        """Internal method to evaluate the underlying model. Users should
        not call this method direclty, but instead call
        `~sbpy.thermal.ThermalClass.calculate_flux` to
        obtain a flux density estimate.

        Parameters
        ----------
        x: `astropy.units` quantity, `~numpy.ndarray`, or list
           Wavelengths at which to evaluate model. If a list of floats is
           provided, wavelengths must be in units of micron.
        jy: bool, optional
           Flux density units to be used: if ``True``,
           Janskys are used; if ``False``, units of
           :math: `W m ^ {-1} {\mu}m ^ {-1}` are used. Default: ``True``

        Notes
        -----
        This method requires the following fields in ``self.phys``:
           * ``diam``: target diameter
           * ``emissivity``: target emissivity

        This method requires the following fields in ``self.ephem``:
           * ``heliodist``: heliocentric distance of the target
           * ``obsdist``: distance of the target from the observer
           * ``subsolartemp``: subsolar temperature for each epoch
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
        flux = self._flux(2, x, jy=jy)/np.pi

        if _no_unit:
            return flux.value
        else:
            return flux


class NEATM(ThermalClass, Fittable1DModel):
    """Implementation of the Near-Earth Asteroid Thermal Model(NEATM) as
    defined by`Harris(1998)
    <https://ui.adsabs.harvard.edu/link_gateway/1998Icar..131..291H/doi:10.1006/icar.1997.5865>`_. This
    is the fixed-:math:`\eta` version of this model.

    This class derives from both `~sbpy.thermal.ThermalClass` and
    `~astropy.modeling.Fittable1DModel`. Fitting parameters are the target
    diameter(``self.phys['diam']``) and the target Bond albedo
    (``self.phys['bondalbedo']``).

    """

    def __init__(self, *pargs):
        """If a `~sbpy.data.Phys` object is provided for initiation but
        does not contain a beaming parameter ``eta``,
        :math: `\eta = 1` is assumed."""

        ThermalClass.__init__(self, *pargs)

        # assume eta=1 if none is provided
        try:
            self.phys['eta']
        except KeyError:
            self.phys.add_column([1.0]*max([len(self.phys), 1]),
                                 'eta')

    def evaluate(self, x, *pargs, jy=True):
        """Internal method to evaluate the underlying model. Users should
        not call this method direclty, but instead call
        `~sbpy.thermal.ThermalClass.calculate_flux` to
        obtain a flux density estimate.

        Parameters
        ----------
        x: `astropy.units` quantity, `~numpy.ndarray`, or list
            Wavelengths at which to evaluate model. If a list of floats is
            provided, wavelengths must be in units of micron.
        jy: bool, optional
            Flux density units to be used: if ``True``,
            Janskys are used; if ``False``, units of
            : math: `W m ^ {-1} {\mu}m ^ {-1}` are used. Default: ``True``

        Notes
        -----
        This method requires the following fields in ``self.phys``:
           * ``diam``: target diameter
           * ``emissivity``: target emissivity

        This method requires the following fields in ``self.ephem``:
           * ``heliodist``: heliocentric distance of the target
           * ``obsdist``: distance of the target from the observer
           * ``subsolartemp``: subsolar temperature for each epoch
        """
        _no_unit = False
        if not hasattr(x, 'unit'):
            _no_unit = True

        # set parameters and input as quantities
        x = self._apply_unit(x, u.micron)

        # use pargs, if provided...
        if len(pargs) == 2:
            diam = self._apply_unit(pargs[0], u.km)
            subsolartemp_au = self._apply_unit(pargs[1][0], u.K,
                                               equiv=u.temperature())

            self.phys._table['diam'] = diam
            self.ephem._table['subsolartemp'] = (
                subsolartemp_au /
                np.sqrt(self.ephem['heliodist'].to('au').data))
            # or pull the parameters from self.phys
        else:
            diam = self._apply_unit(self.phys['diam'][0], u.km)
            subsolartemp_au = self._apply_unit(
                np.mean(self.ephem['subsolartemp'] *
                        np.sqrt(self.ephem['heliodist'].to('au').data)),
                u.K)

        flux = self._flux(3, x, jy)/np.pi

        if _no_unit:
            return flux.value
        else:
            return flux
