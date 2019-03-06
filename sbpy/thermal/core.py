# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy Thermal Module

created on June 27, 2017
"""

import numpy as np

import astropy.units as u
from astropy import constants as const
from ._thermal import integrate_planck

__all__ = ['ThermalClass', 'STM', 'FRM', 'NEATM']


class ThermalClass():

    def __init__(self, phys, eph):
        """phys should only refer to a single object, while eph can have
        multiple epochs"""
        self.phys = phys
        self.eph = eph

    def subsolartemp(self):
        return (const.L_sun/(4*np.pi*const.au.to(u.m)**2) *
                (1.-self.phys['A']) /
                (self.eph['r'].to('au').value**2*self.phys['eta'] *
                 const.sigma_sb*self.phys['emissivity']))**0.25

    def _flux(self, model, wavelengths, jy=True):
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

        if jy:
            flux_unit = u.astrophys.Jy
        else:
            flux_unit = u.W/(u.micron*u.m**2)

        # make sure `wavelengths` is a list of floats in units of microns
        if isinstance(wavelengths, (float, int)):
            wavelengths = [wavelengths]*u.micron
        elif isinstance(wavelengths, u.Quantity):
            if wavelengths.isscalar:
                wavelengths = [wavelengths.to(u.micron)]*u.micron
        elif isinstance(wavelengths, (list, np.ndarray)):
            for i in range(len(wavelengths)):
                if isinstance(wavelengths[i], u.Quantity):
                    wavelengths[i] = wavelengths[i].to(u.micron)
            wavelengths *= u.micron

        return (self.phys['emissivity'] * self.phys['diam']**2 /
                self.eph['delta']**2 *
                np.pi * const.h * const.c**2 / wavelengths**5 *
                integrate_planck(
                    model, 0, np.pi/2, np.array(wavelengths),
                    self.subsolartemp().to('K').value,
                    self.eph['alpha'].to('deg').value)).to(
                        flux_unit,
                        equivalencies=u.spectral_density(wavelengths))

    def fit(self, eph):
        """Fit thermal model to observations stored in `sbpy.data.Ephem` instance

        Parameters
        ----------
        eph : `sbpy.data.Ephem` instance, mandatory
            provide object ephemerides and flux measurements

        Examples
        --------
        >>> from sbpy.thermal import STM
        >>> stmfit = STM.fit(eph) # doctest: +SKIP

        not yet implemented

        """


class STM(ThermalClass):

    def flux(self, wavelengths, jy=True):
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

        Examples
        --------
        >>> from astropy.time import Time
        >>> from astropy import units as u
        >>> from sbpy.thermal import STM
        >>> from sbpy.data import Ephem, Phys
        >>> epoch = Time('2019-03-12 12:30:00', scale='utc')
        # doctest: +REMOTE_DATA
        >>> eph = Ephem.from_horizons('2015 HW', location='568', epochs=epoch)
        >>> phys = PhysProp('diam'=0.3*u.km, 'pv'=0.3) # doctest: +SKIP
        >>> lam = np.arange(1, 20, 5)*u.micron # doctest: +SKIP
        >>> flux = STM.flux(phys, eph, lam) # doctest: +SKIP
        """

        return self._flux(1, wavelengths, jy)


class FRM(ThermalClass):

    def flux(self, wavelengths, jy=True):
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

        Examples
        --------
        >>> from astropy.time import Time
        >>> from astropy import units as u
        >>> from sbpy.thermal import STM
        >>> from sbpy.data import Ephem, Phys
        >>> epoch = Time('2019-03-12 12:30:00', scale='utc')
        # doctest: +REMOTE_DATA
        >>> eph = Ephem.from_horizons('2015 HW', location='568', epochs=epoch)
        >>> phys = PhysProp('diam'=0.3*u.km, 'pv'=0.3) # doctest: +SKIP
        >>> lam = np.arange(1, 20, 5)*u.micron # doctest: +SKIP
        >>> flux = STM.flux(phys, eph, lam) # doctest: +SKIP
        """

        return self._flux(2, wavelengths, jy)


class NEATM(ThermalClass):

    def __init__(self, *args):

        super().__init__(*args)

        from .. import bib
        bib.register('sbpy.thermal.NEATM', {'method': '1998Icar..131..291H'})

    def flux(self, wavelengths, jy=True):
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

        Examples
        --------
        >>> from astropy.time import Time
        >>> from astropy import units as u
        >>> from sbpy.thermal import STM
        >>> from sbpy.data import Ephem, Phys
        >>> epoch = Time('2019-03-12 12:30:00', scale='utc')
        # doctest: +REMOTE_DATA
        >>> eph = Ephem.from_horizons('2015 HW', location='568', epochs=epoch)
        >>> phys = PhysProp('diam'=0.3*u.km, 'pv'=0.3) # doctest: +SKIP
        >>> lam = np.arange(1, 20, 5)*u.micron # doctest: +SKIP
        >>> flux = STM.flux(phys, eph, lam) # doctest: +SKIP
        """

        return self._flux(3, wavelengths, jy)/np.pi
