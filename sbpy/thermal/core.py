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

    def _flux(lam):
        """Model flux density for a given wavelength `lam`, or a list/array thereof

        Parameters
        ----------
        phys : `sbpy.data.Phys` instance, mandatory
            provide physical properties
        eph : `sbpy.data.Ephem` instance, mandatory
            provide object ephemerides
        lam : `astropy.units` quantity or list-like, mandatory
            wavelength or list thereof

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

    def flux(self, lam):

        # check that wavelengths is list, not float, not int, not list of Quantities
        return (self.phys['emissivity'] * self.phys['diam']**2 /
                self.eph['delta']**2 *
                np.pi * const.h * const.c**2 / lam**5 *
                integrate_planck(1, 0, np.pi/2, np.array(lam),
                                 self.subsolartemp().to('K').value,
                                 self.eph['alpha'].to('deg').value)).to(
            #            u.W/(u.micron*u.m**2),
            #            equivalencies=u.spectral_density(lam))
                                     u.astrophys.Jy,
                                     equivalencies=u.spectral_density(lam))


class FRM(ThermalClass):
    pass


class NEATM(ThermalClass):
    def __init__(self):
        from .. import bib
        bib.register('sbpy.thermal.NEATM', {'method': '1998Icar..131..291H'})
