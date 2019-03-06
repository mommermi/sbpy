import _thermalmodels
import numpy as np
import astropy.units as u
from astropy.constants import astropyconst13 as const
from scipy import integrate
import timeit

# python STM implementation


def A_from_G_and_pv(G, pv):
    """ calculate bond albedo from slope parameter and geometric albedo"""
    return (0.29 + 0.684*G)*pv


# W/(m**2 * K**4) --> same as neosurvey_model.c
stefan_boltzmann_constant = 5.67e-8
kb = 1.381e-23                        # Boltzmann constant, m^2*kg / (K*s^2)
plancks_constant = 6.633e-24          # J*s
speed_of_light = 299792458            # m/s, taken from neosurvey_model.c

# h*c^2 in W m^2 (=/1e-24), taken from neosurvey_model.c
HC2 = 5.95521967e7
# h*c/kb using um in c --> can keep lambda in um, taken from neosurvey_model.c
HCK = 14387.6866

# solar_constant = 1361.               # W/(m**2), Kopp and Lean 2011
SOLARCONST = 1367.                    # W/(m**2), taken from neosurvey_model.c

AU = 149597870.691                    # Astronomical Units in kilometers

G = 0.15                              # slope parameter
pv = 0.2                              # geometric albedo
# bond albedo, function called from first_model.py
A = A_from_G_and_pv(G, pv)
lam = 5.                              # lambda in units of micro meters --> um
# bolometric emissivity, equals 1 for black body, less for real "grey" body
emissivity = 0.9
eta = 1.                              # "fudge factor", unitless
# zero degree angle, written as alpha, equivalent to "full moon"
phase_angle = 0
d = 0.11831                           # diameter in km
Delta = 0.1                           # geocentric distance in Au
r = 1.1                               # heliocentric distance in Au


def Tss(A, r, eta, emissivity):
    # calculates sub-solar temperature with "beaming parameter", eta
    return (SOLARCONST*(1.-A) / (r**2. * eta*stefan_boltzmann_constant*emissivity))**0.25


subsolartemp = Tss(A, r, eta, emissivity)

print(3.335640952e14 * lam**2 * emissivity*d**2.*np.pi*HC2 /
      ((Delta*AU)**2.*lam**5.) *
      _thermalmodels.integrate_planck(1, 0, np.pi/2, lam, subsolartemp, 0))

print(3.335640952e14 * lam**2 * emissivity*d**2.*np.pi*HC2 /
      ((Delta*AU)**2.*lam**5.) *
      _thermalmodels.integrate_planck(2, 0, np.pi/2, lam, subsolartemp, 0))

print(3.335640952e14 * lam**2 * emissivity*d**2.*HC2 /
      ((Delta*AU)**2.*lam**5.) *
      _thermalmodels.integrate_planck(3, 0, np.pi/2, lam, subsolartemp, 0))


for lam in np.arange(5, 100, 0.1):
    print(lam, (3.335640952e14 * lam**2 * emissivity*d**2.*HC2 /
                ((Delta*AU)**2.*lam**5.) *
                _thermalmodels.integrate_planck(3, 0, np.pi/2, lam, subsolartemp, 0)))
