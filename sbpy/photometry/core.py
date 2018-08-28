# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy Photometry Module

created on June 23, 2017
"""

__all__ = ['DiskIntegratedModelClass', 'HG', 'HG12', 'HG1G2',
           'DiskFunctionModel', 'LommelSeeliger', 'Lambert', 'LunarLambert',
           'PhaseFunctionModel', 'ROLOPhase',
           'ResolvedPhotometricModelClass', 'ROLO']

import numpy as np
from scipy.integrate import quad
from astropy.modeling import FittableModel, Fittable1DModel, Fittable2DModel, Parameter
from astropy import units


def ref2mag(ref, radius, M_sun=None):
    """Convert average bidirectional reflectance to reduced magnitude

    Parameters
    ----------
    ref : num, astropy.units.Quantity
        Average bidirectional reflectance
    radius : num, astropy.units.Quantity
        Radius of object
    M_sun : number, optional
        The magnitude of the Sun, default is -26.74

    Returns
    -------
    The same type as input, reduced magnitude (at r=delta=1 au)

    Examples
    --------
    >>> from astropy import units as u
    >>> mag = ref2mag(0.1, 460) # doctest: +SKIP
    >>> print(mag) # doctest: +SKIP
    >>> mag = ref2mag(0.1, 460*u.km) # doctest: +SKIP
    >>> print(mag) # doctest: +SKIP
    """

    if M_sun is None:
        M_sun = -26.74
    Q = False
    if isinstance(ref, units.Quantity):
        ref = ref.value
        Q = True
    if isinstance(radius, units.Quantity):
        radius = radius.to('km').value
        Q = True
    if isinstance(M_sun, units.Quantity):
        M_sun = M_sun.to('mag').value
        Q = True

    mag = M_sun-2.5*np.log10(ref*np.pi*radius*radius*units.km.to('au')**2)
    if Q:
        return mag*units.mag
    else:
        return mag


def mag2ref(mag, radius, M_sun=None):
    """Convert reduced magnitude to average bidirectional reflectance

    Parameters
    ----------
    mag : num, astropy.units.Quantity
        Reduced magnitude
    radius : num, astropy.units.Quantity
        Radius of object
    M_sun : number, optional
        The magnitude of the Sun, default is -26.74

    Returns
    -------
    The same type as input, average bidirectional reflectance

    Examples
    --------
    >>> from astropy import units as u
    >>> ref = mag2ref(2.08, 460) # doctest: +SKIP
    >>> print(ref) # doctest: +SKIP
    >>> ref = mag2ref(2.08, 460*u.km) # doctest: +SKIP
    >>> print(ref) # doctest: +SKIP

    """

    if M_sun is None:
        M_sun = -26.74
    Q = False
    if isinstance(mag, units.Quantity):
        mag = mag.value
        Q = True
    if isinstance(radius, units.Quantity):
        radius = radius.to('km').value
        Q = True
    if isinstance(M_sun, units.Quantity):
        M_sun = M_sun.to('mag').value
        Q = True

    ref = 10**((M_sun-mag)*0.4)/(np.pi*radius*radius*units.km.to('au')**2)
    if Q:
        return ref/units.sr
    else:
        return ref


class spline(object):
    '''Spline with function values at nodes and the first derivatives at
    both ends.  Outside the data grid the extrapolations are linear based
    on the first derivatives at the corresponding ends.
    '''

    def __init__(self, x, y, dy):
        x = np.asarray(x)
        y = np.asarray(y)
        dy = np.asarray(dy)
        self.x, self.y, self.dy = x, y, dy
        n = len(y)
        h = x[1:]-x[:-1]
        r = (y[1:]-y[:-1])/(x[1:]-x[:-1])
        B = np.zeros((n-2,n))
        for i in range(n-2):
            k = i+1
            B[i,i:i+3] = [h[k], 2*(h[k-1]+h[k]), h[k-1]]
        C = np.empty((n-2,1))
        for i in range(n-2):
            k = i+1
            C[i] = 3*(r[k-1]*h[k]+r[k]*h[k-1])
        C[0] = C[0]-dy[0]*B[0,0]
        C[-1] = C[-1]-dy[1]*B[-1,-1]
        B = B[:,1:n-1]
        from numpy.linalg import solve
        dys = solve(B, C)
        dys = np.array([dy[0]] + [tmp for tmp in dys.flatten()] + [dy[1]])
        A0 = y[:-1]
        A1 = dys[:-1]
        A2 = (3*r-2*dys[:-1]-dys[1:])/h
        A3 = (-2*r+dys[:-1]+dys[1:])/h**2
        self.coef = np.array([A0, A1, A2, A3]).T
        self.polys = []
        from numpy.polynomial.polynomial import Polynomial
        for c in self.coef:
            self.polys.append(Polynomial(c))
        self.polys.insert(0, Polynomial([1,self.dy[0]]))
        self.polys.append(Polynomial([self.y[-1]-self.x[-1]*self.dy[-1], self.dy[-1]]))

    def __call__(self, x):
        x = np.asarray(x)
        out = np.zeros_like(x)
        idx = x < self.x[0]
        if idx.any():
            out[idx] = self.polys[0](x[idx])
        for i in range(len(self.x)-1):
            idx = (self.x[i] <= x ) & (x < self.x[i+1])
            if idx.any():
                out[idx] = self.polys[i+1](x[idx]-self.x[i])
        idx = (x >= self.x[-1])
        if idx.any():
            out[idx] = self.polys[-1](x[idx])
        return out


class DiskIntegratedModelClass(Fittable1DModel):
    """Base class for disk-integrated phase function model

    Examples
    --------
    >>> # Define a disk-integrated phase function model
    >>> import numpy as np
    >>> from astropy.modeling import Parameter
    >>>
    >>> class LinearPhaseFunc(DiskIntegratedModelClass): # doctest: +SKIP
    >>>
    >>>     _unit = 'mag' # doctest: +SKIP
    >>>     H = Parameter() # doctest: +SKIP
    >>>     S = Parameter() # doctest: +SKIP
    >>>
    >>>     @staticmethod # doctest: +SKIP
    >>>     def evaluate(a, H, S): # doctest: +SKIP
    >>>         return H + S * a # doctest: +SKIP
    >>>
    >>> linear_phasefunc = LinearPhaseFunc(5, 2.29, radius=300) # doctest: +SKIP
    >>> pha = np.linspace(0, 180, 200) # doctest: +SKIP
    >>> f,ax = plt.subplots(2, 1, sharex=True) # doctest: +SKIP
    >>> ax[0].plot(pha, linear_phasefunc(np.deg2rad(pha))) # doctest: +SKIP
    >>> ax[0].set_ylim([13, 4]) # doctest: +SKIP
    >>> ax[1].plot(pha, linear_phasefunc.ref(np.deg2rad(pha))) # doctest: +SKIP
    >>> geoalb = linear_phasefunc.geoalb # doctest: +SKIP
    >>> bondalb = linear_phasefunc.bondalb # doctest: +SKIP
    >>> print('Geometric albedo is {0:.3}, Bond albedo is {1:.3}'.format(geoalb, bondalb)) # doctest: +SKIP
    """

    _unit = None

    def __init__(self, *args, radius=None, M_sun=None, **kwargs):
        super(Fittable1DModel, self).__init__(*args, **kwargs)
        self.radius = radius
        self.M_sun = M_sun

    def _check_unit(self):
        if self._unit is None:
            raise ValueError('the unit of phase function is unknown')

    @property
    def geoalb(self):
        """Geometric albedo"""
        return np.pi*self.ref(0)

    @property
    def bondalb(self):
        """Bond albedo"""
        return self.geoalb*self.phase_integral()

    @property
    def phaseint(self):
        """Phase integral"""
        return None

    def fit(self, eph):
        """Fit photometric model to photometric data stored in sbpy.data.Ephem
        object

        Parameters
        ----------
        eph : `sbpy.data.Ephem` instance, mandatory
            photometric data; must contain `phase` (phase angle) and `mag`
            (apparent magnitude) columns; `mag_err` optional

        Returns
        -------
        fit Chi2

        Examples
        --------
        >>> from sbpy.photometry import HG # doctest: +SKIP
        >>> from sbpy.data import Misc # doctest: +SKIP
        >>> eph = Misc.mpc_observations('Bennu') # doctest: +SKIP
        >>> hg = HG() # doctest: +SKIP
        >>> chi2 = hg.fit(eph) # doctest: +SKIP

        not yet implemented

        """

    def distance_module(self, eph):
        """Account magnitudes for distance module (distance from observer,
        distance to the Sun); return modified magnitudes

        Parameters
        ----------
        eph : list or array, mandatory
            phase angles

        Returns
        -------
        sbpy.data.Ephem instance

        Examples
        --------
        TBD

        not yet implemented

        """

    def mag(self, pha, **kwargs):
        """Calculate phase function in magnitude

        Parameters
        ----------
        pha : number or array_like of numbers
            Phase angles

        Returns
        -------
        Numpy array of magnitude

        Examples
        --------
        >>> import numpy as np
        >>> from matplotlib import pyplot as plt # doctest: +SKIP
        >>> ceres_hg = HG(3.4, 0.12) # doctest: +SKIP
        >>> pha = np.linspace(0, 180, 200) # doctest: +SKIP
        >>> plt.plot(pha, ceres_hg.mag(np.deg2rad(pha))) # doctest: +SKIP

        """
        self._check_unit()
        out = self(pha, **kwargs)
        if self._unit == 'mag':
            return out
        else:
            if self.radius is None:
                raise ValueError('cannot calculate phase funciton in magnitude because the size of object is unknown')
            return ref2mag(out, self.radius, M_sun=self.M_sun)

    def ref(self, pha, normalized=None, **kwargs):
        """Calculate phase function in average bidirectional reflectance

        Parameters
        ----------
        pha : number or array_like of numbers
            Phase angles

        Returns
        -------
        Numpy array of average bidirectional reflectance

        Examples
        --------
        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> ceres_hg = HG(3.4, 0.12, radius=480) # doctest: +SKIP
        >>> pha = np.linspace(0, 180, 200) # doctest: +SKIP
        >>> plt.plot(pha, ceres_hg.ref(np.deg2rad(pha))) # doctest: +SKIP

        """
        self._check_unit()
        out = self(pha, **kwargs)
        if self._unit == 'ref':
            if normalized is not None:
                out /= self(normalized, **kwargs)
            return out
        else:
            if self.radius is None:
                raise ValueError('cannot calculate phase function in reflectance unit because the size of object is unknown')
            out = mag2ref(out, self.radius, M_sun=self.M_sun)
            if normalized is not None:
                out /= mag2ref(self(normalized, **kwargs), self.radius, M_sun=self.M_sun)
            return out

    def phase_integral(self, integrator=quad):
        """Calculate phase integral.

        If property `self.phaseint` is not loaded (has a `None` value), then
        the phase integral will be calculated with numerical integration.
        Otherwise `self.phase.int` will be returned.

        Parameters
        ----------
        integrator : function, optinonal
            Numerical integrator, default is `scipy.integrate.quad`.  If caller
            supplies a numerical integrator, it must has the same return
            signature as `scipy.integrator.quad`, i.e., a tuple of (y, ...),
            where `y` is the result of numerical integration

        Returns
        -------
        Number, phase integral

        Examples
        --------
        >>> ceres_hg = HG(3.4, 0.12, radius=480) # doctest: +SKIP
        >>> print(ceres_hg.phase_integral()) # doctest: +SKIP

        """
        if hasattr(self, 'phaseint'):
            if self.phaseint is not None:
                return self.phaseint
        else:
            integrand = lambda x: 2*self.ref(x, normalized=0.)*np.sin(x)
            return integrator(integrand, 0, np.pi)[0]


class HG(DiskIntegratedModelClass):
    """HG photometric phase model (Bowell et al. 1989)"""

    _unit = 'mag'
    H = Parameter(description='H parameter')
    G = Parameter(description='G parameter')

    @staticmethod
    def _hgphi(pha, i):
        """Core function in IAU HG phase function model

        Parameters
        ----------
        pha : number or array_like of numbers
            Phase angle
        i   : int in [1, 2]
            Choose the form of function

        Returns
        -------
        numpy array of float

        Examples
        --------
        TBD

        Note
        ----
        See Bowell et al. (1989), Eq. A4.
        """

        if i not in [1,2]:
            raise ValueError('i needs to be 1 or 2, {0} received'.format(i))

        a, b, c = [3.332, 1.862], [0.631, 1.218], [0.986, 0.238]
        pha_half = pha*0.5
        sin_pha = np.sin(pha)
        tan_pha_half = np.tan(pha_half)
        w = np.exp(-90.56 * tan_pha_half * tan_pha_half)
        phiis = 1 - c[i-1]*sin_pha/(0.119+1.341*sin_pha-0.754*sin_pha*sin_pha)
        phiil = np.exp(-a[i-1] * tan_pha_half**b[i-1])
        return w*phiis + (1-w)*phiil

    @staticmethod
    def evaluate(pha, hh, gg):
        return hh-2.5*np.log10((1-gg)*HG._hgphi(pha, 1)+gg*HG._hgphi(pha,2))

    @staticmethod
    def fit_deriv(pha, hh, gg):
        if hasattr(pha,'__iter__'):
            ddh = np.ones_like(pha)
        else:
            ddh = 1.
        phi1 = HG._hgphi(pha,1)
        phi2 = HG._hgphi(pha,2)
        ddg = -1.085736205*(-phi1+phi2)/((1-gg)*phi1+gg*phi2)
        return [ddh, ddg]


class HG12BaseClass(DiskIntegratedModelClass):
    """Base class for IAU HG1G2 model and HG12 model"""

    _unit = 'mag'

    @property
    def _G1(self):
        return None

    @property
    def _G2(self):
        return None

    @property
    def phaseint(self):
        """Phase integral, q
        Based on Muinonen et al. (2010) Eq. 22
        """
        return 0.009082+0.4061*self._G1+0.8092*self._G2

    @property
    def phasecoeff(self):
        """Phase coefficient, k
        Based on Muinonen et al. (2010) Eq. 23
        """
        return -(30*self._G1+9*self._G2)/(5*np.pi*float(self._G1+self._G2))

    @property
    def oe_amp(self):
        """Opposition effect amplitude, `$\zeta-1$`
        Based on Muinonen et al. (2010) Eq. 24)
        """
        tmp = float(self._G1+self._G2)
        return (1-tmp)/tmp

    _phi1v = np.deg2rad([7.5, 30., 60, 90, 120, 150]),[7.5e-1, 3.3486016e-1, 1.3410560e-1, 5.1104756e-2, 2.1465687e-2, 3.6396989e-3],[-1.9098593, -9.1328612e-2]
    _phi1 = spline(*_phi1v)
    _phi2v = np.deg2rad([7.5, 30., 60, 90, 120, 150]),[9.25e-1, 6.2884169e-1, 3.1755495e-1, 1.2716367e-1, 2.2373903e-2, 1.6505689e-4],[-5.7295780e-1, -8.6573138e-8]
    _phi2 = spline(*_phi2v)
    _phi3v = np.deg2rad([0.0, 0.3, 1., 2., 4., 8., 12., 20., 30.]),[1., 8.3381185e-1, 5.7735424e-1, 4.2144772e-1, 2.3174230e-1, 1.0348178e-1, 6.1733473e-2, 1.6107006e-2, 0.],[-1.0630097, 0]
    _phi3 = spline(*_phi3v)


class HG1G2(HG12BaseClass):
    """HG1G2 photometric phase model (Muinonen et al. 2010)"""

    H = Parameter(description='H parameter')
    G1 = Parameter(description='G1 parameter')
    G2 = Parameter(description='G2 parameter')

    @property
    def _G1(self):
        return self.G1

    @property
    def _G2(self):
        return self.G2

    @staticmethod
    def evaluate(ph, h, g1, g2):
        return h-2.5*np.log10(g1*HG1G2._phi1(ph)+g2*HG1G2._phi2(ph)+(1-g1-g2)*HG1G2._phi3(ph))

    @staticmethod
    def fit_deriv(ph, h, g1, g2):
        """Need to check the formula"""
        if hasattr(ph, '__iter__'):
            ddh = np.ones_like(ph)
        else:
            ddh = 1.
        phi1 = HG1G2._phi1(ph)
        phi2 = HG1G2._phi2(ph)
        phi3 = HG1G2._phi3(ph)
        dom = (g1*phi1+g2*phi2+(1-g1-g2)*phi3)
        ddg1 = -1.085736205*(phi1-phi3)/dom
        ddg2 = -1.085736205*(phi2-phi3)/dom
        return [ddh, ddg1, ddg2]


class HG12(HG12BaseClass):
    """HG12 photometric phase model (Muinonen et al. 2010)"""

    H = Parameter(description='H parameter')
    G12 = Parameter(description='G12 parameter')

    @property
    def _G1(G12):
        if G12<0.2:
            return 0.7527*G12+0.06164
        else:
            return 0.9529*G12+0.02162

    @property
    def _G2(G12):
        if G12<0.2:
            return -0.9612*G12+0.6270
        else:
            return -0.6125*G12+0.5572

    @staticmethod
    def evaluate(ph, h, g):
        g1 = HG12._G1(g)
        g2 = HG12._G2(g)
        return HG1G2.evaluate(ph, h, g1, g2)

    @staticmethod
    def fit_deriv(ph, h, g):
        """Need to check formula"""
        if hasattr(ph, '__iter__'):
            ddh = np.ones_like(ph)
        else:
            ddh = 1.
        g1 = HG12._G1(g)
        g2 = HG12._G2(g)
        phi1 = HG1G2._phi1(ph)
        phi2 = HG1G2._phi2(ph)
        phi3 = HG1G2._phi3(ph)
        dom = (g1*phi1+g2*phi2+(1-g1-g2)*phi3)
        if g<0.2:
            p1 = 0.7527
            p2 = -0.9612
        else:
            p1 = 0.9529
            p2 = -0.6125
        ddg = -1.085736205*((phi1-phi3)*p1+(phi2-phi3)*p2)/dom
        return [ddh, ddg]


class DiskFunctionModel(FittableModel):
    """Base class for disk-function model"""
    pass


class LommelSeeliger(DiskFunctionModel):
    """Lommel-Seeliger model class"""

    inputs = ('i','e')
    outputs = ('d',)

    @staticmethod
    def evaluate(i, e):
        mu0 = np.cos(i)
        mu = np.cos(e)
        return mu0/(mu0+mu)


class Lambert(DiskFunctionModel):
    """Lambert model class"""

    inputs = ('i',)
    outputs = ('d',)

    @staticmethod
    def evaluate(i):
        return np.cos(i)


class LunarLambert(DiskFunctionModel):
    """Lunar-Lambert model, or McEwen model class"""
    inputs = ('i', 'e')
    outputs = ('d',)

    L = Parameter(default=0.1, description='Partition parameter')

    @staticmethod
    def evaluate(i, e, L):
        return (1-L) * LommelSeeliger.evaluate(i, e) + L * Lambert.evaluate(i)


class PhaseFunctionModel(FittableModel):
    """Base class for phase function model"""
    inputs = ('a',)
    outputs = ('f',)


class ROLOPhase(PhaseFunctionModel):
    """ROLO phase function model class"""
    A0 = Parameter(default=0.1, min=0., description='ROLO A0 parameter')
    A1 = Parameter(default=0.1, min=0., description='ROLO A1 parameter')
    C0 = Parameter(default=0.1, min=0., description='ROLO C0 parameter')
    C1 = Parameter(default=0.1, description='ROLO C1 parameter')
    C2 = Parameter(default=0.1, description='ROLO C2 parameter')
    C3 = Parameter(default=0.1, description='ROLO C3 parameter')
    C4 = Parameter(default=0.1, description='ROLO C4 parameter')

    @staticmethod
    def evaluate(pha, c0, c1, a0, a1, a2, a3, a4):
        pha2 = pha*pha
        return c0*np.exp(-c1*pha)+a0+a1*pha+a2*pha2+a3*pha*pha2+a4*pha2*pha2

    @staticmethod
    def fit_deriv(pha, c0, c1, a0, a1, a2, a3, a4):
        pha2 = pha*pha
        dc0 = np.exp(-c1*pha)
        if hasattr(pha, '__iter__'):
            dda = np.ones(len(pha))
        else:
            dda = 1.
        return [dc0, -c0*c1*dc0, dda, pha, pha2, pha*pha2, pha2*pha2]


class ResolvedPhotometricModelClass(object):
    """Base class for disk-resolved photometric model"""
    # composite model as the product of a disk function and a phase function
    pass


class ROLO(ResolvedPhotometricModelClass):
    """ROLO disk-resolved photometric model"""
    pass


# class Photometry():

#     def diam2mag(phys, eph, model=None):
#         """Function to calculate the apparent bightness of a body from its physical properties and ephemerides"""


