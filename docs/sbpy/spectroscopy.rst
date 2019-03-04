.. doctest-skip-all

Spectroscopy Module (`sbpy.spectroscopy`)
=========================================

Introduction
------------

`~sbpy.spectroscopy` provides routines for the modeling and analysis of emission spectra of gas in comets and reflection spectra of asteroid surfaces.  Sub-modules `sun` and `vega` control `sbpy`'s photometric calibration.

JPLSpec Constants and Line Intensity Integral Conversion
--------------------------------------------------------

`sbpy.spectroscopy` has a function called ``molecular_data`` which takes care of
querying the JPL Molecular Spectral Catalog through the use of
`astroquery.jplspec` and calculates all the necessary constants needed for both
production rate and Einstein coefficient calculations. The function
``intensity_conversion`` takes care of converting the intensity line integral at
300 K found in JPL Spec catalog and convert it closer to the temperature given
by the user.

.. code-block:: python

   >>> from sbpy.spectroscopy import molecular_data, intensity_conversion
   >>> import astropy.units as u
   >>> temp_estimate = 33. * u.K
   >>> vgas = 0.8 * u.km / u.s
   >>> mol_tag = 27001
   >>> transition_freq = 265.886434 * u.MHz
   >>> mol_data = molecular_data(temp_estimate, transition_freq, mol_tag, vgas)
   >>> intl = intensity_conversion(temp_estimate, transition_freq, mol_tag, vgas)
   >>> co
     [<Quantity 265886.18 MHz>,
      <Quantity 37.5 K>,
      <Quantity 0.08412014 MHz nm2>,
      424.32634842758375,
      53.91380704697852,
      1.7317,
      <Quantity 6.62607004e-34 J s>,
      <Quantity 1.38064852e-23 J / K>,
      <Quantity 2.99792458e+08 m / s>,
      <Quantity 800. m / s>,
      <Quantity 3.52359898e-22 J>,
      <Quantity 1.76181853e-22 J>]

   >>> intl
      <Quantity 3.99731047636546 MHz nm2>


Einstein coefficient
--------------------

`sbpy.spectroscopy` offers a function to calculate the Einstein coefficient
for a specific molecule and transition frequency.

.. code-block:: python

   >>> from sbpy.spectroscopy import einstein_coeff
   >>> e = einstein_coeff(temp_estimate, transition_freq, mol_tag, vgas)
   >>> e
      <Quantity 0.0008601294364222543 1 / s>


Production Rate Calculations
----------------------------

`sbpy.spectroscopy` provides several models to calculate production rates in comets.
One of the models followed by this module is based on the following paper:

| Drahus et al. September 2012. The Sources of HCN and CH3OH and the
| Rotational Temperature in Comet 103P/Hartley 2 from Time-resolved
| Millimeter Spectroscopy. The Astrophysical Journal, Volume 756,
| Issue 1.

This model does not include photodissociation. The model uses the functions for
constants, conversion, and Einstein coefficients as well as a JPLHorizons
lookup to obtain the distance to the comet at observation time. The following
example shows the usage of the module. For more information on the parameters
that are optional or needed for the module follow the link under Reference/API
section.

.. code-block:: python

  >>> from sbpy.spectroscopy import prodrate_np
  >>> temp_estimate = 33. * u.K
  >>> target = '900918'
  >>> vgas = 0.8 * u.km / u.s
  >>> aper = 30 * u.m
  >>> b = 1.13
  >>> mol_tag = 27001
  >>> transition_freq = 265.886434 * u.MHz
  >>> spectra = 1.22 * u.K * u.km / u.s
  >>> time = '2010-11-3 00:48:06'
  >>> q = prodrate_np(spectra, temp_estimate, transition_freq,
                            mol_tag, time, target, vgas, aper,
                            b=b, id_type='id')

  >>> q
  <Quantity 1.0432591198553935e+25 1 / s>

Another model included in the module is based off of the model in the following
literature:

| Haser 1957, Bulletin de la Societe Royale des Sciences de Liege 43, 740.
| Newburn and Johnson 1978, Icarus 35, 360-368.

This model takes in an initial guess for the production rate, and uses the
module found in ``sbpy.activity.gas`` to find a ratio between the model model
total number of molecules and the number of molecules calculated from the data
to scale the model Q and output the new production rate from the result. This
model does account for the effects of photolysis.

.. code-block:: python

  >>> from sbpy.activity.gas import Haser
  >>> coma = Haser(Q, v, parent)
  >>> Q = spec.production_rate(coma, molecule='H2O')

  >>> Q_estimate = 2.8*10**(28) / u.s
  >>> transition_freq = (230.53799 * u.GHz).to('MHz')
  >>> aper = 10 * u.m
  >>> mol_tag = 28001
  >>> temp_estimate = 25. * u.K
  >>> target = 'C/2016 R2'
  >>> b = 0.74
  >>> vgas = 0.5 * u.km / u.s

  >>> time = '2017-12-22 05:24:20'
  >>> spectra = 0.26 * u.K * u.km / u.s

  >>> parent = photo_timescale('CO') * vgas

  >>> coma = Haser(Q_estimate, vgas, parent)

  >>> Q = spec.production_rate(coma, spectra, temp_estimate,
                               transition_freq, mol_tag, time, target,
                               aper=aper, b=b)

  >>> print(Q)
      <Quantity [1.64403219e+28] 1 / s>

.. _sbpy_spectral_standards:      

Spectral standards and photometric calibration
----------------------------------------------
`sbpy` has the ability to return spectra of the Sun and Vega.  There are built-in spectra for each.  The spectra are additionally used for `sbpy`'s photometric calibration.  Users may provide their own spectra.

For Vega, the spectrum of Bohlin (2014) is the default.  It is downloaded and cached from STScI, as needed.  Four solar spectra are built-in:

  * E490_2014 - E490 (2014) standard.
  * E490_2014LR - A low resolution version of the E490 standard.
  * Kurucz1993 - Kurucz (1993) model.
  * Castelli1996 - Castelli model from Colina et al. (1996).

The E490 spectra are included with `sbpy`, and the Kurucz and Castell spectra are downloaded as needed from `STScI's reference data <http://www.stsci.edu/hst/observatory/crds/astronomical_catalogs.html>`_.

`sbpy` has a `Sun` class that encapsulates solar spectra.  The class can be initialized with the default spectrum using `from_default`:

.. doctest-requires:: synphot

   >>> from sbpy.spectroscopy.sun import Sun
   >>> sun = Sun.from_default()
   >>> print(sun)
   <Sun: E490-00a (2014) reference solar spectrum (Table 3)>

The names of the built-in sources are stored as an internal array.  Any built-in spectrum can initialize a `Sun` object using `from_builtin`:

.. doctest-requires:: synphot

   >>> from sbpy.spectroscopy.sun import Sun, sources
   >>> print(sources.available)
   ['E490_2014', 'E490_2014LR', 'Kurucz1993', 'Castelli1996']
   >>> sun = Sun.from_builtin('E490_2014LR')
   >>> print(sun)
   <Sun: E490-00a (2014) low resolution reference solar spectrum (Table 4)>

The solar spectrum in current use is controlled with `default_sun`:

.. doctest-requires:: synphot

   >>> from sbpy.spectroscopy.sun import Sun, default_sun
   >>> default_sun.set('E490_2014LR')
   <ScienceState default_sun: <Sun: E490-00a (2014) low resolution reference solar spectrum (Table 4)>>
   >>> # E490 low-resolution spectrum in effect for all of sbpy
   >>> sun = Sun.from_default()
   >>> print(sun)
   <Sun: E490-00a (2014) low resolution reference solar spectrum (Table 4)>

`default_sun` can also be used as a context manager to temporarily change the default spectrum:

.. doctest-requires:: synphot

   >>> from sbpy.spectroscopy.sun import Sun, default_sun
   >>> default_sun.set('E490_2014')  # E490 in effect
   <ScienceState default_sun: <Sun: E490-00a (2014) reference solar spectrum (Table 3)>>
   >>> with default_sun.set('E490_2014LR'):
   ...   # E490 low-res in effect
   ...   print(Sun.from_default())
   <Sun: E490-00a (2014) low resolution reference solar spectrum (Table 4)>
   >>> # Back to module default.
   >>> print(Sun.from_default())
   <Sun: E490-00a (2014) reference solar spectrum (Table 3)>

Provide your own solar spectrum with the `Sun` class:

.. doctest-requires:: synphot

   >>> from sbpy.spectroscopy.sun import Sun, default_sun
   >>> with default_sun.set(Sun.from_file('sun.txt')):  # doctest: +SKIP
   ...   # sun.txt in effect

See `~Sun` for more information on ways to create solar spectra.

In a similar manner, Vega spectra are accessed and controlled via `Vega` and `default_vega`:

.. doctest-requires:: synphot

   >>> from sbpy.spectroscopy.vega import Vega, default_vega
   >>> print(Vega.from_default())     # doctest: +SKIP
   <Vega: Spectrum of Bohlin 2014>
   >>> with default_vega.set(Vega.from_file('vega.txt')):  # doctest: +SKIP
   ...   # vega.txt in effect


Observe the Sun through a filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`sbpy` simulates observations of comets and asteroids through filter bandpasses.  To support this functionality, the `Sun` and `Vega` classes have `filt` methods that return simulated flux densities.

Get the default solar spectrum, and observe it through the Johnson V-band filter, using `synphot`'s built-in support:

.. doctest-requires:: synphot

   >>> from sbpy.spectroscopy.sun import Sun
   >>> sun = Sun.from_default()
   >>> wave, mag = sun.filt('johnson_v', unit='vegamag')                                                                          # doctest: +REMOTE_DATA
   >>> print('{}\n- Johnson V = {:.2f} at {:.0f}'.format(sun.description, mag, wave))                                           # doctest: +REMOTE_DATA
   E490-00a (2014) reference solar spectrum (Table 3)
   - Johnson V = -26.77 VEGAMAG at 5502 Angstrom

Repeat with the Castelli 1996 solar spectrum:

.. doctest-requires:: synphot

   >>> from sbpy.spectroscopy.sun import Sun, default_sun
   >>> with default_sun.set('Castelli1996'):
   ...   sun = Sun.from_default()                                                                                                 # doctest: +REMOTE_DATA
   ...   wave, mag = sun.filt('johnson_v', unit='vegamag')                                                                        # doctest: +REMOTE_DATA
   ...   print('{}\n- Johnson V = {:.2f} at {:.0f}'.format(sun.description, mag, wave))                                         # doctest: +REMOTE_DATA
   Castelli model, scaled and presented by Colina et al. (1996)
   - Johnson V = -26.78 VEGAMAG at 5502 Angstrom

Finally, observe the Sun through *Hubble*/WFC3's F438W and F606W filters:

.. doctest-requires:: synphot

   >>> import synphot
   >>> from sbpy.spectroscopy.sun import Sun
   >>> fn = 'ftp://ftp.stsci.edu/cdbs/comp/wfc3/wfc3uvis1_f438w_mjd_006_syn.fits'
   >>> f438w = synphot.SpectralElement.from_file(fn)                                                               # doctest: +REMOTE_DATA +IGNORE_OUTPUT
   >>> fn = 'ftp://ftp.stsci.edu/cdbs/comp/wfc3/wfc3uvis1_f606w_mjd_006_syn.fits'
   >>> f606w = synphot.SpectralElement.from_file(fn)                                                               # doctest: +REMOTE_DATA +IGNORE_OUTPUT
   >>> sun = Sun.from_builtin('E490_2014')
   >>> for name, bp in zip(('F438W', 'F606W'), (f438w, f606w)):                                                                   # doctest: +REMOTE_DATA
   ...     print(name)
   ...     for unit in ('vegamag', 'ABmag', 'W/(m2 um)'):
   ...         wave, obs = sun.filt(bp, unit=unit)
   ...         print('  {:.2f} at {:.0f}'.format(obs, wave))
   F438W
     -26.08 VEGAMAG at 4351 Angstrom
     -26.24 mag(AB) at 4351 Angstrom
     1799.80 W / (m2 um) at 4351 Angstrom
   F606W
     -26.94 VEGAMAG at 5957 Angstrom
     -26.88 mag(AB) at 5957 Angstrom
     1707.19 W / (m2 um) at 5957 Angstrom


Plot solar spectra
^^^^^^^^^^^^^^^^^^

Solar spectra in Sun objects can be plotted at the native resolution of the data, or rebinned. Plot the Castelli 1996 spectrum at the native resolution, and at a resolution of ~25:

.. doctest-skip::

   >>> import astropy.units as u
   >>> import numpy as np
   >>> import matplotlib.pyplot as plt
   >>> from sbpy.spectroscopy.sun import Sun
   >>> # Create an array of wavelengths for a particular spectral resolution
   >>> R = 25  # goal spectral resolution
   >>> wrange = 0.3, 0.8  # wavelength range
   >>> d = 1 + 1 / R
   >>> n = int(np.ceil(np.log(wrange[1] / wrange[0]) / np.log(d)))
   >>> wave_binned = wrange[0] * d**np.arange(n) * u.um
   >>> # Get the built-in Castelli 1996 model and rebin
   >>> sun = Sun.from_builtin('Castelli1996')
   >>> fluxd_binned = sun(wave_binned, unit='W / (m2 um)')
   >>> # Plot
   >>> plt.plot(sun.wave.to('um'), sun.fluxd.to('W/(m2 um)'),
   ...          ls='steps-mid', color='#1f77b4', label='C96')
   >>> plt.plot(wave_binned, fluxd_binned, ls='steps-mid',
   ...          color='#ff7f0e', label='C96, R~{}'.format(R))
   >>> plt.setp(plt.gca(), xlim=wrange, xlabel='Wavelength (μm)', ylabel='Flux density (W/(m2 μm)')
   >>> plt.legend()
   >>> plt.tight_layout()

.. plot::

   import astropy.units as u
   import numpy as np
   import matplotlib.pyplot as plt
   from sbpy.spectroscopy.sun import Sun
   R = 25  # goal spectral resolution
   wrange = 0.3, 0.8  # wavelength range
   d = 1 + 1 / R
   n = int(np.ceil(np.log(wrange[1] / wrange[0]) / np.log(d)))
   wave_binned = wrange[0] * d**np.arange(n) * u.um
   sun = Sun.from_builtin('Castelli1996')
   fluxd_binned = sun(wave_binned, unit='W / (m2 um)')
   plt.plot(sun.wave.to('um'), sun.fluxd.to('W/(m2 um)'), ls='steps-mid', color='#1f77b4', label='C96')
   plt.plot(wave_binned, fluxd_binned, ls='steps-mid', color='#ff7f0e', label='C96, R~{}'.format(R))
   plt.setp(plt.gca(), xlim=wrange, xlabel='Wavelength (μm)', ylabel='Flux density (W/(m2 μm)')
   plt.legend()
   plt.tight_layout()


Reference/API
-------------
.. automodapi:: sbpy.spectroscopy
    :no-heading:
