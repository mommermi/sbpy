
.. _alternative_fieldnames:

Alternative Field Names
=======================

The following table lists alternative field names accepted by `sbpy`
when accessing `~sbpy.data.DataClass` objects, i.e.,
`~sbpy.data.Ephem`, `~sbpy.data.Orbit`, or `~sbpy.data.Phys` objects.

As an example, heliocentric distance can be addressed a ``'r'`` or
``'heldist'``:

    >>> from sbpy.data import Ephem
    >>> ceres = Ephem.from_horizons('Ceres')
    >>> print(ceres['r']) # doctest: +IGNORE_OUTPUT
    [2.69866993] AU
    >>> print(ceres['heldist']) # doctest: +IGNORE_OUTPUT
    [2.69866993] AU

The list of alternative field names is always up to date, but not
complete. The source list is located as
``sbpy.data.conf.fieldnames`` in file ``sbpy/data/__init__.py``. If you
think an important alternative
is missing, please suggest it by opening an issue. However, keep in mind
that each alternative field name has to be *unique* and *unambiguous*.


List of Alternative Field Names
-------------------------------

=================================== ===================================================================================================================
                        Description                                                                                                   Alternative Names
=================================== ===================================================================================================================
              **Target Identifier**                                                                                  ``targetname``, ``id``, ``Object``
                    **Inclination**                                                                                            ``i``, ``inc``, ``incl``
                          **Epoch**                                                              ``epoch``, ``datetime_jd``, ``JD``, ``Date``, ``date``
**Longitude of the Ascending Node**                                                                                             ``Omega``, ``longnode``
      **Argument of the Periapsis**                                                                                                   ``w``, ``argper``
          **Heliocentric Distance**                                                                               ``r``, ``rh``, ``r_hel``, ``heldist``
       **Distance to the Observer**                                                                                   ``delta``, ``Delta``, ``obsdist``
                **Right Ascension**                                                                                                      ``ra``, ``RA``
                    **Declination**                                                                                           ``dec``, ``DEC``, ``Dec``
                        **RA Rate**                                              ``ra_rate``, ``RA_rate``, ``ra_rates``, ``RA_rates``, ``dRA``, ``dra``
               **RA*cos(Dec) Rate**                        ``RA*cos(Dec)_rate``, ``dra cos(dec)``, ``dRA cos(Dec)``, ``dra*cos(dec)``, ``dRA*cos(Dec)``
                       **Dec Rate** ``dec_rate``, ``DEC_rate``, ``Dec_rate``, ``dec_rates``, ``DEC_rates``, ``Dec_rates``, ``dDec``, ``dDEC``, ``ddec``
                  **Proper Motion**                                                                                           ``mu``, ``Proper motion``
        **Proper Motion Direction**                                                                                         ``Direction``, ``directin``
              **Solar Phase Angle**                                                                                ``alpha``, ``phaseangle``, ``Phase``
               **Solar Elongation**                                      ``elong``, ``solarelong``, ``solarelongation``, ``elongation``, ``Elongation``
               **V-band Magnitude**                                                                                                     ``V``, ``Vmag``
**Heliocentric Ecliptic Longitude**                                                      ``hlon``, ``EclLon``, ``ecllon``, ``HelEclLon``, ``helecllon``
 **Heliocentric Ecliptic Latitude**                                                      ``hlat``, ``EclLat``, ``ecllat``, ``HelEclLat``, ``helecllat``
                      **Elevation**                                                  ``el``, ``EL``, ``elevation``, ``alt``, ``altitude``, ``Altitude``
                        **Azimuth**                                                                                         ``az``, ``AZ``, ``azimuth``
               **Lunar Elongation**                          ``lunar_elong``, ``elong_moon``, ``elongation_moon``, ``lunar_elongation``, ``lunarelong``
           **x Velocity Component**                                                                                           ``vx``, ``dx``, ``dx/dt``
           **y Velocity Component**                                                                                           ``vy``, ``dy``, ``dy/dt``
           **z Velocity Component**                                                                                           ``vz``, ``dz``, ``dz/dt``
                       **Diameter**                                                                                ``d``, ``D``, ``diam``, ``diameter``
                         **Radius**                                                                                                   ``R``, ``radius``
        **V-band Geometric Albedo**                                                            ``pv``, ``pV``, ``p_v``, ``p_V``, ``albedo``, ``geoalb``
=================================== ===================================================================================================================
