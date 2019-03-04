.. _status page:

Status Page
===========

This page indicates the progress on specific modules, classes, and
functions.

While this page is mainly intented for internal purposes, it provides
insight into functionality that is already available and can be
tested.

If you are or have been working on code, please indicate its status
here.

Legend
~~~~~~

.. raw:: html

    <style>
         .planned:before {
              color: #cbcbcb;
              content: "⬤";
         }
         .dev:before {
              color: #ffad00;
              content: "⬤";
         }
         .stable:before {
              color: #4e72c3;
              content: "⬤";
         }
         .mature:before {
              color: #03a913;
              content: "⬤";
         }
         .pendingdep:before {
              color: #a84b03;
              content: "⬤";
         }
         .deprecated:before {
              color: #ff0000;
              content: "⬤";
         }
    </style>

    <table align='center'>
      <tr>
        <td align='center'><span class="planned"></span></td>
        <td>Planned</td>
      </tr>
      <tr>
        <td align='center'><span class="dev"></span></td>
        <td>Actively developed, be prepared for possible significant changes.</td>
      </tr>
      <tr>
        <td align='center'><span class="stable"></span></td>
        <td>Reasonably stable, but potentially incomplete; any significant changes/additions will generally include backwards-compatiblity.</td>
      </tr>
      <tr>
        <td align='center'><span class="mature"></span></td>
        <td>Mature.  Additions/improvements possible, but no major changes planned. </td>
      </tr>
      <tr>
        <td align='center'><span class="pendingdep"></span></td>
        <td>Pending deprecation.  Might be deprecated in a future version.</td>
      </tr>
      <tr>
        <td align='center'><span class="deprecated"></span></td>
        <td>Deprecated.  Might be removed in a future version.</td>
      </tr>
    </table>



Current development status
~~~~~~~~~~~~~~~~~~~~~~~~~~

The current development status of `sbpy` sub-modules is indicated in
the following tables.  The column `Dev` indicates the lead developer
for the respective sub-packages and classes.


`sbpy.data`
-----------

.. raw:: html

    <table border="1" class="docutils stability" align='center'>
        <tr>
            <th class="head">
                Sub-Packages and Classes
            </th>
            <th class="head">
                &nbsp;
            </th>
            <th class="head">
                Dev
            </th>
            <th class="head">
                Comments
            </th>
        </tr>
        <tr>
            <td>
                <em>sbpy.data.DataClass</em>
            </td>
            <td align='center'>
                <span class="stable"></span>
            </td>
            <td align='center'>
                MM
            </td>
            <td>
                basic functionality established, alternative property names implemented; tests and documentation available
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.data.Names</em>
            </td>
            <td align='center'>
                <span class="stable"></span>
            </td>
            <td align='center'>
                MM
            </td>
             <td>
                asteroid and comet name parsing established, asteroid_or_comet implemented; <em>sbpy.data.Names.altident</em> not yet implemented
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.data.Ephem</em>
            </td>
            <td align='center'>
                <span class="stable"></span>
            </td>
            <td align='center'>
                MM
            </td>
            <td>
	       jplhorizons and Minor Planet Center queries implemented, OpenOrb ephemeris computation available; imcce, lowell queries tbd.
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.data.Orbit</em>
            </td>
            <td align='center'>
                <span class="stable"></span>
            </td>
            <td align='center'>
                MM
            </td>
            <td>
                jplhorizons query implemented, OpenOrb orbit transformations and orbit propagation implemented; mpc, imcce, lowell queries, as well as OpenOrb ranging tbd.
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.data.Phys</em>
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td align='center'>
                MM
            </td>
            <td>
                crude jplsbdb query implemented, no tests or documentation available; lowell query tbd.
            </td>
        </tr>
    </table>


`sbpy.activity`
---------------

.. raw:: html

    <table border="1" class="docutils stability" align='center'>
        <tr>
            <th class="head">
                Sub-Packages and Classes
            </th>
            <th class="head">
                &nbsp;
            </th>
            <th class="head">
                Dev
            </th>
            <th class="head">
                Comments
            </th>
        </tr>
        <tr>
            <td>
                <em>sbpy.activity.dust</em>
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td align='center'>
                MSK
            </td>
            <td>
                Halley-Marcus phase function implemented; Afρ class: conversion to/from flux density for a specific wavelength is implemented, using a filter bandpass and conversion to/from magnitudes is in development; εfρ class: conversion to/from flux density implemented for a specific wavelength, using a filter bandpass and conversion to/from magnitudes is in development; Syndynes and synchrones tbd.
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.activity.gas</em>
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td align='center'>
                MSK
            </td>
             <td>
                Haser model implemented, final tests tbd.; Vectorial model TBD (use source code from M. Festou?)
            </td>
        </tr>
    </table>

    
`sbpy.photometry`
-----------------

.. raw:: html

    <table border="1" class="docutils stability" align='center'>
        <tr>
            <th class="head">
                Sub-Packages and Classes
            </th>
            <th class="head">
                &nbsp;
            </th>
            <th class="head">
                Dev
            </th>
            <th class="head">
                Comments
            </th>
        </tr>
        <tr>
            <td>
                <em>sbpy.photometry</em>
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td align='center'>
                JYL
            </td>
            <td>
                disk integrated phase functions implemented: HG, HG12, HG1G2, linear phasecurve; disk-resolved phase functions tbd.
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.photometry.hapke</em>
            </td>
            <td align='center'>
                <span class="planned"></span>
            </td>
            <td align='center'>
                JYL
            </td>
             <td>
                Hapke light scattering functions tbd.
            </td>
        </tr>
    </table>


`sbpy.shape`
------------

.. raw:: html

    <table border="1" class="docutils stability" align='center'>
        <tr>
            <th class="head">
                Sub-Packages and Classes
            </th>
            <th class="head">
                &nbsp;
            </th>
            <th class="head">
                Dev
            </th>
            <th class="head">
                Comments
            </th>
        </tr>
        <tr>
            <td>
                <em>sbpy.shape.lightcurve</em>
            </td>
            <td align='center'>
                <span class="planned"></span>
            </td>
            <td align='center'>
                MM
            </td>
            <td>
                lightcurve periodicity modeling tools and wrappers (periodograms, Fourier analysis) tbd.
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.shape.inversion</em>
            </td>
            <td align='center'>
                <span class="planned"></span>
            </td>
            <td align='center'>
                MM
            </td>
             <td>
                Kaasalainen lightcurve inversion tool interface tbd.
            </td>
        </tr>
    </table>

    
`sbpy.spectroscopy`
-------------------

.. raw:: html

    <table border="1" class="docutils stability" align='center'>
        <tr>
            <th class="head">
                Sub-Packages and Classes
            </th>
            <th class="head">
                &nbsp;
            </th>
            <th class="head">
                Dev
            </th>
            <th class="head">
                Comments
            </th>
        </tr>
        <tr>
            <td>
                <em>sbpy.spectroscopy</em>
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td align='center'>
                MdVB
            </td>
            <td>
                some preliminary methods for absorption and emission spectroscopy implemented
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.spectroscopy.reflectance</em>
            </td>
            <td align='center'>
                <span class="planned"></span>
            </td>
            <td align='center'>
                MM
            </td>
            <td>
                tools for identification of asteroid reflectance spectra tbd.
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.spectroscopy.sun</em>
            </td>
            <td align='center'>
                <span class="stable"></span>
            </td>
            <td align='center'>
                MSK
            </td>
            <td>
                <em>sbpy.spectroscopy.sun</em> spectral model implemented and fully tested.
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.spectroscopy.vega</em>
            </td>
            <td align='center'>
                <span class="stable"></span>
            </td>
            <td align='center'>
                MSK
            </td>
            <td>
                <em>sbpy.spectroscopy.vega</em> spectral model implemented and fully tested.
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.spectroscopy.spectrophotometry</em>
            </td>
            <td align='center'>
                <span class="planned"></span>
            </td>
            <td align='center'>
                MM
            </td>
             <td>
                spectrophotometry tools tbd.
            </td>
        </tr>
    </table>

`sbpy.thermal`
--------------

.. raw:: html

    <table border="1" class="docutils stability" align='center'>
        <tr>
            <th class="head">
                Sub-Packages and Classes
            </th>
            <th class="head">
                &nbsp;
            </th>
            <th class="head">
                Dev
            </th>
            <th class="head">
                Comments
            </th>
        </tr>
        <tr>
            <td>
                <em>sbpy.thermal</em>
            </td>
            <td align='center'>
                <span class="planned"></span>
            </td>
            <td align='center'>
                MM
            </td>
            <td>
                STM, FRM, NEATM thermal model implementations tbd, fitting and modeling routines tbd. 
            </td>
        </tr>
   </table>

    
`sbpy.imageanalysis`
--------------------

.. raw:: html

    <table border="1" class="docutils stability" align='center'>
        <tr>
            <th class="head">
                Sub-Packages and Classes
            </th>
            <th class="head">
                &nbsp;
            </th>
            <th class="head">
                Dev
            </th>
            <th class="head">
                Comments
            </th>
        </tr>
        <tr>
            <td>
                <em>sbpy.imageanalysis.comettools</em>
            </td>
            <td align='center'>
                <span class="planned"></span>
            </td>
            <td align='center'>
                MSK
            </td>
            <td>
                comet coma image enhancement tools and image handling tbd.
            </td>
        </tr>
        <tr>
            <td>
                <em>sbpy.imageanalysis.psfsubtraction</em>
            </td>
            <td align='center'>
                <span class="planned"></span>
            </td>
            <td align='center'>
                MM
            </td>
            <td>
                 PSF subtraction techniques and wrappers tbd.
            </td>
        </tr>
    </table>


`sbpy.obsutil`
--------------

.. raw:: html

    <table border="1" class="docutils stability" align='center'>
        <tr>
            <th class="head">
                Sub-Packages and Classes
            </th>
            <th class="head">
                &nbsp;
            </th>
            <th class="head">
                Dev
            </th>
            <th class="head">
                Comments
            </th>
        </tr>
        <tr>
            <td>
                <em>sbpy.obsutil</em>
            </td>
            <td align='center'>
                <span class="planned"></span>
            </td>
            <td align='center'>
                MSK/MM
            </td>
            <td>
                finder charts, general observability and peak observability, planning tools, etc. tbd.
            </td>
        </tr>
    </table>


`sbpy.bib`
----------

.. raw:: html

    <table border="1" class="docutils stability" align='center'>
        <tr>
            <th class="head">
                Sub-Packages and Classes
            </th>
            <th class="head">
                &nbsp;
            </th>
            <th class="head">
                Dev
            </th>
            <th class="head">
                Comments
            </th>
        </tr>
        <tr>
            <td>
                <em>sbpy.bib</em>
            </td>
            <td align='center'>
                <span class="stable"></span>
            </td>
            <td align='center'>
                MSK/MdVB/MM
            </td>
            <td>
                reference tracking and handling implemented, formatted output for reference lists 
            </td>
        </tr>
    </table>
