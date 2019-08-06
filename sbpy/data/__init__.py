# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
sbpy.data
---------

:author: Michael Mommert (mommermiscience@gmail.com)
"""


class Conf():

    # acceptable field names for DataClass
    fieldnames_info = [
        # General
        {'description': 'Target Identifier',
         'fieldnames': ['targetname', 'id', 'Object'],
         'provenance': ['orbit', 'ephem', 'obs', 'phys'],
         'dimension': None},
        {'description': 'Target Designation',
         'fieldnames': ['desig', 'designation'],
         'provenance': ['orbit', 'ephem', 'obs', 'phys'],
         'dimension': None},
        {'description': 'Target Number',
         'fieldnames': ['number'],
         'provenance': ['orbit', 'ephem', 'obs', 'phys'],
         'dimension': None},
        {'description': 'Target Name',
         'fieldnames': ['name'],
         'provenance': ['orbit', 'ephem', 'obs', 'phys'],
         'dimension': None},

        {'description': 'Epoch',
         'fieldnames': ['epoch', 'datetime',
                        'Date', 'date', 'Time', 'time'],
         'provenance': ['orbit', 'ephem', 'obs'],
         'dimension': '`~astropy.time.Time`'},

        # Orbital Elements
        {'description': 'Semi-Major Axis',
         'fieldnames': ['a', 'sma'],
         'provenance': ['orbit'],
         'dimension': 'length'},
        {'description': 'Eccentricity',
         'fieldnames': ['e', 'ecc'],
         'provenance': ['orbit'],
         'dimension': None},
        {'description': 'Inclination',
         'fieldnames': ['i', 'inc', 'incl'],
         'provenance': ['orbit'],
         'dimension': 'angle'},
        {'description': 'Perihelion Distance',
         'fieldnames': ['q', 'periheldist'],
         'provenance': ['orbit'],
         'dimension': 'length'},
        {'description': 'Aphelion Distance',
         'fieldnames': ['Q', 'apheldist'],
         'provenance': ['orbit'],
         'dimension': 'length'},
        {'description': 'Longitude of the Ascending Node',
         'fieldnames': ['Omega', 'longnode', 'node'],
         'provenance': ['orbit'],
         'dimension': 'angle'},
        {'description': 'Argument of the Periapsis',
         'fieldnames': ['w', 'argper'],
         'provenance': ['orbit'],
         'dimension': 'angle'},
        {'description': 'Mean Anomaly',
         'fieldnames': ['M', 'mean_anom'],
         'provenance': ['orbit'],
         'dimension': 'angle'},
        {'description': 'True Anomaly',
         'fieldnames': ['v', 'true_anom', 'true_anomaly'],
         'provenance': ['orbit', 'ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Arc Length',
         'fieldnames': ['arc', 'arc_length'],
         'provenance': ['orbit'],
         'dimension': 'angle'},
        {'description': 'Delta-v',
         'fieldnames': ['delta_v', 'delta-v'],
         'provenance': ['orbit', 'phys'],
         'dimension': 'velocity'},
        {'description': 'Minimum Orbit Intersection Distance wrt Mercury',
         'fieldnames': ['moid_mercury'],
         'provenance': ['orbit', 'phys'],
         'dimension': 'length'},
        {'description': 'Minimum Orbit Intersection Distance wrt Earth',
         'fieldnames': ['moid_earth'],
         'provenance': ['orbit', 'phys'],
         'dimension': 'length'},
        {'description': 'Minimum Orbit Intersection Distance wrt Venus',
         'fieldnames': ['moid_venus'],
         'provenance': ['orbit', 'phys'],
         'dimension': 'length'},
        {'description': 'Minimum Orbit Intersection Distance wrt Mars',
         'fieldnames': ['moid_mars'],
         'provenance': ['orbit', 'phys'],
         'dimension': 'length'},
        {'description': 'Minimum Orbit Intersection Distance wrt Jupiter',
         'fieldnames': ['moid_jupiter'],
         'provenance': ['orbit', 'phys'],
         'dimension': 'length'},
        {'description': 'Minimum Orbit Intersection Distance wrt Saturn',
         'fieldnames': ['moid_saturn'],
         'provenance': ['orbit', 'phys'],
         'dimension': 'length'},
        {'description': 'Minimum Orbit Intersection Distance wrt Uranus',
         'fieldnames': ['moid_uranus'],
         'provenance': ['orbit', 'phys'],
         'dimension': 'length'},
        {'description': 'Minimum Orbit Intersection Distance wrt Neptune',
         'fieldnames': ['moid_neptune'],
         'provenance': ['orbit', 'phys'],
         'dimension': 'length'},
        {'description': 'Tisserand Parameter wrt Jupiter',
         'fieldnames': ['Tj', 'tj'],
         'provenance': ['orbit', 'phys'],
         'dimension': None},
        {'description': 'MPC Orbit Type',
         'fieldnames': ['mpc_orb_type'],
         'provenance': ['orbit', 'phys'],
         'dimension': None},
        {'description': 'Epoch of Perihelion Passage',
         'fieldnames': ['Tp'],
         'provenance': ['orbit'],
         'dimension': '`~astropy.time.Time`'},
        {'description': 'Orbital Period',
         'fieldnames': ['P', 'period'],
         'provenance': ['orbit', 'phys'],
         'dimension': 'time'},

        # Ephemerides properties
        {'description': 'Heliocentric Distance',
         'fieldnames': ['r', 'rh', 'r_hel', 'heldist'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'length'},
        {'description': 'Heliocentric Radial Velocity',
         'fieldnames': ['r_rate', 'rh_rate', 'rdot', 'r-dot', 'rhdot',
                        'rh-dot'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'velocity'},
        {'description': 'Distance to the Observer',
         'fieldnames': ['delta', 'Delta', 'obsdist'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'length'},
        {'description': 'Observer-Target Radial Velocity',
         'fieldnames': ['delta_rate', 'deltadot', 'delta-dot',
                        'deldot', 'del-dot'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'velocity'},
        {'description': 'Right Ascension',
         'fieldnames': ['ra', 'RA'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Declination',
         'fieldnames': ['dec', 'DEC'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Right Ascension Rate',
         'fieldnames': ['ra_rate', 'RA_rate', 'ra_rates', 'RA_rates',
                        'dRA', 'dra'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angular velocity'},
        {'description': 'RA*cos(Dec) Rate',
         'fieldnames': ['RA*cos(Dec)_rate', 'dra cos(dec)',
                        'dRA cos(Dec)', 'dra*cos(dec)', 'dRA*cos(Dec)'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angular velocity'},
        {'description': 'Declination Rate',
         'fieldnames': ['dec_rate', 'DEC_rate', 'Dec_rate', 'dec_rates',
                        'DEC_rates', 'Dec_rates', 'dDec', 'dDEC', 'ddec'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angular velocity'},
        {'description': 'Proper Motion',
         'fieldnames': ['mu', 'Proper motion'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angular velocity'},
        {'description': 'Proper Motion Direction',
         'fieldnames': ['Direction', 'direction'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Solar Phase Angle',
         'fieldnames': ['alpha', 'phaseangle', 'Phase', 'phase'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Solar Elongation Angle',
         'fieldnames': ['elong', 'solarelong',
                        'solarelongation', 'elongation', 'Elongation'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'V-band Magnitude',
         'fieldnames': ['V', 'Vmag'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'magnitude'},
        {'description': 'Heliocentric Ecliptic Longitude',
         'fieldnames': ['hlon', 'EclLon', 'ecllon', 'HelEclLon',
                        'helecllon'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Heliocentric Ecliptic Latitude',
         'fieldnames': ['hlat', 'EclLat', 'ecllat', 'HelEclLat',
                        'helecllat'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Horizontal Elevation',
         'fieldnames': ['el', 'EL', 'elevation', 'alt', 'altitude',
                        'Altitude'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Horizontal Azimuth',
         'fieldnames': ['az', 'AZ', 'azimuth'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Lunar Elongation',
         'fieldnames': ['lunar_elong', 'elong_moon', 'elongation_moon',
                        'lunar_elongation', 'lunarelong'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'X State Vector Component',
         'fieldnames': ['x', 'X', 'x_vec'],
         'provenance': ['orbit', 'ephem', 'obs'],
         'dimension': 'length'},
        {'description': 'Y State Vector Component',
         'fieldnames': ['y', 'Y', 'y_vec'],
         'provenance': ['orbit', 'ephem', 'obs'],
         'dimension': 'length'},
        {'description': 'Z State Vector Component',
         'fieldnames': ['z', 'Z', 'z_vec'],
         'provenance': ['orbit', 'ephem', 'obs'],
         'dimension': 'length'},
        {'description': 'X Velocity Vector Component',
         'fieldnames': ['vx', 'dx', 'dx/dt'],
         'provenance': ['orbit', 'ephem', 'obs'],
         'dimension': 'velocity'},
        {'description': 'Y Velocity Vector Component',
         'fieldnames': ['vy', 'dy', 'dy/dt'],
         'provenance': ['orbit', 'ephem', 'obs'],
         'dimension': 'velocity'},
        {'description': 'Z Velocity Vector Component',
         'fieldnames': ['vz', 'dz', 'dz/dt'],
         'provenance': ['orbit', 'ephem', 'obs'],
         'dimension': 'velocity'},
        {'description': 'X heliocentric position vector',
         'fieldnames': ['x_h', 'X_h'],
         'provenance': ['orbit', 'ephem', 'obs'],
         'dimension': 'length'},
        {'description': 'Y heliocentric position vector',
         'fieldnames': ['y_h', 'Y_h'],
         'provenance': ['orbit', 'ephem', 'obs'],
         'dimension': 'length'},
        {'description': 'Z heliocentric position vector',
         'fieldnames': ['z_h', 'Z_h'],
         'provenance': ['orbit', 'ephem', 'obs'],
         'dimension': 'length'},

        {'description': 'Comet Total Absolute Magnitude',
         'fieldnames': ['m1', 'M1'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'magnitude'},
        {'description': 'Comet Nuclear Absolute Magnitude',
         'fieldnames': ['m2', 'M2'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'magnitude'},
        {'description': 'Total Magnitude Scaling Factor',
         'fieldnames': ['k1', 'K1'],
         'provenance': ['ephem', 'obs'],
         'dimension': None},
        {'description': 'Nuclear Magnitude Scaling Factor',
         'fieldnames': ['k2', 'K2'],
         'provenance': ['ephem', 'obs'],
         'dimension': None},
        {'description': 'Phase Coefficient',
         'fieldnames': ['phase_coeff', 'Phase_coeff'],
         'provenance': ['ephem', 'obs'],
         'dimension': None},
        {'description': 'Information on Solar Presence',
         'fieldnames': ['solar_presence', 'Solar_presence'],
         'provenance': ['ephem', 'obs'],
         'dimension': None},
        {'description': 'Information on Moon and target status',
         'fieldnames': ['status_flag', 'Status_flag'],
         'provenance': ['ephem', 'obs'],
         'dimension': None},
        {'description': 'Apparent Right Ascension',
         'fieldnames': ['RA_app', 'ra_app'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Apparent Declination',
         'fieldnames': ['DEC_app', 'dec_app'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Azimuth Rate (dAZ*cosE)',
         'fieldnames': ['az_rate', 'AZ_rate'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angular velocity'},
        {'description': 'Elevation Rate (d(ELV)/dt)',
         'fieldnames': ['el_rate', 'EL_rate'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angular velocity'},
        {'description': 'Satellite Position Angle',
         'fieldnames': ['sat_pang', 'Sat_pang'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Local Sidereal Time',
         'fieldnames': ['siderealtime', 'Siderealtime'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'time'},
        {'description': 'Target Optical Airmass',
         'fieldnames': ['airmass', 'Airmass'],
         'provenance': ['ephem', 'obs'],
         'dimension': None},
        {'description': 'V Magnitude Extinction',
         'fieldnames': ['vmagex', 'Vmagex'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'magnitude'},
        {'description': 'Surface Brightness',
         'fieldnames': ['Surfbright', 'surfbright'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'magnitude/angle^2'},
        {'description': 'Fraction of Illumination',
         'fieldnames': ['frac_illum', 'Frac_illum'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'percent'},
        {'description': 'Illumination Defect',
         'fieldnames': ['defect_illum', 'Defect_illum'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Target-primary angular separation',
         'fieldnames': ['targ_sep', 'Targ_sep'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Target-primary visibility',
         'fieldnames': ['targ_vis', 'Targ_vis'],
         'provenance': ['ephem', 'obs'],
         'dimension': None},
        {'description': 'Angular width of target',
         'fieldnames': ['targ_width', 'Targ_width'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Apparent planetodetic longitude',
         'fieldnames': ['pldetic_long', 'Pldetic_long'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Apparent planetodetic latitude',
         'fieldnames': ['pldetic_lat', 'Pldetic_lat'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Apparent planetodetic Solar longitude',
         'fieldnames': ['pltdeticSol_long', 'PltdeticSol_long'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Apparent planetodetic Solar latitude',
         'fieldnames': ['pltdeticSol_lat', 'PltdeticSol_lat'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Target sub-solar point position angle',
         'fieldnames': ['subsol_ang', 'Subsol_ang'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Target sub-solar point angle distance',
         'fieldnames': ['subsol_dist', 'Subsol_dist'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Target North pole position angle',
         'fieldnames': ['npole_angle', 'Npole_angle'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Target North pole position distance',
         'fieldnames': ['npole_dist', 'Npole_dist'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Observation centric ecliptic longitude',
         'fieldnames': ['obs_ecl_long', 'Obs_ecl_long'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Observation centric ecliptic latitude',
         'fieldnames': ['obs_ecl_lat', 'Obs_ecl_lat'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'One-way light time',
         'fieldnames': ['lighttime', 'Lighttime'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'time'},
        {'description': 'Target center velocity wrt Sun',
         'fieldnames': ['vel_sun', 'Vel_sun'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'velocity'},
        {'description': 'Target center velocity wrt Observer',
         'fieldnames': ['vel_obs', 'Vel_obs'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'velocity'},
        {'description': 'Lunar illumination',
         'fieldnames': ['lun_illum', 'Lun_illum'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'percent'},
        {'description': 'Apparent interfering body elongation wrt observer',
         'fieldnames': ['ib_elong', 'IB_elong'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Interfering body illumination',
         'fieldnames': ['ib_illum', 'IB_illum'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'percent'},
        {'description': 'Observer primary target angle',
         'fieldnames': ['targ_angle_obs', 'Targ_angle_obs'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Orbital plane angle',
         'fieldnames': ['orbangle_plane', 'Orbangle_plane'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'deg'},
        {'description': 'Constellation ID containing target',
         'fieldnames': ['constellation', 'Constellation'],
         'provenance': ['ephem', 'obs'],
         'dimension': None},
        {'description': 'Target North Pole RA',
         'fieldnames': ['targ_npole_ra', 'targ_npole_RA'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Target North Pole DEC',
         'fieldnames': ['targ_npole_dec', 'targ_npole_DEC'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Galactic Longitude',
         'fieldnames': ['glx_long', 'Glx_long'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Galactic Latitude',
         'fieldnames': ['glx_lat', 'Glx_lat'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Local apparent solar time',
         'fieldnames': ['solartime'],
         'provenance': ['ephem', 'obs'],
         'dimension': None},
        {'description': 'Observer light time from Earth',
         'fieldnames': ['earthlighttime', 'Earthlighttime'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'time'},
        {'description': '3 sigma positional uncertainty RA',
         'fieldnames': ['RA_3sigma', 'ra_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': '3 sigma positional uncertainty DEC',
         'fieldnames': ['DEC_3sigma', 'dec_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': '3 sigma positional uncertainty semi-major axis',
         'fieldnames': ['sma_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': '3 sigma positional uncertainty semi-minor axis',
         'fieldnames': ['smi_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': '3 sigma positional uncertainty position angle',
         'fieldnames': ['posangle_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': '3 sigma positional uncertainty ellipse area',
         'fieldnames': ['area_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angular area'},
        {'description': '3 sigma positional uncertainty root sum square',
         'fieldnames': ['rss_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': '3 sigma range uncertainty',
         'fieldnames': ['r_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'length'},
        {'description': '3 sigma range rate uncertainty',
         'fieldnames': ['r_rate_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'velocity'},
        {'description': '3 sigma doppler radar uncertainty at S-band',
         'fieldnames': ['sband_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'frequency'},
        {'description': '3 sigma doppler radar uncertainty at X-band',
         'fieldnames': ['xband_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'frequency'},
        {'description': '3 sigma doppler round-trip delay uncertainty',
         'fieldnames': ['dopdelay_3sigma'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'time'},
        {'description': 'Local apparent hour angle',
         'fieldnames': ['locapp_hourangle'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'time'},
        {'description': 'True phase angle',
         'fieldnames': ['true_phaseangle'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Phase angle bisector longitude',
         'fieldnames': ['pab_long'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Phase angle bisector latitude',
         'fieldnames': ['pab_lat'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Absolute V-band Magnitude',
         'fieldnames': ['abs_V', 'abs_Vmag'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'magnitude'},
        {'description': 'Satellite X-position',
         'fieldnames': ['sat_X', 'sat_x'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Satellite Y-position',
         'fieldnames': ['sat_y', 'sat_Y'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},
        {'description': 'Atmospheric Refraction',
         'fieldnames': ['atm_refraction', 'refraction'],
         'provenance': ['ephem', 'obs'],
         'dimension': 'angle'},

        # Physical properties (dependent on other properties)
        {'description': 'Infrared Beaming Parameter',
         'fieldnames': ['eta', 'Eta'],
         'provenance': ['ephem', 'obs'],
         'dimension': None},
        {'description': 'Temperature',
         'fieldnames': ['temp', 'Temp', 'temperature', 'Temperature'],
         'provenance': ['phys', 'ephem', 'obs'],
         'dimension': 'temperature'},


        # Physical properties (static)
        {'description': 'Effective Diameter',
         'fieldnames': ['d', 'D', 'diam', 'diameter', 'Diameter'],
         'provenance': ['phys'],
         'dimension': 'length'},
        {'description': 'Effective Radius',
         'fieldnames': ['R', 'radius'],
         'provenance': ['phys'],
         'dimension': 'length'},
        {'description': 'Geometric Albedo',
         'fieldnames': ['pv', 'pV', 'p_v', 'p_V', 'geomalb'],
         'provenance': ['phys'],
         'dimension': None},
        {'description': 'Bond Albedo',
         'fieldnames': ['A', 'bondalbedo'],
         'provenance': ['phys'],
         'dimension': None},
        {'description': 'Emissivity',
         'fieldnames': ['emissivity', 'Emissivity'],
         'provenance': ['phys'],
         'dimension': None},
        {'description': 'Absolute Magnitude',
         'fieldnames': ['absmag', 'H'],
         'provenance': ['phys', 'ephem', 'orbit'],
         'dimension': 'magnitude'},
        {'description': 'Photometric Phase Slope Parameter',
         'fieldnames': ['G', 'slope'],
         'provenance': ['phys', 'ephem', 'orbit'],
         'dimension': None},
        {'description': 'Molecule Identifier',
         'fieldnames': ['mol_tag', 'mol_name'],
         'provenance': ['phys'],
         'dimension': None},
        {'description': 'Transition frequency',
         'fieldnames': ['t_freq'],
         'provenance': ['phys'],
         'dimension': 'frequency'},
        {'description': 'Integrated line intensity at 300 K',
         'fieldnames': ['lgint300'],
         'provenance': ['phys'],
         'dimension': 'intensity'},
        {'description': 'Integrated line intensity at designated Temperature',
         'fieldnames': ['intl', 'lgint'],
         'provenance': ['phys'],
         'dimension': 'intensity'},
        {'description': 'Partition function at 300 K',
         'fieldnames': ['partfn300'],
         'provenance': ['phys'],
         'dimension': None},
        {'description': 'Partition function at designated temperature',
         'fieldnames': ['partfn'],
         'provenance': ['phys'],
         'dimension': None},
        {'description': 'Upper state degeneracy',
         'fieldnames': ['dgup'],
         'provenance': ['phys'],
         'dimension': None},
        {'description': 'Upper level energy in Joules',
         'fieldnames': ['eup_j', 'eup_J'],
         'provenance': ['phys'],
         'dimension': 'energy'},
        {'description': 'Lower level energy in Joules',
         'fieldnames': ['elo_j', 'elo_J'],
         'provenance': ['phys'],
         'dimension': 'energy'},
        {'description': 'Degrees of freedom',
         'fieldnames': ['degfr', 'ndf', 'degfreedom'],
         'provenance': ['phys'],
         'dimension': None},
        {'description': 'Einstein Coefficient',
         'fieldnames': ['au', 'eincoeff'],
         'provenance': ['phys'],
         'dimension': '1/time'},
        {'description': 'Timescale * r^2',
         'fieldnames': ['beta', 'beta_factor'],
         'provenance': ['phys'],
         'dimension': 'time * length^2'},
        {'description': 'Total Number',
         'fieldnames': ['totnum', 'total_number_nocd' 'total_number'],
         'provenance': ['phys'],
         'dimension': None},
        {'description': 'Column Density from Bockelee Morvan et al. 2004',
         'fieldnames': ['cdensity', 'col_density'],
         'provenance': ['phys'],
         'dimension': '1/length^2'},
        # {'description': '',
        #  'fieldnames': [],
        #  'provenance': [],
        #  'dimension': None},
    ]

    # use this code snippet to identify duplicate field names:
    # from sbpy.data import conf
    # import collections
    # a = sum(conf.fieldnames, [])
    # print([item for item, count in collections.Counter(a).items()
    #        if count > 1])

    # list of fieldnames; each element a list of alternatives
    fieldnames = [prop['fieldnames'] for prop in fieldnames_info]

    fieldname_idx = {}
    for idx, field in enumerate(fieldnames):
        for alt in field:
            fieldname_idx[alt] = idx

    # field equivalencies defining conversions
    # key defines target quantity; dict with source quantity and function
    # for conversion
    # conversions considered as part of DataClass._translate_columns
    field_eq = {'R': {'d': lambda r: r/2},
                # diameter to radius}
                'd': {'R': lambda d: d*2}
                }

    # definitions for use of pyoorb in Orbits
    oorb_timeScales = {'UTC': 1, 'UT1': 2, 'TT': 3, 'TAI': 4}
    oorb_elemType = {'CART': 1, 'COM': 2, 'KEP': 3, 'DEL': 4, 'EQX': 5}

    oorb_orbit_fields = {'COM': ['id', 'q', 'e', 'incl', 'Omega',
                                 'w', 'Tp_jd', 'orbtype', 'epoch',
                                 'epoch_scale', 'H', 'G'],
                         'KEP': ['id', 'a', 'e', 'incl', 'Omega', 'w', 'M',
                                 'orbtype', 'epoch', 'epoch_scale', 'H',
                                 'G'],
                         'CART': ['id', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                                  'orbtype', 'epoch', 'epoch_scale', 'H',
                                  'G']}
    oorb_orbit_units = {'COM': [None, 'au', None, 'deg', 'deg',
                                'deg', 'd', None, 'd',
                                None, 'mag', None],
                        'KEP': [None, 'au', None, 'deg', 'deg', 'deg', 'deg',
                                None, 'd', None, 'mag', None],
                        'CART': [None, 'au', 'au', 'au', 'au/d', 'au/d',
                                 'au/d', None, 'd', None, 'mag', None]}

    oorb_ephem_full_fields = [
        'MJD', 'RA', 'DEC', 'RA*cos(Dec)_rate', 'DEC_rate',
        'alpha', 'elong', 'r', 'Delta', 'V', 'pa', 'TopEclLon',
        'TopEclLat', 'OppTopEclLon', 'OppTopEclLat',
        'HelEclLon', 'HelEclLat', 'OppHelEclLon',
        'OppHelEclLat', 'EL', 'ELsun', 'ELmoon',
        'lunarphase', 'lunarelong', 'x', 'y', 'z',
        'vx', 'vy', 'vz', 'obsx', 'obsy', 'obsz',
        'trueanom']

    oorb_ephem_full_units = [
        'd', 'deg', 'deg', 'deg/d', 'deg/d', 'deg',
        'deg', 'au', 'au', 'mag', 'deg', 'deg',
        'deg', 'deg', 'deg',
        'deg', 'deg', 'deg',
        'deg', 'deg', 'deg', 'deg',
        None, 'deg', 'au', 'au', 'au',
        'au/d', 'au/d', 'au/d', 'au', 'au', 'au', 'deg']

    oorb_ephem_basic_fields = [
        'MJD', 'RA', 'DEC', 'RA*cos(Dec)_rate', 'DEC_rate',
        'alpha', 'elong', 'r', 'Delta', 'V', 'trueanom']

    oorb_ephem_basic_units = [
        'd', 'deg', 'deg', 'deg/d', 'deg/d', 'deg',
        'deg', 'au', 'au', 'mag', 'deg']

    # definitions for MPC orbits: MPC field name: [sbpy field name, unit]
    mpc_orbit_fields = {
        'absolute_magnitude': ['absmag', 'mag'],
        'aphelion_distance': ['Q', 'au'],
        'arc_length': ['arc', 'day'],
        'argument_of_perihelion': ['w', 'deg'],
        'ascending_node': ['Omega', 'deg'],
        'delta_v': ['delta_v', 'km/s'],
        'designation': ['desig', None],
        'earth_moid': ['moid_earth', 'au'],
        'eccentricity': ['e', None],
        'epoch_jd': ['epoch', 'time_jd_utc'],
        'inclination': ['i', 'deg'],
        'jupiter_moid': ['moid_jupiter', 'au'],
        'mars_moid': ['moid_mars', 'au'],
        'mean_anomaly': ['M', None],
        'mercury_moid': ['moid_mercury', None],
        'name': ['name', None],
        'number': ['number', None],
        'orbit_type': ['mpc_orbit_type', None],
        'perihelion_date_jd': ['Tp', 'time_jd_utc'],
        'perihelion_distance': ['q', 'au'],
        'period': ['P', 'year'],
        'phase_slope': ['G', None],
        'saturn_moid': ['moid_saturn', 'au'],
        'semimajor_axis': ['a', 'au'],
        'tisserand_jupiter': ['Tj', None],
        'uranus_moid': ['moid_uranus', 'au'],
        'venus_moid': ['moid_venus', 'au']
    }


conf = Conf()

from .core import DataClass, DataClassError, QueryError, TimeScaleWarning
from .decorators import *
from .ephem import Ephem
from .orbit import Orbit
from .phys import Phys
from .obs import Obs
from .names import Names, natural_sort_key

__all__ = ['DataClass', 'Ephem', 'Obs', 'Orbit', 'Phys', 'Names',
           'conf', 'Conf', 'DataClassError', 'quantity_to_dataclass',
           'QueryError', 'TimeScaleWarning']
