double integrate_planck (int model, double a, double b,
			 double wavelength, double T0, double phaseangle);
double integrand_stm (double alpha, double wavelength, double T0);
double integrand_frm (double latitude, double wavelength, double T0);
double integrand_neatm_longitude (double longitude, double wavelength, double T0);
double integrand_neatm_latitude (double latitude, double wavelength, double T0);
