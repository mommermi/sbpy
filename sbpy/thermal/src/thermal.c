#include "thermal.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

// setup for Romberg integrator and physics
#define JMAX 20          // number of iterations used in integration
#define EPS  1E-5        // relative accuracy required to stop integrator
#define HCK  14387.6866  // h*c/k using c in um/s
#define PI   M_PI
#define PI2  M_PI_2

// introduce auxiliary variables for NEATM
static double neatm_latitude_static;
static double neatm_longitude_upperlimit, neatm_longitude_lowerlimit;
static double neatm_phaseangle;

double integrate_planck (int model, double a, double b, double wavelength,
		  double T0, double phaseangle)
/* Romberg Integrator for thermal model Planck Function integration in one dimension
   this function is not intended for use by the sbpy user 
   based on "Numerical Recipes in C", Press et al. 1988, Cambridge University Press 
   (using the integration within C is 20 times faster than using scipy.integrate) 

   Parameters
   ----------
   model : int
       model integrand identifier (1: STM, 2: FRM, 3: NEATM)
   a : float
       lower boundary for integration (radians)
   b : float
       upper boundary for integration (radians)
   wavelengths : iterable of floats
       wavelengths at which to evaluate integral (micron)
   T0 : float 
       subsolar temperature (K)
   phaseangle : float 
       solar phase angle (radians, only relevant for NEATM) 

   Returns
   -------
   list : values of the integral at provided wavelengths
*/
{
  double I[JMAX+1][JMAX+1];
  double h[JMAX+1];
  double sum;
  int n, i, j;

  double (*integrand)(double, double, double);

  switch (model)
    {
    case 1:
      integrand = &integrand_stm;
      break;
    case 2:
      integrand = &integrand_frm;
      break;
    case 3:  // covers the NEATM latitude integration; longitude is handled internally
      integrand = &integrand_neatm_latitude;
      neatm_phaseangle = phaseangle;
      //define upper and lower integration limits for longitude
      if (phaseangle >= 0)
	{
	  neatm_longitude_lowerlimit = phaseangle-PI2;
	  neatm_longitude_upperlimit = PI2;
	}
	else
	{
	  neatm_longitude_lowerlimit = -PI2;
	  neatm_longitude_upperlimit = phaseangle+PI2;
	}
      break;
    case 30:  // covers the NEATM longitude integration; handled internally; do not use
      integrand = &integrand_neatm_longitude; 
      break;
    }
	
  //calculate I_{1,1} = trapezoidal rule
  h[1] = (b-a);
  I[1][1] = 0.5*h[1]*((*integrand)(a, wavelength, T0) 
		      + (*integrand)(b, wavelength, T0));

  for (n=2; n<=JMAX; n++)
    {	
      h[n] = h[n-1]*0.5;
		
      //calculate I_{n,1} using less calculation steps
      for (i=1, sum=0; i<=pow(2,(n-2)); i++)
	sum += (*integrand)(a+(2*i-1)*h[n], wavelength, T0);
      I[n][1] = 0.5*I[n-1][1] + h[n]*sum;
		
      //calculate I_{n,1} using existing results
      int m = n, k = 1;
      for (j=1; j<n; j++)
	{
	  k++;
	  m--;
			
	  //calculate I_{m,k}
	  I[m][k] = I[m+1][k-1] + (I[m+1][k-1] - I[m][k-1])
	    /(pow(2,(2*k-2)) - 1);
	}
		
      //check for nan
      if (isnan(I[1][n]))
	{
	  return -9998;  // error code for nan
	}
		
      //do at least 5 iterations
      if (n > 5)
	//check for convergence within epsilon and special case
	//of I[1][j] == 0 for (0<j<6)
	if ((fabs(I[1][n]-I[1][n-1])/I[1][n] < EPS) ||
	     (I[1][n] == 0 && I[1][n-1] == 0))
	  return I[1][n];
    }
  
  return -9999;  // error code for divergence
}


/* ------------------ thermal model integrand implementations ---------------------- */

/* Integrand for the Standard Thermal Model (STM, Morrison and Lebofsky 1979). Input 
   arguments for this integrand are:
   - alpha: angular distance between solar incidence and local zenith
            (subsolar point is located at alpha=0)
   - wavelength: wavelength (micron)
   - T0: subsolar temperature (K) */ 
double integrand_stm (double alpha, double wavelength, double T0)
{
	double exparg = HCK/(wavelength*T0*pow(cos(alpha), 0.25));
	
	if ((exparg > 50) || (isnan(exparg)))
	    return 0;
	else
	    return (cos(alpha)*sin(alpha)/(exp(exparg)-1));
}


/* Integrand for the Fast Rotating Model (FRM, Lebofsky and Spencer 1989). Input 
   arguments for this integrand are:
   - latitude: latitude on sphere (equator and subsolar point at latitude=0)
   - wavelength: wavelength (micron)
   - T0: subsolar temperature (K) */ 
double integrand_frm (double latitude, double wavelength, double T0)
{
	double exparg = HCK/(wavelength*T0*pow(cos(latitude), 0.25));
	
	if ((exparg > 50) || (isnan(exparg)))
	    return 0;
	else
	    return (cos(latitude)*cos(latitude)/(exp(exparg)-1));
}


/* Integrand for the longitude component of the  Near-Earth Asteroid Model (NEATM, 
   Harris 1998). Input arguments for this integrand are:
   - longitude: longitude on sphere (equator and subsolar point at latitude=0)
   - wavelength: wavelength (micron)
   - T0: subsolar temperature (K) 
   This function is only called internally. */ 
double integrand_neatm_longitude (double longitude, double wavelength, double T0)
{
  double exparg = HCK/(wavelength*T0*pow(cos(neatm_latitude_static)*
					 cos(longitude),0.25));
	
  if ((exparg > 50) || (isnan(exparg)))
    return 0;
  else
    return (cos(neatm_latitude_static)*cos(neatm_latitude_static)*
	    cos(longitude-neatm_phaseangle)/(exp(exparg)-1));
}

/* Integrand for the latitude component of the Near-Earth Asteroid Model (NEATM, 
   Harris 1998). Input arguments for this integrand are:
   - longitude: longitude on sphere (equator and subsolar point at latitude=0)
   - wavelength: wavelength (micron)
   - T0: subsolar temperature (K) 
   This function will handle the longitude intergration. */ 
double integrand_neatm_latitude (double latitude, double wavelength, double T0)
{
  neatm_latitude_static = latitude;

  // evaluate longitude integral internally (model code 30)
  return integrate_planck(30, neatm_longitude_lowerlimit, neatm_longitude_upperlimit,
			  wavelength, T0, neatm_phaseangle);
}
