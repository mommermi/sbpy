#include <Python.h>
#include <numpy/arrayobject.h>
#include "thermal.h"

static char module_docstring[] =
    "sbpy sub-module to calculate thermal model surface temperature distributions.";
static char integrate_planck_docstring[] =
  "Romberg Integrator for thermal model Planck function integration in one dimension\n\nThis function is not intended for use by the sbpy user; based on 'Numerical Recipes in C', Press et al. 1988, Cambridge University Press.\n\nParameters\n----------\nmodel : int\n    model integrand identifier (1: STM, 2: FRM, 3: NEATM)\na : float\n    lower boundary for integration (radians)\nb : float\n    upper boundary for integration (radians)\nwavelengths : iterable of n floats\n    n wavelengths at which to evaluate integral, one per epoch (micron)\nsubsolartemp : iterable of n floats\n    n subsolar temperatures (K), one per epoch\nphaseangle : iterable of n floats\n    n solar phase angles (radians, only relevant for NEATM), one for epoch\n\nReturns\n-------\results_array : numpy array of calculated flux densities with length n\n";

static PyObject *thermal_integrate_planck(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"integrate_planck", thermal_integrate_planck,
     METH_VARARGS, integrate_planck_docstring},
    {NULL, NULL, 0, NULL}
};

/* --------------------------- Module Interface -------------------------------- */

PyMODINIT_FUNC PyInit__thermal(void)
{
    
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_thermal",
        module_docstring,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    /* Load `numpy` functionality. */
    import_array();
    
    return module;
}

/* ----------------------------- Integrator Interface -------------------------- */

static PyObject *thermal_integrate_planck(PyObject *self, PyObject *args)
{
  int model;
  double a, b;
  PyObject *wavelengths_obj, *subsolartemp_obj, *phaseangle_obj;

 
  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "iddOOO", &model, &a, &b, &wavelengths_obj,
			&subsolartemp_obj, &phaseangle_obj))
        return NULL;

  PyObject *wavelengths_array = PyArray_FROM_OTF(wavelengths_obj,
						 NPY_DOUBLE,
						 NPY_IN_ARRAY);
  PyObject *subsolartemp_array = PyArray_FROM_OTF(subsolartemp_obj,
						  NPY_DOUBLE,
						  NPY_IN_ARRAY);
  PyObject *phaseangle_array = PyArray_FROM_OTF(phaseangle_obj,
						NPY_DOUBLE,
						NPY_IN_ARRAY);

  /* If that didn't work, throw an exception. */
  if ((wavelengths_array == NULL) || (subsolartemp_array == NULL) ||
      (phaseangle_array == NULL)) { 
    Py_XDECREF(wavelengths_array);
    return NULL;
  }
  
  /* extract number of wavelengths provided */
  int n_wavelengths = (int)PyArray_DIM(wavelengths_array, 0);
  
  /* extract number of ephemerides provided */
  int n_eph = (int)PyArray_DIM(phaseangle_array, 0);

  if (n_wavelengths != n_eph)
    {
      PyErr_SetString(PyExc_RuntimeError,
		      "require one wavelength per epoch.");
      return NULL;
    }
      
  /* extract Python arrays into C arrays */
  double *wavelengths = (double*)PyArray_DATA(wavelengths_array);
  double *subsolartemps = (double*)PyArray_DATA(subsolartemp_array);
  double *phaseangles = (double*)PyArray_DATA(phaseangle_array);  
    
  /* Call the integrator for each wavelength and append results to a list*/
  int i;
  double results[n_wavelengths];
  for (i=0; i<n_wavelengths; i++)
    {
      double value = integrate_planck(model, a, b, wavelengths[i],
				      subsolartemps[i], phaseangles[i]);
      
      /* Resolve error codes */
      switch ((int)value) {
      case -9999: 
	PyErr_SetString(PyExc_RuntimeError,
		  "Temperature distribution integration did not converge.");
	return NULL;
	break;
      case -9998:
	PyErr_SetString(PyExc_RuntimeError,
	       "Temperature distribution integration returned NaN.");
	return NULL;
	break;
      }

      /* assign value to results array */
      results[i] = value;
    }

  /* prepare output array */
  npy_intp dims[] = {n_eph};
  PyObject *results_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  memcpy(PyArray_DATA(results_array), results, sizeof(results));

  /* Clean up. */
  Py_DECREF(wavelengths_array);
  Py_DECREF(subsolartemp_array);
  Py_DECREF(phaseangle_array);  
  
  return results_array;
}
