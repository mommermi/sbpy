# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys
from distutils.extension import Extension

C_THERMAL_PKGDIR = os.path.relpath(os.path.dirname(__file__))

SRC_FILES = [os.path.join(C_THERMAL_PKGDIR, filename)
             for filename in ['src/thermal.c',
                              'src/_thermal.c']]

extra_compile_args = ['-UNDEBUG']
if not sys.platform.startswith('win'):
    extra_compile_args.append('-fPIC')


def get_extensions():
    # Add '-Rpass-missed=.*' to ``extra_compile_args`` when compiling with clang
    # to report missed optimizations
    _thermal_ext = Extension(name='sbpy.thermal._thermal',
                             sources=SRC_FILES,
                             extra_compile_args=extra_compile_args,
                             language='c')

    return [_thermal_ext]


# original
# setup(
#    ext_modules=[
#        Extension("_thermalmodels", ["_thermalmodels.c", "thermalmodels.c"])],
#    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
#)
