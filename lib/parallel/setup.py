import numpy as np
import glob
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext = Extension('_psparse',
    ['psparse.pyx', 'cs_gaxpy.c'],
    extra_compile_args=['-fopenmp', '-O3', '-ffast-math'],
    include_dirs = [np.get_include(), '.'],
    extra_link_args=['-fopenmp'])
    
setup(
    cmdclass = {'build_ext': build_ext},
    py_modules = ['psparse',],
    ext_modules = [ext]
)