from setuptools import setup
import numpy
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

#libraries=["m"],
ext_modules=[
    Extension("mul_sparse",
              ["./pyleml/mul_sparse.pyx", "./pyleml/cs_gaxpy.c"],
              extra_compile_args=['-fopenmp', '-O3', '-ffast-math'],
              include_dirs=[numpy.get_include()],
              extra_link_args=['-fopenmp']
              )]

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='leml',
      version='0.1',
      cmdclass = {"build_ext": build_ext},
      ext_modules = ext_modules,
      description='Python version of the LEML algorithm',
      url='https://github.com/AnthonyMRios/pyleml',
      author='Anthony Rios',
      author_email='anthonymrios@gmail.com',
      classifiers=[
          'Intended Audience :: Science/Research'
      ],
      packages=['pyleml'],
      zip_safe=False)
