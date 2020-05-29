from distutils.core import setup

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages

module = [Extension('cython_explainer', sources=['cython_explainer.pyx'])]

setup(
    name='cython_explainer',
    version='0.0.1',
    find_packages=find_packages(),
    ext_modules=cythonize(module),
    include_dirs=[numpy.get_include()]
)
