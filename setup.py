# cython --cplus -X boundscheck=False -X cdivision=True *.pyx
# python setup.py build_ext --inplace

from distutils.core import setup
from distutils.core import Extension
import os
from os.path import join
import numpy
import warnings

def configuration():
    from numpy.distutils.misc_util import Configuration

    config = Configuration('')
    
    config.add_extension(name='atoms',
                         sources=['atoms.cpp'])
    
    config.add_extension(name='cd_solver',
                         sources=['cd_solver.cpp'])
        
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup( **configuration().todict() )
