import os

import numpy as np

from distutils.core import setup
from Cython.Distutils import build_ext, Extension


descr = 'Efficient implementation of a generic CD solver'


DISTNAME = 'cd_solver'
DESCRIPTION = descr
MAINTAINER = 'Olivier Fercoq'
MAINTAINER_EMAIL = 'olivier.fercoq@telecom-paristech.fr'
LICENSE = 'Apache License, Version 2.0'
DOWNLOAD_URL = 'https://bitbucket.org/ofercoq/cd_solver.git'
URL = 'https://bitbucket.org/ofercoq/cd_solver/src'


setup(name='cd_solver',
      description=DESCRIPTION,
      long_description=open('README').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('atoms',
                    sources=['atoms.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives={'boundscheck':False,
                                             'cdivision':True,
                                             'wraparound':False},
                    extra_compile_args=["-O3"]),
          Extension('helpers',
                    sources=['helpers.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives={'boundscheck':False,
                                             'cdivision':True,
                                             'wraparound':False},
                    extra_compile_args=["-O3"]),
          Extension('algorithms',
                    sources=['algorithms.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives={'boundscheck':False,
                                             'cdivision':True,
                                             'wraparound':False},
                    extra_compile_args=["-O3"]),
          Extension('algorithms_purecd',
                    sources=['algorithms_purecd.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives={'boundscheck':False,
                                             'cdivision':True,
                                             'wraparound':False},
                    extra_compile_args=["-O3"]),
          Extension('screening',
                    sources=['screening.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives={'boundscheck':False,
                                             'cdivision':True,
                                             'wraparound':False},
                    extra_compile_args=["-O3"]),
          Extension('cd_solver_',
                    sources=['cd_solver_.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives={'boundscheck':False,
                                             'cdivision':True,
                                             'wraparound':True},
                    extra_compile_args=["-O3"]),
                 ],
    )
