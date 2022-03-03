from setuptools import setup, find_packages, Extension, dist
from setuptools.command.build_ext import build_ext
# dist.Distribution().fetch_build_eggs(['numpy>=1.12'])
import numpy as np  # noqa
# import cython

descr = 'Efficient implementation of a generic CD solver'


DISTNAME = 'cd_solver'
DESCRIPTION = descr
MAINTAINER = 'Olivier Fercoq'
MAINTAINER_EMAIL = 'olivier.fercoq@telecom-paristech.fr'
LICENSE = 'Apache License, Version 2.0'
DOWNLOAD_URL = 'https://bitbucket.org/ofercoq/cd_solver.git'
URL = 'https://bitbucket.org/ofercoq/cd_solver/src'
VERSION = '0.1'

cython_directives = {'boundscheck': "False",
                     'cdivision': "True",
                     'wraparound': "False",
                     'language_level': "3"}

setup(name='cd_solver',
      description=DESCRIPTION,
      version=VERSION,
      long_description=open('README').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=find_packages(),
      #   setup_requires=["cython"],
      install_requires=["numpy", "scipy"],
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('cd_solver.atoms',
                    sources=['cd_solver/atoms.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives=cython_directives,
                    extra_compile_args=["-O3"]),
          Extension('cd_solver.helpers',
                    sources=['cd_solver/helpers.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives=cython_directives,
                    extra_compile_args=["-O3"]),
          Extension('cd_solver.algorithms',
                    sources=['cd_solver/algorithms.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives=cython_directives,
                    extra_compile_args=["-O3"]),
          Extension('cd_solver.algorithms_purecd',
                    sources=['cd_solver/algorithms_purecd.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives=cython_directives,
                    extra_compile_args=["-O3"]),
          Extension('cd_solver.screening',
                    sources=['cd_solver/screening.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives=cython_directives,
                    extra_compile_args=["-O3"]),
          Extension('cd_solver.cd_solver_',
                    sources=['cd_solver/cd_solver_.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    cython_directives=cython_directives,
                    extra_compile_args=["-O3"]),
      ],
      )
