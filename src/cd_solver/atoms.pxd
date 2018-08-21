# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True atoms.pyx


from libc.math cimport fabs, sqrt, log2, log, exp
cimport numpy as np
import numpy as np
from scipy import linalg

cimport cython
import warnings
from libc.stdlib cimport malloc, free

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT32_t


cdef enum MODE:
    VAL = 0
    GRAD = 1
    PROX = 2
    PROX_CONJ = 3
    LIPSCHITZ = 4

cdef int STRING_LONG_ENOUGH = 6

cdef DOUBLE my_eval(unsigned char* func_string, DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode=*,
                       DOUBLE prox_param=*, DOUBLE prox_param2=*) nogil

# func_string can be:
#    square
#    abs
#    zero