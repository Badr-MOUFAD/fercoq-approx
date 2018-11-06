# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True atoms.pyx


from libc.math cimport fabs, sqrt, log2, log, exp
from libc.math cimport pow, cos, acos, sinh, cosh, asinh, acosh, copysign, fmax
from libc.math cimport M_PI
cimport numpy as np

cimport cython

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t


cdef enum MODE:
    VAL = 0
    VAL_CONJ = 1
    GRAD = 2
    PROX = 3
    PROX_CONJ = 4
    LIPSCHITZ = 5
    IS_KINK = 6

cdef int STRING_LONG_ENOUGH = 6

cdef DOUBLE my_eval(unsigned char* func_string, DOUBLE[:] x,
                        DOUBLE[:] buff, int nb_coord, MODE mode=*,
                        DOUBLE prox_param=*, DOUBLE prox_param2=*) nogil

# func_string can be:
#    "square", "abs", "norm2", "linear", "log1pexp", "box_zero_one", 
#    "eq_const", "ineq_cont", "zero"
