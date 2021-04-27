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
    POLAR_SUPPORT_KINK = 7


cdef DOUBLE INF = 1e20
cdef DOUBLE useless_param = 0.

ctypedef DOUBLE (*atom)(DOUBLE[:], DOUBLE[:], int, MODE, DOUBLE, DOUBLE) nogil

cdef atom string_to_func(bytes func_string)


cdef DOUBLE square(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE abs(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE max0x(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE norm2(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE linear(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE log1pexp(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE logsumexp(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE box_zero_one(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE eq_const(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE ineq_const(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE second_order_cone(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE zero(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil

cdef DOUBLE error_atom(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil
