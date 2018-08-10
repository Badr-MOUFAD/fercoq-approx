# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False atoms.pyx

# definitions in atoms.pxd
#from libc.math cimport fabs, sqrt, log2
#cimport numpy as np
#import numpy as np
#from scipy import linalg
#
#cimport cython
#import warnings
#
#ctypedef np.float64_t DOUBLE
#ctypedef np.int32_t INT32_t

cdef DOUBLE INF = 1e30

cdef DOUBLE my_eval(unsigned char* func_string, DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode=VAL,
                        DOUBLE prox_param=1.) nogil:
    # Evaluate function func which is given as a chain of characters
    # (I did not manage to send lists of functions directly from python to cython)
    if func_string[0] == "s":
        return square(x, buff, nb_coord, mode, prox_param)
    elif func_string[0] == "a":
        return abs(x, buff, nb_coord, mode, prox_param)
    elif func_string[0] == "l":
        if func_string[1] == "i":
            return linear(x, buff, nb_coord, mode, prox_param)
        elif func_string[1] == "o":
            return log1pexp(x, buff, nb_coord, mode, prox_param)
    elif func_string[0] == "b":
        return box_zero_one(x, buff, nb_coord, mode, prox_param)
    elif func_string[0] == "z":
        return zero(x, buff, nb_coord, mode, prox_param)
        
        

cdef DOUBLE square(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
    # Function x -> x**2
    if mode == GRAD:
        buff[0] = 2. * x[0]
        return buff[0]
    elif mode == PROX:
        buff[0] = x[0] / (1. + 2. * prox_param)
        return buff[0]
    elif mode == LIPSCHITZ:
        buff[0] = 2.
        return buff[0]
    else:  # mode == VAL
        return x[0] * x[0]

    
cdef inline DOUBLE sign(DOUBLE x) nogil:
    if x < 0:
        return -1
    return 1

cdef inline DOUBLE max(DOUBLE x, DOUBLE y) nogil:
    if x < y:
        return y
    return x

cdef inline DOUBLE min(DOUBLE x, DOUBLE y) nogil:
    if x < y:
        return x
    return y

cdef DOUBLE abs(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
    # Function x -> |x|
    if mode == GRAD:
        buff[0] = sign(x[0])
        return buff[0]
    elif mode == PROX:
        buff[0] = sign(x[0]) * max(0., fabs(x[0]) - prox_param)
        return buff[0]
    elif mode == LIPSCHITZ:
        buff[0] = INF
        return buff[0]
    else:  # mode == VAL
        return fabs(x[0])


cdef DOUBLE linear(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
    # Function x -> x
    if mode == GRAD:
        buff[0] = 1.
        return buff[0]
    elif mode == PROX:
        buff[0] = x[0] - prox_param
        return buff[0]
    elif mode == LIPSCHITZ:
        buff[0] = 0.
        return buff[0]
    else:  # mode == VAL
        return x[0]


cdef DOUBLE log1pexp(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
    # Function x in [0,1]
    cdef DOUBLE exp_x
    if mode == GRAD:
        buff[0] = 0.
        if x[0] > 0.:
            buff[0] = 1. / (1. + exp(-x[0]))
        else:
            exp_x = exp(x[0])
            buff[0] = exp_x / (1. + exp_x)
        return buff[0]
    elif mode == PROX:
        # not coded yet
        buff[0] = 1e30
        return buff[0]
    elif mode == LIPSCHITZ:
        buff[0] = 1. / 4.
        return buff[0]
    else:  # mode == VAL
        if x[0] > 30.:
            return x[0]
        return log(1.+exp(x[0]))


cdef DOUBLE box_zero_one(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
    # Function x in [0,1]
    if mode == GRAD:
        buff[0] = 0.
        return buff[0]
    elif mode == PROX:
        buff[0] = min(1., max(0., x[0]))
        return buff[0]
    elif mode == LIPSCHITZ:
        buff[0] = INF
        return buff[0]
    else:  # mode == VAL
        if x[0] > 1.:
            return INF
        elif x[0] < 0.:
            return INF
        return 0

    
cdef DOUBLE zero(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
    # Function x -> 0
    if mode == GRAD:
        buff[0] = 0.
        return buff[0]
    elif mode == PROX:
        buff[0] = x[0]
        return buff[0]
    elif mode == LIPSCHITZ:
        buff[0] = 0.
        return buff[0]
    else:  # mode == VAL
        return 0.

