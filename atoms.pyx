# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False atoms.pyx

# definitions in atoms.pxd
import numpy as np

# Atom support for gap safe screening is defined in the file screening.pyx


cdef DOUBLE INF = 1e20

cdef atom string_to_func(bytes func_string):
    if func_string[0] == 's':
        return square
    elif func_string[0] == 'a':
        return abs
    elif func_string[0] == 'n':
        return norm2
    elif func_string[0] == 'l':
        if func_string[1] == 'i':
            return linear
        elif func_string[1] == 'o':
            if func_string[3] == '1':
                return log1pexp
            if func_string[3] == 's':
                return logsumexp
    elif func_string[0] == 'b':
        return box_zero_one
    elif func_string[0] == 'e':
        return eq_const
    elif func_string[0] == 'i':
        return ineq_const
    elif func_string[0] == 'z':
        return zero
    else:
        return error_atom  # error


cdef DOUBLE val_conj_not_implemented(atom func,
                DOUBLE[:] x, DOUBLE[:] buff, int nb_coord) nogil:
    # Approximate f*(x) by sup <x, z> - f(z) - alpha/2. ||z||**2
    # with alpha very small (prone to numerical errors)
    cdef int i
    cdef DOUBLE val_conj = 0.
    for i in range(nb_coord):
        x[i] = INF * x[i]
    func(x, buff, nb_coord, PROX, INF, useless_param)
    for i in range(nb_coord):
        x[i] = x[i] / INF

    for i in range(nb_coord):
        val_conj += x[i] * buff[i]
        val_conj -= 0.5 / INF * buff[i]**2
    val_conj -= func(buff, buff, nb_coord, VAL, useless_param, useless_param)
    return val_conj


cdef DOUBLE prox_conj(atom func, DOUBLE[:] x,
                        DOUBLE[:] buff, int nb_coord,
                        DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # prox_{a f*}(x) = x - a prox{1/a f}(x/a)
    # prox_{a (ch)*}(y) = y - a prox{1/a (ch)}(y/a)
    cdef int i
    for i in range(nb_coord):
        x[i] /= prox_param  # trick to save a bit of memory
    func(x, buff, nb_coord, PROX,
                prox_param2/prox_param, 1.)
    for i in range(nb_coord):
        x[i] *= prox_param  # we undo the trick
        buff[i] = x[i] - prox_param * buff[i]
    return buff[0]


cdef DOUBLE square(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # Function x -> x**2
    cdef int i
    cdef DOUBLE val = 0.
    if mode == GRAD:
        for i in range(nb_coord):
            buff[i] = 2. * x[i]
        return buff[0]
    elif mode == PROX:
        for i in range(nb_coord):
            buff[i] = x[i] / (1. + 2. * prox_param)
        return buff[0]
    elif mode == PROX_CONJ:
        return prox_conj(square, x, buff, nb_coord, prox_param, prox_param2)
    elif mode == LIPSCHITZ:
        buff[0] = 2.
        return buff[0]
    elif mode == IS_KINK:
        return 0
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(square, x, buff, nb_coord)
    else:  # mode == VAL
        for i in range(nb_coord):
            val += x[i] * x[i]
        return val

    
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

cdef DOUBLE abs(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # Function x -> |x|
    cdef int i
    cdef DOUBLE val
    if mode == GRAD:
        for i in range(nb_coord):
            buff[i] = sign(x[i])
        return buff[0]
    elif mode == PROX:
        for i in range(nb_coord):
            buff[i] = sign(x[i]) * max(0., fabs(x[i]) - prox_param)
        return buff[0]
    elif mode == PROX_CONJ:
        return prox_conj(abs, x, buff, nb_coord, prox_param, prox_param2)
    elif mode == LIPSCHITZ:
        buff[0] = INF
        return buff[0]
    elif mode == IS_KINK:
        for i in range(nb_coord):
            if x[i] != 0:
                return 0
        return 1
    elif mode == VAL_CONJ:
        for i in range(nb_coord):
            if fabs(x[i]) > 1.00000001:
                return INF
        return 0 # val_conj_not_implemented(ABS, x, buff, nb_coord)
    else:  # mode == VAL
        val = 0.
        for i in range(nb_coord):
            val += fabs(x[i])
        return val


cdef DOUBLE norm2(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # Function x -> ||x||_2
    # the dimension of the space on which we compute the norm is given by nb_coord
    cdef int i
    cdef DOUBLE val = 0.
    for i in range(nb_coord):
        val += x[i] ** 2
    val = sqrt(val)

    if mode == GRAD:
        if val != 0:
            for i in range(nb_coord):
                buff[i] = x[i] / val
        else:
            for i in range(nb_coord):
                buff[i] = 0
        return buff[0]
    elif mode == PROX:
        if val > prox_param:
            for i in range(nb_coord):
                buff[i] = x[i] * (1. - prox_param / val)
        else:
            for i in range(nb_coord):
                buff[i] = 0.
        return buff[0]
    elif mode == PROX_CONJ:
        return prox_conj(norm2, x, buff, nb_coord, prox_param, prox_param2)
    elif mode == LIPSCHITZ:
        buff[0] = INF
        return buff[0]
    elif mode == IS_KINK:
        for i in range(nb_coord):
            if x[i] != 0:
                return 0
        return 1
    elif mode == VAL_CONJ:
        if val > 1.00000001:
            return INF
        return 0.
    else:  # mode == VAL
        return val


cdef DOUBLE linear(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # Function x -> x
    cdef int i
    cdef DOUBLE val = 0.
    if mode == GRAD:
        for i in range(nb_coord):
            buff[i] = 1.
        return buff[0]
    elif mode == PROX:
        for i in range(nb_coord):
            buff[i] = x[i] - prox_param
        return buff[0]
    elif mode == PROX_CONJ:
        return prox_conj(linear, x, buff, nb_coord, prox_param, prox_param2)
    elif mode == LIPSCHITZ:
        buff[0] = 0.
        return buff[0]
    elif mode == IS_KINK:
        return 0
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(linear, x, buff, nb_coord)
    else:  # mode == VAL
        for i in range(nb_coord):
            val += x[0]
        return val


cdef DOUBLE log1pexp(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # Function log(1+exp(x))
    cdef int i
    cdef DOUBLE val = 0.
    cdef DOUBLE exp_x
    if mode == GRAD:
        for i in range(nb_coord):
            if x[i] > 0.:
                buff[i] = 1. / (1. + exp(-x[i]))
            else:
                exp_x = exp(x[i])
                buff[i] = exp_x / (1. + exp_x)
        return buff[0]
    elif mode == PROX:
        # not coded yet
        for i in range(nb_coord):
            buff[i] = 1e30
        return buff[0]
    elif mode == PROX_CONJ:
        return prox_conj(log1pexp, x, buff, nb_coord, prox_param, prox_param2)
    elif mode == LIPSCHITZ:
        buff[0] = 1. / 4.
        return buff[0]
    elif mode == IS_KINK:
        return 0
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(log1pexp, x, buff, nb_coord)
    else:  # mode == VAL
        for i in range(nb_coord):
            if x[i] > 30.:
                val += x[i]
            else:
                val += log(1.+exp(x[i]))
        return val


cdef DOUBLE logsumexp(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # Function log(exp(x_0)+...+exp(x_n))
    cdef int i
    cdef DOUBLE max_x = x[0]
    cdef DOUBLE sum_exp_x = 0.

    if mode == GRAD or mode == VAL:
        for i in range(1, nb_coord):
            if x[i] > max_x:
                max_x = x[i]
        for i in range(nb_coord):
            sum_exp_x += exp(x[i] - max_x)

    if mode == GRAD:
        for i in range(nb_coord):
            buff[i] = exp(x[i] - max_x) / sum_exp_x
        return buff[0]
    elif mode == PROX:
        # not coded yet
        for i in range(nb_coord):
            buff[i] = 1e30
        return buff[0]
    elif mode == PROX_CONJ:
        return prox_conj(logsumexp, x, buff, nb_coord, prox_param, prox_param2)
    elif mode == LIPSCHITZ:
        buff[0] = 0.5
        return buff[0]
    elif mode == IS_KINK:
        return 0
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(logsumexp, x, buff, nb_coord)
    else:  # mode == VAL
        return max_x + log(sum_exp_x)

    
cdef DOUBLE box_zero_one(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # Function x in [0,1]
    cdef int i
    cdef DOUBLE val = 0.
    if mode == GRAD:
        for i in range(nb_coord):
            buff[i] = 0.
        return buff[0]
    elif mode == PROX:
        for i in range(nb_coord):
            buff[i] = min(1., max(0., x[i]))
        return buff[0]
    elif mode == PROX_CONJ:
        return prox_conj(box_zero_one, x, buff, nb_coord, prox_param, prox_param2)
    elif mode == LIPSCHITZ:
        buff[0] = INF
        return buff[0]
    elif mode == IS_KINK:
        for i in range(nb_coord):
            if x[i] > 0 and x[i] < 1:
                return 0
        return 1
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(box_zero_one, x, buff, nb_coord)
    else:  # mode == VAL
        for i in range(nb_coord):
            if x[i] > 1.:
                val += INF
            elif x[i] < 0.:
                val += INF
        return val


cdef DOUBLE eq_const(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # Function x == 0
    cdef int i
    cdef DOUBLE val = 0.
    if mode == GRAD:
        for i in range(nb_coord):
            buff[i] = 0.
        return buff[0]
    elif mode == PROX:
        for i in range(nb_coord):
            buff[i] = 0.
        return buff[0]
    elif mode == PROX_CONJ:
        return prox_conj(eq_const, x, buff, nb_coord, prox_param, prox_param2)
    elif mode == LIPSCHITZ:
        buff[0] = INF
        return buff[0]
    elif mode == IS_KINK:
        return 1
    elif mode == VAL_CONJ:
        return 0.
        # return val_conj_not_implemented(EQ_CONST, x, buff, nb_coord)
    else:  # mode == VAL
        for i in range(nb_coord):
            if x[i] > 0:
                val += INF
            elif x[i] < 0.:
                val += INF
        return val


cdef DOUBLE ineq_const(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # Function x >= 0
    cdef int i
    cdef DOUBLE val = 0.
    if mode == GRAD:
        for i in range(nb_coord):
            buff[i] = 0.
        return buff[0]
    elif mode == PROX:
        for i in range(nb_coord):
            buff[i] = max(0., x[i])
        return buff[0]
    elif mode == PROX_CONJ:
        return prox_conj(ineq_const, x, buff, nb_coord, prox_param, prox_param2)
    elif mode == LIPSCHITZ:
        buff[0] = INF
        return buff[0]
    elif mode == IS_KINK:
        for i in range(nb_coord):
            if x[i] > 0:
                return 0
        return 1
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(ineq_const, x, buff, nb_coord)
    else:  # mode == VAL
        for i in range(nb_coord):
            if x[i] < 0:
                val += INF
        return val

    
cdef DOUBLE zero(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # Function x -> 0
    cdef int i
    if mode == GRAD:
        for i in range(nb_coord):
            buff[i] = 0.
        return buff[0]
    elif mode == PROX:
        for i in range(nb_coord):
            buff[i] = x[i]
        return buff[0]
    elif mode == PROX_CONJ:
        return prox_conj(zero, x, buff, nb_coord, prox_param, prox_param2)
    elif mode == LIPSCHITZ:
        buff[0] = 0.
        return buff[0]
    elif mode == IS_KINK:
        return 0
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(zero, x, buff, nb_coord)
    else:  # mode == VAL
        return 0.


cdef DOUBLE error_atom(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # Nonsense function
    cdef int i
    if mode == GRAD:
        for i in range(nb_coord):
            buff[i] = INF
        return buff[0]
    elif mode == PROX:
        for i in range(nb_coord):
            buff[i] = INF
        return buff[0]
    elif mode == PROX_CONJ:
        return prox_conj(error_atom, x, buff, nb_coord, prox_param, prox_param2)
    elif mode == LIPSCHITZ:
        buff[0] = INF
        return buff[0]
    elif mode == IS_KINK:
        return 0
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(error_atom, x, buff, nb_coord)
    else:  # mode == VAL
        return INF
