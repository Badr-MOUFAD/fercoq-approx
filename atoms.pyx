# cython: profile=True

# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False atoms.pyx

# definitions in atoms.pxd

cdef DOUBLE INF = 1e30


def string_to_enum(func_string):
    if func_string[0] == 's':
        return SQUARE
    elif func_string[0] == 'a':
        return ABS
    elif func_string[0] == 'n':
        return NORM2
    elif func_string[0] == 'l':
        if func_string[1] == 'i':
            return LINEAR
        elif func_string[1] == 'o':
            if func_string[3] == '1':
                return LOG1PEXP
            if func_string[3] == 's':
                return LOGSUMEXP
    elif func_string[0] == 'b':
        return BOX_ZERO_ONE
    elif func_string[0] == 'e':
        return EQ_CONST
    elif func_string[0] == 'i':
        return INEQ_CONST
    elif func_string[0] == 'z':
        return ZERO
    else:
        return -1  # error


cdef DOUBLE val_conj_not_implemented(FUNCTION func,
                DOUBLE[:] x, DOUBLE[:] buff, int nb_coord) nogil:
    # Approximate f*(x) by sup <x, z> - f(z) - alpha/2. ||z||**2
    # with alpha very small (prone to numerical errors)
    cdef int i
    cdef DOUBLE val_conj = 0.
    for i in range(nb_coord):
        x[i] = INF * x[i]
    my_eval(func, x, buff, nb_coord, PROX, INF)
    for i in range(nb_coord):
        x[i] = x[i] / INF

    for i in range(nb_coord):
        val_conj += x[i] * buff[i]
        val_conj -= 0.5 / INF * buff[i]**2
    val_conj -= my_eval(func, buff, buff, nb_coord, VAL)
    return val_conj


cdef DOUBLE prox_conj(FUNCTION func, DOUBLE[:] x,
                        DOUBLE[:] buff, int nb_coord,
                        DOUBLE prox_param, DOUBLE prox_param2) nogil:
    # prox_{a f*}(x) = x - a prox{1/a f}(x/a)
    # prox_{a (ch)*}(y) = y - a prox{1/a (ch)}(y/a)
    cdef int i
    for i in range(nb_coord):
        x[i] /= prox_param  # trick to save a bit of memory
    my_eval(func, x, buff, nb_coord, PROX,
                prox_param=prox_param2/prox_param)
    for i in range(nb_coord):
        x[i] *= prox_param  # we undo the trick
        buff[i] = x[i] - prox_param * buff[i]
    return buff[0]


cdef DOUBLE my_eval(FUNCTION func, DOUBLE[:] x,
                        DOUBLE[:] buff, int nb_coord, MODE mode=VAL,
                        DOUBLE prox_param=1., DOUBLE prox_param2=1.) nogil:
    # Evaluate function func which is given as a chain of characters
    # (I did not manage to send lists of functions directly from python to cython)
    if mode == PROX_CONJ:
        return prox_conj(func, x, buff, nb_coord, prox_param, prox_param2)

    if func == SQUARE:
        return square(x, buff, nb_coord, mode, prox_param)
    elif func == ABS:
        return abs(x, buff, nb_coord, mode, prox_param)
    elif func == NORM2:
        return norm2(x, buff, nb_coord, mode, prox_param)
    elif func == LINEAR:
        return linear(x, buff, nb_coord, mode, prox_param)
    elif func == LOG1PEXP:
        return log1pexp(x, buff, nb_coord, mode, prox_param)
    elif func == LOGSUMEXP:
        return logsumexp(x, buff, nb_coord, mode, prox_param)
    elif func == BOX_ZERO_ONE:
        return box_zero_one(x, buff, nb_coord, mode, prox_param)
    elif func == EQ_CONST:
        return eq_const(x, buff, nb_coord, mode, prox_param)
    elif func == INEQ_CONST:
        return ineq_const(x, buff, nb_coord, mode, prox_param)
    elif func == ZERO:
        return zero(x, buff, nb_coord, mode, prox_param)
    else:
        return -INF
    # TODO: quadratic...


cdef inline DOUBLE square(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
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
    elif mode == LIPSCHITZ:
        buff[0] = 2.
        return buff[0]
    elif mode == IS_KINK:
        return 0
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(SQUARE, x, buff, nb_coord)
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

cdef inline DOUBLE abs(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
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


cdef inline DOUBLE norm2(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
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


cdef inline DOUBLE linear(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
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
    elif mode == LIPSCHITZ:
        buff[0] = 0.
        return buff[0]
    elif mode == IS_KINK:
        return 0
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(LINEAR, x, buff, nb_coord)
    else:  # mode == VAL
        for i in range(nb_coord):
            val += x[0]
        return val


cdef inline DOUBLE log1pexp(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
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
    elif mode == LIPSCHITZ:
        buff[0] = 1. / 4.
        return buff[0]
    elif mode == IS_KINK:
        return 0
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(LOG1PEXP, x, buff, nb_coord)
    else:  # mode == VAL
        for i in range(nb_coord):
            if x[i] > 30.:
                val += x[i]
            else:
                val += log(1.+exp(x[i]))
        return val


cdef inline DOUBLE logsumexp(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
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
    elif mode == LIPSCHITZ:
        buff[0] = 0.5
        return buff[0]
    elif mode == IS_KINK:
        return 0
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(LOGSUMEXP, x, buff, nb_coord)
    else:  # mode == VAL
        return max_x + log(sum_exp_x)

    
cdef inline DOUBLE box_zero_one(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
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
    elif mode == LIPSCHITZ:
        buff[0] = INF
        return buff[0]
    elif mode == IS_KINK:
        for i in range(nb_coord):
            if x[i] > 0 and x[i] < 1:
                return 0
        return 1
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(BOX_ZERO_ONE, x, buff, nb_coord)
    else:  # mode == VAL
        for i in range(nb_coord):
            if x[i] > 1.:
                val += INF
            elif x[i] < 0.:
                val += INF
        return val


cdef inline DOUBLE eq_const(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
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


cdef inline DOUBLE ineq_const(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
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
    elif mode == LIPSCHITZ:
        buff[0] = INF
        return buff[0]
    elif mode == IS_KINK:
        for i in range(nb_coord):
            if x[i] > 0:
                return 0
        return 1
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(INEQ_CONST, x, buff, nb_coord)
    else:  # mode == VAL
        for i in range(nb_coord):
            if x[i] < 0:
                val += INF
        return val

    
cdef inline DOUBLE zero(DOUBLE[:] x, DOUBLE[:] buff, int nb_coord, MODE mode, DOUBLE prox_param=1.) nogil:
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
    elif mode == LIPSCHITZ:
        buff[0] = 0.
        return buff[0]
    elif mode == IS_KINK:
        return 0
    elif mode == VAL_CONJ:
        return val_conj_not_implemented(ZERO, x, buff, nb_coord)
    else:  # mode == VAL
        return 0.

