# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True algorithms.pyx

# C definitions in algorithms.pxd
import numpy as np
from scipy import sparse

cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


def find_dual_variables_to_update(UINT32_t n,
                                  UINT32_t[:] blocks, UINT32_t[:] blocks_h,
                                  UINT32_t[:] Ah_indptr, UINT32_t[:] Ah_indices,
                                  UINT32_t[:] inv_blocks_h, pb):

    cdef UINT32_t ii, i, j, l, lh, coord, nb_coord
    if n == 1:
        dual_vars_to_update_ = [np.arange(pb.Ah.nnz).tolist()]
        return dual_vars_to_update_

    dual_vars_to_update_ = [[] for ii in range(n)]
    for ii in range(n):
        nb_coord = blocks[ii+1] - blocks[ii]
        for i in range(nb_coord):
            coord = blocks[ii] + i
            for lh in range(Ah_indptr[coord], Ah_indptr[coord+1]):
                j = inv_blocks_h[Ah_indices[lh]]
                l = 0
                while l < len(dual_vars_to_update_[ii]) and \
                  inv_blocks_h[Ah_indices[<int> dual_vars_to_update_[ii][l]]] <= j:
                    l += 1
                dual_vars_to_update_[ii].insert(l, lh)

    return dual_vars_to_update_


cdef void dual_prox_operator(DOUBLE[:] buff, DOUBLE[:] buff_y,
                             DOUBLE[:] y, DOUBLE[:] prox_y,
                             DOUBLE[:] rhx, int len_pb_h, atom* h,
                             UINT32_t[:] blocks_h, DOUBLE[:] dual_step_size,
                             DOUBLE[:] ch) nogil:

    cdef UINT32_t j, l
    for j in range(len_pb_h):
        for l in range(blocks_h[j+1]-blocks_h[j]):
            buff_y[l] = y[blocks_h[j]+l] \
                             + rhx[blocks_h[j]+l] * dual_step_size[j]
        h[j](buff_y, buff, blocks_h[j+1]-blocks_h[j],
             PROX_CONJ, dual_step_size[j], ch[j])
        for l in range(blocks_h[j+1]-blocks_h[j]):
            prox_y[blocks_h[j]+l] = buff[l]


def compute_Ah_nnz_perrow(UINT32_t n, UINT32_t[:] Ah_nnz_perrow,
                          UINT32_t[:] blocks, UINT32_t[:] blocks_h,
                          UINT32_t[:] Ah_indptr, UINT32_t[:] Ah_indices,
                          UINT32_t[:] inv_blocks_h, pb, gather_blocks_h=False):
    cdef UINT32_t i, ii, lh, jh, j, coord, nb_coord, current_index
    Ah_block_summary_indptr = np.zeros(n+1, dtype=np.uint32)
    Ah_block_summary_indices = np.empty(pb.Ah.nnz, dtype=np.uint32)
    current_index = 0
    if gather_blocks_h == 1:
        len_summary = len(pb.h)
    else:
        len_summary = pb.Ah.shape[0]
    for ii in range(n):
        Ah_block_summary_indptr[ii+1] = Ah_block_summary_indptr[ii]
        nb_coord = blocks[ii+1] - blocks[ii]
        for i in range(nb_coord):
            coord = blocks[ii] + i
            for lh in range(Ah_indptr[coord], Ah_indptr[coord+1]):
                jh = Ah_indices[lh]
                if gather_blocks_h == 1:
                    j = inv_blocks_h[jh]
                else:
                    j = jh
                Ah_block_summary_indices[current_index] = j
                Ah_block_summary_indptr[ii+1] += 1
                current_index += 1
    Ah_block_summary = sparse.csc_matrix((np.ones(current_index),
                                          Ah_block_summary_indices[:current_index],
                                          Ah_block_summary_indptr),
                                         (len_summary, n))
    Ah_block_summary.tocoo().tocsc()  # remove double indices
    Ah_block_summary = (Ah_block_summary!=0)
    Ah_block_summary = np.maximum(1, np.array(Ah_block_summary.sum(axis=1)).ravel())
    if gather_blocks_h == 1:
        for j in range(len(pb.h)):
            for jh in range(blocks_h[j],blocks_h[j+1]):
                Ah_nnz_perrow[jh] = Ah_block_summary[j]
    else:
        for jh in range(pb.Ah.shape[0]):
            Ah_nnz_perrow[jh] = Ah_block_summary[jh]


cdef void one_step_coordinate_descent(DOUBLE[:] x,
        DOUBLE[:] y, DOUBLE[:] Sy, DOUBLE[:] prox_y,
        DOUBLE[:] rhx, DOUBLE[:] rf, DOUBLE[:] rhy, DOUBLE[:] rhy_ii,
        DOUBLE[:] rQ,
        DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff, DOUBLE[:] x_ii,
        DOUBLE[:] grad,
        UINT32_t[:] blocks, UINT32_t[:] blocks_f, UINT32_t[:] blocks_h,
        UINT32_t[:] Af_indptr, UINT32_t[:] Af_indices, DOUBLE[:] Af_data,
        DOUBLE[:] cf, DOUBLE[:] bf,
        DOUBLE[:] Dg_data, DOUBLE[:] cg, DOUBLE[:] bg,
        UINT32_t[:] Ah_indptr, UINT32_t[:] Ah_indices, DOUBLE[:] Ah_data,
        UINT32_t[:] inv_blocks_f,
        UINT32_t[:] inv_blocks_h, UINT32_t[:] Ah_nnz_perrow,
        UINT32_t[:] Ah_col_indices, UINT32_t[:,:] dual_vars_to_update,
        DOUBLE[:] ch, DOUBLE[:] bh,
        UINT32_t[:] Q_indptr, UINT32_t[:] Q_indices, DOUBLE[:] Q_data,
        atom* f, atom* g, atom* h,
        int f_present, int g_present, int h_present,
        DOUBLE[:] primal_step_size, DOUBLE[:] dual_step_size,
        int sampling_law, UINT32_t* rand_r_state,
        UINT32_t[:] active_set, UINT32_t n_active, 
        UINT32_t[:] focus_set, UINT32_t n_focus, UINT32_t n,
        UINT32_t per_pass,
        DOUBLE* change_in_x, DOUBLE* change_in_y) nogil:
    # Algorithm described in O. Fercoq and P. Bianchi (2015).
    #     A coordinate descent primal-dual algorithm with large step size
    #     and possibly non separable functions. arXiv preprint arXiv:1508.04625.

    # XXX out of bounds error for TV toy problem without f
    
    cdef UINT32_t ii, i, coord, j, jh, l, lh, jj, j_prev
    cdef UINT32_t nb_coord
    cdef int focus_on_kink_or_not = 0
    cdef DOUBLE dy, dxi
    if n_active == 0:
        return
    for f_iter in range(n * per_pass):
        if sampling_law == 0:
            ii = rand_int(n_active, rand_r_state)
            ii = active_set[ii]
        else:  # sampling_law == 1:
            # probability 1/2 to focus on non-kink points
            focus_on_kink_or_not = rand_int(2, rand_r_state)
            if focus_on_kink_or_not == 0 or n_focus == 0:
                ii = rand_int(n_active, rand_r_state)
                ii = active_set[ii]
            else:
                ii = rand_int(n_focus, rand_r_state)
                ii = focus_set[ii]  # focus_set is contained in active_set
    
        nb_coord = blocks[ii+1] - blocks[ii]
        if h_present is True:
            for i in range(nb_coord):
                coord = blocks[ii] + i
                rhy_ii[i] = rhy[coord]

            # Apply prox of h* in the dual space
            j = -10
            for i in range(dual_vars_to_update[ii][0]):
                lh = dual_vars_to_update[ii][1+i]
                jh = Ah_indices[lh]  # jh in [0, Ah.shape[0][
                j_prev = j
                j = inv_blocks_h[jh]
                if (i == 0 or j != j_prev):
                    for l in range(blocks_h[j+1]-blocks_h[j]):
                        buff_y[l] = Sy[blocks_h[j]+l] \
                                      + rhx[blocks_h[j]+l] * dual_step_size[j]
                    h[j](buff_y, buff,
                                blocks_h[j+1]-blocks_h[j],
                                PROX_CONJ,
                                dual_step_size[j],
                                ch[j])
                    for l in range(blocks_h[j+1]-blocks_h[j]):
                        prox_y[blocks_h[j]+l] = buff[l]
                # else: we have already computed prox_y[blocks_h[j]:blocks_h[j+1]], so nothing to do. Moreover, we can update Sy[jh] safely.

                # update y
                dy = prox_y[jh] - y[lh]
                y[lh] = prox_y[jh]
                change_in_y[0] += fabs(dy)
                # update Sy
                Sy[jh] += 1. / Ah_nnz_perrow[jh] * dy
                # update rhy
                rhy[Ah_col_indices[lh]] += Ah_data[lh] * dy

        for i in range(nb_coord):
            coord = blocks[ii] + i
            x_ii[i] = x[coord]

            # Compute gradient of quadratic form and do gradient step
            x[coord] -= primal_step_size[ii] * rQ[coord]

            # Compute gradient of f and do gradient step
            if f_present is True:
                grad[i] = 0.
                j = -10
                for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                    jj = Af_indices[l]
                    j_prev = j
                    j = inv_blocks_f[jj]
                    if f[j] == square:
                        # hard code for the important special case of square loss
                        grad[i] += 2 * cf[j] * Af_data[l] * rf[jj]
                    else:
                        if l == Af_indptr[coord] or j != j_prev:
                            f[j](rf[blocks_f[j]:blocks_f[j+1]], buff,
                                 blocks_f[j+1]-blocks_f[j], GRAD,
                                 useless_param, useless_param)
                        # else: we have already computed it
                        #   good for dense Af but not optimal for diagonal Af
                        grad[i] += cf[j] * Af_data[l] * buff[jj - blocks_f[j]]
                x[coord] -= primal_step_size[ii] * grad[i]
            if h_present is True:
                x[coord] -= primal_step_size[ii] * (2*rhy[coord] - rhy_ii[i])

        # Apply prox of g
        if g_present is True:
            for i in range(nb_coord):
                coord = blocks[ii] + i
                buff_x[i] = Dg_data[ii] * x[coord] - bg[coord]
            g[ii](buff_x, buff, nb_coord, PROX,
                  cg[ii]*Dg_data[ii]*Dg_data[ii]*primal_step_size[ii],
                  useless_param)
            for i in range(nb_coord):
                coord = blocks[ii] + i
                x[coord] = (buff[i] + bg[coord]) / Dg_data[ii]

        # Update residuals
        for i in range(nb_coord):
            coord = blocks[ii] + i
            if x_ii[i] != x[coord]:
                dxi = x[coord] - x_ii[i]
                for l in range(Q_indptr[coord], Q_indptr[coord+1]):
                    j = Q_indices[l]
                    rQ[j] += Q_data[l] * dxi
                if f_present is True:
                    for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                        j = Af_indices[l]
                        rf[j] += Af_data[l] * dxi
                if h_present is True:
                    for lh in range(Ah_indptr[coord], Ah_indptr[coord+1]):
                        jh = Ah_indices[lh]
                        rhx[jh] += Ah_data[lh] * dxi
                change_in_x[0] += fabs(dxi)
    return


# c function to solve the cubic
cdef DOUBLE root3(DOUBLE a, DOUBLE b, DOUBLE c, DOUBLE d) nogil:
    """
    Finds the unique positive root of
    a X**3 + b X**2 + c X + d  when a, b, c > 0 and d < 0
    cf https://en.wikipedia.org/wiki/Cubic_function for the formulas
    """
    cdef DOUBLE p
    cdef DOUBLE q
    cdef DOUBLE t
    cdef DOUBLE angle
    cdef DOUBLE r

    p = (3 * a * c - pow(b, 2) ) / 3. / pow(a,2)
    q = (2 * pow(b,3) - 9 * a * b * c + 27 * pow(a,2) * d) / 27. / pow(a,3)
    
    if 4 * pow(p,3) + 27 * pow(q,2) <= 0:  # three real roots
        angle = 1./3. * acos(3./2. * q / p * sqrt(-3 / p))
        t = 2 * sqrt(- p / 3.) * \
          fmax(fmax(cos(angle), cos(angle + 2./3. * M_PI)),
                   cos(angle - 2./3. * M_PI))
    else:  # one real root
        if p < 0:
            t = - 2 * copysign(sqrt(-p / 3.), q) * \
            cosh(1./3. * acosh(-3./2. * fabs(q) / p * sqrt(-3 / p)))
        else:
            t = - 2 * sqrt(p / 3.) * \
            sinh(1./3. * asinh(3./2. * q / p * sqrt(3 / p)))
    r = t - b / 3. / a
    return r


cdef DOUBLE next_theta(DOUBLE theta, int h_present=0) nogil:
    cdef DOUBLE theta2 = theta * theta
    if h_present is False:
        theta = (sqrt(theta2 * theta2 + 4 * theta2) - theta2) / 2.
    else:
        theta = root3(1., 1., theta2, -theta2)
    return theta


cdef void one_step_accelerated_coordinate_descent(DOUBLE[:] x,
        DOUBLE[:] xe, DOUBLE[:] xc, DOUBLE[:] y_center, DOUBLE[:] prox_y,
        DOUBLE[:] rhxe, DOUBLE[:] rhxc, DOUBLE[:] rfe, DOUBLE[:] rfc,
        DOUBLE[:] rhy, DOUBLE[:] rQe, DOUBLE[:] rQc,
        DOUBLE* theta, DOUBLE theta0, DOUBLE* c_theta,
        DOUBLE* beta,
        DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff, DOUBLE[:] xe_ii,
        DOUBLE[:] xc_ii, DOUBLE[:] grad,
        UINT32_t[:] blocks, UINT32_t[:] blocks_f, UINT32_t[:] blocks_h,
        UINT32_t[:] Af_indptr, UINT32_t[:] Af_indices, DOUBLE[:] Af_data,
        DOUBLE[:] cf, DOUBLE[:] bf,
        DOUBLE[:] Dg_data, DOUBLE[:] cg, DOUBLE[:] bg,
        UINT32_t[:] Ah_indptr, UINT32_t[:] Ah_indices, DOUBLE[:] Ah_data,
        UINT32_t[:] inv_blocks_f,
        UINT32_t[:] inv_blocks_h, UINT32_t[:] Ah_nnz_perrow,
        UINT32_t[:] Ah_col_indices, UINT32_t[:,:] dual_vars_to_update,
        DOUBLE[:] ch, DOUBLE[:] bh,
        UINT32_t[:] Q_indptr, UINT32_t[:] Q_indices, DOUBLE[:] Q_data,
        atom* f, atom* g, atom* h,
        int f_present, int g_present, int h_present,
        DOUBLE[:] Lf, DOUBLE[:] norm2_columns_Ah, 
        int sampling_law, UINT32_t* rand_r_state,
        UINT32_t[:] active_set, UINT32_t n_active, 
        UINT32_t[:] focus_set, UINT32_t n_focus, UINT32_t n,
        UINT32_t per_pass,
        DOUBLE* change_in_x) nogil:
    # if h_present is False and restart_period == 0:
    # Algorithm described in O. Fercoq and P. Richtárik. (2015).
    #   Accelerated, parallel, and proximal coordinate descent.
    #   SIAM Journal on Optimization, 25(4), 1997-2023.
    # if h_present is False and restart_period > 0:
    # Algorithm described in O. Fercoq and Z. Qu. (2018).
    #   Restarting the accelerated coordinate descent method with a rough
    #   strong convexity estimate. arXiv preprint arXiv:1803.05771.
    # if h_present is True:
    # Algorithm described in A. Alacaoglu, Q. Tran-Dinh, O. Fercoq and V. Cevher.
    #   (2017). Smooth primal-dual coordinate descent algorithms for nonsmooth
    #   convex optimization. In NIPS proceedings (pp. 5852-5861).

    cdef UINT32_t i, ii, coord, j, jh, l, lh, jj, j_prev
    cdef DOUBLE dy, dxei, dxci
    cdef DOUBLE primal_step_size
    cdef UINT32_t nb_coord
    
    for f_iter in range(n * per_pass):
        if sampling_law == 0:
            ii = rand_int(n_active, rand_r_state)
            ii = active_set[ii]
        else:  # sampling_law == 1:
            # probability 1/2 to focus on non-kink points
            focus_on_kink_or_not = rand_int(2, rand_r_state)
            if focus_on_kink_or_not == 0 or n_focus == 0:
                ii = rand_int(n_active, rand_r_state)
                ii = active_set[ii]
            else:
                ii = rand_int(n_focus, rand_r_state)
                ii = focus_set[ii]  # focus_set is contained in active_set
            
        primal_step_size = 1. / fmax(1e-30, Lf[ii] + norm2_columns_Ah[ii] / beta[0])
        nb_coord = blocks[ii+1] - blocks[ii]
        if h_present is True:
            # dual_step_size[:] = 1. / beta[0]
            # initialize rhy[blocks[ii]:blocks[ii+1]]
            for i in range(nb_coord):
                coord = blocks[ii] + i
                rhy[coord] = 0

            # Apply prox of h* in the dual space
            j = -10
            for i in range(dual_vars_to_update[ii][0]):
                lh = dual_vars_to_update[ii][1+i]
                jh = Ah_indices[lh]  # jh in [0, Ah.shape[0][
                j_prev = j
                j = inv_blocks_h[jh]
                if (i == 0 or j != j_prev):
                    for l in range(blocks_h[j+1]-blocks_h[j]):
                        jj = blocks_h[j]+l
                        buff_y[l] = y_center[jj] + (rhxe[jj] + c_theta[0] * rhxc[jj]) / beta[0]
                    h[j](buff_y, buff,
                                blocks_h[j+1]-blocks_h[j],
                                PROX_CONJ,
                                1./beta[0],
                                ch[j])
                    for l in range(blocks_h[j+1]-blocks_h[j]):
                        jj = blocks_h[j]+l
                        prox_y[jj] = buff[l]
                # else: we have already computed prox_y[blocks_h[j]:blocks_h[j+1]], so nothing to do.

                rhy[Ah_col_indices[lh]] += Ah_data[lh] * prox_y[jh]

        for i in range(nb_coord):
            coord = blocks[ii] + i
            xe_ii[i] = xe[coord]
            xc_ii[i] = xc[coord]
            
            # Compute gradient of quadratic form and do gradient step
            xe[coord] -= primal_step_size * theta0 / theta[0] * \
              (rQe[coord] + c_theta[0] * rQc[coord])

            # Compute gradient of f and do gradient step
            if f_present is True:
                grad[i] = 0.
                j = -10
                for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                    jj = Af_indices[l]
                    j_prev = j
                    j = inv_blocks_f[jj]
                    if f[j] == square:
                        # hard code for the important special case of square loss
                        grad[i] += 2 * cf[j] * Af_data[l] * (rfe[jj] + \
                                  c_theta[0] * rfc[jj])
                    else:
                        if l == Af_indptr[coord] or j != j_prev:
                            for jh in range(blocks_f[j+1] - blocks_f[j]):
                                buff_x[jh] = rfe[blocks_f[j] + jh] + \
                                  c_theta[0] * rfc[blocks_f[j] + jh]
                            f[j](buff_x, buff,
                                 blocks_f[j+1]-blocks_f[j], GRAD,
                                 useless_param, useless_param)
                        # else: we have already computed it
                        #   good for dense Af but not optimal for diagonal Af
                        grad[i] += cf[j] * Af_data[l] * buff[jj - blocks_f[j]]
                xe[coord] -= primal_step_size * theta0 / theta[0] * grad[i]
            if h_present is True:
                xe[coord] -= primal_step_size * theta0 / theta[0] * rhy[coord]

        # Apply prox of g
        if g_present is True:
            for i in range(nb_coord):
                coord = blocks[ii] + i
                buff_x[i] = Dg_data[ii] * xe[coord] - bg[coord]
            g[ii](buff_x, buff, nb_coord, PROX,
                  cg[ii]*Dg_data[ii]*Dg_data[ii]*primal_step_size*theta0/theta[0],
                  useless_param)
            for i in range(nb_coord):
                coord = blocks[ii] + i
                xe[coord] = (buff[i] + bg[coord])/ Dg_data[ii]

        # Update xc variable
        for i in range(nb_coord):
            coord = blocks[ii] + i
            xc[coord] -= (xe[coord] - xe_ii[i]) * \
                             (1. - theta[0] / theta0) / c_theta[0]

        # Update residuals
        for i in range(nb_coord):
            coord = blocks[ii] + i
            if xe_ii[i] != xe[coord]:
                dxei = xe[coord] - xe_ii[i]
                dxci = xc[coord] - xc_ii[i]
                for l in range(Q_indptr[coord], Q_indptr[coord+1]):
                    j = Q_indices[l]
                    rQe[j] += Q_data[j] * dxei
                    rQc[j] += Q_data[j] * dxci
                if f_present is True:
                    for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                        j = Af_indices[l]
                        rfe[j] += Af_data[l] * dxei
                        rfc[j] += Af_data[l] * dxci
                if h_present is True:
                    for lh in range(Ah_indptr[coord], Ah_indptr[coord+1]):
                        jh = Ah_indices[lh]
                        rhxe[jh] += Ah_data[lh] * dxei
                        rhxc[jh] += Ah_data[lh] * dxci
                change_in_x[0] += fabs(dxei)

        # Update momentum parameters
        theta[0] = next_theta(theta[0], h_present)
        c_theta[0] *= (1. - theta[0])
        if h_present is True:
            beta[0] /= (1. + theta[0])

    return


def variable_restart(restart_history, iter, restart_period, next_period,
                     fixed_restart_period=False):
    if (iter % next_period) != next_period - 1:
        return (False, next_period)
    else:
        if fixed_restart_period == False:
            j = 0
            while j < len(restart_history) and restart_history[j] == 1:
                j += 1
            for i in range(j):
                restart_history[i] = 0
            if j < len(restart_history):
                restart_history[j] = 1
            else:
                restart_history.append(1)

            return (True, restart_period * 2**j)
        else:
            return (True, restart_period)
