# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True algorithms.pyx

# C definitions in algorithms.pxd
import numpy as np
from algorithms import find_dual_variables_to_update


cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


def compute_theta_s_tri_pd(Ah, UINT32_t n,
                        UINT32_t[:] blocks, UINT32_t[:] blocks_h,
                        UINT32_t[:] Ah_indptr, UINT32_t[:] Ah_indices,
                        UINT32_t[:] inv_blocks_h, UINT32_t keep_all=0):
    cdef UINT32_t i, ii, lh, jh, j
    dual_vars_to_update_ = find_dual_variables_to_update(n, blocks, blocks_h,
                                  Ah_indptr, Ah_indices, inv_blocks_h)

    # At the moment, this function does not take into account a possible
    #   block diagonal structure of Ah.
    
    # update dual_vars_to_update in order to go from entries of the matrix
    #   to actual dual variable indices, and also include all computed entries
    #   if keep_all == 1.
    dual_vars_to_update_2 = [[] for ii in range(n)]
    for ii in range(n):
        j = -10
        for i in range(len(dual_vars_to_update_[ii])):
            lh = dual_vars_to_update_[ii][i]
            jh = Ah_indices[lh]  # jh in [0, Ah.shape[0][
            if keep_all == 1:
                j_prev = j
                j = inv_blocks_h[jh]
                if (i == 0 or j != j_prev):
                    for l in range(blocks_h[j+1]-blocks_h[j]):
                        jj = blocks_h[j]+l
                        dual_vars_to_update_2[ii].append(jj)
            else:
                lenlisti = len(dual_vars_to_update_2[ii])
                if (i == 0) or (dual_vars_to_update_2[ii][lenlisti - 1] != jh):
                    dual_vars_to_update_2[ii].append(jh)

    # Add dummy dependences if the probability to select a dual variable
    #   is not uniform
    # It would be better to add dummy dependences related to the blocks of h
    #   but we leave this for future work.
    m = max([len(dual_vars_to_update_2[ii]) for ii in range(n)])
    for ii in range(n):
        while len(dual_vars_to_update_2[ii]) < m:
            random_numbers = list(np.random.randint(0, Ah.shape[0],
                              m - len(dual_vars_to_update_2[ii])))
            dual_vars_to_update_2[ii] = sorted(set(dual_vars_to_update_2[ii]
                                                       + random_numbers))

    theta_s_tri_pd = [m] * n

    return theta_s_tri_pd, dual_vars_to_update_2


cdef void s_tri_pd(DOUBLE[:] x,
        DOUBLE[:] y, DOUBLE[:] prox_y, DOUBLE[:] rhx,
        DOUBLE[:] rhx_jj, DOUBLE[:] rf, DOUBLE[:] rQ,
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
        DOUBLE [:] theta_s_tri_pd,
        int sampling_law, UINT32_t* rand_r_state,
        UINT32_t[:] active_set, UINT32_t n_active, 
        UINT32_t[:] focus_set, UINT32_t n_focus, UINT32_t n,
        UINT32_t per_pass,
        DOUBLE* change_in_x, DOUBLE* change_in_y) nogil:
    # Algorithm developed by A. Alacaoglu and O. Fercoq
    # dual_vars_to_update contains indices in [0, Ah.shape[0])

    cdef UINT32_t ii, i, coord, j, jh, l, lh, jj, j_prev
    cdef UINT32_t nb_coord
    cdef int focus_on_kink_or_not = 0
    cdef DOUBLE dy, dxi
    cdef DOUBLE rhprox_y_i
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

            # Apply prox of h* in the dual space
            j = -10
            for i in range(dual_vars_to_update[ii][0]):
                jh = dual_vars_to_update[ii][1+i]
                j_prev = j
                j = inv_blocks_h[jh]
                if (i == 0 or j != j_prev):
                    for l in range(blocks_h[j+1]-blocks_h[j]):
                        rhx_jj[blocks_h[j]+l] = rhx[blocks_h[j]+l]
                        # we will need this backup later on
                        buff_y[l] = y[blocks_h[j]+l] \
                                      + rhx[blocks_h[j]+l] * dual_step_size[j]
                    h[j](buff_y, buff,
                                blocks_h[j+1]-blocks_h[j],
                                PROX_CONJ,
                                dual_step_size[j],
                                ch[j])
                    for l in range(blocks_h[j+1]-blocks_h[j]):
                        prox_y[blocks_h[j]+l] = buff[l]
                # else: we have already computed prox_y[blocks_h[j]:blocks_h[j+1]], so nothing to do.

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
                # compute rhproxy[i]
                rhprox_y_i = 0.
                for lh in range(Ah_indptr[coord], Ah_indptr[coord+1]):
                    rhprox_y_i += Ah_data[lh] * prox_y[Ah_indices[lh]]
                    # we are only accessing entries of prox_y that have been
                    #     updated above.
                # update x[i]
                x[coord] -= primal_step_size[ii] * rhprox_y_i

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

        # update y
        for i in range(nb_coord):
            coord = blocks[ii] + i
            for i in range(dual_vars_to_update[ii][0]):
                jh = dual_vars_to_update[ii][1+i]
                j = inv_blocks_h[jh]
                dy = prox_y[jh] + dual_step_size[j] * theta_s_tri_pd[ii] * \
                  (rhx[jh] - rhx_jj[jh]) - y[jh]
                y[jh] += dy
                change_in_y[0] += fabs(dy)

    return

