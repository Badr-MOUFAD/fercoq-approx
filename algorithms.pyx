# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True algorithms.pyx

# definitions in algorithms.pxd


def find_dual_variables_to_update(UINT32_t n,
                                  UINT32_t[:] blocks, UINT32_t[:] blocks_h,
                                  UINT32_t[:] Ah_indptr, UINT32_t[:] Ah_indices,
                                  UINT32_t[:] inv_blocks_h):

    cdef UINT32_t ii, i, j, l, lh, coord, nb_coord
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


cdef void one_step_coordinate_descent(int ii, DOUBLE[:] x,
        DOUBLE[:] y, DOUBLE[:] Sy, DOUBLE[:] prox_y,
        DOUBLE[:] rhx, DOUBLE[:] rf, DOUBLE[:] rhy, DOUBLE[:] rhy_ii,
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
        unsigned char** f, unsigned char** g, unsigned char** h,
        int f_present, int g_present, int h_present,
        DOUBLE[:] primal_step_size, DOUBLE[:] dual_step_size,
        DOUBLE* change_in_x, DOUBLE* change_in_y) nogil:

    cdef UINT32_t i, coord, j, jh, l, lh, jj, j_prev
    cdef UINT32_t nb_coord = blocks[ii+1] - blocks[ii]
    cdef DOUBLE dy

    if h_present is True:
        for i in range(nb_coord):
            coord = blocks[ii] + i
            rhy_ii[i] = rhy[coord]

        # Apply prox of h* in the dual space
        for i in range(dual_vars_to_update[ii][0]):
            lh = dual_vars_to_update[ii][1+i]
            jh = Ah_indices[lh]  # jh in [0, Ah.shape[0][
            j = inv_blocks_h[Ah_indices[lh]]
            if (i == 0 or j != inv_blocks_h[Ah_indices[dual_vars_to_update[ii][i]]]):
                for l in range(blocks_h[j+1]-blocks_h[j]):
                    buff_y[l] = Sy[blocks_h[j]+l] \
                                  + rhx[blocks_h[j]+l] * dual_step_size[jh]
                my_eval(h[j], buff_y, buff,
                            nb_coord=blocks_h[j+1]-blocks_h[j],
                            mode=PROX_CONJ,
                            prox_param=dual_step_size[jh],
                            prox_param2=ch[j])
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

        # Compute gradient of f and do gradient step
        if f_present is True:
            grad[i] = 0.
            for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                jj = Af_indices[l]
                if l > Af_indptr[coord]:
                    j_prev = j
                j = inv_blocks_f[jj]
                if l == Af_indptr[coord] or j != j_prev:
                    my_eval(f[j], rf[blocks_f[j]:blocks_f[j+1]], buff,
                            nb_coord=blocks_f[j+1]-blocks_f[j], mode=GRAD)
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
        my_eval(g[ii], buff_x, buff, nb_coord=nb_coord,
                    mode=PROX, prox_param=cg[ii]*Dg_data[ii]**2*primal_step_size[ii])
        for i in range(nb_coord):
            coord = blocks[ii] + i
            x[coord] = (buff[i] + bg[coord])/ Dg_data[ii]

    # Update residuals
    for i in range(nb_coord):
        coord = blocks[ii] + i
        if x_ii[i] != x[coord]:
            if f_present is True:
                for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                    j = Af_indices[l]
                    rf[j] += Af_data[l] * (x[coord] - x_ii[i])
            if h_present is True:
                for lh in range(Ah_indptr[coord], Ah_indptr[coord+1]):
                    jh = Ah_indices[lh]
                    rhx[jh] += Ah_data[lh] * (x[coord] - x_ii[i])
            change_in_x[0] += fabs(x_ii[i] - x[coord])
    return


cdef DOUBLE next_theta(DOUBLE theta, int h_present==0) nogil:
    return theta  # todo


cdef void one_step_accelerated_coordinate_descent(int ii, DOUBLE[:] x,
        DOUBLE[:] xe, DOUBLE[:] xc, DOUBLE[:] y_center, DOUBLE[:] prox_y,
        DOUBLE[:] rhxe, DOUBLE[:] rhxc, DOUBLE[:] rfe, DOUBLE[:] rfx,
        DOUBLE[:] rhy_ii, DOUBLE* theta, DOUBLE theta0, DOUBLE* c_theta,
        DOUBLE* beta,
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
        unsigned char** f, unsigned char** g, unsigned char** h,
        int f_present, int g_present, int h_present,
        DOUBLE[:] primal_step_size, DOUBLE[:] dual_step_size,
        DOUBLE* change_in_x, DOUBLE* change_in_y) nogil:

    cdef UINT32_t i, coord, j, jh, l, lh, jj, j_prev
    cdef UINT32_t nb_coord = blocks[ii+1] - blocks[ii]
    cdef DOUBLE dy

    if h_present is True:
        # initialize rhy_ii
        for i in range(nb_coord):
            coord = blocks[ii] + i
            rhy_ii[i] = 0

        # Apply prox of h* in the dual space
        for i in range(dual_vars_to_update[ii][0]):
            lh = dual_vars_to_update[ii][1+i]
            jh = Ah_indices[lh]  # jh in [0, Ah.shape[0][
            j = inv_blocks_h[Ah_indices[lh]]
            if (i == 0 or j != inv_blocks_h[Ah_indices[dual_vars_to_update[ii][i]]]):
                for l in range(blocks_h[j+1]-blocks_h[j]):
                    jj = blocks_h[j]+l
                    buff_y[l] = y_center[jj] + (rhxe[jj] + c_theta[0] * rhxc[jj]) \
                                        * dual_step_size[jh]
                my_eval(h[j], buff_y, buff,
                            nb_coord=blocks_h[j+1]-blocks_h[j],
                            mode=PROX_CONJ,
                            prox_param=dual_step_size[jh],
                            prox_param2=ch[j])
                for l in range(blocks_h[j+1]-blocks_h[j]):
                    jj = blocks_h[j]+l
                    prox_y[jj] = buff[l]
            # else: we have already computed prox_y[blocks_h[j]:blocks_h[j+1]], so nothing to do.

            rhy[Ah_col_indices[lh]] += Ah_data[lh] * prox_y[jh]

    for i in range(nb_coord):
        coord = blocks[ii] + i
        x_ii[i] = xe[coord]

        # Compute gradient of f and do gradient step
        if f_present is True:
            grad[i] = 0.
            for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                jj = Af_indices[l]
                if l > Af_indptr[coord]:
                    j_prev = j
                j = inv_blocks_f[jj]
                if l == Af_indptr[coord] or j != j_prev:
                    for jh in range(blocks_f[j+1] - blocks_f[j]):
                        buff_x[jh] = rfe[blocks_f[j] + jh] + \
                          c_theta[0] * rfc[blocks_f[j] + jh]
                    my_eval(f[j], buff_x, buff,
                            nb_coord=blocks_f[j+1]-blocks_f[j], mode=GRAD)
                # else: we have already computed it
                #   good for dense Af but not optimal for diagonal Af
                grad[i] += cf[j] * Af_data[l] * buff[jj - blocks_f[j]]
            xe[coord] -= primal_step_size[ii] * theta0 / theta[0] * grad[i]
        if h_present is True:
            xe[coord] -= primal_step_size[ii] * theta0 / theta[0] * rhy_ii[i]

    # Apply prox of g
    if g_present is True:
        for i in range(nb_coord):
            coord = blocks[ii] + i
            buff_x[i] = Dg_data[ii] * xe[coord] - bg[coord]
        my_eval(g[ii], buff_x, buff, nb_coord=nb_coord,
                    mode=PROX, prox_param=cg[ii]*Dg_data[ii]**2*primal_step_size[ii]*theta0/theta[0])
        for i in range(nb_coord):
            coord = blocks[ii] + i
            xe[coord] = (buff[i] + bg[coord])/ Dg_data[ii]

    for i in range(nb_coord):
        coord = blocks[ii] + i
        xc[coord] -= (xe[coord] - x_ii[i]) * \
                         (1 - theta[0] / theta0) / c_theta[0]


    ####  ----------  To code from here -------- ####
    # Update residuals
    for i in range(nb_coord):
        coord = blocks[ii] + i
        if x_ii[i] != x[coord]:
            if f_present is True:
                for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                    j = Af_indices[l]
                    rf[j] += Af_data[l] * (x[coord] - x_ii[i])
            if h_present is True:
                for lh in range(Ah_indptr[coord], Ah_indptr[coord+1]):
                    jh = Ah_indices[lh]
                    rhx[jh] += Ah_data[lh] * (x[coord] - x_ii[i])
            change_in_x[0] += fabs(x_ii[i] - x[coord])
    return
