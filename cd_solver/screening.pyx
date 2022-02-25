# C definitions in screening.pxd
import numpy as np

cdef DOUBLE INF = 1e20


cdef int is_in_domain_polar_support(atom func, DOUBLE[:] x,
                                        UINT32_t nb_coord,
                                        DOUBLE radius, UINT32_t kink_number):
    # Checks whether the infinity-ball of center x and radius radius is included in
    #  the domain of the polar support of the subdifferential of the atom
    #  at a given kink. The other functions assume every point is in the domain
    cdef UINT32_t i
    if func == box_zero_one:
        for i in range(nb_coord):
            if kink_number % 2 == 0:
                # kink[i] = 0
                if x[i] - radius < 0:
                    return 0
            if kink_number % 2 == 1:
                # kink[i] = 1
                if x[i] + radius > 0:
                    return 0
            kink_number = kink_number // 2
        return 1
    elif func == ineq_const:
        for i in range(nb_coord):
            # kink[i] = 0
            if x[i] - radius < 0:
                return 0
        return 1
    return 1


cdef DOUBLE polar_matrix_norm(atom func, UINT32_t* Af_indptr,
                                  UINT32_t nb_coord, UINT32_t[:] Af_indices,
                                  DOUBLE[:] Af_data, DOUBLE Qii,
                                  UINT32_t kink_number):
    # equivalent of matrix norm for polar support function of the
    #    subdifferential of g at kink points
    # Af: sparse sub-matrix corresponding to the atom
    # Qii: nonzero if i-ith subblock of Q is nonzero
    cdef UINT32_t i, l
    cdef DOUBLE val = 0.
    cdef DOUBLE val_i
    if func == abs:
        if kink_number == 0:
            # val = np.max(np.sqrt(Af.multiply(Af).sum(axis=0)))
            for i in range(nb_coord):
                if Qii > 0:
                    val_i = 1
                else:
                    val_i = 0.
                for l in range(Af_indptr[i], Af_indptr[i+1]):
                    val_i += Af_data[l] * Af_data[l]
                val = fmax(val, val_i)
            return sqrt(val)
        else:
            return -1
    elif func == norm2:
        if kink_number == 0:
            # Frobenius norm is larger that spectral norm but easier to compute
            # val = sqrt(Af.multiply(Af).sum())
            for i in range(nb_coord):
                if Qii > 0:
                    val_i = nb_coord
                else:
                    val_i = 0.
                for l in range(Af_indptr[i], Af_indptr[i+1]):
                    val_i += Af_data[l] * Af_data[l]
                val += val_i
            return sqrt(val)
    elif func == box_zero_one:
        if nb_coord > 10:
            print('gap safe screening: warning, too many kinks to check, '
                       'consider using smaller blocks.')
        if kink_number >= 2 ** nb_coord:
            return -1
        else:
            return 0
            # we do not bother with domain issues here
            # (cf function is_in_domain_polar_support)
        
    # code for no more kink point
    return -1.


cdef DOUBLE polar_support_kink(atom func, DOUBLE[:] x,
                                   DOUBLE[:] kink,
                                   UINT32_t nb_coord, int kink_number):
    # polar support function of the subdifferential of g
    #     at kink points
    # returns also the kink
    cdef DOUBLE val
    cdef UINT32_t i
    if func == abs:
        if kink_number == 0:
            val = 0.
            for i in range(nb_coord):
                kink[i] = 0.
                val = fmax(val, fabs(x[i]))
            return val
    if func == norm2:
        if kink_number == 0:
            val = 0.
            for i in range(nb_coord):
                kink[i] = 0.
                val = val + x[i] * x[i]
            return sqrt(val)
    if func == box_zero_one:
        if kink_number >= 2 ** nb_coord:
            return -1
        val = 0.
        for i in range(nb_coord):
            if kink_number % 2 == 0:
                kink[i] = 0
                if x[i] < 0:
                    val += INF
            if kink_number % 2 == 1:
                kink[i] = 1
                if x[i] > 0:
                    val += INF
            kink_number = kink_number // 2
        return val

    # code for no more kink point            
    return -1.


cdef DOUBLE polar_support_dual_domain(atom func, DOUBLE[:] x,
                                          UINT32_t nb_coord) nogil:
    # polar support function of the domain of g*
    cdef DOUBLE val
    cdef UINT32_t i
    if func == abs:
       val = 0.
       for i in range(nb_coord):
           val = fmax(val, fabs(x[i]))
       return val
    if func == norm2:
       val = 0.
       for i in range(nb_coord):
           val = val + x[i] * x[i]
       return sqrt(val)
    if func == box_zero_one:
        return 0.
    # error code for not implemented
    return -1.


cdef DOUBLE dual_scaling(DOUBLE[:] z, DOUBLE[:] AfTz, DOUBLE[:] w,
                             UINT32_t n_active,
                             UINT32_t[:] active_set, pb, atom* g,
                             DOUBLE[:] buff_x):
    cdef UINT32_t i, ii, iii, coord, nb_coord
    cdef DOUBLE scaling = 1.
    cdef int raise_not_implemented_warning = 0
    for iii in range(n_active):
        ii = active_set[iii]
        nb_coord = pb.blocks[ii+1] - pb.blocks[ii]
        for i in range(nb_coord):
            coord = pb.blocks[ii] + i
            buff_x[i] = -AfTz[coord] - w[coord]
        norm_dom_g_i = polar_support_dual_domain(g[ii], buff_x, nb_coord) \
                           / (pb.cg[ii] * fabs(pb.Dg.data[0][ii]))  # fabs missing in the cd_solver paper
        scaling = fmax(scaling, norm_dom_g_i)
        if norm_dom_g_i < 0:
            raise_not_implemented_warning = 1
            break

    if raise_not_implemented_warning:
        print('polar_support_dual_domain not implemented for at least '
                          'one of the atoms: dual scaling skipped')
        scaling = 1.

    else:
        for i in range(len(np.array(z))):
            z[i] /= scaling
        for i in range(len(np.array(AfTz))):
            AfTz[i] /= scaling
        for i in range(len(np.array(w))):
            w[i] /= scaling

    return scaling



cdef UINT32_t do_gap_safe_screening(UINT32_t[:] active_set,
                              UINT32_t n_active_prev, pb,
                              atom* f, atom* g, atom* h, DOUBLE[:] Lf,
                              DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx,
                              DOUBLE[:] rQ, DOUBLE[:] prox_y,
                              DOUBLE[:] z, DOUBLE[:] AfTz,
                              DOUBLE[:] xe, DOUBLE[:] xc, DOUBLE[:] rfe,
                              DOUBLE[:] rfc, DOUBLE[:] rQe, DOUBLE[:] rQc,
                              DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                              g_norms_Af, norms_Af, DOUBLE max_Lf, int accelerated):
    cdef UINT32_t i, ii, iii, l, j, kink_number, coord, nb_coord
    cdef UINT32_t n_active = n_active_prev
    cdef DOUBLE beta = 0.
    cdef DOUBLE gamma = 1. / INF
    cdef DOUBLE[:] w = np.array(rQ).copy()

    cdef DOUBLE scaling = dual_scaling(z, AfTz, w, n_active,
                                           active_set, pb, g, buff_x)

    # Scale dual vector and compute duality gap
    gap = compute_smoothed_gap(pb, f, g, h, x,
                                   rf, rhx, rQ, prox_y, z, AfTz, w, x, prox_y,
                                   buff_x, buff_y, buff,
                                   &beta, &gamma, compute_z=False,
                                   compute_gamma=False)

    # Screen
    iii = 0
    while iii < n_active:
        ii = active_set[iii]
        nb_coord = pb.blocks[ii+1] - pb.blocks[ii]
        for i in range(nb_coord):
            coord = pb.blocks[ii] + i
            buff_x[i] = AfTz[coord] + w[coord]
        kink_number = 0
        while True:
            polar_support_value = polar_support_kink(g[ii], buff_x,
                                   buff, nb_coord, kink_number)
            if polar_support_value == -1.:
                # code for no more kinks
                break
            else:
                radius = sqrt(2 * gap / max_Lf)
                if (polar_support_value + radius * g_norms_Af[ii][kink_number]) \
                       / (pb.Dg.data[0][ii] * pb.cg[ii]) < 1 \
                       and is_in_domain_polar_support(g[ii], buff_x, nb_coord,
                                radius * norms_Af[ii], kink_number):
                    # set x[ii] = x*[ii] = x_k[ii] and update residuals
                    for i in range(nb_coord):
                        coord = pb.blocks[ii] + i
                        buff[i] = (buff[i] + pb.bg[coord]) / pb.Dg.data[0][ii]
                        dxi = buff[i] - x[coord]
                        if dxi != 0:
                            for l in range(pb.Q.indptr[coord], pb.Q.indptr[coord+1]):
                                j = pb.Q.indices[l]
                                rQ[j] += pb.Q.data[l] * dxi
                            if pb.f_present == True:
                                for l in range(pb.Af.indptr[coord], pb.Af.indptr[coord+1]):
                                    j = pb.Af.indices[l]
                                    rf[j] += pb.Af.data[l] * dxi
                        x[coord] = buff[i]
                        if accelerated == True:
                            dxei = buff[i] - xe[coord]
                            dxci = - xc[coord]
                            if dxei != 0 and dxci != 0:
                                for l in range(pb.Q.indptr[coord], pb.Q.indptr[coord+1]):
                                    j = pb.Q.indices[l]
                                    rQe[j] += pb.Q.data[l] * dxei
                                    rQc[j] += pb.Q.data[l] * dxci
                                if pb.f_present == True:
                                    for l in range(pb.Af.indptr[coord], pb.Af.indptr[coord+1]):
                                        j = pb.Af.indices[l]
                                        rfe[j] += pb.Af.data[l] * dxei
                                        rfc[j] += pb.Af.data[l] * dxci
                            xe[coord] = buff[i]
                            xc[coord] = 0.
                    # remove variable ii from the active set
                    active_set[iii] = active_set[n_active - 1]
                    n_active = n_active - 1
                    iii = iii - 1
                    break  # go to next variable
            kink_number = kink_number + 1
        iii = iii + 1
    
    return n_active


cdef UINT32_t update_focus_set(UINT32_t[:] focus_set, UINT32_t n_active,
                         UINT32_t[:] active_set,
                         atom* g, pb, DOUBLE[:] x,
                         DOUBLE[:] buff_x, DOUBLE[:] buff):
    # We focus on variables that are not at kinks
    cdef UINT32_t j, ii, nb_coord, i, coord
    cdef UINT32_t n_focus = 0
    for j in range(n_active):
        ii = active_set[j]
        nb_coord = pb.blocks[ii+1] - pb.blocks[ii]
        for i in range(nb_coord):
            coord = pb.blocks[ii] + i
            buff_x[i] = pb.Dg.data[0][ii] * x[coord] - pb.bg[coord]
        if g[ii](buff_x, buff, nb_coord, IS_KINK,
                     useless_param, useless_param) == 0:
            focus_set[n_focus] = ii
            n_focus += 1
    return n_focus
