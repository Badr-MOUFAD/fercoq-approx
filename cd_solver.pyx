# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True cd_solver.pyx


from atoms cimport *
# bonus: same imports as in atoms

from libc.stdlib cimport malloc, free

import numpy as np
from scipy import linalg
from scipy import sparse
import warnings
import time
import sys

from helpers cimport compute_primal_value, compute_smoothed_gap
from algorithms cimport one_step_coordinate_descent
from algorithms cimport one_step_accelerated_coordinate_descent
from algorithms cimport RAND_R_MAX
from screening cimport polar_matrix_norm, do_gap_safe_screening, update_focus_set

from algorithms import find_dual_variables_to_update, variable_restart

# The following three functions are copied from Scikit Learn.


class Problem:
      # defines the optimization problem
      # min_x sum_j cf[j] * f_j (Af[j] x - bf[j])
      # 	      	    + cg * g (Dg x - bg) + ch * h (Ah x - bh)
      
      def __init__(self, N, blocks=None, x_init=None, y_init=None,
                         f=None, cf=None, Af=None, bf=None, blocks_f=None,
                         g=None, cg=None, Dg=None, bg=None,
                         h=None, ch=None, Ah=None, bh=None, blocks_h=None,
                         h_takes_infinite_values=None):
            # N is the number of variables
            # blocks codes all blocks. It starts with 0 and terminates with N.
            # The default is N block of size 1.
            #
            # f, g and h are lists of strings that code for the name of a
            # convex function defined in atoms.pyx
            #
            # The rest of the parameters are arrays and matrices
            # We only allow blocks_g to be equal to blocks (for easier implementation)

            self.N = N
            if blocks is None:
                  blocks = np.arange(N+1, dtype=np.uint32)
            self.blocks = np.array(blocks, dtype=np.uint32)
            if x_init is None:
                  self.x_init = np.zeros(N)
                  
            if f is not None:
                  self.f_present = True
                  if cf is None:
                        cf = np.ones(len(f))
                  if len(cf) != len(f):
                        raise Warning("cf should have the same length as f.")
                  if Af is None:
                        raise Warning("Af must be defined if f is.")
                  Af = sparse.csc_matrix(Af)
                  if Af.shape[1] != N:
                        raise Warning("dimensions of Af and x do not match.")
                  if bf is None:
                        bf = np.zeros(Af.shape[0])
                  if len(bf) != Af.shape[0]:
                        raise Warning("dimensions of Af and bf do not match.")
            else:
                self.f_present = False
                f = []
                cf = Af = bf = np.empty(0)
            if blocks_f is None:
                blocks_f = np.arange(len(f)+1, dtype=np.uint32)
            if len(blocks_f) != len(f) + 1 or blocks_f[-1] != Af.shape[0]:
                  raise Warning("blocks_f seems to be ill defined.")
            

            if g is not None:
                  self.g_present = True
                  if len(g) != len(self.blocks) - 1:
                        raise Warning("blocks for g and x should match.")
                  if cg is None:
                        cg = np.ones(len(g))
                  if len(cg) != len(g):
                        raise Warning("cg should have the same length as g.")
                  if Dg is None:
                        Dg = sparse.eye(len(g))
                  if sparse.isspmatrix_dia(Dg) is not True:
                        raise Warning("Dg must be a sparse diagonal matrix.")
                  if Dg.shape[1] != len(g):
                        raise Warning("dimensions of Dg and g do not match.")
                  if bg is None:
                        bg = np.zeros(N)
                  if len(bg) != N:
                        raise Warning("dimensions of bg and x do not match.")
            else:
                self.g_present = False
                g = []
                cg = bg = np.empty(0)
                Dg = 0 * sparse.eye(1)

            if h is not None:
                  self.h_present = True
                  if ch is None:
                        ch = np.ones(len(h))
                  if len(ch) != len(h):
                        raise Warning("ch should have the same length as h.")
                  if Ah is None:
                        raise Warning("Ah must be defined if h is.")
                  Ah = sparse.csc_matrix(Ah)
                  if Ah.shape[1] != N:
                        raise Warning("dimensions of Ah and x do not match.")
                  elif bh is None:
                        bh = np.zeros(Ah.shape[0])
                  if len(bh) != Ah.shape[0]:
                        raise Warning("dimensions of Dh and bh do not match.")
                  if h_takes_infinite_values is None:
                      if (any([h[j] == 'eq_const' for j in range(len(h))]) or
                              any([h[j] == 'box_zero_one' for j in range(len(h))]) or
                              any([h[j] == 'ineq_const' for j in range(len(h))])
                              ):
                            h_takes_infinite_values = True
                      else:
                            h_takes_infinite_values = False

            else:
                self.h_present = False
                h = []
                ch = Ah = bh = np.empty(0)
                h_takes_infinite_values = False
            if blocks_h is None:
                blocks_h = np.arange(len(h)+1, dtype=np.uint32)
            if len(blocks_h) != len(h) + 1 or blocks_h[-1] != Ah.shape[0]:
                    raise Warning("blocks_h seems to be ill defined.")

            self.f = f
            self.cf = np.array(cf, dtype=float)
            self.Af = sparse.csc_matrix(Af)
            self.bf = np.array(bf, dtype=float)
            self.blocks_f = np.array(blocks_f, dtype=np.uint32)
            self.g = g
            self.cg = np.array(cg, dtype=float)
            self.Dg = Dg
            self.bg = np.array(bg, dtype=float)
            self.h = h
            self.ch = np.array(ch, dtype=float)
            self.Ah = sparse.csc_matrix(Ah)
            self.bh = np.array(bh, dtype=float)
            self.blocks_h = np.array(blocks_h, dtype=np.uint32)
            if y_init == None:
                  y_init = np.zeros(self.Ah.shape[0])
            self.y_init = y_init
            self.h_takes_infinite_values = h_takes_infinite_values



def coordinate_descent(pb, int max_iter=1000, max_time=1000.,
                           verbose=0, print_style='classical',
                           min_change_in_x=1e-15, tolerance=0,
                           check_period=10, step_size_factor=1.,
                           sampling='uniform', int accelerated=False,
                           int restart_period=0, callback=None, int per_pass=1,
                           screening=None):
    # pb is a Problem as defined above
    # max_iter: maximum number of passes over the data
    # max_time: in seconds
    # verbose: if positive, time between two prints
    # print_style: 'classical' or 'smoothed_gap'
    # min_change_in_x: stopping criterion
    # tolerance: stopping criterion wrt smoothed gap
    # check_period: period for smoothed gap computation
    # step_size_factor: number to balance primal and dual step sizes
    # sampling: either 'uniform' or 'kink_half'
    # accelerated: if True, the algorithm with momentum is implemented
    # restart_period: initial restart period for accelerated method
    # per_pass: number of times we go through the data before releasing the gil
    # screening: if screening == 'gapsafe': do gap safe screening (Ndiaye et al)
    #
    # For details on the algorithms, see algorithms.pyx
    
    #--------------------- Prepare data ----------------------#
    
    cdef UINT32_t ii, j, jj, k, l, i, coord, lh, jh
    cdef UINT32_t f_iter
    cdef UINT32_t nb_coord

    # Problem pb
    cdef DOUBLE[:] x = pb.x_init.copy()
    cdef DOUBLE[:] y
    cdef DOUBLE[:] y_center

    cdef UINT32_t N = pb.N
    cdef UINT32_t[:] blocks = pb.blocks
    cdef UINT32_t[:] blocks_f = pb.blocks_f
    cdef UINT32_t[:] blocks_h = pb.blocks_h
    cdef UINT32_t n = len(pb.blocks) - 1

    cdef DOUBLE[:] cf = pb.cf
    cdef UINT32_t[:] Af_indptr = np.array(pb.Af.indptr, dtype=np.uint32)
    cdef UINT32_t[:] Af_indices = np.array(pb.Af.indices, dtype=np.uint32)
    cdef DOUBLE[:] Af_data = np.array(pb.Af.data, dtype=float)
    cdef DOUBLE[:] bf = pb.bf
    cdef DOUBLE[:] cg = pb.cg
    cdef DOUBLE[:] Dg_data = np.array(pb.Dg.data[0], dtype=float)
    cdef DOUBLE[:] bg = pb.bg
    cdef DOUBLE[:] ch = pb.ch
    cdef UINT32_t[:] Ah_indptr = np.array(pb.Ah.indptr, dtype=np.uint32)
    cdef UINT32_t[:] Ah_indices = np.array(pb.Ah.indices, dtype=np.uint32)  # I do not know why but the order of the indices is changed here...
    cdef DOUBLE[:] Ah_data = np.array(pb.Ah.data, dtype=float)  # Fortunately, it seems that the same thing happens here.
    cdef UINT32_t[:] Ah_nnz_perrow = np.array((pb.Ah!=0).sum(axis=1), dtype=np.uint32).ravel()
    
    cdef DOUBLE[:] bh = pb.bh

    cdef int f_present = pb.f_present
    cdef atom* f
    if f_present is True:
        f = <atom*>malloc(len(pb.f)*sizeof(atom))
        for j in range(len(pb.f)):
            if sys.version_info[0] > 2 and isinstance(pb.f[j], bytes) == True:
                pb.f[j] = pb.f[j].encode()
            f[j] = string_to_func(<bytes>pb.f[j])
    else:
        f = <atom*>malloc(0)  # just to remove uninitialized warning

    cdef int g_present = pb.g_present
    cdef atom* g
    if g_present is True:
        g = <atom*>malloc(len(pb.g)*sizeof(atom))
        for ii in range(len(pb.g)):
            if sys.version_info[0] > 2 and isinstance(pb.g[ii], bytes) == True:
                pb.g[ii] = pb.g[ii].encode()
            g[ii] = string_to_func(<bytes>pb.g[ii])
    else:
        g = <atom*>malloc(0)  # just to remove uninitialized warning

    cdef int h_present = pb.h_present
    cdef atom* h
    if h_present is True:
        h = <atom*>malloc(len(pb.h)*sizeof(atom))
        for jh in range(len(pb.h)):
            if sys.version_info[0] > 2 and isinstance(pb.h[jh], bytes) == True:
                pb.h[jh] = pb.h[jh].encode()
            h[jh] = string_to_func(<bytes>pb.h[jh])
    else:
        h = <atom*>malloc(0)  # just to remove uninitialized warning
    cdef int h_takes_infinite_values = pb.h_takes_infinite_values

    # We have two kind of dual vectors so the user may use any of them to initialize
    if accelerated == False:
        if pb.y_init.shape[0] == pb.Ah.nnz or h_present is False:
            y = pb.y_init.copy()
        else:
            y = np.zeros(pb.Ah.nnz, dtype=float)
            for i in range(N):
                for lh in range(Ah_indptr[i], Ah_indptr[i+1]):
                    y[lh] = pb.y_init[Ah_indices[lh]]
    else:
        if pb.y_init.shape[0] == pb.Ah.shape[0] or h_present is False:
            y_center = pb.y_init.copy()
        else:
            y_center = np.zeros(pb.Ah.shape[0], dtype=float)
            for i in range(N):
                for lh in range(Ah_indptr[i], Ah_indptr[i+1]):
                    # we take only the last copy of pb.y_init[lh]
                    y_center[Ah_indices[lh]] = pb.y_init[lh]

    cdef UINT32_t[:] inv_blocks_f = np.zeros(pb.Af.shape[0], dtype=np.uint32)

    if f_present is True:
        for j in range(len(pb.f)):
            for i in range(blocks_f[j+1] - blocks_f[j]):
                inv_blocks_f[blocks_f[j]+i] = j
                
    cdef UINT32_t[:] Ah_col_indices = np.empty(Ah_indices.shape[0], dtype=np.uint32)
    if h_present is True:
        for i in range(N):
            for lh in range(Ah_indptr[i], Ah_indptr[i+1]):
                Ah_col_indices[lh] = i

    cdef UINT32_t[:] inv_blocks_h = np.zeros(pb.Ah.shape[0], dtype=np.uint32)
    cdef UINT32_t[:,:] dual_vars_to_update
    if h_present is True:
        # As h is not required to be separable, we need some preprocessing
        # to detect what dual variables need to be processed
        for jh in range(len(pb.h)):
            for i in range(blocks_h[jh+1] - blocks_h[jh]):
                inv_blocks_h[blocks_h[jh]+i] = jh
        dual_vars_to_update_ = find_dual_variables_to_update(n, blocks, blocks_h,
                                                Ah_indptr, Ah_indices, inv_blocks_h)
        dual_vars_to_update = np.empty((n,1+max([len(dual_vars_to_update_[ii])
                                                        for ii in range(n)])),
                                            dtype=np.uint32)
        for ii in range(n):
            dual_vars_to_update[ii][0] = len(dual_vars_to_update_[ii])
            for i in range(len(dual_vars_to_update_[ii])):
                dual_vars_to_update[ii][i+1] = dual_vars_to_update_[ii][i]
    else:
        dual_vars_to_update = np.empty((0,0), dtype=np.uint32)

    # Definition of residuals
    cdef DOUBLE[:] rf
    if f_present is True:
        rf = pb.Af * x - pb.bf
    else:
        rf = np.empty(0)
    cdef DOUBLE[:] rhx
    if h_present is True:
        rhx = pb.Ah * x - pb.bh
    else:
        rhx = np.empty(0)
    cdef DOUBLE[:] rhy = np.zeros(pb.Ah.shape[1])
    cdef DOUBLE[:] Sy = np.zeros(pb.Ah.shape[0])
    if h_present is True and accelerated is False:
        # Sy is the mean of the duplicates of y
        for i in range(N):
            for l in range(Ah_indptr[i], Ah_indptr[i+1]):
                Sy[Ah_indices[l]] += y[l] / Ah_nnz_perrow[Ah_indices[l]]
                rhy[i] += Ah_data[l] * y[l]

    # Arrays for accelerated version
    cdef DOUBLE[:] xe
    cdef DOUBLE[:] xc
    cdef DOUBLE[:] rfe
    cdef DOUBLE[:] rfc
    cdef DOUBLE[:] rhxe
    cdef DOUBLE[:] rhxc
    if accelerated == True:
        xe = x.copy()
        xc = np.zeros(x.shape[0])
        rfe = np.array(rf).copy()
        rfc = np.zeros(rf.shape[0])
        rhxe = np.array(rhx).copy()
        rhxc = np.zeros(rhx.shape[0])
    else:
        xe = np.empty(0)
        xc = np.empty(0)
        rfe = np.empty(0)
        rfc = np.empty(0)
        rhxe = np.empty(0)
        rhxc = np.empty(0)

    cdef DOUBLE theta0 = 1. / n
    cdef DOUBLE theta = theta0
    cdef DOUBLE c_theta = 1.
    cdef DOUBLE beta0 = 1e-30
    cdef DOUBLE beta
    restart_history = []
    next_period = restart_period

    cdef UINT32_t rand_r_state_seed = np.random.randint(RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    # buffers and auxiliary variables
    max_nb_coord = <int> np.max(np.diff(pb.blocks))
    max_nb_coord_h = <int> np.max(np.hstack((np.zeros(1), np.diff(pb.blocks_h))))
    max_nb_coord_f = <int> np.max(np.hstack((np.zeros(1), np.diff(pb.blocks_f))))
    cdef DOUBLE[:] grad = np.zeros(max_nb_coord)
    cdef DOUBLE[:] x_ii = np.zeros(max_nb_coord)
    cdef DOUBLE[:] prox_y = np.zeros(pb.Ah.shape[0])
    cdef DOUBLE[:] rhy_ii = np.zeros(max_nb_coord)
    cdef DOUBLE[:] buff_x = np.zeros(max_nb_coord)
    cdef DOUBLE[:] buff_y = np.zeros(max_nb_coord_h)
    cdef DOUBLE[:] buff = np.zeros(max([max_nb_coord, max_nb_coord_h, max_nb_coord_f]))
    cdef DOUBLE[:] xc_ii
    cdef DOUBLE[:] xe_ii
    if accelerated == True:
        xc_ii = np.zeros(max_nb_coord)
        xe_ii = np.zeros(max_nb_coord)

    # Compute Lipschitz constants
    cdef DOUBLE[:] tmp_Lf = np.zeros(len(pb.f))
    cdef DOUBLE max_Lf = 0.
    for j in range(len(pb.f)):
        tmp_Lf[j] = cf[j] * f[j](buff_x, buff, blocks_f[j+1]-blocks_f[j],
                         LIPSCHITZ, useless_param, useless_param)
        max_Lf = fmax(max_Lf, tmp_Lf[j])
    cdef DOUBLE[:] Lf = 1e-30 * np.ones(n)
    if f_present is True:
        for ii in range(n):
            # if block size is > 1, we use the inequality frobenius norm > 2-norm
            #   (not optimal)
            for i in range(blocks[ii+1] - blocks[ii]):
                coord = blocks[ii] + i
                for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                    jj = Af_indices[l]
                    j = inv_blocks_f[jj]
                    Lf[ii] += Af_data[l]**2 * tmp_Lf[j]
    del tmp_Lf
    cdef DOUBLE[:] primal_step_size = 1. / np.array(Lf)
    cdef DOUBLE[:] dual_step_size = np.zeros(pb.Ah.shape[0])
    cdef DOUBLE[:] norm2_columns_Ah = np.zeros(n)
    if h_present is True:
        if accelerated is True:
            for i in range(n):
                norm2_columns_Ah[i] = (pb.Ah[:,blocks[i]:blocks[i+1]]\
                                .multiply(pb.Ah[:,blocks[i]:blocks[i+1]])).sum()
        else:
            for i in range(n):
                norm2_columns_Ah[i] = ((pb.Ah[:,blocks[i]:blocks[i+1]]\
                                .multiply(pb.Ah[:,blocks[i]:blocks[i+1]])).T\
                                .multiply(np.array(Ah_nnz_perrow))).sum()
        dual_step_size = np.maximum(
            1. / np.sqrt(np.array(norm2_columns_Ah) + 1e-30),
            np.array(Lf) / (np.array(norm2_columns_Ah) + 1e-30)) \
                            * step_size_factor
        if accelerated == True:
            beta0 = 1. / np.maximum(1e-30, np.max(np.array(dual_step_size)))
        else:
            primal_step_size = 0.9 / (Lf + np.array(dual_step_size)
                                          * np.array(norm2_columns_Ah))
    beta = beta0

    cdef int sampling_law = 0  # default, uniform coordinate sampling probability
    if sampling == 'kink_half':
        sampling_law = 1
    cdef UINT32_t n_active = n
    cdef UINT32_t n_focus = n
    cdef UINT32_t[:] active_set = np.arange(n, dtype=np.uint32)
    cdef DOUBLE[:] z = np.zeros(pb.Af.shape[0])  # useful for screening
    cdef DOUBLE[:] AfTz = np.zeros(N)  # useful for screening
    if screening == 'gapsafe' and h_present is True:
        print('Gap safe screening not analyzed when h is present,'
                  'we are deactivating it.')
        screening = None
    if screening == 'gapsafe':
        g_norms_Af = [[] for ii in range(n)]
        norms_Af = np.zeros(n)
        for ii in range(n):
            nb_coord = blocks[ii+1] - blocks[ii]
            k = 0
            l = Af_indptr[blocks[ii]]
            norms_Af[ii] = polar_matrix_norm(abs,
                            &Af_indptr[blocks[ii]], nb_coord,
                            Af_indices, Af_data, 0)
            while True:
                polar_support_value = polar_matrix_norm(g[ii],
                            &Af_indptr[blocks[ii]], nb_coord,
                            Af_indices, Af_data, k)
                if polar_support_value == -1.:
                    # code for no more kinks
                    break
                else:
                    g_norms_Af[ii].append(polar_support_value)
                k = k + 1

    cdef UINT32_t[:] focus_set
    if sampling_law == 1:
        focus_set = np.arange(n, dtype=np.uint32)
        n_focus = n
        if accelerated == True:
            theta0 = 0.5 / n
            theta = theta0
    else:
        focus_set = np.empty(0, dtype=np.uint32)

    cdef UINT32_t iter_last_check = -10
    cdef UINT32_t offset = 0
    cdef DOUBLE primal_val = 0.
    cdef DOUBLE infeas = 0.
    cdef DOUBLE dual_val = 0.
    cdef DOUBLE beta_print = 0.
    cdef DOUBLE gamma_print = 0.
        
    cdef DOUBLE change_in_x
    cdef DOUBLE change_in_y
    smoothed_gap, beta_print, gamma_print = 0., 0., 0.

    #----------------------- Main loop ----------------------------#
    init_time = time.time()
    if verbose > 0:
        if print_style == 'classical':
            if h_present is True and h_takes_infinite_values is False:
                print("elapsed time \t iter \t function value  "
                          "change in x \t change in y")
            elif h_present is True and h_takes_infinite_values is True:
                print("elapsed time \t iter \t function value  "
                          "infeasibility \t change in x \t change in y")
            else:
                print("elapsed time \t iter \t function value  change in x")
        elif print_style == 'smoothed_gap':
                print("elapsed time\titer\tfunction value infeasibility"
                              "\tsmoothed gap \tbeta     gamma  "
                              "\tchange in x\tchange in y")

    nb_prints = 0

    # code in the case bloks_g = blocks only for the moment
    for iter in range(0, max_iter, per_pass):
        if callback is not None:
            if callback(x, Sy, rf, rhx): break

        change_in_x = 0.
        change_in_y = 0.

        if accelerated == False:
            one_step_coordinate_descent(x,
                    y, Sy, prox_y, rhx, rf, rhy, rhy_ii,
                    buff_x, buff_y, buff, x_ii, grad,
                    blocks, blocks_f, blocks_h,
                    Af_indptr, Af_indices, Af_data, cf, bf,
                    Dg_data, cg, bg,
                    Ah_indptr, Ah_indices, Ah_data,
                    inv_blocks_f,
                    inv_blocks_h, Ah_nnz_perrow, Ah_col_indices,
                    dual_vars_to_update,
                    ch, bh,
                    f, g, h, f_present, g_present, h_present,
                    primal_step_size, dual_step_size,
                    sampling_law, rand_r_state, active_set, n_active,
                    focus_set, n_focus, n,
                    per_pass, &change_in_x, &change_in_y)
        else:
            one_step_accelerated_coordinate_descent(x,
                    xe, xc, y_center, prox_y, rhxe, rhxc, rfe, rfc,
                    rhy, &theta, theta0, &c_theta, &beta,
                    buff_x, buff_y, buff, xe_ii, xc_ii, grad,
                    blocks, blocks_f, blocks_h,
                    Af_indptr, Af_indices, Af_data, cf, bf,
                    Dg_data, cg, bg, Ah_indptr, Ah_indices, Ah_data,
                    inv_blocks_f, inv_blocks_h, Ah_nnz_perrow,
                    Ah_col_indices, dual_vars_to_update, ch, bh,
                    f, g, h, f_present, g_present, h_present,
                    Lf, norm2_columns_Ah, 
                    sampling_law, rand_r_state, active_set, n_active,
                    focus_set, n_focus, n,
                    per_pass, &change_in_x)

        elapsed_time = time.time() - init_time
        if verbose > 0 or tolerance > 0:
            if ((verbose > 0 and elapsed_time > nb_prints * verbose)
                    or change_in_x + change_in_y < min_change_in_x
                    or elapsed_time > max_time
                    or iter >= max_iter-1):
                print_time = True
                nb_prints += 1
            else:
                print_time = False
            if tolerance > 0 and iter - iter_last_check >= check_period:
                check_time = True
            else:
                check_time = False
            if print_time == True or check_time == True:
                # Compute value
                if accelerated is True:
                    for i in range(N):
                        x[i] = xe[i] + c_theta * xc[i]
                    if f_present is True:
                        for j in range(pb.Af.shape[0]):
                            rf[j] = rfe[j] + c_theta * rfc[j]
                    if h_present is True:
                        for l in range(pb.Ah.shape[0]):
                            rhx[l] = rhxe[l] + c_theta * rhxc[l]
                            
                compute_primal_value(pb, f, g, h, x, rf, rhx,
                                         buff_x, buff_y, buff,
                                         &primal_val, &infeas)
                if print_style == 'classical' and print_time == True:
                    if h_present is True and h_takes_infinite_values is False:
                        print("%.5f \t %d \t %+.5e \t %.5e \t %.5e"
                                  %(elapsed_time, iter, primal_val,
                                        change_in_x, change_in_y))
                    elif h_present is True and h_takes_infinite_values is True:
                        print("%.5f \t %d \t %+.5e \t %.5e \t %.5e \t %.5e"
                                  %(elapsed_time, iter, primal_val, infeas,
                                        change_in_x, change_in_y))
                    else:  # h_present is False
                        print("%.5f \t %d \t %+.5e \t %.5e"
                                  %(elapsed_time, iter, primal_val, change_in_x))
                elif print_style == 'smoothed_gap' or tolerance > 0:
                    # When we print, we check
                    if h_present is True:
                        beta_print = max(infeas, 1e-20)
                        for j in range(len(pb.h)):
                            if accelerated == False:
                                # Compute one more prox_h* in case Sy is not feasible
                                for l in range(blocks_h[j+1]-blocks_h[j]):
                                    buff_y[l] = Sy[blocks_h[j]+l] \
                                      + rhx[blocks_h[j]+l] * dual_step_size[j]
                            else:  # accelerated == True:
                                # Compute dual vector
                                for l in range(blocks_h[j+1]-blocks_h[j]):
                                    buff_y[l] = y_center[blocks_h[j]+l] \
                                      + rhx[blocks_h[j]+l] / beta

                            h[j](buff_y, buff,
                                 blocks_h[j+1]-blocks_h[j],
                                 PROX_CONJ,
                                 dual_step_size[j],
                                 ch[j])
                            for l in range(blocks_h[j+1]-blocks_h[j]):
                                prox_y[blocks_h[j]+l] = buff[l]
                    else:
                        beta_print = 0

                    smoothed_gap = compute_smoothed_gap(pb, f, g, h, x,
                                        rf, rhx, prox_y, z, AfTz,
                                        buff_x, buff_y, buff,
                                        &beta_print, &gamma_print)

                    if print_style == 'smoothed_gap' and print_time == True:
                        print("%.5f \t %d\t%+.5e\t%.5e\t%.5e\t%.2e %.1e\t%.5e\t%.5e"
                              %(elapsed_time, iter, primal_val, infeas,
                                smoothed_gap, beta_print, gamma_print,
                                change_in_x, change_in_y))
                    if smoothed_gap < tolerance and beta_print < tolerance \
                           and gamma_print < tolerance:
                        print("Target tolerance reached: stopping "
                                  "the algorithm at smoothed_gap=%.5e, "
                                  "beta=%.5e, gamma=%.5e"
                                  %(smoothed_gap, beta_print, gamma_print))
                        break
                iter_last_check = iter

        if screening == 'gapsafe' and iter == iter_last_check:
            # AfTz was computed just before when checking tolerance or printing
            n_active = do_gap_safe_screening(active_set, n_active,
                              pb, f, g, h, Lf,
                              x, rf, rhx, prox_y, z, AfTz,
                              xe, xc, rfe, rfc, buff_x, buff_y, buff,
                              g_norms_Af, norms_Af, max_Lf, accelerated)
        if sampling_law == 1 and g_present is True:
            n_focus = update_focus_set(focus_set, n_active, active_set,
                                           g, pb, x, buff_x, buff)

        if accelerated is True and restart_period > 0:
            do_restart, next_period = variable_restart(restart_history,
                                        iter - offset, restart_period, next_period)
            if do_restart is True:
                xe = np.array(xe) + c_theta * np.array(xc)
                rfe = np.array(rfe) + c_theta * np.array(rfc)
                xc = np.zeros(x.shape[0])
                rfc = np.zeros(rf.shape[0])
                if h_present is True:
                    rhxe = np.array(rhxe) + c_theta * np.array(rhxc)
                    rhxc = np.zeros(rhx.shape[0])
                    y_center = np.array(prox_y).copy()  # heuristic
                if sampling_law == 1:
                    theta0 = 0.5 / n_active
                else:
                    theta0 = 1. / n_active
                theta = theta0
                beta = beta0
                c_theta = 1.
                offset = iter

                
        if elapsed_time > max_time:
            print("Time limit reached: stopping the algorithm after %f s"
                      %elapsed_time)
            break
        if change_in_x + change_in_y < min_change_in_x:
            print("Not enough change in iterates (||x(t+1) - x(t)|| = %.5e): "
                      "stopping the algorithm" %change_in_x)
            break

    pb.performance_stats = {"Time (s)": elapsed_time, "Iterations": iter,
                                "Smoothed Gap": [smoothed_gap, beta_print, gamma_print],
                                "Change in x": change_in_x, "Change_in_y": change_in_x,
                                "Primal value": primal_val, "Infeasibility": infeas}
    pb.sol = np.array(x).copy()
    if accelerated is False:
        pb.dual_sol = np.array(Sy).copy()
        pb.dual_sol_duplicated = np.array(y).copy()
    else:
        pb.dual_sol = np.array(prox_y).copy()

    free(f)
    free(g)
    free(h)
