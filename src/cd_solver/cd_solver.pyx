# cython --cplus -X boundscheck=False -X cdivision=True cd_solver.pyx


from libc.math cimport fabs, sqrt, log2
cimport numpy as np
import numpy as np
from scipy import linalg
from scipy import sparse

cimport cython
import warnings
from libc.stdlib cimport malloc, free

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t

import time

from atoms cimport *

# The following three functions are copied from Scikit Learn.

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end



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
            # convex functions of the type
            # cdef double function(double x, bool val=True,
            #           bool grad=False, bool Lipschitz=False,
            #           bool prox=False, double prox_param=1.,
            #           double* buffer)
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
                        raise Warning("cf should have the same length as f")
                  if Af is None:
                        raise Warning("Af must be defined if f is")
                  Af = sparse.csc_matrix(Af)
                  if Af.shape[1] != N:
                        raise Warning("dimensions of Af and x do not match")
                  if bf is None:
                        bf = np.zeros(Af.shape[0])
                  if len(bf) != Af.shape[0]:
                        raise Warning("dimensions of Af and bf do not match")
            else:
                self.f_present = False
                f = []
                cf = Af = bf = np.empty(0)
            if blocks_f is None:
                blocks_f = np.arange(len(f)+1, dtype=np.uint32)
            

            if g is not None:
                  self.g_present = True
                  if len(g) != len(self.blocks) - 1:
                        raise Warning("blocks for g and x should match")
                  if cg is None:
                        cg = np.ones(len(g))
                  if len(cg) != len(g):
                        raise Warning("cg should have the same length as g")
                  if Dg is None:
                        Dg = sparse.eye(len(g))
                  if sparse.isspmatrix_dia(Dg) is not True:
                        raise Warning("Dg must be a sparse diagonal matrix")
                  if Dg.shape[1] != len(g):
                        raise Warning("dimensions of Dg and g do not match")
                  if bg is None:
                        bg = np.zeros(N)
                  if len(bg) != N:
                        raise Warning("dimensions of bg and x do not match")
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
                        raise Warning("ch should have the same length as h")
                  if Ah is None:
                        raise Warning("Ah must be defined if h is")
                  Ah = sparse.csc_matrix(Ah)
                  if Ah.shape[1] != N:
                        raise Warning("dimensions of Ah and x do not match")
                  elif bh is None:
                        bh = np.zeros(Ah.shape[0])
                  if len(bh) != Ah.shape[0]:
                        raise Warning("dimensions of Dh and bh do not match")
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
            else:
                if len(blocks_h) != len(h) + 1:
                    raise Warning("dimensions of h and blocks_h do not match")

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
                  y_init = np.zeros(self.Ah.nnz)
            self.y_init = y_init
            self.h_takes_infinite_values = h_takes_infinite_values


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


cdef void compute_primal_value(pb, unsigned char** f, unsigned char** g, unsigned char** h,
                             DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx,
                             DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                             DOUBLE* val, DOUBLE* infeas):
    cdef UINT32_t ii, i, j, jh, l, coord, nbcoord
    val[0] = 0.
    infeas[0] = 0.
    if pb.f_present is True:
        for j in range(len(pb.f)):
            val[0] += pb.cf[j] * my_eval(f[j],
                                      rf[pb.blocks_f[j]:pb.blocks_f[j+1]],
                                      buff,
                                      nb_coord=pb.blocks_f[j+1]-pb.blocks_f[j])
    if pb.g_present is True:
        for ii in range(len(pb.g)):
            nb_coord = pb.blocks[ii+1] - pb.blocks[ii]
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                buff_x[i] = pb.Dg.data[0][ii] * x[coord] - pb.bg[coord]
            val[0] += pb.cg[ii] * my_eval(g[ii], buff_x, buff,
                                           nb_coord=nb_coord)
    if pb.h_present is True:
        if pb.h_takes_infinite_values == False:
            for jh in range(len(pb.h)):
                val[0] += pb.ch[jh] * my_eval(h[jh],
                                        rhx[pb.blocks_h[jh]:pb.blocks_h[jh+1]],
                                        buff,
                                        nb_coord=pb.blocks_h[jh+1] - pb.blocks_h[jh])
        if pb.h_takes_infinite_values == True:
            for jh in range(len(pb.h)):
                for l in range(pb.blocks_h[jh+1]-pb.blocks_h[jh]):
                    coord = pb.blocks_h[jh]+l
                    buff_y[l] = rhx[coord]
                # project rhx onto the domain of h
                my_eval(h[jh], buff_y, buff,
                            nb_coord=pb.blocks_h[jh+1]-pb.blocks_h[jh],
                            mode=PROX, prox_param=1e-20)
                for l in range(pb.blocks_h[jh+1]-pb.blocks_h[jh]):
                    coord = pb.blocks_h[jh]+l
                    infeas[0] += fabs(buff[l] - rhx[coord])


cdef DOUBLE compute_smoothed_gap(pb, unsigned char** f, unsigned char** g, unsigned char** h,
                             DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx, DOUBLE[:] Sy,
                             DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                             DOUBLE* beta, DOUBLE* gamma):
    cdef UINT32_t ii, i, j, jh, l, coord, nbcoord
    cdef DOUBLE val = 0.
    z = np.zeros(pb.Af.shape[0])  # dual variable associated to f(Af x - bf)
    if pb.f_present is True:
        for j in range(len(pb.f)):
            my_eval(f[j], rf[pb.blocks_f[j]:pb.blocks_f[j+1]], buff,
                    nb_coord=pb.blocks_f[j+1]-pb.blocks_f[j], mode=GRAD)
            for l in range(pb.blocks_f[j+1]-pb.blocks_f[j]):
                coord = pb.blocks_f[j] + l
                z[coord] = pb.cf[j] * buff[l]
        val += z.dot(np.array(rf)) + pb.bf.dot(np.array(rf))   # = f(Af x - bf) + f*(z)
        print('contrib f:', val)
        AfTz = pb.Af.T.dot(z)
    else:
        AfTz = np.zeros(pb.N)
    if pb.h_present is True:
        # Theory would require us to do one more prox_h* but we do not do it
        AhTSy = pb.Ah.T.dot(np.array(Sy))
    else:
        AhTSy = np.zeros(pb.N)

    cdef DOUBLE INF = 1e20
    cdef DOUBLE[:] xbar = np.zeros(pb.N, dtype=float)
    cdef DOUBLE[:] ybar = np.zeros(pb.Ah.shape[0], dtype=float)
    if pb.g_present is True:
        val_g = 0.
        # compute g(x)
        for ii in range(len(pb.g)):
            nb_coord = pb.blocks[ii+1] - pb.blocks[ii]
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                buff_x[i] = pb.Dg.data[0][ii] * x[coord] - pb.bg[coord]
            val_g += pb.cg[ii] * my_eval(g[ii], buff_x, buff,
                                           nb_coord=nb_coord)
        print('g(x) = ', val_g)

        # estimate dual infeasibility
        dual_infeas = 0.
        for ii in range(len(pb.g)):
            nb_coord = pb.blocks[ii+1] - pb.blocks[ii]
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                buff_x[i] = 1. / pb.Dg.data[0][ii] * (
                    - (AfTz[coord] + AhTSy[coord])
                    + pb.bg[coord])
                # project -AfTz - AhTSy onto the domain of g*
            my_eval(g[ii], buff_x, buff,
                            nb_coord=nb_coord,
                            mode=PROX_CONJ, prox_param=1./INF, prox_param2=pb.cg[ii])
            for i in range(nb_coord):
                dual_infeas += fabs(buff[i] - buff_x[i])

        #print(dual_infeas)
        gamma[0] = max(1./INF, dual_infeas)
        # compute g*_gamma(-AfTz - AhTSy;x) = -(AfTz + AhTSy)(xbar) - g(xbar) - gamma/2 ||x-xbar||**2
        val_g1 = 0.
        val_g2 = 0.
        val_g3 = 0.
        # note that g deals with bg directly in the prox
        for ii in range(len(pb.g)):
            # compute xbar
            nb_coord = pb.blocks[ii+1] - pb.blocks[ii]
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                buff_x[i] = pb.Dg.data[0][ii] * (
                    x[coord] - 1. / gamma[0] * (AfTz[coord] + AhTSy[coord])) - pb.bg[coord]
            my_eval(g[ii], buff_x, buff,
                        nb_coord=pb.blocks[ii+1]-pb.blocks[ii],
                        mode=PROX, prox_param=pb.cg[ii]*pb.Dg.data[0][ii]**2/gamma[0])
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                xbar[coord] = 1. / pb.Dg.data[0][ii] * (buff[i] + pb.bg[coord])

            # compute g*_gamma(-AfTz - AhTSy;x)
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                buff_x[i] = pb.Dg.data[0][ii] * xbar[coord] - pb.bg[coord]
            val_g1 -= pb.cg[ii] * my_eval(g[ii], buff_x, buff, nb_coord=nb_coord)
            for i in range(nb_coord):
                val_g2 -= (AfTz[coord] + AhTSy[coord]) * xbar[coord]
                val_g3 -= gamma[0] / 2. * (xbar[coord] - x[coord])**2
        # print('contrib g:', val_g, val_g1, val_g2, val_g3, np.array(xbar), np.array(x), np.array(AfTz))
        val += val_g + val_g1 + val_g2 + val_g3

    if pb.h_present is True:
        val_h = 0.
        # compute h*(Sy) + bh.Sy = Sybar.Sy - h(Sybar) + bh.Sy
        for jh in range(len(pb.h)):
            nb_coord = pb.blocks_h[jh+1] - pb.blocks_h[jh]
            for l in range(nb_coord):
                coord = pb.blocks_h[jh] + l
                buff_y[l] = Sy[coord] + INF * Sy[coord]
            my_eval(h[jh], buff_y, buff, nb_coord=nb_coord,
                            mode=PROX, prox_param=pb.ch[jh]*INF)
            for l in range(nb_coord):
                buff_y[l] = buff[l]
            h_Sybar = my_eval(h[jh], buff_y, buff, nb_coord=nb_coord,
                            mode=VAL)
            val_h -= h_Sybar
            for l in range(nb_coord):
                val_h += buff_y[l] * Sy[coord]
                val_h += pb.bh[coord] * Sy[coord]

        if pb.h_takes_infinite_values == False:
            # compute h(Ah x - bh)
            for jh in range(len(pb.h)):
                val_h += pb.ch[jh] * my_eval(h[jh],
                                        rhx[pb.blocks_h[jh]:pb.blocks_h[jh+1]],
                                        buff,
                                        nb_coord=pb.blocks_h[jh+1] - pb.blocks_h[jh])
        if pb.h_takes_infinite_values == True:
            # compute h_beta(Ah x - bh; Sy) = (Ah x - bh) ybar - h*(ybar) - beta/2 ||Sy - ybar||**2
            beta[0] = max(1./INF, beta[0])
            for jh in range(len(pb.h)):
                nb_coord = pb.blocks_h[jh+1] - pb.blocks_h[jh]
                # compute ybar
                for l in range(nb_coord):
                    coord = pb.blocks_h[jh] + l
                    buff_y[l] = Sy[coord] + 1. / beta[0] * rhx[coord]
                my_eval(h[jh], buff_y, buff,
                            nb_coord=pb.blocks_h[jh+1]-pb.blocks[jh],
                            mode=PROX_CONJ, prox_param=1./beta[0], prox_param2=pb.ch[jh])

                # compute -h*(ybar) = -ybar.ybarbar + h(ybarbar)
                for l in range(nb_coord):
                    coord = pb.blocks_h[jh] + l
                    ybar[coord] = buff[l]
                    buff_y[l] = Sy[coord] + INF * ybar[coord]
                my_eval(h[jh], buff_y, buff, nb_coord=nb_coord,
                            mode=PROX, prox_param=pb.ch[jh]*INF)
                for l in range(nb_coord):
                    buff_y[l] = buff[l]
                h_ybarbar = my_eval(h[jh], buff_y, buff, nb_coord=nb_coord,
                            mode=VAL)
                val_h += h_ybarbar
                for l in range(nb_coord):
                    coord = pb.blocks_h[jh] + l
                    val_h -= buff_y[l] * ybar[coord]
                    val_h += rhx[coord] * ybar[coord] - beta[0] / 2. * (Sy[coord] - ybar[coord])**2
        # print('contrib h:', val_h, h_ybarbar, np.array(rhx), np.array(ybar), np.array(Sy))
        val += val_h
    return val


def coordinate_descent(pb, max_iter=1000, max_time=1000., verbose=0, print_style='classical',
                           min_change_in_x=1e-15, step_size_factor=1., callback=None):

    #--------------------- Prepare data ----------------------#
    
    cdef UINT32_t ii, j, k, l, i, coord, lh, jh
    cdef UINT32_t f_iter
    cdef UINT32_t nb_coord

    # Problem pb
    cdef DOUBLE[:] x = pb.x_init.copy()
    cdef DOUBLE[:] y = pb.y_init.copy()
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
    cdef DOUBLE[:] Ah_data = np.array(pb.Ah.data, dtype=float)  # Fortunately, it seems that the same thin happens here.
    cdef UINT32_t[:] Ah_nnz_perrow = np.array((pb.Ah!=0).sum(axis=1), dtype=np.uint32).ravel()
    cdef DOUBLE[:] bh = pb.bh

    cdef int f_present = pb.f_present
    cdef unsigned char** f
    if f_present is True:
        f = <unsigned char**>malloc(len(pb.f)*sizeof(char*))
        for j in range(len(pb.f)):
            f[j] = pb.f[j]

    cdef int g_present = pb.g_present
    cdef unsigned char** g
    if g_present is True:
        g = <unsigned char**>malloc(len(pb.g)*sizeof(char*))
        for ii in range(len(pb.g)):
            g[ii] = pb.g[ii]

    cdef int h_present = pb.h_present
    cdef unsigned char** h
    if h_present is True:
        h = <unsigned char**>malloc(len(pb.h)*sizeof(char*))
        for jh in range(len(pb.h)):
            h[jh] = pb.h[jh]
    cdef int h_takes_infinite_values = pb.h_takes_infinite_values

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
    cdef DOUBLE[:] Sy = np.zeros(pb.Ah.shape[0])  # Sy is the mean of the duplicates of y
    if h_present is True:
        for i in range(N):
            for l in range(Ah_indptr[i], Ah_indptr[i+1]):
                Sy[Ah_indices[l]] += y[l] / Ah_nnz_perrow[Ah_indices[l]]

    cdef UINT32_t rand_r_state_seed = np.random.randint(RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    # buffers and auxiliary variables
    max_nb_coord = <int> np.max(np.diff(pb.blocks))
    max_nb_coord_h = <int> np.max(np.hstack((np.zeros(1), np.diff(pb.blocks_h))))
    cdef DOUBLE[:] grad = np.zeros(max_nb_coord)
    cdef DOUBLE[:] x_ii = np.zeros(max_nb_coord)
    cdef DOUBLE[:] prox_y = np.zeros(pb.Ah.shape[0])
    cdef DOUBLE[:] rhy = np.zeros(max_nb_coord)
    cdef DOUBLE[:] rhy_new = np.zeros(max_nb_coord)
    cdef DOUBLE[:] buff_x = np.zeros(max_nb_coord)
    cdef DOUBLE[:] buff_y = np.zeros(max_nb_coord_h)
    cdef DOUBLE[:] buff = np.zeros(max(max_nb_coord, max_nb_coord_h))

    # Compute Lipschitz constants
    cdef DOUBLE[:] tmp_Lf = np.zeros(len(pb.f))
    for j in range(len(pb.f)):
        tmp_Lf[j] = my_eval(f[j], buff_x, buff, nb_coord=blocks_f[j+1]-blocks_f[j],
                                mode=LIPSCHITZ)
    cdef DOUBLE[:] Lf = 1e-30 * np.ones(n)
    if f_present is True:
        for ii in range(n):
            # if block size is > 1, we use the inequality frobenius norm > 2-norm (not optimal)
            for i in range(blocks[ii+1] - blocks[ii]):
                coord = blocks[ii] + i
                for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                    j = Af_indices[l]
                    Lf[ii] += cf[j] * Af_data[l]**2 * tmp_Lf[j]
    del tmp_Lf
    cdef DOUBLE[:] primal_step_size = 1. / np.array(Lf)
    cdef DOUBLE[:] dual_step_size = np.zeros(pb.Ah.shape[0])
    if h_present is True:
        norm2_columns_Ah = np.array((pb.Ah.multiply(pb.Ah)).sum(axis=0)).ravel()
        dual_step_size = np.max(np.maximum(1. / np.sqrt(norm2_columns_Ah + 1e-30),
                                        np.array(Lf) / (norm2_columns_Ah + 1e-30))) \
                            * step_size_factor * np.ones(pb.Ah.shape[0])
        primal_step_size = 0.9 / (Lf + dual_step_size[0] * norm2_columns_Ah * np.max(np.array(Ah_nnz_perrow)))

    cdef DOUBLE primal_val = 0.
    cdef DOUBLE infeas = 0.
    cdef DOUBLE dual_val = 0.
    cdef DOUBLE beta = 0.
    cdef DOUBLE gamma = 0.
        
    cdef DOUBLE change_in_x
    cdef DOUBLE change_in_y

    #----------------------- Main loop ----------------------------#
    
    init_time = time.time()
    if verbose > 0:
        if print_style == 'classical':
            if h_present is True and h_takes_infinite_values is False:
                print("elapsed time \t iter \t function value  change in x \t change in y")
            elif h_present is True and h_takes_infinite_values is True:
                print("elapsed time \t iter \t function value  infeasibility \t change in x \t change in y")
            else:
                print("elapsed time \t iter \t function value  change in x")
        elif print_style == 'smoothed_gap':
                            print("elapsed time\titer\tfunction value infeasibility\tsmoothed gap \tbeta     gamma  \tchange in x\tchange in y")

        nb_prints = 0

    # code in the case bloks_g = blocks only for the moment
    for iter in range(0,int(max_iter),2):
        if callback is not None:
            if callback(x, Sy, rf, rhx): break
        change_in_x = 0.
        change_in_y = 0.
        for f_iter in range(2*n):
            # with nogil:
                ii = rand_int(n, rand_r_state)
                nb_coord = blocks[ii+1] - blocks[ii]

                if h_present is True:
                    # compute rhy = Ah.T D(m) Sy
                    for i in range(nb_coord):
                        coord = blocks[ii] + i
                        rhy[i] = 0.
                        for l in range(Ah_indptr[coord], Ah_indptr[coord+1]):
                            jh = Ah_indices[l]
                            rhy[i] += Ah_data[l] * Sy[jh]

                # Apply prox of h* in the dual space
                if h_present is True:
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
                        # else: we have already computed prox_y[blocks_h[j]:blocks_h[j+1]], so nothing to do

                        # update Sy
                        Sy[jh] += 1. / Ah_nnz_perrow[jh] * (prox_y[jh] - y[lh])
                        # update y
                        change_in_y += fabs(prox_y[jh] - y[lh])
                        y[lh] = prox_y[jh]

                for i in range(nb_coord):
                    coord = blocks[ii] + i
                    x_ii[i] = x[coord]

                    # Compute gradient of f and do gradient step
                    if f_present is True:
                        grad[i] = 0.
                        for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                            j = Af_indices[l]
                            # TODO: code for the case blocks_f[j+1]-blocks_f[j] > 1
                            my_eval(f[j], rf[blocks_f[j]:blocks_f[j+1]], buff,
                                            nb_coord=blocks_f[j+1]-blocks_f[j], mode=GRAD)
                            grad[i] += cf[j] * Af_data[l] * buff[0]
                        x[coord] -= primal_step_size[ii] * grad[i]
                    if h_present is True:
                        # compute rhy_new = Ah.T D(m) Sy
                        rhy_new[i] = 0.
                        for l in range(Ah_indptr[coord], Ah_indptr[coord+1]):
                            jh = Ah_indices[l]
                            rhy_new[i] += Ah_data[l] * Sy[jh]
                        x[coord] -= primal_step_size[ii] * (2*rhy_new[i] - rhy[i])

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
                    change_in_x += fabs(x_ii[i] - x[coord])

        elapsed_time = time.time() - init_time
        if verbose > 0:
            if (elapsed_time > nb_prints * verbose
                    or change_in_x + change_in_y < min_change_in_x or elapsed_time > max_time
                    or iter >= max_iter-2):
                # Compute value
                compute_primal_value(pb, f, g, h, x, rf, rhx, buff_x, buff_y, buff,
                                         &primal_val, &infeas)
                if print_style == 'classical':
                    if h_present is True and h_takes_infinite_values is False:
                        print("%.5f \t %d \t %+.5e \t %.5e \t %.5e"
                                  %(elapsed_time, iter, primal_val, change_in_x, change_in_y))
                    elif h_present is True and h_takes_infinite_values is True:
                        print("%.5f \t %d \t %+.5e \t %.5e \t %.5e \t %.5e"
                                  %(elapsed_time, iter, primal_val, infeas, change_in_x, change_in_y))
                    else:  # h_present is False
                        print("%.5f \t %d \t %+.5e \t %.5e"
                                  %(elapsed_time, iter, primal_val, change_in_x))
                elif print_style == 'smoothed_gap':
                    beta = infeas
                    smoothed_gap = compute_smoothed_gap(pb, f, g, h, x, rf, rhx, Sy,
                                                    buff_x, buff_y, buff, &beta, &gamma)
                    
                    print("%.5f \t %d\t%+.5e\t%.5e\t%.5e\t%.2e %.1e\t%.5e\t%.5e"
                              %(elapsed_time, iter, primal_val, infeas, smoothed_gap, beta, gamma, change_in_x, change_in_y))

                nb_prints += 1

        if elapsed_time > max_time:
            print("Time limit reached: stopping the algorithm after %f s"
                      %elapsed_time)
            break
        if change_in_x + change_in_y < min_change_in_x:
            print("Not enough change in iterates (||x(t+1) - x(t)|| = %.5e): "
                      "stopping the algorithm" %change_in_x)
            break

    pb.sol = np.array(x).copy()
    pb.dual_sol = np.array(Sy).copy()
    pb.dual_sol_duplicated = np.array(y).copy()

    if f_present is True:
        free(f)
    if g_present is True:
        free(g)
    if h_present is True:
        free(h)


def check_grad(f, x, nb_coord=1):
    x = np.array(x, dtype='float')
    cdef DOUBLE[:] x_ = np.array(x)
    cdef DOUBLE[:] grad = np.array(x).copy()
    my_eval(f, x_, grad, nb_coord, mode=GRAD)
    cdef DOUBLE[:] grad_finite_diffs = np.array(x).copy()
    cdef DOUBLE[:] x_shift = np.array(x).copy()
    cdef int i
    cdef DOUBLE error = 0.
    for i in range(nb_coord):
        x_shift[i] = x[i] + 1e-6
        grad_finite_diffs[i] = (my_eval(f, x_shift, grad, nb_coord=1, mode=VAL)
                                    - my_eval(f, x_, grad, nb_coord=1, mode=VAL)) / 1e-6
        x_shift[i] = x[i]
        error += (grad_finite_diffs[i] - grad[i])**2
    return sqrt(error), np.array(grad), np.array(grad_finite_diffs)
