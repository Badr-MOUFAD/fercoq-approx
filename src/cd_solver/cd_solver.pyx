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
      
      def __init__(self, N, blocks=None, x_init=None,
                         f=None, cf=None, Af=None, bf=None, blocks_f=None,
                         g=None, cg=None, Dg=None, bg=None,
                         h=None, ch=None, Ah=None, bh=None, blocks_h=None):
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
                  self.blocks = np.arange(N+1, dtype=np.uint32)
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
                  if Af.shape[1] != N:
                        raise Warning("dimensions of Af and x do not match")
                  if bf is None:
                        bf = np.zeros(Af.shape[0])
                  if len(bf) != Af.shape[0]:
                        raise Warning("dimensions of Af and bf do not match")
                  if blocks_f is None:
                        blocks_f = np.arange(len(f)+1, dtype=np.uint32)
                    
            else:
                self.f_present = False
                cf = Af = bf = np.empty(0)

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
                  if Ah.shape[1] != N:
                        raise Warning("dimensions of Ah and x do not match")
                  elif bh is None:
                        bh = np.zeros(Ah.shape[0])
                  if len(bh) != Ah.shape[0]:
                        raise Warning("dimensions of Dh and bh do not match")
                  if blocks_h is None:
                        blocks_h = np.arange(len(h)+1, dtype=np.uint32)
            else:
                self.h_present = False
                ch = Ah = bh = np.empty(0)

            self.f = f
            self.cf = np.array(cf, dtype=float)
            self.Af = sparse.csc_matrix(Af)
            self.bf = np.array(bf, dtype=float)
            self.blocks_f = blocks_f
            self.g = g
            self.cg = np.array(cg, dtype=float)
            self.Dg = Dg
            self.bg = np.array(bg, dtype=float)
            self.h = h
            self.ch = np.array(ch, dtype=float)
            self.Ah = sparse.csc_matrix(Ah)
            self.bh = np.array(bh, dtype=float)
            self.blocks_h = blocks_h


def coordinate_descent(pb, max_iter=1000, verbose=0, min_change_in_x=1e-20):

    cdef int ii, j, k, l, i, coord
    cdef int f_iter
    cdef int nb_coord

    # Problem pb
    cdef DOUBLE[:] x = pb.x_init.copy()
    cdef int N
    cdef UINT32_t[:] blocks = pb.blocks
    cdef UINT32_t[:] blocks_f = pb.blocks_f
    cdef UINT32_t[:] blocks_h = pb.blocks_h

    
    cdef int n = len(pb.blocks) - 1
    cdef DOUBLE[:] cf = pb.cf
    cdef int[:] Af_indptr = pb.Af.indptr
    cdef int[:] Af_indices = pb.Af.indices
    cdef DOUBLE[:] Af_data = np.array(pb.Af.data, dtype=float)
    cdef DOUBLE[:] bf = pb.bf
    cdef DOUBLE[:] cg = pb.cg
    cdef DOUBLE[:] Dg_data = np.array(pb.Dg.data[0], dtype=float)
    cdef DOUBLE[:] bg = pb.bg
    cdef DOUBLE[:] ch = pb.ch
    cdef int[:] Ah_indptr = pb.Ah.indptr
    cdef int[:] Ah_indices = pb.Ah.indices
    cdef DOUBLE[:] Ah_data = np.array(pb.Ah.data, dtype=float)
    cdef DOUBLE[:] bh = pb.bh

    cdef int f_present = pb.f_present
    cdef unsigned char** f = <unsigned char**>malloc(len(pb.f)*sizeof(char*))
    if f_present is True:
        for j in range(len(pb.f)):
            f[j] = pb.f[j]

    cdef int g_present = pb.g_present
    cdef unsigned char** g = <unsigned char**>malloc(len(pb.g)*sizeof(char*))
    if g_present is True:
        for ii in range(len(pb.g)):
            g[ii] = pb.g[ii]

    cdef int h_present = pb.h_present

    # Definition of residuals
    cdef DOUBLE[:] rf
    if f_present is True:
        rf = pb.Af * x - pb.bf
    cdef DOUBLE[:] rh
    # if pb.h is not None:
    #    rh = pb.Ah * x - pb.bh

    cdef UINT32_t rand_r_state_seed = np.random.randint(RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    max_nb_coord = np.max(np.diff(pb.blocks))
    cdef DOUBLE[:] grad = np.zeros(max_nb_coord)
    cdef DOUBLE[:] x_ii = np.zeros(max_nb_coord)
    cdef DOUBLE[:] buff_x = np.zeros(max_nb_coord)
    cdef DOUBLE[:] buff = np.zeros(max_nb_coord)

    # Compute Lipschitz constants
    cdef DOUBLE[:] tmp_Lf = np.zeros(len(pb.f))
    for j in range(len(pb.f)):
        tmp_Lf[j] = my_eval(f[j], buff_x, buff, nb_coord=blocks_f[j+1]-blocks_f[j],
                                mode=LIPSCHITZ)
    cdef DOUBLE[:] Lf = np.zeros(n)
    if f_present is True:
        for ii in range(n):
            for i in range(blocks[ii+1] - blocks[ii]):
                coord = blocks[ii] + i
                for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                    j = Af_indices[l]
                    Lf[ii] += cf[j] * Af_data[l]**2 * tmp_Lf[j]
            Lf[ii] = np.maximum(Lf[ii], 1e-30)
    del tmp_Lf

    cdef DOUBLE change_in_x

    init_time = time.time()
    if verbose > 0:
        print("elapsed time \t iter \t function value  change in x")
        nb_prints = 0

    # code in the case n = N only for the moment
    for iter in range(max_iter):
        change_in_x = 0.
        for f_iter in range(n):
            # with nogil:
                ii = rand_int(n, rand_r_state)
                nb_coord = blocks[ii+1] - blocks[ii]
                for i in range(nb_coord):
                    coord = blocks[ii] + i
                    x_ii[i] = x[coord]

                    # Compute gradient of f and do gradient step
                    if f_present is True:
                        grad[i] = 0.
                        for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                            j = Af_indices[l]
                            my_eval(f[j], rf[blocks_f[j]:blocks_f[j+1]], buff,
                                            nb_coord=blocks_f[j+1]-blocks_f[j], mode=GRAD)
                            grad[i] += cf[j] * Af_data[l] * buff[i]
                        x[coord] = x[coord] - 1./Lf[ii]*grad[i]
                
                # Apply prox of g
                if g_present is True:
                    for i in range(nb_coord):
                        coord = blocks[ii] + i
                        buff_x[i] = Dg_data[ii] * x[coord] - bg[coord]
                    my_eval(g[ii], buff_x, buff, nb_coord=nb_coord,
                            mode=PROX, prox_param=cg[ii]*Dg_data[ii]**2/Lf[ii])
                    for i in range(nb_coord):
                        coord = blocks[ii] + i
                        x[coord] = buff[i]

                # if pb.h is not None:
                     # Not coded yet"
                
                # Update residuals
                for i in range(nb_coord):
                    coord = blocks[ii] + i
                    if x_ii[i] != x[coord]:
                        if f_present is True:
                            for l in range(Af_indptr[coord], Af_indptr[coord+1]):
                                j = Af_indices[l]
                                rf[j] += Af_data[l] * (x[coord] - x_ii[i])
                    change_in_x += fabs(x_ii[i] - x[coord])

        if verbose > 0:
             elapsed_time = time.time() - init_time
             if (elapsed_time > nb_prints * verbose
                    or change_in_x < min_change_in_x):
                 # Compute value
                 val = 0.
                 if f_present is True:
                     for j in range(len(pb.f)):
                         val += cf[j] * my_eval(f[j], rf[blocks_f[j]:blocks_f[j+1]],
                                                    buff, nb_coord=blocks_f[j+1]-blocks_f[j])
                 if g_present is True:
                     for ii in range(len(pb.g)):
                         nb_coord = blocks[ii+1] - blocks[ii]
                         for i in range(nb_coord):
                            coord = blocks[ii] + i
                            buff_x[i] = Dg_data[ii] * x[coord] - bg[coord]
                         val += cg[ii] * my_eval(g[ii], buff_x, buff, nb_coord=nb_coord)

                 print("%.5f \t %d \t %+.5e \t %.5e"
                           %(elapsed_time, iter, val, change_in_x))
                 nb_prints += 1

        if change_in_x < min_change_in_x:
            break;

    pb.sol = np.array(x)

    free(f)
    free(g)


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
