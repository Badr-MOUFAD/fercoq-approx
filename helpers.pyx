# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True helpers.pyx

# C definitions in helpers.pxd

import numpy as np
import sys


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
        val += z.dot(np.array(rf)) + pb.bf.dot(z)   # = f(Af x - bf) + f*(z) + bf.dot(z)
        # print('contrib f:', val)
        AfTz = pb.Af.T.dot(z)
    else:
        AfTz = np.zeros(pb.N)
    if pb.h_present is True:
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
        # print('g(x) = ', val_g)

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
                        nb_coord=nb_coord,
                        mode=PROX, prox_param=pb.cg[ii]*pb.Dg.data[0][ii]**2/gamma[0])
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                xbar[coord] = 1. / pb.Dg.data[0][ii] * (buff[i] + pb.bg[coord])

            # compute g*_gamma(-AfTz - AhTSy;x)
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                buff_x[i] = pb.Dg.data[0][ii] * xbar[coord] - pb.bg[coord]
            val_g1 -= pb.cg[ii] * my_eval(g[ii], buff_x, buff, nb_coord=nb_coord)
            # print('g_ii(xbar_ii) = ', pb.cg[ii] * my_eval(g[ii], buff_x, buff, nb_coord=nb_coord))
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                val_g2 -= (AfTz[coord] + AhTSy[coord]) * xbar[coord]
                val_g3 -= gamma[0] / 2. * (xbar[coord] - x[coord])**2
        # print('contrib g:', val_g, val_g1, val_g2, val_g3, np.array(xbar), np.array(x), np.array(AfTz))
        val += val_g + val_g1 + val_g2 + val_g3

    if pb.h_present is True:
        val_h = 0.
        val_hh = 0.
        val_h2 = 0.
        # compute h*(Sy) + bh.Sy
        test = 0.
        for jh in range(len(pb.h)):
            nb_coord = pb.blocks_h[jh+1] - pb.blocks_h[jh]
            for l in range(nb_coord):
                coord = pb.blocks_h[jh] + l
                buff_y[l] = Sy[coord] / pb.ch[jh]
            val_h += pb.ch[jh] * my_eval(h[jh], buff_y, buff, nb_coord=nb_coord,
                            mode=VAL_CONJ)
            for l in range(nb_coord):
                coord = pb.blocks_h[jh] + l
                val_hh += pb.bh[coord] * Sy[coord]

        if pb.h_takes_infinite_values == False:
            # compute h(Ah x - bh)
            for jh in range(len(pb.h)):
                val_h2 += pb.ch[jh] * my_eval(h[jh],
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
                            nb_coord=nb_coord,
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
                val_h2 += h_ybarbar
                for l in range(nb_coord):
                    coord = pb.blocks_h[jh] + l
                    val_h2 -= buff_y[l] * ybar[coord]
                    val_h2 += rhx[coord] * ybar[coord] - beta[0] / 2. * (Sy[coord] - ybar[coord])**2
        # print('contrib h:', val_h, val_hh, val_h2, np.array(Sy))
        val += val_h + val_hh + val_h2
    return val


def check_grad(f, x, nb_coord=1, shift=1e-6):
    if sys.version_info[0] > 2:
        f = f.encode()
    cdef unsigned char* f_str = <bytes> f
    cdef DOUBLE[:] x_ = np.array(x, dtype='float')
    cdef DOUBLE[:] grad = np.array(x_).copy()
    my_eval(f_str, x_, grad, nb_coord, mode=GRAD)
    cdef DOUBLE[:] grad_finite_diffs = np.array(x_).copy()
    cdef DOUBLE[:] x_shift = np.array(x_).copy()
    cdef int i
    cdef DOUBLE error = 0.
    for i in range(nb_coord):
        x_shift[i] = x_[i] + shift
        grad_finite_diffs[i] = (my_eval(f_str, x_shift, grad, nb_coord=nb_coord, mode=VAL)
                                    - my_eval(f_str, x_, grad, nb_coord=nb_coord, mode=VAL)) / shift
        x_shift[i] = x_[i]
        error += (grad_finite_diffs[i] - grad[i])**2
    return sqrt(error), np.array(grad), np.array(grad_finite_diffs)


def my_eval_python(f, x, nb_coord=1, mode=0):
    if sys.version_info[0] > 2:
        f = f.encode()
    cdef unsigned char* f_str = <bytes> f
    cdef DOUBLE[:] x_ = np.array(x, dtype='float')
    cdef DOUBLE[:] buff_x = np.array(x_).copy()
    val = my_eval(f_str, x_, buff_x, nb_coord, mode=mode)
    return val, np.array(buff_x)
