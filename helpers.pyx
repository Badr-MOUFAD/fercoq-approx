# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True helpers.pyx

# C definitions in helpers.pxd
import numpy as np
import sys

cdef DOUBLE INF = 1e20

cdef void compute_primal_value(pb, atom* f, atom* g, atom* h,
                             DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx,
                             DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                             DOUBLE* val, DOUBLE* infeas):
    cdef UINT32_t ii, i, j, jh, l, coord, nbcoord
    val[0] = 0.
    infeas[0] = 0.
    if pb.f_present is True:
        for j in range(len(pb.f)):
            val[0] += pb.cf[j] * f[j](rf[pb.blocks_f[j]:pb.blocks_f[j+1]],
                                      buff,
                                      pb.blocks_f[j+1]-pb.blocks_f[j],
                                      VAL, useless_param, useless_param)
    if pb.g_present is True:
        for ii in range(len(pb.g)):
            nb_coord = pb.blocks[ii+1] - pb.blocks[ii]
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                buff_x[i] = pb.Dg.data[0][ii] * x[coord] - pb.bg[coord]
            val[0] += pb.cg[ii] * g[ii](buff_x, buff,
                                        nb_coord, VAL, useless_param,
                                        useless_param)
    if pb.h_present is True:
        if pb.h_takes_infinite_values == False:
            for jh in range(len(pb.h)):
                val[0] += pb.ch[jh] * h[jh](
                                        rhx[pb.blocks_h[jh]:pb.blocks_h[jh+1]],
                                        buff,
                                        pb.blocks_h[jh+1]-pb.blocks_h[jh],
                                        VAL, useless_param, useless_param)
        if pb.h_takes_infinite_values == True:
            for jh in range(len(pb.h)):
                for l in range(pb.blocks_h[jh+1]-pb.blocks_h[jh]):
                    coord = pb.blocks_h[jh]+l
                    buff_y[l] = rhx[coord]
                # project rhx onto the domain of h
                h[jh](buff_y, buff,
                            pb.blocks_h[jh+1]-pb.blocks_h[jh],
                            PROX, 1e-20, useless_param)
                for l in range(pb.blocks_h[jh+1]-pb.blocks_h[jh]):
                    coord = pb.blocks_h[jh]+l
                    infeas[0] += fabs(buff[l] - rhx[coord])


cdef DOUBLE compute_smoothed_gap(pb, atom* f, atom* g, atom* h,
                             DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx,
                             DOUBLE[:] Sy, DOUBLE[:] z, DOUBLE[:] AfTz,
                             DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                             DOUBLE* beta, DOUBLE* gamma, compute_z=True):
    # We output z and AfTz because it is useful when doing variable screening
    cdef UINT32_t ii, i, j, jh, l, coord, nbcoord
    cdef DOUBLE val = 0.
    if compute_z is True:
        # z is the dual variable associated to f(Af x - bf)
        if pb.f_present is True:
            for j in range(len(pb.f)):
                f[j](rf[pb.blocks_f[j]:pb.blocks_f[j+1]], buff,
                     pb.blocks_f[j+1]-pb.blocks_f[j], GRAD,
                     useless_param, useless_param)
                for l in range(pb.blocks_f[j+1]-pb.blocks_f[j]):
                    coord = pb.blocks_f[j] + l
                    z[coord] = pb.cf[j] * buff[l]
            val += np.array(z).dot(np.array(rf)) + pb.bf.dot(np.array(z))
            #        = f(Af x - bf) + f*(z) + bf.dot(z)
            AfTz_ = pb.Af.T.dot(np.array(z))
            for i in range(pb.N):
                AfTz[i] = AfTz_[i]  # otherwise the pointer seems to be broken
        # else: AfTz is initialized with np.zeros(pb.N)
    else:
        if pb.f_present is True:
            for j in range(len(pb.f)):
                val += pb.cf[j] * f[j](rf[pb.blocks_f[j]:pb.blocks_f[j+1]], buff,
                     pb.blocks_f[j+1]-pb.blocks_f[j], VAL,
                     useless_param, useless_param)
                for l in range(pb.blocks_f[j+1]-pb.blocks_f[j]):
                    coord = pb.blocks_f[j] + l
                    buff_x[l] = z[coord] / pb.cf[j]
                val +=  pb.cf[j] * f[j](buff_x, buff,
                     pb.blocks_f[j+1]-pb.blocks_f[j], VAL_CONJ,
                     useless_param, useless_param)
            val += pb.bf.dot(np.array(z))
            
    if pb.h_present is True:
        AhTSy = pb.Ah.T.dot(np.array(Sy))
    else:
        AhTSy = np.zeros(pb.N)

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
            val_g += pb.cg[ii] * g[ii](buff_x, buff, nb_coord, VAL,
                                       useless_param, useless_param)

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
            g[ii](buff_x, buff, nb_coord,
                  PROX_CONJ, 1./INF, pb.cg[ii])
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
            g[ii](buff_x, buff, nb_coord, PROX,
                  pb.cg[ii]*pb.Dg.data[0][ii]**2/gamma[0], useless_param)
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                xbar[coord] = 1. / pb.Dg.data[0][ii] * (buff[i] + pb.bg[coord])

            # compute g*_gamma(-AfTz - AhTSy;x)
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                buff_x[i] = pb.Dg.data[0][ii] * xbar[coord] - pb.bg[coord]
            val_g1 -= pb.cg[ii] * g[ii](buff_x, buff, nb_coord, VAL,
                                        useless_param, useless_param)
            for i in range(nb_coord):
                coord = pb.blocks[ii] + i
                val_g2 -= (AfTz[coord] + AhTSy[coord]) * xbar[coord]
                val_g3 -= gamma[0] / 2. * (xbar[coord] - x[coord])**2
        val += val_g + val_g1 + val_g2 + val_g3
        # print('contrib_g=', val_g + val_g1 + val_g2 + val_g3, val_g, val_g1, val_g2, val_g3)

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
            val_h += pb.ch[jh] * h[jh](buff_y, buff, nb_coord,
                                       VAL_CONJ, useless_param, useless_param)
            for l in range(nb_coord):
                coord = pb.blocks_h[jh] + l
                val_hh += pb.bh[coord] * Sy[coord]

        if pb.h_takes_infinite_values == False:
            # compute h(Ah x - bh)
            for jh in range(len(pb.h)):
                val_h2 += pb.ch[jh] * h[jh](
                                        rhx[pb.blocks_h[jh]:pb.blocks_h[jh+1]],
                                        buff,
                                        pb.blocks_h[jh+1] - pb.blocks_h[jh],
                                        VAL, useless_param, useless_param)
        if pb.h_takes_infinite_values == True:
            # compute h_beta(Ah x - bh; Sy) = (Ah x - bh) ybar - h*(ybar) - beta/2 ||Sy - ybar||**2
            beta[0] = max(1./INF, beta[0])
            for jh in range(len(pb.h)):
                nb_coord = pb.blocks_h[jh+1] - pb.blocks_h[jh]
                # compute ybar
                for l in range(nb_coord):
                    coord = pb.blocks_h[jh] + l
                    buff_y[l] = Sy[coord] + 1. / beta[0] * rhx[coord]
                h[jh](buff_y, buff, nb_coord, PROX_CONJ,
                      1./beta[0], pb.ch[jh])

                # compute -h*(ybar) = -ybar.ybarbar + h(ybarbar)
                for l in range(nb_coord):
                    coord = pb.blocks_h[jh] + l
                    ybar[coord] = buff[l]
                    buff_y[l] = Sy[coord] + INF * ybar[coord]
                h[jh](buff_y, buff, nb_coord, PROX,
                      pb.ch[jh]*INF, useless_param)
                for l in range(nb_coord):
                    buff_y[l] = buff[l]
                h_ybarbar = h[jh](buff_y, buff, nb_coord, VAL,
                                  useless_param, useless_param)
                val_h2 += h_ybarbar
                for l in range(nb_coord):
                    coord = pb.blocks_h[jh] + l
                    val_h2 -= buff_y[l] * ybar[coord]
                    val_h2 += rhx[coord] * ybar[coord] - beta[0] / 2. * (Sy[coord] - ybar[coord])**2
        # print('contrib h:', val_h, val_hh, val_h2, np.array(Sy))
        val += val_h + val_hh + val_h2
    return val


def check_grad(f, x, nb_coord=1, shift=1e-6):
    if sys.version_info[0] > 2 and isinstance(f, bytes) == True:
        f = f.encode()
    cdef atom func = string_to_func(<bytes> f)
    cdef DOUBLE[:] x_ = np.array(x, dtype='float')
    cdef DOUBLE[:] grad = np.array(x_).copy()
    func(x_, grad, nb_coord, GRAD, useless_param, useless_param)
    cdef DOUBLE[:] grad_finite_diffs = np.array(x_).copy()
    cdef DOUBLE[:] x_shift = np.array(x_).copy()
    cdef int i
    cdef DOUBLE error = 0.
    for i in range(nb_coord):
        x_shift[i] = x_[i] + shift
        grad_finite_diffs[i] = (func(x_shift, grad, nb_coord, VAL, useless_param, useless_param)
                                    - func(x_, grad, nb_coord, VAL, useless_param, useless_param)) / shift
        x_shift[i] = x_[i]
        error += (grad_finite_diffs[i] - grad[i])**2
    return sqrt(error), np.array(grad), np.array(grad_finite_diffs)


def my_eval_python(f, x, nb_coord=1, mode=0, param=1., param2=1.):
    if sys.version_info[0] > 2 and isinstance(f, bytes) == True:
        f = f.encode()
    cdef atom func = string_to_func(<bytes> f)
    cdef DOUBLE[:] x_ = np.array(x, dtype='float')
    cdef DOUBLE[:] buff_x = np.array(x_).copy()
    val = func(x_, buff_x, nb_coord, mode, param, param2)
    return val, np.array(buff_x)
