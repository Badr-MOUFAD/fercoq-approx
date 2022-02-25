# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True helpers.pyx

# C definitions in helpers.pxd
import numpy as np
import sys

cdef DOUBLE INF = 1e20

cdef void compute_primal_value(pb, atom* f, atom* g, atom* h,
                             DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx,
                             DOUBLE[:] rQ,
                             DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                             DOUBLE* val, DOUBLE* infeas):

    cdef UINT32_t N = pb.N
    cdef UINT32_t[:] blocks = pb.blocks
    cdef UINT32_t[:] blocks_f = pb.blocks_f
    cdef UINT32_t[:] blocks_h = pb.blocks_h
    cdef DOUBLE[:] cf = pb.cf
    cdef DOUBLE[:] bf = pb.bf
    cdef DOUBLE[:] cg = pb.cg
    cdef DOUBLE[:] Dg_data = np.array(pb.Dg.data[0], dtype=float)
    cdef DOUBLE[:] bg = pb.bg
    cdef DOUBLE[:] ch = pb.ch
    cdef DOUBLE[:] bh = pb.bh

    cdef UINT32_t ii, i, j, jh, l, coord, nbcoord
    val[0] = 0.
    infeas[0] = 0.
    for i in range(N):
        val[0] += 0.5 * x[i] * rQ[i]

    if pb.f_present is True:
        for j in range(len(pb.f)):
            val[0] += cf[j] * f[j](rf[blocks_f[j]:blocks_f[j+1]],
                                      buff,
                                      blocks_f[j+1]-blocks_f[j],
                                      VAL, useless_param, useless_param)
    if pb.g_present is True:
        for ii in range(len(pb.g)):
            nb_coord = blocks[ii+1] - blocks[ii]
            for i in range(nb_coord):
                coord = blocks[ii] + i
                buff_x[i] = Dg_data[ii] * x[coord] - bg[coord]
            val[0] += cg[ii] * g[ii](buff_x, buff,
                                        nb_coord, VAL, useless_param,
                                        useless_param)
    if pb.h_present is True:
        if pb.h_takes_infinite_values == False:
            for jh in range(len(pb.h)):
                val[0] += ch[jh] * h[jh](
                                        rhx[blocks_h[jh]:blocks_h[jh+1]],
                                        buff,
                                        blocks_h[jh+1]-blocks_h[jh],
                                        VAL, useless_param, useless_param)
        if pb.h_takes_infinite_values == True:
            for jh in range(len(pb.h)):
                for l in range(blocks_h[jh+1]-blocks_h[jh]):
                    coord = blocks_h[jh]+l
                    buff_y[l] = rhx[coord]
                # project rhx onto the domain of h
                h[jh](buff_y, buff,
                            blocks_h[jh+1]-blocks_h[jh],
                            PROX, 1e-20, useless_param)
                for l in range(blocks_h[jh+1]-blocks_h[jh]):
                    coord = blocks_h[jh]+l
                    infeas[0] += (buff[l] - rhx[coord]) ** 2
            infeas[0] = sqrt(infeas[0])


cdef DOUBLE compute_smoothed_gap(pb, atom* f, atom* g, atom* h,
                             DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx,
                             DOUBLE[:] rQ, DOUBLE[:] Sy, DOUBLE[:] z,
                             DOUBLE[:] AfTz, DOUBLE[:] w,
                             DOUBLE[:] x_center, DOUBLE[:] y_center,
                             DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                             DOUBLE* beta, DOUBLE* gamma, compute_z=True,
                             compute_gamma=True):
    # We output z and AfTz because it is useful when doing variable screening
    cdef UINT32_t ii, i, j, jh, l, coord, nbcoord
    cdef DOUBLE val = 0.

    cdef UINT32_t N = pb.N
    cdef UINT32_t[:] blocks = pb.blocks
    cdef UINT32_t[:] blocks_f = pb.blocks_f
    cdef UINT32_t[:] blocks_h = pb.blocks_h
    cdef DOUBLE[:] cf = pb.cf
    cdef DOUBLE[:] bf = pb.bf
    cdef DOUBLE[:] cg = pb.cg
    cdef DOUBLE[:] Dg_data = np.array(pb.Dg.data[0], dtype=float)
    cdef DOUBLE[:] bg = pb.bg
    cdef DOUBLE[:] ch = pb.ch
    cdef DOUBLE[:] bh = pb.bh
    
    if compute_z is True:
        # w is the dual variable associated to 0.5 xT Q x
        w = rQ.copy()
        val += np.array(x).dot(np.array(w))

        # z is the dual variable associated to f(Af x - bf)
        if pb.f_present is True:
            for j in range(len(pb.f)):
                f[j](rf[blocks_f[j]:blocks_f[j+1]], buff,
                     blocks_f[j+1]-blocks_f[j], GRAD,
                     useless_param, useless_param)
                for l in range(blocks_f[j+1]-blocks_f[j]):
                    coord = blocks_f[j] + l
                    z[coord] = cf[j] * buff[l]
            val += np.array(z).dot(np.array(rf)) + np.array(bf).dot(np.array(z))
            #        = f(Af x - bf) + f*(z) + bf.dot(z)
            AfTz_ = pb.Af.T.dot(np.array(z))
            for i in range(N):
                AfTz[i] = AfTz_[i]  # otherwise the pointer seems to be broken
        # else: AfTz is initialized with np.zeros(N)
    else:
        max_w = np.linalg.norm(np.array(w), np.inf)
        max_rQ = np.linalg.norm(np.array(rQ), np.inf)
        if max_w > 0:
            # we add 0.5 (x Q x + w inv(Q) w), knowing that Q x = rQ and w = rQ/scaling
            val += 0.5 * np.array(x).dot(np.array(rQ))
            val += 0.5 * np.array(x).dot(np.array(w)) * max_w / max_rQ
        if pb.f_present is True:
            for j in range(len(pb.f)):
                val += cf[j] * f[j](rf[blocks_f[j]:blocks_f[j+1]], buff,
                     blocks_f[j+1]-blocks_f[j], VAL,
                     useless_param, useless_param)
                for l in range(blocks_f[j+1]-blocks_f[j]):
                    coord = blocks_f[j] + l
                    buff_x[l] = z[coord] / cf[j]
                val +=  cf[j] * f[j](buff_x, buff,
                     blocks_f[j+1]-blocks_f[j], VAL_CONJ,
                     useless_param, useless_param)
            val += np.array(bf).dot(np.array(z))
    # print('contrib_f=', val, np.array(z).dot(np.array(rf)), np.array(bf).dot(np.array(z)))

    if pb.h_present is True:
        AhTSy = pb.Ah.T.dot(np.array(Sy))
    else:
        AhTSy = np.zeros(N)

    cdef DOUBLE[:] xbar = np.zeros(N, dtype=float)
    cdef DOUBLE[:] ybar = np.zeros(pb.Ah.shape[0], dtype=float)
    if pb.g_present is True:
        val_g = 0.
        # compute g(x)
        for ii in range(len(pb.g)):
            nb_coord = blocks[ii+1] - blocks[ii]
            for i in range(nb_coord):
                coord = blocks[ii] + i
                buff_x[i] = Dg_data[ii] * x[coord] - bg[coord]
            val_g += cg[ii] * g[ii](buff_x, buff, nb_coord, VAL,
                                       useless_param, useless_param)

        if compute_gamma == True:
            # estimate dual infeasibility
            dual_infeas = 0.
            for ii in range(len(pb.g)):
                nb_coord = blocks[ii+1] - blocks[ii]
                for i in range(nb_coord):
                    coord = blocks[ii] + i
                    buff_x[i] = 1. / Dg_data[ii] * (
                        - (AfTz[coord] + AhTSy[coord] + w[coord]))
                    # project -AfTz - AhTSy - w onto the domain of g*
                g[ii](buff_x, buff, nb_coord,
                      PROX_CONJ, 1./INF, cg[ii])
                for i in range(nb_coord):
                    dual_infeas += (Dg_data[ii] * (buff[i] - buff_x[i])) ** 2
            dual_infeas = sqrt(dual_infeas)
            gamma[0] = max(1./INF, dual_infeas)

        # compute g*_gamma(-AfTz - AhTSy - w;x_center) = -(AfTz + AhTSy + w)(xbar) - g(xbar) - gamma/2 ||x-x_center||**2
        val_g1 = 0.
        val_g2 = 0.
        val_g3 = 0.
        # note that g deals with bg directly in the prox
        for ii in range(len(pb.g)):
            # compute xbar
            nb_coord = blocks[ii+1] - blocks[ii]
            for i in range(nb_coord):
                coord = blocks[ii] + i
                buff_x[i] = Dg_data[ii] * (
                    x_center[coord] - 1. / gamma[0] * \
                       (AfTz[coord] + AhTSy[coord] + w[coord])) - bg[coord]
            g[ii](buff_x, buff, nb_coord, PROX,
                  cg[ii]*Dg_data[ii]**2/gamma[0], useless_param)
            for i in range(nb_coord):
                coord = blocks[ii] + i
                xbar[coord] = 1. / Dg_data[ii] * (buff[i] + bg[coord])

            # compute g*_gamma(-AfTz - AhTSy;x)
            for i in range(nb_coord):
                coord = blocks[ii] + i
                buff_x[i] = buff[i]  # = Dg_data[ii] * xbar[coord] - bg[coord]
            val_g1 -= cg[ii] * g[ii](buff_x, buff, nb_coord, VAL,
                                        useless_param, useless_param)
            for i in range(nb_coord):
                coord = blocks[ii] + i
                val_g2 -= (AfTz[coord] + AhTSy[coord] + w[coord]) * xbar[coord]
                val_g3 -= gamma[0] / 2. * (xbar[coord] - x_center[coord])**2
        val += val_g + val_g1 + val_g2 + val_g3
        # print('contrib_g=', val_g + val_g1 + val_g2 + val_g3, val_g, val_g1, val_g2, val_g3)

    if pb.h_present is True:
        val_h = 0.
        val_hh = 0.
        val_h2 = 0.
        # compute h*(Sy) + bh.Sy
        test = 0.
        for jh in range(len(pb.h)):
            nb_coord = blocks_h[jh+1] - blocks_h[jh]
            for l in range(nb_coord):
                coord = blocks_h[jh] + l
                buff_y[l] = Sy[coord] / ch[jh]
            val_h += ch[jh] * h[jh](buff_y, buff, nb_coord,
                                       VAL_CONJ, useless_param, useless_param)
            for l in range(nb_coord):
                coord = blocks_h[jh] + l
                val_hh += bh[coord] * Sy[coord]

        if pb.h_takes_infinite_values == False:
            # compute h(Ah x - bh)
            for jh in range(len(pb.h)):
                val_h2 += ch[jh] * h[jh](
                                        rhx[blocks_h[jh]:blocks_h[jh+1]],
                                        buff,
                                        blocks_h[jh+1] - blocks_h[jh],
                                        VAL, useless_param, useless_param)
        if pb.h_takes_infinite_values == True:
            # compute h_beta(Ah x - bh; y_center) = (Ah x - bh) ybar - h*(ybar) - beta/2 ||y_center - ybar||**2
            beta[0] = max(1./INF, beta[0])
            for jh in range(len(pb.h)):
                nb_coord = blocks_h[jh+1] - blocks_h[jh]
                # compute ybar
                for l in range(nb_coord):
                    coord = blocks_h[jh] + l
                    buff_y[l] = y_center[coord] + 1. / beta[0] * rhx[coord]
                h[jh](buff_y, buff, nb_coord, PROX_CONJ,
                      1./beta[0], ch[jh])

                # compute -h*(ybar) = -ybar.ybarbar + h(ybarbar)
                for l in range(nb_coord):
                    coord = blocks_h[jh] + l
                    ybar[coord] = buff[l]
                    buff_y[l] = y_center[coord] + INF * ybar[coord]
                h[jh](buff_y, buff, nb_coord, PROX,
                      ch[jh]*INF, useless_param)
                for l in range(nb_coord):
                    buff_y[l] = buff[l]
                h_ybarbar = h[jh](buff_y, buff, nb_coord, VAL,
                                  useless_param, useless_param)
                val_h2 += h_ybarbar
                for l in range(nb_coord):
                    coord = blocks_h[jh] + l
                    val_h2 -= buff_y[l] * ybar[coord]
                    val_h2 += rhx[coord] * ybar[coord] - beta[0] / 2. * (y_center[coord] - ybar[coord])**2
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
