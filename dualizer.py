from .cd_solver_ import *
import scipy.sparse as sp
import numpy as np
import copy
norm = np.linalg.norm

def fenchel(f):
    # returns [a, cf, af, bf, l] such that f*(y) = cf a(af y - bf) + l y
    if f == "abs":
        return ["box_zero_one", 1, 0.5, -0.5, 0]
    elif f == "box_zero_one":
        return ["abs", 0.5, 1., 0., 0.5]
    elif f == "square":
        return ["square", 0.25, 1., 0., 0.]
    elif f == "eq_const":
        return ["zero", 1., 1., 0., 0.]
    elif f == "zero":
        return ["eq_const", 1., 1., 0., 0.]
    elif f == "ineq_const":
        return ["ineq_const", 1., -1, 0., 0.]
    elif f == "second_order_cone":
        return ["second_order_cone", 1., -1, 0., 0.]
    else:
        print("fenchel conjugate not implemented for " + f)
        return ["error_atom", 0., 0., 0., 0.]


def dualizer(pb_in_):
    pb_in = copy.copy(pb_in_)
    if pb_in.f_present == 1:
        if len(pb_in.f) == 1 and pb_in.f[0] == "linear":
            af_in = pb_in.Af.toarray().ravel()
        else:
            print("Dualizer only available when fenchel conjugate is available."
                   "Please put differentiable function in h and the linear"
                   "function in f.")
            return -1
    else:
        af_in = 0

    if norm(pb_in.bg) + norm(pb_in.bh) > 0:
        f = ["linear"]
        if pb_in.g_present == 0:
            Af = pb_in.bh
            bf = [0]
        else:
            Dgbg = 0 * pb_in.bg
            for i in range(len(pb_in.g)):
                Dgbg[pb_in.blocks[i]:pb_in.blocks[i+1]] = \
                    pb_in.Dg.data[0][i] * pb_in.bg[pb_in.blocks[i]:pb_in.blocks[i+1]]
            Af = pb_in.bh - (pb_in.Ah.dot(Dgbg)).T
            bf = [-np.sum(af_in * Dgbg)]
    else:
        f = None
        Af = pb_in.bh
        bf = None

    if pb_in.h_present == 1:
        cg = pb_in.ch.copy()
        Dg = sp.diags(1 / cg)
        blocks = pb_in.blocks_h.copy()
        N = blocks[-1]
        bg = np.zeros(N)
        g = []
        for i, h_k in enumerate(pb_in.h):
            atom_fench, c_fench, a_fench, b_fench, l_fench = fenchel(h_k)
            g.append(atom_fench)
            Af[blocks[i]:blocks[i+1]] += cg[i] * l_fench
            cg[i] *= c_fench
            Dg.data[0,i] *= a_fench
            bg[i] *= a_fench
            bg[i] += b_fench

    if pb_in.g_present == 0:
        pb_in.g = ["zero"]
        pb_in.blocks = np.array([0, pb_in.N])
        pb_in.Dg = sp.eye(1)
        pb_in.cg = [1]
        pb_in.g_present = 1
        pb_in.bg = [0]
    if pb_in.g_present == 1:
        ch = pb_in.cg.copy()
        blocks_h = pb_in.blocks.copy()
        Dgcg = 0 * pb_in.bg
        for i in range(len(pb_in.g)):
            Dgcg[pb_in.blocks[i]:pb_in.blocks[i+1]] = \
                pb_in.Dg.data[0][i] * pb_in.cg[i]
        Ah = -(pb_in.Ah.multiply(1 / Dgcg)).T
        Ah = sp.csr_matrix(Ah)  # for fast row access
        bh = af_in / Dgcg
        h = []
        for k, g_i in enumerate(pb_in.g):
            atom_fench, c_fench, a_fench, b_fench, l_fench = fenchel(g_i)
            h.append(atom_fench)
            Af += ch[k] * l_fench * \
                  np.array(np.sum(Ah[blocks_h[k]:blocks_h[k+1], :], axis=0)).ravel()
            ch[k] *= c_fench
            if a_fench != 1:
                Ah[blocks_h[k]:blocks_h[k+1], :] *= a_fench
                bh[k] *= a_fench
            bh[k] += b_fench

        Ah = sp.csc_matrix(Ah)
        N = Ah.shape[1]  # this should be equal to blocks[-1]

    pb_out = cd_solver.Problem(N=N, f=f, Af=Af, bf=bf,
                               g=g, Dg=Dg, cg=cg, bg=bg, blocks=blocks,
                               h=h, Ah=Ah, ch=ch, bh=bh, blocks_h=blocks_h)

    return pb_out


def my_str_mult(number):
    if number == 1:
        return ""
    elif number == -1:
        return "-"
    else:
        return str(number)

def my_str_add(number):
    if number == 0:
        return ""
    elif number >= 0:
        return "+"+str(number)
    else:  #  number < 0
        return str(number)

def print_problem(pb):
    seq="min_x "
    if pb.Q.nnz > 0:
        seq += "+ 0.5 * ("
        for i in range(len(pb.Q.indptr)):
            for j in range(pb.Q.indptr[i], pb.Q.indptr[i+1]):
                if pb.Q.data[j] >= 0:
                    seq += "+"
                seq += my_str_mult(pb.Q.data[j])+"x["+str(pb.Q.indices[j])+"]x["+str(i)+"]"
        seq += ") "
    if pb.f_present == 1:
        AfT = sp.csc_matrix(pb.Af.T)
        for j, fj in enumerate(pb.f):
            seq += " + "+my_str_mult(pb.cf[j])+fj+"("
            for jj in range(pb.blocks_f[j], pb.blocks_f[j+1]):
                for i in range(AfT.indptr[jj], AfT.indptr[jj+1]):
                    if AfT.data[i] >= 0:
                        seq += "+"
                    seq += my_str_mult(AfT.data[i])+"x["+str(AfT.indices[i])+"]"
                seq += my_str_add(-pb.bf[jj])
                seq += ")"
    if pb.g_present == 1:
        for i, gi in enumerate(pb.g):
            seq += " + "+my_str_mult(pb.cg[i])+gi+"("
            for ii in range(pb.blocks[i], pb.blocks[i+1]):
                if pb.Dg.data[0,i] >= 0:
                    seq += "+"
                seq += my_str_mult(pb.Dg.data[0,i])+"x["+str(ii)+"]"
                seq += my_str_add(-pb.bg[ii])
                seq += ")"
    if pb.h_present == 1:
        AhT = sp.csc_matrix(pb.Ah.T)
        for k, hk in enumerate(pb.h):
            seq += " + "+my_str_mult(pb.ch[k])+hk+"("
            for kk in range(pb.blocks_h[k], pb.blocks_h[k+1]):
                for i in range(AhT.indptr[kk], AhT.indptr[kk+1]):
                    if AhT.data[i] >= 0:
                        seq += "+"
                    seq += my_str_mult(AhT.data[i])+"x["+str(AhT.indices[i])+"]"
                seq += my_str_add(-pb.bh[kk])
                seq += ")"
    print(seq)
    return seq

def test():
    A = np.array([[0, 1, 2, 3], [4, 5, -6, 7]])
    b = np.array([-1, 1])
    alpha = max(1e-6, norm(A.T.dot(b), np.inf)) / 10
    pb_primal = cd_solver.Problem(N=A.shape[1], h=["square"]*A.shape[0],
                                  ch=[0.5]*A.shape[0], bh=b, Ah=A,
                              g=["abs"]*A.shape[1], cg=[alpha]*A.shape[1])
    pb_dual = dualizer(pb_primal)
    print('Primal Lasso problem')
    s = print_problem(pb_primal)
    print('Dual Lasso problem')
    s = print_problem(pb_dual)

    print('The final values should be opposite')
    cd_solver.coordinate_descent(pb_primal, verbose = 0.001)

    cd_solver.coordinate_descent(pb_dual, verbose = 0.001)

    b = np.array([0, 3])
    pb_primal2 = cd_solver.Problem(N=A.shape[1], h=["eq_const"]*A.shape[0],
                              bh=b, Ah=A,
                              g=["ineq_const"]*A.shape[1],
                              f=["linear"], Af=np.ones(A.shape[1]))
    pb_dual2 = dualizer(pb_primal2)
    print('Primal LP')
    s = print_problem(pb_primal2)
    print('Dual LP')
    s = print_problem(pb_dual2)

    print('The final values should be opposite')
    cd_solver.coordinate_descent(pb_primal2, verbose = 0.001, min_change_in_x=0)

    cd_solver.coordinate_descent(pb_dual2, verbose = 0.001, min_change_in_x=0)
