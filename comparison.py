# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
import numpy as np

import scipy.sparse as sp
from helpers import check_grad
import time

# imports for loading datasets
from scipy import io
from sklearn.datasets.mldata import fetch_mldata
from sklearn.externals.joblib import Memory
import sys
sys.path.append("../tv_l1_solver")
from load_poldrack import load_gain_poldrack

# imports for solvers
import cd_solver
from sklearn import linear_model
from sklearn import svm

def cd_pure_python(f, Af, bf, g, max_iter=100):
    N = Af.shape[1]
    x = np.zeros(N)
    r = np.matrix(Af.dot(x) - bf).T
    Lip = (Af.multiply(Af)).sum(axis=0) * f(r, mode='Lipschitz')
    step_size = 1. / Lip
    for k in range(max_iter):
        print('iter = ', k)
        for f_iter in range(N):
            i = np.random.randint(N)
            x_ii = x[i]
            tmp = g[i](x[i] - step_size[0,i] * Af[:,i].T.dot(f(r, mode='grad'))[0,0],
                        mode='prox',
                        prox_param=step_size[0,i])
            x[i] = tmp
            if x[i] != x_ii:
                r = r + (x[i] - x_ii) * Af[:,i]
    return x


if 0:
    print("Lasso on Leukemia")
    dataset = 'leukemia'
    data = fetch_mldata(dataset)
    X = data.data
    X = sp.csc_matrix(X)
    y = data.target
    alpha = 0.1 * np.linalg.norm(X.T.dot(y), np.inf)

    max_iter = 100

    def half_squares(r, mode):
        if mode == 'Lipschitz':
            return 1.
        elif mode == 'grad':
            return r

    def abs_prox(x, mode, prox_param=1.):
        if mode == 'prox':
            return np.sign(x) * np.maximum(0., np.abs(x) - prox_param)

    g_i = lambda x, mode, prox_param: abs_prox(x, mode, prox_param * alpha)

    start_time = time.time()
    cd_pure_python(half_squares, X, y, [g_i]*X.shape[1], max_iter=max_iter)
    time_cd_pure_python_leukemia_lasso = time.time() - start_time

    pb_leukemia_lasso = cd_solver.Problem(N=X.shape[1],
                                          f=["square"],
                                          blocks_f = [0, X.shape[0]],
                                          Af=X,
                                          bf=y,
                                          cf=[0.5],
                                          g=["abs"] * X.shape[1],
                                          cg=[alpha] * X.shape[1])

    #import pstats, cProfile
    #
    #import pyximport
    #pyximport.install()
    #
    #cProfile.runctx("cd_solver.coordinate_descent(pb_leukemia_lasso, max_iter=max_iter, verbose=0, per_pass=10)", globals(), locals(), "Profile.prof")
    #
    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats()

    start_time = time.time()
    cd_solver.coordinate_descent(pb_leukemia_lasso, max_iter=max_iter, verbose=0)
    time_cd_solver_leukemia_lasso = time.time() - start_time

    start_time = time.time()
    alphas, coefs, gaps = linear_model.lasso_path(X, y, alphas=[alpha/X.shape[0]],
                                         precompute=False, tol=1e-15,
                                         verbose=0, max_iter=max_iter,
                                         selection='random')
    time_sklearn_leukemia_lasso = time.time() - start_time

    print(time_cd_pure_python_leukemia_lasso,
              time_cd_solver_leukemia_lasso, time_sklearn_leukemia_lasso)
    print(np.linalg.norm(coefs.ravel() - pb_leukemia_lasso.sol) / np.linalg.norm(coefs.ravel()))


if 1:
    print("dual SVM on rcv1")
    data = io.loadmat('/data/ofercoq/datasets/Classification/rcv1_train.binary.mat')

    X = data['X'].astype(np.float)
    y = data['y'].astype(np.float).ravel()

    max_iter_svm = 10
    alpha = 0.1
    C = 1. / alpha

    def f_svm(r, mode):
        if mode == 'Lipschitz':
            return 1.
        elif mode == 'grad':
            return np.concatenate((r[0:-1], [[1]]))

    def proj_01(x, mode, prox_param=1.):
        if mode == 'prox':
            return np.maximum(0, np.minimum(1, x))

    g_i = lambda x, mode, prox_param: proj_01(x, mode, prox_param * alpha)
    
    start_time = time.time()
    cd_pure_python(f_svm, sp.vstack([X.T.multiply(y), -np.ones(X.shape[0])], format="csc"), np.zeros(X.shape[1]+1), [g_i]*X.shape[1], max_iter=max_iter_svm)
    time_cd_pure_python_rcv1_svm = time.time() - start_time
    
    pb_rcv1_svm = cd_solver.Problem(N=X.shape[0],
                                    f=["square"] * X.shape[1] + ["linear"],
                                    Af=sp.vstack([X.T.multiply(y), -np.ones(X.shape[0])], format="csc"),
                                    bf=np.zeros(X.shape[1] + 1),
                                    cf=[0.5/alpha] * X.shape[1] + [1],
                                    g=["box_zero_one"] * X.shape[0])
    start_time = time.time()
    cd_solver.coordinate_descent(pb_rcv1_svm, max_iter=max_iter_svm, verbose=0, print_style='smoothed_gap')
    time_cd_solver_rcv1_svm = time.time() - start_time

    start_time = time.time()
    clf = svm.LinearSVC(C=1./alpha, loss='hinge', penalty='l2', fit_intercept=False,
                            max_iter=max_iter_svm, tol=1e-15)
    clf.fit(X, y)
    time_liblinear_rcv1_svm = time.time() - start_time

    print(time_cd_pure_python_rcv1_svm, time_cd_solver_rcv1_svm, time_liblinear_rcv1_svm)
    print(np.linalg.norm(clf.coef_.ravel() - X.T.dot(y*pb_rcv1_svm.sol) * 0.5 / alpha)/np.linalg.norm(clf.coef_.ravel()))

    w1 = clf.coef_.ravel()
    val1 = np.sum(np.maximum(0, 1 - y * X.dot(w1))) + 0.5 * alpha * np.linalg.norm(w1)**2


    x2 = pb_rcv1_svm.sol
    w2 = X.T.dot(y*x2) / alpha
    val2 = np.sum(np.maximum(0, 1 - y * X.dot(w2))) + 0.5 * alpha * np.linalg.norm(w2)**2

    vald2 = 0.5 / alpha * np.linalg.norm(X.T.dot(y*x2))**2 - np.sum(x2)


