# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>

import numpy as np
import copy

import scipy.sparse as sp
import cd_solver

# imports for loading datasets
from libsvmdata import fetch_libsvm
from sklearn.datasets import fetch_openml

if 0:
    # Check gradients
    print('Testing gradients:')
    test = check_grad('square', [1], nb_coord=1)
    print('square', test[0])
    test = check_grad('linear', [1], nb_coord=1)
    print('linear', test[0])
    test = check_grad('log1pexp', [1], nb_coord=1)
    print('log1pexp', test[0])
    test = check_grad('logsumexp', [1, -2, 3], nb_coord=3)
    print('logsumexp', test[0])


if 1:
    # test theta_purecd
    from cd_solver.algorithms_purecd import dual_vars_to_update_pure_cd
    from cd_solver.algorithms import find_dual_variables_to_update
    Ah = np.array([[-1, 1], [2, 0]])
    Ah = sp.csc_matrix(Ah)
    m, n = Ah.shape
    pb = cd_solver.Problem(N=Ah.shape[1], h=['zero']*2, Ah=Ah)
    dual_vars_to_update_ = find_dual_variables_to_update(np.uint32(n),
                                                         np.arange(
                                                             n+1, dtype=np.uint32),
                                                         np.arange(
                                                             m+1, dtype=np.uint32),
                                                         np.array(
                                                             Ah.indptr, dtype=np.uint32),
                                                         np.array(
                                                             Ah.indices, dtype=np.uint32),
                                                         np.arange(m, dtype=np.uint32), pb)
    dual_vars_to_update2 = dual_vars_to_update_pure_cd(np.uint32(n),
                                                       np.arange(
                                                           n+1, dtype=np.uint32),
                                                       np.arange(
                                                           m+1, dtype=np.uint32),
                                                       np.array(
                                                           Ah.indptr, dtype=np.uint32),
                                                       np.array(
                                                           Ah.indices, dtype=np.uint32),
                                                       np.arange(m, dtype=np.uint32), pb)
    print(dual_vars_to_update_, dual_vars_to_update2)


def smoothed_gap(pb, x, y):
    pb_ = copy.copy(pb)
    pb_.x_init = x
    pb_.y_init = y
    cd_solver.cd_solver.coordinate_descent(pb_, max_iter=1, algorithm=None,
                                           verbose=0.1, print_style='smoothed_gap')
    if pb_.performance_stats['Smoothed Gap'][2] > 1e-10:
        while pb_.performance_stats['Smoothed Gap'][2] < pb_.performance_stats['Smoothed Gap'][0]:
            gamma_print = pb_.performance_stats['Smoothed Gap'][2] * 2.
            cd_solver.cd_solver.coordinate_descent(pb_, max_iter=1, algorithm=None,
                                                   verbose=0., print_style='smoothed_gap',
                                                   gamma_print_=gamma_print)
    return pb_.performance_stats['Smoothed Gap']


probs = [0, 6, 7, 8]

for prob in probs:
    if prob == 0:
        f = ["square", "square"]
        cf = [0.5]*2
        bf = [1, -8]
        A = np.array([[1, 4], [4, 3]])

        g = ["abs", "abs"]

        pb_toy = cd_solver.Problem(N=2, f=f, Af=A, bf=bf, cf=cf, g=g)

        for algorithm in ['cd', 'approx', 'pure-cd', 's-pdhg']:
            cd_solver.coordinate_descent(
                pb_toy, max_iter=100, verbose=0.5, print_style='smoothed_gap', min_change_in_x=0., algorithm=algorithm)

    if (prob >= 1 and prob <= 4) or prob == 6 or prob == 11 or prob == 13:
        dataset = 'leukemia'
        X, y = fetch_openml(dataset, return_X_y=True)
        X = sp.csc_matrix(X)
        y = np.array([1 if label == "AML" else -1 for label in y])

    if prob == 1:
        # Lasso
        pb_leukemia_lasso = cd_solver.Problem(N=X.shape[1],
                                              f=["square"] * X.shape[0],
                                              Af=X,
                                              bf=y,
                                              cf=[0.5] * X.shape[0],
                                              g=["abs"] * X.shape[1],
                                              cg=[0.1*np.linalg.norm(X.T.dot(y), np.inf)] * X.shape[1])

        pb_leukemia_lasso_acc = copy.copy(pb_leukemia_lasso)
        pb_leukemia_lasso_screen = copy.copy(pb_leukemia_lasso)

        print("Lasso on Leukemia")
        cd_solver.coordinate_descent(pb_leukemia_lasso, max_iter=1000,
                                     verbose=1., print_style='smoothed_gap', tolerance=1e-10)

        print("Lasso on Leukemia with screening")
        cd_solver.coordinate_descent(pb_leukemia_lasso_screen, max_iter=1000,
                                     verbose=1., print_style='smoothed_gap', tolerance=1e-10,
                                     screening='gapsafe')

        print("Lasso on Leukemia with screening, momentum and variable restart")
        cd_solver.coordinate_descent(pb_leukemia_lasso_acc, max_iter=1000,
                                     verbose=1., print_style='smoothed_gap', tolerance=1e-10,
                                     algorithm='smart-cd', restart_period=4, screening='gapsafe')

    if prob == 2:
        # Logistic regression
        print("logistic regression on Leukemia")
        pb_leukemia_logreg = cd_solver.Problem(N=X.shape[1],
                                               f=["log1pexp"] * X.shape[0],
                                               Af=(X.T.multiply(y)).T,
                                               bf=0*y,
                                               cf=[1] * X.shape[0],
                                               g=["square"] * X.shape[1],
                                               cg=[0.5*0.01*np.linalg.norm(X.T.dot(y), np.inf)] * X.shape[1])

        cd_solver.coordinate_descent(
            pb_leukemia_logreg, max_iter=150, verbose=2., print_style='smoothed_gap')

    if prob == 3:
        # SVM
        print("dual SVM on Leukemia")
        alpha = 1000
        pb_leukemia_svm = cd_solver.Problem(N=X.shape[0],
                                            f=["square"] * X.shape[1] +
                                            ["linear"] * X.shape[0],
                                            Af=sp.vstack(
                                                [X.T.multiply(y), -sp.eye(X.shape[0])], format="csc"),
                                            bf=np.zeros(
                                                X.shape[1] + X.shape[0]),
                                            cf=[0.5/alpha] * X.shape[1] +
                                            [1] * X.shape[0],
                                            g=["box_zero_one"] * X.shape[0])

        pb_leukemia_svm_screen = copy.copy(pb_leukemia_svm)

        cd_solver.coordinate_descent(
            pb_leukemia_svm, max_iter=1000, verbose=0.5, print_style='smoothed_gap', tolerance=1e-14)
        print("dual SVM on Leukemia with gap safe screening")

        cd_solver.coordinate_descent(pb_leukemia_svm_screen, max_iter=1000, verbose=0.5,
                                     print_style='smoothed_gap', screening='gapsafe', tolerance=1e-14)

    if prob == 4:
        # Lasso by ISTA
        print("Lasso on Leukemia by ISTA")
        pb_leukemia_lasso = cd_solver.Problem(N=X.shape[1],
                                              f=["square"] * X.shape[0],
                                              Af=X,
                                              bf=y,
                                              cf=[0.5] * X.shape[0],
                                              g=["abs"],
                                              blocks=[0, X.shape[1]],
                                              cg=[0.1*np.linalg.norm(X.T.dot(y), np.inf)])

        cd_solver.coordinate_descent(
            pb_leukemia_lasso, max_iter=100, verbose=0.5, print_style='smoothed_gap')

    if prob == 5:
        # basic cd_solver.Problem with constraints
        print('basic cd_solver.Problem with constraints')
        f = ["square", "square"]
        cf = [0.5, 0.5]
        bf = [1, -0.5]
        Af = np.eye(2)

        h = ["eq_const", "eq_const"]
        Ah = np.array([[-1, 1], [2, 0]])

        pb_toy_const = cd_solver.Problem(
            N=2, f=f, Af=Af, bf=bf, cf=cf, h=h, Ah=Ah)

        pb_toy_const_smartcd = copy.copy(pb_toy_const)

        cd_solver.coordinate_descent(
            pb_toy_const, max_iter=1000, verbose=0.001, print_style='smoothed_gap')

        print('basic cd_solver.Problem with constraints by SMART-CD')
        cd_solver.coordinate_descent(pb_toy_const_smartcd, max_iter=1000, verbose=0.01,
                                     print_style='smoothed_gap', algorithm='smart-cd', restart_period=10)

    if prob == 6:
        # SVM with intercept

        print("dual SVM with intercept on Leukemia")
        alpha = 1000
        Xred = np.linalg.cholesky(np.array((X.dot(X.T)).todense()))
        pb_leukemia_svm_intercept = cd_solver.Problem(N=X.shape[0],
                                                      f=["square"] *
                                                      Xred.shape[1] +
                                                      ["linear"],
                                                      Af=sp.vstack(
                                                          [Xred.T * y, -np.ones((1, X.shape[0]))], format="csc"),
                                                      bf=np.zeros(
                                                          Xred.shape[1]+1),
                                                      cf=[0.5/alpha] *
                                                      Xred.shape[1] + [1],
                                                      g=["box_zero_one"] *
                                                      X.shape[0],
                                                      h=["eq_const"],
                                                      Ah=sp.csc_matrix(y)
                                                      )

        cd_solver.coordinate_descent(
            pb_leukemia_svm_intercept, max_iter=10000, verbose=0.5, print_style='smoothed_gap')

        print("dual SVM with intercept on Leukemia, no Cholesky")
        Q = 1. / alpha * (X.T.multiply(y)).T.dot(X.T.multiply(y))
        pb_leukemia_svm_intercept_nochol = cd_solver.Problem(N=X.shape[0],
                                                             Q=Q,
                                                             f=["linear"],
                                                             Af=-
                                                             np.ones(
                                                                 (1, X.shape[0])),
                                                             bf=np.zeros(1),
                                                             cf=[1],
                                                             g=["box_zero_one"] *
                                                             X.shape[0],
                                                             h=["eq_const"],
                                                             Ah=sp.csc_matrix(
                                                                 y)
                                                             )

        pb_leukemia_svm_intercept_nochol_smart_cd = copy.copy(
            pb_leukemia_svm_intercept_nochol)

        cd_solver.coordinate_descent(
            pb_leukemia_svm_intercept_nochol, max_iter=10000, verbose=0.5, print_style='smoothed_gap')

        cd_solver.coordinate_descent(pb_leukemia_svm_intercept_nochol, max_iter=10000,
                                     verbose=0.5, print_style='smoothed_gap', algorithm='smart-cd', restart_period=10)

    if prob == 7:
        print("dual SVM with intercept on RCV1")

        # data = svmlight_format.load_svmlight_file('/home/ofercoq/scikit_learn_data/mldata/rcv1_train.binary')
        X, y = fetch_libsvm("rcv1.binary")
        C = 1. / X.shape[0]
        alpha = 0.25 / X.shape[0]

        pb_rcv1_svm_intercept = cd_solver.Problem(N=X.shape[0],
                                                  f=["square"] *
                                                  X.shape[1] + ["linear"],
                                                  Af=sp.vstack(
                                                      [X.T.multiply(y), -np.ones((1, X.shape[0]))], format="csc"),
                                                  bf=np.zeros(X.shape[1] + 1),
                                                  cf=[C] * X.shape[1] + [1],
                                                  g=["box_zero_one"] *
                                                  X.shape[0],
                                                  Dg=alpha*sp.eye(X.shape[0]),
                                                  h=["eq_const"],
                                                  Ah=sp.csc_matrix(y)
                                                  )
        for algorithm in ['pure-cd', 'vu-condat-cd', 'smart-cd']:
            print(algorithm + ' without restart')
            cd_solver.coordinate_descent(pb_rcv1_svm_intercept, max_iter=1000, verbose=20., print_style='smoothed_gap', step_size_factor=alpha,
                                         # sampling='kink_half',
                                         algorithm=algorithm, average=1,
                                         restart_period=0)
        for algorithm in ['pure-cd', 'smart-cd']:
            print(algorithm + ' with restart')
            cd_solver.coordinate_descent(pb_rcv1_svm_intercept, max_iter=1000, verbose=20., print_style='smoothed_gap', step_size_factor=alpha,
                                         # sampling='kink_half',
                                         algorithm=algorithm, average=1,
                                         restart_period=200)

    if prob == 8:
        print("TV regularized least squares on toy dataset")

        X = np.array([[1, 2, 3, 4, 5, 6, 7], [-7, -6, -5, -4, -3, -2, -1]])
        y = [0, 2]

        alpha = 1*1e-2

        mask = np.array([[[True, True], [True, True]],
                         [[False, True], [True, True]]])
        integer_mask = np.cumsum(mask).reshape(mask.shape) * mask
        ravelling_array = np.cumsum(mask == mask).reshape(mask.shape) - 1
        correspondance = ravelling_array[mask]

        N = np.prod(mask.shape)

        X = sp.csr_matrix(X)
        Af = sp.csr_matrix(
            (X.data, correspondance[X.indices], X.indptr), (X.shape[0], N))

        Dx = sp.diags([-np.ones(mask.shape[0]),
                       np.ones(mask.shape[0])], offsets=[0, 1])
        Dy = sp.diags([-np.ones(mask.shape[1]),
                       np.ones(mask.shape[1])], offsets=[0, 1])
        Dz = sp.diags([-np.ones(mask.shape[2]),
                       np.ones(mask.shape[2])], offsets=[0, 1])

        Dx = sp.kron(Dx, sp.eye(mask.shape[1]*mask.shape[2]))
        Dy = sp.kron(sp.eye(mask.shape[0]), sp.kron(Dy, sp.eye(mask.shape[2])))
        Dz = sp.kron(sp.eye(mask.shape[0]*mask.shape[1]), Dz)

        threeDgradient = sp.vstack([Dx, Dy, Dz], format='csc')
        threeDgradient.eliminate_zeros()
        # reorder the matrix
        threeDgradient = sp.csc_matrix((threeDgradient.data, 3 * (threeDgradient.indices %
                                                                  N) + threeDgradient.indices // N, threeDgradient.indptr), (3*N, N))

        pb_toy_tvl1 = cd_solver.Problem(N=N,
                                        f=["square"] * X.shape[0],
                                        Af=Af,
                                        bf=y,
                                        cf=[0.5] * X.shape[0],
                                        h=["norm2"] * N,
                                        ch=[1.] * N,
                                        blocks=np.array([0, N]),
                                        blocks_h=np.arange(0, 3*N + 1, 3),
                                        Ah=alpha*threeDgradient
                                        )

        # ['vu-condat-cd', 'smart-cd', 'pure-cd', 's-pdhg']:
        for algorithm in ['pure-cd']:
            print("TV regularized least squares on toy dataset by %s" % algorithm)
            cd_solver.coordinate_descent(pb_toy_tvl1, max_iter=200000,
                                         verbose=0.1, max_time=5, print_style='smoothed_gap',
                                         tolerance=1e-14, algorithm=algorithm,
                                         average=1, restart_period=100, fixed_restart_period=False,
                                         check_period=1000, min_change_in_x=0)

    if prob == 9:
        try:
            print("l1+TV regularized least squares on fmri dataset")

            mem = Memory(cachedir='cache', verbose=3)
            X, y, subjects, mask, affine = mem.cache(
                load_gain_poldrack)(smooth=0, folder='../tv_l1_solver')

            l1_ratio = 0.5
            alpha = 1e-2

            integer_mask = np.cumsum(mask).reshape(mask.shape) * mask
            ravelling_array = np.cumsum(mask == mask).reshape(mask.shape) - 1
            correspondance = ravelling_array[mask]

            N = np.prod(mask.shape)

            X = sp.csr_matrix(X)
            Af = sp.csr_matrix(
                (X.data, correspondance[X.indices], X.indptr), (X.shape[0], N))

            Dx = sp.diags([-np.ones(mask.shape[0]),
                           np.ones(mask.shape[0])], offsets=[0, 1])
            Dy = sp.diags([-np.ones(mask.shape[1]),
                           np.ones(mask.shape[1])], offsets=[0, 1])
            Dz = sp.diags([-np.ones(mask.shape[2]),
                           np.ones(mask.shape[2])], offsets=[0, 1])

            Dx = sp.kron(Dx, sp.eye(mask.shape[1]*mask.shape[2]))
            Dy = sp.kron(sp.eye(mask.shape[0]),
                         sp.kron(Dy, sp.eye(mask.shape[2])))
            Dz = sp.kron(sp.eye(mask.shape[0]*mask.shape[1]), Dz)

            threeDgradient = sp.vstack([Dx, Dy, Dz], format='csc')
            threeDgradient.eliminate_zeros()
            # reorder the matrix
            threeDgradient = sp.csc_matrix((threeDgradient.data, 3 * (
                threeDgradient.indices % N) + threeDgradient.indices // N, threeDgradient.indptr), (3*N, N))

            pb_fmri_tvl1 = cd_solver.Problem(N=N,
                                             f=["square"] * X.shape[0],
                                             Af=Af,
                                             bf=y,
                                             cf=[0.5] * X.shape[0],
                                             g=["abs"] * N,
                                             cg=[alpha*l1_ratio] * N,
                                             h=["norm2"] * N,
                                             ch=[(1-l1_ratio)] * N,
                                             blocks_h=np.arange(0, 3*N + 1, 3),
                                             Ah=alpha*threeDgradient
                                             )

            cd_solver.coordinate_descent(pb_fmri_tvl1, max_iter=100, verbose=20.,
                                         max_time=100., step_size_factor=10., print_style='smoothed_gap')
        except:
            print('fMRI dataset not loaded')
    if prob == 10:
        # LP  --  min c.dot(x) : x >= 0, Mx <= b
        print('basic LP')
        d = 3
        n = 4
        M = np.array([[2, 4, 5, 7], [1, 1, 2, 2], [1, 2, 3, 3]])
        c = -np.array([7, 9, 18, 17])
        b = np.array([41, 17, 24])

        pb_basic_lp = cd_solver.Problem(N=n,
                                        f=["linear"],
                                        Af=c,
                                        g=["ineq_const"]*n,
                                        h=["ineq_const"]*d,
                                        Ah=-M,
                                        bh=-b
                                        )

        cd_solver.coordinate_descent(
            pb_basic_lp, max_iter=1000000, verbose=1., max_time=10., print_style='smoothed_gap')

    if prob == 11:
        # Lasso
        print("Lasso on Leukemia with kink detection")
        pb_leukemia_lasso = cd_solver.Problem(N=X.shape[1],
                                              f=["square"] * X.shape[0],
                                              Af=X,
                                              bf=y,
                                              cf=[0.5] * X.shape[0],
                                              g=["abs"] * X.shape[1],
                                              cg=[0.1*np.linalg.norm(X.T.dot(y), np.inf)] * X.shape[1])

        cd_solver.coordinate_descent(pb_leukemia_lasso, max_iter=100,
                                     verbose=5, print_style='smoothed_gap',
                                     sampling='kink_half')

    if prob == 12:
        # multinomial logistic regression
        print("Multinomial logistic regression on iris")
        dataset = 'iris'
        data = fetch_mldata(dataset)
        X = data.data
        X = sp.csc_matrix(X)
        y = data.target
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # reorganise data, row blocks, one block per observation,
        #   containing all classes and column blocks,
        #   one block per feature, containing all classes.
        n_class = 3
        Y = np.array([y == 1, y == 2, y == 3])
        tmp = X.T.dot(1. / n_class - Y.T)
        lambda_max = np.sqrt(np.max(np.sum(tmp*tmp, axis=1)))
        Y = Y.T.ravel()  # .reshape((1, np.prod(Y.shape)))
        XX = sp.kron(X, sp.eye(n_class))

        N = XX.shape[1]

        blocks = np.arange(0, XX.shape[1]+1, n_class)
        f = ["logsumexp"] * int(XX.shape[0] / n_class) + ["linear"]
        Af = sp.vstack([XX, XX.T.dot(Y)], format='csc')
        blocks_f = np.arange(0, XX.shape[0]+1, n_class)
        blocks_f = np.concatenate((blocks_f, [blocks_f[-1]+1]))

        pb_iris_multinomial = cd_solver.Problem(N=N,
                                                f=f,
                                                Af=Af,
                                                bf=np.zeros(Af.shape[0]),
                                                cf=[1.]*len(f),
                                                blocks_f=blocks_f,
                                                g=["norm2"]*n_features,
                                                blocks=blocks,
                                                cg=[0.5*lambda_max] * n_features)

        cd_solver.coordinate_descent(pb_iris_multinomial, max_iter=5000,
                                     verbose=1, print_style='smoothed_gap',
                                     sampling='kink_half', min_change_in_x=0)

    if prob == 13:
        # Sparse logistic regression
        print("sparse logistic regression on Leukemia")
        pb_leukemia_sparse_logreg = cd_solver.Problem(N=X.shape[1],
                                                      f=["log1pexp"] *
                                                      X.shape[0],
                                                      Af=(X.T.multiply(y)).T,
                                                      bf=np.zeros(X.shape[0]),
                                                      cf=[1] * X.shape[0],
                                                      g=["abs"] * X.shape[1],
                                                      cg=[0.5*np.linalg.norm(X.T.dot(0.5 - (1+y)/2.), np.inf)] * X.shape[1])

        cd_solver.coordinate_descent(pb_leukemia_sparse_logreg, max_iter=150, verbose=2.,
                                     print_style='gap', screening='gapsafe', min_change_in_x=0, tolerance=1e-12)
