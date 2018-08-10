import numpy as np

import scipy.sparse as sp
import cd_solver

from sklearn.datasets.mldata import fetch_mldata


probs = [0, 1, 2, 3]

for prob in probs:
    if prob == 0:
        f = ["square", "square"]
        cf = [0.5]*2
        bf = [1,-8]
        A = np.array([[1,4],[4, 3]])

        g = ["abs", "abs"]


        pb_toy = cd_solver.Problem(N=2, f=f, Af=A, bf=bf, cf=cf, g=g)

        cd_solver.coordinate_descent(pb_toy, max_iter=10, verbose=0.00001)

    if prob >= 1:
        dataset = 'leukemia'
        data = fetch_mldata(dataset)
        X = data.data
        X = sp.csc_matrix(X)
        y = data.target

    if prob == 1:
        # Lasso
        print("Lasso on Leukemia")
        pb_leukemia_lasso = cd_solver.Problem(N=X.shape[1],
                                              f=["square"] * X.shape[0],
                                              Af=X,
                                              bf=y,
                                              cf=[0.5] * X.shape[0],
                                              g=["abs"] * X.shape[1],
                                              cg=[0.1*np.linalg.norm(X.T.dot(y), np.inf)] * X.shape[1])

        cd_solver.coordinate_descent(pb_leukemia_lasso, max_iter=100, verbose=0.5)

    if prob == 2:
        # Logistic regression
        print("logistic regression on Leukemia")
        pb_leukemia_logreg = cd_solver.Problem(N=X.shape[1],
                                               f=["log1pexp"] * X.shape[0],
                                               Af=X,
                                               bf=y,
                                               cf=[1] * X.shape[0],
                                               g=["square"] * X.shape[1],
                                               cg=[0.5*0.01*np.linalg.norm(X.T.dot(y), np.inf)] * X.shape[1])

        cd_solver.coordinate_descent(pb_leukemia_logreg, max_iter=300, verbose=0.5)

    if prob == 3:
        # SVM
        print("dual SVM on Leukemia")
        alpha = 1000
        pb_leukemia_svm = cd_solver.Problem(N=X.shape[0],
                                            f=["square"] * X.shape[1] + ["linear"] * X.shape[0],
                                            Af=sp.vstack([X.T.multiply(y), -sp.eye(X.shape[0])], format="csc"),
                                            bf=np.zeros(X.shape[1] + X.shape[0]),
                                            cf=[0.5/alpha] * X.shape[1] + [1] * X.shape[0],
                                            g=["box_zero_one"] * X.shape[0])

        cd_solver.coordinate_descent(pb_leukemia_svm, max_iter=100, verbose=0.5)
