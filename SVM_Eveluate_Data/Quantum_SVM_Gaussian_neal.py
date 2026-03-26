import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import model_selection
from sklearn.datasets import make_circles
import plotly.graph_objects as go
from numba import njit

import neal

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc
)

#Q metric 생성

@njit
def Gaussian_kernel(n, m, gamma, X_1, X_2):
    P = np.exp(-1 * gamma * ((X_1[n, :] - X_2[m, :]) @ (X_1[n, :] - X_2[m, :])))

    return(P)

def Q_metric(N_train, X_train, y_train, B, K, xi, gamma):
    """
    Returns
    -------
    K_train_train : (N, N) ndarray
        Gaussian kernel matrix
    Q_upper : (K*N, K*N) ndarray
        Upper-triangular QUBO matrix ready for neal
    """
    N_var = K * N_train

    # 1) kernel matrix
    K_train_train = np.zeros((N_train, N_train), dtype=float)
    for n in range(N_train):
        for m in range(N_train):
            K_train_train[n, m] = Gaussian_kernel(n, m, gamma, X_train, X_train)

    # 2) build symmetric Q_tilde from paper Eq. (13)
    Q_tilde = np.zeros((N_var, N_var), dtype=float)

    for n in range(N_train):
        for m in range(N_train):
            ymn = y_train[n] * y_train[m]
            kmn = K_train_train[n, m]

            for k in range(K):
                Bk = B ** k
                for j in range(K):
                    idx1 = K * n + k
                    idx2 = K * m + j

                    val = 0.5 * (B ** (k + j)) * ymn * (kmn + xi)

                    if n == m and k == j:
                        val -= Bk

                    Q_tilde[idx1, idx2] = val

    # 3) convert symmetric Q_tilde -> upper-triangular QUBO matrix Q
    Q_upper = np.zeros_like(Q_tilde)

    for i in range(N_var):
        Q_upper[i, i] = Q_tilde[i, i]
        for j in range(i + 1, N_var):
            Q_upper[i, j] = Q_tilde[i, j] + Q_tilde[j, i]

    return K_train_train, Q_upper

#Solver 구하기

def neal_Solver(Q_upper):
    sampler = neal.SimulatedAnnealingSampler()

    Q_dict = {}
    n = Q_upper.shape[0]

    for i in range(n):
        if Q_upper[i, i] != 0.0:
            Q_dict[(i, i)] = float(Q_upper[i, i])

        for j in range(i + 1, n):
            if Q_upper[i, j] != 0.0:
                Q_dict[(i, j)] = float(Q_upper[i, j])

    sampleset = sampler.sample_qubo(
        Q_dict,
        num_reads=10000,
        beta_range=(0.1, 10),
        num_sweeps=1000
    )

    top_solutions = sampleset.lowest(20)
    return top_solutions

#학습

def Solution(sol, top_k, n_th):
    solutions = []

    for i, rec in enumerate(sol.data(['sample', 'energy'])):
        if i == top_k:
            break
        solutions.append((rec.sample, rec.energy))

    N_sol = len(solutions)
    
    X = solutions[n_th][0]
    energy = solutions[n_th][1]

    keys = sorted(X.keys())
    x_opt = np.array([X[i] for i in keys], dtype=int)

    return x_opt, energy

def alpha_value(N_train, x_opt, B, K):

    X_matrix = x_opt.reshape(-1, K)
    alpha = np.zeros(N_train)


    for n in range(N_train):
        a = 0
        
        for k in range(K):
            a = a + (B**k) * X_matrix[n][k]

        alpha[n] = a

    return alpha


def Gaussian_Parameter(i, gamma, z, X_train):
    # z: (2,), X_train: (N,2)
    diff = X_train[i, :] - z
    return np.exp(-gamma * (diff @ diff))

@njit
def Gaussian_HyperPlane(xx, yy, X_train, y_train, alpha, gamma, b, C):
    # 타입/shape 안정화 (매우 중요)
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    alpha   = np.asarray(alpha,   dtype=np.float64).ravel()
    HP = np.where((alpha > 0) & (alpha < C))[0]
    gamma   = float(gamma)
    b       = float(b)

    N = X_train.shape[0]
    Z = np.zeros(xx.shape, dtype=np.float64)

    for r in range(xx.shape[0]):
        for c in range(xx.shape[1]):
            z = np.array([xx[r, c], yy[r, c]], dtype=np.float64)

            s = 0.0
            for i in HP:
                s += Gaussian_Parameter(i, gamma, z, X_train) * alpha[i] * y_train[i]

            Z[r, c] = s + b   # b는 한 번만 더함

    return Z

def b_value_eq7(alpha, C, y_train, K_train_train, eps=1e-8):
    alpha = np.asarray(alpha).ravel()
    y_train = np.asarray(y_train).ravel()
    K_train_train = np.asarray(K_train_train)

    sv = (alpha > eps) & (alpha < C - eps)

    if not np.any(sv):
        return 0.0

    idx = np.where(sv)[0]
    b_list = []

    for i in idx:
        b_i = y_train[i] - np.sum(alpha * y_train * K_train_train[:, i])
        b_list.append(b_i)

    return float(np.mean(b_list))


def b_value(alpha, C, y_train, K_train_train, n_grid=2001, margin=2.0):
    alpha = np.asarray(alpha).ravel()
    y_train = np.asarray(y_train).ravel()
    K_train_train = np.asarray(K_train_train)

    scores0 = (alpha * y_train) @ K_train_train
    b0 = b_value_eq7(alpha, C, y_train, K_train_train)

    smin = np.min(scores0)
    smax = np.max(scores0)
    span = max(smax - smin, 1.0)

    y_bin = (y_train == 1).astype(int)

    best_b = b0
    best_acc = -1.0

    for b in np.linspace(b0 - margin * span, b0 + margin * span, n_grid):
        pred = ((scores0 + b) >= 0).astype(int)
        acc = np.mean(pred == y_bin)

        if acc > best_acc:
            best_acc = acc
            best_b = b

    return float(best_b)
def Train_Graph(ax, X_train, y_train, alpha, K, gamma, C, K_train_train): 

# 2-D graph############################################################################################################
    h = 0.01
    x_min, x_max = X_train[:, 0].min()-1, X_train[:, 0].max()+1
    y_min, y_max = X_train[:, 1].min()-1, X_train[:, 1].max()+1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # b를 모르면 일단 0으로 두고 경계 모양을 먼저 확인 가능
    b = b_value(alpha, C, y_train, K_train_train)
    
    Z = Gaussian_HyperPlane(xx, yy, X_train, y_train, alpha, gamma, b, C)

    ax.contourf(xx, yy, Z,
                levels=[Z.min(), 0, Z.max()],
                colors=['#87CEEB', '#8B4513'],
                alpha=0.5)

    ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)  # 결정경계 강조

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)

    ax.set_title(f'Quantum Gaussian (gamma = {gamma}) SVM')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
#######################################################################################################################

# 3-D graph############################################################################################################
    # fig = go.Figure()

    # fig.add_trace(
    #     go.Surface(
    #         x=xx,
    #         y=yy,
    #         z=Z,
    #         colorscale='RdBu',
    #         opacity=0.85,
    #         colorbar=dict(title='f(x, y)')
    #     )
    # )


    # fig.update_layout(
    #     title='RBF (Gaussian) qSVM Decision Surface',
    #     scene=dict(
    #         xaxis_title='Sepal Length',
    #         yaxis_title='Sepal Width',
    #         zaxis_title='Decision value f(x,y)'
    # ))


    # fig.show()
#######################################################################################################################

#X_test Data

def Test_evlauation(X_train, X_test, y_train, alpha, K_train_train, gamma, C):

    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    K_train_test = np.zeros((N_train, N_test))

    for n in range(N_train):
        for m in range(N_test):
            K_train_test[n, m] = Gaussian_kernel(n, m, gamma, X_train, X_test)

    scores_test = (alpha * y_train) @ K_train_test + b_value(alpha, C, y_train, K_train_train)

    return scores_test

def Test_Graph(X_train, X_test, y_train, y_test, alpha, K_train_train, gamma, C):

# 2-D graph############################################################################################################
    h = 0.01

    x_min, x_max = X_test[:, 0].min()-1 , X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min()-1 , X_test[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]  # (N_grid, 2)

    # grid에 대한 decision score f(x) 계산: f(x) = Σ α_i y_i k(x_i, x) + b
    N_train = X_train.shape[0]
    N_grid = grid.shape[0]

    K_train_grid = np.zeros((N_train, N_grid))
    for i in range(N_train):
        for j in range(N_grid):
            K_train_grid[i, j] = Gaussian_kernel(i, j, gamma, X_train, grid)

    scores_grid = (alpha * y_train) @ K_train_grid + b_value(alpha, C, y_train, K_train_train)
    Z = scores_grid.reshape(xx.shape)

    return (Z)

    # plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], colors=['#87CEEB', '#8B4513'], alpha=0.5)
    # plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired)

    # plt.title('Gaussian SVM')
    # plt.xlabel('Sepal Length')
    # plt.ylabel('Sepal Width')
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.show()
#######################################################################################################################

# 3-D graph############################################################################################################
    # fig = go.Figure()

    # fig.add_trace(
    #     go.Surface(
    #         x=xx,
    #         y=yy,
    #         z=Z,
    #         colorscale='RdBu',
    #         opacity=0.85,
    #         colorbar=dict(title='f(x, y)')
    #     )
    # )


    # fig.update_layout(
    #     title='RBF (Gaussian) qSVM Decision Surface',
    #     scene=dict(
    #         xaxis_title='Sepal Length',
    #         yaxis_title='Sepal Width',
    #         zaxis_title='Decision value f(x,y)'
    # ))


    # fig.show()
#######################################################################################################################

#평가

def evaluate_train(y_true, alpha, K_train_train, C, threshold=0.0):
    y_true = np.asarray(y_true)
    scores_train = (alpha * y_true) @ K_train_train + b_value(alpha, C, y_true, K_train_train)
    scores_train = np.asarray(scores_train)

    if set(np.unique(y_true)) == {-1, 1}:
        y_true_bin = (y_true == 1).astype(int)
    else:
        y_true_bin = y_true.astype(int)

    # Accuracy (threshold 기반)
    y_pred = (scores_train >= threshold).astype(int)
    accuracy = accuracy_score(y_true_bin, y_pred)

    # AUROC / AUPRC (threshold-independent)
    auroc = roc_auc_score(y_true_bin, scores_train)
    auprc = average_precision_score(y_true_bin, scores_train)

    return accuracy, auroc, auprc, scores_train

def evaluate_test(y_true, decision_scores, threshold=0.0):
    y_true = np.asarray(y_true)
    decision_scores = np.asarray(decision_scores)

    # {-1, +1} → {0, 1}
    if set(np.unique(y_true)) == {-1, 1}:
        y_true_bin = (y_true == 1).astype(int)
    else:
        y_true_bin = y_true.astype(int)

    # Accuracy (threshold 기반)
    y_pred = (decision_scores >= threshold).astype(int)
    accuracy = accuracy_score(y_true_bin, y_pred)

    # AUROC / AUPRC (threshold-independent)
    auroc = roc_auc_score(y_true_bin, decision_scores)
    auprc = average_precision_score(y_true_bin, decision_scores)

    return accuracy, auroc, auprc

def Evaluate(X_train, X_test, y_train, y_test, alpha, K_train_train, gamma, C):
    acc, auroc, auprc = evaluate_test(
        y_test,
        Test_evlauation(X_train, X_test, y_train, alpha, K_train_train, gamma, C)
    )

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test AUROC    : {auroc:.4f}")
    print(f"Test AUPRC    : {auprc:.4f}")

    y_true = np.asarray(y_test).ravel()
    if set(np.unique(y_true)) == {-1, 1}:
        y_true = (y_true == 1).astype(int)

    scores = np.asarray(Test_evlauation(X_train, X_test, y_train, alpha, K_train_train, gamma, C)).ravel()

    # ROC 계산
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (SVM)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def Evaluate_Overfitting(acc_train, acc_test, auroc_train, auroc_test, auprc_train, auprc_test):

    gap_acc   = acc_train   - acc_test
    gap_auroc = auroc_train - auroc_test
    gap_auprc = auprc_train - auprc_test
    
    return gap_acc, gap_auroc, gap_auprc

# def Hinge_Loss(X_train, X_test, y_train, y_test, alpha, K_train_train, scores_train, gamma, C):
#     scores_test = Test_evlauation(X_train, X_test, y_train, alpha, K_train_train, gamma, C)

#     loss_train = np.maximum(0, 1 - (y_train * scores_train))
#     loss_test = np.maximum(0, 1 - (y_test * scores_test))

#     loss_train_mean = np.mean(loss_train)
#     loss_test_mean  = np.mean(loss_test)

#     return loss_train_mean, loss_test_mean

def Primal(alpha, K_train_train, y_train, C):
    J_w =  0.5 * ((alpha * y_train) @ K_train_train @ (alpha * y_train))
    scores = (alpha * y_train) @ K_train_train + b_value(alpha, C, y_train, K_train_train)

    # slack
    xi = np.maximum(0, 1 - y_train * scores)

    # sum xi
    J_xi = np.sum(xi)

    return J_w, J_xi