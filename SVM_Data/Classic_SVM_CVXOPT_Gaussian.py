from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection
from sklearn.datasets import make_circles
import os
os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#TF_CPP_MIN_LOG_LEVEL: 텐서 플로우 로그
#0: 모든 로그 출력(default)
#1: INFO 로그 필터
#2: INFO, WARNING 로그 필터
#3: INFO, WARNING, ERROR 로그 필터

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from cvxopt import matrix, solvers
import plotly.graph_objects as go
from numba import njit

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc
)

#Solver 구하기

@njit
def Gaussian_kernel(n, m, gamma, X_1, X_2):
    K = np.exp(-1 * gamma * ((X_1[n, :] - X_2[m, :]) @ (X_1[n, :] - X_2[m, :])))

    return(K)

def Solver_SVM (P, q, G, h, A, b):
    sol = solvers.qp(
    matrix(P),
    matrix(q),
    matrix(G),
    matrix(h),
    matrix(A, tc = "d"),
    matrix(b, tc = "d")
    )

    return(sol)

#학습
def Solver_Parameter(N_train, X_train, y_train, gamma, C):
    I = np.identity(N_train)

    K_train_train = np.zeros((N_train, N_train))

    for n in range(N_train):
        for m in range(N_train):
            K_train_train[n, m] = Gaussian_kernel(n, m, gamma,  X_train, X_train)

    Y = np.diag(y_train)
    P = Y @ K_train_train @ Y
    q = q = -np.ones((N_train, 1))

    G = np.vstack([-I, I])
    h = np.hstack([np.zeros(N_train), C*np.ones(N_train)])

    A = y_train.astype(float).reshape(1, N_train)
    b = np.array([0.0])

    return P, q, G, h, A, b, K_train_train

@njit
def Gaussian_Parameter(i, gamma, z, X_train):
    # z: (2,), X_train: (N,2)
    diff = X_train[i, :] - z              # (2,)
    return np.exp(-gamma * (diff @ diff))

@njit
def Gaussian_HyperPlane(xx, yy, X_train, y_train, alpha, gamma, b):
    # 타입/shape 안정화 (매우 중요)
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    HP = np.where((alpha >= 0) & (alpha <= C))[0]
    alpha   = np.asarray(alpha,   dtype=np.float64).ravel()
    gamma   = float(gamma)
    b       = float(b)

    N = X_train.shape[0]
    Z = np.zeros(xx.shape, dtype=np.float64)

    for r in range(xx.shape[0]):
        for c in range(xx.shape[1]):
            z = np.array([xx[r, c], yy[r, c]], dtype=np.float64)

            s = 0.0
            for i in  HP:
                s += Gaussian_Parameter(i, gamma, z, X_train) * alpha[i] * y_train[i]

            Z[r, c] = s + b   # b는 한 번만 더함

    return Z

def b_value(alpha, C, y, K):
    alpha = np.asarray(alpha, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    K = np.asarray(K, dtype=float)

    HP = np.where((alpha >= 0) & (alpha <= C))[0]

    g = (alpha * y) @ K

    b = np.mean(y[HP] - g[HP])

    return b

def Train_Graph(X_train, y_train, alpha, K, gamma, C): 

# 2-D graph############################################################################################################
    h = 0.01
    x_min, x_max = X_train[:, 0].min()-1, X_train[:, 0].max()+1
    y_min, y_max = X_train[:, 1].min()-1, X_train[:, 1].max()+1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # b를 모르면 일단 0으로 두고 경계 모양을 먼저 확인 가능
    b = b_value(alpha, C, y_train, K)
    
    Z = Gaussian_HyperPlane(xx, yy, X_train, y_train, alpha, gamma, b=b)

    plt.contourf(xx, yy, Z,
                levels=[Z.min(), 0, Z.max()],
                colors=['#87CEEB', '#8B4513'],
                alpha=0.5)

    plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)  # 결정경계 강조

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)

    plt.title('Gaussian (RBF) SVM')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
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

def Test_Graph(X_train, X_test, y_train, y_test, alpha, K, gamma, C):

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

    scores_grid = (alpha * y_train) @ K_train_grid + b_value(alpha, C, y_train, K)
    Z = scores_grid.reshape(xx.shape)

    return Z

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

def Evaluate(y_test, alpha, K_train_train):
    acc, auroc, auprc = evaluate_test(
        y_test,
        Test_evlauation(alpha, K_train_train)
    )

    print("#" * 25)
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test AUROC    : {auroc:.4f}")
    print(f"Test AUPRC    : {auprc:.4f}")
    print("#" * 25)

    y_true = np.asarray(y_test).ravel()
    if set(np.unique(y_true)) == {-1, 1}:
        y_true = (y_true == 1).astype(int)

    scores = np.asarray(Test_evlauation(alpha, K_train_train)).ravel()

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

def Hinge_Loss(X_train, X_test, y_train, y_test, alpha, K_train_train, scores_train, gamma, C):
    scores_test = Test_evlauation(X_train, X_test, y_train, alpha, K_train_train, gamma, C)

    loss_train = np.maximum(0, 1 - (y_train * scores_train))
    loss_test = np.maximum(0, 1 - (y_test * scores_test))

    loss_train_mean = np.mean(loss_train)
    loss_test_mean  = np.mean(loss_test)

    return loss_train_mean, loss_test_mean
def Lagrangian(alpha, K_train_train, energy, y_train):

    lagrangian_w = ( 0.5 * ((alpha * y_train) @ K_train_train @ (alpha * y_train)) - sum(alpha) )

    lagrangian_b = -1 * sum(alpha * y_train * b_value(alpha, y_train, K_train_train))

    lagrangian_xi = energy - lagrangian_w - lagrangian_b

    return lagrangian_w, lagrangian_b, lagrangian_xi

def Primal(alpha, K_train_train, y_train, C):
    J_w =  0.5 * ((alpha * y_train) @ K_train_train @ (alpha * y_train))
    scores = (alpha * y_train) @ K_train_train + b_value(alpha, C, y_train, K_train_train)

    # slack
    xi = np.maximum(0, 1 - y_train * scores)

    # sum xi
    J_xi = np.sum(xi)

    return J_w, J_xi