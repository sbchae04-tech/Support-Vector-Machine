import numpy as np

def Processing(X, Y, n_train, raw, col):
    train_X_set = X[:n_train, :]
    train_Y_set = 2 * Y[:n_train] - 1

    X_test = X[n_train:, :]
    y_test  = 2 * Y[n_train:] - 1

    X_train = np.array(train_X_set).reshape(raw, col, 2)
    y_train = np.array(train_Y_set).reshape(raw, col)

    return X_train, y_train, X_test, y_test