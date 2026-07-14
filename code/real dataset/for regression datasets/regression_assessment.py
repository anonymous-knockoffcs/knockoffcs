# Output: SVR_predicted_test_labels_for_{method}.npy, SVR model for {method}.npy
# RFR_predicted_test_labels_for_{method}.npy, RFR model for {method}.npy
# RD_predicted_test_labels_for_{method}.npy, RD model for {method}.npy
# SGD_predicted_test_labels_for_{method}.npy, SGD model for {method}.npy
# MLPR_predicted_test_labels_for_{method}.npy, MLPR model for {method}.npy


import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import xgboost

def _SVR(train_X, train_labels, test_X, test_labels, method):
    svr = SVR()
    svr.fit(train_X, train_labels)
    predicted_test_labels = svr.predict(test_X)
    mse = mean_squared_error(test_labels, predicted_test_labels)
    #print("SVR mse:", mse)
    #filename = f"SVR_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return mse

def _RFR(train_X, train_labels, test_X, test_labels, method):
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(train_X, train_labels)
    predicted_test_labels = rfr.predict(test_X)
    mse = mean_squared_error(test_labels, predicted_test_labels)
    #print("RFR mse:", mse)
    #filename = f"RFR_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return mse

def _ridge(train_X, train_labels, test_X, test_labels, method):
    rd = Ridge()
    rd.fit(train_X, train_labels)
    predicted_test_labels = rd.predict(test_X)
    mse = mean_squared_error(test_labels, predicted_test_labels)
    #print("RD mse:", mse)
    #filename = f"RD_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return mse

def _XGB(train_X, train_labels, test_X, test_labels, method):
    xgb = xgboost.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
    xgb.fit(train_X, train_labels)
    predicted_test_labels = xgb.predict(test_X)
    mse = mean_squared_error(test_labels, predicted_test_labels)
    #print("XGBoost mse:", mse)
    #filename = f"XGBoost_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return mse

def _linear(train_X, train_labels, test_X, test_labels, method):
    lr = LinearRegression()
    lr.fit(train_X, train_labels)
    predicted_test_labels = lr.predict(test_X)
    mse = mean_squared_error(test_labels, predicted_test_labels)
    #print("Linear mse:", mse)
    #filename = f"Linear_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return mse

def _bayesian_ridge(train_X, train_labels, test_X, test_labels, method):
    br = BayesianRidge()
    br.fit(train_X, train_labels)
    predicted_test_labels = br.predict(test_X)
    mse = mean_squared_error(test_labels, predicted_test_labels)
    #print("BayesianRidge mse:", mse)
    #filename = f"BayesianRidge_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return mse
def _knn(train_X, train_labels, test_X, test_labels, method):
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(train_X, train_labels)
    predicted_test_labels = knn.predict(test_X)
    mse = mean_squared_error(test_labels, predicted_test_labels)
    #print("KNN mse:", mse)
    #filename = f"KNN_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return mse

def _lasso(train_X, train_labels, test_X, test_labels, method):
    lasso = Lasso(alpha=1.0)
    lasso.fit(train_X, train_labels)
    predicted_test_labels = lasso.predict(test_X)
    mse = mean_squared_error(test_labels, predicted_test_labels)
    #print("Lasso mse:", mse)
    #filename = f"Lasso_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return mse

def assessment(train_X, train_labels, test_X, test_labels, method_name, high_dim, knockoff_selection_ratio):
    if method_name == "knockoffCS":
        method = f"{method_name}-{high_dim}-{knockoff_selection_ratio}"
    else:
        method = f"{method_name}-{high_dim}"
    svr = _SVR(train_X, train_labels, test_X, test_labels, method)
    rfr = _RFR(train_X, train_labels, test_X, test_labels, method)
    ridge = _ridge(train_X, train_labels, test_X, test_labels, method)
    linear = _linear(train_X, train_labels, test_X, test_labels, method)
    bayesian_ridge = _bayesian_ridge(train_X, train_labels, test_X, test_labels, method)
    knn = _knn(train_X, train_labels, test_X, test_labels, method)
    lasso = _lasso(train_X, train_labels, test_X, test_labels, method)
    XGB = _XGB(train_X, train_labels, test_X, test_labels, method)
    mse_score = {
        'method': method,
        'SVR': svr,
        'RF': rfr,
        'Ridge': ridge,
        'Linear': linear,
        'BayesianRidge': bayesian_ridge,
        'KNN': knn,
        'Lasso': lasso,
        'XGB': XGB
    }
    return mse_score


def XGBoostModel(train_X, train_labels, test_X, test_labels, method):
    xgb = xgboost.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
    xgb.fit(train_X, train_labels)
    predicted_test_labels = xgb.predict(test_X)
    mse = mean_squared_error(test_labels, predicted_test_labels)
    print(f"{method} XGBoost mse:", mse)
    filename = f"XGBoost_predicted_test_labels_for_{method}.npy"
    np.save(filename, test_labels, allow_pickle=True)


lasso10_signal = np.load('lasso10_signal.npy',  allow_pickle=True)
lasso1_signal = np.load('lasso1_signal.npy',  allow_pickle=True)
lasso01_signal = np.load('lasso01_signal.npy',  allow_pickle=True)
lasso001_signal = np.load('lasso001_signal.npy',  allow_pickle=True)
lasso0001_signal = np.load('lasso0001_signal.npy',  allow_pickle=True)
OMP_signal = np.load('OMP_signal.npy', allow_pickle=True)
knockoffCS_signal = np.load('knockoffCS_signal.npy', allow_pickle=True)
labels = np.load("Original Dataset_labels.npy", allow_pickle=True)

total_size = lasso10_signal.shape[0]
high_dim = lasso10_signal.shape[1]
knockoff_selection_ratio = 0.01
print(total_size)
train_size = 3500 
all_indices = np.arange(total_size)
np.random.shuffle(all_indices)
train_indices = all_indices[:train_size]
test_indices = all_indices[train_size:]

lasso10_signal_train = lasso10_signal[train_indices]
lasso10_signal_test = lasso10_signal[test_indices]
lasso1_signal_train = lasso1_signal[train_indices]
lasso1_signal_test = lasso1_signal[test_indices]
lasso01_signal_train = lasso01_signal[train_indices]
lasso01_signal_test = lasso01_signal[test_indices]
lasso001_signal_train = lasso001_signal[train_indices]
lasso001_signal_test = lasso001_signal[test_indices]
lasso0001_signal_train = lasso0001_signal[train_indices]
lasso0001_signal_test = lasso0001_signal[test_indices]
OMP_signal_train = OMP_signal[train_indices]
OMP_signal_test = OMP_signal[test_indices]
knockoffCS_signal_train = knockoffCS_signal[train_indices]
knockoffCS_signal_test = knockoffCS_signal[test_indices]
labels_train = labels[train_indices]
labels_test = labels[test_indices]

lasso10_mse = assessment(lasso10_signal_train, labels_train, lasso10_signal_test, labels_test, 'lasso10', high_dim, knockoff_selection_ratio)
lasso1_mse = assessment(lasso1_signal_train, labels_train, lasso1_signal_test, labels_test, 'lasso1', high_dim, knockoff_selection_ratio)
lasso01_mse = assessment(lasso01_signal_train, labels_train, lasso01_signal_test, labels_test, 'lasso01', high_dim, knockoff_selection_ratio)
lasso001_mse = assessment(lasso001_signal_train, labels_train, lasso001_signal_test, labels_test, 'lasso001', high_dim, knockoff_selection_ratio)
lasso0001_mse = assessment(lasso0001_signal_train, labels_train, lasso0001_signal_test, labels_test, 'lasso0001', high_dim, knockoff_selection_ratio)
OMP_mse = assessment(OMP_signal_train, labels_train, OMP_signal_test, labels_test, 'OMP', high_dim, knockoff_selection_ratio)
knockoffCS_mse = assessment(knockoffCS_signal_train, labels_train, knockoffCS_signal_test, labels_test, 'knockoffCS', high_dim, knockoff_selection_ratio)

import sys
from datetime import datetime

original_stdout = sys.stdout

with open("output-add more models with knn and lasso.txt", "w", encoding="utf-8") as file:
    sys.stdout = file  
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp,'\n')
    print(lasso10_mse)
    print(lasso1_mse)
    print(lasso01_mse)
    print(lasso001_mse)
    print(lasso0001_mse)
    print(OMP_mse)
    print(knockoffCS_mse)
    sys.stdout = original_stdout
