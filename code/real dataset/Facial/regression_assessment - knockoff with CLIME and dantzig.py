# Output: SVR_predicted_test_labels_for_{method}.npy, SVR model for {method}.npy
# RFR_predicted_test_labels_for_{method}.npy, RFR model for {method}.npy
# RD_predicted_test_labels_for_{method}.npy, RD model for {method}.npy
# SGD_predicted_test_labels_for_{method}.npy, SGD model for {method}.npy
# MLPR_predicted_test_labels_for_{method}.npy, MLPR model for {method}.npy


import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
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

def assessment(train_X, train_labels, test_X, test_labels, method_name, high_dim, knockoff_selection_ratio):
    if method_name == "knockoffCS":
        method = f"{method_name}-{high_dim}-{knockoff_selection_ratio}"
    else:
        method = f"{method_name}-{high_dim}"
    svr = _SVR(train_X, train_labels, test_X, test_labels, method)
    rfr = _RFR(train_X, train_labels, test_X, test_labels, method)
    ridge = _ridge(train_X, train_labels, test_X, test_labels, method)
    XGB = _XGB(train_X, train_labels, test_X, test_labels, method)
    mse_score = {
        'method': method,
        'SVR': svr,
        'RF': rfr,
        'Ridge': ridge,
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


#dantzig01_signal = np.load('knockoffCS_signal_Dantzig.npy',  allow_pickle=True)
CLIME01_signal = np.load('knockoffCS_signal_CLIME.npy',  allow_pickle=True)
labels = np.load("Original Dataset_labels.npy", allow_pickle=True)


total_size = CLIME01_signal.shape[0]
high_dim = CLIME01_signal.shape[1]
knockoff_selection_ratio = 0.01

train_size = 3500
all_indices = np.arange(total_size)
np.random.shuffle(all_indices)
train_indices = all_indices[:train_size]
test_indices = all_indices[train_size:]

#dantzig01_signal_train = dantzig01_signal[train_indices]
#dantzig01_signal_test = dantzig01_signal[test_indices]
CLIME01_signal_train = CLIME01_signal[train_indices]
CLIME01_signal_test = CLIME01_signal[test_indices]
labels_train = labels[train_indices]
labels_test = labels[test_indices]

#dantzig01_signal_mse = assessment(dantzig01_signal_train, labels_train, dantzig01_signal_test, labels_test, 'knockoff with dantzig01', high_dim, knockoff_selection_ratio)
CLIME01_signal_mse = assessment(CLIME01_signal_train, labels_train, CLIME01_signal_test, labels_test, 'knockoff with CLIME01', high_dim, knockoff_selection_ratio)

import sys
from datetime import datetime

original_stdout = sys.stdout 

with open("output-knockoff with CLIME and dantzig.txt", "w", encoding="utf-8") as file:
    sys.stdout = file 
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp,'\n')
    #print(dantzig01_signal_mse)
    print(CLIME01_signal_mse)
    sys.stdout = original_stdout 
