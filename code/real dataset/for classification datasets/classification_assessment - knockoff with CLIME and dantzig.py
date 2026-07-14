# Output: SVC_predicted_test_labels_for_{method}.npy, SVC model for {method}.npy
# RF_predicted_test_labels_for_{method}.npy, RF model for {method}.npy
# log_predicted_test_labels_for_{method}.npy, log model for {method}.npy
# mlpc_predicted_test_labels_for_{method}.npy, mlpc model for {method}.npy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost

def _SVC(train_X, train_labels, test_X, test_labels, method):
    svc = SVC(decision_function_shape='ovr')
    svc.fit(train_X, train_labels)
    predicted_test_labels = svc.predict(test_X)
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    #print("SVC accuracy:", accuracy)
    #filename = f"SVC_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return accuracy

def _RF(train_X, train_labels, test_X, test_labels, method):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_X, train_labels)
    predicted_test_labels = rf.predict(test_X)
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    #print("RF accuracy:", accuracy)
    #filename = f"RF_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return accuracy

def _logisticRegresion(train_X, train_labels, test_X, test_labels, method):
    log = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
    log.fit(train_X, train_labels)
    predicted_test_labels = log.predict(test_X)
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    #print("log accuracy:", accuracy)
    #filename = f"log_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return accuracy

def _MLPC(train_X, train_labels, test_X, test_labels, method):
    mlpc = MLPClassifier(random_state=1, max_iter=300)
    mlpc.fit(train_X, train_labels)
    predicted_test_labels = mlpc.predict(test_X)
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    #print("mlpc accuracy:", accuracy)
    #filename = f"mlpc_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return accuracy

def _xgb(train_X, train_labels, test_X, test_labels, method):
    xgb = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(train_X, train_labels)
    predicted_test_labels = xgb.predict(test_X)
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    #print("xgb accuracy:", accuracy)
    #filename = f"xgb_predicted_test_labels_for_{method}.npy"
    #np.save(filename, test_labels, allow_pickle=True)
    return accuracy

def assessment(train_X, train_labels, test_X, test_labels, method_name, high_dim, knockoff_selection_ratio):
    if method_name == "knockoffCS":
        method = f"{method_name}-{high_dim}-{knockoff_selection_ratio}"
    else:
        method = f"{method_name}-{high_dim}"
    SVC = _SVC(train_X, train_labels, test_X, test_labels, method)
    RF = _RF(train_X, train_labels, test_X, test_labels, method)
    log = _logisticRegresion(train_X, train_labels, test_X, test_labels, method)
    MLPC = _MLPC(train_X, train_labels, test_X, test_labels, method)
    XGB = _xgb(train_X, train_labels, test_X, test_labels, method)
    accuracy_score = {
        'method': method,
        'SVC': SVC,
        'RF': RF,
        'log': log,
        'MLPC': MLPC,
        'XGB' : XGB,
    }
    return accuracy_score


def xgb_model(train_X, train_labels, test_X, test_labels, method):
    xgb = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(train_X, train_labels)
    predicted_test_labels = xgb.predict(test_X)
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    print(f"{method} xgb accuracy:", accuracy)
    filename = f"xgb_predicted_test_labels_for_{method}.npy"
    np.save(filename, test_labels, allow_pickle=True)



dantzig01_signal = np.load('knockoffCS_signal_Dantzig.npy',  allow_pickle=True)
CLIME01_signal = np.load('knockoffCS_signal_CLIME.npy',  allow_pickle=True)
labels = np.load("Original Dataset_labels.npy", allow_pickle=True)

total_size = CLIME01_signal.shape[0]
high_dim = CLIME01_signal.shape[1]
knockoff_selection_ratio = 0.01

train_size = 2500 
all_indices = np.arange(total_size)
np.random.shuffle(all_indices)
train_indices = all_indices[:train_size]
test_indices = all_indices[train_size:]

dantzig01_signal_train = dantzig01_signal[train_indices]
dantzig01_signal_test = dantzig01_signal[test_indices]
CLIME01_signal_train = CLIME01_signal[train_indices]
CLIME01_signal_test = CLIME01_signal[test_indices]
labels_train = labels[train_indices]
labels_test = labels[test_indices]

dantzig01_signal_mse = assessment(dantzig01_signal_train, labels_train, dantzig01_signal_test, labels_test, 'knockoff with dantzig01', high_dim, knockoff_selection_ratio)
CLIME01_signal_mse = assessment(CLIME01_signal_train, labels_train, CLIME01_signal_test, labels_test, 'knockoff with CLIME01', high_dim, knockoff_selection_ratio)

import sys
from datetime import datetime

original_stdout = sys.stdout  

with open("output-knockoff with CLIME and dantzig.txt", "w", encoding="utf-8") as file:
    sys.stdout = file  
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp,'\n')
    print(dantzig01_signal_mse)
    print(CLIME01_signal_mse)
    sys.stdout = original_stdout


