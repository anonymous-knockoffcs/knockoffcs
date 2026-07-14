import numpy as np
import cvxpy as cp
from tqdm import tqdm
import pickle

knockoff_selection_ratio = 0.01
high_dim = 500

variables_with_error= np.load("transformed_variables.npy", allow_pickle=True)

# Read measurement matrix from file
with open("Another measurement matrix.pkl", "rb") as f:
    A = pickle.load(f)


def clime_compressive_sensing(A, y, lambda_):
    """
    Compressed sensing support set recovery using CLIME-style optimization
    Parameters:
        X: measurement matrix (n x p)
        y: observation vector (n,)
        lambda_: regularization parameter
    Returns:
        beta_hat: estimated sparse signal
        support: estimated support set
    """
    p = A.shape[1]

    # define variables
    beta = cp.Variable(p)

    # goal: min ||beta||_1
    objective = cp.Minimize(cp.norm(beta, 1))

    # constraint：||X beta - y||_inf <= lambda
    constraints = [cp.max(cp.abs(A @ beta - y)) <= lambda_]

    # solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # evaluate beta
    beta_hat = beta.value

    # support set recovery: Choose index where |beta_i| > tau
    tau = 1e-3  # Threshold, needs to be adjusted according to the problem
    support = np.where(np.abs(beta_hat) > tau)[0]

    return beta_hat, support



CLIME01_signal_list = []

for i in tqdm(range(variables_with_error.shape[1])):
    # Run CLIME-style optimization
    beta_hat, support_hat = clime_compressive_sensing(A,variables_with_error[:, i], 0.1)
    CLIME01_signal_list.append(beta_hat)

CLIME01_signal = np.array(CLIME01_signal_list)

np.save('CLIME01_signal.npy', CLIME01_signal, allow_pickle=True)
