import numpy as np
import cvxpy as cp
from tqdm import tqdm
import pickle
import mosek

knockoff_selection_ratio = 0.01
high_dim = 500

variables_with_error= np.load("transformed_variables.npy", allow_pickle=True)

# Read measurement matrix from file
with open("Another measurement matrix.pkl", "rb") as f:
    A = pickle.load(f)


def dantzig_compressive_sensing(A, y, lambda_):
    """
    Compressed sensing support set recovery using dantzig-style optimization
    Parameters:
        X: measurement matrix (n x p)
        y: observation vector (n,)
        lambda_: regularization parameter
    Returns:
        beta_hat: estimated sparse signal
        support: estimated support set
    """
    p = A.shape[1]
    ATA = A.T @ A
    # define variables
    beta = cp.Variable(p)

    # goal: min ||beta||_1
    objective = cp.Minimize(cp.norm(beta, 1))

    # constraint: ||X^T (y - X beta)||_inf <= lambda
    # constraint: ||X beta - y||_inf <= lambda
    constraints = [cp.max(cp.abs(ATA @ beta - A.T @ y)) <= lambda_]

    # solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    # evaluate beta
    beta_hat = beta.value

    # support set recovery: Choose index where |beta_i| > tau
    tau = 1e-3  # Threshold, needs to be adjusted according to the problem
    support = np.where(np.abs(beta_hat) > tau)[0]

    return beta_hat, support


def dantzig_selector_admm(X, y, lambda_, rho=1.0, max_iter=1000, tol=1e-4):
    """
    Implement Dantzig Selector using ADMM
    Parameters:
        X: design matrix (n x p)
        y: response vector (n,)
        lambda_: regularization parameter
        rho: ADMM penalty parameter
        max_iter: maximum number of iterations
        tol: convergence tolerance
    Returns:
        beta: estimated regression coefficients
    """
    n, p = X.shape
    Xt = X.T
    XtX = Xt @ X
    Xty = Xt @ y

    # initialization
    beta = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)

    # ADMM iteration
    for _ in range(max_iter):
        # Update beta: (XtX + rho I) beta = Xty - z + u
        beta_old = beta.copy()
        A = XtX + rho * np.eye(p)
        b = Xty - z + u
        beta = np.linalg.solve(A, b)

        # Update z: project onto ||z||_inf <= lambda
        z = np.clip(Xt @ (y - X @ beta) + u, -lambda_, lambda_)

        # Update u
        u = u + Xt @ (y - X @ beta) - z

        # Check convergence
        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta


dantzig01_signal_list = []

for i in tqdm(range(variables_with_error.shape[1])):
    # Run Dantzig-style optimization
    beta_hat = dantzig_selector_admm(A,variables_with_error[:, i], 0.1)
    dantzig01_signal_list.append(beta_hat)

dantzig01_signal = np.array(dantzig01_signal_list)

np.save('dantzig01_signal.npy', dantzig01_signal, allow_pickle=True)
