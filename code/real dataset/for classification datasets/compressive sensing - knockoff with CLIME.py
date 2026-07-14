import numpy as np
from tqdm import tqdm
import pickle
from unittest import mock
from scipy.linalg import _decomp as linalg_decomp
from knockpy.knockoff_filter import KnockoffFilter
from knockpy.knockoffs import GaussianSampler
from sklearn.covariance import LedoitWolf
from knockpy.knockoff_stats import FeatureStatistic, combine_Z_stats
from sklearn.linear_model import LinearRegression
import cvxpy as cp

knockoff_selection_ratio = 0.01
high_dim = 500

variables_with_error= np.load("transformed_variables.npy", allow_pickle=True)

# Read measurement matrix from file
with open("measurement matrix.pkl", "rb") as f:
    A = pickle.load(f)

class ClimeStatistic(FeatureStatistic):
    """ CLIME-style compressive sensing statistic wrapper class """

    def __init__(self):
        super().__init__()

    def fit(
        self,
        X,
        Xk,
        y,
        groups=None,
        antisym="cd",
        group_agg="avg",
        cv_score=False,
        lambda_=0.1,  # Default regularization parameter
        tau=1e-3,     # Default support set threshold
        **kwargs,
    ):
        """
        Wraps the FeatureStatistic class but uses CLIME-style compressive sensing
        coefficients as variable importances.

        Parameters
        ----------
        X : np.ndarray
            the ``(n, p)``-shaped design matrix
        Xk : np.ndarray
            the ``(n, p)``-shaped matrix of knockoffs
        y : np.ndarray
            ``(n,)``-shaped response vector
        groups : np.ndarray
            For group knockoffs, a p-length array of integers from 1 to
            num_groups such that ``groups[j] == i`` indicates that variable `j`
            is a member of group `i`. Defaults to None (regular knockoffs).
        antisym : str
            The antisymmetric function used to create (ungrouped) feature
            statistics. Three options:
            - "CD" (Difference of absolute vals of coefficients),
            - "SM" (Signed maximum).
            - "SCD" (Simple difference of coefficients - NOT recommended)
        group_agg : str
            For group knockoffs, specifies how to turn individual feature
            statistics into grouped feature statistics. Two options:
            "sum" and "avg".
        cv_score : bool
            If true, score the feature statistic's predictive accuracy
            using cross validation. (Not implemented for CLIME)
        lambda_ : float
            Regularization parameter for CLIME
        tau : float
            Threshold for support recovery in CLIME
        kwargs : dict
            Extra kwargs (ignored in this implementation)

        Returns
        -------
        W : np.ndarray
            an array of feature statistics. This is ``(p,)``-dimensional
            for regular knockoffs and ``(num_groups,)``-dimensional for
            group knockoffs.
        """
        # Set default grouping
        p = X.shape[1]
        if groups is None:
            groups = np.arange(1, p + 1, 1)

        # Combine X and Xk for joint calculation
        X_full = np.hstack([X, Xk])

        # Standardize y (optional, ensures numerical stability)
        y = (y - np.mean(y)) / np.std(y)

        # Step 1: Compute Z statistics using CLIME-style compressed sensing
        beta_hat, _ = clime_compressive_sensing(
            A=X_full,
            y=y,
            lambda_=lambda_,
            tau=tau
        )

        # Step 2: Use beta_hat as Z (first p entries are original features, last p entries are knockoff features)
        Z = beta_hat  # beta_hat is already in shape (2p,)

        # Step 3: Combine Z statistics
        W_group = combine_Z_stats(Z, groups, antisym=antisym, group_agg=group_agg)

        # Save values for later use
        self.Z = Z
        self.groups = groups
        self.W = W_group

        # Cross-validation scores (not implemented)
        if cv_score:
            self.score = None
            self.score_type = "not_implemented"
            print("Warning: cv_score is not implemented for CLIME.")

        return W_group

def clime_compressive_sensing(A, y, lambda_, tau=1e-3):
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

    # constraint：||A beta - y||_inf <= lambda
    constraints = [cp.max(cp.abs(A @ beta - y)) <= lambda_]

    # solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    # evaluate beta
    beta_hat = beta.value

    # support set recovery: Choose index where |beta_i| > tau
    support = np.where(np.abs(beta_hat) > tau)[0]

    return beta_hat, support


# generate a knockoff matrix
kfilter = KnockoffFilter(
    fstat=ClimeStatistic(),
    ksampler='gaussian',
    knockoff_kwargs={"method": "mvr"}
)

def generate_knockoffs_for_A(A, method="mvr"):
    """
    Given an (m, n) measurement matrix A, generate its knockoff version A_k.
    Automatically handles anomalies in covariance matrix estimation or sampling.
    """
    X = A.copy()

    # Step 1: Safely estimate covariance matrix Sigma
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lw = LedoitWolf().fit(X)
            Sigma = lw.covariance_
    except Exception as e:
        print(f"LedoitWolf failed with {type(e).__name__}: {e}")

        # Replace scipy's eigh implementation with eig
        def patched_eig(a, lower=None, check_finite=False):
            w, vr = np.linalg.eig(a)
            return w, vr

        with mock.patch.object(linalg_decomp, 'eigh', patched_eig):
            print("Using patched eig() instead of eigh() for covariance estimation.")
            lw = LedoitWolf().fit(X)
            Sigma = lw.covariance_

    mu = X.mean(axis=0)

    # Step 2: Knockoff sampling; prevent internal linear algebra errors
    try:
        sampler = GaussianSampler(X=X, mu=mu, Sigma=Sigma, method=method)
        Xk = sampler.sample_knockoffs()
    except np.linalg.LinAlgError as e:
        raise RuntimeError("Knockoff sampling failed due to linear algebra issue.") from e
    return Xk

Ak = generate_knockoffs_for_A(A)


def run_experiment(y):
    results = {}
    # Compute knockoffs and feature statistics
    # Apply knockoff filter with FDR control
    fdr_selected, W = kfilter.forward(X=A, y=y, Xk=Ak)
    #print(f"W:{W}")
    # Calculate total number of variables and the number to select (10%)
    num_vars = len(W)
    num_to_select = int(np.ceil(knockoff_selection_ratio * num_vars))

   # Get indices of variables with highest W scores, using 'selected' directly
    selected = np.argsort(W)[-num_to_select:]
    #print(selected)

    # Estimate non-zero coefficients
    x_hat_knockoff = np.zeros(high_dim)
    if len(selected) > 0:
        lr = LinearRegression()
        x_hat_knockoff[selected] = lr.fit(A[:, selected], y).coef_

    return x_hat_knockoff

knockoffCS_signal_list_CLIME = []

for i in tqdm(range(variables_with_error.shape[1])):
    results = run_experiment(variables_with_error[:, i])
    knockoffCS_signal_list_CLIME.append(results)

knockoffCS_signal = np.array(knockoffCS_signal_list_CLIME)

np.save('knockoffCS_signal_CLIME.npy', knockoffCS_signal, allow_pickle=True)
