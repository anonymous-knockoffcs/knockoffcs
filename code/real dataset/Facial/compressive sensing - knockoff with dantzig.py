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


class DantzigStatistic(FeatureStatistic):
    """ Dantzig Selector statistic wrapper class using ADMM """

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
        lambda_=0.1,  # default regularization parameter
        rho=1.0,      # default ADMM penalty parameter
        max_iter=1000,  # default maximum number of iterations
        tol=1e-4,     # default convergence tolerance
        **kwargs,
    ):
        """
        Wraps the FeatureStatistic class but uses Dantzig Selector coefficients
        computed via ADMM as variable importances.

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
            using cross validation. (Not implemented for Dantzig Selector)
        lambda_ : float
            Regularization parameter for Dantzig Selector
        rho : float
            ADMM penalty parameter
        max_iter : int
            Maximum number of iterations for ADMM
        tol : float
            Convergence tolerance for ADMM
        kwargs : dict
            Extra kwargs (ignored in this implementation)

        Returns
        -------
        W : np.ndarray
            an array of feature statistics. This is ``(p,)``-dimensional
            for regular knockoffs and ``(num_groups,)``-dimensional for
            group knockoffs.
        """
        # Set default groups
        p = X.shape[1]
        if groups is None:
            groups = np.arange(1, p + 1, 1)

        # Combine X and X_k for joint computation
        X_full = np.hstack([X, Xk])

        # Standardize y (optional, for numerical stability)
        y = (y - np.mean(y)) / np.std(y)

        # Step 1: Compute Z statistics using the Dantzig Selector
        beta = dantzig_selector_admm(
            X=X_full,
            y=y,
            lambda_=lambda_,
            rho=rho,
            max_iter=max_iter,
            tol=tol
        )

        # Step 2: Use beta_hat as Z (first p entries are original features, last p entries are knockoff features)
        Z = beta  # beta 已经是 (2p,) 形状

        # Step 3: 组合 Z 统计量
        W_group = combine_Z_stats(Z, groups, antisym=antisym, group_agg=group_agg)

        # 保存值以供后续使用
        self.Z = Z
        self.groups = groups
        self.W = W_group

        # 交叉验证分数（未实现）
        if cv_score:
            self.score = None
            self.score_type = "not_implemented"
            print("Warning: cv_score is not implemented for Dantzig Selector.")

        return W_group

def dantzig_selector_admm(X, y, lambda_, rho=1.0, max_iter=1000, tol=1e-4):
    """
    使用 ADMM 实现 Dantzig Selector
    参数：
        X: 设计矩阵 (n x p)
        y: 响应向量 (n,)
        lambda_: 正则化参数
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛容差
    返回：
        beta: 估计的回归系数
    """
    n, p = X.shape
    Xt = X.T
    XtX = Xt @ X
    Xty = Xt @ y

    # 初始化
    beta = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)

    # ADMM 迭代
    for _ in range(max_iter):
        # 更新 beta：(XtX + rho I) beta = Xty - z + u
        beta_old = beta.copy()
        A = XtX + rho * np.eye(p)
        b = Xty - z + u
        beta = np.linalg.solve(A, b)

        # 更新 z：投影到 ||z||_inf <= lambda
        z = np.clip(Xt @ (y - X @ beta) + u, -lambda_, lambda_)

        # 更新 u
        u = u + Xt @ (y - X @ beta) - z

        # 检查收敛
        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta



# 生成 knockoff 矩阵（只需生成一次）
kfilter = KnockoffFilter(
    fstat=DantzigStatistic(),
    ksampler='gaussian',
    knockoff_kwargs={"method": "mvr"}
)

def generate_knockoffs_for_A(A, method="mvr"):
    """
    给定一个 (m, n) 的测量矩阵 A，生成其 knockoff 版本 A_k。
    自动处理协方差矩阵估计或采样中的异常。
    """
    X = A.copy()

    # Step 1: 安全估计协方差矩阵 Sigma
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lw = LedoitWolf().fit(X)
            Sigma = lw.covariance_
    except Exception as e:
        print(f"LedoitWolf failed with {type(e).__name__}: {e}")

        # 替换 scipy 的 eigh 实现为 eig
        def patched_eig(a, lower=None, check_finite=False):
            w, vr = np.linalg.eig(a)
            return w, vr

        with mock.patch.object(linalg_decomp, 'eigh', patched_eig):
            print("Using patched eig() instead of eigh() for covariance estimation.")
            lw = LedoitWolf().fit(X)
            Sigma = lw.covariance_

    mu = X.mean(axis=0)

    # Step 2: Knockoff sampling，防止内部线性代数出错
    try:
        sampler = GaussianSampler(X=X, mu=mu, Sigma=Sigma, method=method)
        Xk = sampler.sample_knockoffs()
    except np.linalg.LinAlgError as e:
        raise RuntimeError("Knockoff sampling failed due to linear algebra issue.") from e
    return Xk

# 从文件中读取测量矩阵
Ak = generate_knockoffs_for_A(A)


def run_experiment(y):
    results = {}
    # Compute knockoffs and feature statistics
    # Apply knockoff filter with FDR control
    fdr_selected, W = kfilter.forward(X=A, y=y, Xk=Ak)
    #print(f"W:{W}")
    # 计算变量总数和需要选择的变量个数（10%）
    num_vars = len(W)
    num_to_select = int(np.ceil(knockoff_selection_ratio * num_vars))

    # 获取W分数最高的变量索引，直接用selected表示
    selected = np.argsort(W)[-num_to_select:]
    #print(selected)

    # Estimate non-zero coefficients
    x_hat_knockoff = np.zeros(high_dim)
    if len(selected) > 0:
        lr = LinearRegression()
        x_hat_knockoff[selected] = lr.fit(A[:, selected], y).coef_

    return x_hat_knockoff

knockoffCS_signal_list_Dantzig = []

for i in tqdm(range(variables_with_error.shape[1])):
    results = run_experiment(variables_with_error[:, i])
    knockoffCS_signal_list_Dantzig.append(results)

knockoffCS_signal = np.array(knockoffCS_signal_list_Dantzig)

np.save('knockoffCS_signal_Dantzig.npy', knockoffCS_signal, allow_pickle=True)
