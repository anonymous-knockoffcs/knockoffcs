import numpy as np
from sklearn.linear_model import Lasso, orthogonal_mp,LinearRegression
from scipy.linalg import toeplitz
import pandas as pd
from itertools import product

num_batches = 10
trials = 20
# Simulation parameters
param_grid = {
    'knockoff_selection_ratio': [0.02],
    'n': [500, 1000],
    'm': [50, 100, 200],
    's': [5, 10],
    'snr': [2, 10, 30, 50],
    'correlation_type': ['block']
}

class CS_Simulator:
    def __init__(self, n=1000, m=200, s=20, snr=20, correlation_type='block', knockoff_selection_ratio=0.1):
        self.n = int(n)  # Signal dimension
        self.m = int(m)  # Measurements
        self.s = int(s)  # Sparsity
        self.snr = float(snr)  # Signal-to-noise ratio (dB)
        self.corr_type = correlation_type
        self.A, self.x_true = self._generate_design_matrix()
        self.knockoff_selection_ratio = float(knockoff_selection_ratio)
    def _generate_design_matrix(self):
        """Generate correlated design matrix with PD check"""
        # Add small epsilon to ensure positive definiteness
        eps = 1e-6
    
        if self.corr_type == 'block':
            # Block-diagonal correlation with size 5
            block_size = 5
            n_blocks = self.n // block_size
            cov = np.zeros((self.n, self.n))
            for i in range(n_blocks):
                block = np.full((block_size, block_size), 0.6)
                np.fill_diagonal(block, 1.0)
                start_idx = i * block_size
                end_idx = (i+1) * block_size
                cov[start_idx:end_idx, start_idx:end_idx] = block
            # Handle remaining dimensions
            remaining = self.n % block_size
            if remaining > 0:
                cov[-remaining:, -remaining:] = np.eye(remaining)
            
        elif self.corr_type == 'exponential':
            # Ensure valid exponential decay
            rho = 0.8  # Reduce correlation strength
            cov = toeplitz(rho**np.arange(0, self.n))
        else:
            cov = np.eye(self.n)
    
        # Ensure positive definiteness
        cov += eps * np.eye(self.n)

        # Verify PD before Cholesky
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Add more regularization if needed
            cov += (eps * 10) * np.eye(self.n)
            L = np.linalg.cholesky(cov)
        
        A = np.random.randn(self.m, self.n) @ L.T
        A = A / np.linalg.norm(A, axis=0)
    
        # Generate sparse signal
        x_true = np.zeros(self.n)
        supp = np.random.choice(self.n, self.s, replace=False)
        x_true[supp] = np.random.randn(self.s) * (1 + np.abs(np.random.randn(self.s)))
        return A, x_true

    
    def _add_noise(self, y_clean):
        """Add noise with specified SNR"""
        signal_power = np.mean(y_clean**2)
        sigma = np.sqrt(signal_power / (10**(self.snr/10)))
        return y_clean + sigma * np.random.randn(len(y_clean))

    def run_experiment(self, methods=['lasso', 'omp', 'knockoff']):
        """Run full simulation trial"""

        y_clean = self.A @ self.x_true
        y = self._add_noise(y_clean)  # Add noise to the signal
        results = {}

        # Common metrics calculator
        def compute_metrics(x_hat):
            assert not np.ma.isMaskedArray(x_hat), "compute metrics: x_hat is unexpectedly a MaskedArray!"
            TP = np.sum((x_hat != 0) & (self.x_true != 0))
            FP = np.sum((x_hat != 0) & (self.x_true == 0))
            FN = np.sum((x_hat == 0) & (self.x_true != 0))
            return {
                'FDR': FP / max(1, (TP + FP)),  # False Discovery Rate
                'Power': TP / self.s,           # Statistical power
                'RelError': np.linalg.norm(x_hat - self.x_true) / np.linalg.norm(self.x_true),  # Relative error
                'ReconstructionError': np.sum((y - self.A @ x_hat) ** 2)  # Reconstruction error
            }

        # LASSO implementation
        if 'lasso' in methods:
            lasso = Lasso(alpha=0.1 * np.sqrt(np.log(self.n) / self.m))
            lasso.fit(self.A, y)
            results['LASSO'] = compute_metrics(lasso.coef_)

        # Orthogonal Matching Pursuit
        if 'omp' in methods:
            omp_coef = orthogonal_mp(self.A, y)
            results['OMP'] = compute_metrics(omp_coef)

        # Generate MVR knockoffs
        if 'knockoff' in methods:
            from knockpy.knockoff_filter import KnockoffFilter

            # Compute knockoffs and feature statistics
            kfilter = KnockoffFilter(
                fstat='lcd',
                ksampler='gaussian',
                knockoff_kwargs={"method": "mvr"}
            )
            print(self.A.shape)
            # Apply knockoff filter with FDR control
            fdr_selected, W = kfilter.forward(X=self.A, y=y)
            print(f"W:{W}")
            print(f"x_true:{self.x_true}")
            # Compute the total number of variables and the number to select (10%)
            num_vars = len(W)
            num_to_select = int(np.ceil(self.knockoff_selection_ratio * num_vars))

            # Get indices of variables with the highest W scores; use `selected`
            selected = np.argsort(W)[-num_to_select:]
            print(selected)

            # Estimate non-zero coefficients
            x_hat_knockoff = np.zeros(self.n)
            if len(selected) > 0:
                lr = LinearRegression()
                x_hat_knockoff[selected] = lr.fit(self.A[:, selected], y).coef_
            print(f"x_hat_knockoff:{x_hat_knockoff}")
            results['Knockoff'] = compute_metrics(x_hat_knockoff)

        return results


def split_grid(param_grid, num_batches):
    """Split the parameter grid into the specified number of batches"""
    param_combinations = list(product(
        param_grid['knockoff_selection_ratio'],
        param_grid['n'],
        param_grid['m'],
        param_grid['s'],
        param_grid['snr'],
        param_grid['correlation_type']
    ))
    return np.array_split(param_combinations, num_batches)

def parameter_sweep(param_grid, trials=10, num_batches=num_batches):
    """Run experiments for each batch of the parameter grid and output results"""
    param_batches = split_grid(param_grid, num_batches)

    for batch_idx, batch in enumerate(param_batches):
        results = []

        for knockoff_selection_ratio, n, m, s, snr, corr in batch:
            print(f'Batch {batch_idx + 1}, knockoff_selection_ratio:{knockoff_selection_ratio}, m:{m}, n:{n}, s:{s}, snr:{snr}, corr:{corr}')

            method_results = {method: {'FDR': [], 'Power': [], 'RelError': [], 'ReconstructionError': []}
                              for method in ['LASSO', 'OMP', 'Knockoff']}

            for _ in range(trials):
                simulator = CS_Simulator(n=n, m=m, s=s, snr=snr, correlation_type=corr, knockoff_selection_ratio=knockoff_selection_ratio)
                exp_result = simulator.run_experiment()

                for method in method_results:
                    if method in exp_result:
                        for metric in method_results[method]:
                                method_results[method][metric].append(exp_result[method][metric])

            result_entry = {
                'knockoff_selection_ratio': knockoff_selection_ratio,
                'correlation_type': corr,
                'm': m,
                'n': n,
                's': s,
                'snr': snr,
            }

            # Compute mean and standard deviation for each method's results
            for method in method_results:
                for metric in method_results[method]:
                    values = method_results[method][metric]
                    if values:
                        result_entry[f'{method}_{metric}_mean'] = np.mean(values)
                        result_entry[f'{method}_{metric}_std'] = np.std(values)
                    else:
                        result_entry[f'{method}_{metric}_mean'] = None
                        result_entry[f'{method}_{metric}_std'] = None

            results.append(result_entry)

        # Save results for each batch
        df = pd.DataFrame(results)
        df.to_csv(f'parameter_sweep_results_with_std_batch_{batch_idx + 1}.csv', index=False)
        print(f'Batch {batch_idx + 1} saved as "parameter_sweep_results_batch_{batch_idx + 1}.csv"')

    print("All batches completed successfully.")


# Example usage and visualization
if __name__ == "__main__":
    parameter_sweep(param_grid, trials)
