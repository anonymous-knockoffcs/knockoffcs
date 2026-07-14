import numpy as np
from sklearn.linear_model import Lasso, orthogonal_mp
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product


trials = 1
# Simulation parameters
param_grid = {
    'fdr': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'n': [500],
    'm': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    's': [5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450],
    'snr': [0.5, 1, 1.5, 2, 5, 10, 15, 20, 30, 40, 50],
    'correlation_type': ['block', 'exponential']
}

class CS_Simulator:
    def __init__(self, n=1000, m=200, s=20, snr=20, correlation_type='block', fdr=0.1):
        self.n = n  # Signal dimension
        self.m = m  # Measurements
        self.s = s  # Sparsity
        self.snr = snr  # Signal-to-noise ratio (dB)
        self.corr_type = correlation_type
        self.A, self.x_true = self._generate_design_matrix()
        self.fdr = fdr
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
            # Apply knockoff filter with FDR control
            selected, W = kfilter.forward(X=self.A, y=y, fdr=self.fdr)

            # Ensure selected is an integer array
            selected = np.array(selected).astype(int)
            if selected.ndim != 1:
                selected = selected.flatten()  # Ensure it's a one-dimensional array
            print(selected)  # Check type and shape

            # Estimate non-zero coefficients
            x_hat_knockoff = np.zeros(self.n)
            if len(selected) > 0:
                #x_hat_knockoff[selected] = np.linalg.lstsq(
                    #self.A[:, selected], y, rcond=None
                #)[0]
                x_hat_knockoff[selected] = Lasso(alpha=0.1).fit(self.A[:, selected], y).coef_
            results['Knockoff'] = compute_metrics(x_hat_knockoff)

        return results


def split_grid(param_grid, num_batches):
    """Split the parameter grid into the specified number of batches."""
    param_combinations = list(product(
        param_grid['fdr'],
        param_grid['n'],
        param_grid['m'],
        param_grid['s'],
        param_grid['snr'],
        param_grid['correlation_type']
    ))
    return np.array_split(param_combinations, num_batches)

def parameter_sweep(param_grid, trials=10, num_batches=5):
    """Run experiments sequentially for each batch of the parameter grid and output results."""
    param_batches = split_grid(param_grid, num_batches)

    for batch_idx, batch in enumerate(param_batches):
        results = []

        for fdr, n, m, s, snr, corr in batch:
            print(f'Batch {batch_idx + 1}, fdr:{fdr}, m:{m}, n:{n}, s:{s}, snr:{snr}, corr:{corr}')

            method_results = {method: {'FDR': [], 'Power': [], 'RelError': [], 'ReconstructionError': []}
                              for method in ['LASSO', 'OMP', 'Knockoff']}

            for _ in range(trials):
                simulator = CS_Simulator(n=n, m=m, s=s, snr=snr, correlation_type=corr, fdr=fdr)
                exp_result = simulator.run_experiment()

                for method in method_results:
                    if method in exp_result:
                        for metric in method_results[method]:
                                method_results[method][metric].append(exp_result[method][metric])

                result_entry = {
                    'trials': trials,
                    'fdr': fdr,
                    'correlation_type': corr,
                    'm': m,
                    'n': n,
                    's': s,
                    'snr': snr,
                }

                # Average results for each method
                for method in method_results:
                    for metric in method_results[method]:
                        values = method_results[method][metric]
                        result_entry[f'{method}_{metric}'] = sum(values) / len(values) if values else None

                results.append(result_entry)

            # Save results for each batch
            df = pd.DataFrame(results)
            df.to_csv(f'parameter_sweep_results_batch_{batch_idx + 1}.csv', index=False)
            print(f'Batch {batch_idx + 1} saved as "parameter_sweep_results_batch_{batch_idx + 1}.csv"')

        print("All batches completed successfully.")

    return pd.DataFrame(results)

import pandas as pd

# Example usage and visualization
if __name__ == "__main__":
    df = parameter_sweep(param_grid)

    # Save results
    df.to_csv('cs_simulation_results.csv', index=False)

    # # Create reconstruction error table
    # reconstruction_df = df[['trials', 'correlation_type', 'm','n','s', 'snr',
    #                         'LASSO_ReconstructionError',
    #                         'OMP_ReconstructionError',
    #                         'Knockoff_ReconstructionError']]
    # Create reconstruction error table
    reconstruction_df = df[['correlation_type', 'm', 'n', 's', 'snr','fdr',
                            'LASSO_ReconstructionError',
                            'OMP_ReconstructionError',
                            'Knockoff_ReconstructionError']]

    # Print and save
    print("Reconstruction Error Table:")
    print(reconstruction_df)
    reconstruction_df.to_csv('reconstruction_error_table.csv', index=False)

    # Generate summary plots
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'LASSO': 'red', 'OMP': 'blue', 'Knockoff': 'green'}  # Define distinct colors for each method

    for method in ['LASSO', 'OMP', 'Knockoff']:
        fdr_col = f'{method}_FDR'
        power_col = f'{method}_Power'
        relerror_col = f'{method}_RelError'

        if fdr_col in df.columns and power_col in df.columns:
            sc1 = ax[0].scatter(
                df[fdr_col],
                df[power_col],
                label=method,
                color=colors[method]
            )

        if 'snr' in df.columns and relerror_col in df.columns:
            sc2 = ax[1].scatter(
                df['snr'],
                df[relerror_col],
                color=colors[method]
            )

        if 'm' in df.columns and 'n' in df.columns and power_col in df.columns:
            sc3 = ax[2].scatter(
                df['m'] / df['n'],  # Check if 'n' exists
                df[power_col],
                color=colors[method]
            )

    # **Manually adjust axis ranges**
    for ax_ in ax:
        if ax_.collections:  # Ensure scatter contains data
            x_values = np.concatenate([col.get_offsets()[:, 0] for col in ax_.collections])
            y_values = np.concatenate([col.get_offsets()[:, 1] for col in ax_.collections])

            # **Use the 5%-95% range to compute axis limits to avoid extreme outliers**
            x_min, x_max = np.percentile(x_values, [5, 95])
            y_min, y_max = np.percentile(y_values, [5, 95])

            # **Increase margins appropriately**
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1

            ax_.set_xlim(x_min - x_margin, x_max + x_margin)
            ax_.set_ylim(y_min - y_margin, y_max + y_margin)

    # **Add axis titles**
    ax[0].set_xlabel('FDR')
    ax[0].set_ylabel('Power')
    ax[0].set_title('FDR-Power Tradeoff')

    ax[1].set_xlabel('SNR (dB)')
    ax[1].set_ylabel('Relative Error')
    ax[1].set_title('SNR vs Relative Error')

    ax[2].set_xlabel('Measurement Ratio (m/n)')
    ax[2].set_ylabel('Power')
    ax[2].set_title('Measurement Ratio vs Power')

    # Collect artists and labels from the first subplot
    handles, labels = ax[0].get_legend_handles_labels()
    # Add a global legend to the entire figure
    fig.legend(handles, labels, loc='upper center', ncol=3)

    plt.tight_layout()
    plt.savefig('simulation_summary.png')


