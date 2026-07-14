import numpy as np
from sklearn.linear_model import Lasso, orthogonal_mp, LinearRegression
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt


trials = 1
# Simulation parameters
param_grid = {
    'knockoff_selection_ratio': [0.6],
    'n': [500],
    'm': [500],
    's': [300],
    'snr': [1],
    'correlation_type': ['block']
}
# param_grid = {
#     'knockoff_selection_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     'n': [500],
#     'm': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
#     's': [5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450],
#     'snr': [0.5, 1, 1.5, 2, 5, 10, 15, 20, 30, 40, 50],
#     'correlation_type': ['block', 'exponential']
# }

class CS_Simulator:
    def __init__(self, n=1000, m=200, s=20, snr=20, correlation_type='block', knockoff_selection_ratio=0.1):
        self.n = n  # Signal dimension
        self.m = m  # Measurements
        self.s = s  # Sparsity
        self.snr = snr  # Signal-to-noise ratio (dB)
        self.corr_type = correlation_type
        self.A, self.x_true = self._generate_design_matrix()
        self.knockoff_selection_ratio = knockoff_selection_ratio
    def _generate_design_matrix(self):
        """Generate correlated design matrix with PD check"""
        # Add small epsilon to ensure positive definiteness
        eps = 1e-6
    
        if self.corr_type == 'block':
            # Block-diagonal correlation with size 5
            block_size = 5
            n_blocks = self.n // block_size
            cov = np.zeros((self.n, self.n))
            print(f"cov = np.zeros((self.n, self.n)):{cov.shape}")
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
            print(f"L = np.linalg.cholesky(cov): {L.shape}")
        except np.linalg.LinAlgError:
            # Add more regularization if needed
            cov += (eps * 10) * np.eye(self.n)
            L = np.linalg.cholesky(cov)
        
        A = np.random.randn(self.m, self.n) @ L.T
        print(f"A = np.random.randn(self.m, self.n) @ L.T:{A.shape}")
        A = A / np.linalg.norm(A, axis=0)
        print(f"A = A / np.linalg.norm(A, axis=0): {A.shape}")
    
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

def parameter_sweep(param_grid):
    # Run trials
    results = []
    for corr in param_grid['correlation_type']:
        for m in param_grid['m']:
            for n in param_grid['n']:
                for s in param_grid['s']:
                    for snr in param_grid['snr']:
                        for knockoff_selection_ratio in param_grid['knockoff_selection_ratio']:
                            print(f'corr:{corr}, m:{m},n:{n},s:{s},snr:{snr},knockoff_selection_ratio:{knockoff_selection_ratio}')
                            # Initialize results for each repeated trial
                            method_results = {method: {'FDR': [], 'Power': [], 'RelError': [], 'ReconstructionError': []}
                                          for method in ['LASSO', 'OMP', 'Knockoff']}
                            for _ in range(trials):  # Number of trials per config
                                simulator = CS_Simulator(
                                    n=n, m=m, s=s, snr=snr, correlation_type=corr, knockoff_selection_ratio = knockoff_selection_ratio
                                )
                                exp_result = simulator.run_experiment()
                                for method in method_results:
                                    if method in exp_result:
                                        for metric in method_results[method]:
                                            method_results[method][metric].append(exp_result[method][metric])

                            # Construct a standard DataFrame row entry
                            result_entry = {
                                'correlation_type': corr,  # Explicitly store `correlation_type`
                                'm': m,
                                'n': n,
                                's': s,
                                'snr': snr,
                                'knockoff_selection_ratio': knockoff_selection_ratio
                            }

                            # Average results for each method
                            for method in method_results:
                                for metric in method_results[method]:
                                    values = np.array(method_results[method][metric], dtype=float)  # Convert to `ndarray`
                                    result_entry[f'{method}_{metric}'] = np.nanmean(values) if values.size > 0 else None

                        # # Construct standard DataFrame row entry
                        # result_entry = {
                        #     'trials': _,
                        #     'correlation_type': corr,  # Explicitly store `correlation_type`
                        #     'm': m,
                        #     'n': n,
                        #     's': s,
                        #     'snr': snr,
                        # }
                        #
                        # # Parse LASSO, OMP, Knockoff metrics from `exp_result`
                        # for method in ['LASSO', 'OMP', 'Knockoff']:
                        #     if method in exp_result:
                        #         result_entry[f'{method}_FDR'] = exp_result[method]['FDR']
                        #         result_entry[f'{method}_Power'] = exp_result[method]['Power']
                        #         result_entry[f'{method}_RelError'] = exp_result[method]['RelError']
                        #         result_entry[f'{method}_ReconstructionError'] = exp_result[method][
                        #             'ReconstructionError']
                        #     else:
                        #         result_entry[f'{method}_FDR'] = None
                        #         result_entry[f'{method}_Power'] = None
                        #         result_entry[f'{method}_RelError'] = None
                        #         result_entry[f'{method}_ReconstructionError'] = None

                        results.append(result_entry)

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
    reconstruction_df = df[['correlation_type', 'm', 'n', 's', 'snr','knockoff_selection_ratio',
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

            # **Use 5%-95% range to compute axis limits to avoid outlier influence**
            x_min, x_max = np.percentile(x_values, [5, 95])
            y_min, y_max = np.percentile(y_values, [5, 95])

            # **Add appropriate margins**
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
    # Add a global legend across the figure
    fig.legend(handles, labels, loc='upper center', ncol=3)

    plt.tight_layout()
    plt.savefig('simulation_summary.png')


