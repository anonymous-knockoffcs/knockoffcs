import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Set style for academic publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Create a figure
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# Set up data
snr_values = [2, 10, 30, 50]

# LASSO data (s=5)
lasso_fdr_s5 = [0.200, 0.000, 0.000, 0.000]
lasso_power_s5 = [0.333, 0.067, 0.333, 0.067]

# OMP data (s=5)
omp_fdr_s5 = [0.940, 0.920, 0.900, 0.900]
omp_power_s5 = [0.600, 0.800, 1.000, 1.000]

# Knockoff data (s=5)
knockoff_fdr_s5 = [0.733, 0.667, 0.500, 0.500]
knockoff_power_s5 = [0.533, 0.667, 1.000, 1.000]

# LASSO data (s=10)
lasso_fdr_s10 = [0.000, 0.222, 0.000, 0.000]
lasso_power_s10 = [0.100, 0.267, 0.200, 0.100]

# OMP data (s=10)
omp_fdr_s10 = [0.913, 0.867, 0.807, 0.800]
omp_power_s10 = [0.433, 0.667, 0.967, 1.000]

# Knockoff data (s=10)
knockoff_fdr_s10 = [0.567, 0.300, 0.067, 0.000]
knockoff_power_s10 = [0.433, 0.700, 0.933, 1.000]

# Plot data points for s=5
lasso_s5 = ax.scatter(lasso_fdr_s5, lasso_power_s5, s=100, c='blue', marker='o', label='LASSO (s=5)', alpha=0.7, edgecolors='black')
omp_s5 = ax.scatter(omp_fdr_s5, omp_power_s5, s=100, c='orange', marker='s', label='OMP (s=5)', alpha=0.7, edgecolors='black')
knockoff_s5 = ax.scatter(knockoff_fdr_s5, knockoff_power_s5, s=100, c='green', marker='^', label='Knockoff (s=5)', alpha=0.7, edgecolors='black')

# Plot data points for s=10
lasso_s10 = ax.scatter(lasso_fdr_s10, lasso_power_s10, s=100, c='blue', marker='o', label='LASSO (s=10)', alpha=0.7, edgecolors='black', linestyle='--')
omp_s10 = ax.scatter(omp_fdr_s10, omp_power_s10, s=100, c='orange', marker='s', label='OMP (s=10)', alpha=0.7, edgecolors='black', linestyle='--')
knockoff_s10 = ax.scatter(knockoff_fdr_s10, knockoff_power_s10, s=100, c='green', marker='^', label='Knockoff (s=10)', alpha=0.7, edgecolors='black', linestyle='--')

# Add dashed lines to connect points for s=5
for i in range(len(snr_values)-1):
    ax.plot([lasso_fdr_s5[i], lasso_fdr_s5[i+1]], [lasso_power_s5[i], lasso_power_s5[i+1]], 'b--', alpha=0.5)
    ax.plot([omp_fdr_s5[i], omp_fdr_s5[i+1]], [omp_power_s5[i], omp_power_s5[i+1]], 'orange', linestyle='--', alpha=0.5)
    ax.plot([knockoff_fdr_s5[i], knockoff_fdr_s5[i+1]], [knockoff_power_s5[i], knockoff_power_s5[i+1]], 'g--', alpha=0.5)

# Add dashed lines to connect points for s=10
for i in range(len(snr_values)-1):
    ax.plot([lasso_fdr_s10[i], lasso_fdr_s10[i+1]], [lasso_power_s10[i], lasso_power_s10[i+1]], 'b:', alpha=0.5)
    ax.plot([omp_fdr_s10[i], omp_fdr_s10[i+1]], [omp_power_s10[i], omp_power_s10[i+1]], 'orange', linestyle=':', alpha=0.5)
    ax.plot([knockoff_fdr_s10[i], knockoff_fdr_s10[i+1]], [knockoff_power_s10[i], knockoff_power_s10[i+1]], 'g:', alpha=0.5)

# Add SNR annotations for Knockoff s=10
for i, snr in enumerate(snr_values):
    if i > 0:
        dx = knockoff_fdr_s10[i] - knockoff_fdr_s10[i-1]
        dy = knockoff_power_s10[i] - knockoff_power_s10[i-1]
        ax.annotate('',
                   xy=(knockoff_fdr_s10[i], knockoff_power_s10[i]),
                   xytext=(knockoff_fdr_s10[i-1], knockoff_power_s10[i-1]),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7))
    ax.annotate(f'SNR={snr}',
               xy=(knockoff_fdr_s10[i], knockoff_power_s10[i]),
               xytext=(5, 5),
               textcoords='offset points',
               fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# Add a rectangle to highlight the ideal region
ideal_region = patches.Rectangle((0, 0.8), 0.2, 0.2, linewidth=2, edgecolor='red', facecolor='none',
                                linestyle='--', label='Ideal Region')
ax.add_patch(ideal_region)
ax.annotate('Ideal Region\n(Low FDR, High Power)',
           xy=(0.1, 0.9),
           xytext=(0.25, 0.75),
           arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
           fontsize=12,
           color='red',
           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# Set plot limits and add grid
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, linestyle='--', alpha=0.7)

# Add labels and title
ax.set_xlabel('False Discovery Rate (FDR)')
ax.set_ylabel('Power (True Positive Rate)')
ax.set_title('FDR-Power Trade-off for Compressive Sensing Methods (m=100, n=500)')

# Add custom legend with all elements
ideal_proxy = patches.Rectangle((0, 0), 1, 1, fc='none', ec='red', linestyle='--', label='Ideal Region')
handles = [lasso_s5, omp_s5, knockoff_s5, lasso_s10, omp_s10, knockoff_s10, ideal_proxy]
labels = ['LASSO (s=5)', 'OMP (s=5)', 'Knockoff (s=5)', 'LASSO (s=10)', 'OMP (s=10)', 'Knockoff (s=10)', 'Ideal Region']
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=True, shadow=True)

# Add text explaining the arrows (position adjusted to avoid overlap)
ax.text(0.5, 0.15, 'Arrows indicate increasing SNR (2→10→30→50)',
        horizontalalignment='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# Add text explaining the trade-off (position adjusted to avoid overlap)
ax.text(0.98, 0.02,
        'Methods closer to the top-left corner\nperform better (higher power, lower FDR)',
        horizontalalignment='right',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# Adjust layout to accommodate legend and text
plt.tight_layout(rect=[0, 0.1, 1, 1])

# Define the list of image titles
_mfajlsdf98q21_image_title_list = ["FDR-Power Trade-off for Compressive Sensing Methods"]

# Show the plot
plt.show()
