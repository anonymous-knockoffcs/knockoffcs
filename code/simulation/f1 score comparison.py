data = {
    (100, 500): {
        (5, 'LASSO'): [0.470, 0.126, 0.500, 0.126],
        (5, 'OMP'): [0.109, 0.145, 0.182, 0.182],
        (5, 'KnockoffCS'): [0.356, 0.444, 0.667, 0.667],
        (10, 'LASSO'): [0.182, 0.398, 0.333, 0.182],
        (10, 'OMP'): [0.145, 0.222, 0.322, 0.333],
        (10, 'KnockoffCS'): [0.433, 0.700, 0.933, 1.000],
    },
    (100, 1000): {
        (5, 'LASSO'): [0.235, 0.126, 0.126, 0.235],
        (5, 'OMP'): [0.057, 0.076, 0.095, 0.095],
        (5, 'KnockoffCS'): [0.213, 0.240, 0.400, 0.400],
        (10, 'LASSO'): [0.179, 0.000, 0.235, 0.286],
        (10, 'OMP'): [0.091, 0.127, 0.164, 0.182],
        (10, 'KnockoffCS'): [0.244, 0.422, 0.533, 0.578],
    },
    (200, 500): {
        (5, 'LASSO'): [0.000, 0.126, 0.000, 0.126],
        (5, 'OMP'): [0.133, 0.133, 0.182, 0.182],
        (5, 'KnockoffCS'): [0.489, 0.489, 0.622, 0.667],
        (10, 'LASSO'): [0.064, 0.333, 0.064, 0.064],
        (10, 'OMP'): [0.200, 0.267, 0.333, 0.333],
        (10, 'KnockoffCS'): [0.567, 0.767, 0.967, 1.000],
    },
    (200, 1000): {
        (5, 'LASSO'): [0.333, 0.000, 0.000, 0.235],
        (5, 'OMP'): [0.063, 0.082, 0.095, 0.095],
        (5, 'KnockoffCS'): [0.240, 0.347, 0.400, 0.400],
        (10, 'LASSO'): [0.235, 0.126, 0.126, 0.182],
        (10, 'OMP'): [0.115, 0.145, 0.182, 0.182],
        (10, 'KnockoffCS'): [0.400, 0.556, 0.667, 0.667],
    }
}

ratios_lasso = []
ratios_omp = []

for (m, n), configs in data.items():
    for s in [5, 10]:
        kcs_vals = configs.get((s, 'KnockoffCS'))
        lasso_vals = configs.get((s, 'LASSO'))
        omp_vals = configs.get((s, 'OMP'))

        # Point-wise division
        if kcs_vals and lasso_vals:
            for k, l in zip(kcs_vals, lasso_vals):
                if l > 0:
                    ratios_lasso.append(k / l)
        if kcs_vals and omp_vals:
            for k, o in zip(kcs_vals, omp_vals):
                if o > 0:
                    ratios_omp.append(k / o)

# Compute averages
avg_lasso = sum(ratios_lasso) / len(ratios_lasso)
avg_omp = sum(ratios_omp) / len(ratios_omp)

print(f"Average KnockoffCS / LASSO F1 ratio: {avg_lasso:.2f}x")
print(f"Average KnockoffCS / OMP F1 ratio: {avg_omp:.2f}x")
