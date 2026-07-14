import pandas as pd

# Read CSV file
file_path = "result.csv"
df = pd.read_csv(file_path)

# Select required columns
fdr_power_columns = ["m", "n", "s", "snr",  "LASSO_FDR", "OMP_FDR", "Knockoff_FDR","LASSO_Power", "OMP_Power", "Knockoff_Power"]
fdr_power_df = df[fdr_power_columns].copy()
# Keep SNR as integer
fdr_power_df["snr"] = fdr_power_df["snr"].round(0).astype(int)
#fdr_power_df["knockoff_selection_ratio"] = fdr_power_df["knockoff_selection_ratio"].map(lambda x: f"{x:.2f}")  # Keep one decimal place
# Round FDR and power data
fdr_power_df[["LASSO_FDR", "OMP_FDR", "Knockoff_FDR","LASSO_Power", "OMP_Power", "Knockoff_Power"]] = fdr_power_df[["LASSO_FDR", "OMP_FDR", "Knockoff_FDR","LASSO_Power", "OMP_Power", "Knockoff_Power"]].round(3)
fdr_power_df["Lasso_F1"] = 2 * (1 - fdr_power_df["LASSO_FDR"]) * fdr_power_df["LASSO_Power"] / (1 - fdr_power_df["LASSO_FDR"] + fdr_power_df["LASSO_Power"])
fdr_power_df["OMP_F1"] = 2 * (1 - fdr_power_df["OMP_FDR"]) * fdr_power_df["OMP_Power"] / (1 - fdr_power_df["OMP_FDR"] + fdr_power_df["OMP_Power"])
fdr_power_df["Knockoff_F1"] = 2 * (1 - fdr_power_df["Knockoff_FDR"]) * fdr_power_df["Knockoff_Power"] / (1 - fdr_power_df["Knockoff_FDR"] + fdr_power_df["Knockoff_Power"])
#fdr_power_df.drop(inplace = True,columns=["LASSO_FDR", "OMP_FDR", "Knockoff_FDR","LASSO_Power", "OMP_Power", "Knockoff_Power"])

# Convert to LaTeX table
fdr_power_latex_table = fdr_power_df.to_latex(index=False, float_format="%.3f")
# Output LaTeX code
print(fdr_power_latex_table)

error_columns = ["m", "n", "s", "snr",  "LASSO_RelError", "OMP_RelError", "Knockoff_RelError","LASSO_ReconstructionError", "OMP_ReconstructionError", "Knockoff_ReconstructionError"]
error_df = df[error_columns].copy()
# Keep SNR as integer
error_df["snr"] = error_df["snr"].round(0).astype(int)
#error_df["knockoff_selection_ratio"] = error_df["knockoff_selection_ratio"].map(lambda x: f"{x:.2f}")  # Keep one decimal place
# Round error data if needed
#error_df[["LASSO_RelError", "OMP_RelError", "Knockoff_RelError","LASSO_ReconstructionError", "OMP_ReconstructionError", "Knockoff_ReconstructionError"]] = error_df[["LASSO_RelError", "OMP_RelError", "Knockoff_RelError","LASSO_ReconstructionError", "OMP_ReconstructionError", "Knockoff_ReconstructionError"]].round(3)
# Convert to LaTeX table
#error_latex_table = error_df.to_latex(index=False, float_format="%.3f")
# Convert to LaTeX table
error_latex_table = error_df.to_latex(index=False)
# Output LaTeX code
print(error_latex_table)
