import arviz as az
import matplotlib.pyplot as plt
import os

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load trace from Phase 4
trace = az.from_netcdf("outputs/bayesian_sem_trace.nc")

# === SUMMARY ===
print("\nPOSTERIOR SUMMARY:\n")
summary = az.summary(trace, 
                     var_names=[
                         "lambda_panic", "lambda_risk", "lambda_vol", 
                         "beta_profit", "beta_rt", "beta_exit"
                     ])
print(summary)

# === TRACE PLOTS ===
print("\nPlotting trace plots...")
az.plot_trace(trace, 
              var_names=["lambda_panic", "lambda_risk", "lambda_vol"], 
              figsize=(10, 6))
plt.tight_layout()
plt.savefig("outputs/trace_factors.png", dpi=300)
plt.close()

# === FOREST PLOT ===
print("Plotting forest plot of structural coefficients...")

az.plot_forest(
    trace,
    var_names=["beta_profit", "beta_rt", "beta_exit"],
    combined=True,
    figsize=(8, 4),
    hdi_prob=0.94  # ← this is the correct version
)

plt.title("Structural Effects of Latent Trait (θ)")
plt.tight_layout()
plt.savefig("outputs/forest_structural.png", dpi=300)
plt.close()

print("All Phase 5 outputs saved in /outputs")
