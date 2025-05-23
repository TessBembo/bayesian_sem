import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load and prep data
df = pd.read_csv("data/full_simulated_data.csv")

columns = [
    "panic_response", "risk_preference", "volatility_avoidance",
    "net_profit", "reaction_time", "market_exit_day"
]

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)
data = {col: df_scaled[col].values for col in columns}
n = df_scaled.shape[0]

# MODEL + SAMPLING
if __name__ == "__main__":
    with pm.Model() as bsem:

        # Latent trait
        theta = pm.Normal("theta", mu=0, sigma=1, shape=n)

        # Measurement model
        lambda_panic = pm.Normal("lambda_panic", mu=1, sigma=0.5)
        lambda_risk = pm.Normal("lambda_risk", mu=1, sigma=0.5)
        lambda_vol = pm.Normal("lambda_vol", mu=1, sigma=0.5)

        pm.Normal("panic_obs", mu=lambda_panic * theta, sigma=1, observed=data["panic_response"])
        pm.Normal("risk_obs", mu=lambda_risk * theta, sigma=1, observed=data["risk_preference"])
        pm.Normal("vol_obs", mu=lambda_vol * theta, sigma=1, observed=data["volatility_avoidance"])

        # Structural model
        beta_profit = pm.Normal("beta_profit", mu=0, sigma=1)
        beta_rt = pm.Normal("beta_rt", mu=0, sigma=1)
        beta_exit = pm.Normal("beta_exit", mu=0, sigma=1)

        sigma_profit = pm.Exponential("sigma_profit", 1)
        sigma_rt = pm.Exponential("sigma_rt", 1)
        sigma_exit = pm.Exponential("sigma_exit", 1)

        pm.Normal("profit_obs", mu=beta_profit * theta, sigma=sigma_profit, observed=data["net_profit"])
        pm.Normal("rt_obs", mu=beta_rt * theta, sigma=sigma_rt, observed=data["reaction_time"])
        pm.Normal("exit_obs", mu=beta_exit * theta, sigma=sigma_exit, observed=data["market_exit_day"])

        # Sampling
        trace = pm.sample(
            draws=1000,
            tune=1000,
            target_accept=0.9,
            return_inferencedata=True
        )

        # Save trace
        az.to_netcdf(trace, "outputs/bayesian_sem_trace.nc")
        print("Trace successfully saved to outputs/bayesian_sem_trace.nc")


