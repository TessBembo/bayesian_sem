# Bayesian Structural Equation Modeling of Latent Traits

This project simulates a latent psychological trait (θ) and uses Bayesian structural equation modeling (BSEM) to recover it from observed behaviors and predict downstream outcomes. The goal is to demonstrate how abstract cognitive profiles—like holistic vs analytic thinking—can influence financial behavior under uncertainty.

## Concept

We simulate a population of agents, each with a latent trait score. This unobserved trait affects:
- Panic response
- Risk preference
- Volatility avoidance

These behavioral indicators are noisy but informative reflections of the underlying trait.

The latent trait also influences downstream financial outcomes:
- Net profit
- Reaction time
- Market exit timing

This structure mirrors real-world situations where hidden psychological variables shape both observed behavior and financial decisions.

## Modeling Approach

We use PyMC to:
- Define a Bayesian measurement model of the latent trait using three behavioral indicators
- Fit a structural model predicting three financial outcomes from the latent trait
- Sample from the full posterior using the NUTS algorithm

All data were simulated in advance and standardized.

## Results

The model recovers posterior distributions for:

- Measurement model parameters (`lambda_*`)  
- Structural regression paths (`beta_*`)

### Structural Effects of Latent Trait (θ)

![Structural Forest Plot](outputs/forest_structural.png)

This plot shows the 94% highest density intervals (HDIs) for how the latent trait θ influences financial outcomes.  
For example, θ has a positive estimated effect on market exit timing (agents with higher θ scores tend to stay longer before exiting).

### Behavioral Loadings (Indicators of θ)

![Measurement Forest Plot](outputs/forest_measurement.png)

This plot shows how strongly each behavioral indicator (panic response, risk preference, volatility avoidance) loads onto the latent trait θ.  
Stronger loadings mean that indicator is more informative of the underlying psychological profile.

---

### Posterior File

This repo includes `outputs/bayesian_sem_trace.nc`, a NetCDF file containing the full posterior samples from the PyMC model.  
This allows you to reload results and make new plots without re-running the sampling.

To load it:

```python
import arviz as az
trace = az.from_netcdf("outputs/bayesian_sem_trace.nc")


## File Structure

- `phase_1_2_bayesian_sem.py`: Simulates the latent trait and behavioral indicators  
- `phase_3_bayesian_sem.py`: Simulates financial outcomes based on traits  
- `phase_4_bayesian_sem.py`: Builds and samples the Bayesian model using PyMC  
- `phase_5_bayesian_sem.py`: Visualizes the posterior distributions  
- `outputs/`: Contains saved trace file and graphs  
- `data/`: Input dataset (`full_simulated_data.csv`)  

## Requirements

- Python 3.10+
- PyMC ≥ 4.0
- ArviZ ≥ 0.12
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn

Install with:

```bash
pip install pymc arviz pandas numpy matplotlib seaborn scikit-learn
