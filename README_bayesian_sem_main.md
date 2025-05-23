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
- Trait loadings (`lambda_*`)
- Regression paths to outcomes (`beta_*`)

### Example Output: Structural Coefficient Forest Plot

![Structural Forest Plot](outputs/forest_structural.png)

> This plot shows the 94% highest density intervals (HDIs) for how the latent trait θ affects each outcome. Intervals that exclude zero suggest strong directional effects.

## File Structure

- `phase_1_2_generate_traits.py`: Simulates the latent trait and behavioral indicators  
- `phase_3_generate_outcomes.py`: Simulates financial outcomes based on traits  
- `phase_4_bayesian_sem.py`: Builds and samples the Bayesian model using PyMC  
- `phase_5_bayesian_sem.py`: Visualizes the posterior distributions  
- `outputs/`: Contains saved trace file and graphs  
- `data/`: Input dataset (`full_simulated_data.csv`)  

## 🔧 Requirements

- Python 3.10+
- PyMC ≥ 4.0
- ArviZ ≥ 0.12
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn

Install with:

```bash
pip install pymc arviz pandas numpy matplotlib seaborn scikit-learn
