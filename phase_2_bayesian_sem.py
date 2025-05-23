import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Number of agents
n_agents = 500

# 1. Latent trait θ
theta = np.random.normal(0, 1, size=n_agents)

# 2. Generate observed indicators (linear relationships + noise)

# Panic Response: inverse relation to θ (holistic → more panic)
panic_response = 1 - (0.5 * theta + np.random.normal(0, 0.3, size=n_agents))

# Risk Preference: positive relation to θ (analytic → more risk)
risk_preference = 0.5 * theta + np.random.normal(0, 0.3, size=n_agents)

# Volatility Avoidance: inverse relation to θ
volatility_avoidance = 1 - (0.4 * theta + np.random.normal(0, 0.25, size=n_agents))

# Optional: Normalize to 0–1 range for clarity
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

panic_response = normalize(panic_response)
risk_preference = normalize(risk_preference)
volatility_avoidance = normalize(volatility_avoidance)

# 3. Assemble into DataFrame
df = pd.DataFrame({
    'theta': theta,
    'panic_response': panic_response,
    'risk_preference': risk_preference,
    'volatility_avoidance': volatility_avoidance
})

# 4. Plot relationships
plt.figure(figsize=(14, 4))
for i, col in enumerate(['panic_response', 'risk_preference', 'volatility_avoidance']):
    plt.subplot(1, 3, i+1)
    sns.scatterplot(x='theta', y=col, data=df, alpha=0.6)
    plt.title(f'Trait vs. {col.replace("_", " ").title()}')
    plt.xlabel("Latent Trait (θ)")
    plt.ylabel(col.replace("_", " ").title())
plt.tight_layout()
plt.show()
df.to_csv("data/behavioral_indicators.csv", index=False)
