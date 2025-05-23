import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  

# Set seed for reproducibility
np.random.seed(42)

# Number of agents
n_agents = 500

# 1. Simulate latent trait theta for each agent
theta = np.random.normal(loc=0, scale=1, size=n_agents)  # θ ~ N(0,1)

# 2. Define behavior modifier: e.g., volatility sensitivity as a function of theta
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

volatility_sensitivity = sigmoid(-theta)  # Inverse: lower theta = higher sensitivity

# 3. Plot trait distribution
plt.figure(figsize=(10, 4))
sns.histplot(theta, bins=30, kde=True, color='skyblue')
plt.title("Latent Trait Distribution (θ: Holistic → Analytic)", fontsize=13)
plt.xlabel("Trait Score (θ)")
plt.ylabel("Agent Count")
plt.show()

# 4. Plot trait vs. behavior
plt.figure(figsize=(8, 5))
plt.scatter(theta, volatility_sensitivity, alpha=0.6, color='purple')
plt.title("Trait vs. Volatility Sensitivity")
plt.xlabel("Latent Trait (θ)")
plt.ylabel("Volatility Sensitivity")
plt.grid(True)
plt.show()
