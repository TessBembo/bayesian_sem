import pandas as pd
import numpy as np

# Load data from Phase 2
df = pd.read_csv("data/behavioral_indicators.csv")

n_days = 100
initial_price = 100

# 1. Simulate fake market: random walk with volatility
daily_returns = np.random.normal(loc=0.0005, scale=0.02, size=n_days)
price_path = initial_price * np.cumprod(1 + daily_returns)

# 2. Simulate outcomes for each agent
profits = []
exit_days = []
reaction_times = []

for _, row in df.iterrows():
    panic = row["panic_response"]
    risk = row["risk_preference"]
    avoid = row["volatility_avoidance"]

    # Market exit logic: exit earlier if panic is high
    exit_day = int(np.clip(n_days * (1 - panic + np.random.normal(0, 0.05)), 10, n_days))
    exit_days.append(exit_day)

    # Simulate investment outcome
    entry_price = price_path[0]
    exit_price = price_path[exit_day - 1]
    volatility_penalty = avoid * np.std(price_path[:exit_day])
    gain = (exit_price - entry_price) - volatility_penalty
    profits.append(gain)

    # Reaction time: how quickly the agent reacts to market events
    reaction_time = np.clip(1 + 5 * (1 - risk) + np.random.normal(0, 0.5), 1, 10)
    reaction_times.append(reaction_time)

# 3. Add outcomes to DataFrame
df["net_profit"] = profits
df["market_exit_day"] = exit_days
df["reaction_time"] = reaction_times

# 4. Save for Phase 4
df.to_csv("data/full_simulated_data.csv", index=False)

# Optional: quick peek
print(df.head())
