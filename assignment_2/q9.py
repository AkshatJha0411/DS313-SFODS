import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
n = 20   # Total number of customers
p = 0.3  # Probability of a customer needing a connection
c = 10   # Number of available modems
num_trials = 100000  # Number of Monte Carlo simulations

# Analytical computation
prob_exceeding_c = 1 - sum(stats.binom.pmf(k, n, p) for k in range(c+1))

# Monte Carlo Simulation
exceed_count = 0
for _ in range(num_trials):
    connections_needed = np.random.binomial(n, p)  # Simulate one scenario
    if connections_needed > c:
        exceed_count += 1

# Estimated probability from simulation
simulated_prob = exceed_count / num_trials

# Print results
print(f"Analytical probability P(X > {c}): {prob_exceeding_c:.6f}")
print(f"Simulated probability P(X > {c}) (Monte Carlo): {simulated_prob:.6f}")

# Plot the histogram
samples = np.random.binomial(n, p, num_trials)
plt.hist(samples, bins=np.arange(0, n+2)-0.5, density=True, alpha=0.6, color='b', edgecolor='black')
plt.axvline(c + 0.5, color='r', linestyle='dashed', label=f"Threshold c = {c}")
plt.title(f"Histogram of Connections Needed (n={n}, p={p})")
plt.xlabel("Number of Customers Needing Connection")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
