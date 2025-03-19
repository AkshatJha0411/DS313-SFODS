import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
lambda_poisson = 3  # Mean of Poisson distribution
p_geom = 0.5  # Probability for Geometric distribution
num_samples = 10000  # Number of samples

# Generate random samples
X_samples = np.random.poisson(lambda_poisson, num_samples)
Y_samples = np.random.geometric(p_geom, num_samples) - 1  # Adjust to match definition

# Compute Z = X + Y
Z_samples = X_samples + Y_samples

# Compute empirical probabilities
values, counts = np.unique(Z_samples, return_counts=True)
empirical_probs = counts / num_samples

# Compute theoretical probabilities
max_n = max(values)
theoretical_probs = []
for n in range(max_n + 1):
    prob_n = sum(stats.poisson.pmf(k, lambda_poisson) * stats.geom.pmf(n - k + 1, p_geom)
                 for k in range(n + 1))
    theoretical_probs.append(prob_n)

# Plot results
plt.figure(figsize=(10, 5))
plt.bar(values, empirical_probs, alpha=0.6, label="Simulated", color='blue')
plt.plot(range(max_n + 1), theoretical_probs, 'ro-', markersize=4, label="Theoretical", color='red')
plt.xlabel("Z = X + Y")
plt.ylabel("Probability")
plt.title("Comparison of Empirical and Theoretical Distribution")
plt.legend()
plt.show()