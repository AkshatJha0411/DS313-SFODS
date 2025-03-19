import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters
n = 10  # Number of trials
p = 0.5  # Success probability
num_samples = 10000  # Number of samples

# Generate binomial samples
X = np.random.binomial(n, p, num_samples)
Y = np.random.binomial(n, p, num_samples)
Z = X + Y  # Sum of two independent Bin(n, p) variables

# Compute empirical probabilities
unique, counts = np.unique(Z, return_counts=True)
empirical_probs = counts / num_samples

# Theoretical binomial probabilities
k_values = np.arange(0, 2*n + 1)
theoretical_probs = binom.pmf(k_values, 2*n, p)

# Plot results
plt.bar(unique, empirical_probs, alpha=0.6, label='Empirical', color='blue')
plt.plot(k_values, theoretical_probs, 'ro-', label='Theoretical', markersize=5)
plt.xlabel('Sum (Z)')
plt.ylabel('Probability')
plt.title(f'Empirical vs Theoretical Binomial Distribution (n={n}, p={p})')
plt.legend()
plt.grid()
plt.show()