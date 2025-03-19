import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Number of simulations
n_samples = 100000

# Generate random variables X, Y, Z each taking values {1,2} with equal probability
X = np.random.choice([1, 2], size=n_samples)
Y = np.random.choice([1, 2], size=n_samples)
Z = np.random.choice([1, 2], size=n_samples)

# Compute derived random variables
W = X * Y * Z
V = X * Y + X * Z + Y * Z
U = X**2 + Y * Z

# Function to compute empirical PMF
def compute_pmf(data):
    pmf = Counter(data)
    for key in pmf:
        pmf[key] /= len(data)
    return pmf

# Compute and display PMFs
for var, name in zip([W, V, U], ['W', 'V', 'U']):
    pmf = compute_pmf(var)
    
    plt.bar(pmf.keys(), pmf.values(), width=0.4)
    plt.xlabel(name)
    plt.ylabel("Probability")
    plt.title(f"Empirical PMF of {name}")
    plt.xticks(sorted(pmf.keys()))
    #plt.show()
    
    print(f"PMF of {name}: {dict(sorted(pmf.items()))}\n")

# Compute expectations
E_W = np.mean(W)
E_V = np.mean(V)
E_U = np.mean(U)

print(W.shape, V.shape, U.shape)  # Should all be (n_samples,)

# Compute expectations of products
E_WV = np.mean(W * V)
E_WU = np.mean(W * U)
E_VU = np.mean(V * U)

print(f"E[WV] (Empirical) = {E_WV:.5f}, E[WV] (Analytical) = {18:.5f}")
print(f"E[WU] (Empirical) = {E_WU:.5f}, E[WU] (Analytical) = {9:.5f}")
print(f"E[VU] (Empirical) = {E_VU:.5f}, E[VU] (Analytical) = {21:.5f}")

Cov_WV = np.mean(W * V) - np.mean(W) * np.mean(V)
Cov_WU = np.mean(W * U) - np.mean(W) * np.mean(U)
Cov_VU = np.mean(V * U) - np.mean(V) * np.mean(U)

# Print covariance results
print(f"Empirical Cov(W, V) = {Cov_WV:.4f}")
print(f"Empirical Cov(W, U) = {Cov_WU:.4f}")
print(f"Empirical Cov(V, U) = {Cov_VU:.4f}")
