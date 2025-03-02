import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import binom, poisson, geom

# Set random seed for reproducibility
np.random.seed(42)

### (a) Binomial Distribution ###

def binomial_inversion(N, p):
    """
    Generate a single Binomial(N, p) sample using the inversion method.
    """
    # Pre-compute PMF for k = 0,...,N
    pmf = [math.comb(N, k) * (p ** k) * ((1-p) ** (N-k)) for k in range(N+1)]
    cdf = np.cumsum(pmf)
    u = np.random.uniform()
    # Find the smallest k for which cdf[k] >= u
    for k, F in enumerate(cdf):
        if u <= F:
            return k
    return N  # fallback

# Parameters for Binomial
N_binom = 10
p_binom = 0.3
num_samples = 10000

# Generate samples
binom_samples = np.array([binomial_inversion(N_binom, p_binom) for _ in range(num_samples)])

# Theoretical PMF
k_vals = np.arange(0, N_binom+1)
theoretical_binom_pmf = [math.comb(N_binom, k) * (p_binom ** k) * ((1-p_binom) ** (N_binom-k)) for k in k_vals]

### (b) Poisson Distribution ###

def poisson_inversion(lam):
    """
    Generate a single Poisson(lam) sample using the inversion method (Knuth's algorithm).
    """
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        u = np.random.uniform()
        p *= u
        k += 1
    return k - 1

# Parameter for Poisson
lam = 4.0

# Generate samples
poisson_samples = np.array([poisson_inversion(lam) for _ in range(num_samples)])

# Theoretical PMF for Poisson over a suitable range
k_poisson = np.arange(0, np.max(poisson_samples)+1)
theoretical_poisson_pmf = [math.exp(-lam) * (lam ** k) / math.factorial(k) for k in k_poisson]

### (c) Geometric Distribution ###

def geometric_inversion(p):
    """
    Generate a single geometric sample (support: 1,2,...) using the inversion method.
    """
    u = np.random.uniform()
    # Solve for k in: 1 - (1-p)^k >= u  --> k >= log(1-u)/log(1-p)
    k = math.ceil(np.log(1-u) / np.log(1-p))
    return k

# Parameter for Geometric
p_geom = 0.3

# Generate samples
geometric_samples = np.array([geometric_inversion(p_geom) for _ in range(num_samples)])

# Theoretical PMF for Geometric (support: 1,2,...)
k_geom = np.arange(1, np.max(geometric_samples)+1)
theoretical_geom_pmf = [p_geom * ((1-p_geom)**(k-1)) for k in k_geom]

### Plotting the results ###

fig, axes = plt.subplots(3, 1, figsize=(8, 12))
fig.tight_layout(pad=4.0)

# Plot for Binomial
axes[0].hist(binom_samples, bins=np.arange(-0.5, N_binom+1.5, 1), density=True, alpha=0.6, color='skyblue', edgecolor='black')
axes[0].plot(k_vals, theoretical_binom_pmf, 'ro-', label='Theoretical PMF')
axes[0].set_title(f'Binomial Distribution (N={N_binom}, p={p_binom})')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Probability')
axes[0].legend()

# Plot for Poisson
axes[1].hist(poisson_samples, bins=np.arange(-0.5, np.max(poisson_samples)+1.5, 1), density=True, alpha=0.6, color='lightgreen', edgecolor='black')
axes[1].plot(k_poisson, theoretical_poisson_pmf, 'ro-', label='Theoretical PMF')
axes[1].set_title(f'Poisson Distribution (λ={lam})')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Probability')
axes[1].legend()

# Plot for Geometric
axes[2].hist(geometric_samples, bins=np.arange(0.5, np.max(geometric_samples)+1.5, 1), density=True, alpha=0.6, color='lightcoral', edgecolor='black')
axes[2].plot(k_geom, theoretical_geom_pmf, 'ro-', label='Theoretical PMF')
axes[2].set_title(f'Geometric Distribution (p={p_geom})')
axes[2].set_xlabel('k')
axes[2].set_ylabel('Probability')
axes[2].legend()

plt.show()
