import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm

n = 10**6

# For exponential distribution:
lam = 2.0
X_exp = np.random.exponential(scale=1/lam, size=n)
F_X_exp = 1 - np.exp(-lam*X_exp)
plt.figure(figsize=(8,4))
plt.hist(F_X_exp, bins=50, density=True, alpha=0.7, label="Exp Transform")
plt.plot(np.linspace(0,1,100), np.ones(100), 'r--', label='Unif(0,1)')
plt.title("Probability Integral Transform for Exponential")
plt.xlabel("y")
plt.ylabel("Density")
plt.legend()
plt.show()

# For Gaussian distribution:
mu, sigma = 5, 2
X_norm = np.random.normal(loc=mu, scale=sigma, size=n)
F_X_norm = norm.cdf(X_norm, loc=mu, scale=sigma)
plt.figure(figsize=(8,4))
plt.hist(F_X_norm, bins=50, density=True, alpha=0.7, label="Gaussian Transform")
plt.plot(np.linspace(0,1,100), np.ones(100), 'r--', label='Unif(0,1)')
plt.title("Probability Integral Transform for Gaussian")
plt.xlabel("y")
plt.ylabel("Density")
plt.legend()
plt.show()
