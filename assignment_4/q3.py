import numpy as np
import matplotlib.pyplot as plt

# Parameters:
lam = 1.5  # for example, lambda value
n = 10**6

# Generate Exp(lambda) samples:
X_exp = np.random.exponential(scale=1/lam, size=n)

# Define Y = exp(X) and Z = min(X,3)
Y = np.exp(X_exp)
Z = np.minimum(X_exp, 3)

# Empirical CDF for Y:
y_vals = np.linspace(0.9, np.max(Y), 1000)
cdf_Y = [np.mean(Y <= y) for y in y_vals]

plt.figure(figsize=(8,4))
plt.plot(y_vals, cdf_Y, label="Empirical CDF of Y")
plt.title("CDF of Y = exp(X)")
plt.xlabel("y")
plt.ylabel("F_Y(y)")
plt.legend()
plt.grid(True)
plt.show()

# Empirical CDF for Z:
z_vals = np.linspace(0, 4, 1000)
cdf_Z = [np.mean(Z <= z) for z in z_vals]

plt.figure(figsize=(8,4))
plt.plot(z_vals, cdf_Z, label="Empirical CDF of Z")
plt.title("CDF of Z = min(X,3)")
plt.xlabel("z")
plt.ylabel("F_Z(z)")
plt.legend()
plt.grid(True)
plt.show()
