import numpy as np
import matplotlib.pyplot as plt

# Simulation for X:
n = 10**6
U = np.random.rand(n)
X = np.where(U < 0.5, 0, U - 0.5)

# Compute the empirical CDF:
x_vals = np.linspace(-0.1, 0.6, 1000)
cdf_empirical = [np.mean(X <= x) for x in x_vals]

# Plot CDF:
plt.figure(figsize=(8,4))
plt.plot(x_vals, cdf_empirical, label='Empirical CDF')
plt.axvline(0, color='red', linestyle='--', label='x=0')
plt.title("Empirical CDF of X")
plt.xlabel("x")
plt.ylabel("F_X(x)")
plt.legend()
plt.grid(True)
plt.show()

# Check value at x=0
print("Empirical F_X(0) =", np.mean(X<=0))

# Characteristic function (empirical approximation):
u = 2.0  # choose a sample value for u
phi_empirical = np.mean(np.exp(1j*u*X))
phi_theoretical = 0.5 + (np.exp(1j*u*0.5)-1)/(1j*u)
print("Empirical Phi_X(u):", phi_empirical)
print("Theoretical Phi_X(u):", phi_theoretical)
