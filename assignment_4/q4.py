import numpy as np
import matplotlib.pyplot as plt

# (a) Transformation Y = X^3:
X = np.random.uniform(-2*np.pi, 2*np.pi, 10**6)
Y_a = X**3

plt.figure(figsize=(8,4))
plt.hist(Y_a, bins=100, density=True, alpha=0.7, color='skyblue')
plt.title("Histogram of Y = X^3")
plt.xlabel("y")
plt.ylabel("Density")
plt.show()

# (b) Transformation Y = X^4:
Y_b = X**4

plt.figure(figsize=(8,4))
plt.hist(Y_b, bins=100, density=True, alpha=0.7, color='lightgreen')
plt.title("Histogram of Y = X^4")
plt.xlabel("y")
plt.ylabel("Density")
plt.show()

# (c) Transformation Y = 2*sin(3X+0.698):
Y_c = 2*np.sin(3*X + 0.698)

plt.figure(figsize=(8,4))
plt.hist(Y_c, bins=100, density=True, alpha=0.7, color='salmon')
plt.title("Histogram of Y = 2*sin(3X+0.698)")
plt.xlabel("y")
plt.ylabel("Density")
plt.show()
