import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_samples = 1000000
np.random.seed(42)

# Generate samples
X = np.random.uniform(0, 1, num_samples)
Y = np.random.uniform(0, 1, num_samples)
Z = np.random.uniform(0, 1, num_samples)

# Probability that all roots are real
real_roots = Y**2 >= 4 * X * Z
p_real = np.mean(real_roots)
print(f"Probability that all roots are real: {p_real:.4f}")

# Given real roots, probability that at least one root > 1
X_real = X[real_roots]
Y_real = Y[real_roots]
Z_real = Z[real_roots]

# Compute roots
discriminant = Y_real**2 - 4 * X_real * Z_real
sqrt_disc = np.sqrt(discriminant)
root1 = (-Y_real + sqrt_disc) / (2 * X_real)
root2 = (-Y_real - sqrt_disc) / (2 * X_real)

# Check if any root > 1
any_root_gt_1 = (root1 > 1) | (root2 > 1)
p_gt_1_given_real = np.mean(any_root_gt_1)
print(f"Probability at least one root > 1 given real roots: {p_gt_1_given_real:.4f}")

# Plotting
labels = ['All roots real', 'At least one root > 1 given real']
analytical = [np.log(2)/6 + 1/18, 0.0]
simulated = [p_real, p_gt_1_given_real]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, analytical, width, label='Analytical')
rects2 = ax.bar(x + width/2, simulated, width, label='Simulated')

ax.set_ylabel('Probability')
ax.set_title('Comparison of Analytical and Simulated Probabilities')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.savefig('probabilities_plot.png')
plt.show()