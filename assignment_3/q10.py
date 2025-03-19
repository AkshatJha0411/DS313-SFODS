import numpy as np
import matplotlib.pyplot as plt

# Inverse CDF Sampling
def sample_X(n):
    u = np.random.uniform(0, 1, n)  # Uniform samples
    samples = np.zeros(n)
    
    # CDF thresholds (computed correctly)
    cdf_1 = 2*(0.5**2)/3  # P(X < 0.5)
    cdf_2 = 1 - np.exp(-0.75*1.5)  # P(X < 1.5)
    cdf_3 = 1 - 0.05  # P(X < 2)
    
    # Apply inverse transform sampling based on given CDF
    mask1 = (u < cdf_1)
    mask2 = (u >= cdf_1) & (u < cdf_2)
    mask3 = (u >= cdf_2) & (u < cdf_3)
    mask4 = (u >= cdf_3)
    
    samples[mask1] = np.sqrt(3*u[mask1]/2)  # Solving for x in F_X(x)
    samples[mask2] = -np.log(1 - (u[mask2] - cdf_1)) / 0.75  # Exponential part
    samples[mask3] = 2 * (u[mask3] - cdf_2) + 1.5  # Linear part
    samples[mask4] = 2  # Values at x >= 2
    
    return samples

# Generate samples
n_samples = 100000  # Increased sample size
samples = sample_X(n_samples)

# Plot histogram and compare with analytical pdf
x_vals = np.linspace(0, 2, 1000)
f_x_vals = np.piecewise(x_vals, 
                         [x_vals < 0, (x_vals >= 0) & (x_vals < 0.5), 
                          (x_vals >= 0.5) & (x_vals < 1.5), (x_vals >= 1.5) & (x_vals < 2), x_vals >= 2],
                         [0, lambda x: 4*x/3, lambda x: 0.75*np.exp(-0.75*x), 0.5, 0])

plt.figure(figsize=(10,5))
plt.hist(samples, bins=100, density=True, alpha=0.6, label='Simulated')  # Increased bins
plt.plot(x_vals, f_x_vals, 'r-', linewidth=2, label='Analytical pdf')  # Thicker line for visibility
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Histogram of Simulated Samples vs. Analytical PDF')
plt.legend()
plt.show()
