import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm

# Create folder to save plots
os.makedirs("plots", exist_ok=True)

def simulate_box_muller(n_samples=10**6, bins=100):
    # Generate U ~ Unif(0, 2π)
    U = np.random.uniform(0, 2*np.pi, size=n_samples)
    
    # Generate Z ~ Exp(1)
    Z = np.random.exponential(scale=1.0, size=n_samples)
    
    # Transform to X, Y
    R = np.sqrt(2 * Z)
    X = R * np.cos(U)
    Y = R * np.sin(U)

    # Plot histogram for X
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(X, bins=bins, stat='density', label='Simulated X', color='skyblue', kde=False)
    x_vals = np.linspace(-4, 4, 500)
    plt.plot(x_vals, norm.pdf(x_vals), 'r-', label='Standard Normal PDF')
    plt.title('Distribution of X')
    plt.xlabel('X')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    # Plot histogram for Y
    plt.subplot(1, 2, 2)
    sns.histplot(Y, bins=bins, stat='density', label='Simulated Y', color='lightgreen', kde=False)
    plt.plot(x_vals, norm.pdf(x_vals), 'r-', label='Standard Normal PDF')
    plt.title('Distribution of Y')
    plt.xlabel('Y')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    filename = "plots/box_muller_xy_distribution.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    # Correlation check
    correlation = np.corrcoef(X, Y)[0, 1]
    print(f"Correlation between X and Y (should be ~0): {correlation:.5f}")

    return X, Y


X, Y = simulate_box_muller()
