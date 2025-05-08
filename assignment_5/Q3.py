import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)

def analytical_pdf_y_given_x_ge_alpha(y, alpha):
    if y < alpha**2:
        return 0
    else:
        return np.exp(-(np.sqrt(y) - alpha)) / (2 * np.sqrt(y))

def simulate_and_plot(alpha, n_samples=10**6, bins=100):
    x_samples = np.random.exponential(scale=1.0, size=n_samples)
    x_filtered = x_samples[x_samples >= alpha]
    y_filtered = x_filtered ** 2

    plt.figure(figsize=(8, 5))
    sns.histplot(y_filtered, bins=bins, stat='density', label='Simulated', kde=False, color='skyblue')
    y_vals = np.linspace(alpha**2, np.percentile(y_filtered, 99), 500)
    f_vals = [analytical_pdf_y_given_x_ge_alpha(y, alpha) for y in y_vals]
    plt.plot(y_vals, f_vals, label='Analytical', color='red', linewidth=2)

    plt.title(f'Conditional PDF f_Y(y | X ≥ {alpha})')
    plt.xlabel('y')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    filename = f"plots/conditional_pdf_alpha_{alpha}.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Alpha: {alpha}, Plot saved to {filename}")
    return y_filtered

# Call the function
alphas = [0, 0.5, 1, 1.5, 2]
for alpha in alphas:
    simulate_and_plot(alpha)
