import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def simulate_dice_rolls(num_trials=100000):
    outcomes = [(np.random.randint(1, 7), np.random.randint(1, 7)) for _ in range(num_trials)]
    return outcomes

def compute_joint_pmf(outcomes, case):
    counts = Counter()
    for a, b in outcomes:
        if case == 'a':
            x, y = max(a, b), a + b
        elif case == 'b':
            x, y = a, max(a, b)
        elif case == 'c':
            x, y = min(a, b), max(a, b)
        counts[(x, y)] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def theoretical_pmf(case):
    pmf = {}
    for a in range(1, 7):
        for b in range(1, 7):
            if case == 'a':
                x, y = max(a, b), a + b
                count = 1 if y == 2*x else 2 if x < y/2 <= x and y <= 2*x else 0
            elif case == 'b':
                x, y = a, max(a, b)
                count = y if x == y else 1 if x < y else 0
            elif case == 'c':
                x, y = min(a, b), max(a, b)
                count = 1 if x == y else 2 if x < y else 0
            pmf[(x, y)] = count / 36
    return pmf

def plot_pmf(simulated_pmf, theoretical_pmf, case):
    x_vals = sorted(set(k[0] for k in simulated_pmf.keys()))
    y_vals = sorted(set(k[1] for k in simulated_pmf.keys()))
    sim_matrix = np.zeros((len(x_vals), len(y_vals)))
    theo_matrix = np.zeros((len(x_vals), len(y_vals)))
    
    for (x, y), p in simulated_pmf.items():
        sim_matrix[x_vals.index(x), y_vals.index(y)] = p
    for (x, y), p in theoretical_pmf.items():
        theo_matrix[x_vals.index(x), y_vals.index(y)] = p
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(sim_matrix, xticklabels=y_vals, yticklabels=x_vals, ax=axes[0], cmap='Blues', annot=True)
    axes[0].set_title(f'Simulated PMF for Case {case}')
    axes[0].set_xlabel('Y values')
    axes[0].set_ylabel('X values')
    
    sns.heatmap(theo_matrix, xticklabels=y_vals, yticklabels=x_vals, ax=axes[1], cmap='Reds', annot=True)
    axes[1].set_title(f'Theoretical PMF for Case {case}')
    axes[1].set_xlabel('Y values')
    axes[1].set_ylabel('X values')
    
    plt.show()

def main():
    num_trials = 100000
    outcomes = simulate_dice_rolls(num_trials)
    for case in ['a', 'b', 'c']:
        sim_pmf = compute_joint_pmf(outcomes, case)
        theo_pmf = theoretical_pmf(case)
        plot_pmf(sim_pmf, theo_pmf, case)

if __name__ == "__main__":
    main()
