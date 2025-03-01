import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------------
# Part (i): Negative Binomial Distribution
# ---------------------------

# Parameters for simulation
p = 0.3       # probability of heads
k = 5         # number of heads required
num_experiments = 100000

# Simulate the number of flips needed to get k heads
flips_count = []
for i in range(num_experiments):
    count = 0
    heads = 0
    while heads < k:
        count += 1
        if np.random.rand() < p:
            heads += 1
    flips_count.append(count)

# Obtain simulation distribution
unique, counts = np.unique(flips_count, return_counts=True)
sim_prob = counts / num_experiments

# Compute analytical probabilities for n from k to the maximum number observed
n_values = np.arange(k, max(unique) + 1)
analytical_prob = [math.comb(n - 1, k - 1) * (p ** k) * ((1 - p) ** (n - k)) for n in n_values]

# Plotting for Part (i)
plt.figure(figsize=(10, 5))
plt.bar(unique, sim_prob, alpha=0.6, label="Simulation", width=0.8, color='skyblue')
plt.plot(n_values, analytical_prob, 'ro-', label="Analytical", markersize=5)
plt.xlabel("Number of flips")
plt.ylabel("Probability")
plt.title("Negative Binomial Distribution: Number of flips to obtain {} heads".format(k))
plt.legend()
plt.show()

# ---------------------------
# Part (ii): Distribution of X = Heads - Tails
# ---------------------------

n = 50  # Number of coin tosses in each experiment
X_values = [] 

# Simulate n tosses and compute X for each experiment
for i in range(num_experiments):
    tosses = np.random.rand(n) < p
    heads = np.sum(tosses)
    # Alternatively, X = 2*heads - n
    X = heads - (n - heads)
    X_values.append(X)

# Obtain simulation distribution for X
unique_X, counts_X = np.unique(X_values, return_counts=True)
sim_prob_X = counts_X / num_experiments

# Calculate the analytical PMF for X
# X can only take even-spaced values: -n, -n+2, ..., n
x_possible = np.arange(-n, n + 1, 2)
analytical_prob_X = []
for x in x_possible:
    h = (x + n) // 2  # derived from x = 2h - n
    prob = math.comb(n, h) * (p ** h) * ((1 - p) ** (n - h))
    analytical_prob_X.append(prob)

# Plotting for Part (ii)
plt.figure(figsize=(10, 5))
plt.bar(unique_X, sim_prob_X, alpha=0.6, label="Simulation", width=1.0, color='lightgreen')
plt.plot(x_possible, analytical_prob_X, 'ro-', label="Analytical", markersize=5)
plt.xlabel("X (Heads - Tails)")
plt.ylabel("Probability")
plt.title("Distribution of X = Heads - Tails for {} coin tosses".format(n))
plt.legend()
plt.show()
