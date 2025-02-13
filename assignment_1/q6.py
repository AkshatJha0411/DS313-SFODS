import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Simulation parameters
num_trials = 1000000

# Probability mappings
def generate_register_mapping_a():
    return [np.random.choice([0, 1], p=[2**(-i) / (2**(-i) + (1 - 2**(-i))), 1 - 2**(-i) / (2**(-i) + (1 - 2**(-i)))]) for i in range(1, 9)]

def generate_register_mapping_b(p_ev=0.6, p_od=0.4):
    return [np.random.choice([0, 1], p=[p_ev, 1 - p_ev]) if i % 2 == 0 else np.random.choice([0, 1], p=[p_od, 1 - p_od]) for i in range(8)]

def check_A1(register):
    return all(register[i] != register[i+1] for i in range(len(register) - 1))

def check_A2(register):
    return sum(register) == 4

def check_A3(register):
    target = [0, 1, 1, 0, 0, 1, 1, 0]
    for i in range(8):
        if register[i:] + register[:i] == target:
            return True
    return False

# Monte Carlo simulation
def simulate(mapping_func):
    A1_count, A2_count, A3_count, A1_A3_count, A2_A3_count = 0, 0, 0, 0, 0
    
    for _ in range(num_trials):
        reg = mapping_func()
        A1, A2, A3 = check_A1(reg), check_A2(reg), check_A3(reg)
        
        if A1:
            A1_count += 1
        if A2:
            A2_count += 1
        if A3:
            A3_count += 1
        if A1 and A3:
            A1_A3_count += 1
        if A2 and A3:
            A2_A3_count += 1
    
    P_A1, P_A2, P_A3 = A1_count / num_trials, A2_count / num_trials, A3_count / num_trials
    P_A1_given_A3 = A1_A3_count / A3_count if A3_count > 0 else 0
    P_A2_given_A3 = A2_A3_count / A3_count if A3_count > 0 else 0
    
    return P_A1, P_A2, P_A3, P_A1_given_A3, P_A2_given_A3

# Run simulation for both mappings
results_a = simulate(generate_register_mapping_a)
results_b = simulate(generate_register_mapping_b)

# Print results
print("Mapping A:", results_a)
print("Mapping B:", results_b)

# Plotting for Mapping A
labels = ["P(A1)", "P(A2)", "P(A3)", "P(A1|A3)", "P(A2|A3)"]
x = np.arange(len(labels))

fig, ax = plt.subplots()
ax.bar(x, results_a, width=0.5, label='Mapping A', color='blue')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Probability")
ax.set_title("Monte Carlo Simulation - Mapping A")
ax.legend()
plt.show()

# Plotting for Mapping B
fig, ax = plt.subplots()
ax.bar(x, results_b, width=0.5, label='Mapping B', color='green')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Probability")
ax.set_title("Monte Carlo Simulation - Mapping B")
ax.legend()
plt.show()
