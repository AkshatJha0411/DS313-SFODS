import numpy as np
import matplotlib.pyplot as plt

N = 1000
sw_f1 = 0  # Count of working server given first attempt failed
f1 = 0  # Count of first attempt failures
sw_f1_f2 = 0  # Count of working server given first and second attempts failed
f1_f2 = 0  # Count of first and second failures

for _ in range(N):
    server_working = np.random.choice([True, False], p=[0.8, 0.2])
    first_fail = not server_working or np.random.rand() < 0.1
    if first_fail:
        f1 += 1
        if server_working:
            sw_f1 += 1
        second_fail = not server_working or np.random.rand() < 0.1
        if second_fail:
            f1_f2 += 1
            if server_working:
                sw_f1_f2 += 1

p_f1 = f1 / N
p_sw_f1 = sw_f1 / f1 if f1 > 0 else 0
p_f2_f1 = f1_f2 / f1 if f1 > 0 else 0
p_sw_f1_f2 = sw_f1_f2 / f1_f2 if f1_f2 > 0 else 0

calculated_values = [0.28, 0.2857, 0.7429, 0.0385]
simulated_values = [p_f1, p_sw_f1, p_f2_f1, p_sw_f1_f2]
labels = ["P(F1)", "P(SW | F1)", "P(F2 | F1)", "P(SW | F1, F2)"]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, calculated_values, width, label='Calculated', color='blue')
rects2 = ax.bar(x + width/2, simulated_values, width, label='Simulated', color='orange')

ax.set_ylabel("Probability")
ax.set_title("Comparison of Calculated vs Simulated Probabilities")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.ylim(0, 1)
plt.show()

print(f"Simulated P(F1) = {p_f1:.4f}")
print(f"Simulated P(SW | F1) = {p_sw_f1:.4f}")
print(f"Simulated P(F2 | F1) = {p_f2_f1:.4f}")
print(f"Simulated P(SW | F1, F2) = {p_sw_f1_f2:.4f}")
