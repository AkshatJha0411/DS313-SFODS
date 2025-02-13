import numpy as np
import matplotlib.pyplot as plt

N = 100000
fc_h2 = 0
h2 = 0

for _ in range(N):
    fc = np.random.choice([True, False])
    if fc:
        t = np.random.choice(["H", "T"], size=2, p=[0.5, 0.5])
    else:
        t = ["H", "H"]
    if t[0] == "H" and t[1] == "H":
        h2 += 1
        if fc:
            fc_h2 += 1

p_est = fc_h2 / h2
labels = ["Analytical", "Simulated"]
values = [0.2, p_est]

plt.bar(labels, values, color=['blue', 'orange'])
plt.ylim(0, 0.3)
plt.ylabel("Probability")
plt.title("P(Fair Coin | Two Heads)")
plt.show()

print(f"Simulated probability: {p_est:.4f}")