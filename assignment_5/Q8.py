import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the joint pdf f_UV(u, v)
def f_UV(u, v):
    return 9 * (u ** 2) * (v ** 2)

# Define the joint pdf f_XY(x, y)
def f_XY(x, y):
    if 0 <= x <= 3 and 0 <= y <= x / 3:
        return (9 * y ** 2) / x
    else:
        return 0

# Create grids for U and V
u = np.linspace(0, 1, 100)
v = np.linspace(0, 1, 100)
U, V = np.meshgrid(u, v)
Z_UV = f_UV(U, V)

# Create grids for X and Y
x = np.linspace(0, 3, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z_XY = np.zeros_like(X)
for i in range(100):
    for j in range(100):
        Z_XY[i, j] = f_XY(X[i, j], Y[i, j])

# Plot f_UV(u, v)
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(U, V, Z_UV, cmap='viridis')
ax1.set_title('Joint PDF $f_{UV}(u, v)$')
ax1.set_xlabel('u')
ax1.set_ylabel('v')
ax1.set_zlabel('$f_{UV}(u, v)$')

# Plot f_XY(x, y)
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_XY, cmap='viridis')
ax2.set_title('Joint PDF $f_{XY}(x, y)$')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('$f_{XY}(x, y)$')

plt.tight_layout()
plt.savefig('joint_pdfs_plot.png')
plt.show()