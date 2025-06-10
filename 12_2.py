import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

dr = 0.1
dt = 0.5  #dt = 0.002（穩定）
K = 0.1
alpha = 1 / (4 * K)
r = np.arange(0.5, 1.0 + dr, dr)
t = np.arange(0, 10 + dt, dt)
Nr = len(r)
Nt = len(t)
λ = alpha * dt / dr**2

T_init = 200 * (r - 0.5)


# (a) Forward Difference
Tf = np.zeros((Nt, Nr))
Tf[0, :] = T_init

for n in range(0, Nt - 1):
    for i in range(1, Nr - 1):
        Tf[n + 1, i] = Tf[n, i] + λ * (
            (Tf[n, i + 1] - 2 * Tf[n, i] + Tf[n, i - 1]) +
            (dr / r[i]) * (Tf[n, i + 1] - Tf[n, i - 1]) / 2
        )
    # 邊界條件
    Tf[n + 1, -1] = 100 + 40 * t[n + 1]  # T(1, t)
    Tf[n + 1, 0] = (Tf[n + 1, 1]) / (1 + 3 * dr)  # Robin condition


# (b) Backward Difference
Tb = np.zeros((Nt, Nr))
Tb[0, :] = T_init

A = np.zeros((Nr, Nr))
for i in range(1, Nr - 1):
    ri = r[i]
    A[i, i - 1] = -λ * (1 - dr / (2 * ri))
    A[i, i] = 1 + 2 * λ
    A[i, i + 1] = -λ * (1 + dr / (2 * ri))
A[0, 0] = 1 + 3 * dr
A[0, 1] = -1
A[-1, -1] = 1

for n in range(0, Nt - 1):
    b = Tb[n, :].copy()
    b[0] = 0
    b[-1] = 100 + 40 * t[n + 1]
    Tb[n + 1, :] = solve(A, b)

# (c) Crank-Nicolson Method
Tc = np.zeros((Nt, Nr))
Tc[0, :] = T_init

Ac = np.zeros((Nr, Nr))
Bc = np.zeros((Nr, Nr))
for i in range(1, Nr - 1):
    ri = r[i]
    Ac[i, i - 1] = -λ / 2 * (1 - dr / (2 * ri))
    Ac[i, i] = 1 + λ
    Ac[i, i + 1] = -λ / 2 * (1 + dr / (2 * ri))
    Bc[i, i - 1] = λ / 2 * (1 - dr / (2 * ri))
    Bc[i, i] = 1 - λ
    Bc[i, i + 1] = λ / 2 * (1 + dr / (2 * ri))
Ac[0, 0], Ac[0, 1] = 1 + 3 * dr, -1
Ac[-1, -1] = 1
Bc[0, 0], Bc[0, 1] = 1 + 3 * dr, -1
Bc[-1, -1] = 1

for n in range(0, Nt - 1):
    b = Bc @ Tc[n, :]
    b[0] = 0
    b[-1] = 100 + 40 * t[n + 1]
    Tc[n + 1, :] = solve(Ac, b)

# 繪圖比較
r_grid, t_grid = np.meshgrid(r, t)

fig = plt.figure(figsize=(18, 5))
gs = gridspec.GridSpec(1, 3)

methods = [Tf, Tb, Tc]
titles = ['(a) Forward Difference', '(b) Backward Difference', '(c) Crank-Nicolson']

for i in range(3):
    ax = fig.add_subplot(gs[0, i], projection='3d')
    ax.plot_surface(r_grid, t_grid, methods[i], cmap='viridis')
    ax.set_title(titles[i])
    ax.set_xlabel('r')
    ax.set_ylabel('t')
    ax.set_zlabel('T')

plt.tight_layout()
plt.show()
