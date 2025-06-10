import numpy as np
import matplotlib.pyplot as plt

dx = dt = 0.1
x = np.arange(0, 1 + dx, dx)
t = np.arange(0, 1 + dt, dt)

Nx = len(x)
Nt = len(t)
λ = dt / dx  # wave speed ratio

p = np.zeros((Nt, Nx))

p[0, :] = np.cos(2 * np.pi * x)

# 初始條件：∂p/∂t(x, 0) = 2π sin(2πx)
# 使用 central difference 初始化第一步 (j=1)
for i in range(1, Nx - 1):
    p[1, i] = (p[0, i] + dt * 2 * np.pi * np.sin(2 * np.pi * x[i]) +
               0.5 * λ**2 * (p[0, i + 1] - 2 * p[0, i] + p[0, i - 1]))

# 邊界條件
p[:, 0] = 1
p[:, -1] = 2

# 時間迭代（從 j=1 開始）
for j in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        p[j + 1, i] = (2 * (1 - λ**2) * p[j, i] +
                       λ**2 * (p[j, i + 1] + p[j, i - 1]) -
                       p[j - 1, i])

plt.figure(figsize=(8, 4))
for j in range(0, Nt, 2):
    plt.plot(x, p[j], label=f't={t[j]:.1f}')
plt.title("Wave Equation: p(x, t)")
plt.xlabel("x")
plt.ylabel("p")
plt.legend()
plt.grid(True)
plt.show()

# 建立 meshgrid for 3D plot
X, T = np.meshgrid(x, t)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, p, cmap='viridis')
ax.set_title("3D Surface Plot of p(x, t)")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("p")
plt.tight_layout()
plt.show()