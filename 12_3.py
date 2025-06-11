import numpy as np
import matplotlib.pyplot as plt

# 格點設定
Nr = 10
Ntheta = 10

r_min, r_max = 0.5, 1.0
theta_min, theta_max = 0, np.pi / 3

dr = (r_max - r_min) / Nr
dtheta = (theta_max - theta_min) / Ntheta

r = np.linspace(r_min, r_max, Nr + 1)
theta = np.linspace(theta_min, theta_max, Ntheta + 1)

# 內部點數量
n = Nr - 1
m = Ntheta - 1
size = n * m

# 建立 A 矩陣與 F 向量
A = np.zeros((size, size))
F = np.zeros(size)

for i in range(1, Nr):
    for j in range(1, Ntheta):
        index = (i - 1) * m + (j - 1)
        ri = r[i]
        alpha = (ri * dtheta / dr) ** 2

        A[index, index] = -2 * (1 + alpha)
        if i > 1:
            A[index, index - m] = 1
        else:
            F[index] -= 1 * 50  # T = 50 at r = 0.5

        if i < Nr - 1:
            A[index, index + m] = 1
        else:
            F[index] -= 1 * 100  # T = 100 at r = 1.0

        if j > 1:
            A[index, index - 1] = alpha
        else:
            F[index] -= alpha * 0  # T = 0 at theta = 0

        if j < Ntheta - 1:
            A[index, index + 1] = alpha
        else:
            F[index] -= alpha * 0  # T = 0 at theta = pi/3

# 解線性系統 AU = F
U = np.linalg.solve(A, F)

# 重組解至 T 陣列
T = np.zeros((Nr + 1, Ntheta + 1))
for i in range(1, Nr):
    for j in range(1, Ntheta):
        idx = (i - 1) * m + (j - 1)
        T[i, j] = U[idx]

# 邊界條件
T[0, :] = 50
T[Nr, :] = 100
T[:, 0] = 0
T[:, Ntheta] = 0

# 建立極座標網格與轉換為直角座標
R, Theta = np.meshgrid(r, theta)   
# X = R * np.cos(Theta)
# Y = R * np.sin(Theta)
T = np.array(T).T 

# 繪製 3D 曲面圖
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, Theta, T, cmap='viridis')

ax.set_box_aspect([1, 1, 0.5])
ax.view_init(elev=30, azim=120)
ax.set_title('Temperature Distribution T(r, θ) via Gauss Elimination')
ax.set_xlabel('r')
ax.set_ylabel('θ')
ax.set_zlabel('T(r,θ)')
plt.tight_layout()
plt.show()
