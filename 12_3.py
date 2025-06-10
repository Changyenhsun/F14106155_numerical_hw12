import numpy as np
import matplotlib.pyplot as plt

Nr = 10 
Ntheta = 10  

r_min = 0.5
r_max = 1.0
theta_min = 0
theta_max = np.pi / 3

dr = (r_max - r_min) / Nr
dtheta = (theta_max - theta_min) / Ntheta

r = np.linspace(r_min, r_max, Nr + 1)
theta = np.linspace(theta_min, theta_max, Ntheta + 1)

T = np.zeros((Nr + 1, Ntheta + 1))

T[0, :] = 50            # T(r=0.5, theta)
T[Nr, :] = 100          # T(r=1.0, theta)
T[:, 0] = 0             # T(r, theta=0)
T[:, Ntheta] = 0        # T(r, theta=pi/3)

max_iter = 10000
tolerance = 1e-6

for iteration in range(max_iter):
    T_old = T.copy()
    for i in range(1, Nr):
        for j in range(1, Ntheta):
            ri = r[i]
            alpha = (ri * dtheta / dr)**2

            T[i, j] = 1 / (2 * (1 + alpha)) * (
                T_old[i + 1, j] + T_old[i - 1, j] +
                alpha * (T_old[i, j + 1] + T_old[i, j - 1])
            )

    if np.max(np.abs(T - T_old)) < tolerance:
        print(f"Converged at iteration {iteration}")
        break

R, Theta = np.meshgrid(r, theta, indexing='ij')
X = R * np.cos(Theta)
Y = R * np.sin(Theta)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, T, cmap='viridis')

ax.set_box_aspect([1, 1, 0.5])  # x:y:z 的比例
ax.view_init(elev=30, azim=120)
ax.set_title('Temperature Distribution T(r, θ)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.show()

