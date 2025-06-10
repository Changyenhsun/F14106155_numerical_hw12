import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
h = k = 0.1 * pi
Nx = int(pi / h)  
Ny = int((pi / 2) / k)  

x = np.linspace(0, pi, Nx + 1)
y = np.linspace(0, pi / 2, Ny + 1)

u = np.zeros((Nx + 1, Ny + 1))


for j in range(Ny + 1):
    u[0, j] = np.cos(y[j])           # u(0, y) = cos(y)
    u[Nx, j] = -np.cos(y[j])         # u(pi, y) = -cos(y)

for i in range(Nx + 1):
    u[i, 0] = np.cos(x[i])           # u(x, 0) = cos(x)
    u[i, Ny] = 0                     # u(x, pi/2) = 0

max_iter = 10000
tolerance = 1e-6

for iteration in range(max_iter):
    u_old = u.copy()
    for i in range(1, Nx):
        for j in range(1, Ny):
            fxy = x[i] * y[j]
            u[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] +
                             u_old[i, j+1] + u_old[i, j-1] -
                             h**2 * fxy)

    # 收斂判斷
    if np.max(np.abs(u - u_old)) < tolerance:
        print(f'Converged at iteration {iteration}')
        break

X, Y = np.meshgrid(x, y, indexing='ij')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_title('Solution of u(x, y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.show()
