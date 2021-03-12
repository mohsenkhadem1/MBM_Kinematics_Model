import numpy as np
from MBM import MBM_Model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import matplotlib.pyplot as plt
import time

start_time = time.time()
# Joint variables (pulling or pushing rods 1,2,4, and 5)
q = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
u0 = np.array([0.0, 0.0, 0.0, 0.0])
length_seg1 = 150e-3  # length of long rods
length_seg2 = 100e-3  # length of shorter rods
length_rod = 0e-3  # length of the inner shaft
# force on robot tip along x, y, and z direction
f = np.array([0, 0, 0]).reshape(3, 1)
# distributed force on robot tip along x, y, and z direction
w = np.array([0, 0, 0]).reshape(3, 1)
# mimimum step size for the solver
tol = 0.005
error = np.array([0])

MBM = MBM_Model(length_seg1, length_seg2, length_rod, w, f, q, tol)
u0 = MBM.minimize(u0)
x_a = MBM.r[-1, :].reshape(3, 1)  # actual position of the robot end-effector

# calculating initial Jacobian
eps = 0.000001
pos_init = x_a.reshape(3, )
Jac0 = np.empty((3, 5))
for j in range(0, 5):
    q[j] = q[j] + eps
    MBM.q = q[0:4]
    MBM.insert = q[4]
    MBM.minimize(np.array(u0))
    pos_new = MBM.r[-1, :].reshape(3, )
    Jac0[:, j] = (pos_new - pos_init) / eps
    q[j] = q[j] - eps

x_d = x_a + np.array([5e-3, 5e-3, 0]).reshape(3, 1)

K = 1 * np.identity(3)
dq = -np.linalg.pinv(Jac0) @ K @ (x_d - x_a)

for i in range(0, 70):
    MBM.reset()
    u0 = MBM.minimize(u0)
    x_a = MBM.r[-1, :].reshape(3, 1)  # actual position of the robot end-effector
    dx = x_a - x_d
    X = 0.0001  # learning rate
    Jac = Jac0 + (X / (dq.T @ dq).ravel()) * (dx - Jac0 @ dq) @ dq.T
    dq = -np.linalg.pinv(Jac) @ K @ (x_d - x_a)
    q = 0.01 * dq.reshape(5, ) + q
    Jac0 = Jac
    MBM.insert = q[4]
    MBM.q = q[0:4]
    error = np.concatenate((error, np.linalg.norm(x_d - x_a)), axis=None)
    print(np.linalg.norm(x_d - x_a))

# Plotting the robot and principal axes of manipulability ellipsoids
print("--- %s seconds ---" % (time.time() - start_time))
fig = plt.figure()
# ax = plt.axes(projection='3d')
# # plot the robot shape
# ax.plot(MBM.r[:, 0], MBM.r[:, 1], MBM.r[:, 2], '-b', label='CTR Robot')
# ax.auto_scale_xyz([np.amin(MBM.r[:, 0]), np.amax(MBM.r[:, 0]) + 0.01],
#                   [np.amin(MBM.r[:, 1]), np.amax(MBM.r[:, 1]) + 0.01],
#                   [np.amin(MBM.r[:, 2]), np.amax(MBM.r[:, 2]) + 0.01])
# ax.set_xlabel('X [mm]')
# ax.set_ylabel('Y [mm]')
# ax.set_zlabel('Z [mm]')
plt.plot(range(0, 71), error)
plt.grid(True)
#plt.legend()
plt.show()
