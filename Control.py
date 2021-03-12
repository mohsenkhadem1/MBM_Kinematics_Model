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
length_seg1 = 550e-3  # length of long rods
length_seg2 = 500e-3  # length of shorter rods
length_rod = 0e-3  # length of the inner shaft
# force on robot tip along x, y, and z direction
f = np.array([0, 0, 0]).reshape(3, 1)
# distributed force on robot tip along x, y, and z direction
w = np.array([0, 0, 0]).reshape(3, 1)
# mimimum step size for the solver
tol = 0.01
MBM = MBM_Model(length_seg1, length_seg2, length_rod, w, f, q, 0.01)

L_d = np.array([length_seg2, length_seg2, length_seg1, length_seg1]).reshape(4, 1) \
      + q[0:4].reshape(4, 1)  # desired length of cables

u0 = np.array([0.0, 0.0, 0.0, 0.0]).reshape(4, 1)

x_d = np.array([0.0, 0.0, 0.56]).reshape(3, 1)  # desired position of the robot end-effector
dq = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # initial velocity of cables
start_time = time.time()

for i in np.linspace(0, 10, num=50):
    dt = 0.01
    # start_time = time.time()
    q = dt * dq.reshape(5, ) + q
    # MBM.insert = q[4, 0]
    # MBM.q = q[0:4, 0]
    MBM.insert = q[4]
    MBM.q = q[0:4]
    MBM.reset()
    # K1 = 70 * np.identity(4)
    K2 = 1 * np.identity(3)
    u0 = MBM.minimize(np.array(u0))
    x_a = MBM.r[-1, :].reshape(3, 1)  # actual position of the robot end-effector

    # # find new initial curvature using observer
    # L = np.array([MBM.Length0[0], MBM.Length0[1], MBM.Length0[3], MBM.Length0[4]]).reshape(4, 1)  # actual length of
    # # cables
    # L_d = np.array([length_seg2, length_seg2, length_seg1, length_seg1]).reshape(4, 1) \
    #       + q[0:4, 0].reshape(4, 1)  # desired length of cables
    # dU = np.linalg.pinv(MBM.grad) @ (K1 @ (L_d - L))
    # u0 = u0 + dt * dU
    # print(np.linalg.norm(L_d - L))

    # calculating Jacobian
    eps = 0.00001
    pos_init = x_a.reshape(3, )
    Jac = np.empty((3, 5))
    for j in range(0, 5):
        q[j] = q[j] + eps
        MBM.q = q[0:4]
        MBM.insert = q[4]
        # L_d = np.array([length_seg2, length_seg2, length_seg1, length_seg1]).reshape(4, 1) \
        #       + q[0:4, 0].reshape(4, 1)  # desired length of cables
        # dU = np.linalg.pinv(MBM.grad) @ (K1 @ (L_d - L))
        # u0 = u0 + dt * dU
        # MBM.ode_obs_solver(u0.ravel())
        MBM.minimize(np.array(u0))
        pos_new = MBM.r[-1, :].reshape(3, )
        Jac[:, j] = (pos_new - pos_init) / eps
        q[j] = q[j] - eps

    # find joint inputs using inverse kinematics
    dq = np.linalg.pinv(Jac) @ K2 @ (x_d - x_a)
    print(np.linalg.norm(x_d - x_a))

# Plotting the robot shape
fig = plt.figure()
ax = plt.axes(projection='3d')
# plot the robot shape
ax.plot(MBM.r[:, 0], MBM.r[:, 1], MBM.r[:, 2], '-b', label='CTR Robot')
ax.auto_scale_xyz([np.amin(MBM.r[:, 0]), np.amax(MBM.r[:, 0]) + 0.01],
                  [np.amin(MBM.r[:, 1]), np.amax(MBM.r[:, 1]) + 0.01],
                  [0, np.amax(MBM.r[:, 2]) + 0.01])
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')

plt.grid(True)
plt.legend()
plt.show()
