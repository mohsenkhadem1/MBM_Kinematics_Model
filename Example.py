
import numpy as np
from MBM import MBM_Model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import matplotlib.pyplot as plt
import time

start_time = time.time()
# Joint variables (pulling or pushing rods 1,2,4, and 5)
q = np.array([0.002, 0, -0.005, 0])
length_seg1 = 550e-3  # length of long rods
length_seg2 = 500e-3  # length of shorter rods
length_rod = 0e-3  # length of the inner rod
# force on robot tip along x, y, and z direction
f = np.array([0, 0, 0]).reshape(3, 1)
# distributed force on robot tip along x, y, and z direction
w = np.array([0, 0, 0]).reshape(3, 1)
tol = 0.01
MBM = MBM_Model(length_seg1, length_seg2, length_rod, w, f, q, tol)
MBM.minimize(np.array([0, 0, 0, 0]))

# Plotting the robot and principal axes of manipulability ellipsoids
print("--- %s seconds ---" % (time.time() - start_time))
fig = plt.figure()
ax = plt.axes(projection='3d')
# plot the robot shape
ax.plot(MBM.r[:, 0], MBM.r[:, 1], MBM.r[:, 2], '-b', label='CTR Robot')
ax.auto_scale_xyz([np.amin(MBM.r[:, 0]), np.amax(MBM.r[:, 0]) + 0.01],
                [np.amin(MBM.r[:, 1]), np.amax(MBM.r[:, 1]) + 0.01],
                [np.amin(MBM.r[:, 2]), np.amax(MBM.r[:, 2]) + 0.01])
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
plt.grid(True)
plt.legend()
plt.show()
