import numpy as np
from math import pi, pow
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import optimize
import time


class MBM_Model:
    def __init__(self, length_seg1, length_seg2, length_rod, w, f, q, Tol):
        self.accuracy = Tol  # ode solver maximum step size
        self.eps = 1.e-4
        self.q = q  # rods pull/push
        self.f = f.astype(float).reshape(3, 1)  # external point force at the tip
        self.w = w.astype(float).reshape(3, 1)  # external distributed force
        self.L1 = length_seg1  # length of the long cables
        self.L2 = length_seg2  # length of the small cables
        self.L3 = length_rod  # length of the inner rod
        r_od = 2.3e-3 / 2  # outer diameter of the main-backbone
        r_in = 2e-3 / 2  # inner diameter of the main-backbone
        r_od2 = 1.5e-3 / 2  # outer diameter of the inner rod
        r_in2 = 1e-3 / 2  # inner diameter of the inner rod
        r = 0.475e-3 / 2  # diameter of the rods
        self.delta = 2e-3  # rods distance from the center
        E1 = 30e9  # stiffness of the main backbone
        G1 = 5e9  # torsional stiffness of the main backbone
        E2 = 70e9  # stiffness of the inner rod
        G2 = 15e9  # torsional stiffness of the inner rod

        # segmenting the robot
        self.S = np.array([self.L3, self.L2 - self.L3, self.L1 - self.L2 - self.L3])
        # s is segmented abscissa of the robot after template
        I1 = 3 * ((pi * pow(r, 4)) / 64 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             3 * ((pi * pow(r, 4)) / 64 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 64 + \
             (pi * (pow(r_in2, 4) - pow(r_od2, 4))) / 64
        I2 = 3 * ((pi * pow(r, 4)) / 64 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             3 * ((pi * pow(r, 4)) / 64 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 64
        I3 = 3 * ((pi * pow(r, 4)) / 64 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 64 + \
             (pi * (pow(r_in2, 4) - pow(r_od2, 4))) / 64
        J1 = 3 * ((pi * pow(r, 4)) / 32 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             3 * ((pi * pow(r, 4)) / 32 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 32 + \
             (pi * (pow(r_in2, 4) - pow(r_od2, 4))) / 32
        J2 = 3 * ((pi * pow(r, 4)) / 32 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             3 * ((pi * pow(r, 4)) / 32 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 32
        J3 = 3 * ((pi * pow(r, 4)) / 32 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 32 + \
             (pi * (pow(r_in2, 4) - pow(r_od2, 4))) / 32
        self.EI = np.array([(E1 + E2) * I1, E1 * I2, E1 * I3])
        self.GJ = np.array([(G1 + G2) * J1, G1 * J2, G1 * J3])

        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)  # initial position of robot
        self.Length0 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)  # initial length of rods
        self.alpha_0 = 0  # initial twist angle of the robot
        self.R_0 = np.array(
            [[np.cos(self.alpha_0), -np.sin(self.alpha_0), 0], [np.sin(self.alpha_0), np.cos(self.alpha_0), 0],
             [0, 0, 1]]) \
            .reshape(9, 1)  # initial rotation matrix
        self.Length = np.empty((0, 6))
        self.r = np.empty((0, 3))

    def reset(self):
        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)  # initial position of robot
        self.alpha_0 = 0  # initial twist angle of the robot
        self.R_0 = np.array(
            [[np.cos(self.alpha_0), -np.sin(self.alpha_0), 0], [np.sin(self.alpha_0), np.cos(self.alpha_0), 0],
             [0, 0, 1]]) \
            .reshape(9, 1)  # initial rotation matrix
        self.Length = np.empty((0, 6))
        self.Length0 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)  # initial length of rods
        self.r = np.empty((0, 3))

        # segmenting the robot
        r_od = 2.3e-3 / 2  # outer diameter of the main-backbone
        r_in = 2e-3 / 2  # inner diameter of the main-backbone
        r_od2 = 1.5e-3 / 2  # outer diameter of the inner rod
        r_in2 = 1e-3 / 2  # inner diameter of the inner rod
        r = 0.475e-3 / 2  # diameter of the rods
        E1 = 30e9  # stiffness of the main backbone
        G1 = 5e9  # torsional stiffness of the main backbone
        E2 = 70e9  # stiffness of the inner rod
        G2 = 15e9  # torsional stiffness of the inner rod
        self.S = np.array([self.L3, self.L2 - self.L3, self.L1 - self.L2 - self.L3])
        # s is segmented abscissa of the robot after template
        I1 = 3 * ((pi * pow(r, 4)) / 64 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             3 * ((pi * pow(r, 4)) / 64 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 64 + \
             (pi * (pow(r_in2, 4) - pow(r_od2, 4))) / 64
        I2 = 3 * ((pi * pow(r, 4)) / 64 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             3 * ((pi * pow(r, 4)) / 64 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 64
        I3 = 3 * ((pi * pow(r, 4)) / 64 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 64 + \
             (pi * (pow(r_in2, 4) - pow(r_od2, 4))) / 64
        J1 = 3 * ((pi * pow(r, 4)) / 32 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             3 * ((pi * pow(r, 4)) / 32 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 32 + \
             (pi * (pow(r_in2, 4) - pow(r_od2, 4))) / 32
        J2 = 3 * ((pi * pow(r, 4)) / 32 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             3 * ((pi * pow(r, 4)) / 32 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 32
        J3 = 3 * ((pi * pow(r, 4)) / 32 + pi * pow(self.delta, 2) * pow(r, 2)) + \
             (pi * (pow(r_in, 4) - pow(r_od, 4))) / 32 + \
             (pi * (pow(r_in2, 4) - pow(r_od2, 4))) / 32
        self.EI = np.array([(E1 + E2) * I1, E1 * I2, E1 * I3])
        self.GJ = np.array([(G1 + G2) * J1, G1 * J2, G1 * J3])

    # ordinary differential equations for CTR with 3 tubes
    def ode_eq(self, s, y, i, l):
        # first two elements of y are curvatures along x and y
        # next 12 elements are r and R
        # next 6 elements are the length of the first, second, and third cables, respectively
        dydt = np.empty([20, 1])
        U = np.array([[y[0]], [y[1]], [0.0]])  # Vector of curvatures
        K = np.diag(np.array([self.EI[i], self.EI[i], self.GJ[i]]))
        K_inv = np.diag(np.array([1 / np.sum(self.EI[i]), 1 / np.sum(self.EI[i]), 1 / np.sum(self.GJ[i])]))
        U_hat = np.array([[0, -U[2, 0], U[1, 0]], [U[2, 0], 0, -U[0, 0]], [-U[1, 0], U[0, 0], 0]])
        e3 = np.array([[0.0], [0.0], [1.0]])
        e3_hat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        R = np.array(
            [[y[5], y[6], y[7]], [y[8], y[9], y[10]], [y[11], y[12], y[13]]])  # rotation matrix

        # odes
        du = -K_inv @ (U_hat @ K @ U + e3_hat @ R.T @ (self.f + (l - s) * self.w) )
        dr = R @ e3
        dR = (R @ U_hat).ravel()

        a = np.pi / 6
        b = 2 * np.pi / 3
        Rb = np.identity(3)
        dl1 = np.linalg.norm(e3 + U_hat @ Rb @ np.array([[self.delta], [0], [0]]))
        Rb = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
        dl2 = np.linalg.norm(e3 + U_hat @ Rb @ np.array([[self.delta], [0], [0]]))
        Rb = np.array([[np.cos(2 * b), -np.sin(2 * b), 0], [np.sin(2 * b), np.cos(2 * b), 0], [0, 0, 1]])
        dl3 = np.linalg.norm(e3 + U_hat @ Rb @ np.array([[self.delta], [0], [0]]))
        Rb = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        dl4 = np.linalg.norm(e3 + U_hat @ Rb @ np.array([[self.delta], [0], [0]]))
        Rb = np.array([[np.cos(a + b), -np.sin(a + b), 0], [np.sin(a + b), np.cos(a + b), 0], [0, 0, 1]])
        dl5 = np.linalg.norm(e3 + U_hat @ Rb @ np.array([[self.delta], [0], [0]]))
        Rb = np.array(
            [[np.cos(a + 2 * b), -np.sin(a + 2 * b), 0], [np.sin(a + 2 * b), np.cos(a + 2 * b), 0], [0, 0, 1]])
        dl6 = np.linalg.norm(e3 + U_hat @ Rb @ np.array([[self.delta], [0], [0]]))

        dydt[0] = du[0, 0]
        dydt[1] = du[1, 0]
        dydt[2] = dr[0, 0]
        dydt[3] = dr[1, 0]
        dydt[4] = dr[2, 0]
        dydt[14], dydt[15], dydt[16], dydt[17], dydt[18], dydt[19] = dl1, dl2, dl3, dl4, dl5, dl6

        for k in range(5, 14):
            dydt[k] = dR[k - 5]

        return dydt.ravel()

    def ode_solver(self, u_init):
        u_init = u_init.reshape(4, 1)
        u1_xy_0 = np.array([[u_init[0, 0]], [u_init[1, 0]]])
        u2_xy_0 = np.array([[u_init[2, 0]], [u_init[3, 0]]])
        self.reset()
        for seg in range(0, len(self.S)):
            # Initial conditions: 2 initial curvature, 3 initial position, 9 initial rotation matrix, 6 length of rods
            y_0 = np.vstack((u1_xy_0, self.r_0, self.R_0, self.Length0)).ravel()
            ell = np.sum(self.S[seg:])
            s = solve_ivp(lambda s, y: self.ode_eq(s, y, seg, ell),
                          [0, self.S[seg]],
                          y_0, method='RK23', max_step=self.accuracy)
            ans = s.y.transpose()
            self.r = np.vstack((self.r, ans[:, (2, 3, 4)]))
            self.Length = np.vstack((self.Length, ans[:, (14, 15, 16, 17, 18, 19)]))
            # new boundary conditions for next segment
            self.r_0 = self.r[-1, :].reshape(3, 1)
            R = np.array(ans[-1, 5:14]).reshape(3, 3)
            self.R_0 = R.reshape(9, 1)
            if seg >= 1:
                u1_xy_0 = u2_xy_0.reshape(2, 1)
            else:
                u1_xy_0 = np.array(ans[-1, (0, 1)]).reshape(2, 1)
            if seg > 1:
                self.Length0[3:, ] = np.array(ans[-1, (17, 18, 19)]).reshape(3, 1)
            else:
                self.Length0 = np.array(ans[-1, (14, 15, 16, 17, 18, 19)]).reshape(6, 1)
        # cost function for bpv solver includes 4 values: distance between the actual and
        # estimated length of rod 1,2,4, and 5
        Cost = np.array([self.Length0[0, 0] - (self.L2 + self.q[0]), self.Length0[1, 0] - (self.L2 + self.q[1]),
                         self.Length0[3, 0] - (self.L1 + self.q[2]), self.Length0[4, 0] - (self.L1 + self.q[3])])

        print(Cost)
        return Cost

    # Solving the BVP problem using built-in scipy minimize module
    def minimize(self, u_init):
        u0 = u_init
        res = optimize.anderson(self.ode_solver, u0, f_tol=1e-4)
        # another option for scalar cost function is  is res = optimize.minimize(self.ode_solver, u0,
        # method='Powell', options={'gtol': 1e-3, 'maxiter': 1000})
        return res


def main():

    # Joint variables (pulling or pushing rods 1,2,4, and 5)
    q = np.array([0.002, 0, -0.005, 0])
    length_seg1 = 550e-3  # length of long rods
    length_seg2 = 500e-3  # length of shorter rods
    length_rod = 0e-3  # length of the inner rod
    # force on robot tip along x, y, and z direction
    f = np.array([2, 0, 0]).reshape(3, 1)
    # distributed force on robot tip along x, y, and z direction
    w = np.array([0, 0, 0]).reshape(3, 1)

    # Use this command if you wish to use initial value problem (ivp) solver (less accurate but faster)
    MBM = MBM_Model(length_seg1, length_seg2, length_rod, w, f, q, 0.01)
    MBM.minimize(np.array([0, 0, 0, 0]))
    # Plotting the robot and principal axes of manipulability ellipsoids
    print("--- %s seconds ---" % (time.time() - start_time))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # plot the robot shape
    ax.plot(MBM.r[:, 0], MBM.r[:, 1], MBM.r[:, 2], '-b', label='CTR Robot')
    # ax.auto_scale_xyz([np.amin(MBM.r[:, 0]), np.amax(MBM.r[:, 0]) + 0.01],
    #                   [np.amin(MBM.r[:, 1]), np.amax(MBM.r[:, 1]) + 0.01],
    #                   [np.amin(MBM.r[:, 2]), np.amax(MBM.r[:, 2]) + 0.01])
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    plt.grid(True)
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()
