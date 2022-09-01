import numpy as np


class UsvAsmc():
    def __init__(self):
        # USV model coefficients
        self.X_u_dot = -2.25
        self.Y_v_dot = -23.13
        self.Y_r_dot = -1.31
        self.N_v_dot = -16.41
        self.N_r_dot = -2.79
        self.Xuu = 0
        self.Yvv = -99.99
        self.Yvr = -5.49
        self.Yrv = -5.49
        self.Yrr = -8.8
        self.Nvv = -5.49
        self.Nvr = -8.8
        self.Nrv = -8.8
        self.Nrr = -3.49
        self.m = 30
        self.Iz = 4.1
        self.B = 0.41
        self.c = 0.78

        # ASMC gains
        self.k_u = 0.1
        self.k_psi = 0.2
        self.kmin_u = 0.05
        self.kmin_psi = 0.2
        self.k2_u = 0.02
        self.k2_psi = 0.1
        self.mu_u = 0.05
        self.mu_psi = 0.1
        self.lambda_u = 0.001
        self.lambda_psi = 1

        # Second order filter gains (for r_d)
        self.f1 = 2.0
        self.f2 = 2.0
        self.f3 = 2.0

        self.so_filter = np.zeros(7)

        self.last = np.zeros(9)
        self.aux_vars = np.array([0, 0, 0])
        self.integral_step = 0.01

    # position = eta
    # velocity = upsilon
    def compute(self, action, position, velocity):
        x_dot_last, y_dot_last, psi_dot_last, u_dot_last, v_dot_last, r_dot_last, e_u_last, ka_dot_u_last, ka_dot_psi_last = self.last
        psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot = self.so_filter

        u, v, r = velocity
        x, y, psi = position

        e_u_int, ka_u, ka_psi = self.aux_vars

        # Create model related vectors
        position = np.array([x, y, psi])
        velocity = np.array([u, v, r])
        eta_dot_last = np.array([x_dot_last, y_dot_last, psi_dot_last])
        upsilon_dot_last = np.array([u_dot_last, v_dot_last, r_dot_last])

        for _ in range(10):
            beta = np.math.asin(velocity[1] / (0.001 + np.hypot(velocity[0], velocity[1])))
            chi = psi + beta
            # chi = np.where(np.greater(np.abs(chi), np.pi), (np.sign(chi)) * (np.abs(chi) - 2 * np.pi), chi)

            # Compute the desired heading
            psi_d = chi + action[1]
            # psi_d = ak + action[1]
            # psi_d = np.where(np.greater(np.abs(psi_d), np.pi), (np.sign(psi_d)) * (np.abs(psi_d) - 2 * np.pi), psi_d)

            # Second order filter to compute desired yaw rate
            r_d = (psi_d - psi_d_last) / self.integral_step
            psi_d_last = psi_d
            o_dot_dot = (((r_d - o_last) * self.f1) - (self.f3 * o_dot_last)) * self.f2
            o_dot = self.integral_step * (o_dot_dot + o_dot_dot_last) / 2 + o_dot
            o = self.integral_step * (o_dot + o_dot_last) / 2 + o
            r_d = o
            o_last = o
            o_dot_last = o_dot
            o_dot_dot_last = o_dot_dot

            # Compute variable hydrodynamic coefficients
            Xu = -25
            Xuu = 0
            if abs(velocity[0]) > 1.2:
                Xu = 64.55
                Xuu = -70.92

            Yv = 0.5 * (-40 * 1000 * abs(velocity[1])) * \
                 (1.1 + 0.0045 * (1.01 / 0.09) - 0.1 * (0.27 / 0.09) + 0.016 * (np.power((0.27 / 0.09), 2)))
            Yr = 6 * (-3.141592 * 1000) * \
                 np.sqrt(np.power(velocity[0], 2) + np.power(velocity[1], 2)) * 0.09 * 0.09 * 1.01
            Nv = 0.06 * (-3.141592 * 1000) * \
                 np.sqrt(np.power(velocity[0], 2) + np.power(velocity[1], 2)) * 0.09 * 0.09 * 1.01
            Nr = 0.02 * (-3.141592 * 1000) * \
                 np.sqrt(np.power(velocity[0], 2) + np.power(velocity[1], 2)) * 0.09 * 0.09 * 1.01 * 1.01

            # Rewrite USV model in simplified components f and g
            g_u = 1 / (self.m - self.X_u_dot)
            g_psi = 1 / (self.Iz - self.N_r_dot)
            f_u = (((self.m - self.Y_v_dot) * velocity[1] * velocity[2] + (
                    Xuu * np.abs(velocity[0]) + Xu * velocity[0])) / (self.m - self.X_u_dot))
            f_psi = (((-self.X_u_dot + self.Y_v_dot) * velocity[0] * velocity[1] + (Nr * velocity[2])) / (
                    self.Iz - self.N_r_dot))

            # Compute heading error
            e_psi = psi_d - position[2]
            e_psi = np.where(np.greater(np.abs(e_psi), np.pi), (np.sign(e_psi)) * (np.abs(e_psi) - 2 * np.pi), e_psi)
            e_psi_dot = r_d - velocity[2]

            # Compute desired speed (unnecessary if DNN gives it)
            u_d = action[0]

            # Compute speed error
            e_u = u_d - velocity[0]
            e_u_int = self.integral_step * (e_u + e_u_last) / 2 + e_u_int
            e_u_last = e_u

            # Create sliding surfaces for speed and heading
            sigma_u = e_u + self.lambda_u * e_u_int
            sigma_psi = e_psi_dot + self.lambda_psi * e_psi

            # Compute ASMC gain derivatives
            ka_dot_u = np.where(np.greater(ka_u, self.kmin_u), self.k_u * np.sign(np.abs(sigma_u) - self.mu_u),
                                self.kmin_u)
            ka_dot_psi = np.where(np.greater(ka_psi, self.kmin_psi),
                                  self.k_psi * np.sign(np.abs(sigma_psi) - self.mu_psi), self.kmin_psi)

            # Compute gains
            ka_u = self.integral_step * (ka_dot_u + ka_dot_u_last) / 2 + ka_u
            ka_dot_u_last = ka_dot_u

            ka_psi = self.integral_step * (ka_dot_psi + ka_dot_psi_last) / 2 + ka_psi
            ka_dot_psi_last = ka_dot_psi

            # Compute ASMC for speed and heading
            ua_u = (-ka_u * np.power(np.abs(sigma_u), 0.5) * np.sign(sigma_u)) - (self.k2_u * sigma_u)
            ua_psi = (-ka_psi * np.power(np.abs(sigma_psi), 0.5) * np.sign(sigma_psi)) - (self.k2_psi * sigma_psi)

            # Compute control inputs for speed and heading
            tx = ((self.lambda_u * e_u) - f_u - ua_u) / g_u
            tz = ((self.lambda_psi * e_psi) - f_psi - ua_psi) / g_psi

            # Compute both thrusters and saturate their values
            tport = (tx / 2) + (tz / self.B)
            tstbd = (tx / (2 * self.c)) - (tz / (self.B * self.c))

            tport = np.where(np.greater(tport, 36.5), 36.5, tport)
            tport = np.where(np.less(tport, -30), -30, tport)
            tstbd = np.where(np.greater(tstbd, 36.5), 36.5, tstbd)
            tstbd = np.where(np.less(tstbd, -30), -30, tstbd)

            # Compute USV model matrices
            M = np.array([[self.m - self.X_u_dot, 0, 0],
                          [0, self.m - self.Y_v_dot, 0 - self.Y_r_dot],
                          [0, 0 - self.N_v_dot, self.Iz - self.N_r_dot]])

            T = np.array([tport + self.c * tstbd, 0, 0.5 * self.B * (tport - self.c * tstbd)])

            CRB = np.array([[0, 0, 0 - self.m * velocity[1]],
                            [0, 0, self.m * velocity[0]],
                            [self.m * velocity[1], 0 - self.m * velocity[0], 0]])

            CA = np.array(
                [[0, 0, 2 * ((self.Y_v_dot * velocity[1]) + ((self.Y_r_dot + self.N_v_dot) / 2) * velocity[2])],
                 [0, 0, 0 - self.X_u_dot * self.m * velocity[0]],
                 [2 * (((0 - self.Y_v_dot) * velocity[1]) - ((self.Y_r_dot + self.N_v_dot) / 2) * velocity[2]),
                  self.X_u_dot * self.m * velocity[0], 0]])

            C = CRB + CA

            Dl = np.array([[0 - Xu, 0, 0],
                           [0, 0 - Yv, 0 - Yr],
                           [0, 0 - Nv, 0 - Nr]])

            Dn = np.array([[Xuu * abs(velocity[0]), 0, 0],
                           [0, self.Yvv * abs(velocity[1]) + self.Yvr * abs(velocity[2]), self.Yrv *
                            abs(velocity[1]) + self.Yrr * abs(velocity[2])],
                           [0, self.Nvv * abs(velocity[1]) + self.Nvr * abs(velocity[2]),
                            self.Nrv * abs(velocity[1]) + self.Nrr * abs(velocity[2])]])

            D = Dl - Dn

            # Compute acceleration and velocity in body
            upsilon_dot = np.matmul(np.linalg.inv(
                M), (T - np.matmul(C, velocity) - np.matmul(D, velocity)))
            velocity = self.integral_step * (upsilon_dot +
                                             upsilon_dot_last) / 2 + velocity  # integral
            upsilon_dot_last = upsilon_dot

            # Rotation matrix
            J = np.array([[np.cos(position[2]), -np.sin(position[2]), 0],
                          [np.sin(position[2]), np.cos(position[2]), 0],
                          [0, 0, 1]])

            # Compute NED position
            eta_dot = np.matmul(J, velocity)  # transformation into local reference frame
            position = self.integral_step * (eta_dot + eta_dot_last) / 2 + position  # integral
            eta_dot_last = eta_dot

            self.last = np.array(
                [eta_dot_last[0], eta_dot_last[1], eta_dot_last[2], upsilon_dot_last[0], upsilon_dot_last[1],
                 upsilon_dot_last[2], e_u_last, ka_dot_u_last, ka_dot_psi_last])

            self.so_filter = np.array([psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot])
            self.aux_vars = np.array([e_u_int, ka_u, ka_psi])
            return position, velocity
