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

        self.so_filter = None

        self.last = None
        self.aux_vars = np.array([0,0,0])
        self.integral_step = 0.01

    def compute_asmc(self, state, position, action, last_values, so_filter, aux_vars):
        x_dot_last, y_dot_last, psi_dot_last, u_dot_last, v_dot_last, r_dot_last, e_u_last, Ka_dot_u_last, Ka_dot_psi_last = last_values
        psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot = so_filter

        u, v, r, ye, ye_dot, chi_ak, u_ref= state[0], state[1], state[2], state[3], state[4], state[5], state[6]
        x, y, psi = position
        e_u_int, Ka_u, Ka_psi = self.aux_vars

        # Create model related vectors
        eta = np.array([x, y, psi])
        upsilon = np.array([u, v, r])
        eta_dot_last = np.array([x_dot_last, y_dot_last, psi_dot_last])
        upsilon_dot_last = np.array([u_dot_last, v_dot_last, r_dot_last])

        for i in range(10):
            beta = np.math.asin(upsilon[1] / (0.001 + np.hypot(upsilon[0], upsilon[1])))
            chi = psi + beta

            # Compute the desired heading
            psi_d = chi + action[1]

            # Second order filter to compute desired yaw rate
            r_d = (psi_d - psi_d_last) / self.integral_step
            psi_d_last = psi_d
            o_dot_dot = (((r_d - o_last) * self.f1) - (self.f3 * o_dot_last)) * self.f2
            o_dot = (self.integral_step) * (o_dot_dot + o_dot_dot_last) / 2 + o_dot
            o = (self.integral_step) * (o_dot + o_dot_last) / 2 + o
            r_d = o
            o_last = o
            o_dot_last = o_dot
            o_dot_dot_last = o_dot_dot

            # Compute variable hydrodynamic coefficients
            Xu = -25
            Xuu = 0
            if (abs(upsilon[0]) > 1.2):
                Xu = 64.55
                Xuu = -70.92

            Yv = 0.5 * (-40 * 1000 * abs(upsilon[1])) * \
                 (1.1 + 0.0045 * (1.01 / 0.09) - 0.1 * (0.27 / 0.09) + 0.016 * (np.power((0.27 / 0.09), 2)))
            Yr = 6 * (-3.141592 * 1000) * \
                 np.sqrt(np.power(upsilon[0], 2) + np.power(upsilon[1], 2)) * 0.09 * 0.09 * 1.01
            Nv = 0.06 * (-3.141592 * 1000) * \
                 np.sqrt(np.power(upsilon[0], 2) + np.power(upsilon[1], 2)) * 0.09 * 0.09 * 1.01
            Nr = 0.02 * (-3.141592 * 1000) * \
                 np.sqrt(np.power(upsilon[0], 2) + np.power(upsilon[1], 2)) * 0.09 * 0.09 * 1.01 * 1.01

            # Rewrite USV model in simplified components f and g
            g_u = 1 / (self.m - self.X_u_dot)
            g_psi = 1 / (self.Iz - self.N_r_dot)
            f_u = (((self.m - self.Y_v_dot) * upsilon[1] * upsilon[2] + (
                    Xuu * np.abs(upsilon[0]) + Xu * upsilon[0])) / (self.m - self.X_u_dot))
            f_psi = (((-self.X_u_dot + self.Y_v_dot) * upsilon[0] * upsilon[1] + (Nr * upsilon[2])) / (
                    self.Iz - self.N_r_dot))

            # Compute heading error
            e_psi = psi_d - eta[2]
            e_psi = np.where(np.greater(np.abs(e_psi), np.pi), (np.sign(e_psi)) * (np.abs(e_psi) - 2 * np.pi), e_psi)
            e_psi_dot = r_d - upsilon[2]

            # Compute desired speed (unnecessary if DNN gives it)
            u_d = action[0]

            # Compute speed error
            e_u = u_d - upsilon[0]
            e_u_int = self.integral_step * (e_u + e_u_last) / 2 + e_u_int

            # Create sliding surfaces for speed and heading
            sigma_u = e_u + self.lambda_u * e_u_int
            sigma_psi = e_psi_dot + self.lambda_psi * e_psi

            # Compute ASMC gain derivatives
            Ka_dot_u = np.where(np.greater(Ka_u, self.kmin_u), self.k_u * np.sign(np.abs(sigma_u) - self.mu_u),
                                self.kmin_u)
            Ka_dot_psi = np.where(np.greater(Ka_psi, self.kmin_psi),
                                  self.k_psi * np.sign(np.abs(sigma_psi) - self.mu_psi), self.kmin_psi)

            # Compute gains
            Ka_u = self.integral_step * (Ka_dot_u + Ka_dot_u_last) / 2 + Ka_u
            Ka_dot_u_last = Ka_dot_u

            Ka_psi = self.integral_step * (Ka_dot_psi + Ka_dot_psi_last) / 2 + Ka_psi
            Ka_dot_psi_last = Ka_dot_psi

            # Compute ASMC for speed and heading
            ua_u = (-Ka_u * np.power(np.abs(sigma_u), 0.5) * np.sign(sigma_u)) - (self.k2_u * sigma_u)
            ua_psi = (-Ka_psi * np.power(np.abs(sigma_psi), 0.5) * np.sign(sigma_psi)) - (self.k2_psi * sigma_psi)

            # Compute control inputs for speed and heading
            Tx = ((self.lambda_u * e_u) - f_u - ua_u) / g_u
            Tz = ((self.lambda_psi * e_psi) - f_psi - ua_psi) / g_psi

            # Compute both thrusters and saturate their values
            Tport = (Tx / 2) + (Tz / self.B)
            Tstbd = (Tx / (2 * self.c)) - (Tz / (self.B * self.c))

            Tport = np.where(np.greater(Tport, 36.5), 36.5, Tport)
            Tport = np.where(np.less(Tport, -30), -30, Tport)
            Tstbd = np.where(np.greater(Tstbd, 36.5), 36.5, Tstbd)
            Tstbd = np.where(np.less(Tstbd, -30), -30, Tstbd)

            # Compute USV model matrices
            M = np.array([[self.m - self.X_u_dot, 0, 0],
                          [0, self.m - self.Y_v_dot, 0 - self.Y_r_dot],
                          [0, 0 - self.N_v_dot, self.Iz - self.N_r_dot]])

            T = np.array([Tport + self.c * Tstbd, 0, 0.5 * self.B * (Tport - self.c * Tstbd)])

            CRB = np.array([[0, 0, 0 - self.m * upsilon[1]],
                            [0, 0, self.m * upsilon[0]],
                            [self.m * upsilon[1], 0 - self.m * upsilon[0], 0]])

            CA = np.array([[0, 0, 2 * ((self.Y_v_dot * upsilon[1]) + ((self.Y_r_dot + self.N_v_dot) / 2) * upsilon[2])],
                           [0, 0, 0 - self.X_u_dot * self.m * upsilon[0]],
                           [2 * (((0 - self.Y_v_dot) * upsilon[1]) - ((self.Y_r_dot + self.N_v_dot) / 2) * upsilon[2]),
                            self.X_u_dot * self.m * upsilon[0], 0]])

            C = CRB + CA

            Dl = np.array([[0 - Xu, 0, 0],
                           [0, 0 - Yv, 0 - Yr],
                           [0, 0 - Nv, 0 - Nr]])

            Dn = np.array([[Xuu * abs(upsilon[0]), 0, 0],
                           [0, self.Yvv * abs(upsilon[1]) + self.Yvr * abs(upsilon[2]), self.Yrv *
                            abs(upsilon[1]) + self.Yrr * abs(upsilon[2])],
                           [0, self.Nvv * abs(upsilon[1]) + self.Nvr * abs(upsilon[2]),
                            self.Nrv * abs(upsilon[1]) + self.Nrr * abs(upsilon[2])]])

            D = Dl - Dn

            # Compute acceleration and velocity in body
            upsilon_dot = np.matmul(np.linalg.inv(
                M), (T - np.matmul(C, upsilon) - np.matmul(D, upsilon)))
            upsilon = (self.integral_step) * (upsilon_dot +
                                              upsilon_dot_last) / 2 + upsilon  # integral
            upsilon_dot_last = upsilon_dot

            # Rotation matrix
            J = np.array([[np.cos(eta[2]), -np.sin(eta[2]), 0],
                          [np.sin(eta[2]), np.cos(eta[2]), 0],
                          [0, 0, 1]])

            # Compute NED position
            eta_dot = np.matmul(J, upsilon)  # transformation into local reference frame
            eta = (self.integral_step) * (eta_dot + eta_dot_last) / 2 + eta  # integral
            eta_dot_last = eta_dot

            psi = eta[2]

        self.last = np.array(
            [eta_dot_last[0], eta_dot_last[1], eta_dot_last[2], upsilon_dot_last[0], upsilon_dot_last[1],
             upsilon_dot_last[2], e_u_last, Ka_dot_u_last, Ka_dot_psi_last])

        self.so_filter = np.array([psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot])
        self.aux_vars = np.array([e_u_int, Ka_u, Ka_psi])

        return eta, upsilon, psi