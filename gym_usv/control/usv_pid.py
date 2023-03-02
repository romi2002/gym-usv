import numpy as np


class UsvPID():
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

        # PID gains
        self.kp_u = 1.6
        self.ki_u = 0.2
        self.kd_u = 0.1
        self.kp_psi = 22.625
        self.kd_psi = 10

        self.k_ak = 5.72
        self.k_ye = 0.5
        self.sigma_ye = 1.

        # Second order filter gains (for r_d)
        self.f1 = 2.0
        self.f2 = 2.0
        self.f3 = 2.0

        self.so_filter = np.zeros(7)

        self.last = np.zeros(9)
        self.aux_vars = np.array([0, 0, 0])
        self.integral_step = 0.01

        self.perturb_step = 0

    def _wrap_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    # position = eta
    # velocity = upsilon
    def compute(self, action, position, velocity, do_perturb):
        infos = []

        for _ in range(10):
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
            # print(thuster_perturbation)

            info = {}

            # TODO WRAP ANGLE
            beta = np.math.asin(velocity[1] / (0.001 + np.hypot(velocity[0], velocity[1])))
            psi_d = self._wrap_angle(psi + action[1] + beta)
            info['psi_d'] = psi_d

            Xu = -25
            Xuu = 0
            if (abs(velocity[0]) > 1.2):
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

            g_u = 1 / (self.m - self.X_u_dot)
            g_psi = 1 / (self.Iz - self.N_r_dot)

            f_u = (((self.m - self.Y_v_dot) * velocity[1] * velocity[2] + (
                        Xuu * np.abs(velocity[0]) + Xu * velocity[0])) / (self.m - self.X_u_dot))
            f_psi = (((-self.X_u_dot + self.Y_v_dot) * velocity[0] * velocity[1] + (Nr * velocity[2])) / (
                        self.Iz - self.N_r_dot))

            e_psi = self._wrap_angle(psi_d - position[2])
            info['e_psi'] = e_psi
            e_psi_dot = 0 - velocity[2]

            abs_e_psi = np.abs(e_psi)

            u_psi = 1 / (1 + np.exp(10 * (abs_e_psi * (2 / np.pi) - 0.5)))

            u_d = action[0]
            info['u_d'] = u_d

            e_u = u_d - velocity[0]
            info['e_u'] = e_u
            e_u_int = self.integral_step * (e_u + e_u_last) / 2 + e_u_int
            e_u_dot = (e_u - e_u_last) / self.integral_step

            ua_u = (self.kp_u * e_u) + (self.ki_u * e_u_int) + (self.kd_u * e_u_dot)
            ua_psi = (self.kp_psi * e_psi) + (self.kd_psi * e_psi_dot)

            Tx = (-f_u + ua_u) / g_u
            Tz = (-f_psi + ua_psi) / g_psi

            tport = (Tx / 2) + (Tz / self.B)
            tstbd = (Tx / (2 * self.c)) - (Tz / (self.B * self.c))
            tport = np.clip(tport, -30, 30)
            tstbd = np.clip(tstbd, -30, 30)

            #if thuster_perturbation is not None:
            #    tport += thuster_perturbation[0]
            #    tstbd += thuster_perturbation[1]

            info['tport'] = tport
            info['tstbd'] = tstbd

            # Compute USV model matrices
            M = np.array([[self.m - self.X_u_dot, 0, 0],
                          [0, self.m - self.Y_v_dot, 0 - self.Y_r_dot],
                          [0, 0 - self.N_v_dot, self.Iz - self.N_r_dot]])

            T = np.array([tport + self.c * tstbd, 0, 0.5 * self.B * (tport - self.c * tstbd)])

            # Rotation matrix
            J = np.array([[np.cos(position[2]), -np.sin(position[2]), 0],
                          [np.sin(position[2]), np.cos(position[2]), 0],
                          [0, 0, 1]])

            # Step increases every 0.01sec
            perturb_force = np.zeros(3)
            info['perturb_force_x'] = 0
            info['perturb_force_y'] = 0
            info['perturb_force_local'] = 0
            if do_perturb:
                freq = 15
                magnitude = 5
                x = self.perturb_step * self.integral_step
                k = freq * (2 * np.pi)
                force_x = np.cos(x * k) * magnitude
                force_y = np.cos(x + k + 10) * magnitude
                perturb_force = np.array([force_x, force_y, 0]) @ J
                info['perturb_force_x'] = force_x
                info['perturb_force_y'] = force_y
                info['perturb_force_local'] = perturb_force

            #T += perturb_force
            self.perturb_step += 1

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

            # Compute NED position
            eta_dot = np.matmul(J, velocity)  # transformation into local reference frame
            position = self.integral_step * (eta_dot + eta_dot_last) / 2 + position  # integral
            eta_dot_last = eta_dot

            self.last = np.array(
                [eta_dot_last[0], eta_dot_last[1], eta_dot_last[2], upsilon_dot_last[0], upsilon_dot_last[1],
                 upsilon_dot_last[2], e_u_last, ka_dot_u_last, ka_dot_psi_last])

            self.so_filter = np.array([psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot])
            self.aux_vars = np.array([e_u_int, ka_u, ka_psi])
            infos.append(info)
        return position, velocity, infos
