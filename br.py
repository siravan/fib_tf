#!/home/shahriar/anaconda3/bin/python
import tensorflow as tf
import numpy as np
from screen import Screen
from ionic import IonicModel


class Fenton4v(IonicModel):
    def __init__(self, props):
        IonicModel.__init__(self, props)


    def differentiate(self, U, V, W, S):
        """ the state differentiation for the 4v model """
        # constants for the Fenton 4v left atrial action potential model
        tau_vp = 3.33
        tau_vn1 = 19.2
        tau_vn = tau_vn1
        tau_wp = 160.0
        tau_wn1 = 75.0
        tau_wn2 = 75.0
        tau_d = 0.065
        tau_si = 31.8364
        tau_so = tau_si
        tau_0 = 39.0
        tau_a = 0.009
        u_c = 0.23
        u_w = 0.146
        u_0 = 0.0
        u_m = 1.0
        u_csi = 0.8
        u_so = 0.3
        r_sp = 0.02
        r_sn = 1.2
        k_ = 3.0
        a_so = 0.115
        b_so = 0.84
        c_so = 0.02

        def H(x):
            """ the step function """
            return (1 + tf.sign(x)) * 0.5

        def G(x):
            """ the step function """
            return (1 - tf.sign(x)) * 0.5

        I_fi = -V * H(U - u_c) * (U - u_c) * (u_m - U) / tau_d
        I_si = -W * S / tau_si
        I_so = (0.5 * (a_so - tau_a) * (1 + tf.tanh((U - b_so) / c_so)) +
               (U - u_0) * G(U - u_so) / tau_so + H(U - u_so) * tau_a)

        dU = -(I_fi + I_si + I_so)
        dV = tf.where(U > u_c, -V / tau_vp, (1 - V) / tau_vn)
        dW = tf.where(U > u_c, -W / tau_wp, tf.where(U > u_w, (1 - W) / tau_wn2, (1 - W) / tau_wn1))
        r_s = (r_sp - r_sn) * H(U - u_c) + r_sn
        dS = r_s * (0.5 * (1 + tf.tanh((U - u_csi) * k_)) - S)

        return dU, dV, dW, dS


    def solve(self, U, V, W, S, U0):
        """ Explicit Euler ODE solver """
        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
        with jit_scope():
            dU, dV, dW, dS = self.differentiate(U, V, W, S)

            U1 = U0 + self.dt * dU + self.diff * self.dt * self.laplace(U0)
            V1 = V + self.dt * dV
            W1 = W + self.dt * dW
            S1 = S + self.dt * dS

            return tf.group(
                U.assign(U1),
                V.assign(V1),
                W.assign(W1),
                S.assign(S1)
                )

    def define(self):
        """
            Create a tensorflow graph to run the Fenton 4v model

            Args:
                N: height (pixels)
                M: width (pixels)

            Returns:
                A model dict
        """
        # the initial values of the state variables
        u_init = np.zeros([self.height, self.width], dtype=np.float32)
        v_init = np.ones([self.height, self.width], dtype=np.float32)
        w_init = np.ones([self.height, self.width], dtype=np.float32)
        s_init = np.zeros([self.height, self.width], dtype=np.float32)

        # S1 stimulation: vertical along the left side
        u_init[:,1] = 1.0

        # prepare for S2 stimulation as part of the cross-stimulation protocol
        s2 = np.zeros([self.height, self.width], dtype=np.float32)
        s2[:self.height//2, :self.width//2] = 1.0

        # define the graph...
        with tf.device('/device:GPU:0'):
            # Create variables for simulation state
            U  = tf.Variable(u_init, name='U')
            V  = tf.Variable(v_init, name='V')
            W  = tf.Variable(w_init, name='W')
            S  = tf.Variable(s_init, name='S')

            # enforcing the no-flux boundary condition
            paddings = tf.constant([[1,1], [1,1]])
            U0 = tf.pad(U[1:-1,1:-1], paddings, 'SYMMETRIC', name='U0')

            self.ode_op = self.solve(U, V, W, S, U0)

            # Operation for S2 stimulation
            self.s2_op = U.assign(tf.maximum(U, s2))
            self.U = U

    def normalized_vlt(self):
        return self.U.eval()


class BeelerReuter(IonicModel):
    def __init__(self, props):
        IonicModel.__init__(self, props)
        self.min_v = -90.0    # mV
        self.max_v = 30.0     # mV
        self.ab_coef = np.array(
                [[0.0005, 0.083,  50.,    0.0,    0.0,    0.057,  1.0],   # ca_x1
                 [0.0013, -0.06,  20.,    0.0,    0.0,    -0.04,  1.0],   # cb_x1
                 [0.0000, 0.0,    47.,    -1.0,   47.,    -0.1,   -1.0],  # ca_m
                 [40.,    -0.056, 72.,    0.0,    0.0,    0.0,    0.0],   # cb_m
                 [0.126,  -.25,   77.,    0.0,    0.0,    0.0,    0.0],   # ca_h
                 [1.7,    0.0,    22.5,   0.0,    0.0,    -0.082, 1.0],   # cb_h
                 [0.055,  -.25,   78.0,   0.0,    0.0,    -0.2,   1.0],   # ca_j
                 [0.3,    0.0,    32.,    0.0,    0.0,    -0.1,   1.0],   # cb_j
                 [2*0.095,  -0.01,  -5.,    0.0,    0.0,    -0.072, 1.0],   # ca_d
                 [2*0.07,   -0.017, 44.,    0.0,    0.0,    0.05,   1.0],   # cb_d
                 [2*0.012,  -0.008, 28.,    0.0,    0.0,    0.15,   1.0],   # ca_f
                 [2*0.0065, -0.02,  30.,    0.0,    0.0,    -0.2,   1.0]],  # cb_f
                dtype=np.float32)

    def define(self):
        # the initial values of the state variables
        v_init = np.full([self.height, self.width], -84.624, dtype=np.float32)
        c_init = np.full([self.height, self.width], 1e-4, dtype=np.float32)
        m_init = np.full([self.height, self.width], 0.01, dtype=np.float32)
        h_init = np.full([self.height, self.width], 0.988, dtype=np.float32)
        j_init = np.full([self.height, self.width], 0.975, dtype=np.float32)
        d_init = np.full([self.height, self.width], 0.003, dtype=np.float32)
        f_init = np.full([self.height, self.width], 0.994, dtype=np.float32)
        xi_init = np.full([self.height, self.width], 0.0001, dtype=np.float32)

        # S1 stimulation: vertical along the left side
        v_init[:,1] = 10.0

        # prepare for S2 stimulation as part of the cross-stimulation protocol
        s2 = np.full([self.height, self.width], self.min_v, dtype=np.float32)
        s2[:self.height//2, :self.width//2] = 10.0

        # define the graph...
        with tf.device('/device:GPU:0'):
            # Create variables for simulation state
            V  = tf.Variable(v_init, name='V')
            C  = tf.Variable(c_init, name='C')
            M  = tf.Variable(m_init, name='M')
            H  = tf.Variable(h_init, name='H')
            J  = tf.Variable(j_init, name='J')
            D  = tf.Variable(d_init, name='D')
            F  = tf.Variable(f_init, name='F')
            XI  = tf.Variable(xi_init, name='XI')

            state = [V, C, M, H, J, D, F, XI]

            self.ode_op = self.solve(state, self.enforce_boundary(V))

            # Operation for S2 stimulation
            self.s2_op = V.assign(tf.maximum(V, s2))
            self.V = V


    def solve(self, state, V0):
        """ Explicit Euler ODE solver """
        V, C, M, H, J, D, F, XI = state
        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
        with jit_scope():
        #if True:
            if not self.cheby:
                xi_inf, xi_tau = self.calc_inf_tau(V0, self.ab_coef[0], self.ab_coef[1], 'xi')
                m_inf, m_tau = self.calc_inf_tau(V0, self.ab_coef[2], self.ab_coef[3], 'm')
                h_inf, h_tau = self.calc_inf_tau(V0, self.ab_coef[4], self.ab_coef[5], 'h')
                j_inf, j_tau = self.calc_inf_tau(V0, self.ab_coef[6], self.ab_coef[7], 'j')
                d_inf, d_tau = self.calc_inf_tau(V0, self.ab_coef[8], self.ab_coef[9], 'd')
                f_inf, f_tau = self.calc_inf_tau(V0, self.ab_coef[10], self.ab_coef[11], 'f')
            else:
                x = (V0 - 0.5*(self.max_v+self.min_v)) / (0.25*(self.max_v-self.min_v))
                T1x2 = tf.identity(x, name='T1x2')
                T2 = tf.subtract(0.5*T1x2*T1x2, 1.0, name='T2')
                T3 = tf.subtract(T1x2*T2, 0.5*T1x2, name='T3')
                T4 = tf.subtract(T1x2*T3, T2, name='T4')
                T5 = tf.subtract(T1x2*T4, T3, name='T5')
                T6 = tf.subtract(T1x2*T5, T4, name='T6')
                T7 = tf.subtract(T1x2*T6, T5, name='T7')
                T8a = tf.multiply(T1x2, T7, name='T8a')

                Ts = [T1x2, T2, T3, T4, T5, T6, T7, T8a]

                v, α, β = self.calc_alpha_beta_np()

                xi_inf = self.chebyshev_poly(Ts, v, α[:,0]/(α[:,0]+β[:,0]))
                m_inf = self.chebyshev_poly(Ts, v, α[:,1]/(α[:,1]+β[:,1]))
                h_inf = self.chebyshev_poly(Ts, v, α[:,2]/(α[:,2]+β[:,2]))
                j_inf = self.chebyshev_poly(Ts, v, α[:,3]/(α[:,3]+β[:,3]))
                d_inf = self.chebyshev_poly(Ts, v, α[:,4]/(α[:,4]+β[:,4]))
                f_inf = self.chebyshev_poly(Ts, v, α[:,5]/(α[:,5]+β[:,5]))

                xi_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,0]+β[:,0]))
                m_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,1]+β[:,1]))
                h_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,2]+β[:,2]))
                j_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,3]+β[:,3]))
                d_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,4]+β[:,4]))
                f_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,5]+β[:,5]))


            dt = self.dt
            XI1 = tf.clip_by_value(xi_inf - (xi_inf - XI) * tf.exp(-dt/xi_tau), 0.0, 1.0, name='XI1')
            M1 = tf.clip_by_value(m_inf - (m_inf - M) * tf.exp(-dt/m_tau), 0.0, 1.0, name='M1')
            H1 = tf.clip_by_value(h_inf - (h_inf - H) * tf.exp(-dt/h_tau), 0.0, 1.0, name='H1')
            J1 = tf.clip_by_value(j_inf - (j_inf - J) * tf.exp(-dt/j_tau), 0.0, 1.0, name='J1')
            D1 = tf.clip_by_value(d_inf - (d_inf - D) * tf.exp(-dt/d_tau), 0.0, 1.0, name='D1')
            F1 = tf.clip_by_value(f_inf - (f_inf - F) * tf.exp(-dt/f_tau), 0.0, 1.0, name='F1')

            # Current Multipliers
            C_K1 = 1.0
            C_x1 = 1.0
            C_Na = 1.0
            C_s = 1.0
            D_Ca = 0.0
            D_Na = 0.0
            g_s = 0.09
            g_Na = 4.0
            g_NaC = 0.005
            ENa = 50.0 + D_Na
            C_m = 1.0

            k = tf.exp(0.04 * V0, name='k')
            iK1 = (C_K1 * (0.35 *(4*(29.64*k - 1) / ( 69.41*k*k + 8.33*k) +
                        0.2 * ((V0 + 23) / (1 - 0.3985 / k )))))

            ix1 = (C_x1 * XI * 0.8 * (21.76*k - 1) / (4.055*k))

            iNa = C_Na * (g_Na*M*M*M*H*J + g_NaC) * (V0 - ENa)

            ECa = D_Ca - 82.3 - 13.0278 * tf.log(C)
            iCa = C_s * g_s * D * F * (V0 - ECa)

            I_sum = iK1 + ix1 + iNa + iCa

            V1 = tf.clip_by_value(V0 + self.diff * self.dt * self.laplace(V0) - dt * I_sum / C_m, -85.0, 25.0)

            dC = -1.0e-7*iCa + 0.07*(1.0e-7 - C)
            C1 = C + dt * dC

            # return tf.no_op()

            return tf.group(
                tf.assign(V, V1, name='set_V'),
                tf.assign(C, C1, name='set_C'),
                tf.assign(M, M1, name='set_M'),
                tf.assign(H, H1, name='set_H'),
                tf.assign(J, J1, name='set_J'),
                tf.assign(D, D1, name='set_D'),
                tf.assign(F, F1, name='set_F'),
                tf.assign(XI, XI1, name='set_X')
                )

    def calc_alpha_bata(self, v, c):
        if c[3] == 0:
            return ((c[0] * tf.exp(c[1]*(v+c[2]), name='exp_A')) /
                    (tf.exp(c[5]*(v+c[2]), name='exp_B') + c[6]))

        return ((c[0] * tf.exp(c[1]*(v+c[2]), name='exp_C') + c[3] * (v+c[4])) /
                (tf.exp(c[5]*(v+c[2]), name='exp_D') + c[6]))

    def calc_inf_tau(self, v, c, d, name):
        with tf.name_scope(name) as scope:
            alpha = self.calc_alpha_bata(v, c)
            beta = self.calc_alpha_bata(v, d)
            return (tf.realdiv(alpha, alpha+beta, name='inf'),
                    tf.reciprocal(alpha+beta, name='tau'))

    def calc_alpha_beta_np(self):
            v = np.linspace(self.min_v, self.max_v, 1001)
            x = np.outer(v, np.ones(self.ab_coef.shape[0]))
            y = ((self.ab_coef[:,0] * np.exp(self.ab_coef[:,1] * (x+self.ab_coef[:,2])) +
                    self.ab_coef[:,3] * (x+self.ab_coef[:,4])) /
                    (np.exp(self.ab_coef[:,5] * (x+self.ab_coef[:,2])) + self.ab_coef[:,6]))
            alpha = y[...,::2]
            beta = y[...,1::2]
            return v, alpha, beta

    def convert_chebyshev(self, x, y, deg=8):
        cheb = np.polynomial.chebyshev.Chebyshev.fit(x, y, deg)
        return cheb.coef

    def chebyshev_poly(self, Ts, x, y):
        c = self.convert_chebyshev(x, y)
        return (c[0] + (0.5*c[1])*Ts[0] + c[2]*Ts[1] + c[3]*Ts[2] + c[4]*Ts[3] +
                c[5]*Ts[4] + (c[6]-c[8])*Ts[5] + c[7]*Ts[6] + c[8]*Ts[7])

    def normalized_vlt(self):
        v = self.V.eval()
        return (v - self.min_v) / (self.max_v - self.min_v)



if __name__ == '__main__':
    props = {
        'width': 512,
        'height': 512,
        'dt': 0.1,
        'diff': 0.809,
        'samples': 20000,
        's2_time': 3000,
        'cheby': True,
        'timeline': False,
        'timeline_name': 'timeline_br.json',
        'save_graph': False
    }

    model = BeelerReuter(props)
    model.define()
    im = Screen(model.height, model.width, 'Beeler-Reuter Model')
    model.run(im)
