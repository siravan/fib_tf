#!/home/shahriar/anaconda3/bin/python
"""
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler
    @2017 Shahriar Iravanian (siravan@emory.edu)
"""

import tensorflow as tf
import numpy as np
from screen import Screen
from ionic import IonicModel

class BeelerReuter(IonicModel):
    """
        The modified 8-variable Beeler-Reuter model:

        Reconstruction of the action potential of ventricular myocardial fibres,
        Beeler, G.W. and Reuter, H. 1977, Journal of Physiology, 268, 177-210.
        PubMed ID: 874889
    """

    def __init__(self, props):
        IonicModel.__init__(self, props)
        self.min_v = -90.0    # mV
        self.max_v = 30.0     # mV

        # note: the first column for d and f gates is multiplies by 2
        # to reduce the calcium current activation/deactivation gates by
        # 50% in order to have spiral waves
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

    def define(self, s1=True):
        """
            Defines the tensorflow model
            It sets ode_op, s2_op and V used by other methods
        """
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
        if s1:
            v_init[:,1] = 10.0

        # prepare for S2 stimulation as part of the cross-stimulation protocol
        s2 = np.full([self.height, self.width], self.min_v, dtype=np.float32)
        # s2[:self.height//2, :self.width//2] = 10.0
        s2[:self.height//4, :self.width//2] = 10.0
        s2[self.height//4*3:, :self.width//2] = 10.0

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

            states = [[V, C, M, H, J, D, F, XI]]

            if self.skip:
                s = self.solve(states[0], self.enforce_boundary(states[0][0]), 10)
                states.append(s)
                for i in range(9):
                    V0 = self.enforce_boundary(states[-1][0])
                    # V0 = states[-1][0]
                    states.append(self.solve(states[-1], V0, 0))
            else:
                s = self.solve(states[0], self.enforce_boundary(states[0][0]), 1)
                states.append(s)
                for i in range(4):
                    V0 = self.enforce_boundary(states[-1][0])
                    # V0 = states[-1][0]
                    states.append(self.solve(states[-1], V0, 1))

            V1, C1, M1, H1, J1, D1, F1, XI1 = states[-1]

            self._ode_op = tf.group(
                tf.assign(V, V1, name='set_V'),
                tf.assign(C, C1, name='set_C'),
                tf.assign(M, M1, name='set_M'),
                tf.assign(H, H1, name='set_H'),
                tf.assign(J, J1, name='set_J'),
                tf.assign(D, D1, name='set_D'),
                tf.assign(F, F1, name='set_F'),
                tf.assign(XI, XI1, name='set_X')
                )

            # Operation for S2 stimulation
            self._s2_op = V.assign(tf.maximum(V, s2))
            self._V = V  # V is needed by self.image()


    def solve(self, state, V0, n=1):
        """ Explicit Euler ODE solver """
        V, C, M, H, J, D, F, XI = state
        with self.jit_scope():
        #if True:
            # calculating Chebyshev polynomials of the first kind
            # using T_0(x) = 1, T_1(x) = 2x, and the recurrence relationship
            # T_n(x) = 2xT_{n-1} - T_{n-2}
            # note: x ranges from -2 to 2 and is twice the actual Chebyshev input
            x = (V0 - 0.5*(self.max_v+self.min_v)) / (0.25*(self.max_v-self.min_v))
            # T1x2 is T1 x 2
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

            dt = self.dt

            m_inf = self.chebyshev_poly(Ts, v, α[:,1]/(α[:,1]+β[:,1]))
            h_inf = self.chebyshev_poly(Ts, v, α[:,2]/(α[:,2]+β[:,2]))
            m_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,1]+β[:,1]))
            h_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,2]+β[:,2]))

            M1 = self.rush_larsen(M, m_inf, m_tau, dt, name='M1')
            H1 = self.rush_larsen(H, h_inf, h_tau, dt, name='H1')

            if n > 0:
                xi_inf = self.chebyshev_poly(Ts, v, α[:,0]/(α[:,0]+β[:,0]))
                j_inf = self.chebyshev_poly(Ts, v, α[:,3]/(α[:,3]+β[:,3]))
                d_inf = self.chebyshev_poly(Ts, v, α[:,4]/(α[:,4]+β[:,4]))
                f_inf = self.chebyshev_poly(Ts, v, α[:,5]/(α[:,5]+β[:,5]))

                xi_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,0]+β[:,0]))
                j_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,3]+β[:,3]))
                d_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,4]+β[:,4]))
                f_tau = self.chebyshev_poly(Ts, v, 1.0/(α[:,5]+β[:,5]))

                XI1 = self.rush_larsen(XI, xi_inf, xi_tau, dt*n)
                J1 = self.rush_larsen(J, j_inf, j_tau, dt*n)
                D1 = self.rush_larsen(D, d_inf, d_tau, dt*n)
                F1 = self.rush_larsen(F, f_inf, f_tau, dt*n)
            else:
                XI1 = XI
                J1 = J
                D1 = D
                F1 = F

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

            V1 = (tf.clip_by_value(V0 + self.diff * self.dt * self.laplace(V0)
                    - dt * I_sum / C_m, -85.0, 25.0))

            dC = -1.0e-7*iCa + 0.07*(1.0e-7 - C)
            C1 = C + dt * dC

            return [V1, C1, M1, H1, J1, D1, F1, XI1]


    def calc_alpha_bata_tf(self, v, c):
        """
            direct calculation of α/β in the non-Chebychev mode
        """
        if c[3] == 0:
            return ((c[0] * tf.exp(c[1]*(v+c[2]), name='exp_A')) /
                    (tf.exp(c[5]*(v+c[2]), name='exp_B') + c[6]))

        return ((c[0] * tf.exp(c[1]*(v+c[2]), name='exp_C') + c[3] * (v+c[4])) /
                (tf.exp(c[5]*(v+c[2]), name='exp_D') + c[6]))

    def calc_inf_tau(self, v, c, d, name):
        """
            direct calculation of inf/tau in the non-Chebychev mode
        """
        with tf.name_scope(name) as scope:
            α = self.calc_alpha_bata_tf(v, c)
            β = self.calc_alpha_bata_tf(v, d)
            return α/(α+β), 1.0/(α+β)

    def calc_alpha_beta_np(self):
        """
            Defintion time calculation of α and β to be used
            by the Chebyshev routines
        """
        v = np.linspace(self.min_v, self.max_v, 1001)
        x = np.outer(v, np.ones(self.ab_coef.shape[0]))
        y = ((self.ab_coef[:,0] * np.exp(self.ab_coef[:,1] * (x+self.ab_coef[:,2])) +
                self.ab_coef[:,3] * (x+self.ab_coef[:,4])) /
                (np.exp(self.ab_coef[:,5] * (x+self.ab_coef[:,2])) + self.ab_coef[:,6]))
        alpha = y[...,::2]
        beta = y[...,1::2]
        return v, alpha, beta

    def chebyshev_poly(self, Ts, x, y, deg=8):
        """
            Defines an 8 order Chebyshev polymonial estimating y at x

            Ts is a list of Tensors as [T1x2, T2, T3, T4, T5, T6, T7, T8a],
            where T1x2 is 2*T1, T8a is T8 + T6, and Tn is the nth
            Chebyshev polynomial of the first kind. The reason for T1x2 and T8a
            instead of T1 and T8 is fewer multipilications to improve
            performance.
        """
        c = np.polynomial.chebyshev.Chebyshev.fit(x, y, deg).coef
        return (c[0] + (0.5*c[1])*Ts[0] + c[2]*Ts[1] + c[3]*Ts[2] + c[4]*Ts[3] +
                c[5]*Ts[4] + (c[6]-c[8])*Ts[5] + c[7]*Ts[6] + c[8]*Ts[7])

    def pot(self):
        return self._V

    def image(self):
        """
            Returns a [height x width] float ndarray in the range 0 to 1
            that encodes V in grayscale
        """
        v = self._V.eval()
        return (v - self.min_v) / (self.max_v - self.min_v)



if __name__ == '__main__':
    if True:
        config = {
            'width': 512,
            'height': 512,
            'dt': 0.1,
            'skip': True,
            'dt_per_plot': 1,
            'diff': 2.0,
            'samples': 2000,
            's2_time': 300,
            'cheby': True,
            'timeline': False,
            'timeline_name': 'timeline_br.json',
            'save_graph': False
        }
    else:
        config = {
            'width': 512,
            'height': 512,
            'dt': 0.1,
            'skip': False,
            'dt_per_plot': 2,
            'diff': 0.809,
            'samples': 4000,
            's2_time': 600,
            'cheby': True,
            'timeline': False,
            'timeline_name': 'timeline_br.json',
            'save_graph': False
        }

    model = BeelerReuter(config)
    model.define()
    # note: change the following line to im = None to run without a screen
    im = Screen(model.height, model.width, 'Beeler-Reuter Model')
    model.run(im)
