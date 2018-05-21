#!/usr/bin/env python
"""
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler

    Copyright 2017-2018 Shahriar Iravanian (siravan@emory.edu)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
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
        super().__init__(props)
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
        super().define()
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

            states = [(V, C, M, H, J, D, F, XI)]

            if self.skip:
                s = self.solve(states[0], 5)
                states.append(s)
                for i in range(4):
                    states.append(self.solve(states[-1], 0))
                self.dt_per_step = 5
            else:
                for i in range(5):
                    states.append(self.solve(states[-1], 1))
                self.dt_per_step = 5

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

            self._V = V  # V is needed by self.image()


    def solve(self, state, n=1):
        """ Explicit Euler ODE solver """
        V, C, M, H, J, D, F, XI = state
        V0 = self.enforce_boundary(V)
        dt = self.dt

        with self.jit_scope():
            if self.cheby:
                M1, H1, J1, D1, F1, XI1 = self.update_gates_with_cheby(V0, M, H, J, D, F, XI, n)
            else:
                M1, H1, J1, D1, F1, XI1 = self.update_gates_without_cheby(V0, M, H, J, D, F, XI, n)

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

            return (V1, C1, M1, H1, J1, D1, F1, XI1)

    def update_gates_without_cheby(self, V0, M, H, J, D, F, XI, n):
        """
            updates the six gating variables without using the Chebyshev Polynomials
            n is the number of steps to advance the slow gates
            m and h are always advanced one step
        """
        dt = self.dt

        m_inf, m_tau = self.calc_inf_tau(V0, self.ab_coef[2], self.ab_coef[3], 'm')
        h_inf, h_tau = self.calc_inf_tau(V0, self.ab_coef[4], self.ab_coef[5], 'h')

        M1 = self.rush_larsen(M, m_inf, m_tau, dt, name='M1')
        H1 = self.rush_larsen(H, h_inf, h_tau, dt, name='H1')

        if n > 0:
            xi_inf, xi_tau = self.calc_inf_tau(V0, self.ab_coef[0], self.ab_coef[1], 'xi')
            j_inf, j_tau = self.calc_inf_tau(V0, self.ab_coef[6], self.ab_coef[7], 'j')
            d_inf, d_tau = self.calc_inf_tau(V0, self.ab_coef[8], self.ab_coef[9], 'd')
            f_inf, f_tau = self.calc_inf_tau(V0, self.ab_coef[10], self.ab_coef[11], 'f')

            XI1 = self.rush_larsen(XI, xi_inf, xi_tau, dt*n)
            J1 = self.rush_larsen(J, j_inf, j_tau, dt*n)
            D1 = self.rush_larsen(D, d_inf, d_tau, dt*n)
            F1 = self.rush_larsen(F, f_inf, f_tau, dt*n)
        else:   # n == 0
            XI1 = XI
            J1 = J
            D1 = D
            F1 = F

        return M1, H1, J1, D1, F1, XI1

    def update_gates_with_cheby(self, V0, M, H, J, D, F, XI, n, deg=8):
        """
            updates the six gating variables using the Chebyshev Polynomials
            n is the number of steps to advance the slow gates
            m and h are always advanced one step
        """
        dt = self.dt
        # note: x ranges from -1 to +1 and is input to Chebyshev polynomials
        x = (V0 - 0.5*(self.max_v+self.min_v)) / (0.5*(self.max_v-self.min_v))

        # Ts is a list [S_0,...,S_{\deg}], where S_i is the leading term of
        # S_i
        Ts = self.calc_chebyshev_leading(x, 8)

        v, α, β = self.calc_alpha_beta_np()
        
        m_inf = self.expand_chebyshev(Ts, v, α[:,1]/(α[:,1]+β[:,1]))
        h_inf = self.expand_chebyshev(Ts, v, α[:,2]/(α[:,2]+β[:,2]))
        m_tau = self.expand_chebyshev(Ts, v, 1.0/(α[:,1]+β[:,1]))
        h_tau = self.expand_chebyshev(Ts, v, 1.0/(α[:,2]+β[:,2]))

        M1 = self.rush_larsen(M, m_inf, m_tau, dt, name='M1')
        H1 = self.rush_larsen(H, h_inf, h_tau, dt, name='H1')

        if n > 0:
            xi_inf = self.expand_chebyshev(Ts, v, α[:,0]/(α[:,0]+β[:,0]))
            j_inf = self.expand_chebyshev(Ts, v, α[:,3]/(α[:,3]+β[:,3]))
            d_inf = self.expand_chebyshev(Ts, v, α[:,4]/(α[:,4]+β[:,4]))
            f_inf = self.expand_chebyshev(Ts, v, α[:,5]/(α[:,5]+β[:,5]))

            xi_tau = self.expand_chebyshev(Ts, v, 1.0/(α[:,0]+β[:,0]))
            j_tau = self.expand_chebyshev(Ts, v, 1.0/(α[:,3]+β[:,3]))
            d_tau = self.expand_chebyshev(Ts, v, 1.0/(α[:,4]+β[:,4]))
            f_tau = self.expand_chebyshev(Ts, v, 1.0/(α[:,5]+β[:,5]))

            XI1 = self.rush_larsen(XI, xi_inf, xi_tau, dt*n)
            J1 = self.rush_larsen(J, j_inf, j_tau, dt*n)
            D1 = self.rush_larsen(D, d_inf, d_tau, dt*n)
            F1 = self.rush_larsen(F, f_inf, f_tau, dt*n)
        else:   # n == 0
            XI1 = XI
            J1 = J
            D1 = D
            F1 = F

        return M1, H1, J1, D1, F1, XI1


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

    def calc_chebyshev_leading(self, x, deg):
        """
            calculates S_i (the leading term of T_i) based on
            an input tensor x
        """
        assert(deg > 1)
        T0 = 1.0
        T1 = x
        Ts = [T0, T1]
        for i in range(deg-1):
            T = 2*x*Ts[-1]
            Ts.append(T)
        return Ts

    def expand_chebyshev(self, Ts, x, y, deg=0):
        """
            Defines an order deg Chebyshev polymonial estimating y sampled at x

            Ts is a list of [S_0,...,S_{\deg}] of S_i tensors, where
            S_i is the leading terms of the Chebyshev polynomial T_i.
        """
        if deg == 0:
            deg = len(Ts) - 1

        # c is the coefficients of the least squares fit to
        # the data y sampled at x
        c = np.polynomial.chebyshev.Chebyshev.fit(x, y, deg).coef

        # a is the chebyshev polynomials coefficients
        # a[i,j] is the coefficient of x^j in T_i
        a = np.zeros([deg+1, deg+1], dtype=np.int)
        a[0,0] = 1      # T_0 = 1
        a[1,1] = 1      # T_1 = x
        for i in range(2, deg+1):
            a[i,1:] += 2*a[i-1,:-1]     # + 2x T_{i-1}
            a[i,:] -= a[i-2,:]          # - T_{i-2}
        a //= np.diag(a)    # S_i is the leading term of T_i
        # transform best fit coefficient from a T_i basis to an S_i basis
        d = np.matmul(np.transpose(a), c)

        r = d[0]    # note T_0 = S_0 = Ts[0] == 1.0
        for i in range(1, deg+1):
            r += d[i] * Ts[i]
        return r

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
    config = {
        'width': 512,           # screen width in pixels
        'height': 512,          # screen height in pixels
        'dt': 0.1,              # integration time step in ms
        'dt_per_plot': 10,      # screen refresh interval in dt unit
        'diff': 0.809,          # diffusion coefficient
        'duration': 1000,       # simulation duration in ms
        'skip': False,          # optimization flag: activate multi-rate
        'cheby': True,          # optimization flag: activate Chebysheb polynomials
        'timeline': False,      # flag to save a timeline (profiler)
        'timeline_name': 'timeline_br.json',
        'save_graph': False     # flag to save the dataflow graph
    }

    model = BeelerReuter(config)
    model.add_hole_to_phase_field(150, 200, 40) # center=(150,200), radius=40
    model.define()
    model.add_pace_op('s2', 'luq', 10.0)

    # note: change the following line to im = None to run without a screen
    # im = None
    im = Screen(model.height, model.width, 'Beeler-Reuter Model')

    s2 = model.millisecond_to_step(300)     # 300 ms

    for i in model.run(im):
        if i == s2:
            model.fire_op('s2')
