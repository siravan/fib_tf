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

class Courtemanche(IonicModel):
    """
        The modified Courtemanche atrial model
    """

    def __init__(self, props):
        super().__init__(props)
        self.min_v = -100.0    # mV
        self.max_v = 50.0     # mV

    def calc_inter(self, V, mod=np):
        R = 8.3143          # R in membrane (joule/mole_kelvin).
        T = 310             # T in (kelvin).
        F = 96.4867         # F in membrane (coulomb/millimole).
        Cm = 100            #  Cm in membrane (picoF).
        Na_o = 140          #  Na_o (millimolar).
        g_K1 = 0.09         #  g_K1 (nanoS/picoF).
        K_Q10 = 3           #  transient_outward_K_current (dimensionless).
        g_Kr = 0.029411765  #  rapid_delayed_rectifier_K_current (nanoS/picoF).
        Ca_o = 1.8          #  (millimolar).
        I_NaCa_max = 1600   #  Na_Ca_exchanger_current(picoA/picoF).
        K_mNa = 87.5        #  Na_Ca_exchanger_current (millimolar).
        K_mCa = 1.38        #  Na_Ca_exchanger_current (millimolar).
        K_sat = 0.1         #  Na_Ca_exchanger_current (dimensionless).
        gamma_ = 0.35       #  Na_Ca_exchanger_current (dimensionless).
        sigma = 1.0         #  sodium_potassium_pump (dimensionless).

        def where(cond, x, y):
            ret = mod.where(cond, x, y)
            if type(ret) is np.ndarray and ret.shape == ():
                ret = float(ret)
            return ret

        inter = {}

        inter['d_infinity'] = mod.reciprocal(1.0 + mod.exp((V + 10.0) / -8.0))

        # note: V + 10 is changed to V + 10.0001 to suppress a warning passing V = -10
        inter['tau_d'] = where(
            mod.abs(V + 10.0001) < 1.0e-10,
            4.579 / (1.0 + mod.exp((V + 10.0) / -6.24)),
            (1.0 - mod.exp((V + 10.0001) / -6.24)) / (0.0350000 * (V + 10.0001) * (1.0 + mod.exp((V + 10.0001) / -6.24)))
        )

        inter['f_infinity'] = mod.exp(-(V + 28.0) / 6.9) / (1.0 + mod.exp(-(V + 28.0) / 6.9))
        inter['tau_f'] = 9.0 * mod.reciprocal(0.0197000 * mod.exp(-mod.square(0.0337) * mod.square(V + 10.0)) + 0.02)

        inter['tau_w'] = where(
            mod.abs(V - 7.9) < 1.0e-10,
            (6.0 * 0.2) / 1.3,
            (6.0 * (1.0 - mod.exp(-(V - 7.9) / 5.0))) / ((1.0 + 0.3 * mod.exp(-(V - 7.9) / 5.0)) * 1.0 * (V - 7.9))
        )

        inter['w_infinity'] = 1.0 - mod.reciprocal(1.0 + mod.exp(-(V - 40.0) / 17.0))

        alpha_m = where(
            mod.abs(V - -47.13) < 0.001,
            3.2,
            (0.32 * (V + 47.13)) / (1.0 - mod.exp(-0.1 * (V + 47.13)))
        )

        beta_m = 0.08 * mod.exp(-V / 11.0)

        inter['m_inf'] = alpha_m / (alpha_m + beta_m)
        inter['tau_m'] = mod.reciprocal(alpha_m + beta_m)

        alpha_h = where(
            V < -40.0,
            0.135 * mod.exp((V + 80.0) / -6.8),
            0.0
        )

        beta_h = where(
            V < -40.0,
            3.56 * mod.exp(0.079 * V) + 310000. * mod.exp(0.35 * V),
            mod.reciprocal(0.13 * (1.0 + mod.exp((V + 10.66) / -11.1)))
        )

        inter['h_inf'] = alpha_h / (alpha_h + beta_h)
        inter['tau_h'] = mod.reciprocal(alpha_h + beta_h)

        alpha_j = where(
            V < -40.0,
            ((-127140. * mod.exp(0.2444 * V) - 3.474e-05 * mod.exp(-0.04391 * V)) * (V + 37.78)) / (1.0 + mod.exp(0.311 * (V + 79.23))),
            0.0
        )

        beta_j = where(
            V < -40.0,
            (0.1212 * mod.exp(-0.01052 * V)) / (1.0 + mod.exp(-0.1378 * (V + 40.14))),
            (0.3 * mod.exp(-2.535e-07 * V)) / (1.0 + mod.exp(-0.1 * (V + 32.0)))
        )

        inter['j_inf'] = alpha_j / (alpha_j + beta_j)
        inter['tau_j'] = mod.reciprocal(alpha_j + beta_j)

        alpha_oa = 0.65 * mod.reciprocal(mod.exp((V - -10.0) / -8.5) + mod.exp(((V - -10.0) - 40.0) / -59.0))
        beta_oa = 0.65 * mod.reciprocal(2.5 + mod.exp(((V - -10.0) + 72.0) / 17.0))

        inter['tau_oa'] = mod.reciprocal(alpha_oa + beta_oa) / K_Q10
        inter['oa_infinity'] = mod.reciprocal(1.0 + mod.exp(((V - -10.0) + 10.47) / -17.54))

        alpha_oi = mod.reciprocal(18.53 + 1.0 * mod.exp(((V - -10.0) + 103.7) / 10.95))
        beta_oi = mod.reciprocal(35.56 + 1.0 * mod.exp(((V - -10.0) - 8.74) / -7.44))

        inter['tau_oi'] = mod.reciprocal(alpha_oi + beta_oi) / K_Q10
        inter['oi_infinity'] = mod.reciprocal(1.0 + mod.exp(((V - -10.0) + 33.1) / 5.3))

        alpha_ua = 0.65 * mod.reciprocal(mod.exp((V - -10.0) / -8.5) + mod.exp(((V - -10.0) - 40.0) / -59.0))
        beta_ua = 0.65 * mod.reciprocal(2.5 + mod.exp(((V - -10.0) + 72.0) / 17.0))

        inter['tau_ua'] = mod.reciprocal(alpha_ua + beta_ua) / K_Q10
        inter['ua_infinity'] = mod.reciprocal(1.0 + mod.exp(((V - -10.0) + 20.3) / -9.6))

        alpha_ui = mod.reciprocal(21.0 + 1.0 * mod.exp(((V - -10.0) - 195.000) / -28.0))
        beta_ui = mod.reciprocal(mod.exp(((V - -10.0) - 168.0) / -16.0))

        inter['tau_ui'] = mod.reciprocal(alpha_ui + beta_ui) / K_Q10
        inter['ui_infinity'] = mod.reciprocal(1.0 + mod.exp(((V - -10.0) - 109.45) / 27.48))

        alpha_xr = where(
            mod.abs(V + 14.1) < 1.0e-10,
            0.0015,
            (0.0003 * (V + 14.1)) / (1.0 - mod.exp((V + 14.1) / -5.0))
        )

        beta_xr = where(
            mod.abs(V - 3.3328) < 1.0e-10,
            0.000378361,
            (7.3898e-05 * (V - 3.3328)) / (mod.exp((V - 3.3328) / 5.1237) - 1.0)
        )

        inter['tau_xr'] = mod.reciprocal(alpha_xr + beta_xr);
        inter['xr_infinity'] = mod.reciprocal(1.0 + mod.exp((V + 14.1) / -6.5))

        alpha_xs = where(
            mod.abs(V - 19.9) < 1.0e-10,
            0.00068,
            (4.0e-05 * (V - 19.9)) / (1.0 - mod.exp((V - 19.9) / -17.0))
        )

        beta_xs = where(
            mod.abs(V - 19.9) < 1.0e-10,
            0.000315,
            (3.5e-05 * (V - 19.9)) / (mod.exp((V - 19.9) / 9.0) - 1.0)
        )

        inter['tau_xs'] = 0.5 * mod.reciprocal(alpha_xs + beta_xs)
        inter['xs_infinity'] = mod.sqrt(mod.reciprocal(1.0 + mod.exp((V - 19.9) / -12.7)))

        inter['g_Kur'] = 0.005 + 0.05 / (1.0 + mod.exp((V - 15.0) / -13.0))

        inter['f_NaK'] = mod.reciprocal(1.0 + 0.1245 * mod.exp((-0.1 * F * V) / (R * T)) + 0.0365 * sigma * mod.exp((-F * V) / (R * T)))

        i_NaCad = (K_mNa*K_mNa*K_mNa + Na_o*Na_o*Na_o) * (K_mCa + Ca_o) * (1.0 + K_sat * mod.exp(((gamma_ - 1.0) * V * F) / (R * T)))

        inter['i_NaCaa'] = (Cm * I_NaCa_max * (mod.exp((gamma_ * F * V) / (R * T)) * Ca_o)) / i_NaCad

        inter['i_NaCab'] = (Cm * I_NaCa_max * (mod.exp(((gamma_ - 1.0) * F * V) / (R * T)) * (Na_o*Na_o*Na_o))) / i_NaCad

        inter['i_K1a'] = (Cm * g_K1) / (1.0 + mod.exp(0.07 * (V + 80.0)))  # * (V - E_K)

        inter['i_Kra'] = (Cm * g_Kr) / (1.0 + mod.exp((V + 15.0) / 22.4))  # * state[_xr_] * (V - E_K)

        return inter

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
        a #= np.diag(a)    # S_i is the leading term of T_i
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
