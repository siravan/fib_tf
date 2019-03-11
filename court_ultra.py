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
from functools import partial

class Courtemanche(IonicModel):
    """
        The modified Courtemanche atrial model
    """

    def __init__(self, props):
        super().__init__(props)
        self.min_v = -100.0     # mV
        self.max_v = 50.0       # mV
        self.depol = -81.0      # mV
        self.chronic = True
        self.fast_states = ['V', '_Na_i_', '_m_', '_h_']

    def init_state_variable(self, state, name, value):
        if name in state:
            print('Warning! The state variable arlready exists')
        state[name] = np.full([self.height, self.width], value, dtype=np.float32)

    def define(self, s1=True, state=None):
        """
            Defines the tensorflow model
            It sets ode_op, s2_op and V used by other methods
        """
        super().define()
        if state is None:
            state = {}
            self.init_state_variable(state, 'V', -81.18)
            self.init_state_variable(state, '_Na_i_', 1.117e+01)
            # self.init_state_variable(state, '_Na_i_', 1.3e+01)
            self.init_state_variable(state, '_m_', 2.98e-3)
            self.init_state_variable(state, '_h_', 9.649e-1)
            self.init_state_variable(state, '_j_', 9.775e-1)
            self.init_state_variable(state, '_K_i_', 1.39e+02)
            self.init_state_variable(state, '_oa_', 3.043e-2)
            self.init_state_variable(state, '_oi_', 9.992e-1)
            self.init_state_variable(state, '_ua_', 4.966e-3)
            self.init_state_variable(state, '_ui_', 9.986e-1)
            self.init_state_variable(state, '_xr_', 3.296e-5)
            self.init_state_variable(state, '_xs_', 1.869e-2)
            self.init_state_variable(state, '_Ca_i_', 1.013e-4)
            self.init_state_variable(state, '_d_', 1.367e-4)
            self.init_state_variable(state, '_f_', 9.996e-1)
            self.init_state_variable(state, '_f_Ca_', 7.755e-1)
            self.init_state_variable(state, '_Ca_rel_', 1.488)
            self.init_state_variable(state, '_u_', 0.0)
            self.init_state_variable(state, '_v_', 1.0)
            self.init_state_variable(state, '_w_', 0.9992)
            self.init_state_variable(state, '_Ca_up_', 1.488)

            if self.ultra_slow:
                self.init_state_variable(state, '_us_', 0.72)   # steady-state at 500 ms

            # S1 stimulation: vertical along the left side
            if s1:
                state['V'][:,:25] = 20.0

        # define the graph...
        with tf.device('/device:GPU:0'):
            # Create variables for simulation state
            State = {}
            for s in state:
                State[s] = tf.Variable(state[s])

            State1, inter = self.solve(State)
            self.dt_per_step = 1

            fasts = []
            slows = []

            # for s in State:
            #     if s in self.fast_states:
            #         fasts.append(tf.assign(State[s], State1[s]))
            #     else:
            #         slows.append(tf.assign(State[s], State1[s]))

            for s in State:
                fasts.append(tf.assign(State[s], State1[s]))

            self._ode_op = tf.group(*fasts)
            self._ops['slow'] = tf.group(*slows)
            self._V = State['V']  # V is needed by self.image()
            self._State = State
            self._Inter = inter

            Trend = tf.Variable(np.zeros([2], dtype=np.float32))
            self._ops['trend'] = tf.group(
                tf.assign(Trend[0], self._V[self.width//2,self.height//8])
                # tf.assign(Trend[1], State['_us_'][self.width//2,self.height//8])
            )
            self._Trend = Trend


    def euler(self, g, Rate, dt):
        return g + Rate * dt

    def δt(self, name):
        return self.dt
        # if name in self.fast_states:
        #     return self.dt
        # else:
        #     return self.dt * 10

    def solve(self, State):
        """ Explicit Euler ODE solver """
        V0 = State['V']
        V = self.enforce_boundary(V0)

        R = 8.3143              # (joule/mole_kelvin).
        T = 310;                # (kelvin).
        F = 96.4867             # (coulomb/millimole).
        Cm = 100                # Cm is Cm in membrane (picoF).
        g_Na = 7.8              # fast_sodium_current (nanoS/picoF).
        Na_o = 140              # (millimolar).
        K_o = 5.4               # (millimolar).
        g_to = 0.1652           # transient_outward_K_current (nanoS/picoF).
        g_Ks = 0.12941176       # slow_delayed_rectifier_K_current (nanoS/picoF).
        g_Ca_L = 0.12375        # L_type_Ca_channel (nanoS/picoF).
        Km_Na_i = 10            # sodium_potassium_pump (millimolar).
        Km_K_o = 1.5            # sodium_potassium_pump (millimolar).
        i_NaK_max = 0.59933874  # sodium_potassium_pump (picoA/picoF).
        i_CaP_max = 0.275       # sarcolemmal_calcium_pump_current (picoA/picoF).
        g_B_Na = 0.0006744375   # background_currents (nanoS/picoF).
        g_B_Ca = 0.001131       # background_currents (nanoS/picoF).
        g_B_K = 0               # background_currents (nanoS/picoF).
        Ca_o = 1.8              # (millimolar).
        K_rel = 30              # Ca_release_current_from_JSR (per_millisecond).
        tau_tr = 180            # transfer_current_from_NSR_to_JSR (millisecond).
        I_up_max = 0.005        # Ca_uptake_current_by_the_NSR (millimolar/millisecond).
        K_up = 0.00092          # Ca_uptake_current_by_the_NSR (millimolar).
        Ca_up_max = 15          # Ca_leak_current_by_the_NSR (millimolar).
        CMDN_max = 0.05         # CMDN_max is CMDN_max in Ca_buffers (millimolar).
        TRPN_max = 0.07         # TRPN_max is TRPN_max in Ca_buffers (millimolar).
        CSQN_max = 10           # CSQN_max is CSQN_max in Ca_buffers (millimolar).
        Km_CMDN = 0.00238       # Km_CMDN is Km_CMDN in Ca_buffers (millimolar).
        Km_TRPN = 0.0005        # Km_TRPN is Km_TRPN in Ca_buffers (millimolar).
        Km_CSQN = 0.8           # Km_CSQN is Km_CSQN in Ca_buffers (millimolar).
        V_cell = 20100          # intracellular_ion_concentrations (micrometre_3).
        V_i = V_cell * 0.68     # intracellular_ion_concentrations (micrometre_3).
        tau_f_Ca = 2.0          # L_type_Ca_channel_f_Ca_gate (millisecond).
        tau_u = 8.0             # Ca_release_current_from_JSR_u_gate (millisecond).
        V_rel = 0.0048 * V_cell # intracellular_ion_concentrations (micrometre_3).
        V_up = 0.0552 * V_cell  # intracellular_ion_concentrations (micrometre_3).

        State1 = {}

        if self.chronic:
            chronic = 1.0
        else:
            chronic = 0.0

        with self.jit_scope():
            inter = self.calc_inter(V, tf)

            State1['_d_'] = self.rush_larsen(State['_d_'], inter['d_infinity'], inter['tau_d'], self.δt('_d_'))
            State1['_f_'] = self.rush_larsen(State['_f_'], inter['f_infinity'], inter['tau_f'], self.δt('_f_'))
            State1['_w_'] = self.rush_larsen(State['_w_'], inter['w_infinity'], inter['tau_w'], self.δt('_d_'))
            State1['_m_'] = self.rush_larsen(State['_m_'], inter['m_inf'], inter['tau_m'], self.δt('_m_'))
            State1['_h_'] = self.rush_larsen(State['_h_'], inter['h_inf'], inter['tau_h'], self.δt('_h_'))
            State1['_j_'] = self.rush_larsen(State['_j_'], inter['j_inf'], inter['tau_j'], self.δt('_j_'))
            State1['_oa_'] = self.rush_larsen(State['_oa_'], inter['oa_infinity'], inter['tau_oa'], self.δt('_oa_'))
            State1['_oi_'] = self.rush_larsen(State['_oi_'], inter['oi_infinity'], inter['tau_oi'], self.δt('_oi_'))
            State1['_ua_'] = self.rush_larsen(State['_ua_'], inter['ua_infinity'], inter['tau_ua'], self.δt('_ua_'))
            State1['_ui_'] = self.rush_larsen(State['_ui_'], inter['ui_infinity'], inter['tau_ui'], self.δt('_ui_'))
            State1['_xr_'] = self.rush_larsen(State['_xr_'], inter['xr_infinity'], inter['tau_xr'], self.δt('_xr_'))
            State1['_xs_'] = self.rush_larsen(State['_xs_'], inter['xs_infinity'], inter['tau_xs'], self.δt('_xs_'))

            if self.ultra_slow:
                State1['_us_'] = self.rush_larsen(State['_us_'], inter['us_infinity'], inter['tau_us'], self.δt('_us_'))

            f_Ca_infinity = tf.reciprocal(1.0 + State['_Ca_i_'] / 0.00035)
            State1['_f_Ca_'] = self.rush_larsen(State['_f_Ca_'], f_Ca_infinity, tau_f_Ca, self.δt('_f_Ca_'))

            E_K = ((R * T) / F) * tf.log(K_o / State['_K_i_'])
            i_K1 = inter['i_K1a'] * (V - E_K)
            i_to = (1.0-0.5*chronic) * Cm * g_to * tf.pow(State['_oa_'], 3) * State['_oi_'] * (V - E_K)
            i_Kur = (1.0-0.5*chronic) * Cm * inter['g_Kur'] * tf.pow(State['_ua_'], 3) * State['_ui_'] * (V - E_K)
            i_Kr = inter['i_Kra'] * State['_xr_'] * (V - E_K)
            i_Ks = Cm * g_Ks * tf.square(State['_xs_']) * (V - E_K)
            i_NaK = ((Cm * i_NaK_max * inter['f_NaK']) / (1.0 + tf.sqrt(tf.pow(Km_Na_i / State['_Na_i_'], 3.0)))) * (K_o / (K_o + Km_K_o))
            i_B_K = Cm * g_B_K * (V - E_K)

            State1['_K_i_'] = self.euler(
                State['_K_i_'],
                (2.0 * i_NaK - (i_K1 + i_to + i_Kur + i_Kr + i_Ks + i_B_K)) / (V_i * F),
                self.δt('_K_i_')
            )

            E_Na = ((R * T) / F) * tf.log(Na_o / State['_Na_i_'])
            i_Na = Cm * g_Na * tf.pow(State['_m_'], 3) * State['_h_'] * State['_j_'] * (V - E_Na)
            if self.ultra_slow:
                i_Na *= State['_us_']

            i_NaCa = inter['i_NaCaa'] * tf.pow(State['_Na_i_'], 3) - inter['i_NaCab'] * State['_Ca_i_']
            i_B_Na = Cm * g_B_Na * (V - E_Na)

            State1['_Na_i_'] = self.euler(
                State['_Na_i_'],
                (-3.0 * i_NaK - (3.0 * i_NaCa + i_B_Na + i_Na)) / (V_i * F),
                self.δt('_Na_i_')
            )

            i_st = 0.0
            i_Ca_L = (1.0-0.7*chronic) * Cm * g_Ca_L * State['_d_'] * State['_f_'] * State['_f_Ca_'] * (V - 65.0)
            i_CaP = (Cm * i_CaP_max * State['_Ca_i_']) / (0.0005 + State['_Ca_i_'])
            E_Ca = ((R * T) / (2.0 * F)) * tf.log(Ca_o / State['_Ca_i_'])
            i_B_Ca = Cm * g_B_Ca * (V - E_Ca)

            DV = self.euler(
                V,
                -(i_Na + i_K1 + i_to + i_Kur + i_Kr + i_Ks + i_B_Na + i_B_Ca + i_NaK + i_CaP + i_NaCa + i_Ca_L + i_st) / Cm,
                self.δt('V')
            )

            State1['V'] = DV + self.diff * self.δt('V') * self.laplace(V)


            i_rel = K_rel * tf.square(State['_u_']) * State['_v_'] * State['_w_'] * (State['_Ca_rel_'] - State['_Ca_i_'])
            i_tr = (State['_Ca_up_'] - State['_Ca_rel_']) / tau_tr

            State1['_Ca_rel_'] = self.euler(
                State['_Ca_rel_'],
                (i_tr - i_rel) * tf.reciprocal(1.0 + (CSQN_max * Km_CSQN) / tf.square(State['_Ca_rel_'] + Km_CSQN)),
                self.δt('_Ca_rel_')
            )

            Fn = 1000.0 * (1.0e-15 * V_rel * i_rel - (1.0e-15 / (2.0 * F)) * (0.5 * i_Ca_L - 0.2 * i_NaCa))
            u_infinity = tf.reciprocal(1.0 + tf.exp(-(Fn - 3.4175e-13) / 1.367e-15))
            State1['_u_'] = self.rush_larsen(State['_u_'], u_infinity, tau_u, self.δt('_u_'))

            tau_v = 1.91 + 2.09 * u_infinity
            v_infinity = 1.0 - tf.reciprocal(1.0 + tf.exp(-(Fn - 6.835e-14) / 1.367e-15))
            State1['_v_'] = self.rush_larsen(State['_v_'], v_infinity, tau_v, self.δt('_v_'))

            i_up = I_up_max / (1.0 + K_up / State['_Ca_i_'])
            i_up_leak = (I_up_max * State['_Ca_up_']) / Ca_up_max

            State1['_Ca_up_'] = self.euler(
                State['_Ca_up_'],
                i_up - (i_up_leak + (i_tr * V_rel) / V_up),
                self.δt('_Ca_up_')
            )

            B1 = (2.0 * i_NaCa - (i_CaP + i_Ca_L + i_B_Ca)) / (2.0 * V_i * F) + (V_up * (i_up_leak - i_up) + i_rel * V_rel) / V_i
            B2 = 1.0 + (TRPN_max * Km_TRPN) / tf.square(State['_Ca_i_'] + Km_TRPN) + (CMDN_max * Km_CMDN) / tf.square(State['_Ca_i_'] + Km_CMDN)

            State1['_Ca_i_'] = self.euler(
                State['_Ca_i_'],
                B1 / B2,
                self.δt('_Ca_i_')
            )

            for s in State:
                if not s in State1:
                    print('Warning! Missing New State: %s' % s)

            return State1, inter

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

        ϵ = V * 1e-20

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
            ϵ + ((6.0 * 0.2) / 1.3),
            (6.0 * (1.0 - mod.exp(-(V - 7.9) / 5.0))) / ((1.0 + 0.3 * mod.exp(-(V - 7.9) / 5.0)) * 1.0 * (V - 7.9))
        )

        inter['w_infinity'] = 1.0 - mod.reciprocal(1.0 + mod.exp(-(V - 40.0) / 17.0))

        alpha_m = where(
            mod.abs(V - -47.13) < 0.001,
            ϵ + 3.2,
            (0.32 * (V + 47.13)) / (1.0 - mod.exp(-0.1 * (V + 47.13)))
        )

        beta_m = 0.08 * mod.exp(-V / 11.0)

        inter['m_inf'] = alpha_m / (alpha_m + beta_m)
        inter['tau_m'] = mod.reciprocal(alpha_m + beta_m)

        alpha_h = where(
            V < -40.0,
            0.135 * mod.exp((V + 80.0) / -6.8),
            ϵ
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
            ϵ
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
            ϵ + 0.0015,
            (0.0003 * (V + 14.1)) / (1.0 - mod.exp((V + 14.1) / -5.0))
        )

        beta_xr = where(
            mod.abs(V - 3.3328) < 1.0e-10,
            ϵ + 0.000378361,
            (7.3898e-05 * (V - 3.3328)) / (mod.exp((V - 3.3328) / 5.1237) - 1.0)
        )

        inter['tau_xr'] = mod.reciprocal(alpha_xr + beta_xr);
        inter['xr_infinity'] = mod.reciprocal(1.0 + mod.exp((V + 14.1) / -6.5))

        alpha_xs = where(
            mod.abs(V - 19.9) < 1.0e-10,
            ϵ + 0.00068,
            (4.0e-05 * (V - 19.9)) / (1.0 - mod.exp((V - 19.9) / -17.0))
        )

        beta_xs = where(
            mod.abs(V - 19.9) < 1.0e-10,
            ϵ + 0.000315,
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

        V_us = -83.0
        K_us = 23.0
        alpha_us = 3e-5 * (0.5 * (1 - mod.tanh((V - V_us) / K_us)))
        beta_us = 1e-5 * (0.5 * (1 + mod.tanh((V - (V_us + 30)) / K_us)))
        inter['us_infinity'] = alpha_us / (alpha_us + beta_us)
        inter['tau_us'] = mod.reciprocal(alpha_us + beta_us)

        return inter

    def pot(self):
        return self._V

    def image(self):
        """
            Returns a [height x width] float ndarray in the range 0 to 1
            that encodes V in grayscale
        """
        v = self._V.eval()
        return (v - self.min_v) / (self.max_v - self.min_v)

def cl_observer(m, cyclelengths, i0, i, cl):
    na = m._State['_Na_i_'].eval()
    mean_na = np.average(na, weights=m.phase)

    us_infinity = m._State['_f_Ca_'].eval()
    mean_ca = np.average(us_infinity, weights=m.phase)

    if m.ultra_slow:
        us = m._State['_us_'].eval()
        mean_us = np.average(us, weights=m.phase)

        us_infinity = m._Inter['us_infinity'].eval()
        mean_us_infinity = np.average(us_infinity, weights=m.phase)

        tau_us = m._Inter['tau_us'].eval()
        mean_tau_us = np.average(tau_us, weights=m.phase)

        cyclelengths.append([i0+i, cl, mean_na, mean_ca, mean_us, mean_us_infinity, mean_tau_us])
        print('%d:\t%d\t%.3f\t%.3f\t%.5f\t%.5f\t%.0f' % (i+i0, cl, mean_na, mean_ca, mean_us, mean_us_infinity, mean_tau_us))
    else:
        cyclelengths.append([i0+i, cl, mean_na, mean_ca])
        print('%d:\t%d\t%.3f\t%.3f' % (i+i0, cl, mean_na, mean_ca))


def run_small(config, im, cyclelengths, radius=50, i0=0):
    m = Courtemanche(config)
    m.add_hole_to_phase_field(m.width//2, m.height//2, radius)
    m.add_hole_to_phase_field(m.width//2, m.height//2, m.width//2-6, neg=True)
    m.define()
    m.add_pace_op('s2', 'luq', 10.0)
    m.cl_observer = partial(cl_observer, m, cyclelengths, i0)

    s2 = m.millisecond_to_step(300)

    for i in m.run(im, keep_state=True, block=False):
        if i % 10 == 0:
            m.fire_op('slow')
        if i == s2:
            m.fire_op('s2')
        if i % 5000 == 0:
            image = m.image()
            phase = m.phase
            # cutoff -55 mV
            ρ = np.sum(image[phase > 1e-3] < 0.2) / np.sum(phase > 1e-3)
            print('ρ = %.4f' % ρ)

    np.save('state_small', m.state)  # save the state as a npy file
    return m.state

def run_large(config, im, cyclelengths, radius, i0=0):
    m = Courtemanche(config)
    m.add_hole_to_phase_field(m.width//2, m.height//2, radius)
    # m.add_hole_to_phase_field(m1.width//2, m1.height//2, m1.width//2-6, neg=True)
    state = np.load('state_small.npy').item(0)
    m.define(state=state)
    m.cl_observer = partial(cl_observer, m, cyclelengths, i0)

    for i in m.run(im, keep_state=True, block=False):
        if i % 10 == 0:
            m.fire_op('slow')

    np.save('state_large', m.state)  # save the state as a npy file
    return m.state


if __name__ == '__main__':
    config = {
        'width': 512,           # screen width in pixels
        'height': 512,          # screen height in pixels
        'dt': 0.1,              # integration time step in ms
        'dt_per_plot': 10,      # screen refresh interval in dt unit
        'diff': 1.5,            # diffusion coefficient
        'duration': 10000,       # simulation duration in ms
        'skip': False,          # optimization flag: activate multi-rate
        'cheby': True,          # optimization flag: activate Chebysheb polynomials
        'timeline': False,      # flag to save a timeline (profiler)
        'timeline_name': 'timeline_court.json',
        'save_graph': False,    # flag to save the dataflow graph
        'ultra_slow': False
    }

    # note: change the following line to im = None to run without a screen
    # im = None
    im = Screen(config['height'], config['width'], 'Courtemanche Model')
    cyclelengths = []

    run_small(config, im, cyclelengths, radius=10)
    # i0 = int(config['duration'] / config['dt'])
    # run_large(config, im, cyclelengths, 100, i0)
    # im.save('100.png')

    # data = np.asarray(data)
    # np.savetxt('vol_na_2.dat', data)

    # np.savetxt('test.dat', cyclelengths)
