#!/home/shahriar/anaconda3/bin/python
"""
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler
    @2017 Shahriar Iravanian (siravan@emory.edu)
"""

import tensorflow as tf
import numpy as np
from screen import Screen
from ionic import IonicModel

class Fenton4v(IonicModel):
    """
        The Cherry-Ehrlich-Nattel-Fenton (4v) canine left-atrial model

        Cherry EM, Ehrlich JR, Nattel S, Fenton FH. Pulmonary vein reentry--
        properties and size matter: insights from a computational analysis.
        Heart Rhythm. 2007 Dec;4(12):1553-62.
    """

    def __init__(self, props):
        IonicModel.__init__(self, props)
        self.min_v = 0.0
        self.max_v = 1.0

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


    def solve(self, state, U0):
        """ Explicit Euler ODE solver """
        U, V, W, S = state
        with self.jit_scope():
            dU, dV, dW, dS = self.differentiate(U, V, W, S)

            U1 = U0 + self.dt * dU + self.diff * self.dt * self.laplace(U0)
            V1 = V + self.dt * dV
            W1 = W + self.dt * dW
            S1 = S + self.dt * dS

            return U1, V1, W1, S1

    def define(self, s1=True):
        """
            Create a tensorflow graph to run the Fenton 4v model
        """
        super().define()
        # the initial values of the state variables
        u_init = np.zeros([self.height, self.width], dtype=np.float32)
        v_init = np.ones([self.height, self.width], dtype=np.float32)
        w_init = np.ones([self.height, self.width], dtype=np.float32)
        s_init = np.zeros([self.height, self.width], dtype=np.float32)

        # S1 stimulation: vertical along the left side
        if s1:
            u_init[:,1] = 1.0

        # define the graph...
        with tf.device('/device:GPU:0'):
            # Create variables for simulation state
            U  = tf.Variable(u_init, name='U')
            V  = tf.Variable(v_init, name='V')
            W  = tf.Variable(w_init, name='W')
            S  = tf.Variable(s_init, name='S')

            states = [[U, V, W, S]]
            for i in range(10):
                states.append(self.solve(states[-1],
                              self.enforce_boundary(states[-1][0])))
            U1, V1, W1, S1 = states[-1]

            self._ode_op = tf.group(
                U.assign(U1),
                V.assign(V1),
                W.assign(W1),
                S.assign(S1)
                )

            self._U = U

    def pot(self):
        return self._U

    def image(self):
        return self._U.eval()

if __name__ == '__main__':

    config = {
        'width': 512,
        'height': 512,
        'dt': 0.1,
        'dt_per_plot' : 1,
        'diff': 1.5,
        'samples': 10000,
        # 's2_time': 210,
        'cheby': True,
        'timeline': False,
        'timeline_name': 'timeline_4v.json',
        'save_graph': False
    }
    model = Fenton4v(config)

    model.add_hole_to_phase_field(256, 256, 50.0)
    model.define()
    model.add_pace_op('s2', 'luq', 1.0)
    model.add_pace_op('s3', 'right', 1.0)
    # note: change the following line to im = None to run without a screen
    im = Screen(model.height, model.width, 'Fenton 4v Model')

    for t in model.run(im):
        if t == 210:
            model.fire_op('s2')
        if t > 500 and t < 8000 and t % 100 == 0:
            model.fire_op('s3')
