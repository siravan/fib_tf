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
import time
from screen import Screen
from tensorflow.python.client import timeline

def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
  laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
  return simple_conv(x, laplace_k)

def enforce_boundary(X):
    """
        Enforcing the no-flux (Neumann) boundary condition
    """
    paddings = tf.constant([[1,1], [1,1]])
    return tf.pad(X[1:-1,1:-1], paddings, 'SYMMETRIC', name='boundary')


class Fenton4vJIT:
    """
        The Cherry-Ehrlich-Nattel-Fenton (4v) canine left-atrial model

        Cherry EM, Ehrlich JR, Nattel S, Fenton FH. Pulmonary vein reentry--
        properties and size matter: insights from a computational analysis.
        Heart Rhythm. 2007 Dec;4(12):1553-62.
    """

    def __init__(self, props):
        for key, val in config.items():
            setattr(self, key, val)
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


    def solve(self, state):
        """ Explicit Euler ODE solver """
        U, V, W, S = state
        U0 = enforce_boundary(U)

        scope = tf.contrib.compiler.jit.experimental_jit_scope()

        with scope:
            dU, dV, dW, dS = self.differentiate(U, V, W, S)
            U1 = U0 + self.dt * dU + self.diff * self.dt * laplace(U0)
            V1 = V + self.dt * dV
            W1 = W + self.dt * dW
            S1 = S + self.dt * dS

        return U1, V1, W1, S1

    def define(self):
        """
            Create a tensorflow graph to run the Fenton 4v model
        """
        # the initial values of the state variables
        # initial values (u, v, w, s) = (0.0, 1.0, 1.0, 0.0)
        u_init = np.full([self.height, self.width], self.min_v, dtype=np.float32)
        v_init = np.full([self.height, self.width], 1.0, dtype=np.float32)
        w_init = np.full([self.height, self.width], 1.0, dtype=np.float32)
        s_init = np.full([self.height, self.width], 0.0, dtype=np.float32)

        # S1 stimulation: vertical along the left side
        u_init[:,1] = self.max_v

        # define S2 stimulation
        s2_init = np.full([self.height, self.width], self.min_v, dtype=np.float32)
        s2_init[:self.height//2, :self.width//2] = self.max_v

        # define the graph...
        with tf.device('/device:GPU:0'):
            # Create variables for simulation state
            U  = tf.Variable(u_init, name='U')
            V  = tf.Variable(v_init, name='V')
            W  = tf.Variable(w_init, name='W')
            S  = tf.Variable(s_init, name='S')

            state = [U, V, W, S]
            U1, V1, W1, S1 = self.solve(state)

            self._ode_op = tf.group(
                U.assign(U1),
                V.assign(V1),
                W.assign(W1),
                S.assign(S1)
                )

            self._U = U
            self._s2_op = U.assign(tf.maximum(U, s2_init))


    def run(self, im=None):
        """
            Runs the model. The model should be defined first by calling
            self.define()

            Args:
                im: A Screen used to paint the transmembrane potential

            Returns:
                None
        """

        with tf.Session() as sess:
            # start the timer
            then = time.time()
            tf.global_variables_initializer().run()

            # the main loop!
            for i in range(self.samples):
                sess.run(self._ode_op)

                if i == int(self.s2_time / self.dt):
                    sess.run(self._s2_op)

                # draw a frame every 1 ms
                if im and i % self.dt_per_plot == 0:
                    image = self._U.eval()
                    im.imshow(image)

            elapsed = (time.time() - then)

            # options and run_metadata are needed for tracing
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run(self._ode_op, options=options, run_metadata=run_metadata)
            # Create the Timeline object for the last iteration
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('timeline_jit.json', 'w') as f:
                f.write(chrome_trace)

        print('elapsed: %f sec' % elapsed)
        if im:
            im.wait()   # wait until the window is closed


if __name__ == '__main__':

    config = {
        'width': 512,
        'height': 512,
        'dt': 0.1,
        'dt_per_plot' : 10,
        'diff': 1.5,
        'samples': 10000,
        's2_time': 210
    }
    model = Fenton4vJIT(config)
    model.define()
    # note: change the following line to im = None to run without a screen
    im = Screen(model.height, model.width, 'Simple Fenton 4v Model')

    model.run(im)
