#!/home/shahriar/anaconda3/bin/python
"""
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler
    @2017 Shahriar Iravanian (siravan@emory.edu)
"""

import time
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
        # the initial values of the state variables
        u_init = np.zeros([self.height, self.width], dtype=np.float32)
        v_init = np.ones([self.height, self.width], dtype=np.float32)
        w_init = np.ones([self.height, self.width], dtype=np.float32)
        s_init = np.zeros([self.height, self.width], dtype=np.float32)

        # S1 stimulation: vertical along the left side
        if s1:
            u_init[:,1] = 1.0

        # prepare for S2 stimulation as part of the cross-stimulation protocol
        s2 = np.zeros([self.height, self.width], dtype=np.float32)
        s2[:self.height//2, :self.width//2] = 1.0

        #xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        #pace = (np.exp(-((xx - self.width/2)**2 + (yy - self.height/2)**2) / 5.0**2))
        link = calc_link(20, self.width, self.height)

        # define the graph...
        with tf.device('/device:GPU:0'):
            # Create variables for simulation state
            U  = tf.Variable(u_init, name='U')
            V  = tf.Variable(v_init, name='V')
            W  = tf.Variable(w_init, name='W')
            S  = tf.Variable(s_init, name='S')

            states = [[U, V, W, S]]
            for i in range(10):
                if i == 0:
                    U0 = self.enforce_boundary(states[-1][0])
                else:
                    U0 = states[-1][0]
                states.append(self.solve(states[-1], U0))
            U1, V1, W1, S1 = states[-1]

            self._ode_op = tf.group(
                U.assign(U1),
                V.assign(V1),
                W.assign(W1),
                S.assign(S1)
                )

            # Operation for S2 stimulation
            self._s2_op = U.assign(tf.maximum(U, s2))
            self._pace_op = U.assign(tf.maximum(U, link))
            self._U = U

    def image(self):
        return self._U.eval()

def calc_link(n, width, height, sigma=5.0):
    xc = np.random.randint(width, size=(n))
    yc = np.random.randint(height, size=(n))
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    ψ = np.zeros((height, width), dtype=np.float32)
    for x, y in zip(xc, yc):
        ψ += (np.exp(-((xx - x)**2 + (yy - y)**2) / sigma**2))
    ψ = 0.5 * np.clip(ψ, 0, 1)
    return ψ

def coupling_ops(U1, U0, link):
    lr = tf.assign(U1, U1 + link * (U0 - U1))
    rl = tf.assign(U0, U0 + link * (U1 - U0))
    return lr, rl

def sync(m1, m2, lr=None, rl=None, im=None, num_tap=0):
    with tf.Session() as sess:
        # start the timer
        then = time.time()
        tf.global_variables_initializer().run()

        samples = max(m1.samples, m2.samples)

        u1 = []
        u2 = []

        if num_tap > 0:
            tap = np.random.randint(m1.width * m1.height, size=num_tap)
            u1 = np.zeros((samples, num_tap), dtype=np.float32)
            u2 = np.zeros((samples, num_tap), dtype=np.float32)

        for i in range(samples):
            sess.run(m1.ode_op(i))
            sess.run(m2.ode_op(i))
            # fire S2
            if i == m1.s2_time:
                sess.run(m1.s2_op())
            if i == m2.s2_time:
                sess.run(m2.s2_op())

            if i > 2000 and i < 15000 and i % m1.dt_per_plot == 0:
                if i & 1 == 0:
                    sess.run(lr)
                else:
                    sess.run(rl)
                #if i % 90 == 0:
                #    sess.run(m2._pace_op)

            # draw a frame every 1 ms
            if im and i % m1.dt_per_plot == 0:
                image1 = m1.image()
                image2 = m2.image()

                if num_tap > 0:
                    u1[i,:] = image1.ravel()[tap]
                    u2[i,:] = image2.ravel()[tap]

                image = np.concatenate((image1, image2), axis=1)
                im.imshow(image)
                if i > 2000 and np.mean(image2) < 0.01:
                    print('termination detected at %d' % i)
                    break

    print('elapsed: %f sec' % (time.time() - then))
    if im:
        im.wait()   # wait until the window is closed
    return u1, u2

if __name__ == '__main__':
    config = {
        'width': 400,
        'height': 400,
        'dt': 0.1,
        'dt_per_plot' : 1,
        'diff': 1.5,
        'samples': 20000,
        's2_time': 200,
        'cheby': True,
        'timeline': True,
        'timeline_name': 'timeline_4v.json',
        'save_graph': False
    }
    m1 = Fenton4v(config)
    m2 = Fenton4v(config)

    m1.s2_time = -1
    m2.diff = 0.5

    m1.define(False)
    m2.define()


    link = calc_link(3, m1.width, m1.height)
    lr, rl = coupling_ops(m1._U, m2._U, link)

    im = Screen(m1.height, 2*m1.width, 'Fenton 4v Model #1')
    sync(m1, m2, lr, rl, im)
