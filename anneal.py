#!/home/shahriar/anaconda3/bin/python
"""
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler
    @2017 Shahriar Iravanian (siravan@emory.edu)
"""

import time
import json
import tensorflow as tf
import numpy as np
from PIL import Image
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


    def solve(self, state, U0, Diff):
        """ Explicit Euler ODE solver """
        U, V, W, S = state
        with self.jit_scope():
            dU, dV, dW, dS = self.differentiate(U, V, W, S)

            U1 = U0 + self.dt * dU + self.dt * Diff * self.laplace(U0)
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

        # define the graph...
        with tf.device('/device:GPU:0'):
            # Create variables for simulation state
            U  = tf.Variable(u_init, name='U')
            V  = tf.Variable(v_init, name='V')
            W  = tf.Variable(w_init, name='W')
            S  = tf.Variable(s_init, name='S')
            Diff = tf.placeholder(tf.float32, shape=[self.height, self.width], name='Diff')

            states = [[U, V, W, S]]
            for i in range(10):
                states.append(self.solve(states[-1],
                              self.enforce_boundary(states[-1][0]),
                              Diff))
            U1, V1, W1, S1 = states[-1]

            self._ode_op = tf.group(
                U.assign(U1),
                V.assign(V1),
                W.assign(W1),
                S.assign(S1)
                )

            # Operation for S2 stimulation
            self._s2_op = U.assign(tf.maximum(U, s2))
            self._U = U
            self.Diff = Diff

    def pot(self):
        return self._U

    def image(self, sess):
        return sess.run(self._U)

def random_obstacles(n, width, height):
    xy = np.zeros((n, 2), dtype=np.int32)
    xy[:,0] = np.random.randint(width, size=(n))
    xy[:,1] = np.random.randint(height, size=(n))
    return xy

def nodge_obstacles(xy0, width, height):
    xy = np.copy(xy0)
    n = xy.shape[0]
    i = np.random.randint(n)
    xy[i,0] = np.random.randint(width)
    xy[i,1] = np.random.randint(height)
    return xy

def calc_link(xy, width, height, sigma=5.0):
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    ψ = np.zeros((height, width), dtype=np.float32)
    n = xy.shape[0]
    for i in range(n):
        ψ += (np.exp(-((xx - xy[i,0])**2 + (yy - xy[i,1])**2) / sigma**2))
    ψ = 1.0 - np.clip(ψ*2, 0, 1)
    return ψ

def coupling_ops(m1, m2, link):
    U1 = m1.pot()
    U2 = m2.pot()
    lr = tf.assign(U1, U1 + link * (U2 - U1))
    rl = tf.assign(U2, U2 + link * (U1 - U2))
    return lr, rl

def pace_op(m, link):
    U = m.pot()
    return U.assign(tf.maximum(U, m.min_v + 1.8 * link * (m.max_v - m.min_v)))

def anneal(sess, m, Diff, diff, im=None):
    # start the timer
    then = time.time()
    sess.run(tf.global_variables_initializer())

    samples = m.samples
    when = samples

    for i in range(samples):
        sess.run(m.ode_op(i), feed_dict={Diff: diff})
        # fire S2
        if i == m.s2_time:
            sess.run(m.s2_op(), feed_dict={Diff: diff})

        # draw a frame every 1 ms
        if i % m.dt_per_plot == 0:
            image = m.image(sess)

            if i == m.s3_time:
                sess.run(m.s2_op(), feed_dict={Diff: diff})

            if im:
                im.imshow(image)

            if i > 500 and np.max(image) < 0.5:
                when = i
                print('termination detected at %d' % i)
                break

    elapsed = time.time() - then
    # print('elapsed: %.3f\twhen: %d' % (elapsed, when))
    if im:
        pass
        #im.wait()   # wait until the window is closed
    return elapsed, when

def anneal_once(sess, m, xy0):
    xy1 = nodge_obstacles(xy0, m.width, m.height)
    d = 0.45 + 0.1 * np.random.random()
    link = calc_link(xy1, m.width, m.height, sigma=10)
    diff = np.asarray(d * link, dtype=np.float32)
    elapsed, when1 = anneal(sess, m, m.Diff, diff)
    print('elapsed: %.3f' % elapsed)
    return xy1, link, when1


def anneal_multi(sess, m, xy0, k):
    xy1 = nodge_obstacles(xy0, m.width, m.height)
    link = calc_link(xy1, m.width, m.height, sigma=10)
    w = 0

    for i in range(k):
        d = 0.45 + 0.1 * np.random.random()
        diff = np.asarray(d * link, dtype=np.float32)
        elapsed, when1 = anneal(sess, m, m.Diff, diff)
        print('elapsed: %.3f' % elapsed)
        w += 1 / when1

    return xy1, link, k/w


def calc_cycle_length(x):
    w = np.where((x[1:] >= 0.5) & (x[:-1] < 0.5))[0]
    cl = w[1:] - w[:-1]
    return cl

def run_once(plot=True, num_link=10, diff=1.5, pace=False):
    config = {
        'width': 512,
        'height': 512,
        'dt': 0.1,
        'dt_per_plot': 1,
        'diff': diff,
        'samples': 10000,
        's2_time': 250,
        's3_time': 1000,
        'cheby': True,
        'timeline': False,
        'timeline_name': 'timeline_4v.json',
        'save_graph': False,
        'skip': False,
        'pace_cl': 90
    }
    m = Fenton4v(config)
    Diff = m.define()

    # if plot:
    #     im = Screen(m.height, m.width, 'Fenton 4v Model #1')
    # else:
    #     im = False

    im = Screen(m.height, m.width, 'Fenton 4v Model')

    xy0 = random_obstacles(15, m.width, m.height)

    β = 1000.0
    when0 = 1e6
    best_when = when0
    whens = []
    k = 0
    count_since_last_move = 0

    for i in range(500):
        with tf.Session() as sess:
            xy1, link, when1 = anneal_multi(sess, m, xy0, 5)

        print('>: when: %d' % when0)

        if when1 < when0 or np.exp(-(when1 - when0)/β) > np.random.random():
            count_since_last_move = 0
            i_move = i
            im.imshow(link)
            Image.fromarray((link*255).astype(np.uint8)).save('frames/%04d.png' % k)
            k += 1
            xy0 = xy1
            when0 = when1
            print('%d: when: %d (switch)\t best: %d, β: %.3f' % (i+1, when1, best_when, β))
        else:
            if count_since_last_move > 10:
                β = min(β*1.25, 1000.0)
                count_since_last_move = 0
            print('%d: when: %d (stay)\t best: %d, β: %.3f' % (i+1, when1, best_when, β))

        if when1 < best_when:
            best_link = link
            best_when = when1

        whens.append(when1)
        β *= 0.995

    print('best when: %d' % best_when)
    with open('whens.dat', 'wb') as f:
        np.savetxt(f, np.asarray(whens), fmt='%d')
    with open('link.dat', 'wb') as f:
        np.savetxt(f, link, fmt='%.4f')

    im.imshow(link)
    im.wait()

if __name__ == '__main__':
    run_once(plot=False)
