#!/home/shahriar/anaconda3/bin/python
"""
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler
    @2017 Shahriar Iravanian (siravan@emory.edu)
"""

import time
import json
import tensorflow as tf
import numpy as np
from screen import Screen
from fenton import Fenton4v
from br2 import BeelerReuter

def calc_link(n, width, height, sigma=5.0):
    xc = np.random.randint(width, size=(n))
    yc = np.random.randint(height, size=(n))
    # yc = np.full((n), height // 2, dtype=np.float32)
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    ψ = np.zeros((height, width), dtype=np.float32)
    for x, y in zip(xc, yc):
        ψ += (np.exp(-((xx - x)**2 + (yy - y)**2) / sigma**2))
    ψ = 0.5 * np.clip(ψ, 0, 1)
    return ψ, xc, yc

def coupling_ops(m1, m2, link):
    U1 = m1.pot()
    U2 = m2.pot()
    lr = tf.assign(U1, U1 + link * (U2 - U1))
    rl = tf.assign(U2, U2 + link * (U1 - U2))
    return lr, rl

def pace_op(m, link):
    U = m.pot()
    return U.assign(tf.maximum(U, m.min_v + 1.8 * link * (m.max_v - m.min_v)))

def sync(m1, m2, lr=None, rl=None, im=None, pace_cl=0, num_tap=0):
    with tf.Session() as sess:
        # start the timer
        then = time.time()
        tf.global_variables_initializer().run()

        samples = max(m1.samples, m2.samples)
        when = samples

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

            # draw a frame every 1 ms
            if i % m1.dt_per_plot == 0:
                image1 = m1.image()
                image2 = m2.image()

                if num_tap > 0:
                    u1[i,:] = image1.ravel()[tap]
                    u2[i,:] = image2.ravel()[tap]

                if i == m2.s3_time:
                    sess.run(m2.s2_op())

                if i > 5000 and i < 15000:
                    if i & 1 == 0:
                        if lr is not None:
                            sess.run(lr)
                    else:
                        if rl is not None:
                            sess.run(rl)
                    if pace_cl > 0 and i % pace_cl == 0:
                        sess.run(m2._pace_op)

                if im:
                    image = np.concatenate((image1, image2), axis=1)
                    im.imshow(image)

                if i > 500 and np.max(image2) < 0.5:
                    when = i
                    print('termination detected at %d' % i)
                    break
        sess.close()
        print(sess._closed)

    elapsed = time.time() - then
    print('elapsed: %f sec' % elapsed)
    if im:
        im.wait()   # wait until the window is closed
    return elapsed, when, u1, u2

def calc_cycle_length(x):
    w = np.where((x[1:] >= 0.5) & (x[:-1] < 0.5))[0]
    cl = w[1:] - w[:-1]
    return cl

def run_once(plot=True, num_link=10, diff=1.5, pace=False):
    # config = {
    #     'width': 400,
    #     'height': 400,
    #     'dt': 0.1,
    #     'dt_per_plot': 1,
    #     'diff': diff,
    #     'samples': 20000,
    #     's2_time': 250,
    #     's3_time': 1000,
    #     'cheby': True,
    #     'timeline': False,
    #     'timeline_name': 'timeline_4v.json',
    #     'save_graph': False,
    #     'skip': False,
    #     'pace_cl': 90
    # }
    # m1 = Fenton4v(config)
    # m2 = Fenton4v(config)
    # m2.diff = 0.45 + 0.1 * np.random.random()

    config = {
        'width': 512,
        'height': 512,
        'dt': 0.1,
        'skip': True,
        'dt_per_plot': 1,
        'diff': diff,
        'samples': 20000,
        's2_time': 300,
        's3_time': 750,
        'cheby': True,
        'timeline': False,
        'timeline_name': 'timeline_br.json',
        'save_graph': False,
        'pace_cl': 180,
    }
    m1 = BeelerReuter(config)
    m2 = BeelerReuter(config)
    m2.diff = 2 * (0.45 + 0.1 * np.random.random())

    print('diff = %.5f' % m2.diff)
    m1.s2_time = -1

    # link, xc, yc = calc_link(num_link, m1.width, m1.height)
    link, xc, yc = calc_link(num_link, m1.width, m1.height)
    m1.define(False)
    m2.define()

    lr, rl = coupling_ops(m1, m2, link)
    m2._pace_op = pace_op(m2, link)

    # lr, _ = coupling_ops(m1._U, m2._U, link)
    # link, xc, yc = calc_link(num_link, m1.width, m1.height)
    # _, rl = coupling_ops(m1._U, m2._U, link)

    #m2.create_pacer(7)

    if plot:
        im = Screen(m1.height, 2*m1.width, 'Fenton 4v Model #1')
    else:
        im = False

    if pace:
        elapsed, when, u1, u2 = sync(m1, m2, None, None, im, num_tap=1, pace_cl=m2.pace_cl)
    else:
        elapsed, when, u1, u2 = sync(m1, m2, lr, rl, im, num_tap=1)

    config['pace'] = pace
    config['sync'] = not pace
    config['num_link'] = num_link
    config['elapsed'] = elapsed
    config['when'] = when
    if not pace:
        config['u1'] = calc_cycle_length(u1[:,0]).tolist()
    config['u2'] = calc_cycle_length(u2[:,0]).tolist()
    return config

if __name__ == '__main__':
    path = '/home/shahriar/Dropbox/results/tf_sync_br/'

    num_link = np.random.randint(1, 20)
    diff = np.random.random()*1.75 + 0.25
    pace = np.random.random() > 0.8

    config = run_once(plot=False, num_link=num_link, diff=diff, pace=pace)

    if pace:
        name = 'fopr_2Dxx0xx%d.json' % (int(time.time()))
    else:
        name = 'ffsr_2Dxx0xx%d.json' % (int(time.time()))

    with open(path + name, 'w') as f:
        json.dump(config, f, indent=4)
