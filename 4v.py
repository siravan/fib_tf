#!/home/shahriar/anaconda3/bin/python
"""
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler
    @2017 Shahriar Iravanian (siravan@emory.edu)
"""

import numpy as np
import time
import tensorflow as tf
from tensorflow.python.client import timeline
import screen as sc

def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return tf.constant(a, dtype=tf.float32)

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
    # return tf.sigmoid(x * 1000)
    return (tf.sign(x) + 1) * 0.5

def differentiate(U, V, W, S):
    """ the state differentiation for the 4v model """
    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
    with jit_scope():
        I_fi = -V * H(U - u_c) * (U - u_c) * (u_m - U) / tau_d
        I_si = -W * S / tau_si
        I_so = (0.5 * (a_so - tau_a) * (1 + tf.tanh((U - b_so) / c_so)) +
               (U - u_0) * (1 - H(U - u_so)) / tau_so + H(U - u_so) * tau_a)

        dU = -(I_fi + I_si + I_so)
        dV = tf.where(U > u_c, -V / tau_vp, (1 - V) / tau_vn)
        dW = tf.where(U > u_c, -W / tau_wp, tf.where(U > u_w, (1 - W) / tau_wn2, (1 - W) / tau_wn1))
        r_s = (r_sp - r_sn) * H(U - u_c) + r_sn
        dS = r_s * (0.5 * (1 + tf.tanh((U - u_csi) * k_)) - S)

    return dU, dV, dW, dS

def define_model(N, M):
    """
        Create a tensorflow graph to run the Fenton 4v model

        Args:
            N: height (pixels)
            M: width (pixels)

        Returns:
            A model dict
    """
    # the initial values of the state variables
    u_init = np.zeros([N, M], dtype=np.float32)
    v_init = np.ones([N, M], dtype=np.float32)
    w_init = np.ones([N, M], dtype=np.float32)
    s_init = np.zeros([N, M], dtype=np.float32)

    # S1 stimulation: vertical along the left side
    u_init[:,1] = 1.0

    # prepare for S2 stimulation as part of the cross-stimulation protocol
    s2 = np.zeros([N, M], dtype=np.float32)
    s2[:N//2, :N//2] = 1.0

    # define the graph...
    with tf.device('/device:GPU:0'):
        diff = tf.placeholder(tf.float32, shape=())     # the diffusion coefficient
        dt = tf.placeholder(tf.float32, shape=())       # the integration time-step in ms

        # Create variables for simulation state
        U  = tf.Variable(u_init)
        V  = tf.Variable(v_init)
        W  = tf.Variable(w_init)
        S  = tf.Variable(s_init)

        # enforcing the no-flux boundary condition
        paddings = tf.constant([[1,1], [1,1]])
        U0 = tf.pad(U[1:-1,1:-1], paddings, 'SYMMETRIC')
        # U0 = tf.pad(U[1:-1,1:-1], paddings, 'REFLECT')

        # Explicit Euler integration
        dU, dV, dW, dS = differentiate(U0, V, W, S)

        # Operation to update the state
        ode_op = tf.group(
          U.assign(U0 + dt * dU + diff * dt * laplace(U0)),
          V.assign_add(dt * dV),
          W.assign_add(dt * dW),
          S.assign_add(dt * dS))

        # Operation for S2 stimulation
        s2_op = U.assign(tf.maximum(U, s2))

        return {'height': N, 'width': M, 'U': U, 'V': V, 'W': W, 'S': S,
                'ode_op': ode_op, 's2_op': s2_op, 'diff': diff, 'dt': dt}


def run(model, samples=10000, s2_time=2000, diff=1.5, dt=0.1):
    """
        Runs the model

        Args:
            model: A model dict as returned by define_model
            samples: number of sample points
            s2_time: time for firing S2 in the cross-stimulation protocol (in dt unit)
            diff: the diffusion coefficient
            dt: the integration time step in ms

        Returns:
            None
    """
    feed_dict = {model['diff']: diff, model['dt']: dt}

    # the painting canvas
    im = sc.Screen(model['height'], model['width'], '4v Model')

    # unpack the model
    ode_op = model['ode_op']
    s2_op = model['s2_op']
    U = model['U']

    # options and run_metadata are needed for tracing
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    # start the timer
    then = time.time()

    with tf.Session(config=config) as sess:
        # Initialize state to initial conditions
        tf.global_variables_initializer().run()

        # the main loop!
        for i in range(samples):
            if i < samples-1:
                sess.run(ode_op, feed_dict=feed_dict)
            else:   # the last loop, save tracing data
                sess.run(ode_op, feed_dict=feed_dict,
                        options=options, run_metadata=run_metadata)
                # Create the Timeline object for the last iteration
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline.json', 'w') as f:
                    f.write(chrome_trace)

            # fire S2
            if i == s2_time:
                sess.run(s2_op)
            # draw a frame every 1 ms
            if i % 10 == 0:
                im.imshow(U.eval())

    print('elapsed: %f sec' % (time.time() - then))
    im.wait()   # wait until the window is closed


if __name__ == '__main__':
    model = define_model(500, 500)
    run(model, samples=20000)
