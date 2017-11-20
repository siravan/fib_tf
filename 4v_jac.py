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

def fwd_gradients(ys, xs, d_xs):
    """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
    With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
    the vector being pushed forward.
    with minor modifications from https://github.com/renmengye/tensorflow-forward-ad/issues/2
    """
    v = tf.placeholder(ys.dtype, shape=ys.get_shape(), name='grad_v')  # dummy variable
    g = tf.gradients(ys, xs, grad_ys=v)
    if g == [None]:
        return [tf.zeros(ys.get_shape(), dtype=ys.dtype)]
    h = tf.gradients(g, v, grad_ys=d_xs)
    if h == [None]:
        return [tf.zeros(ys.get_shape(), dtype=ys.dtype)]
    return h


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
    """ step function """
    # return tf.sigmoid(x * 1000)
    return (1 + tf.sign(x)) * 0.5

def G(x):
    """ negative step function """
    # return tf.sigmoid(x * 1000)
    return (1 - tf.sign(x)) * 0.5

def differentiate(U, V, W, S):
    """ the state differentiation for the 4v model """
    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
    with jit_scope():
        I_fi = V * H(U - u_c) * (U - u_c) * (U - u_m) / tau_d
        I_si = -W * S / tau_si
        I_so = (0.5 * (a_so - tau_a) * (1 + tf.tanh((U - b_so) / c_so)) +
               G(U - u_so) * (U - u_0) / tau_so + H(U - u_so) * tau_a)

        dU = -(I_fi + I_si + I_so)
        dV = G(U - u_c) * (1 + (-V)) / tau_vn + H(U - u_c) * (-V / tau_vp)
        tau_wn = H(U - u_w) * (tau_wn2 - tau_wn1) + tau_wn1
        dW = G(U - u_c) * (1 + (-W)) / tau_wn + H(U - u_c) * (-W / tau_wp)
        r_s = H(U - u_c) * (r_sp - r_sn) + r_sn
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

        Juu = fwd_gradients(dU, U, dU)
        Juv = fwd_gradients(dU, V, dV)
        Juw = fwd_gradients(dU, W, dW)
        Jus = fwd_gradients(dU, S, dS)

        Jvu = fwd_gradients(dV, U, dU)
        Jvv = fwd_gradients(dV, V, dV)
        # Jvw = fwd_gradients(dV, W, dW)
        # Jvs = fwd_gradients(dV, S, dS)

        Jwu = fwd_gradients(dW, U, dU)
        # Jwv = fwd_gradients(dW, V, dV)
        Jww = fwd_gradients(dW, W, dW)
        # Jws = fwd_gradients(dW, S, dS)

        Jsu = fwd_gradients(dS, U, dU)
        # Jsv = fwd_gradients(dS, V, dV)
        # Jsw = fwd_gradients(dS, W, dW)
        Jss = fwd_gradients(dS, S, dS)

        # Operation to update the state
        ode_op = tf.group(
            U.assign(U0 + dt * dU + 0.5 * dt * dt * (Juu[0] + Juw[0]) + diff * dt * laplace(U0)),
            V.assign_add(dt * dV + 0.5 * dt * dt * (Jvu[0] + Jvv[0])),
            W.assign_add(dt * dW + 0.5 * dt * dt * (Jwu[0] + Jww[0])),
            S.assign_add(dt * dS + 0.5 * dt * dt * Jsu[0]))


        # Operation to update the state
        # ode_op = tf.group(
        #   U.assign(U0 + dt * dU + diff * dt * laplace(U0)),
        #   V.assign_add(dt * dV),
        #   W.assign_add(dt * dW),
        #   S.assign_add(dt * dS))

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

    # print_op = model['print_op']

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
                # sess.run(print_op)
                im.imshow(U.eval())

    print('elapsed: %f sec' % (time.time() - then))
    im.wait()   # wait until the window is closed


if __name__ == '__main__':
    model = define_model(500, 500)
    run(model, samples=20000, dt=0.1)
