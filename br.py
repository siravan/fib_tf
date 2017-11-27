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

def laplace2(x):
    """
        Compute the 2D laplacian of an array directly and without using conv
        faster than laplace(x)
    """
    l = (x[:-2,1:-1] + x[2:,1:-1] + x[1:-1,:-2] + x[1:-1,2:] +
         0.5 * (x[:-2,:-2] + x[2:,:-2] + x[:-2,2:] + x[2:,2:]) -
         6 * x[1:-1,1:-1])
    paddings = tf.constant([[1,1], [1,1]])
    return tf.pad(l, paddings, 'CONSTANT', name='laplacian')

SAMPLES = 512
C_m     = 1.0
diffCoef   = 0.001
minVlt     = -90.0    # mV
maxVlt     = 30.0     # mV

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


ab_coef = np.array([[0.0005, 0.083,  50.,    0.0,    0.0,    0.057,  1.0],   # ca_x1
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
           [2*0.0065, -0.02,  30.,    0.0,    0.0,    -0.2,   1.0]], dtype=np.float32)   # cb_f

############### Direct α/β calculation #####################################

def calc_alpha_bata(v, c):
    if c[3] == 0:
        return ((c[0] * tf.exp(c[1]*(v+c[2]), name='exp_A')) /
                (tf.exp(c[5]*(v+c[2]), name='exp_B') + c[6]))

    return ((c[0] * tf.exp(c[1]*(v+c[2]), name='exp_C') + c[3] * (v+c[4])) /
            (tf.exp(c[5]*(v+c[2]), name='exp_D') + c[6]))

def calc_inf_tau(v, c, d, name):
    with tf.name_scope(name) as scope:
        alpha = calc_alpha_bata(v, c)
        beta = calc_alpha_bata(v, d)
        return (tf.realdiv(alpha, alpha+beta, name='inf'),
                tf.reciprocal(alpha+beta, name='tau'))

################ Helper functions for Chebyshev interpolation #############

def calc_alpha_beta_np():
        v = np.linspace(minVlt, maxVlt, SAMPLES)
        x = np.outer(v, np.ones(ab_coef.shape[0]))
        y = ((ab_coef[:,0] * np.exp(ab_coef[:,1] * (x+ab_coef[:,2])) +
                ab_coef[:,3] * (x+ab_coef[:,4])) /
                (np.exp(ab_coef[:,5] * (x+ab_coef[:,2])) + ab_coef[:,6]))
        alpha = y[...,::2]
        beta = y[...,1::2]
        return v, alpha, beta

def convert_chebyshev(x, y, deg=8):
    cheb = np.polynomial.chebyshev.Chebyshev.fit(x, y, deg)
    return cheb.coef


def chebyshev_poly(Ts, x, y):
    c = convert_chebyshev(x, y)
    return (c[0] + (0.5*c[1])*Ts[0] + c[2]*Ts[1] + c[3]*Ts[2] + c[4]*Ts[3] +
            c[5]*Ts[4] + (c[6]-c[8])*Ts[5] + c[7]*Ts[6] + c[8]*Ts[7])


def solve(V, C, M, H, J, D, F, XI, V0, dt, diff, cheby):
    """ Explicit Euler ODE solver """
    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
    with jit_scope():
    #if True:
        if not cheby:
            xi_inf, xi_tau = calc_inf_tau(V0, ab_coef[0], ab_coef[1], 'xi')
            m_inf, m_tau = calc_inf_tau(V0, ab_coef[2], ab_coef[3], 'm')
            h_inf, h_tau = calc_inf_tau(V0, ab_coef[4], ab_coef[5], 'h')
            j_inf, j_tau = calc_inf_tau(V0, ab_coef[6], ab_coef[7], 'j')
            d_inf, d_tau = calc_inf_tau(V0, ab_coef[8], ab_coef[9], 'd')
            f_inf, f_tau = calc_inf_tau(V0, ab_coef[10], ab_coef[11], 'f')
        else:
            x = (V0 - 0.5*(maxVlt+minVlt)) / (0.25*(maxVlt-minVlt))
            T1x2 = tf.identity(x, name='T1x2')
            T2 = tf.subtract(0.5*T1x2*T1x2, 1.0, name='T2')
            T3 = tf.subtract(T1x2*T2, 0.5*T1x2, name='T3')
            T4 = tf.subtract(T1x2*T3, T2, name='T4')
            T5 = tf.subtract(T1x2*T4, T3, name='T5')
            T6 = tf.subtract(T1x2*T5, T4, name='T6')
            T7 = tf.subtract(T1x2*T6, T5, name='T7')
            T8a = tf.multiply(T1x2, T7, name='T8a')

            Ts = [T1x2, T2, T3, T4, T5, T6, T7, T8a]

            v, α, β = calc_alpha_beta_np()

            xi_inf = chebyshev_poly(Ts, v, α[:,0]/(α[:,0]+β[:,0]))
            m_inf = chebyshev_poly(Ts, v, α[:,1]/(α[:,1]+β[:,1]))
            h_inf = chebyshev_poly(Ts, v, α[:,2]/(α[:,2]+β[:,2]))
            j_inf = chebyshev_poly(Ts, v, α[:,3]/(α[:,3]+β[:,3]))
            d_inf = chebyshev_poly(Ts, v, α[:,4]/(α[:,4]+β[:,4]))
            f_inf = chebyshev_poly(Ts, v, α[:,5]/(α[:,5]+β[:,5]))

            xi_tau = chebyshev_poly(Ts, v, 1.0/(α[:,0]+β[:,0]))
            m_tau = chebyshev_poly(Ts, v, 1.0/(α[:,1]+β[:,1]))
            h_tau = chebyshev_poly(Ts, v, 1.0/(α[:,2]+β[:,2]))
            j_tau = chebyshev_poly(Ts, v, 1.0/(α[:,3]+β[:,3]))
            d_tau = chebyshev_poly(Ts, v, 1.0/(α[:,4]+β[:,4]))
            f_tau = chebyshev_poly(Ts, v, 1.0/(α[:,5]+β[:,5]))


        XI1 = tf.clip_by_value(xi_inf - (xi_inf - XI) * tf.exp(-dt/xi_tau), 0.0, 1.0, name='XI1')
        M1 = tf.clip_by_value(m_inf - (m_inf - M) * tf.exp(-dt/m_tau), 0.0, 1.0, name='M1')
        H1 = tf.clip_by_value(h_inf - (h_inf - H) * tf.exp(-dt/h_tau), 0.0, 1.0, name='H1')
        J1 = tf.clip_by_value(j_inf - (j_inf - J) * tf.exp(-dt/j_tau), 0.0, 1.0, name='J1')
        D1 = tf.clip_by_value(d_inf - (d_inf - D) * tf.exp(-dt/d_tau), 0.0, 1.0, name='D1')
        F1 = tf.clip_by_value(f_inf - (f_inf - F) * tf.exp(-dt/f_tau), 0.0, 1.0, name='F1')

        # iK1 = (C_K1 * 0.35 *(4*(tf.exp(0.04 * (V0 + 85)) - 1) /
        #     (tf.exp(0.08 * (V0 + 53)) + tf.exp(0.04 * (V0 + 53))) +
        #     0.2 * ((V0 + 23) / (1 - tf.exp(-0.04 * (V0 + 23))))))
        #ix1 = (C_x1 * XI * 0.8 * ( tf.exp(0.04 * (V0 + 77)) - 1) /
        #    tf.exp(0.04 * (V0 + 35)))

        k = tf.exp(0.04 * V0, name='k')
        iK1 = (C_K1 * (0.35 *(4*(29.64*k - 1) / ( 69.41*k*k + 8.33*k) +
                    0.2 * ((V0 + 23) / (1 - 0.3985 / k )))))

        ix1 = (C_x1 * XI * 0.8 * (21.76*k - 1) / (4.055*k))

        iNa = C_Na * (g_Na*M*M*M*H*J + g_NaC) * (V0 - ENa)

        ECa = D_Ca - 82.3 - 13.0278 * tf.log(C)
        iCa = C_s * g_s * D * F * (V0 - ECa)

        I_sum = iK1 + ix1 + iNa + iCa

        V1 = tf.clip_by_value(V0 + diff * dt * laplace2(V0) - dt * I_sum / C_m, -85.0, 25.0)

        dC = -1.0e-7*iCa + 0.07*(1.0e-7 - C)
        C1 = C + dt * dC

        # return tf.no_op()

        return tf.group(
            tf.assign(V, V1, name='set_V'),
            tf.assign(C, C1, name='set_C'),
            tf.assign(M, M1, name='set_M'),
            tf.assign(H, H1, name='set_H'),
            tf.assign(J, J1, name='set_J'),
            tf.assign(D, D1, name='set_D'),
            tf.assign(F, F1, name='set_F'),
            tf.assign(XI, XI1, name='set_X')
            )

def define_model(height, width, diff=0.809, dt=0.1, cheby=False):
    """
        Create a tensorflow graph to run the Fenton 4v model

        Args:
            N: height (pixels)
            M: width (pixels)

        Returns:
            A model dict
    """
    # the initial values of the state variables
    v_init = np.full([height, width], -84.624, dtype=np.float32)
    c_init = np.full([height, width], 1e-4, dtype=np.float32)
    m_init = np.full([height, width], 0.01, dtype=np.float32)
    h_init = np.full([height, width], 0.988, dtype=np.float32)
    j_init = np.full([height, width], 0.975, dtype=np.float32)
    d_init = np.full([height, width], 0.003, dtype=np.float32)
    f_init = np.full([height, width], 0.994, dtype=np.float32)
    xi_init = np.full([height, width], 0.0001, dtype=np.float32)

    # S1 stimulation: vertical along the left side
    v_init[:,1] = 10.0

    # prepare for S2 stimulation as part of the cross-stimulation protocol
    s2 = np.full([height, width], minVlt, dtype=np.float32)
    s2[:height//2, :width//2] = 10.0

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

        # enforcing the no-flux boundary condition
        paddings = tf.constant([[1,1], [1,1]])
        V0 = tf.pad(V[1:-1,1:-1], paddings, 'SYMMETRIC', name='V0')

        ode_op = solve(V, C, M, H, J, D, F, XI, V0, dt, diff, True)

        # Operation for S2 stimulation
        s2_op = V.assign(tf.maximum(V, s2))

        return {'height': height, 'width': width, 'V': V, 'ode_op': ode_op, 's2_op': s2_op}

def normalize(v):
    return (v - minVlt) / (maxVlt - minVlt)

def run(model, samples=20000, s2_time=3000):
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
    #feed_dict = {model['diff']: diff, model['dt']: dt}
    feed_dict = {}

    # the painting canvas
    im = sc.Screen(model['height'], model['width'], '4v Model')

    # unpack the model
    ode_op = model['ode_op']
    s2_op = model['s2_op']
    V = model['V']

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

        file_writer = tf.summary.FileWriter('logs', sess.graph)

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
                with open('timeline_br.json', 'w') as f:
                    f.write(chrome_trace)

            # fire S2
            if i == s2_time:
                sess.run(s2_op)
            # draw a frame every 1 ms
            if i % 10 == 0:
                im.imshow(normalize(V.eval()))

    print('elapsed: %f sec' % (time.time() - then))
    im.wait()   # wait until the window is closed


if __name__ == '__main__':
    model = define_model(512, 512)
    run(model, samples=20000)
