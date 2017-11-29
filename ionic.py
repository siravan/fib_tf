"""
    A TensorFlow-based 2D Cardiac Electrophysiology Modeler
    @2017 Shahriar Iravanian (siravan@emory.edu)
"""

import numpy as np
import time
import tensorflow as tf
from tensorflow.python.client import timeline
import screen as sc


class IonicModel:
    def __init__(self, props):
        for key, val in props.items():
            setattr(self, key, val)

    def laplace(self, X):
        """
            Compute the 2D laplacian of an array directly and without using conv
        """
        l = (X[:-2,1:-1] + X[2:,1:-1] + X[1:-1,:-2] + X[1:-1,2:] +
             0.5 * (X[:-2,:-2] + X[2:,:-2] + X[:-2,2:] + X[2:,2:]) -
             6 * X[1:-1,1:-1])
        paddings = tf.constant([[1,1], [1,1]])
        return tf.pad(l, paddings, 'CONSTANT', name='laplacian')

    def enforce_boundary(self, X):
        """
            Enforcing the no-flux (Neumann) boundary condition
        """
        paddings = tf.constant([[1,1], [1,1]])
        return tf.pad(X[1:-1,1:-1], paddings, 'SYMMETRIC', name='boundary')

    def rush_larsen(self, g, g_inf, g_tau, dt, name=None):
        return tf.clip_by_value(g_inf - (g_inf - g) * tf.exp(-dt/g_tau), 0.0,
                                1.0, name=name)

    def run(self, im):
        """
            Runs the model. The model should be defined first by calling
            self.define()

            Args:
                im: A Screen used to paint the transmembrane potential

            Returns:
                None
        """

        with tf.Session() as sess:
            if self.save_graph:
                file_writer = tf.summary.FileWriter('logs', sess.graph)

            # start the timer
            then = time.time()
            tf.global_variables_initializer().run()

            # the main loop!
            for i in range(self.samples):
                sess.run(self.ode_op(i))
                # fire S2
                if i == self.s2_time:
                    sess.run(self.s2_op())
                # draw a frame every 1 ms
                if i % 10 == 0:
                    im.imshow(self.image())

            if self.timeline:
                # options and run_metadata are needed for tracing
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                sess.run(self.ode_op(self.samples),
                    options=options, run_metadata=run_metadata)
                # Create the Timeline object for the last iteration
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline_br.json', 'w') as f:
                    f.write(chrome_trace)

        print('elapsed: %f sec' % (time.time() - then))
        im.wait()   # wait until the window is closed

    def define(self):
        """
            A placeholder for a method to be defined in a subclass
            It should define the model and set self.ode_op and
            self.s2_op in addition to any state needed for self.image
        """
        pass

    def image(self):
        """
            A placeholder for a method to be defined in a subclass.
            It should return a [height x width] float ndarray in the range 0 to 1
            that encodes the transmembrane potential in grayscale
        """
        return None

    def ode_op(self, tick):
        if hasattr(self, '_ode_op'):
            return self._ode_op
        elif tick % self.fast_slow_ratio == 0:
            return self._ode_slow_op
        else:
            return self._ode_fast_op

    def s2_op(self):
        return self._s2_op

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def jit_scope(self):
        try:
            scope = tf.contrib.compiler.jit.experimental_jit_scope
        except:
            scope = None

        if scope:
            return scope()
        else:
            return self
