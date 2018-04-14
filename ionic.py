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
    def __init__(self, config):
        for key, val in config.items():
            setattr(self, key, val)
        self.phase = None
        self._ops = {}
        self.defined = False
        self.dt_per_step = 1

    def laplace(self, X):
        """
            Compute the 2D laplacian of an array directly and without using conv
            it also adds the phase field correction if self.phase is defined
        """
        l = (X[:-2,1:-1] + X[2:,1:-1] + X[1:-1,:-2] + X[1:-1,2:] +
             0.5 * (X[:-2,:-2] + X[2:,:-2] + X[:-2,2:] + X[2:,2:]) -
             6 * X[1:-1,1:-1])

        if self.phase is not None:
            if not hasattr(self, 'ϕ'):
                self.ϕ = tf.Variable(self.phase, name='phi')
            l += self.phase_field(X)

        paddings = tf.constant([[1,1], [1,1]])
        return tf.pad(l, paddings, 'CONSTANT', name='laplacian')

    def phase_field(self, X):
        """
            Compute the 2D phase field
            it assumes self.ϕ exists and is the same shape as X
        """
        ϕ = self.ϕ
        f = ((X[2:,1:-1] - X[:-2,1:-1]) * (ϕ[2:,1:-1] - ϕ[:-2,1:-1]) +
             (X[1:-1,2:] - X[1:-1,:-2]) * (ϕ[1:-1,2:] - ϕ[1:-1:,:-2])
             ) / (4 * ϕ[1:-1,1:-1])
        return f

    def add_hole_to_phase_field(self, x, y, radius):
        """
            adds a circular hole, centered at (x,y), to the phase field
            it creates the phase field if it does not already exist

            NOTE: this method should be called before calling define
        """
        if self.defined:
            raise AssertionError('add_hole_to_phase_field should be called before calling define')

        if self.phase is None:
            self.phase = np.ones([self.height, self.width], dtype=np.float32)

        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        dist = np.hypot(xx - x, yy - y)
        self.phase *= np.array(0.5*(np.tanh(dist - radius) + 1.0), dtype=np.float32)
        # we set the minimum phase field at 1e-5 to avoid division by 0 in phase_field
        self.phase = np.maximum(self.phase, 1e-5)

    def enforce_boundary(self, X):
        """
            Enforcing the no-flux (Neumann) boundary condition
        """
        paddings = tf.constant([[1,1], [1,1]])
        return tf.pad(X[1:-1,1:-1], paddings, 'SYMMETRIC', name='boundary')

    def rush_larsen(self, g, g_inf, g_tau, dt, name=None):
        return tf.clip_by_value(g_inf - (g_inf - g) * tf.exp(-dt/g_tau), 0.0,
                                1.0, name=name)

    def add_pace_op(self, name, loc, v):
        """
            adds a pacemaker op to the list of operation
            loc is one of 'left', 'right', 'top', 'bottom',
            'luq' (left upper quadrant), 'llq' (left lower quadrant),
            'ruq' (right upper quadrant), and 'rlq' (right lower quadrant)

            NOTE: this method should be called after calling define
        """
        if not self.defined:
            raise AssertionError('add_hole_to_phase_field should be called after calling define')

        s = np.full([self.height, self.width], self.min_v, dtype=np.float32)
        if loc == 'left':
            s[:,:5] = v
        elif loc == 'right':
            s[:,-5:] = v
        elif loc == 'top':
            s[:5,:] = v
        elif loc == 'bottom':
            s[-5:,:] = v
        elif loc == 'luq':
            s[:self.height//2, :self.width//2] = v
        elif loc == 'llq':
            s[self.height//2:, :self.width//2] = v
        elif loc == 'ruq':
            s[:self.height//2, self.width//2:] = v
        elif loc == 'rlq':
            s[self.height//2:, self.width//2:] = v
        else:
            print('undefined pace location')
        self._ops[name] = self.pot().assign(tf.maximum(self.pot(), s))

    def fire_op(self, name):
        self._sess.run(self._ops[name])

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
            self._sess = sess
            if self.save_graph:
                file_writer = tf.summary.FileWriter('logs', sess.graph)

            # start the timer
            then = time.time()
            tf.global_variables_initializer().run()
            v0 = self.min_v
            last_spike = 0
            samples = int(self.duration / (self.dt_per_step * self.dt))

            # the main loop!
            for i in range(samples):
                sess.run(self.ode_op(i))
                yield i
                # draw a frame every 1 ms
                if im and i % (self.dt_per_plot / self.dt_per_step) == 0:
                    image = self.image()
                    if self.phase is not None:
                        image *= self.phase
                    im.imshow(image)
                    v1 = image[1, self.width//2]
                    if v1 >= 0.5 and v0 < 0.5:
                        cl = (i - last_spike) * self.dt_per_step * self.dt
                        print('wavefront reaches the middle top point at %d, cycle length is %d' % (i, cl))
                        last_spike = i
                    v0 = v1

            if self.timeline:
                # options and run_metadata are needed for tracing
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                sess.run(self.ode_op(self.samples),
                    options=options, run_metadata=run_metadata)
                # Create the Timeline object for the last iteration
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(self.timeline_name, 'w') as f:
                    f.write(chrome_trace)

        print('elapsed: %f sec' % (time.time() - then))
        if im:
            im.wait()   # wait until the window is closed

    def define(self, s1=True):
        """
            A placeholder for a method to be defined in a subclass
            It should define the model and set self.ode_op and
            self.s2_op in addition to any state needed for self.image
        """
        self.defined = True

    def image(self):
        """
            A placeholder for a method to be defined in a subclass.
            It should return a [height x width] float ndarray in the range 0 to 1
            that encodes the transmembrane potential in grayscale
        """
        pass

    def pot(self):
        pass

    def ode_op(self, tick):
        """
            Returns the ODE operation for time tick (in dt unit)
        """
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
        """
            Returns an XLA jit_scope if available; otherwise self is returned
            and provides a dummy Context
        """
        try:
            scope = tf.contrib.compiler.jit.experimental_jit_scope
        except:
            scope = None

        if scope:
            return scope()
        else:
            return self
