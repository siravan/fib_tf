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

import numpy as np
import time
import tensorflow as tf
from tensorflow.python.client import timeline

class IonicModel:
    """
        IonicModel is the base class for cardiac electrophysiology simulation
    """

    def __init__(self, config):
        for key, val in config.items():
            setattr(self, key, val)
        self.phase = None
        self._ops = {}
        self.defined = False
        self.dt_per_step = 1
        self.cl_observer = None

    def laplace(self, X):
        """
            laplace computes the 2D Laplacian of X directly and without using conv
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
            phase_field computes the 2D phase field correction
            it assumes self.ϕ exists and is the same shape as X
        """
        ϕ = self.ϕ
        f = ((X[2:,1:-1] - X[:-2,1:-1]) * (ϕ[2:,1:-1] - ϕ[:-2,1:-1]) +
             (X[1:-1,2:] - X[1:-1,:-2]) * (ϕ[1:-1,2:] - ϕ[1:-1:,:-2])
             ) / (4 * ϕ[1:-1,1:-1])
        return f

    def add_hole_to_phase_field(self, x, y, radius, neg=False):
        """
            add_hole_to_phase_field adds a circular hole, centered at (x,y), to
            the phase field. It creates a phase field if it does not already exist

            if neg == True, the hole remains and its outside is excluded

            NOTE: this method should be called before calling define
        """
        if self.defined:
            raise AssertionError('add_hole_to_phase_field should be called before calling define')

        if self.phase is None:
            self.phase = np.ones([self.height, self.width], dtype=np.float32)

        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        dist = np.hypot(xx - x, yy - y)
        if neg:
            self.phase *= np.array(0.5*(np.tanh(0.1*(radius - dist)) + 1.0), dtype=np.float32)
        else:
            self.phase *= np.array(0.5*(np.tanh(dist - radius) + 1.0), dtype=np.float32)
        # we set the minimum phase field at 1e-5 to avoid division by 0 in phase_field
        self.phase = np.maximum(self.phase, 1e-5)

    def enforce_boundary(self, X):
        """
            enforce_boundary enforces the no-flux (Neumann) boundary condition
            on the medium borders
        """
        paddings = tf.constant([[1,1], [1,1]])
        return tf.pad(X[1:-1,1:-1], paddings, 'SYMMETRIC', name='boundary')

    def rush_larsen(self, g, g_inf, g_tau, dt, name=None):
        """
            rush_larsens is a helper funcion to implement the Rush-Larsen
            direct integration of the gating variables
        """
        return tf.clip_by_value(g_inf - (g_inf - g) * tf.exp(-dt/g_tau), 0.0,
                                1.0, name=name)

    def add_pace_op(self, name, loc, v):
        """
            add_pace_op adds a stimulator/pacing op to the list of operations

            loc is one of
                'left',
                'right',
                'top',
                'bottom',
                'luq' (left upper quadrant),
                'llq' (left lower quadrant),
                'ruq' (right upper quadrant),
                'rlq' (right lower quadrant)

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
        """
            fire_op activates the operation name already added by add_pace_op
        """
        self._sess.run(self._ops[name])

    def run(self, im=None, keep_state=False, block=True):
        """
            runs first defines a TensorFlow Session, attaches the dataflow to it,
            and then run it. The model should be defined first by calling define().
            runs is a generator and is used like

                for i in model.run(im):
                    if i == s2:
                        model.fire_op('s2')

            Args:
                im: A Screen used to paint the transmembrane potential

            Returns:
                step count
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
            self.samples = int(self.duration / (self.dt_per_step * self.dt))

            # the main loop!
            for i in range(self.samples):
                sess.run(self.ode_op(i))
                yield i
                # draw a frame every 1 ms
                if im and i % int(self.dt_per_plot / self.dt_per_step) == 0:
                    image = self.image()
                    if self.phase is not None:
                        image *= self.phase
                    im.imshow(image)
                    v1 = image[20, self.width//2]
                    if v1 >= 0.5 and v0 < 0.5:
                        cl = (i - last_spike) * self.dt_per_step * self.dt
                        if self.cl_observer is None:
                            print('wavefront reaches the middle top point at %d, cycle length is %d' % (i, cl))
                        else:
                            self.cl_observer(i, cl)
                        last_spike = i
                    v0 = v1

            if keep_state:
                self.state = {}
                for s in self._State:
                    self.state[s] = self._State[s].eval()

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
        if block and im:
            im.wait()   # wait until the window is closed

    def millisecond_to_step(self, t):
        """
            millisecond_to_step converts t in milliseconds to a step count
            returns from run()
        """
        return int(t / (self.dt_per_step * self.dt))

    def define(self, s1=True):
        """
            define is a placeholder to be replaced in subclasses
            It should define the model and set self._ode_op
            in addition to any state needed for self.image
        """
        self.defined = True

    def image(self):
        """
            image is a placeholder to be replaced in subclasses
            It should return a [height x width] float ndarray in the range 0 to 1
            that encodes the transmembrane potential in grayscale
        """
        pass

    def pot(self):
        """
            pot is a placeholder to be replaced in subclasses
            it returns the transmembrane voltage
        """
        pass

    def ode_op(self, tick):
        """
            ode_op returns the ODE operation
        """
        if hasattr(self, '_ode_op'):
            return self._ode_op
        elif tick % self.fast_slow_ratio == 0:
            return self._ode_slow_op
        else:
            return self._ode_fast_op

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def jit_scope(self):
        """
            jit_scope returns an XLA jit_scope if available; otherwise self is
            returned as a dummy Context
        """
        try:
            scope = tf.contrib.compiler.jit.experimental_jit_scope
        except:
            scope = None

        if scope:
            return scope()
        else:
            return self
