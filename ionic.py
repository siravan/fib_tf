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
        self.width = props['width']
        self.height = props['height']
        self.dt = props['dt']
        self.diff = props['diff']
        self.cheby = props['cheby']
        self.timeline = props['timeline']
        self.save_graph = props['save_graph']
        self.samples = props['samples']
        self.s2_time = props['s2_time']

    def laplace(self, X):
        """
            Compute the 2D laplacian of an array directly and without using conv
            faster than laplace(x)
        """
        l = (X[:-2,1:-1] + X[2:,1:-1] + X[1:-1,:-2] + X[1:-1,2:] +
             0.5 * (X[:-2,:-2] + X[2:,:-2] + X[:-2,2:] + X[2:,2:]) -
             6 * X[1:-1,1:-1])
        paddings = tf.constant([[1,1], [1,1]])
        return tf.pad(l, paddings, 'CONSTANT', name='laplacian')

    def enforce_boundary(self, X):
        """
            enforcing the no-flux boundary condition
        """
        paddings = tf.constant([[1,1], [1,1]])
        return tf.pad(X[1:-1,1:-1], paddings, 'SYMMETRIC', name='boundary')

    def run(self, im):
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

        with tf.Session() as sess:
            if self.save_graph:
                file_writer = tf.summary.FileWriter('logs', sess.graph)

            # start the timer
            then = time.time()
            tf.global_variables_initializer().run()

            # the main loop!
            for i in range(self.samples):
                sess.run(self.ode_op)
                # fire S2
                if i == self.s2_time:
                    sess.run(self.s2_op)
                # draw a frame every 1 ms
                if i % 10 == 0:
                    im.imshow(self.normalized_vlt())

            if self.timeline:
                # options and run_metadata are needed for tracing
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                sess.run(self.ode_op, options=options, run_metadata=run_metadata)
                # Create the Timeline object for the last iteration
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline_br.json', 'w') as f:
                    f.write(chrome_trace)

        print('elapsed: %f sec' % (time.time() - then))
        im.wait()   # wait until the window is closed
