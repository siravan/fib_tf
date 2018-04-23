# Introduction

**fib_tf** is a python framework developed on the top of [TensorFlow](http://tensorflow.org) for 2D cardiac electrophysiology simulation. While TensorFlow is primarily designed for machine learning applications, it also provides a general framework for performing multidimensional tensor manipulation. The primary goal of **fib_tf** is to test and assess the suitability of TensorFlow for solving systems of stiff ordinary differential equations (ODE) needed for cardiac modeling.

Programming Graphics Processing Units (GPU) using C/C++ CUDA or OpenCL is not a trivial task and is both time-consuming and error-prone. This is where TensorFlow can be very useful. Because TensorFlow takes care of the lower level details of running the model on a GPU, the programmer can focus on the high-level model and be more productive. Additionally, TensorFlow allows running the model on multiple GPUs or even clusters. While it is not expected that a TensorFlow ODE solver beats a handcrafted optimized CUDA kernel, we hope to get a reasonable performance. The main question we are trying to answer is whether this performance is good enough.

# Requirements:

We have tested the software on Ubuntu 14.04 and 16.04 machines with Tensorflow 1.4, Python 3.6, and GTX 1080 or GTX Titan X GPUs (CUDA 8.0, CUDNN 7.0).

The following software packages are needed:

  1. Python 3 (sorry, no python 2.7!)
  2. Numpy
  3. SDL2 (no need for PySDL2, just the basic SDL2 library). You can download it [here](https://wiki.libsdl.org/Installation).
  4. TensorFlow
  5. libcupti (optional, it is needed for profiling)

It should be noted the TensorFlow installables come in different flavors. **fib_tf** can work on CPU only (after changing the line `tf.device('/device:GPU:0'):` to `tf.device('/device:CPU:0'):`), but it would be too slow.

However, just having a GPU-enabled version of Tensorflow is not enough either, rather **you need a TensorFlow version with XLA JIT enabled to have the best performance**. See [details](details.html) for discussion on why this is so important. Fortunately, TensorFlow has an experimental [Just In Time (JIT) compiler](https://www.tensorflow.org/performance/xla/jit) that allows fusing multiple kernels into one. But this feature is still not available in the stock versions of TensorFlow. To enable it, you need to compile Tensorflow from source. This is not the easiest installation process, but is very well doable as long as you just [follow the instructions!](https://www.tensorflow.org/install/install_sources)

# Download

Download **fib_tf** as

`git clone https://github.com/siravan/fib_tf.git`

Change into the `fib_tf` directory and run the models.

# Main Files

- *screen.py*: provides a simple SDL2 screen to plot a 2D numpy array in grayscale.
- *dll.py*: a helper file from PySDL2 to find and load the appropriate SDL2 library (libSDL2.so in Linux).
- *ionic.py*: contains the base class for implementing ionic models.
- *fenton.py*: implements the 4-variable Cherry-Ehrlich-Nattel-Fenton canine left-atrial model.
- *br.py*: implements the 8-variable modified Beeler-Reuter.
- *README.md*: this file!
- *details.html*: A detailed discussion of the software and techniques used.

# Test

You can run the Cherry-Ehrlich-Nattel-Fenton (*4v*) canine left-atrial model as

`python3 fenton.py`

and the 8-variable modified Beeler-Reuter model as

`python3 br.py`

# Documentation

The details of the ionic models and various methods and tricks used to improve the performance are discussed [here](details.html).

To run **fib_tf** models interactively, you first need to import the desired model. For the *4v* model, it is imported as `import fenton`. Next, we need to define a configuration dict. For the *4v* model, it is defined like

```python
config = {
    'width': 512,           # screen width in pixels
    'height': 512,          # screen height in pixels
    'dt': 0.1,              # integration time step in ms
    'dt_per_plot' : 10,     # screen refresh interval in dt unit
    'diff': 1.5,            # diffusion coefficient
    'duration': 1000,       # simulation duration in ms
    'timeline': False,      # flag to save a timeline (profiler)
    'timeline_name': 'timeline_4v.json',
    'save_graph': True      # flag to save the dataflow graph
}
```

For the Beeler-Reuter model, we import it as `import br` and define the configuration dict as:

```python
config = {
    'width': 512,           # screen width in pixels
    'height': 512,          # screen height in pixels
    'dt': 0.1,              # integration time step in ms        
    'dt_per_plot': 10,      # screen refresh interval in dt unit
    'diff': 0.809,          # diffusion coefficient
    'duration': 1000,       # simulation duration in ms
    'skip': False,          # optimization flag: activate multi-rate
    'cheby': True,          # optimization flag: activate Chebysheb polynomials
    'timeline': False,      # flag to save a timeline (profiler)
    'timeline_name': 'timeline_br.json',
    'save_graph': False     # flag to save the dataflow graph
}
```

Then, we need to create a model object. For example:

```python
model = br.BeelerReuter(config)
```

In a minimum application, we construct a dataflow graph by calling `model.define()` and run it as:

```python
for i in model.run():
    pass
```

But this is not very interesting, as there is no visual output. So, let's define a screen object:

```python
from screen import Screen
...
im = Screen(model.height, model.width, 'Beeler-Reuter Model')
```

Now, we can run the model and see what happens:

```python
for i in model.run(im):
    pass
```

A single planar wave! Better, but still not very interesting. We can induce a spiral wave by a S1-S2 stimulation protocol. S1 is implicit, but we need to define S2:

```python
model.add_pace_op('s2', 'luq', 10.0)
```

This define an S2 stimulation operation, where the value of the transmembrane voltage over the *left upper quadrant (luq)* of the screen is set to 10 mV. Now, we modify the main loop to

```python
s2 = model.millisecond_to_step(300)     # 300 ms

for i in model.run(im):
    if i == s2:
        model.fire_op('s2')
```

This time we get a nice spiral wave. Note that we need to convert the desired time of S2 to the number of steps returns from **model.run**.

We can do even better. Let's add a circular obstacle to the medium by adding this line *before* calling `model.define()`:

```python
model.add_hole_to_phase_field(150, 200, 40) # center=(150,200), radius=40
```

Success! We get a spiral wave that anchors to the obstacle:

![](br.png)

---
@2017-2018 [Shahriar Iravanian](siravan@emory.edu)
