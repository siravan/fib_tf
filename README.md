# Introduction

This project contanis few 2D cardiac electrophysiology models using the TensorFlow framework. The primary goal is to test and assess the suitability of Tensorflow for solving systems of stiff ordinary differential equations (ODE) needed for cardiac modeling. While Tensorflow is primarily designed for machine learning applications, it also provides a general framework for performing multidimensional tensor calculations. That said, Tensorflow currently lacks some useful features needed for solving ODEs.

There are more than 100 different ionic models that describe cardiac electriocal activity in various degrees of detail (see this [page](http://www.scholarpedia.org/article/Models_of_cardiac_cell) for an overview of the available models). Most are based on the classic Hodgkin-Huxley model and define the time-evolution of different transmembrane and sarcoplasmic ionic currents in the form of nonlinear first-order ODEs. Generally, the state vector for these models include the transmembrane voltage, various gating variables and ionic concentration. By nature, these models have to deal with different time scales and are therefore classified as *stiff*. Commonly, these models are solved using the explicit Euler method, usually with a closed form for the integration of the gating variables (the Rush-Larsen technique). Other techniques used are the implicit Euler and adaptive time-step methods. Higher order integration methods such as Runge-Kutta (RK4) are not particularly helpful.

To study wave progression, 1D (cable), 2D and 3D cardiac models are used by coupling up to millions of ODEs together. Until few years ago, simulating one second of cardiac activity in a detailed 3D model with realistic ionic currents could easily take upward of an hour. With the advent of massively parallel architectures (specially GPUs), this task can now be done with near real-time efficiency. 

Programming GPUs (e.g., using C/C++ CUDA or OpenCL) is not a trivial task and is both time-consuming and error-prone. This is where Tensorflow can be very useful. Because Tensorflow takes care of the lower level details of running the model on a GPU, the programmer can focus on the high level model and be more productive. In addition, Tensorflow allows running the model on multiple GPUs or even clusters. While it is not expected that Tensorflow beats a hand crafted optimized CUDA kernel, we hope to get a reasonable performance. The main question we are trying to answer is whether this performance is good enough. 

# Quickstart

We have tested the software on Ubuntu 14.04 and 16.04 machines with Tensorflow 1.4, Python 3.6, and GTX 1080 or GTX Titan X GPUs (CUDA 8.0, CUDNN 7.0). 

## Requirements:

  1. Python 3
  2. Numpy
  3. SDL2 (no need for PySDL2, just the basic SDL2 library). You can download it [here](https://wiki.libsdl.org/Installation).
  4. Tensorflow (of course!)
  5. libcupti (optional, to profile the model)

It should be noted the Tensorflow installable come in different flavors. This software can work on CPU only (after changing line with `tf.device('/device:GPU:0'):` to `tf.device('/device:CPU:0'):`), but it would be too slow. However, just having the GPU enabled version of Tensorflow is not enough either, rather **you need a Tensorflow version with XLA JIT enabled**. The way Tensorflow works on GPU is to assign each operation to a CUDA kernel. This works well for machine learning, where each operation does lots of computation. But for solving systeoms of ODEs, most of calculations is not element-wise and we have many light operations. Under this condition, the overhead of kernel launches and data movement to and from the Global GPU memory causes a bottleneck. Fortunately, Tensorflow has an experimental [Just In Time (JIT) compiler](https://www.tensorflow.org/performance/xla/jit) that allows fusing multiple kernels into one. In our experience, enabling JIT makes the code 2-4 times faster. Unfortunatelly, this feature is still not availalbe in the stock versions of Tensorflow. To enable it, you need to compile Tensorflow from [source](https://www.tensorflow.org/install/install_sources).

# Download

## Installation

Download the software using

`git clone https://github.com/siravan/fib_tf.git`

Change into the `fib_tf` directory and run the models. You can run the Cherry-Ehrlich-Nattel-Fenton (4v) canine left-atrial model as

`python3 4v.py`

and the 8-variable modifed Beeler-Reuter as

`python3 br.py`

# Files

- screen.py: provides a simple SDL2 screen to plot a 2D numpy array in grayscale.
- dll.py: a helper file from PySDL2 to find and load the appropriate SDL2 library (libSDL2.so in linux).
- ionic.py: contains the base class for implementing ionic models. 
- 4v.py: implements the 4-variable Cherry-Ehrlich-Nattel-Fenton canine left-atrial model.
- br.py: implements the 8-variable modifed Beeler-Reuter.
- br_multirate.py: an experimental implemention of the 8-variable modifed Beeler-Reuter model  with multirate integration. 


---
©2017 Shahriar Iravanian (siravan@emory.edu). 


