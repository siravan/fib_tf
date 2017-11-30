# Introduction

This project contanis a sample of 2D cardiac electrophysiology models using the TensorFlow framework. 
The primary goal is to test and assess the suitability of Tensorflow for solving systems of stiff ordinary differential
equations (ODE) needed for 2D cardiac modeling. While Tensorflow is primarily designed for machine learning applications, 
it also provides a general framework for performing multidimensional tensor calculations. That said, Tensorflow currently 
lacks some useful features needed for solving ODEs.

There are more than 100 differnt ionic models that describe cardiac electriocal activity in various degrees of detail.
Most are based on the classic Hodgkin-Huxley model and provide equations to find the time-evolution of different
transmembrane and sarcoplasmic ionic currents in the form of nonlinear first-order ODEs. Generally, the state vector for these models include the transmembrane voltage, variuos gating variables and ionic concentration. By nature, these models have to deal with different time scales and are therefore classified as *stiff*. Commonly, these models are solved using the explicit Euler method, usually with a closed form for the integration of the gating variables (the Rush-Larsen technique). Other techniques used are the implicit Euler and adaptive time-step methods. Higher order integration methods such as Runge-Kutta (RK4) are not particularly helpful.

We are not only interested is solving the ionic current ODE for a single cell. To study wave progression, 1D (cable), 2D and 3D cardiac models are used by coupling up to millions of ODEs together. Until few years ago, simulating one second of cardiac activity in a detailed 3D model with realistic ionic currents could easily talk upward of one hour. With the advent of massively parallel architectures (specially on GPUs), this type of calculation can now be done with near real-time efficiency. 

Programming GPUs (e.g., using C/C++ CUDA or OpenCL) is not a trivial task and is time-consuming and error-prone. This is where Tensorflow can be very useful. While it is not expected that Tensorflow beats a hand crafted optimized CUDA kernel, we hope to get reasonable performance. The goal of the current project is to verify this.

# Requirements:

  1. Python3
  2. Numpy
  3. SDL2 (no need to PySDL2, just the basic SDL2 library)
  4. Tensorflow (of course!)
  
We have tested the software on Ubuntu 14.04 and 16.04 machines with Tensorflow 1.4, and GTX 1080 or GTX Titan X GPUs (CUDA 8.0, CUDNN 7.0). It should be noted the Tensorflow installable come in different flavors. This software works on CPU only (after changing line with `tf.device('/device:GPU:0'):` to `tf.device('/device:CPU:0'):`), but it would be too slow. However, just having the GPU enabled version of Tensorflow is not enough. 

**Important**: The way Tensorflow works on GPU is to assign each operation to a CUDA kernel. This works well for machine learning, where each operation does lots of computation. But for solving systeoms of ODEs, most of calculations is not element-wise and we have many light operations. Under this condition, the overhead of kernel launches and data movement to and from the Global GPU memory causes a bottleneck. Fortunately, Tensorflow has an experimental [Just In Time (JIT) compiler](https://www.tensorflow.org/performance/xla/jit) that allows fusing multiple kernels into one. In our experient, enabling JIT makes the code 2-4 times faster. Unfortunatelly, this feature is still not availalbe in the stock versions of Tensorflow. To enable it, you need to compile Tensorflow from [source](https://www.tensorflow.org/install/install_sources).



