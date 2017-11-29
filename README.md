# Introduction

This project contanis a sample of 2D cardiac electrophysiology models using the TensorFlow framework. 
The primary goal is to test and assess the suitability of Tensorflow for solving systems of stiff ordinary differential
equations (ODE) needed for 2D cardiac modeling. While Tensorflow is primarily designed for machine learning applications, 
it also provides a general framework for performing multidimensional tensor calculations. That said, Tensorflow currently 
lacks some useful features needed for solving ODEs.

There are more than 100 differnt ionic models that describe cardiac electriocal activity in variuos degrees of detail.
Most of these models are based on the classic Hodgkin-Huxley model and provide equations to find the time-evolution of various 
transmembrane and sarcoplasmic ionic currents. Generally, the state vector for these models include the transmembrane voltage, 
variuos gating variables and ionic concentration. By nature, these models have to deal with different time scales and are 
therefore classified as *stiff* ODEs. Commonly, these models are solved using the explicit Euler method, usually with a 
closed form for the integration of the gating variables (the Rush-Larsen technique).


We are specially interested in running the Tensorflow applications on GPU (using CUDA).
