---
title: 'fib-tf: A Python framework on the top of [TensorFlow](http://tensorflow.org) for 2D cardiac electrophysiology simulation'
tags:
  - Python
  - computational medicine
  - cardiac electrophysiology
  - TensorFlow
  - GPU
authors:
  - name: Shahriar Iravanian
    orcid: 0000-0003-2132-1543
    affiliation: "1" # (Multiple affiliations must be quoted)  
affiliations:
 - name: Emory University
   index: 1
date: 25 April 2018
bibliography: paper.bib
---

# Summary

`fib_tf` is a python package developed on the top of the machine-learning library TensorFlow for cardiac electrophysiology simulation. While TensorFlow is primarily designed for machine learning, it also provides a framework to perform general-purpose multidimensional tensor manipulation.

The primary goal of `fib_tf` is to test and assess the suitability of TensorFlow for solving systems of stiff ordinary differential equations (ODE), such as those encountered in cardiac modeling. Its main targets are massively parallel hardware architectures (e.g., Graphics Processing Units).

`fib_tf` solves the monodomain reaction-diffusion equations governing cardiac electrical activity by a combination of the finite-difference and explicit Euler methods. It is used to simulate two cardiac ionic models: the 4-variable Cherry-Ehrlich-Nattel-Fenton canine left-atrial and the 8-variable Beeler-Reuter ventricular models.

`fib_tf` serves as a testbed to try various general and TensorFlow-specific optimization techniques. We showed that activating Just-In-Time (JIT) compilation significantly improves the performance. Moreover, by applying a multitude of optimization methods, including dataflow graph unrolling, the Rush-Larsen method, the Chebyshev polynomials approximation, and multi-rate integration, we have achieved a performance within a factor 2-3 of hand-optimized C++ codes. The motivation behind and the details of each method are described in the documentation.

Based on our experiments, TensorFlow applicability is not limited to the machine-learning domain. TensorFlow is a valuable tool for the development of efficient and complex ODE solvers. `fib_tf` can act as a framework for such solvers and is not limited to the cardiac electrophysiology applications. Specially, it is useful in rapid prototyping and testing of new algorithms and methods.

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

Figures can be included like this: ![Example figure.](figure.png)

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
