# Background

[Tensorflow](https://www.tensorflow.org) is a machine-learning framework
developed by Google and initially released publicly in 2015. Based on its
website, "TensorFlow is a tool for machine learning. While it contains a wide
range of functionality, TensorFlow is mainly designed for deep neural network
models." Here, our goal is to test the suitability of Tensorflow for solving
coupled systems of ordinary differential equations (ODE) that arise in the
numerical modeling of cardiac electrophysiology systems.

TensorFlow is primarily, but not exclusively, a machine learning framework.
However, TensorFlow graphs are powerful models that can be useful in variety of
non-machine learning applications. Specially, we are interested in using TensorFlow to solve coupled systems of ordinary differential equations (ODE).

The main power of TensorFlow is the ability to run the model graphs on
heterogeneous hardwares, including CPU, GPU, or even Google developed Tensor
Processing Units (TPU).

# TOC
* [Introduction to **fib_tf**](#introduction-to-fib-tf)
* [*4v* Cardiac Model](#4v-cardiac-model)
* [A Simple 4v Solver](#a-simple-4v-solver)
* [The Root Cause of Slowness](#the-root-cause-of-slowness)
* [Just-In-Time (JIT) Compilation to Rescue!](#just-in-time-jit-compilation-to-rescue-)
* [Other Optimization Tricks](#other-optimization-tricks)
    * [Laplacian](#laplacian)
    * [Graph Unrolling](#graph-unrolling)
* [The Beeler-Reuter Ionic Model](#the-beeler-reuter-ionic-model)
    * [The Rush-Larsen Method](the-rush-larsen-method)
    * [Using the Chebyshev Polynomials](#using-the-chebyshev-polynomials)
    * [Multi-rate Integration](#multi-rate-integration)


# <a name='introduction-to-fib-tf'></a> Introduction to **fib_tf**

**fib_tf** is a python framework developed on the top of TensorFlow for 2D
cardiac electrophysiology simulation using explicit Euler method.

**fib_tf** is designed to run on GPU. Explicit Cardiac electrophysiology models
are particuarly suitable for running on GPU, as the main dependency between ODEs
is through Laplacian, which is only a small fraction of the total computational
cost.

Our goal is to optimize the **fib_tf** code to make its
running time close to a hand-optimized CUDA C++ code running the same model.

In this document, we first describe a straightforward translation of cardiac
models into TensorFlow. We then explains various optimizing tricks needed to
make **fib_tf** performance acceptable.

# <a name='4v-cardiac-model'></a> *4v* Cardiac Model

Let's start by writing a simple 2D solver for the Cherry-Ehrlich-Nattel-Fenton 4-variable atrial model (*4v* from here on). With only four variables, this
model provides a good entry point without being trivial.

The basis of most cardiac electrophysiology simulations is the standard mono-domain reaction-diffusion equation,

\[
  \partial V / \partial t = \nabla (D  \nabla V) - \frac {I_{ion}} {C_m},
\]

where the scalar field $V$ is the transmembrane potential, $D$ is the diffusion tensor, $I_{ion}$ is the sum of all ionic currents, and $C_m$ is the membrane capacitance ($C_m$ is usually set to 1 μF/cm). In general, the diffusion part ($\nabla (D  \nabla V)$) denotes the coupling between neighboring cells and is modeled as a partial differential equation (PDE); whereas, the reaction part (    $I_{ion}/C_m$) is solved as a system of ODEs.

For the sake of simplicity, we assume a uniform isotropic medium (we will add a *phase field* later on to accommodate non-rectangular geometries). Therefore, tensor $D$ is reduced to a scalar $g$, denoting the global diffusion coefficient. We have

\[
  \partial V / \partial t = g\,\nabla^2{V} - \frac {I_{ion}} {C_m}.
 \]

The *4v* model has four state variables: $u$, $v$, $w$, and $s$, where $u$ represents the transmembrane voltage; the other three are gating variables with different time constants. All four variables range from 0 to 1.
The initial condition is $(u, v, w, s) = (0, 1, 1, 0)$, corresponding to a resting cell.

Using the Euler explicit integration, we have

\[
    v(t+\Delta{t}) = v(t) + \Delta{t}\,dv/dt,
\]

where we have made the dependence on $t$ explicit. Similarly,

\[
    w(t+\Delta{t}) = w(t) + \Delta{t}\,dw/dt,
\]

and,

\[
    s(t+\Delta{t}) = s(t) + \Delta{t}\,ds/dt.
\]

The time integration formula for $u$ has an additional term related to the
Laplacian:

\[
    u(t+\Delta{t}) = u(t) + \Delta{t}\,du/dt + g\,\Delta{t}\,\nabla^2{u}.
\]

We need to calculate the time derivatives of the state variables at each time
point. First, we update the dynamic time constants,

\[
  \tau_v^n =
  \begin{cases}
    \tau_{v2}^n & \quad \text{if } u \geq u_v \\
    \tau_{v1}^n & \quad \text{if } u < u_v
  \end{cases},
\]

and,

\[
  r_s =
  \begin{cases}
    r_s^p & \quad \text{if } u \geq u_c \\
    r_s^n & \quad \text{if } u < u_c
  \end{cases}.
\]

Next, the instantaneous currents are calculated:

\[
  I_{fi} =
  \begin{cases}
    -v(u-u_c)(1-u) / \tau_d & \quad \text{if } u \geq u_c \\
    0 & \quad \text{if } u < u_v
  \end{cases},
\]

\[
  I_{si} = -ws / \tau_{si},
\]

\[
  I_{so} =
  \begin{cases}
    \tau_a & \quad \text{if } u \geq u_{so} \\
    u/\tau_0 & \quad \text{if } u < u_{so}
  \end{cases}
\]

and,

\[
  I_{so} = I_{so}' + 0.5 (a_{so} - \tau_a) (1 + \tanh((u-b_{so})/c_{so}) ).
\]

Finally, we obtain the time derivative of the state variables

\[
  du/dt = -(I_{fi} + I_{si} + I_{so}),
\]

\[
  dv/dt =
  \begin{cases}
    (1-v) / \tau_v^p & \quad \text{if } u < u_c \\
    -v / \tau_v^n & \quad \text{if } u \geq u_c
  \end{cases},
\]

\[
  dw/dt =
  \begin{cases}
    (1-w) / \tau_w^p & \quad \text{if } u < u_c \\
    -w / \tau_w^n & \quad \text{if } u \geq u_c
  \end{cases},
\]

and,

\[
  ds/dt = r_s (0.5 (1 + \tanh(k(u-u_{c,si}))) -s).
\]

# <a name='a-simple-4v-solver'></a> A Simple 4v Solver

The TensorFlow website has an [example](https://www.tensorflow.org/tutorials/pdes) of solving a simple partial differential equation. This example served as the starting point of the current project; albeit, in the end, only a tiny fraction of its code remained in the final model!

In this section, we present a straightforward translation of the 4v model into
TensorFlow code by modifying the above example. You can find it source code in **fenton_simple.py**. As we will see, the resulting code is not very efficient. Afterward, we will discuss various modifications to make it faster.

In a typical TensorFlow applications, we first define a TensorFlow graph. A TensorFlow graph is a directed acyclic graph, where nodes correspond to tensors (general multi-dimensional arrays) or operations on tensors, and links denote data movement among the nodes. Such a graph can then be executed once or multiple times.

**fenton_simple.py** begins with few utility functions. **laplace()** and its helper functions (**make_kernel()** and **simple_conv()**) are copied verbatim from the website example. **enforce_boundary()** ensures a no-flux (Neumann) boundary condition.

The TensorFlow example also defines a utility function **DisplayArray()** to visualize the results of simulation. This function only works in a Jupyter notebook environment. I prefer a more general visualization routine that also works from ipython and command line. Therefore, instead of **DisplayArray()**, we import **Screen** from **screen.py** that uses an SDL2 window.

The bulk of the code is contained in the class **Fenton4vSimple**. The model graph is defined in **Fenton4vSimple.define()**, where we introduce four numpy arrays to represent the shape and initial values of the four state variables:

```python
u_init = np.full([self.height, self.width], self.min_v, dtype=np.float32)
v_init = np.full([self.height, self.width], 1.0, dtype=np.float32)
w_init = np.full([self.height, self.width], 1.0, dtype=np.float32)
s_init = np.full([self.height, self.width], 0.0, dtype=np.float32)
```

Then, we define four corresponding tensorFlow variables and link them to the
numpy arrays:

```python
U  = tf.Variable(u_init, name='U')
V  = tf.Variable(v_init, name='V')
W  = tf.Variable(w_init, name='W')
S  = tf.Variable(s_init, name='S')
```

The TensorFlow graph to calculate the new values of the state variables after one time step of the explicit Euler method is generated by calling **Fenton4vSimple.solve()** (which in turns calls **Fenton4vSimple.differentiate()**) from  **Fenton4vSimple.define()** as:

```python
state = [U, V, W, S]
U1, V1, W1, S1 = self.solve(state)
```

Finally, we need to copy back the new values of the state variables. As mentioned above, TensorFlow graphs are generally acyclic. One exception to this rule is **tf.assign** operation that allows such copy backs. We have:

```python
self._ode_op = tf.group(
    U.assign(U1),
    V.assign(V1),
    W.assign(W1),
    S.assign(S1)
    )
```

In order to run a graph, it should be connected to a **tf.Session** that provides a context for execution. This is done in **Fenton4vSimple.run()**, which after creating a session and initializes the graph, runs the graph for the desired number of steps and send the transmembrane voltage array to the screen every 1 ms for visualization.

We can run the model as

```
python3 fenton_simple.py
```

On my desktop computer (1.7 GHz quad-code running Ubuntu 14.04 with an NVidia GTX-1080 GPU), **fenton_simple.py** takes ~11 seconds to run one second of simulation over a 512 x 512 medium (assuming no real-time screen update). This is actually not good! A hand optimized CUDA code runs the same thing in less than a second. In the following sections, we describe optimization tricks that improve the timing by a factor of 3-4.

# <a name='the-root-cause-of-slowness'></a> The Root Cause of Slowness

Before describing the solutions, let's first see if we can find the root cause of the sub-optimal performance of the simple model.

TensorFlow has a powerful [profiler](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/README.md). We profile one time step in **Fenton4vSimple.run()** and save the results as *timeline_simple.json*, which can be visualized in chrome browser by typing *chrome://trace* in the address bar and load the file. The result is

![timeline_simple.json](timeline_simple.png)

Based on the horizontal time bar on the top of the figure, one time step takes a little over $1,200\,\mu\text{s}$. Since we use a $\Delta{t} = 0.1\,\text{ms}$, one second of simulation requires 10000 iterations or ~12 sec.

The main point of the profile is the presence of numerous CUDA kernel launches (each green or purple block signifies a kernel launch). In the baseline TensorFlow, kernels correspond to basic operations like *Add*, *Sub*, *Mul*, and *Conv*. Each kernel launch imposes a timing overhead (the white gaps between the blocks). Even worse is the fact that each kernel needs to read its inputs from and write its outputs to the global GPU memory. Considering that the global memory is the primary bottleneck of the GPU applications, all these reading and writing intermediate results severely degrades the performance.

Fortunately, the TensorFlow developers have recognized this problem and have devise a solution, which is discussed in the next section.

# <a name='just-in-time-jit-compilation-to-rescue-'></a> Just-In-Time (JIT) Compilation to Rescue!

TensorFlow has the very exciting capability of using [JIT compilation](https://www.tensorflow.org/performance/xla/jit) to combine multiple atomic CUDA kernels into one large kernel. JIT compilation mitigates the performance penalty of having many smaller kernels. However, it is still not available in the stock versions of TensorFlow (at least not in version 1.7). To enable it, you need to [compile](https://www.tensorflow.org/install/install_sources) TensorFlow from source.

**fenton_jit.py** is a modified version of **fenton_simple.py**. The main change is in **fenton_simple.solve()**, where the Euler integration steps are moved inside a *jit_scope*.

```python
scope = tf.contrib.compiler.jit.experimental_jit_scope()

with scope:
    dU, dV, dW, dS = self.differentiate(U, V, W, S)
    U1 = U0 + self.dt * dU + self.diff * self.dt * laplace(U0)
    V1 = V + self.dt * dV
    W1 = W + self.dt * dW
    S1 = S + self.dt * dS
```

Just with this change, the model runs more than twice faster (~4.5 seconds per second of simulation). Now, the profile becomes

![timeline_jit.json](timeline_jit.png)

One time step takes a little over $500\,\mu\text{s}$. Also note there are only few blocks. The blue blocks are the fused kernels generated by the JIT compiler and they contain most of the logic of the program.

# <a name='other-optimization-tricks'></a> Other Optimization Tricks

JIT compilation is the main, but not the sole, optimization trick to improve the performance. In this section, we describe few other improvements that together raise the performance of the *4v* solver by another factor of 1.5-2.

The final version of the *4v* solver is **fenton.py**. Note that we have re-factored the code into two files. The model-independent codes are collected in a *virtual* base class **Ionic** (in **ionic.py**). The model-specific code is structured as a class **Fenton4v** (defined in **fenton.py**) that extends **Ionic**. This version of the *4v* solver takes 2.8 seconds per second of simulation.

## <a name='laplacian'></a> Laplacian

Up to this point, we have used the function **laplace()** copied from the PDE [example](https://www.tensorflow.org/tutorials/pdes) of the TensorFlow website. The actual calculation of the Laplacian is done by calling **tf.nn.depthwise_conv2d()**. However, this function is designed with convolutional neural networks in mind and is not the best fit for our application. Instead, a direct calculation of Laplacian with hard-coded coefficients, as is done in **Ionic.laplace()**, improves the performance:

```python
l = (X[:-2,1:-1] + X[2:,1:-1] + X[1:-1,:-2] + X[1:-1,2:] +
     0.5 * (X[:-2,:-2] + X[2:,:-2] + X[:-2,2:] + X[2:,2:]) -
     6 * X[1:-1,1:-1])
```

Here X stands for the transmembrane potential. Note that **Ionic.laplace()** also deals with phase fields, as discussed below.

## <a name='graph-unrolling'></a> Graph Unrolling

Graph unrolling is a spacial case of [loop unrolling](https://en.wikipedia.org/wiki/Loop_unrolling). A simple example of loop unrolling in C is to replace the following *for* loop

```C
for(i = 0; i < 1000; i++) {
    sum += x[i];
}
```

with a code like this

```C
for(i = 0; i < 1000; i+=4) {
    sum += x[i];
    sum += x[i+1];
    sum += x[i+2];
    sum += x[i+3];    
}
```

The goal is to increase the speed by reducing the number of times the for loop logic is checked and applied (Disclaimer: don't use this example! Modern compilers are much better in generating optimized code than most programmers).

Graph unrolling is the application of this idea to TensorFlow graphs. In our *4v* solvers, each time step is 0.1 ms, but the screen is updated every 1 ms. We can combine 10 time steps into one, as is done in **fenton.define()**:

```python
states = [[U, V, W, S]]
for i in range(10):
    states.append(self.solve(states[-1]))
U1, V1, W1, S1 = states[-1]
```

Currently, this trick improves the performance by allowing the TensorFlow optimizer to prune the model graph and to remove some operations. For example, the **tf.pad** operation in **enforce_boundary()** is removed from between the fused kernels. What it does not yet do is to generate a large fused kernel by combining all the individual kernels together. However, I'm not aware of any theoretical reason why this cannot be done. Hopefully, future versions of the TensorFlow JIT compiler become smart enough to handles this situation.

# <a name='the-beeler-reuter-ionic-model'></a> The Beeler-Reuter Ionic Model

We have chosen the [Beeler-Reuter ventricular myocyte model](https://www.ncbi.nlm.nih.gov/pubmed/?term=874889) as our second example. The model can be found in **BeelerReuter.solve()** in the file **br.py**.

The Beeler-Reuter model has eight variables: the transmembrane potential ($v$), sodium-channel activation and inactivation gates ($m$ and $h$, similar to the Hodgkin-Huxley model), with an additional slow inactivation gate ($j$), calcium-channel activation and deactivations gates ($d$ and $f$), a time-dependent outward potassium current gate ($x_1$), and intracellular calcium concentration ($c$). There are four currents: a sodium current ($I_{Na}$), a calcium current ($i_{s}$), and two potassium currents, one time-dependent ($i_{x_1}$) and one time-independent ($i_{K_1}$).

There are eight equations for the time derivation of the state variables. For $v$, it is simply the reaction-diffusion equations

\[
  \partial v / \partial t = \nabla (D  \nabla v) - \frac {i_{Na} + i_{s} + i_{x_1} + i_{K_1}} {C_m}.
\]

For $c$,

\[ dc / dt = -10^{-7} i_s + 0.07 (10^{-7} - c)  .\]

Each of the six gating variables follow a classic Hodgkin-Huxley dynamics. Let $x$ be one of $m$, $h$, $j$, $d$, $f$, or $x_1$. We have

\[ dx / dt = (x_{\infty} - x) / \tau_x,\]

where

\[ x_{\infty} = \frac{\alpha_x(v)}{\alpha_x(v) + \beta_x(v)},\]

and

\[ \tau_x = \frac{1}{\alpha_x(v) + \beta_x(v)}.\]

In the code, $x_{\infty}$ and $\tau_x$ are calculated in **BeelerReuter.calc_inf_tau()**. All $\alpha$s and $\beta$s are derived using the same equation (of course with different values of $C$s),

\[ \alpha = \frac{C_1 e^{C_2(v + C_3)} + C_4(v + C_5) } { e^{C_6(v + C_7) }}.\]

The values of $C$s are found in the file **br.py** as an array **ab_coef** and are used to calculate $\alpha$s and $\beta$s in **BeelerReuter.calc_alpha_beta_np()**. These values are calculated by numpy during the graph definition phase and are considered constants in the TensorFlow graph. **Ionic.rush_larsen()** updates the gating variables (see Rush-Larsen method below) based on the values of $\alpha$s and $\beta$s.

After updating the gating variables, they are used to find the spontaneous ionic currents. We have

\[ i_{Na} = (g_{Na} m^3 h j + g_{NaC})(v - E_{Na}) ,\]

where $g_{Na}$ is the sodium channel conductance, $g_{NaC}$ is the sodium/calcium exchanger conductance, and $E_{Na}$ is sodium reversal potential. Similarly, for the calcium channel,

\[ i_{s} = g_{s} df(v - E_{s}(c)) ,\]

where $g_{s}$ is the calcium current conductance, and $E_{s}(c) = -82.3 - 13.0287 \ln (c)$ is the calcium reversal potential explicitly dependent on the intracellular calcium concentration.

To complete the model, we need to calculate the two potassium currents,

\[ i_{K_1} =
        g_{K_1}.\frac{e^{0.04(v+85)}-1}{e^{0.08(v+53)} + e^{0.04(v+53)}} +
        h_{K_1}.\frac{0.2(v+23)}{1 - e^{-0.04(v+23)}},    
\]

and,

\[i_{x_1} = g_{x_1} . x_1 . \frac{e^{0.04(v+77)}-1}{e^{0.04(v+35)}}, \]

where $g_{K_1}$, $h_{K_1}$, and $g_{x_1}$ are channel conductance. Note that we have used an equivalent form of the last two equations by factoring out the exponentials using an accessory variable $k = \exp(0.04v)$.

We can run the baseline *Beeler-Reuter* solver as

```
python3 br.py
```

Note that the code already employs most of the optimization steps discuss above (JIT compilation, graph unrolling...). On my computer, this model takes 8.5 seconds per second of simulation compared to 2.8 seconds for the *4v* model. Considering twice the number of variables and more complicated formulas, a factor of 3 slow down is reasonable.

But we are not done yet! There are still some optimizations tricks that can improve the performance even further. We will discuss three such techniques: the Rush-Larsen method, using the Chebyshev polynomials and multi-rate integration. These three methods are already coded in **br.py**, but with the exception of the Rush-Larsen, are not active at baseline. You can activate them by editing **br.py** (in the `if __name__ == '__main__':` section on the bottom of the file) and change `cheby` to true to activate the Chebyshev polynomials or `skip` to true to activate the multi-rate integration. We can achieve a factor of 2-2.5 speed up by activating both:


|                   | cheby == False | cheby == True  |
| ----------------- |----------------| ---------------|
| **skip == False** | 8.5 s          |   5.1 s        |
| **skip == True**  | 5.1 s          |   3.9 s        |


## <a name='the-rush-larsen-method'></a> The Rush-Larsen Method

Most cardiac electrophysiology ionic ODEs are *stiff* and require a small integration time step (usually a $\Delta{t}$ of 0.01-0.1 millisecond) to have a stable numerical solution. This small time step is necessary to capture the fast changing dynamics at the time of the action potential upstrokes; but, it is a waste of computational power in other phases of action potentials.

The [Rush-Larsen](https://ieeexplore.ieee.org/document/4122859/) method replaces the explicit Euler integration for the gating variables, i.e., $x(t+\Delta{t}) = x(t) + \Delta{t}\,x'$, with direct integration. The starting point is Eq. XXX, which describes the dynamic of the gating variables in the Beeler-Reuter and other descendants of the Hodgkin-Huxley model, and is reproduced here by making the dependence on $v$ explicit,

\[ dx / dt = \left(x_{\infty}(v) - x \right) / \tau_x(v).\]

If we assume that $x$ changes on a faster time-scale than $x_{\infty}$ and $\tau_x$ (an assumption generally true for the the Hodgkin-Huxley type models), we can considered $x_{\infty}$ and $\tau_x$ constant for the duration of $\Delta{t}$ and perform a direct integration of Eq. XXX to obtain

\[ x(t + \Delta{t}) = x_{\infty} - (x_{\infty} - x)\,e^{-\Delta{t}/\tau_x}. \]

This equation is the basis of the Rush-Larsen method and is coded essentially directly in **Ionin.rush_larsen()**

```python
def rush_larsen(self, g, g_inf, g_tau, dt, name=None):
    return tf.clip_by_value(g_inf - (g_inf - g) * tf.exp(-dt/g_tau), 0.0,
                            1.0, name=name)
```

It allows for the integration of the Beeler-Reuter model with a time step of 0.1 ms, instead of 0.01 ms without the Rush-Larsen trick. This is a factor 10 improvement! It should be noted that the Rush-Larsen method is a general and standard method in solving cardiac electrophysiology ionic models and is not exclusive to TensorFlow or even GPU.

## <a name='using-the-chebyshev-polynomials'></a> Using the Chebyshev Polynomials

Lookup tables are commonly used to increase the speed of solving systems of ODEs. This method is particularly suitable for solving cardiac ionic models, because the right side of many of the equations describing the dynamics of the models depend only of the transmembrane potential. For example, for the Beeler-Reuter model, the time derivatives of the six gating variables are defined by $\alpha(v)$ and $\beta(v)$. It is reasonable to calculate $\alpha$ and $\beta$ for different values of $v$ ahead of time and then just perform a table lookup (with some forms of interpolation) to find the desired values while running the model.

It is reasonably straightforward to implement lookup tables running on a CPU in programming languages like C, C++ and fortran (the usual high-performance computing choices); notwithstanding subtle issues with cache coherence. The situation is more interesting when dealing with GPUs. Lookup tables are located in the global memory. Considering that global memory access is the primary bottleneck of most GPU applications, it is commonly more efficient to recalculate the intermediate parameters each time than doing a table lookup. However, GPU are designed with graphic acceleration in mind and lookup tables are ubiquitous in graphic applications. Therefore, they have specialized hardware to accelerate lookups in the form of *texture memory*. Unfortunately, at least to best of my knowledge, TensorFlow currently does not have an operation to use the texture memory and direct table lookups are exceedingly slow.

It is in this setting that we decided to use [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) to encode the lookup values. Chebyshev polynomials are a valuable tool in numerical analysis and numerical approximation. Here, we only present the bare minimum needed to advance our discussion and cannot do justice to the large and interesting mathematics of the Chebyshev polynomials.

A Chebyshev polynomial of the first kind $T_i(x)$ is a degree $i$ polynomial in variable $x$, defined recursively as

\[ T_0(x) = 1, \]
\[ T_1(x) = x, \]
\[ T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x) \]

Few examples,

\[ T_2(x) = 2x^2 - 1, \]
\[ T_3(x) = 4x^3 - 3x, \]
\[ T_4(x) = 8x^4 - 8x^2 + 1, \]
\[ T_{10}(x) = 512x^{10} - 1280x^8 + 1120x^6 -400x^4 + 50x^2 - 1. \]

Chebyshev polynomial are orthogonal on the interval [-1,1] with respect to the weighting factor $(1-x^2)^{-1/2}$,

\[
    \int_{-1}^{1} T_n(x)T_m(x)\,\frac{dx}{\sqrt{1-x^2}} =
        \begin{cases}
        0       & \quad n \neq m    \\
        \pi     & \quad n = m = 0   \\
        \pi/2   & \quad n = m \neq 0
        \end{cases}
\]

The Chebyshev polynomials can act as a basis set in a similar fashion to the trigonometrical functions in Fourier analysis. Let $f(x)$ be a continuous and piecewise smooth function defined on [1,-1]. It can be expressed as

\[
    f(x) = \sum_{i=0}^{\infty} c_i T_i(x).
\]

For the Beeler-Reuter solver, we employ the Chebyshev polynomials to approximate $g_{\infty}$ and $\tau_g$ for the six gating variables. Let's explore the straightforward way to do this (we will later modify this). We use order 8 polynomials in the solver, but for the sake of simplicity, we use order 4 here to demonstrate the method.

The transmembrane potential ranges between -90 to +30 mV in the Beeler-Reuter model. We remap this range to [-1,1] in **BeelerReuter.update_gates_with_cheby** as

```python
x = (V0 - 0.5*(self.max_v+self.min_v)) / (0.5*(self.max_v-self.min_v))
```

Note than $x$ is a 2D TensorFlow tensor and not a scalar. Next, we can calculate the corresponding Chebyshev polynomials using the recurrence relationship,

```python
T0 = 1.0
T1 = x
T2 = 2*x*T1 - T0
T3 = 2*x*T2 - T1
T4 = 2*x*T3 - T2
```

Assume the goal is to approximate $m_{\infty}$. We calculate $m_{\infty}$ on 5001 points over [-1,1]. Let's call the result $y$. We estimate the best fit Chebyshev coefficients as

```python
c = np.polynomial.chebyshev.Chebyshev.fit(x, y, deg).coef
```

Note that $c$ is calculate by numpy during the setup stage and the resulting coefficients are considered constants as far as TensorFlow is concerned. Now, we can find the value of $m_{\infty}$ at each point as

```python
m_inf = c[0] + c[1]*T1 + c[2]*T2 + c[3]+T3 + c[4]*T4
```

The figure below shows the actual (orange) and approximated (blue) values for $m_{\infty}$:

![](m_inf.png)

We have implemented this method (for order 8 polynomials). It works! But the performance is not great and it confers only a modest performance benefit. I believe the problem arises because the JIT compiler has difficulty optimizing the $T_i$ calculations. The solution is to use instead $S_i$, the leading term of $T_i$, where $S_i$ depends directly only on $S_{i-1}$ and not $S_{i-2}$. This seems to help the optimizer avoid unnecessary copying to and back from the global memory. Therefore,

```python
S0 = 1.0
S1 = x
S2 = 2*x*S1
S3 = 2*x*S2
S4 = 2*x*S3
```

$T_i$s are expanded in term of $S_i$s as

\[ T_0 = S_0 = 1\]
\[ T_1 = S_1 \]
\[ T_2 = S_2 - 1 \]
\[ T_3 = S_3  - 3S_1\]
\[ T_4 = S_4 - 4S_2 + 1\]

Therefore,

```python
m_inf = (c[0]-c[2]+c[4]) + (c[1]-3*c[3])*S1 + (c[2]-4*c[4])*S2 + c[3]+S3 + c[4]*S4
```

In the code, **BeelerReuter.calc_chebyshev_leading** and **BeelerReuter.expand_chebyshev** do these calculation for arbitrary order polynomials. The resulting code runs 1.5-2X faster than the baseline code.


## <a name='multi-rate-integration'></a> Multi-rate Integration
