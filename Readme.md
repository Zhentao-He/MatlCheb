# Chebyshev pseudo-spectral method
Based on
[1]《Spectral Methods in MATLAB》
[2]《Matlab微分方程高效解法：谱方法原理与实现》
[3]《Numerical Recipes》
## Compute derivatives numerically [1]
Given a grid function $\{f(x_i)=:f_i\}$ defined on $N+1$ *Chebyshev-Gauss-Lobatto* points: 
$$
x_i = \cos(\frac{i\pi}{N}), i=0,...,N,
$$
we use the derivative of its Lagrange interpolation function $p(x)$ to approximate its derivative $f^\prime(x_i)\simeq p^\prime(x_i)$.

$$
p(x)=\sum_{i=0}^{N}f_i p_i(x)
$$
where
$$
p_i(x) = \prod_{{j=0},{j\neq i} }^N
\frac{x-x_j}{x_i-x_j}
$$
are called *cardinal functions*.

$$
p^\prime(x_j) = \sum_{i=0}^{N}f_i p^\prime_i(x_j)
= \sum_{i=0}^{N} D_{ji}f_i
$$
$D_{ji}$ is the so-called Chebyshev differentiation matrix (Exercises 6.1) and can given by `cheb.m` (P54).

$\textbf{Comments}$
> Q: Why Spectral methods?
> A.: Spectral methods work well for **smooth** solutions (*Spectral accuracy*, see `Program12.m`).

> Q: Why Chebyshev?
> A: 
> 1. Avoiding Runge Phenomenon.
> 2. No requirement on boundary conditions

> Q: Why pseudo-spectral?
> A: ... (see [3])
> 
$\textbf{More Problems associated with 6.1}$
Differentiation Matrices constructed from the cardinal function
 >How to prove 
 $$
 \sum_{j=0}^{N}D_{ij}=0
 $$
 i.e.
 $$
 0=\sum_{j\neq i}
 (x_i - x_j)
 \prod_{\substack{k\neq i,j \\ l\neq k}}
 (x_k - x_l).
$$

> When $x_{j}=\cos{(j\pi/N)}$, how to compute $a_i/a_j$.

> What basis fuction $p_j(x)$ can give 
$$ 
p^\prime_j(x_i) = \frac {c_i} {c_j} \frac{(-)^{i+j}}{x_i-x_j}
$$


# Applications

## Cauchy Problems of PDEs
Generally, we use spectral method along spacial direction, then spacial derivation operations are transformed into matrix manipulations, where the origin PDE is converted to a set of ODEs along time direction for $\{f_j(T):=f(T,x_j)\}$.

$$
\begin{equation}
    \frac{\mathrm{d}f_j}{\mathrm{d}T}=F(\{f_j\}).
\end{equation}
$$

Given the initial data, one can integrate these ODEs along time direction (`RK4` usually).

$\textbf{Check of spectral accuracy}$

$$
p(x)=\sum_{i=0}^{N}f_i p_i(x)
\quad
\text{(Physical Space)}
$$

$$
p(x)=\sum_{n=0}^{N}a_n T_n(x)
\quad
\text{(Spectral Space)}
$$

$T_n(x)$ are a complete set of basis functions, which, in our context, is $n$-th first kind Chebyshev polynomial
$$
T_n(x)=\cos(n\arccos x),
$$

The $N+1$ Chebyshev-Gauss-Lobatto points are exactly extreme points of the Chebyshev polynomial $T_{N}$.


Transformation between the physical coefficients $f_i$ and spectral coefficients $a_n$ can be achieved by *Fast Cosine Transform* efficiently. (https://zhuanlan.zhihu.com/p/660791156)

Spectral convergence implies that the magnitudes of the coefficients $a_n$ decay exponentially in $n$.

## Boundary Value Problems of ODEs [2]

### See 5.2 in [2]

### Multi-domain spectral methods
See [2202.01794]
$$
[\alpha_2\partial_\sigma^2+\alpha_1\partial_\sigma+\alpha_0]\phi_{lm}=\tilde{\kappa}_{lm}\delta(\sigma-\sigma_p)
$$

$$
\lim_{\epsilon\rightarrow0} 
\left.
\partial_\sigma\tilde\phi
\right|^{\sigma_p+\epsilon}_{\sigma_p-\epsilon}
=\left.
\frac{\tilde{\kappa}_{lm}}{\alpha_2}
\right|_{\sigma_p}
$$

|                |        |        | `j=N1+1` |        |        |                     |
| :-------------: | ------ | ------ | :--------: | ------ | ------ | ------------------- |
|    `i = 1`    | `A1` | `A1` |   `A1`   |        |        | $\sigma=1$        |
|       ...       | `A1` | `A1` |   `A1`   |        |        | `Domain1`         |
|  `i = N1+1`  | `J`  | `J`  |   `J`   | `J`  | `J`  | $\sigma=\sigma_p$ |
|       ...       |        |        |   `A2`   | `A2` | `A2` | `Domain0`         |
| `i = N1+N2+1` |        |        |   `A2`   | `A2` | `A2` | $\sigma=0$   


In this case, the numerical error is estimated by fixing a reference solution obtained with a given high accuracy.

## Eigenvalues Problems (Quasi-Normal Modes)

