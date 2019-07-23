
# DynaFu ICP Math
## Differentiating and Linearising Rt matrices

In dynafu, the warp function looks like the following for each node $i$:


$
\begin{equation*}
f_i(x_i, V_g) = T_{x_i} * V_g = R(x_i) * V_g + t(x_i)
\end{equation*}
$

where ${x_i}$ are the transformation parameters for node $i$ and the rotation is performed around the corresponding node (and not a global reference)

For linearising a transform around the parameters $\mathbf{x}$, we need to find the derivative

$
\begin{equation*}
\displaystyle
\frac{\partial f_i(\mathbf{x} \circ \epsilon,   V_g)}{\partial \epsilon} |_{\epsilon = 0}
\end{equation*}
$

We calculate this as follows:

$
\begin{equation*}
f_i(\mathbf{x} \circ \epsilon, V_g) = f_i(\epsilon, V) = T_{inc} * V
\end{equation*}
$ where $V = f_i(\mathbf{x}, V_g)$ and $T_{inc}$ is the infinitesimal transform with parameters $\epsilon$

According to Lie algebra, each Rt matrix can be represented as $A = e^\xi$ where $\xi$ are the transform parameters. Therefore,


$
\begin{equation*}
f_i(\mathbf{x}, V_g) = e^\xi V
\end{equation*}
$

Therefore,

$
\begin{equation*}
\displaystyle
\frac{\partial f_i(\mathbf{x} \circ \xi,   V_g)}{\partial \xi} |_{\xi = 0} =
\frac{\partial e^\xi V}{\partial \xi} |_{\xi=0} = 
\begin{pmatrix} -[V]_{\times} & I_{3x3} \end{pmatrix}_{3 \times 6}
\end{equation*}
$

Let us denote $\begin{pmatrix} -[V]_{\times} & I_{3x3} \end{pmatrix}$ as $G(V)$ from now on.

This result is mentioned in [this manifold optimisation tutorial](http://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf) (equation 10.23).

With this result, we can now linearise our transformation around $\mathbf{x}$:

$
\begin{equation*}
f_i(x_i, V_g) = G(V) * \epsilon + V
\end{equation*}
$


I suppose the following is an equivalent excerpt from the dynafu paper (Section about efficient optimisation) that mentions this way of calculating derivatives:
> We formulate compositional updates $\hat x$ through the exponential map with a per-node twist $ξ_i ∈ se(3)$, requiring 6 variables per node transform, and perform linearisation  around $ξ_i=  0$. 

As a side note, the derivative $\large \frac{\partial e^\xi}{\partial \xi}|_{\xi=0}$ is called the tangent (esentially the derivative) to the SE(3) manifold (the space in which Rt matrix $T_{inc}$ exists) at identity ($\xi = 0$)

## Estimating Warp Field Parameters
The total energy to be minimised is 

$
E = E_{data} + \lambda E_{reg}
$

#### Data term rearrangement 
$
\displaystyle
E_{data} = \sum_{u \in \Omega} \rho_{Tukey}(N_g^T (V_g - T_u^{-1}\cdot V_c))
$

The quadcopter paper tells us that the following expression has the same minimiser, so we can use this instead:

$
\displaystyle
E_{data} = \sum_{u \in \Omega} w_{Tukey}(r_u) \cdot (r_u)^2
$

where $w_{Tukey}(x) = \rho'(x)/x$ which behaves like a constant term and $r_u = N_g^T (V_g - T_u^{-1}\cdot V_c)$

#### Regularisation term rearrangement
$
\begin{equation}
\displaystyle
E_{reg} = \sum_{i = 0}^n \sum_{j \in \varepsilon(i)} \alpha_{ij} \rho_{Huber} (T_{i}V_g^j - T_{j}V_g^j)
\end{equation}
$

This needs to be changed to the form of weighted least squares to be useful. So incorporate the same rearrangement as the data term and sum over edges instead:

$
\begin{equation}
\displaystyle
E_{reg} = \sum_{e \in E} w_{Huber}(r_e) (r_e)^2
\end{equation}
$

Here $E$ is the set of the directed edges in the regularisation graph between all nodes from current level and the next coarser level. And $w_{Huber}(x) = \alpha_x \rho'(x)/x$

#### Obtaining normal equation

Therefore to solve an iteration, we equate the derivative with 0

$
\begin{equation*}
\large
\frac{\partial E_{data}}{\partial \xi} + \lambda \frac{\partial E_{reg}}{\partial \xi} = 0
\end{equation*}
$

which gives us

$
\begin{equation*}
J_d^T W_d(r_d + J_d\mathbf{\hat x}) + \lambda J_r^T W_r (r_r + J_r\mathbf{\hat x}) = 0
\end{equation*}
$

$
(J_d^T W_d J_d + \lambda J_r^T W_r J_r)\mathbf{\hat x} = -(J_d^T W_d r_d + \lambda J_r^T W_r r_r)
$

Here $W_d$ and $W_r$ are the weight matrices as described in quadcopter paper. However for $W_r, \alpha$ is also incorporated in this matrix

### Calculating Data Term Jacobian ($J_d$) 

Each entry in $J_d$ is as follows for node paramter $x_j$ for each node $j$:

$
\begin{equation*}
\displaystyle
(J_d)_{uj} = \frac{\partial r_u}{\partial x_j} = -N_g^T \frac{\partial T_u^{-1} V_c}{\partial x_j}
\end{equation*}
$


From chain rule

$
\begin{equation*}
\displaystyle
\frac{\partial T_u^{-1} V_c}{\partial x_j} = \frac{\partial T_u^{-1} V_c}{\partial T_u} \frac{\partial T_u}{\partial x_j}
\end{equation*}
$


Equation 7.20 in the manifold optimisation tutorial tells us how to calculate the first term 

$
\begin{equation*}
\displaystyle
\frac{\partial T_u^{-1} V_c}{\partial T_u} = 
\begin{pmatrix}
I_3 \otimes (V_c - t_u)^T & R_u
\end{pmatrix}
\end{equation*}
$

where $\otimes$ denotes the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) and $R_u, t_u$ are the rotation matrix and translation vector of $T_u$ respectively

$
\begin{equation*}
\displaystyle
\frac{\partial T_u}{\partial x_j} = \frac{w_j}{\sum_{k \in N(V_u)} w_k} \frac{\partial T_j}{\partial x_j}
\end{equation*}
$

As per equation 10.15 of the tutorial,

$
\begin{equation*}
\displaystyle
\frac{\partial T_j}{\partial x_j} = 
\begin{pmatrix}
0_{3\times3} & - [R_j^{c1}]_\times \\
0_{3\times3} & - [R_j^{c2}]_\times \\
0_{3\times3} & - [R_j^{c3}]_\times \\
I_{3\times3} & - [t_j]_\times
\end{pmatrix}
\end{equation*}
$

where $R_j^{c1}$ is the first column of the rotation matrix of node $j$, $R_j^{c2}$ is the second column and so on.

Combining these results, we get

$
\begin{equation*}
\displaystyle
\frac{\partial T_u^{-1} V_c}{\partial x_j} =
\frac{w_j}{\sum_{k \in N(V_u)} w_k}
\begin{pmatrix}
I_3 \otimes (V_c - t_u)^T & R_u
\end{pmatrix}
\begin{pmatrix}
0_{3\times3} & - [R_j^{c1}]_\times \\
0_{3\times3} & - [R_j^{c2}]_\times \\
0_{3\times3} & - [R_j^{c3}]_\times \\
I_{3\times3} & - [t_j]_\times
\end{pmatrix}
\end{equation*}
$

$
=
\frac{w_j}{\sum_{k \in N(V_u)} w_k}
\left(
\begin{pmatrix}
R_u & R_j^T [V_c]_\times
\end{pmatrix}
- \left(
\begin{array}{c|c}
0_{3\times3} & R_j^T[t_u]_\times - R_u^T[t_j]_\times
\end{array}
\right)
\right)
$

It may be noted that this result is obtained through a variation of equation 10.25 of the manifold optimisation tutorial.

Therefore,
$
\begin{equation*}
(J_d)_{uj} = \frac{w_j}{\sum_{k \in N(V_u)} w_k}
-\left(
N_g^T \begin{pmatrix}
R_u & R_j^T [V_c]_\times
\end{pmatrix}
- 
N_g^T \left(
\begin{array}{c|c}
0_{3\times3} & R_j^T[t_u]_\times - R_u^T[t_j]_\times
\end{array}
\right)
\right)
\end{equation*}
$

We can rewrite 
$
N_g^T \begin{pmatrix}
R_u & R_j^T [V_c]_\times
\end{pmatrix}
$ as

$
\begin{equation*}
\left(
\begin{pmatrix}
R_u^T \\
(R_j^T [V_c]_\times)^T
\end{pmatrix}
N_g
\right)^T
\end{equation*}
$$
=
\begin{equation*}
\begin{pmatrix}
R_u^T N_g \\
(R_j^T [V_c]_\times)^T N_g
\end{pmatrix}^T
\end{equation*}
$$
=
\begin{equation*}
\begin{pmatrix}
(R_u^T N_g)^T &
N_g^T (R_j^T [V_c]_\times)
\end{pmatrix}
\end{equation*}
$

We know that all rotation matrices are skew symmetric, which means we can replace $R^T$ by $-R$ everywhere.
So now we have:

$
\begin{equation}
(J_d)_{uj} = \frac{w_j}{\sum_{k \in N(V_u)} w_k}
\left(
\begin{pmatrix}
(R_u N_g)^T &
N_g^T (R_j [V_c]_\times)
\end{pmatrix}
-
N_g^T \left(
\begin{array}{c|c}
0_{3\times3} & R_j[t_u]_\times - R_u[t_j]_\times
\end{array}
\right)
\right)
\end{equation}
$

But this expression is only valid if $j \in N(V_u)$, because otherwise $frac{\partial T_u}{\partial x_j} = 0$ in the chain rule, making the entire term 0.

The final expression of the data term Jacobian is:

$
\begin{equation}
(J_d)_{uj} = 
\begin{cases}
\frac{w_j}{\sum_{k \in N(V_u)} w_k}
\left(
\begin{pmatrix}
(R_u N_g)^T &
N_g^T (R_j [V_c]_\times)
\end{pmatrix}
-
N_g^T \left(
\begin{array}{c|c}
0_{3\times3} & R_j[t_u]_\times - R_u[t_j]_\times
\end{array}
\right)
\right) & \text{if  } j \in N(V_u) \\
0 & \text{otherwise}
\end{cases}
\end{equation}
$


### Calculating Regularisation Term Jacobian ($J_r$)

Each row in $J_r$ corresponds to derivative to summand for each edge $e$ and column $k$ corresponds to node $k$ with respect to which the derivative is calculated.

$
\begin{equation*}
\displaystyle
(J_r)_{ek} = 
\frac{\partial ( T_iV_g^j - T_jV_g^j)}{\partial x_k}
=
\begin{cases}
\begin{pmatrix} -[T_iV_g^j] & I_{3x3} \end{pmatrix} & \text {if   }  i = k \\
0 & \text {otherwise}
\end{cases}
\end{equation*}
$

Please note that $T_j$ is constant in all the cases since the corresponding node lies in the next level and there is no $k$ such that $k=j$
