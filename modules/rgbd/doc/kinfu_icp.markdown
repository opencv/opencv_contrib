# ICP point-to-plane odometry algorithm {#kinfu_icp}

This article describes an ICP algorithm used in depth fusion pipelines such as KinectFusion.

The goal of ICP is to align two point clouds, the old one (the existing points and normals in 3D model) and new one (new points and normals, what we want to integrate to the exising model). ICP returns rotation+translation transform between these two point clouds.

The Iterative Closest Point (ICP) minimizes the objective function which is the Point to Plane Distance (PPD) between the corresponding points in two point clouds:

<img src="https://render.githubusercontent.com/render/math?math=E=\sum_{i}\left\|ppd(p_{i}, q_{i}, n_{i})\right\|_{2}\rightarrow0">

### What is ppd(p, q, n)?

Specifically, for each corresponding points ***P*** and ***Q***, it is the distance from the point ***P*** to the plane determined by the point ***Q*** and the normal ***N*** located in the point ***Q***.
Two points ***P*** and ***Q*** are considered correspondent if given current camera pose they are projected in the same pixel.

***p*** - i'th point in the new point cloud

***q*** - i'th point in the old point cloud

***n*** - normal in the point ***q*** in the old point cloud

Therefore, ***ppd(...)*** can be expressed as the dot product of (difference between ***p*** and ***q***) and (***n***):

<img src="https://render.githubusercontent.com/render/math?math=dot(T_{p2q}(p)-q, n)=dot((R\cdot p %2b t)-q,n)=[(R\cdot p %2b t)-q]^{T}\cdot n">

***T(p)*** is a rigid transform of point ***p***:

<img src="https://render.githubusercontent.com/render/math?math=T_{p2q}(p) = (R \cdot  p %2B t)">

where ***R*** - rotation, ***t*** - translation.

***T*** is the transform we search by ICP, its purpose is to bring each point ***p*** closer to the corresponding point ***q*** in terms of point to plane distance.

### How to minimize objective function?

We use the Gauss-Newton method for the function minimization.

In Gauss-Newton method we do sequential steps by changing ***R*** and ***t*** in the direction of the function E decrease, i.e. in the direction of its gradient:

1. At each step we approximate the function ***E*** linearly as its current value plus Jacobian matrix multiplied by ***delta x*** which is concatenated ***delta R*** and ***delta t*** vectors.
2. We find ***delta R*** and ***delta t*** by solving the equation ***E_approx(delta_x) = 0***
3. We apply ***delta R*** and ***delta t*** to current Rt transform and proceed to next iteration

### How to linearize E?

Let's approximate it in infinitesimal neighborhood.

Here's a formula we're going to minimize by changing ***R*** and ***t***:

<img src="https://render.githubusercontent.com/render/math?math=E=\sum\left\|[(R\cdot p %2B t)-q]^{T}\cdot n\right\|_{2}">

While the point to plane distance is linear to both ***R*** and ***t***,  the rotation space is not linear by itself. You can see this in how ***R*** is generated from its rotation angles:

<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+R+%3D+R_%7Bz%7D%28%5Cgamma%29R_%7By%7D%28%5Cbeta+%29R_%7Bx%7D%28%5Calpha%29%3D%0A%5Cbegin%7Bbmatrix%7D%0Acos%28%5Cgamma%29+%26+-sin%28%5Cgamma%29+%26+0+%5C%5C%0Asin%28%5Cgamma%29+%26+cos%28%5Cgamma%29+%26+0%5C%5C%0A0+%26+0+%26+1%0A%5Cend%7Bbmatrix%7D%0A%5Cbegin%7Bbmatrix%7D%0Acos%28%5Cbeta%29+%26+0+%26+sin%28%5Cbeta%29%5C%5C%0A0+%26+1+%26+0%5C%5C%0A-sin%28%5Cbeta%29+%26+0+%26+cos%28%5Cbeta%29%0A%5Cend%7Bbmatrix%7D%0A%5Cbegin%7Bbmatrix%7D%0A1+%26+0+%26+0%5C%5C%0A0+%26+cos%28%5Calpha%29+%26+-sin%28%5Calpha%29%5C%5C%0A0+%26+sin%28%5Calpha%29+%26+cos%28%5Calpha%29%0A%5Cend%7Bbmatrix%7D%0A">

But since we have infinitesimal rotations, ***R*** can be approximated in the following form:

<img src="https://render.githubusercontent.com/render/math?math=R=I %2B Ad\theta">

where ***I*** - unit matrix, ***A*** - member of the three-dimensional special orthogonal group ***so(3)***.

By approaching all sin(t) and cos(t) terms to their limits where ***t --> 0*** we get the following representation:

<img src="https://render.githubusercontent.com/render/math?math=R = I %2B \begin{bmatrix}0 %26 -\gamma  %26 \beta \\ \gamma %26 0 %26 -\alpha \\ -\beta  %26 \alpha  %26 0 \end{bmatrix} = I %2B skew(\begin{bmatrix} \alpha %26 \beta %26 \gamma \end{bmatrix}^{T}) = I %2B skew(R_{shift}) ">

Substituting the approximation of ***R*** back into ***E*** expression, we get:

<img src="https://render.githubusercontent.com/render/math?math=E_{approx}=\sum\left\|[(I %2B skew(R_{shift})) \cdot  p %2B t - q]^{T}  \cdot n \right \|_{2} ">

<img src="https://render.githubusercontent.com/render/math?math=E_{approx} = \sum \left \| [I \cdot  p %2B skew(R_{shift}) \cdot  p %2B t - q]^{T}  \cdot n \right \|_{2} ">

<img src="https://render.githubusercontent.com/render/math?math=E_{approx} = \sum \left \| [skew(R_{shift}) \cdot  p %2B t %2B p- q]^{T}  \cdot n \right \|_{2} ">

Let's introduce a function f which approximates transform shift:

<img src="https://render.githubusercontent.com/render/math?math=f(x, p) = skew(R_{shift}) \cdot  p %2B t">

<img src="https://render.githubusercontent.com/render/math?math=E_{approx} = \sum \left \| [f(x, p) %2B p- q]^{T}  \cdot n \right \|_{2}">

### How to minimize _E_approx_?

***E_approx*** is minimal when its differential (i.e. derivative by argument increase) is zero, so let's find that differential:

<img src="https://render.githubusercontent.com/render/math?math=d(E_{approx}) = \sum_i d(\left \| ppd(T_{approx}(p_i), q_i, n_i) \right \|_2) = ">
<img src="https://render.githubusercontent.com/render/math?math=\sum_i d(ppd(T_{approx}(p_i), q_i, n_i)^2) =">
<img src="https://render.githubusercontent.com/render/math?math=\sum_i 2\cdot ppd(...)\cdot d(ppd(T_{approx}(p_i), q_i, n_i))">

Let's differentiate ***ppd***:

<img src="https://render.githubusercontent.com/render/math?math=d(ppd(T_{approx}(p_i), q_i, n_i)) = d([f(x, p_i) %2b p_i- q_i]^{T}  \cdot n_i) = df(x, p_i)^{T}  \cdot n_i = dx^T f'(x, p_i)^T \cdot n_i">

Here's what we get for all variables x_j from vector x:

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial E}{\partial x_{j}} = \sum [2 \cdot (f(x, p) %2B p - q)^{T} \cdot n] \cdot [f_{j}'(x, p)^{T} \cdot n] = 0">

<img src="https://render.githubusercontent.com/render/math?math=\sum [2 \cdot n^{T} \cdot (f(x, p) %2B p - q)] \cdot [n^{T} \cdot f{}'(x, p)] = 0 ">

Let new variable: <img src="https://render.githubusercontent.com/render/math?math=\triangle p = p - q">

<img src="https://render.githubusercontent.com/render/math?math=\sum [2 \cdot n^{T} \cdot (f(x, p) %2B \triangle p)] \cdot [n^{T} \cdot f{}'(x, p)] = 0">

<img src="https://render.githubusercontent.com/render/math?math=\sum [(f(x, p) %2B \triangle p)^{T} \cdot (n \cdot n^{T})] \cdot f{}'(x, p) = 0">

<img src="https://render.githubusercontent.com/render/math?math=\sum f{}'(x, p)^{T} \cdot [n \cdot n^{T}] \cdot [f(x, p) %2B \triangle p] = 0">

***f(x, p)*** can be represented as a matrix-vector multiplication. To prove that, we have to remember that <img src="https://render.githubusercontent.com/render/math?math=cross(a, b) = skew(a) \cdot b = skew(b)^{T} \cdot a"> :

<img src="https://render.githubusercontent.com/render/math?math=f(x, p) = skew(R_{shift}) \cdot  p %2B t_{shift} = skew(p)^T R_{shift} %2B t_{shift}">
<img src="https://render.githubusercontent.com/render/math?math=f(x, p) = \begin{bmatrix} skew(p)^{T} %26 I_{3\times 3}\end{bmatrix} \cdot \begin{bmatrix} \triangle R %26 \triangle t \end{bmatrix}^{T} = G(p) \cdot x">

***G(p)*** is introduced for simplification.

Since <img src="https://render.githubusercontent.com/render/math?math=d(f(x, p)) = G(p) \cdot dx = f'(x, p) \cdot dx"> we get <img src="https://render.githubusercontent.com/render/math?math=f'(x, p) = G(p)">.

<img src="https://render.githubusercontent.com/render/math?math=\sum f{}'(x, p)^{T} \cdot [n \cdot n^{T}] \cdot [f(x, p)] = \sum f{}'(x, p)^{T} \cdot [n \cdot n^{T}] \cdot [- \triangle p]">

<img src="https://render.githubusercontent.com/render/math?math=\sum G(p)^{T} \cdot [n \cdot n^{T}] \cdot [G(p) \cdot X] = \sum G(p)^{T} \cdot [n \cdot n^{T}] \cdot [- \triangle p]">

Let a new value:

<img src="https://render.githubusercontent.com/render/math?math=C = G(p)^{T} \cdot n">

<img src="https://render.githubusercontent.com/render/math?math=C^{T} = (G(p)^{T} \cdot n)^{T} = n^{T} \cdot G(p)">

Let's make a replacement:

<img src="https://render.githubusercontent.com/render/math?math=\sum C \cdot C^{T} \cdot X = \sum C \cdot n^{T} \cdot [- \triangle p]">

<img src="https://render.githubusercontent.com/render/math?math=\sum C\cdot C^{T}\cdot \begin{bmatrix} \triangle R\\ \triangle t \end{bmatrix} = \sum C \cdot n^{T} \cdot [- \triangle p]">

By solving this equation we get rigid transform shift for each Gauss-Newton iteration.

### How do we apply transform shift?

We generate rotation and translation matrix from the shift and then multiply the current pose matrix by the one we've got.

While the translational part of the shift contributes to the resulting matrix as-is, the rotational part is generated a bit trickier.
The rotation shift is converted from ***so(3)*** to ***SO(3)*** by exponentiation.
In fact, the 3-by-1 rshift vector represents rotation axis multiplied by the rotation angle. We use Rodrigues transform to get rotation matrix from that.
For more details, see [wiki page](https://en.wikipedia.org/wiki/3D_rotation_group).
