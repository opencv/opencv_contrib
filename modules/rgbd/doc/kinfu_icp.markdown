# ICP algorithm in Kinect Fusion {#kinfu_icp}

The Iterative closest point (ICP) function minimizes the PointToPlane Distance (PPD) between the corresponding points in two clouds of points and normals.
Specifically, it is the distance from the point ***P*** to the plane with the normal ***N*** in which the point ***Q*** located

The main equetion, which it needs to minimize:

<img src="https://render.githubusercontent.com/render/math?math=E=\sum\left\|ppd(p_{i},q_{i},n_{i})\right\|_{2}\rightarrow0">


Let's watch what is ***ppd(p,q,n)***

Firstly, we have two clouds of points, old (the existing points and normals in 3-D model) and new (new points and normals, what we want to integrate to the exising model)

***p*** - i'th point in the new cloud of points

***q*** - i'th point in the old cloud of points

***n*** - i'th normal in the old cloud of normals

***ppd(...)***- is the distance ergo its formula is the dot product of (difference between ***p*** and ***q***) and (***n***):

<img src="https://render.githubusercontent.com/render/math?math=dot(T_{p2q}(p)-q, n)=dot((R\cdot p+t)-q,n)=[(R\cdot p+t)-q]^{T}\cdot n">

***T(p)*** - rigid transform of ***p*** point, which brings it closer to the corresponding ***q*** point.

<img src="https://render.githubusercontent.com/render/math?math=T_{p2q}(p) = (R \cdot  p %2B t)">

Where ***R*** - rotation, ***t*** - translation.

We use the Gauss-Newton method for the minimization of function.
In the beginning, we will perform some mathematical operations:

<img src="https://render.githubusercontent.com/render/math?math=E=\sum\left\|[(R\cdot p %2B t)-q]^{T}\cdot n\right\|_{2}">

***R*** is rotation and its formula is complicated:

<img src="https://render.githubusercontent.com/render/math?math=R=R_{z}(\gamma)R_{y}(\beta)R_{x}(\alpha)=(\begin{bmatrix}cos(\gamma)%26-sin(\gamma)%26 0\\sin(\gamma)%26cos(\gamma)%26 0 \\ 0 %26 0 %26 1 \end{bmatrix} \begin{bmatrix}cos(\beta) %26 0 %26 sin(\beta) \\ 0 %26 1 %26 0 \\ -sin(\beta) %26 0 %26 cos(\beta) \end{bmatrix} \begin{bmatrix}1 %26 0 %26 0\\0 %26cos(\alpha) %26 -sin(\alpha)\\0 %26 sin(\alpha) %26 cos(\alpha)\end{bmatrix}">

But we have Infinitesimal rotations, and in that case we have another formula.

<img src="https://render.githubusercontent.com/render/math?math=R=I %2B Ad\theta">

Where ***I*** - unit matrix, ***A*** - member of the three-dimensional special orthogonal group ***so(3)***
In this way:

<img src="https://render.githubusercontent.com/render/math?math=R = I %2B \begin{bmatrix}0 %26 -\gamma  %26 \beta \\ \gamma %26 0 %26 -\alpha \\ -\beta  %26 \alpha  %26 0 \end{bmatrix} = I %2B skew(\begin{bmatrix} \alpha \\ \beta \\ \gamma \\ \end{bmatrix}) = I %2B skew(R_{shift}) ">



Returns to the mathematical operations:

<img src="https://render.githubusercontent.com/render/math?math=E=\sum\left\|[(I %2B skew(R_{shift})) \cdot  p %2B t - q]^{T}  \cdot n \right \|_{2} ">

<img src="https://render.githubusercontent.com/render/math?math=E = \sum \left \| [I \cdot  p %2B skew(R_{shift}) \cdot  p %2B t - q]^{T}  \cdot n \right \|_{2} ">

<img src="https://render.githubusercontent.com/render/math?math=E = \sum \left \| [skew(R_{shift}) \cdot  p %2B t %2B p- q]^{T}  \cdot n \right \|_{2} ">

Let a new function:

<img src="https://render.githubusercontent.com/render/math?math=f(x, p) = skew(R_{shift}) \cdot  p %2B t">

<img src="https://render.githubusercontent.com/render/math?math=E = \sum \left \| [f(x, p) %2B p- q]^{T}  \cdot n \right \|_{2}">

Let's find out differential of ***E***:

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial E}{\partial x_{i}} = \sum [2 \cdot (f(x, p) %2B p - q)^{T} \cdot n] \cdot [f{}'(x, p)^{T} \cdot n] = 0">

<img src="https://render.githubusercontent.com/render/math?math=\sum [2 \cdot n^{T} \cdot (f(x, p) %2B p - q)] \cdot [n^{T} \cdot f{}'(x, p)] = 0 ">

Let new variable: <img src="https://render.githubusercontent.com/render/math?math=\triangle p = p - q">

<img src="https://render.githubusercontent.com/render/math?math=\sum [2 \cdot n^{T} \cdot (f(x, p) %2B \triangle p)] \cdot [n^{T} \cdot f{}'(x, p)] = 0">

<img src="https://render.githubusercontent.com/render/math?math=\sum [(f(x, p) %2B \triangle p)^{T} \cdot (n \cdot n^{T})] \cdot f{}'(x, p) = 0">

<img src="https://render.githubusercontent.com/render/math?math=\sum f{}'(x, p)^{T} \cdot [n \cdot n^{T}] \cdot [f(x, p) %2B \triangle p] = 0">

Let's find out differential of ***f(x)***:

<img src="https://render.githubusercontent.com/render/math?math=d(f(x)) = d(skew(R_{shift}) \cdot  p %2B t) = d(skew(R_{shift})) \cdot  p %2B skew(R_{shift}) \cdot  d(p)) %2B d(t)">

<img src="https://render.githubusercontent.com/render/math?math=d(f(x)) = skew(\triangle R_{shift}) \cdot  p %2B \triangle t">

Let's remember: <img src="https://render.githubusercontent.com/render/math?math=cross(a, b) = skew(a) \cdot b = skew(b)^{T} \cdot a">

<img src="https://render.githubusercontent.com/render/math?math=d(f(x)) = cross(\triangle R_{shift}, p) %2B \triangle t">

<img src="https://render.githubusercontent.com/render/math?math=d(f(x)) = skew(p)^{T} \cdot \triangle R_{shift} %2B \triangle t">

<img src="https://render.githubusercontent.com/render/math?math=d(f(x))=\begin{bmatrix} skew(p)^{T} %26 | %26 I m\end{bmatrix} \cdot \begin{bmatrix} \triangle R_{shift}\\ \triangle t \end{bmatrix} = G(p) \cdot X">

We introduced ***G(p)*** function for simplification

<img src="https://render.githubusercontent.com/render/math?math=\sum f{}'(x, p)^{T} \cdot [n \cdot n^{T}] \cdot [f(x, p)] = \sum f{}'(x, p)^{T} \cdot [n \cdot n^{T}] \cdot [- \triangle p]">

<img src="https://render.githubusercontent.com/render/math?math=\sum G(p)^{T} \cdot [n \cdot n^{T}] \cdot [G(p) \cdot X] = \sum G(p)^{T} \cdot [n \cdot n^{T}] \cdot [- \triangle p]">

Let a new value:

<img src="https://render.githubusercontent.com/render/math?math=C = G(p)^{T} \cdot n">

<img src="https://render.githubusercontent.com/render/math?math=C^{T} = (G(p)^{T} \cdot n)^{T} = n^{T} \cdot G(p)">

Let's make a replacement:

<img src="https://render.githubusercontent.com/render/math?math=\sum C \cdot C^{T} \cdot X = \sum C \cdot n^{T} \cdot [- \triangle p]">

<img src="https://render.githubusercontent.com/render/math?math=\sum C\cdot C^{T}\cdot \begin{bmatrix} \triangle R_{shift}\\ \triangle t \end{bmatrix} = \sum C \cdot n^{T} \cdot [- \triangle p]">

We solve this equation and as result, we have diff Rigid transform
