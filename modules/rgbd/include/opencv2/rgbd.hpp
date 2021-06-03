// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

#ifndef __OPENCV_RGBD_HPP__
#define __OPENCV_RGBD_HPP__

#include "opencv2/rgbd/linemod.hpp"
#include "opencv2/rgbd/depth.hpp"
#include "opencv2/rgbd/kinfu.hpp"
#include "opencv2/rgbd/dynafu.hpp"
#include "opencv2/rgbd/large_kinfu.hpp"
#include "opencv2/rgbd/detail/pose_graph.hpp"
#include "opencv2/rgbd/colored_kinfu.hpp"


/** @defgroup rgbd RGB-Depth Processing
The Iterative closest point (ICP) function minimizes the PointToPoint Distance (PPD) between the corresponding points in two clouds of points and normals.
The main equetion, which it needs to minimize:
\f[ E = \sum \left \| ppd(p_{i}, q_{i}, n_{i}) \right \|_{2} \rightarrow 0 \f]


Let's watch what is ppd(p, q, n)
Firstly, we have two clouds of points, old (the existing points and normals in 3-D model) and new (new points and normals, what we want to integrate to the exising model)
p - \f$ i^{th} \f$ point in the new cloud of points
q - \f$ i^{th} \f$ point in the old cloud of points
n - \f$ i^{th} \f$ normal in the old cloud of normals

\f$ ppd(...) \f$ - is the distance \f$ \rightarrow \f$ its formula is the dot product of (difference between p and q) and (n):
\f[ dot(T_{p2q}(p)-q, n) = dot((R \cdot  p + t) - q, n) = [(R \cdot  p + t)- q]^{T}  \cdot n \f]
\f[ T_{p2q}(p) \f] - rigid transform of p point, which brings it closer to the corresponding q point.
\f[ T_{p2q}(p) = (R \cdot  p + t) \f], where R - rotation, t - translation.

We use the Gauss-Newton method for the minimization of function.
In the beginning, we will perform some mathematical operations:
\f[ E = \sum \left \| [(R \cdot  p + t)- q]^{T}  \cdot n \right \|_{2} \f]

R is rotation and its formula is complicated:
\f[ R = R_{z}(\gamma)R_{y}(\beta )R_{x}(\alpha)=
\begin{bmatrix}
cos(\gamma) & -sin(\gamma) & 0 \\
sin(\gamma) & cos(\gamma) & 0\\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
cos(\beta) & 0 & sin(\beta)\\
0 & 1 & 0\\
-sin(\beta) & 0 & cos(\beta)
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0\\
0 & cos(\alpha) & -sin(\alpha)\\
0 & sin(\alpha) & cos(\alpha)
\end{bmatrix} \f]
But we have Infinitesimal rotations, and in that case we have another formula.
\f[ R = I + Ad\theta \f], where I - unit matrix, A - member of the three-dimensional special orthogonal group SO(3)
In this way:
\f[ R = I + \begin{bmatrix}
0  & -\gamma  & \beta \\
\gamma & 0 & -\alpha \\
-\beta  & \alpha  & 0
\end{bmatrix}
=
I + skew(\begin{bmatrix}
\alpha \\
\beta \\
\gamma \\
\end{bmatrix})
=
I + skew(R_{shift}) \f]

Returns to the mathematical operations:
\f[ E = \sum \left \| [(I + skew(R_{shift})) \cdot  p + t- q]^{T}  \cdot n \right \|_{2} \f]
\f[ E = \sum \left \| [I \cdot  p + skew(R_{shift}) \cdot  p + t- q]^{T}  \cdot n \right \|_{2} \f]
\f[ E = \sum \left \| [skew(R_{shift}) \cdot  p + t + p- q]^{T}  \cdot n \right \|_{2} \f]
let a new function:
\f[ f(x, p) = skew(R_{shift}) \cdot  p + t \f]
\f[ E = \sum \left \| [f(x, p) + p- q]^{T}  \cdot n \right \|_{2} \f]

Let's find out differential of E:
\f[ \frac{\partial E}{\partial x_{i}} = \sum [2 \cdot (f(x, p) + p - q)^{T} \cdot n] \cdot [f{}'(x, p)^{T} \cdot n] = 0 \f]
\f[ \sum [2 \cdot n^{T} \cdot (f(x, p) + p - q)] \cdot [n^{T} \cdot f{}'(x, p)] = 0 \f]
Let new variable: \f$ \triangle p = p - q \f$
\f[ \sum [2 \cdot n^{T} \cdot (f(x, p) + \triangle p)] \cdot [n^{T} \cdot f{}'(x, p)] = 0 \f]
\f[ \sum [(f(x, p) + \triangle p)^{T} \cdot (n \cdot n^{T})] \cdot f{}'(x, p) = 0 \f]
\f[ \sum f{}'(x, p)^{T} \cdot [n \cdot n^{T}] \cdot [f(x, p) + \triangle p] = 0 \f]

Let's find out differential of f(x):
\f[ d(f(x)) = d(skew(R_{shift}) \cdot  p + t) = d(skew(R_{shift})) \cdot  p  + skew(R_{shift}) \cdot  d(p))+ d(t) \f]
\f[ d(f(x)) = skew(\triangle R_{shift}) \cdot  p + \triangle t \f]
let's remember: \f$ cross(a, b) = skew(a) \cdot b = skew(b)^{T} \cdot a \f$
\f[ d(f(x)) = cross(\triangle R_{shift}, p) + \triangle t \f]
\f[ d(f(x)) = skew(p)^{T} \cdot \triangle R_{shift} + \triangle t \f]
\f[ d(f(x)) =
\begin{bmatrix}
 skew(p)^{T} & | & I
\end{bmatrix}
\cdot
\begin{bmatrix}
\triangle R_{shift}\\
\triangle t
\end{bmatrix}
= G(p) \cdot X \f]
We introduced G(p) function for simplification

\f[ \sum f{}'(x, p)^{T} \cdot [n \cdot n^{T}] \cdot [f(x, p)] = \sum f{}'(x, p)^{T} \cdot [n \cdot n^{T}] \cdot [- \triangle p] \f]
\f[ \sum G(p)^{T} \cdot [n \cdot n^{T}] \cdot [G(p) \cdot X] = \sum G(p)^{T} \cdot [n \cdot n^{T}] \cdot [- \triangle p] \f]
let a new value:
\f[ C = G(p)^{T} \cdot n \f]
\f[ C^{T} = (G(p)^{T} \cdot n)^{T} = n^{T} \cdot G(p) \f]

Let's make a replacement:
\f[ \sum C \cdot C^{T} \cdot X = \sum C \cdot n^{T} \cdot [- \triangle p] \f]
\f[ \sum C \cdot C^{T} \cdot
\begin{bmatrix}
\triangle R_{shift}\\
\triangle t
\end{bmatrix}
= \sum C \cdot n^{T} \cdot [- \triangle p] \f]

We solve this equation and as result, we have diff Rigid transform

*/

#endif

/* End of file. */
