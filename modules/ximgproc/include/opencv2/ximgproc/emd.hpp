// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_XIMGPROC_EMD_HPP__
#define __OPENCV_XIMGPROC_EMD_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv { namespace ximgproc {

//! @addtogroup ximgproc
//! @{

/** @brief Computes the "minimal work" distance between two weighted point configurations.

The function computes the earth mover distance and/or a lower boundary of the distance between the
two weighted point configurations. One of the applications described in @cite RubnerSept98,
@cite Rubner2000 is multi-dimensional histogram comparison for image retrieval. EMD is a transportation
problem that is solved using some modification of a simplex algorithm, thus the complexity is
exponential in the worst case, though, on average it is much faster. In the case of a real metric
the lower boundary can be calculated even faster (using linear-time algorithm) and it can be used
to determine roughly whether the two signatures are far enough so that they cannot relate to the
same object.

@param signature1 First signature, a \f$\texttt{size1}\times \texttt{dims}+1\f$ floating-point matrix.
Each row stores the point weight followed by the point coordinates. The matrix is allowed to have
a single column (weights only) if the user-defined cost matrix is used. The weights must be
non-negative and have at least one non-zero value.
@param signature2 Second signature of the same format as signature1 , though the number of rows
may be different. The total weights may be different. In this case an extra "dummy" point is added
to either signature1 or signature2. The weights must be non-negative and have at least one non-zero
value.
@param distType Used metric. See #DistanceTypes.
@param cost User-defined \f$\texttt{size1}\times \texttt{size2}\f$ cost matrix. Also, if a cost matrix
is used, lower boundary lowerBound cannot be calculated because it needs a metric function.
@param lowerBound Optional input/output parameter: lower boundary of a distance between the two
signatures that is a distance between mass centers. The lower boundary may not be calculated if
the user-defined cost matrix is used, the total weights of point configurations are not equal, or
if the signatures consist of weights only (the signature matrices have a single column). You
**must** initialize \*lowerBound . If the calculated distance between mass centers is greater or
equal to \*lowerBound (it means that the signatures are far enough), the function does not
calculate EMD. In any case \*lowerBound is set to the calculated distance between mass centers on
return. Thus, if you want to calculate both distance between mass centers and EMD, \*lowerBound
should be set to 0.
@param flow Resultant \f$\texttt{size1} \times \texttt{size2}\f$ flow matrix: \f$\texttt{flow}_{i,j}\f$ is
a flow from \f$i\f$ -th point of signature1 to \f$j\f$ -th point of signature2 .
 */
CV_EXPORTS float EMD( InputArray signature1, InputArray signature2,
                      int distType, InputArray cost=noArray(),
                      float* lowerBound = nullptr, OutputArray flow = noArray() );

CV_EXPORTS_AS(EMD) float wrapperEMD( InputArray signature1, InputArray signature2,
                      int distType, InputArray cost=noArray(),
                      CV_IN_OUT Ptr<float> lowerBound = Ptr<float>(), OutputArray flow = noArray() );

//! @}

}} // namespace cv::ximgproc

#endif // __OPENCV_XIMGPROC_EMD_HPP__
