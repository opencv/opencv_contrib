// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_ALPHAMAT_INFOFLOW_HPP_
#define _OPENCV_ALPHAMAT_INFOFLOW_HPP_

/** Information Flow algorithm implementaton for alphamatting

This module contains functionality for extracting the
The following four models are implemented:

- EDSR <https://arxiv.org/abs/1707.02921>
- ESPCN <https://arxiv.org/abs/1609.05158>
- FSRCNN <https://arxiv.org/abs/1608.00367>
- LapSRN <https://arxiv.org/abs/1710.01992>

*/
#include <Eigen/Sparse>
using namespace Eigen;

namespace cv{ namespace alphamat{

    void solve(SparseMatrix<double> Wcm,SparseMatrix<double> Wuu,SparseMatrix<double> Wl,SparseMatrix<double> Dcm,
            SparseMatrix<double> Duu,SparseMatrix<double> Dl,SparseMatrix<double> H,SparseMatrix<double> T,
            Mat& ak, Mat& wf, bool useKU, Mat& alpha);

    CV_EXPORTS_W void infoFlow(Mat& image, Mat& tmap, Mat& result, bool useKU, bool trim);

}}
#endif
