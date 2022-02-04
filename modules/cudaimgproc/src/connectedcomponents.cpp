// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::cuda::connectedComponents(InputArray img_, OutputArray labels_, int connectivity,
    int ltype, ConnectedComponentsAlgorithmsTypes ccltype) { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device { namespace imgproc {
        void BlockBasedKomuraEquivalence(const cv::cuda::GpuMat& img, cv::cuda::GpuMat& labels);
}}}}


void cv::cuda::connectedComponents(InputArray img_, OutputArray labels_, int connectivity,
    int ltype, ConnectedComponentsAlgorithmsTypes ccltype) {
    const cv::cuda::GpuMat img = img_.getGpuMat();
    cv::cuda::GpuMat& labels = labels_.getGpuMatRef();

    CV_Assert(img.channels() == 1);
    CV_Assert(connectivity == 8);
    CV_Assert(ltype == CV_32S);
    CV_Assert(ccltype == CCL_BKE || ccltype == CCL_DEFAULT);

    int iDepth = img_.depth();
    CV_Assert(iDepth == CV_8U || iDepth == CV_8S);

    labels.create(img.size(), CV_MAT_DEPTH(ltype));

    if ((ccltype == CCL_BKE || ccltype == CCL_DEFAULT) && connectivity == 8 && ltype == CV_32S) {
        using cv::cuda::device::imgproc::BlockBasedKomuraEquivalence;
        BlockBasedKomuraEquivalence(img, labels);
    }

}

void cv::cuda::connectedComponents(InputArray img_, OutputArray labels_, int connectivity, int ltype) {
    cv::cuda::connectedComponents(img_, labels_, connectivity, ltype, CCL_DEFAULT);
}


#endif /* !defined (HAVE_CUDA) */
