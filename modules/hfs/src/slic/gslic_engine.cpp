// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#ifdef _HFS_CUDA_ON_

#include "slic.hpp"

namespace cv { namespace hfs { namespace slic { namespace engines {

SegEngine::SegEngine(const slicSettings& in_settings)
{
    slic_settings = in_settings;
}

SegEngine::~SegEngine() {}

void SegEngine::performSegmentation(Ptr<UChar4Image> in_img)
{
    source_img->setFrom(in_img, UChar4Image::CPU_TO_CUDA);
    cvtImgSpace(source_img, cvt_img);

    initClusterCenters();
    findCenterAssociation();

    for (int i = 0; i < slic_settings.num_iters; i++)
    {
        updateClusterCenter();
        findCenterAssociation();
    }

    enforceConnectivity();
    cudaDeviceSynchronize();
}


CoreEngine::CoreEngine(const slicSettings& in_settings)
{
    slic_seg_engine = Ptr<SegEngine>(new SegEngineGPU(in_settings));
}

CoreEngine::~CoreEngine() {}

void CoreEngine::setImageSize(int x, int y)
{
    slic_seg_engine->setImageSize(x, y);
}

void CoreEngine::processFrame(Ptr<UChar4Image> in_img)
{
    slic_seg_engine->performSegmentation(in_img);
}

const Ptr<IntImage> CoreEngine::getSegRes()
{
    return slic_seg_engine->getSegMask();
}


}}}}

#endif
