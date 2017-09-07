#pragma once

#include <opencv2/kfusion/types.hpp>
#include <opencv2/kfusion/kinfu.hpp>
#include <opencv2/kfusion/cuda/tsdf_volume.hpp>
#include <opencv2/kfusion/cuda/imgproc.hpp>
#include <opencv2/kfusion/cuda/projective_icp.hpp>
#include "internal.hpp"
#include <iostream>
#include "vector_functions.h"

namespace cv
{
    namespace kfusion
    {
        template<typename D, typename S>
        inline D device_cast(const S& source)
        {
            return *reinterpret_cast<const D*>(source.val);
        }

        template<>
        inline device::Aff3f device_cast<device::Aff3f, Affine3f>(const Affine3f& source)
        {
            device::Aff3f aff;
            Mat3f R = source.rotation();
            Vec3f t = source.translation();
            aff.R = device_cast<device::Mat3f>(R);
            aff.t = device_cast<device::Vec3f>(t);
            return aff;
        }
    }
}
