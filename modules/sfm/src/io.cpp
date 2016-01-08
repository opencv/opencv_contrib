/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include <opencv2/sfm/io.hpp>
#include "io/io_bundler.h"

namespace cv
{
namespace sfm
{

void
importReconstruction(const cv::String &file, OutputArrayOfArrays _Rs,
                     OutputArrayOfArrays _Ts, OutputArrayOfArrays _Ks,
                     OutputArray _points3d, int file_format) {

    std::vector<Matx33d> Rs, Ks;
    std::vector<Vec3d> Ts, points3d;

    if (file_format == SFM_IO_BUNDLER) {
        readBundlerFile(file, Rs, Ts, Ks, points3d);
    } else if (file_format == SFM_IO_VISUALSFM) {
        CV_Error(Error::StsNotImplemented, "The requested function/feature is not implemented");
    } else if (file_format == SFM_IO_OPENSFM) {
        CV_Error(Error::StsNotImplemented, "The requested function/feature is not implemented");
    } else if (file_format == SFM_IO_OPENMVG) {
        CV_Error(Error::StsNotImplemented, "The requested function/feature is not implemented");
    } else if (file_format == SFM_IO_THEIASFM) {
        CV_Error(Error::StsNotImplemented, "The requested function/feature is not implemented");
    } else {
        CV_Error(Error::StsBadArg, "The file format one of SFM_IO_BUNDLER, SFM_IO_VISUALSFM, SFM_IO_OPENSFM, SFM_IO_OPENMVG or SFM_IO_THEIASFM");
    }

    const size_t num_cameras = Rs.size();
    const size_t num_points = points3d.size();

    _Rs.create(num_cameras, 1, CV_64F);
    _Ts.create(num_cameras, 1, CV_64F);
    _Ks.create(num_cameras, 1, CV_64F);
    _points3d.create(num_points, 1, CV_64F);

    for (size_t i = 0; i < num_cameras; ++i) {
        Mat(Rs[i]).copyTo(_Rs.getMatRef(i));
        Mat(Ts[i]).copyTo(_Ts.getMatRef(i));
        Mat(Ks[i]).copyTo(_Ks.getMatRef(i));
    }

    for (size_t i = 0; i < num_points; ++i)
        Mat(points3d[i]).copyTo(_points3d.getMatRef(i));
}


} /* namespace sfm */
} /* namespace cv */