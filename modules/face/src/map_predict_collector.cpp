/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


License Agreement
For Open Source Computer Vision Library
(3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/
#include "opencv2/face/predict_collector.hpp"
#include "opencv2/core/cvstd.hpp"
#include <iterator>
namespace cv {
namespace face {

CV_WRAP bool MapPredictCollector::emit(const int label, const double dist, const int state)
{
    ((void)state);
    //if already in index check which is closer
    if (_idx->find(label) != _idx->end()) {
        double current = (*_idx)[label];
        if (dist < current) {
            (*_idx)[label] = dist;
        }
    }
    else {
        (*_idx)[label] = dist;
    }
    return true;
}

Ptr<std::map<int, double> > MapPredictCollector::getResult()
{
    return _idx;
}

CV_WRAP std::vector<std::pair<int, double> > MapPredictCollector::getResultVector()
{
    std::vector<std::pair<int, double> > result;
    std::copy(_idx->begin(), _idx->end(), std::back_inserter(result));
    return result;
}

CV_WRAP Ptr<MapPredictCollector> MapPredictCollector::create(double threshold)
{
    return Ptr<MapPredictCollector>(new MapPredictCollector(threshold));
}



}
}