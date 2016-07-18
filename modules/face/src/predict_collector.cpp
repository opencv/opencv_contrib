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
namespace cv {
namespace face {


void PredictCollector::init(const int size, const int state) {
    //reserve for some-how usage in descendants
    _size = size;
    _state = state;
}

CV_WRAP bool PredictCollector::defaultFilter(int * label, double * dist, const int state)
{
    // if state provided we should compare it with current state
    if (_state != 0 && _state != state) {
        return false;
    }

    // if exclude label provided we can test it first
    if (_excludeLabel != 0 && _excludeLabel == *label) {
        return false;
    }

    // initially we must recalculate distance by koef iv given
    if (_distanceKoef != 1) {
        *dist = *dist * _distanceKoef;
    }
    // check upper threshold
    if (*dist > _threshold) {
        return false;
    }
    //check inner threshold
    if (*dist < _minthreshold) {
        return false;
    }

    return true;
}

CV_WRAP bool PredictCollector::filter(int* label, double* dist, const int state)
{
    ((void)label);
    ((void)dist);
    ((void)state);
    return true; //no custom logic at base level
}

bool PredictCollector::emit(const int label, const double dist, const int state) {
    ((void)label);
    ((void)dist);
    ((void)state);
    return false; // terminate prediction - no any behavior in base PredictCollector
}

CV_WRAP bool PredictCollector::collect(int label, double dist, const int state)
{
    if (defaultFilter(&label, &dist, state) && filter(&label,&dist,state)) {
        return emit(label, dist, state);
    }
    return true;
}

CV_WRAP int PredictCollector::getSize()
{
    return _size;
}

CV_WRAP void PredictCollector::setSize(int size)
{
    _size = size;
}

CV_WRAP int PredictCollector::getState()
{
    return _state;
}

CV_WRAP void PredictCollector::setState(int state)
{
    _state = state;
}

CV_WRAP int PredictCollector::getExcludeLabel()
{
    return _excludeLabel;
}

CV_WRAP void PredictCollector::setExcludeLabel(int excludeLabel)
{
    _excludeLabel = excludeLabel;
}

CV_WRAP double PredictCollector::getDistanceKoef()
{
    return _distanceKoef;
}

CV_WRAP void PredictCollector::setDistanceKoef(double distanceKoef)
{
    _distanceKoef = distanceKoef;
}

CV_WRAP double PredictCollector::getMinThreshold()
{
    return _minthreshold;
}

CV_WRAP void PredictCollector::setMinThreshold(double minthreshold)
{
    _minthreshold = minthreshold;
}

}
}