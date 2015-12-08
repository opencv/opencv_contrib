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

#ifndef __OPENCV_PREDICT_COLLECTOR_HPP__
#define __OPENCV_PREDICT_COLLECTOR_HPP__
#include <cfloat>
#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
namespace cv {
namespace face {
//! @addtogroup face
//! @{
/** @brief Abstract base class for all strategies of prediction result handling
*/
class CV_EXPORTS_W PredictCollector {
protected:
    double _threshhold;
    int _size;
    int _state;
public:
    /** @brief creates new predict collector with given threshhold */
    PredictCollector(double threshhold = DBL_MAX) :_threshhold(threshhold) {};
    CV_WRAP virtual ~PredictCollector() {}
    /** @brief called once at start of recognition
    @param size total size of prediction evaluation that recognizer could perform
    @param state user defined send-to-back optional value to allow multi-thread, multi-session or aggregation scenarios
    */
    CV_WRAP virtual void init(const int size, const int state = 0);
    /** @brief called with every recognition result
    @param label current prediction label
    @param dist current prediction distance (confidence)
    @param state user defined send-to-back optional value to allow multi-thread, multi-session or aggregation scenarios
    @return true if recognizer should proceed prediction , false - if recognizer should terminate prediction
    */
    CV_WRAP virtual bool emit(const int label, const double dist, const int state = 0); //not abstract while Python generation require non-abstract class
};

/** @brief default predict collector that trace minimal distance with treshhold checking (that is default behavior for most predict logic)
*/
class CV_EXPORTS_W MinDistancePredictCollector : public PredictCollector {
private:
    int _label;
    double _dist;
public:
    /** @brief creates new MinDistancePredictCollector with given threshhold */
    CV_WRAP MinDistancePredictCollector(double threshhold = DBL_MAX) : PredictCollector(threshhold) {
        _label = 0;
        _dist = DBL_MAX;
    };
    CV_WRAP bool emit(const int label, const double dist, const int state = 0);
    /** @brief result label, 0 if not found */
    CV_WRAP int getLabel() const;
    /** @brief result distance (confidence) DBL_MAX if not found */
    CV_WRAP double getDist() const;
    /** @brief factory method to create cv-pointers to MinDistancePredictCollector */
    CV_WRAP static Ptr<MinDistancePredictCollector> create(double threshold = DBL_MAX);
};
//! @}
}
}
#endif