/*M///////////////////////////////////////////////////////////////////////////////////////
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//                        (3-clause BSD License)
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * Neither the names of the copyright holders nor the names of the contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// Author : Awabot SAS
// Copyright (C) 2015, Awabot SAS, all rights reserved.
//
//M*/

#ifndef __OPENCV_GESTURES_PREDICTION_HPP__
#define __OPENCV_GESTURES_PREDICTION_HPP__

#ifdef __cplusplus

#include <string>
#include <opencv2/core.hpp>

/** @defgroup gestures Gestures Recognition module
 */

namespace cv
{
    namespace gestures
    {
        //! @addtogroup gestures
        //! @{
        /* @brief Very simple class to wrap all information about a prediction
         */
        class CV_EXPORTS Prediction
        {
            public:
                /**
                 Default constructor
                 */
                CV_WRAP Prediction();
                /** @overload
                 Full constructor
                 */
                CV_WRAP Prediction(int id, std::string label, float proba = 0.0);

                /**
                 @param id Id of the class corresponding to the prediction
                 */
                CV_WRAP void setClassId(int id);
                /**
                 Returns the id of the class corresponding to the prediction
                 */
                CV_WRAP int getClassId() const;

                /**
                 @param label Label of the class corresponding to the prediction
                 */
                CV_WRAP void setClassLabel(std::string label);
                /**
                 Returns the label of the class corresponding to the prediction
                 */
                CV_WRAP std::string getClassLabel() const;

                /**
                 @param proba Probability of the class
                 */
                CV_WRAP void setProbability(float proba);
                /**
                 Returns the probability of the class
                 */
                CV_WRAP float getProbability() const;

            private:
                int mClassId;
                std::string mClassLabel;
                float mProbability;
        };
        //! @} gestures

        /**
         Comparison operators
         */
        bool operator<(const Prediction& lhs, const Prediction& rhs);
        bool operator>(const Prediction& lhs, const Prediction& rhs);
        bool operator<=(const Prediction& lhs, const Prediction& rhs);
        bool operator>=(const Prediction& lhs, const Prediction& rhs);

    } // namespace gestures
} // namespace cv

#endif // __cpluscplus
#endif // __OPENCV_GESTURES_CLASSIFIER_HPP__
