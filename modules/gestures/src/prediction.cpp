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

#include <opencv2/gestures/prediction.hpp>

namespace cv
{
    namespace gestures
    {
        Prediction::Prediction()
        {
        }
        Prediction::Prediction(int id, std::string label, float proba):
            mClassId(id),
            mClassLabel(label),
            mProbability(proba)
        {
        }

        void Prediction::setClassId(int id)
        {
            CV_Assert(id > 0);
            mClassId = id;
        }
        int Prediction::getClassId() const
        {
            return mClassId;
        }

        void Prediction::setClassLabel(std::string label)
        {
            mClassLabel = label;
        }
        std::string Prediction::getClassLabel() const
        {
            return mClassLabel;
        }

        void Prediction::setProbability(float proba)
        {
            CV_Assert(proba >= 0.0 && proba <= 1.0);
            mProbability = proba;
        }
        float Prediction::getProbability() const
        {
            return mProbability;
        }

        bool operator<(const Prediction& lhs, const Prediction& rhs)
        {
            return lhs.getProbability() < rhs.getProbability();
        }
        bool operator>(const Prediction& lhs, const Prediction& rhs)
        {
            return rhs < lhs;
        }
        bool operator<=(const Prediction& lhs, const Prediction& rhs)
        {
            return !(rhs > lhs);
        }
        bool operator>=(const Prediction& lhs, const Prediction& rhs)
        {
            return !(lhs < rhs);
        }
    } // namespace gestures
} // namespace cv
