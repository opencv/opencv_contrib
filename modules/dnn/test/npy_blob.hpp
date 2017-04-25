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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_DNN_TEST_NPY_BLOB_HPP__
#define __OPENCV_DNN_TEST_NPY_BLOB_HPP__
#include "test_precomp.hpp"
#include "cnpy.h"

namespace cv
{

inline Mat blobFromNPY(const String &path)
{
    cnpy::NpyArray npyBlob = cnpy::npy_load(path.c_str());
    Mat blob = Mat((int)npyBlob.shape.size(), (int*)&npyBlob.shape[0], CV_32F, npyBlob.data).clone();
    npyBlob.destruct();
    return blob;
}

inline void saveBlobToNPY(const Mat &blob, const String &path)
{
    cnpy::npy_save(path.c_str(), blob.ptr<float>(), (unsigned*)&blob.size.p[0], blob.dims);
}

inline size_t shapeTotal(const std::vector<int>& shape)
{
    size_t p = 1, i, n = shape.size();
    for( i = 0; i < n; i++)
        p *= shape[i];
    return p;
}

inline bool shapeEqual(const std::vector<int>& shape1, const std::vector<int>& shape2)
{
    size_t i, n1 = shape1.size(), n2 = shape2.size();
    if( n1 != n2 )
        return false;
    for( i = 0; i < n1; i++ )
        if( shape1[i] != shape2[i] )
            return false;
    return true;
}

inline std::vector<int> getShape(const Mat& m)
{
    return m.empty() ? std::vector<int>() : std::vector<int>(&m.size.p[0], &m.size.p[0] + m.dims);
}

inline std::vector<int> makeShape(int a0, int a1=-1, int a2=-1, int a3=-1, int a4=-1, int a5=-1)
{
    std::vector<int> s;
    s.push_back(a0);
    if(a1 > 0)
    {
        s.push_back(a1);
        if(a2 > 0)
        {
            s.push_back(a2);
            if(a3 > 0)
            {
                s.push_back(a3);
                if(a4 > 0)
                {
                    s.push_back(a4);
                    if(a5 > 0)
                        s.push_back(a5);
                }
            }
        }
    }
    return s;
}

inline std::vector<int> concatShape(const std::vector<int>& a, const std::vector<int>& b)
{
    size_t na = a.size(), nb = b.size();
    std::vector<int> c(na + nb);

    std::copy(a.begin(), a.end(), c.begin());
    std::copy(b.begin(), b.end(), c.begin() + na);

    return c;
}

inline void printShape(const String& name, const std::vector<int>& shape)
{
    printf("%s: [", name.c_str());
    size_t i, n = shape.size();
    for( i = 0; i < n; i++ )
        printf(" %d", shape[i]);
    printf(" ]\n");
}

}

#endif
