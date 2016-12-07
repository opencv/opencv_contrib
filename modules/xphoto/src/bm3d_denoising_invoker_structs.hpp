/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef __OPENCV_BM3D_DENOISING_INVOKER_STRUCTS_HPP__
#define __OPENCV_BM3D_DENOISING_INVOKER_STRUCTS_HPP__

namespace cv
{
namespace xphoto
{

template <typename T, typename DT, typename CT>
class BlockMatch
{
public:
    // Data accessor
    T* data()
    {
        return data_;
    }

    // Const version of data accessor
    const T* data() const
    {
        return data_;
    }

    // Allocate memory for data
    void init(const int &blockSizeSq)
    {
        data_ = new T[blockSizeSq];
    }

    // Release data memory
    void release()
    {
        delete[] data_;
    }

    // Overloaded operator for convenient assignment
    void operator()(const DT &_dist, const CT &_coord_x, const CT &_coord_y)
    {
        dist = _dist;
        coord_x = _coord_x;
        coord_y = _coord_y;
    }

    // Overloaded array subscript operator
    T& operator[](const std::size_t &idx)
    {
        return data_[idx];
    };

    // Overloaded const array subscript operator
    const T& operator[](const std::size_t &idx) const
    {
        return data_[idx];
    };

    // Overloaded comparison operator for sorting
    bool operator<(const BlockMatch& right) const
    {
        return dist < right.dist;
    }

    // Block matching distance
    DT dist;

    // Relative coordinates to the current search window
    CT coord_x;
    CT coord_y;

private:
    // Pointer to the pixel values of the block
    T *data_;
};

class DistAbs
{
    template <typename T>
    struct calcDist_
    {
        static inline int f(const T &a, const T &b)
        {
            return std::abs(a - b);
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 2> >
    {
        static inline int f(const Vec<ET, 2> a, const Vec<ET, 2> b)
        {
            return std::abs((int)(a[0] - b[0])) + std::abs((int)(a[1] - b[1]));
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 3> >
    {
        static inline int f(const Vec<ET, 3> a, const Vec<ET, 3> b)
        {
            return
                std::abs((int)(a[0] - b[0])) +
                std::abs((int)(a[1] - b[1])) +
                std::abs((int)(a[2] - b[2]));
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 4> >
    {
        static inline int f(const Vec<ET, 4> a, const Vec<ET, 4> b)
        {
            return
                std::abs((int)(a[0] - b[0])) +
                std::abs((int)(a[1] - b[1])) +
                std::abs((int)(a[2] - b[2])) +
                std::abs((int)(a[3] - b[3]));
        }
    };

public:
    template <typename T>
    static inline int calcDist(const T &a, const T &b)
    {
        return calcDist_<T>::f(a, b);
    }

    template <typename T>
    static inline int calcDist(const Mat& m, int i1, int j1, int i2, int j2)
    {
        const T a = m.at<T>(i1, j1);
        const T b = m.at<T>(i2, j2);
        return calcDist<T>(a, b);
    }

    template <typename T>
    static inline int calcUpDownDist(T a_up, T a_down, T b_up, T b_down)
    {
        return calcDist<T>(a_down, b_down) - calcDist<T>(a_up, b_up);
    };

    template <typename T>
    static inline T calcBlockMatchingThreshold(const T &blockMatchThrL2, const T &blockSizeSq)
    {
        return (T)(std::sqrt((double)blockMatchThrL2) * blockSizeSq);
    }
};

class DistSquared
{
    template <typename T>
    struct calcDist_
    {
        static inline int f(const T &a, const T &b)
        {
            return (a - b) * (a - b);
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 2> >
    {
        static inline int f(const Vec<ET, 2> a, const Vec<ET, 2> b)
        {
            return (int)(a[0] - b[0])*(int)(a[0] - b[0]) + (int)(a[1] - b[1])*(int)(a[1] - b[1]);
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 3> >
    {
        static inline int f(const Vec<ET, 3> a, const Vec<ET, 3> b)
        {
            return
                (int)(a[0] - b[0])*(int)(a[0] - b[0]) +
                (int)(a[1] - b[1])*(int)(a[1] - b[1]) +
                (int)(a[2] - b[2])*(int)(a[2] - b[2]);
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 4> >
    {
        static inline int f(const Vec<ET, 4> a, const Vec<ET, 4> b)
        {
            return
                (int)(a[0] - b[0])*(int)(a[0] - b[0]) +
                (int)(a[1] - b[1])*(int)(a[1] - b[1]) +
                (int)(a[2] - b[2])*(int)(a[2] - b[2]) +
                (int)(a[3] - b[3])*(int)(a[3] - b[3]);
        }
    };

    template <typename T> struct calcUpDownDist_
    {
        static inline int f(T a_up, T a_down, T b_up, T b_down)
        {
            int A = a_down - b_down;
            int B = a_up - b_up;
            return (A - B)*(A + B);
        }
    };

    template <typename ET, int n> struct calcUpDownDist_<Vec<ET, n> >
    {
    private:
        typedef Vec<ET, n> T;
    public:
        static inline int f(T a_up, T a_down, T b_up, T b_down)
        {
            return calcDist<T>(a_down, b_down) - calcDist<T>(a_up, b_up);
        }
    };

public:
    template <typename T>
    static inline int calcDist(const T &a, const T &b)
    {
        return calcDist_<T>::f(a, b);
    }

    template <typename T>
    static inline int calcDist(const Mat& m, int i1, int j1, int i2, int j2)
    {
        const T a = m.at<T>(i1, j1);
        const T b = m.at<T>(i2, j2);
        return calcDist<T>(a, b);
    }

    template <typename T>
    static inline int calcUpDownDist(T a_up, T a_down, T b_up, T b_down)
    {
        return calcUpDownDist_<T>::f(a_up, a_down, b_up, b_down);
    };

    template <typename T>
    static inline T calcBlockMatchingThreshold(const T &blockMatchThrL2, const T &blockSizeSq)
    {
        return blockMatchThrL2 * blockSizeSq;
    }
};

template <class T>
struct Array2d
{
    T* a;
    int n1, n2;
    bool needToDeallocArray;

    Array2d(const Array2d& array2d) :
        a(array2d.a), n1(array2d.n1), n2(array2d.n2), needToDeallocArray(false)
    {
        if (array2d.needToDeallocArray)
        {
            CV_Error(Error::BadDataPtr, "Copy constructor for self allocating arrays not supported");
        }
    }

    Array2d(T* _a, int _n1, int _n2) :
        a(_a), n1(_n1), n2(_n2), needToDeallocArray(false)
    {
    }

    Array2d(int _n1, int _n2) :
        n1(_n1), n2(_n2), needToDeallocArray(true)
    {
        a = new T[n1*n2];
    }

    ~Array2d()
    {
        if (needToDeallocArray)
            delete[] a;
    }

    T* operator [] (int i)
    {
        return a + i*n2;
    }

    inline T* row_ptr(int i)
    {
        return (*this)[i];
    }
};

template <class T>
struct Array3d
{
    T* a;
    int n1, n2, n3;
    bool needToDeallocArray;

    Array3d(T* _a, int _n1, int _n2, int _n3) :
        a(_a), n1(_n1), n2(_n2), n3(_n3), needToDeallocArray(false)
    {
    }

    Array3d(int _n1, int _n2, int _n3) :
        n1(_n1), n2(_n2), n3(_n3), needToDeallocArray(true)
    {
        a = new T[n1*n2*n3];
    }

    ~Array3d()
    {
        if (needToDeallocArray)
            delete[] a;
    }

    Array2d<T> operator [] (int i)
    {
        Array2d<T> array2d(a + i*n2*n3, n2, n3);
        return array2d;
    }

    inline T* row_ptr(int i1, int i2)
    {
        return a + i1*n2*n3 + i2*n3;
    }
};

}  // namespace xphoto
}  // namespace cv

#endif