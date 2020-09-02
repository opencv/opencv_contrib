// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __OPENCV_MCC_OPERATIONS_HPP__
#define __OPENCV_MCC_OPERATIONS_HPP__

#include <functional>
#include <vector>
#include "opencv2/mcc/utils.hpp"

namespace cv
{
namespace ccm
{

typedef std::function<cv::Mat(cv::Mat)> MatFunc;

/* *\ brief Operation class contains some operarions used for color space
   *        conversion containing linear transformation and non-linear transformation
 */
class Operation
{
public:
    bool linear;
    cv::Mat M;
    MatFunc f;

    Operation() : linear(true), M(cv::Mat()) {};

    Operation(cv::Mat M_) :linear(true), M( M_ ) {};

    Operation(MatFunc f_) : linear(false), f(f_) {};

    virtual ~Operation() {};
    /* *\ brief operator function will run operation*/
    cv::Mat operator()(cv::Mat& abc)
    {
        if (!linear)
        {
            return f(abc);
        }
        if (M.empty())
        {
            return abc;
        }
        return multiple(abc, M);
    };

    /* *\ brief add function will conbine this operation
     *          with  other  linear transformation operation*/
    void add(const Operation& other)
    {
        if (M.empty())
        {
            M = other.M.clone();
        }
        else
        {
            M = M * other.M;
        }
    };

    void clear()
    {
        M = cv::Mat();
    };
};

const Operation IDENTITY_OP( [](cv::Mat x) {return x; } );

class Operations
{
public:
    std::vector<Operation> ops;

    Operations() :ops{ } {};

    Operations(std::initializer_list<Operation> op) :ops{ op } {};

    virtual ~Operations() {};

    /* *\ brief add function will conbine this operation with  other  transformation operations*/
    Operations& add(const Operations& other)
    {
        ops.insert(ops.end(), other.ops.begin(), other.ops.end());
        return *this;
    };

    /* *\ brief run operations to make color conversion*/
    cv::Mat run(cv::Mat abc)
    {
        Operation hd;
        for (auto& op : ops)
        {
            if (op.linear)
            {
                hd.add(op);
            }
            else
            {
                abc = hd(abc);
                hd.clear();
                abc = op(abc);
            }
        }
        abc = hd(abc);
        return abc;
    };
};

const Operations IDENTITY_OPS{ IDENTITY_OP };

} // namespace ccm
} // namespace cv


#endif