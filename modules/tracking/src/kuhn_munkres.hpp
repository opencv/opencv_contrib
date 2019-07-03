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

#pragma once

#include "opencv2/core.hpp"

#include <memory>
#include <vector>


///
/// \brief The KuhnMunkres class
///
/// Solves the assignment problem.
///
class KuhnMunkres {
public:
    KuhnMunkres();

    ///
    /// \brief Solves the assignment problem for given dissimilarity matrix.
    /// It returns a vector that where each element is a column index for
    /// corresponding row (e.g. result[0] stores optimal column index for very
    /// first row in the dissimilarity matrix).
    /// \param dissimilarity_matrix CV_32F dissimilarity matrix.
    /// \return Optimal column index for each row. -1 means that there is no
    /// column for row.
    ///
    std::vector<size_t> Solve(const cv::Mat &dissimilarity_matrix);

private:
    static constexpr int kStar = 1;
    static constexpr int kPrime = 2;

    cv::Mat dm_;
    cv::Mat marked_;
    std::vector<cv::Point> points_;

    std::vector<int> is_row_visited_;
    std::vector<int> is_col_visited_;

    int n_;

    void TrySimpleCase();
    bool CheckIfOptimumIsFound();
    cv::Point FindUncoveredMinValPos();
    void UpdateDissimilarityMatrix(float val);
    int FindInRow(int row, int what);
    int FindInCol(int col, int what);
    void Run();
};

