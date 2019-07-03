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

#include "kuhn_munkres.hpp"

#include <algorithm>
#include <limits>
#include <vector>

KuhnMunkres::KuhnMunkres() : n_() {}

std::vector<size_t> KuhnMunkres::Solve(const cv::Mat& dissimilarity_matrix) {
    CV_Assert(dissimilarity_matrix.type() == CV_32F);
    double min_val;
    cv::minMaxLoc(dissimilarity_matrix, &min_val);
    CV_Assert(min_val >= 0);

    n_ = std::max(dissimilarity_matrix.rows, dissimilarity_matrix.cols);
    dm_ = cv::Mat(n_, n_, CV_32F, cv::Scalar(0));
    marked_ = cv::Mat(n_, n_, CV_8S, cv::Scalar(0));
    points_ = std::vector<cv::Point>(n_ * 2);

    dissimilarity_matrix.copyTo(dm_(
            cv::Rect(0, 0, dissimilarity_matrix.cols, dissimilarity_matrix.rows)));

    is_row_visited_ = std::vector<int>(n_, 0);
    is_col_visited_ = std::vector<int>(n_, 0);

    Run();

    std::vector<size_t> results(marked_.rows, -1);
    for (int i = 0; i < marked_.rows; i++) {
        const auto ptr = marked_.ptr<char>(i);
        for (int j = 0; j < marked_.cols; j++) {
            if (ptr[j] == kStar) {
                results[i] = j;
            }
        }
    }
    return results;
}

void KuhnMunkres::TrySimpleCase() {
    auto is_row_visited = std::vector<int>(n_, 0);
    auto is_col_visited = std::vector<int>(n_, 0);

    for (int row = 0; row < n_; row++) {
        auto ptr = dm_.ptr<float>(row);
        auto marked_ptr = marked_.ptr<char>(row);
        auto min_val = *std::min_element(ptr, ptr + n_);
        for (int col = 0; col < n_; col++) {
            ptr[col] -= min_val;
            if (ptr[col] == 0 && !is_col_visited[col] && !is_row_visited[row]) {
                marked_ptr[col] = kStar;
                is_col_visited[col] = 1;
                is_row_visited[row] = 1;
            }
        }
    }
}

bool KuhnMunkres::CheckIfOptimumIsFound() {
    int count = 0;
    for (int i = 0; i < n_; i++) {
        const auto marked_ptr = marked_.ptr<char>(i);
        for (int j = 0; j < n_; j++) {
            if (marked_ptr[j] == kStar) {
                is_col_visited_[j] = 1;
                count++;
            }
        }
    }

    return count >= n_;
}

cv::Point KuhnMunkres::FindUncoveredMinValPos() {
    auto min_val = std::numeric_limits<float>::max();
    cv::Point min_val_pos(-1, -1);
    for (int i = 0; i < n_; i++) {
        if (!is_row_visited_[i]) {
            auto dm_ptr = dm_.ptr<float>(i);
            for (int j = 0; j < n_; j++) {
                if (!is_col_visited_[j] && dm_ptr[j] < min_val) {
                    min_val = dm_ptr[j];
                    min_val_pos = cv::Point(j, i);
                }
            }
        }
    }
    return min_val_pos;
}

void KuhnMunkres::UpdateDissimilarityMatrix(float val) {
    for (int i = 0; i < n_; i++) {
        auto dm_ptr = dm_.ptr<float>(i);
        for (int j = 0; j < n_; j++) {
            if (is_row_visited_[i]) dm_ptr[j] += val;
            if (!is_col_visited_[j]) dm_ptr[j] -= val;
        }
    }
}

int KuhnMunkres::FindInRow(int row, int what) {
    for (int j = 0; j < n_; j++) {
        if (marked_.at<char>(row, j) == what) {
            return j;
        }
    }
    return -1;
}

int KuhnMunkres::FindInCol(int col, int what) {
    for (int i = 0; i < n_; i++) {
        if (marked_.at<char>(i, col) == what) {
            return i;
        }
    }
    return -1;
}

void KuhnMunkres::Run() {
    TrySimpleCase();
    while (!CheckIfOptimumIsFound()) {
        while (true) {
            auto point = FindUncoveredMinValPos();
            auto min_val = dm_.at<float>(point.y, point.x);
            if (min_val > 0) {
                UpdateDissimilarityMatrix(min_val);
            } else {
                marked_.at<char>(point.y, point.x) = kPrime;
                int col = FindInRow(point.y, kStar);
                if (col >= 0) {
                    is_row_visited_[point.y] = 1;
                    is_col_visited_[col] = 0;
                } else {
                    int count = 0;
                    points_[count] = point;

                    while (true) {
                        int row = FindInCol(points_[count].x, kStar);
                        if (row >= 0) {
                            count++;
                            points_[count] = cv::Point(points_[count - 1].x, row);
                            int col = FindInRow(points_[count].y, kPrime);
                            count++;
                            points_[count] = cv::Point(col, points_[count - 1].y);
                        } else {
                            break;
                        }
                    }

                    for (int i = 0; i < count + 1; i++) {
                        auto& mark = marked_.at<char>(points_[i].y, points_[i].x);
                        mark = mark == kStar ? 0 : kStar;
                    }

                    is_row_visited_ = std::vector<int>(n_, 0);
                    is_col_visited_ = std::vector<int>(n_, 0);

                    marked_.setTo(0, marked_ == kPrime);
                    break;
                }
            }
        }
    }
}

