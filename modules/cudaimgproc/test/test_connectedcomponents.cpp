// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

namespace opencv_test {
namespace {

// This function force a row major order for the labels
template <typename LabelT>
void normalize_labels_impl(Mat& labels) {
    std::map<LabelT, LabelT> map_new_labels;
    LabelT i_max_new_label = 0;

    for (int r = 0; r < labels.rows; ++r) {
        LabelT* const mat_row = labels.ptr<LabelT>(r);
        for (int c = 0; c < labels.cols; ++c) {
            LabelT iCurLabel = mat_row[c];
            if (iCurLabel > 0) {
                if (map_new_labels.find(iCurLabel) == map_new_labels.end()) {
                    map_new_labels[iCurLabel] = ++i_max_new_label;
                }
                mat_row[c] = map_new_labels.at(iCurLabel);
            }
        }
    }
}

void normalize_labels(Mat& labels) {

    int type = labels.type();
    int depth = type & CV_MAT_DEPTH_MASK;
    int chans = 1 + (type >> CV_CN_SHIFT);

    CV_Assert(chans == 1);
    CV_Assert(depth == CV_16U || depth == CV_16S || depth == CV_32S);

    switch (depth) {
    case CV_16U: normalize_labels_impl<ushort>(labels); break;
    case CV_16S: normalize_labels_impl<short>(labels); break;
    case CV_32S: normalize_labels_impl<int>(labels); break;
    default:     CV_Assert(0);
    }
}


////////////////////////////////////////////////////////
// ConnectedComponents

PARAM_TEST_CASE(ConnectedComponents, cv::cuda::DeviceInfo, int, int, cv::cuda::ConnectedComponentsAlgorithmsTypes)
{
    cv::cuda::DeviceInfo devInfo;
    int connectivity;
    int ltype;
    cv::cuda::ConnectedComponentsAlgorithmsTypes algo;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        connectivity = GET_PARAM(1);
        ltype = GET_PARAM(2);
        algo = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(ConnectedComponents, Chessboard_Even)
{
    std::initializer_list<int> sizes{ 16, 16 };
    cv::Mat1b input;
    cv::Mat1i correct_output_int;
    cv::Mat correct_output;

    // Chessboard image with even number of rows and cols
    // Note that this is the maximum number of labels for 4-way connectivity
    {
        input = cv::Mat1b(sizes, {
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
            });

        if (connectivity == 8) {
            correct_output_int = cv::Mat1i(sizes, {
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
                });
        }
        else {
            correct_output_int = cv::Mat1i(sizes, {
                1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0,
                0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16,
                17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0,
                0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0, 32,
                33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 40, 0,
                0, 41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47, 0, 48,
                49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 56, 0,
                0, 57, 0, 58, 0, 59, 0, 60, 0, 61, 0, 62, 0, 63, 0, 64,
                65, 0, 66, 0, 67, 0, 68, 0, 69, 0, 70, 0, 71, 0, 72, 0,
                0, 73, 0, 74, 0, 75, 0, 76, 0, 77, 0, 78, 0, 79, 0, 80,
                81, 0, 82, 0, 83, 0, 84, 0, 85, 0, 86, 0, 87, 0, 88, 0,
                0, 89, 0, 90, 0, 91, 0, 92, 0, 93, 0, 94, 0, 95, 0, 96,
                97, 0, 98, 0, 99, 0, 100, 0, 101, 0, 102, 0, 103, 0, 104, 0,
                0, 105, 0, 106, 0, 107, 0, 108, 0, 109, 0, 110, 0, 111, 0, 112,
                113, 0, 114, 0, 115, 0, 116, 0, 117, 0, 118, 0, 119, 0, 120, 0,
                0, 121, 0, 122, 0, 123, 0, 124, 0, 125, 0, 126, 0, 127, 0, 128
                });
        }
    }

    correct_output_int.convertTo(correct_output, CV_MAT_DEPTH(ltype));

    cv::Mat labels;
    cv::Mat diff;

    cv::cuda::GpuMat d_input;
    cv::cuda::GpuMat d_labels;
    d_input.upload(input);
    EXPECT_NO_THROW(cv::cuda::connectedComponents(d_input, d_labels, connectivity, ltype, algo));
    d_labels.download(labels);
    normalize_labels(labels);

    diff = labels != correct_output;
    EXPECT_EQ(cv::countNonZero(diff), 0);

}

CUDA_TEST_P(ConnectedComponents, Chessboard_Odd)
{
    std::initializer_list<int> sizes{ 15, 15 };
    cv::Mat1b input;
    cv::Mat1i correct_output_int;
    cv::Mat correct_output;

    // Chessboard image with even number of rows and cols
    // Note that this is the maximum number of labels for 4-way connectivity
    {
        input = Mat1b(sizes, {
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
            });

        if (connectivity == 8) {
            correct_output_int = Mat1i(sizes, {
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
                });
        }
        else {
            correct_output_int = Mat1i(sizes, {
                1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8,
                0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0,
                16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23,
                0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0,
                31, 0, 32, 0, 33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38,
                0, 39, 0, 40, 0, 41, 0, 42, 0, 43, 0, 44, 0, 45, 0,
                46, 0, 47, 0, 48, 0, 49, 0, 50, 0, 51, 0, 52, 0, 53,
                0, 54, 0, 55, 0, 56, 0, 57, 0, 58, 0, 59, 0, 60, 0,
                61, 0, 62, 0, 63, 0, 64, 0, 65, 0, 66, 0, 67, 0, 68,
                0, 69, 0, 70, 0, 71, 0, 72, 0, 73, 0, 74, 0, 75, 0,
                76, 0, 77, 0, 78, 0, 79, 0, 80, 0, 81, 0, 82, 0, 83,
                0, 84, 0, 85, 0, 86, 0, 87, 0, 88, 0, 89, 0, 90, 0,
                91, 0, 92, 0, 93, 0, 94, 0, 95, 0, 96, 0, 97, 0, 98,
                0, 99, 0, 100, 0, 101, 0, 102, 0, 103, 0, 104, 0, 105, 0,
                106, 0, 107, 0, 108, 0, 109, 0, 110, 0, 111, 0, 112, 0, 113
                });
        }
    }

    correct_output_int.convertTo(correct_output, CV_MAT_DEPTH(ltype));

    cv::Mat labels;
    cv::Mat diff;

    cv::cuda::GpuMat d_input;
    cv::cuda::GpuMat d_labels;
    d_input.upload(input);
    EXPECT_NO_THROW(cv::cuda::connectedComponents(d_input, d_labels, connectivity, ltype, algo));
    d_labels.download(labels);
    normalize_labels(labels);

    diff = labels != correct_output;
    EXPECT_EQ(cv::countNonZero(diff), 0);

}

CUDA_TEST_P(ConnectedComponents, Maxlabels_8conn_Even)
{
    std::initializer_list<int> sizes{ 16, 16 };
    cv::Mat1b input;
    cv::Mat1i correct_output_int;
    cv::Mat correct_output;

    // Chessboard image with even number of rows and cols
    // Note that this is the maximum number of labels for 4-way connectivity
    {
        input = Mat1b(sizes, {
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            });

        correct_output_int = Mat1i(sizes, {
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0, 32, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 40, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47, 0, 48, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 56, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            57, 0, 58, 0, 59, 0, 60, 0, 61, 0, 62, 0, 63, 0, 64, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            });

    }

    correct_output_int.convertTo(correct_output, CV_MAT_DEPTH(ltype));

    cv::Mat labels;
    cv::Mat diff;

    cv::cuda::GpuMat d_input;
    cv::cuda::GpuMat d_labels;
    d_input.upload(input);
    EXPECT_NO_THROW(cv::cuda::connectedComponents(d_input, d_labels, connectivity, ltype, algo));
    d_labels.download(labels);
    normalize_labels(labels);

    diff = labels != correct_output;
    EXPECT_EQ(cv::countNonZero(diff), 0);

}

CUDA_TEST_P(ConnectedComponents, Maxlabels_8conn_Odd)
{
    std::initializer_list<int> sizes{ 15, 15 };
    cv::Mat1b input;
    cv::Mat1i correct_output_int;
    cv::Mat correct_output;

    // Chessboard image with even number of rows and cols
    // Note that this is the maximum number of labels for 4-way connectivity
    {
        input = Mat1b(sizes, {
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
            });

        correct_output_int = Mat1i(sizes, {
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0, 32,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 40,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47, 0, 48,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 56,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            57, 0, 58, 0, 59, 0, 60, 0, 61, 0, 62, 0, 63, 0, 64
            });

    }

    correct_output_int.convertTo(correct_output, CV_MAT_DEPTH(ltype));

    cv::Mat labels;
    cv::Mat diff;

    cv::cuda::GpuMat d_input;
    cv::cuda::GpuMat d_labels;
    d_input.upload(input);
    EXPECT_NO_THROW(cv::cuda::connectedComponents(d_input, d_labels, connectivity, ltype, algo));
    d_labels.download(labels);
    normalize_labels(labels);

    diff = labels != correct_output;
    EXPECT_EQ(cv::countNonZero(diff), 0);

}

CUDA_TEST_P(ConnectedComponents, Single_Row)
{
    std::initializer_list<int> sizes{ 1, 15 };
    cv::Mat1b input;
    cv::Mat1i correct_output_int;
    cv::Mat correct_output;

    // Chessboard image with even number of rows and cols
    // Note that this is the maximum number of labels for 4-way connectivity
    {
        input = Mat1b(sizes, { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
        correct_output_int = Mat1i(sizes, { 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8 });
    }

    correct_output_int.convertTo(correct_output, CV_MAT_DEPTH(ltype));

    cv::Mat labels;
    cv::Mat diff;

    cv::cuda::GpuMat d_input;
    cv::cuda::GpuMat d_labels;
    d_input.upload(input);
    EXPECT_NO_THROW(cv::cuda::connectedComponents(d_input, d_labels, connectivity, ltype, algo));
    d_labels.download(labels);
    normalize_labels(labels);

    diff = labels != correct_output;
    EXPECT_EQ(cv::countNonZero(diff), 0);

}

CUDA_TEST_P(ConnectedComponents, Single_Column)
{
    std::initializer_list<int> sizes{ 15, 1 };
    cv::Mat1b input;
    cv::Mat1i correct_output_int;
    cv::Mat correct_output;

    // Chessboard image with even number of rows and cols
    // Note that this is the maximum number of labels for 4-way connectivity
    {
        input = Mat1b(sizes, { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
        correct_output_int = Mat1i(sizes, { 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8 });
    }

    correct_output_int.convertTo(correct_output, CV_MAT_DEPTH(ltype));

    cv::Mat labels;
    cv::Mat diff;

    cv::cuda::GpuMat d_input;
    cv::cuda::GpuMat d_labels;
    d_input.upload(input);
    EXPECT_NO_THROW(cv::cuda::connectedComponents(d_input, d_labels, connectivity, ltype, algo));
    d_labels.download(labels);
    normalize_labels(labels);

    diff = labels != correct_output;
    EXPECT_EQ(cv::countNonZero(diff), 0);

}


CUDA_TEST_P(ConnectedComponents, Concentric_Circles)
{
    string img_path = cvtest::TS::ptr()->get_data_path() + "connectedcomponents/concentric_circles.png";
    string exp_path = cvtest::TS::ptr()->get_data_path() + "connectedcomponents/ccomp_exp.png";

    Mat img = imread(img_path, 0);
    EXPECT_FALSE(img.empty());

    Mat exp = imread(exp_path, 0);
    EXPECT_FALSE(exp.empty());

    Mat labels;
    exp.convertTo(exp, ltype);

    GpuMat d_img;
    GpuMat d_labels;
    d_img.upload(img);

    EXPECT_NO_THROW(cv::cuda::connectedComponents(d_img, d_labels, connectivity, ltype, algo));

    d_labels.download(labels);

    normalize_labels(labels);

    Mat diff = labels != exp;
    EXPECT_EQ(cv::countNonZero(diff), 0);

}


INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, ConnectedComponents, testing::Combine(
    ALL_DEVICES,
    testing::Values(8),
    testing::Values(CV_32S),
    testing::Values(cv::cuda::CCL_DEFAULT, cv::cuda::CCL_BKE)
));


}
} // namespace
#endif // HAVE_CUDA
