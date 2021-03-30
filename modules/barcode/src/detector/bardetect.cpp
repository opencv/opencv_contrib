// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#include "../precomp.hpp"
#include "bardetect.hpp"


namespace cv {
namespace barcode {
static constexpr float PI = static_cast<float>(CV_PI);
static constexpr float HALF_PI = static_cast<float>(CV_PI / 2);

#define CALCULATE_SUM(ptr, result) \
    ptr += left_col + integral_cols * top_row;\
    top_left = static_cast<float>(*ptr);\
    ptr += right_col - left_col;\
    top_right = static_cast<float>(*ptr);\
    ptr += (bottom_row - top_row) * integral_cols;\
    bottom_right = static_cast<float>(*ptr);\
    ptr -= right_col - left_col;\
    bottom_left = static_cast<float>(*ptr);\
    ptr -= (bottom_row - top_row) * integral_cols;\
    ptr -= left_col + integral_cols * top_row;\
    result = (bottom_right - bottom_left - top_right + top_left);


inline bool Detect::isValidCoord(const Point &coord, const Size &limit)
{
    if ((coord.x < 0) || (coord.y < 0))
    {
        return false;
    }

    if ((unsigned) coord.x > (unsigned) (limit.width - 1) || ((unsigned) coord.y > (unsigned) (limit.height - 1)))
    {
        return false;
    }

    return true;
}

inline float Detect::computeOrientation(float y, float x)
{
    if (x >= 0)
    {
        return atan(y / x);
    }
    if (y >= 0)
    {
        return (atan(y / x) + PI);
    }
    return (atan(y / x) - PI);
}


void Detect::init(const Mat &src)
{
    const double min_side = std::min(src.size().width, src.size().height);
    if (min_side > 512.0)
    {
        purpose = SHRINKING;
        coeff_expansion = min_side / 512.0;
        width = cvRound(src.size().width / coeff_expansion);
        height = cvRound(src.size().height / coeff_expansion);
        Size new_size(width, height);
        resize(src, resized_barcode, new_size, 0, 0, INTER_AREA);
    }
//        else if (min_side < 512.0) {
//            purpose = ZOOMING;
//            coeff_expansion = 512.0 / min_side;
//            width = cvRound(src.size().width * coeff_expansion);
//            height = cvRound(src.size().height * coeff_expansion);
//            Size new_size(width, height);
//            resize(src, resized_barcode, new_size, 0, 0, INTER_LINEAR);
//
//        }
    else
    {
        purpose = UNCHANGED;
        coeff_expansion = 1.0;
        width = src.size().width;
        height = src.size().height;
        resized_barcode = src.clone();
    }
    medianBlur(resized_barcode, resized_barcode, 3);

}


void Detect::localization()
{

    localization_bbox.clear();
    bbox_scores.clear();

    // get integral image
    preprocess();
    float window_ratio = 0.01f;
    static constexpr float window_ratio_step = 0.02f;
    static constexpr int window_ratio_stepTimes = 6;
    int window_size;
    for (size_t i = 0; i < window_ratio_stepTimes; i++)
    {
        window_size = cvRound(min(width, height) * window_ratio);
        calConsistency(window_size);
        barcodeErode();
        regionGrowing(window_size);
        window_ratio += window_ratio_step;
    }

}


bool Detect::computeTransformationPoints()
{

    bbox_indices.clear();
    transformation_points.clear();
    transformation_points.reserve(bbox_indices.size());
    RotatedRect rect;
    Point2f temp[4];
    const float THRESHOLD_SCORE = float(width * height) / 500.f;
    dnn::NMSBoxes(localization_bbox, bbox_scores, THRESHOLD_SCORE, 0.1f, bbox_indices);

    for (const auto &bbox_index : bbox_indices)
    {
        rect = localization_bbox[bbox_index];
        if (purpose == ZOOMING)
        {
            rect.center /= coeff_expansion;
            rect.size.height /= static_cast<float>(coeff_expansion);
            rect.size.width /= static_cast<float>(coeff_expansion);
        }
        else if (purpose == SHRINKING)
        {
            rect.center *= coeff_expansion;
            rect.size.height *= static_cast<float>(coeff_expansion);
            rect.size.width *= static_cast<float>(coeff_expansion);
        }
        rect.points(temp);
        transformation_points.emplace_back(vector<Point2f>{temp[0], temp[1], temp[2], temp[3]});
    }

    return !transformation_points.empty();
}


void Detect::preprocess()
{
    Mat scharr_x(resized_barcode.size(), CV_32F), scharr_y(resized_barcode.size(), CV_32F), temp;
    Scharr(resized_barcode, scharr_x, CV_32F, 1, 0);
    Scharr(resized_barcode, scharr_y, CV_32F, 0, 1);
    // calculate magnitude of gradient, normalize and threshold
    magnitude(scharr_x, scharr_y, gradient_magnitude);
    threshold(gradient_magnitude, gradient_magnitude, 48, 1, THRESH_BINARY);
    gradient_magnitude.convertTo(gradient_magnitude, CV_8U);
    integral(gradient_magnitude, integral_edges, CV_32F);


    for (int y = 0; y < height; y++)
    {
        //pixels_position.clear();
        auto *const x_row = scharr_x.ptr<float_t>(y);
        auto *const y_row = scharr_y.ptr<float_t>(y);
        auto *const magnitude_row = gradient_magnitude.ptr<uint8_t>(y);
        for (int pos = 0; pos < width; pos++)
        {
            if (magnitude_row[pos] == 0)
            {
                x_row[pos] = 0;
                y_row[pos] = 0;
                continue;
            }
            if (x_row[pos] < 0)
            {
                x_row[pos] *= -1;
                y_row[pos] *= -1;
            }
        }
    }
    integral(scharr_x, temp, integral_x_sq, CV_32F, CV_32F);
    integral(scharr_y, temp, integral_y_sq, CV_32F, CV_32F);
    integral(scharr_x.mul(scharr_y), integral_xy, temp, CV_32F, CV_32F);
}


// Change consistency orientation edge_nums
// depend on width height integral_edges integral_x_sq integral_y_sq integral_xy
void Detect::calConsistency(int window_size)
{
    static constexpr float THRESHOLD_CONSISTENCY = 0.9f;
    int right_col, left_col, top_row, bottom_row;
    float xy, x_sq, y_sq, d, rect_area;
    const float THRESHOLD_AREA = float(window_size * window_size) * 0.42f;
    Size new_size(width / window_size, height / window_size);
    consistency = Mat(new_size, CV_8U), orientation = Mat(new_size, CV_32F), edge_nums = Mat(new_size, CV_32F);

    float top_left, top_right, bottom_left, bottom_right;
    int integral_cols = width + 1;
    const auto *edges_ptr = integral_edges.ptr<float_t>(), *x_sq_ptr = integral_x_sq.ptr<float_t>(), *y_sq_ptr = integral_y_sq.ptr<float_t>(), *xy_ptr = integral_xy.ptr<float_t>();
    for (int y = 0; y < new_size.height; y++)
    {
        auto *consistency_row = consistency.ptr<uint8_t>(y);
        auto *orientation_row = orientation.ptr<float_t>(y);
        auto *edge_nums_row = edge_nums.ptr<float_t>(y);
        if (y * window_size >= height)
        {
            continue;
        }
        top_row = y * window_size;
        bottom_row = min(height, (y + 1) * window_size);

        for (int pos = 0; pos < new_size.width; pos++)
        {

            // then calculate the column locations of the rectangle and set them to -1
            // if they are outside the matrix bounds
            if (pos * window_size >= width)
            {
                continue;
            }
            left_col = pos * window_size;
            right_col = min(width, (pos + 1) * window_size);

            //we had an integral image to count non-zero elements
            CALCULATE_SUM(edges_ptr, rect_area)
            if (rect_area < THRESHOLD_AREA)
            {
                // smooth region
                consistency_row[pos] = 0;
                continue;
            }

            CALCULATE_SUM(x_sq_ptr, x_sq)
            CALCULATE_SUM(y_sq_ptr, y_sq)
            CALCULATE_SUM(xy_ptr, xy)

            // get the values of the rectangle corners from the integral image - 0 if outside bounds
            d = sqrt((x_sq - y_sq) * (x_sq - y_sq) + 4 * xy * xy) / (x_sq + y_sq);
            if (d > THRESHOLD_CONSISTENCY)
            {
                consistency_row[pos] = 255;
                orientation_row[pos] = computeOrientation(x_sq - y_sq, 2 * xy) / 2.0f;
                edge_nums_row[pos] = rect_area;
            }
            else
            {
                consistency_row[pos] = 0;
            }

        }

    }
}

// will change localization_bbox bbox_scores
// will change consistency,
// depend on consistency orientation edge_nums
void Detect::regionGrowing(int window_size)
{
    static constexpr float LOCAL_THRESHOLD_CONSISTENCY = 0.95f, THRESHOLD_RADIAN =
            PI / 30, THRESHOLD_BLOCK_NUM = 35, LOCAL_RATIO = 0.5f, EXPANSION_FACTOR = 1.2f;
    Point pt_to_grow, pt;                       //point to grow

    float src_value;
    float cur_value;
    float edge_num;
    float rect_orientation;
    float sin_sum, cos_sum, counter;
    //grow direction
    static constexpr int DIR[8][2] = {{-1, -1},
                                      {0,  -1},
                                      {1,  -1},
                                      {1,  0},
                                      {1,  1},
                                      {0,  1},
                                      {-1, 1},
                                      {-1, 0}};
    vector<Point2f> growingPoints, growingImgPoints;
    for (int y = 0; y < consistency.rows; y++)
    {
        auto *consistency_row = consistency.ptr<uint8_t>(y);

        for (int x = 0; x < consistency.cols; x++)
        {
            if (consistency_row[x] == 0)
            {
                continue;
            }
            // flag
            consistency_row[x] = 0;
            growingPoints.clear();
            growingImgPoints.clear();

            pt = Point(x, y);
            cur_value = orientation.at<float_t>(pt);
            sin_sum = sin(2 * cur_value);
            cos_sum = cos(2 * cur_value);
            counter = 1;
            edge_num = edge_nums.at<float_t>(pt);
            growingPoints.push_back(pt);
            growingImgPoints.push_back(Point(pt));
            while (!growingPoints.empty())
            {
                pt = growingPoints.back();
                growingPoints.pop_back();
                src_value = orientation.at<float_t>(pt);

                //growing in eight directions
                for (auto i : DIR)
                {
                    pt_to_grow = Point(pt.x + i[0], pt.y + i[1]);

                    //check if out of boundary
                    if (!isValidCoord(pt_to_grow, consistency.size()))
                    {
                        continue;
                    }

                    if (consistency.at<uint8_t>(pt_to_grow) == 0)
                    {
                        continue;
                    }
                    cur_value = orientation.at<float_t>(pt_to_grow);
                    if (abs(cur_value - src_value) < THRESHOLD_RADIAN ||
                        abs(cur_value - src_value) > PI - THRESHOLD_RADIAN)
                    {
                        consistency.at<uint8_t>(pt_to_grow) = 0;
                        sin_sum += sin(2 * cur_value);
                        cos_sum += cos(2 * cur_value);
                        counter += 1;
                        edge_num += edge_nums.at<float_t>(pt_to_grow);
                        growingPoints.push_back(pt_to_grow);                 //push next point to grow back to stack
                        growingImgPoints.push_back(pt_to_grow);
                    }
                }
            }
            //minimum block num
            if (counter < THRESHOLD_BLOCK_NUM)
            {
                continue;
            }
            float local_consistency = (sin_sum * sin_sum + cos_sum * cos_sum) / counter / counter;
            // minimum local gradient orientation_arg consistency_arg
            if (local_consistency < LOCAL_THRESHOLD_CONSISTENCY)
            {
                continue;
            }
            RotatedRect minRect = minAreaRect(growingImgPoints);
            if (edge_num < minRect.size.area() * float(window_size * window_size) * LOCAL_RATIO ||
                counter < minRect.size.area() * LOCAL_RATIO)
            {
                continue;
            }
            const float local_orientation = computeOrientation(cos_sum, sin_sum) / 2.0f;
            // only orientation_arg is approximately equal to the rectangle orientation_arg
            rect_orientation = (minRect.angle) * PI / 180;
            if (minRect.size.width < minRect.size.height)
            {
                rect_orientation += (rect_orientation <= 0 ? HALF_PI : -HALF_PI);
                std::swap(minRect.size.width, minRect.size.height);
            }
            if (abs(local_orientation - rect_orientation) > THRESHOLD_RADIAN &&
                abs(local_orientation - rect_orientation) < PI - THRESHOLD_RADIAN)
            {
                continue;
            }
            minRect.angle = local_orientation * 180.f / PI;
            minRect.size.width *= static_cast<float>(window_size) * EXPANSION_FACTOR;
            minRect.size.height *= static_cast<float>(window_size);
            minRect.center.x = (minRect.center.x + 0.5f) * static_cast<float>(window_size);
            minRect.center.y = (minRect.center.y + 0.5f) * static_cast<float>(window_size);
            localization_bbox.push_back(minRect);
            bbox_scores.push_back(edge_num);

        }
    }
}

inline const std::array<Mat, 4> &getStructuringElement()
{
    static const std::array<Mat, 4> structuringElement{
            Mat_<uint8_t>{{3,   3},
                          {255, 0, 0, 0, 0, 0, 0, 0, 255}}, Mat_<uint8_t>{{3, 3},
                                                                          {0, 0, 255, 0, 0, 0, 255, 0, 0}},
            Mat_<uint8_t>{{3, 3},
                          {0, 0, 0, 255, 0, 255, 0, 0, 0}}, Mat_<uint8_t>{{3, 3},
                                                                          {0, 255, 0, 0, 0, 0, 0, 255, 0}}};
    return structuringElement;
}

// Change mat
void Detect::barcodeErode()
{
    static const std::array<Mat, 4> &structuringElement = getStructuringElement();
    Mat m0, m1, m2, m3;
    dilate(consistency, m0, structuringElement[0]);
    dilate(consistency, m1, structuringElement[1]);
    dilate(consistency, m2, structuringElement[2]);
    dilate(consistency, m3, structuringElement[3]);
    int sum;
    for (int y = 0; y < consistency.rows; y++)
    {
        auto consistency_row = consistency.ptr<uint8_t>(y);
        auto m0_row = m0.ptr<uint8_t>(y);
        auto m1_row = m1.ptr<uint8_t>(y);
        auto m2_row = m2.ptr<uint8_t>(y);
        auto m3_row = m3.ptr<uint8_t>(y);

        for (int pos = 0; pos < consistency.cols; pos++)
        {
            if (consistency_row[pos] != 0)
            {
                sum = m0_row[pos] + m1_row[pos] + m2_row[pos] + m3_row[pos];
                //more than 2 group
                consistency_row[pos] = sum > 600 ? 255 : 0;
            }
        }
    }
}
}
}