// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#include "upcean_decoder.hpp"
#include <vector>
#include <array>


namespace cv {
namespace barcode {

static constexpr int DIVIDE_PART = 16;

bool UPCEANDecoder::findGuardPatterns(const std::vector<uchar> &row, int rowOffset, uchar whiteFirst,
                                      const std::vector<int> &pattern, std::vector<int> counters,
                                      std::pair<int, int> &result)
{
    size_t patternLength = pattern.size();
    size_t width = row.size();
    uchar isWhite = whiteFirst ? WHITE : BLACK;
    rowOffset = (int) (std::find(row.cbegin() + rowOffset, row.cend(), isWhite) - row.cbegin());
    uint counterPosition = 0;
    int patternStart = rowOffset;
    for (uint x = rowOffset; x < width; x++)
    {
        if (row[x] == isWhite)
        {
            counters[counterPosition]++;
        }
        else
        {
            if (counterPosition == patternLength - 1)
            {
                if (patternMatch(counters, pattern, MAX_INDIVIDUAL_VARIANCE) < MAX_AVG_VARIANCE)
                {
                    result.first = patternStart;
                    result.second = x;
                    return true;
                }
                patternStart += counters[0] + counters[1];
                std::copy(counters.begin() + 2, counters.end(), counters.begin());
                counters[patternLength - 2] = 0;
                counters[patternLength - 1] = 0;
                counterPosition--;
            }
            else
            {
                counterPosition++;
            }
            counters[counterPosition] = 1;
            isWhite = (std::numeric_limits<uchar>::max() - isWhite);
        }
    }
    return false;
}

bool UPCEANDecoder::findStartGuardPatterns(const std::vector<uchar> &row, std::pair<int, int> &start_range)
{
    bool is_find = false;
    int next_start = 0;
    while (!is_find)
    {
        std::vector<int> guard_counters{0, 0, 0};
        if (!findGuardPatterns(row, next_start, BLACK, BEGIN_PATTERN(), guard_counters, start_range))
        {
            return false;
        }
        int start = start_range.first;
        next_start = start_range.second;
        int quiet_start = max(start - (next_start - start), 0);
        is_find = (quiet_start != start) &&
                  (std::find(std::begin(row) + quiet_start, std::begin(row) + start, BLACK) == std::begin(row) + start);
    }
    return true;
}

int UPCEANDecoder::decodeDigit(const std::vector<uchar> &row, std::vector<int> &counters, int rowOffset,
                               const std::vector<std::vector<int>> &patterns) const
{
    fillCounter(row, rowOffset, counters);
    int bestMatch = -1;
    int bestVariance = MAX_AVG_VARIANCE; // worst variance we'll accept
    int i = 0;
    for (const auto &pattern : patterns)
    {
        int variance = patternMatch(counters, pattern, MAX_INDIVIDUAL_VARIANCE);
        if (variance < bestVariance)
        {
            bestVariance = variance;
            bestMatch = i;
        }
        i++;
    }
    return std::max(-1, bestMatch);
    // -1 is Mismatch or means error.
}

/*Input a mat and it's position rect, return the decode result */

std::vector<Result> UPCEANDecoder::decodeImg(Mat &mat, const std::vector<std::vector<Point2f>> &pointsArrays) const
{
    CV_Assert(mat.channels() == 1);
    std::vector<Result> will_return;
    Mat gray = mat.clone();
    for (const auto &points : pointsArrays)
    {
        Mat bar_img;
        cutImage(gray, bar_img, points);
        if (bar_img.cols < 500)
        {
            resize(bar_img, bar_img, Size(500, bar_img.rows));
        }
        Result max_result = rectToResult(bar_img, points, DIVIDE_PART, false);
        will_return.push_back(max_result);
    }
    return will_return;
}

Result UPCEANDecoder::decodeImg(const Mat &gray, const vector<Point2f> &points) const
{
    return rectToResult(gray, points, DIVIDE_PART, false);
}

// input image is
Result UPCEANDecoder::rectToResult(const Mat &gray, const std::vector<Point2f> &points, int PART, int directly) const
{
    Mat blur;
    GaussianBlur(gray, blur, Size(0, 0), 25);
    addWeighted(gray, 2, blur, -1, 0, gray);
    gray.convertTo(gray, CV_8UC1, 1, -20);
    //imshow("preprocess", gray);
    threshold(gray, gray, 155, 255, THRESH_OTSU + THRESH_BINARY);
    std::map<std::string, int> result_vote;
    std::map<BarcodeType, int> format_vote;
    int vote_cnt = 0;
    int total_vote = 0;
    std::string max_result;
    BarcodeType max_format = BarcodeType::NONE;
    auto rect_size_height = norm(points[0] - points[1]);
    auto rect_size_width = norm(points[1] - points[2]);
    if (max(rect_size_height, rect_size_width) < this->bits_num)
    {
        return Result{string(), BarcodeType::NONE};
    }

    std::vector<std::pair<Point2i, Point2i>> begin_and_ends;
    const Size2i shape{gray.rows, gray.cols};
    linesFromRect(shape, true, PART, begin_and_ends);
    if (directly)
    {
        linesFromRect(shape, false, PART, begin_and_ends);
    }
    Result barcode;
    for (const auto &i: begin_and_ends)
    {
        const auto &begin = i.first;
        const auto &end = i.second;
        barcode = decodeLine(gray, begin, end);
        if (barcode.result.size() == this->digit_number)
        {
            total_vote++;
            result_vote[barcode.result] += 1;
            if (result_vote[barcode.result] > vote_cnt)
            {
                vote_cnt = result_vote[barcode.result];
                if ((vote_cnt << 1) > total_vote)
                {
                    max_result = barcode.result;
                    max_format = barcode.format;
                }
            }
        }
    }
    return Result(max_result, max_format);
}

Result UPCEANDecoder::decodeLine(const Mat &bar_img, const Point2i &begin, const Point2i &end) const
{
    Result result;
    std::vector<uchar> middle;
    LineIterator line = LineIterator(bar_img, begin, end);
    middle.reserve(line.count);
    for (int cnt = 0; cnt < line.count; cnt++, line++)
    {
        middle.push_back(bar_img.at<uchar>(line.pos()));
    }
    result = this->decode(middle, 0);
    if (result.result.size() != this->digit_number)
    {
        result = this->decode(std::vector<uchar>(middle.crbegin(), middle.crend()), 0);
    }
    return result;
}

/**@Prama img_size is the graph's size ,
* @Prama angle from [0,180)
* 0 is horizontal
* (0-90) top Left to bottom Right
* 90 vertical
* (90-180) lower left to upper right
* */
void UPCEANDecoder::linesFromRect(const Size2i &shape, int angle, int PART,
                                  std::vector<std::pair<Point2i, Point2i>> &results) const
{
    Point2i step = Point2i(shape.height / PART, 0);
    Point2i cbegin = Point2i(shape.height / 2, 0);
    Point2i cend = Point2i(shape.height / 2, shape.width - 1);
    if (angle)
    {
        step = Point2i(0, shape.width / PART);
        cbegin = Point2i(0, shape.width / 2);
        cend = Point2i(shape.height - 1, shape.width / 2);
    }
    results.reserve(results.size() + PART + 1);
    for (int i = 1; i <= (PART >> 1); ++i)
    {
        results.emplace_back(cbegin + i * step, cend + i * step);
        results.emplace_back(cbegin - i * step, cend - i * step);
    }
    results.emplace_back(cbegin, cend);
}


Result UPCEANDecoder::decodeImg(InputArray img) const
{
    auto Mat = img.getMat();
    auto gray = Mat.clone();
    constexpr int PART = 50;
    std::vector<Point2f> real_rect{
            Point2f(0, (float) Mat.rows), Point2f(0, 0), Point2f((float) Mat.cols, 0),
            Point2f((float) Mat.cols, (float) Mat.rows)};
    Result result = rectToResult(Mat, real_rect, PART, true);
    return result;
}


// right for A
const std::vector<std::vector<int>> &get_A_or_C_Patterns()
{
    static const std::vector<std::vector<int>> A_or_C_Patterns{{3, 2, 1, 1}, // 0
                                                               {2, 2, 2, 1}, // 1
                                                               {2, 1, 2, 2}, // 2
                                                               {1, 4, 1, 1}, // 3
                                                               {1, 1, 3, 2}, // 4
                                                               {1, 2, 3, 1}, // 5
                                                               {1, 1, 1, 4}, // 6
                                                               {1, 3, 1, 2}, // 7
                                                               {1, 2, 1, 3}, // 8
                                                               {3, 1, 1, 2}  // 9
    };
    return A_or_C_Patterns;
}

const std::vector<std::vector<int>> &get_AB_Patterns()
{
    static const std::vector<std::vector<int>> AB_Patterns = [] {
        constexpr uint offset = 10;
        auto AB_Patterns_inited = std::vector<std::vector<int>>(offset << 1, std::vector<int>(PATTERN_LENGTH, 0));
        std::copy(get_A_or_C_Patterns().cbegin(), get_A_or_C_Patterns().cend(), AB_Patterns_inited.begin());
        //AB pattern is
        for (uint i = 0; i < offset; ++i)
        {
            for (uint j = 0; j < PATTERN_LENGTH; ++j)
            {
                AB_Patterns_inited[i + offset][j] = AB_Patterns_inited[i][PATTERN_LENGTH - j - 1];
            }
        }
        return AB_Patterns_inited;
    }();
    return AB_Patterns;
}

const std::vector<int> &BEGIN_PATTERN()
{
    // it just need it's 1:1:1(black:white:black)
    static const std::vector<int> BEGIN_PATTERN_(3, 1);
    return BEGIN_PATTERN_;
}

const std::vector<int> &MIDDLE_PATTERN()
{
    // it just need it's 1:1:1:1:1(white:black:white:black:white)
    static const std::vector<int> MIDDLE_PATTERN_(5, 1);
    return MIDDLE_PATTERN_;
}

const std::array<char, 32> &FIRST_CHAR_ARRAY()
{
    // use array to simulation a Hashmap,
    // because the data's size is small,
    // use a hashmap or brute-force search 10 times both can not accept
    static const std::array<char, 32> pattern{
            '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x06', '\x00', '\x00', '\x00', '\x09', '\x00',
            '\x08', '\x03', '\x00', '\x00', '\x00', '\x00', '\x05', '\x00', '\x07', '\x02', '\x00', '\x00', '\x04',
            '\x01', '\x00', '\x00', '\x00', '\x00', '\x00'};
    // length is 32 to ensure the security
    // 0x00000 -> 0  -> 0
    // 0x11010 -> 26 -> 1
    // 0x10110 -> 22 -> 2
    // 0x01110 -> 14 -> 3
    // 0x11001 -> 25 -> 4
    // 0x10011 -> 19 -> 5
    // 0x00111 -> 7  -> 6
    // 0x10101 -> 21 -> 7
    // 0x01101 -> 13 -> 8
    // 0x01011 -> 11 -> 9
    // delete the 1-13's 2 number's bit,
    // it always be A which do not need to count.
    return pattern;
}
}

} // namespace cv