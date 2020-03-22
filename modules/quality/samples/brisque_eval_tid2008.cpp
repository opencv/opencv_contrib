#include <fstream>

#include "opencv2/quality.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ml.hpp"

/*
BRISQUE evaluator using TID2008

TID2008:
http://www.ponomarenko.info/tid2008.htm

[1] N. Ponomarenko, V. Lukin, A. Zelensky, K. Egiazarian, M. Carli,
F. Battisti, "TID2008 - A Database for Evaluation of Full-Reference
Visual Quality Assessment Metrics", Advances of Modern
Radioelectronics, Vol. 10, pp. 30-45, 2009.

[2] N. Ponomarenko, F. Battisti, K. Egiazarian, J. Astola,  V. Lukin
"Metrics performance comparison for color image database", Fourth
international workshop on video processing and quality metrics
for consumer electronics, Scottsdale, Arizona, USA. Jan. 14-16, 2009, 6 p.

*/

namespace {

    // get ordinal ranks of data, fractional ranks assigned for ties.  O(n^2) time complexity
    //  optional binary predicate used for rank ordering of data elements, equality evaluation
    template <typename T, typename PrEqual = std::equal_to<T>, typename PrLess = std::less<T>>
    std::vector<float> rank_ordinal(const T* data, std::size_t sz, PrEqual&& eq = {}, PrLess&& lt = {})
    {
        std::vector<float> result{};
        result.resize(sz, -1);// set all ranks to -1, indicating not yet done

        int rank = 0;
        while (rank < (int)sz)
        {
            std::vector<int> els = {};

            for (int i = 0; i < (int)sz; ++i)
            {
                if (result[i] < 0)//not yet done
                {
                    if (!els.empty())// already found something
                    {
                        if (lt(data[i], data[els[0]]))//found a smaller item, replace existing
                        {
                            els.clear();
                            els.emplace_back(i);
                        }
                        else if (eq(data[i], data[els[0]]))// found a tie, add to vector
                            els.emplace_back(i);
                    }
                    else//els.empty==no current item, add it
                        els.emplace_back(i);
                }
            }

            CV_Assert(!els.empty());

            // compute, assign arithmetic mean
            const auto assigned_rank = (double)rank + (double)(els.size() - 1) / 2.;
            for (auto el : els)
                result[el] = (float)assigned_rank;

            rank += (int)els.size();
        }

        return result;
    }

    template <typename T>
    double pearson(const T* x, const T* y, std::size_t sz)
    {
        // based on https://www.geeksforgeeks.org/program-spearmans-rank-correlation/

        double sigma_x = {}, sigma_y = {}, sigma_xy = {}, sigma_xsq = {}, sigma_ysq = {};
        for (unsigned i = 0; i < sz; ++i)
        {
            sigma_x += x[i];
            sigma_y += y[i];
            sigma_xy += x[i] * y[i];
            sigma_xsq += x[i] * x[i];
            sigma_ysq += y[i] * y[i];
        }

        const double
            num = (sz * sigma_xy - sigma_x * sigma_y)
            , den = std::sqrt(((double)sz*sigma_xsq - sigma_x * sigma_x) * ((double)sz*sigma_ysq - sigma_y * sigma_y))
            ;
        return num / den;
    }

    // https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    template <typename T>
    double spearman(const T* x, const T* y, std::size_t sz)
    {
        // convert x, y to ranked integral vectors
        const auto
            x_rank = rank_ordinal(x, sz)
            , y_rank = rank_ordinal(y, sz)
            ;

        return pearson(x_rank.data(), y_rank.data(), sz);
    }

    // returns cv::Mat of columns: { Distortion Type ID, MOS_Score, Brisque_Score }
    cv::Mat tid2008_eval(const std::string& root, cv::quality::QualityBRISQUE& alg)
    {
        const std::string
            mos_with_names_path = root + "mos_with_names.txt"
            , dist_imgs_root = root + "distorted_images/"
            ;

        cv::Mat result(0, 3, CV_32FC1);

        // distortion types we care about
        static const std::vector<int> distortion_types = {
            10 // jpeg compression
            , 11 // jp2k compression
            , 1 // additive gaussian noise
            , 8 // gaussian blur
        };

        static const int
            num_images = 25 // [I01_ - I25_], file names
            , num_distortions = 4 // num distortions per image
            ;

        // load mos_with_names.  format: { mos, fname }
        std::vector<std::pair<float, std::string>> mos_with_names = {};

        std::ifstream mos_file(mos_with_names_path, std::ios::in);
        while (true)
        {
            std::string line;
            std::getline(mos_file, line);
            if (!line.empty())
            {
                const auto space_pos = line.find(' ');
                CV_Assert(space_pos != line.npos);

                mos_with_names.emplace_back(std::make_pair(
                    (float)std::atof(line.substr(0, space_pos).c_str())
                    , line.substr(space_pos + 1)
                ));
            }

            if (mos_file.peek() == EOF)
                break;
        };

        // foreach image
        //  foreach distortion type
        //      foreach distortion level
        //          distortion type id, mos value, brisque value

        for (int i = 0; i < num_images; ++i)
        {
            for (int ty = 0; ty < (int)distortion_types.size(); ++ty)
            {
                for (int dist = 1; dist <= num_distortions; ++dist)
                {
                    float mos_val = 0.f;

                    const std::string img_name = std::string("i")
                        + (((i + 1) < 10) ? "0" : "")
                        + std::to_string(i + 1)
                        + "_"
                        + ((distortion_types[ty] < 10) ? "0" : "")
                        + std::to_string(distortion_types[ty])
                        + "_"
                        + std::to_string(dist)
                        + ".bmp";

                    // find mos
                    bool found = false;
                    for (const auto& val : mos_with_names)
                    {
                        if (val.second == img_name)
                        {
                            found = true;
                            mos_val = val.first;
                            break;
                        }

                    }

                    CV_Assert(found);

                    // do brisque
                    auto img = cv::imread(dist_imgs_root + img_name);

                    // typeid, mos, brisque
                    cv::Mat row(1, 3, CV_32FC1);
                    row.at<float>(0) = (float)distortion_types[ty];
                    row.at<float>(1) = mos_val;
                    row.at<float>(2) = (float)alg.compute(img)[0];
                    result.push_back(row);

                }// dist
            }//ty
        }//i

        return result;
    }
}

inline void printHelp()
{
    using namespace std;
    cout << "    Demo of comparing BRISQUE quality assessment model against TID2008 database." << endl;
    cout << "    A. Mittal, A. K. Moorthy and A. C. Bovik, 'No Reference Image Quality Assessment in the Spatial Domain'" << std::endl << std::endl;
    cout << "    Usage: program <tid2008_path> <brisque_model_path> <brisque_range_path>" << endl << endl;
}

int main(int argc, const char * argv[])
{
    using namespace cv::ml;

    if (argc != 4)
    {
        printHelp();
        exit(1);
    }

    std::cout << "Evaluating database at " << argv[1] << "..." << std::endl;

    const auto ptr = cv::quality::QualityBRISQUE::create(argv[2], argv[3]);

    const auto data = tid2008_eval( std::string( argv[1] ) + "/", *ptr );

    // create contiguous mats
    const auto mos = data.col(1).clone();
    const auto brisque = data.col(2).clone();

    // calc srocc
    const auto cc = spearman((const float*)mos.data, (const float*)brisque.data, data.rows);
    std::cout << "SROCC: " << cc << std::endl;

    return 0;
}
