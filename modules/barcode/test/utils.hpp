// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CONTRIB_UTILS_HPP
#define OPENCV_CONTRIB_UTILS_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <map>

typedef std::vector<std::string> stringvec;
typedef std::map<std::string, std::string> datasetType;

namespace opencv_test{namespace {
inline stringvec explode(const std::string &s, const char &c)
{
    std::string buff;
    stringvec v;

    for (auto n:s)
    {
        if (n != c) { buff += n; }
        else if (n == c && !buff.empty())
        {
            v.push_back(buff);
            buff = "";
        }
    }
    if (!buff.empty()) { v.push_back(buff); }

    return v;
}

inline datasetType buildDataSet(std::string result_file_path)
{
    std::ifstream result_file;
    datasetType dataset;
    result_file.open(result_file_path);
    std::string line;
    if (result_file.is_open())
    {
        while (std::getline(result_file, line))
        {
            stringvec result = explode(line, ',');
            std::string filename = result[0];
            if (dataset.find(filename) == dataset.end())
            {
                dataset[filename] = result[1];
            }
        }
    }

    result_file.close();
    return dataset;
}
}}
#endif //OPENCV_CONTRIB_UTILS_HPP
