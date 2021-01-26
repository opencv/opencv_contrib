/*
Copyright 2020 ${ALL COMMITTERS}

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/


#ifndef __OPENCV_BARCODE_UTILS_HPP__
#define __OPENCV_BARCODE_UTILS_HPP__

#include <opencv2/imgproc.hpp>

namespace cv{
namespace barcode{

constexpr int OSTU = 0;
constexpr int HYBRID = 1;

void resize(Mat & src, Mat & dst);
void hybridPreprocess(Mat & src, Mat & dst);
void ostuPreprocess(Mat & src, Mat & dst);
void preprocess(Mat & src, Mat & dst, int mode);

}
}
#endif //__OPENCV_BARCODE_UTILS_HPP__
