// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

namespace cv{
    namespace imgaug{
        extern RNG rng;

        namespace det{
            int clamp(int v, int lo, int hi);
            void rotate(int* x, int* y, int cx, int cy, double angle);

            Compose::Compose(std::vector<Ptr<Transform> >& _transforms):
                    transforms(_transforms){};

            void Compose::call(InputArray _src, OutputArray _dst, std::vector<cv::Rect>& target, std::vector<int>& labels) const{
                Mat src = _src.getMat();
                for(cv::imgaug::det::Transform* transform:transforms){
                    transform->call(src, src, target, labels);
                }
                src.copyTo(_dst);
            }

            RandomFlip::RandomFlip(int _flipCode, float _p):
                flipCode(_flipCode), p(_p)
            {
                if(p < 0 || p > 1){
                    CV_Error(Error::Code::StsBadArg, "probability p must be between range 0 and 1");
                }
            };

            void RandomFlip::call(InputArray _src, OutputArray _dst, std::vector<cv::Rect>& target, std::vector<int>& labels) const{
                CV_Assert(target.size() == labels.size());
                bool flag = rng.uniform(0., 1.) < p;

                Mat src = _src.getMat();
                if(!flag){
                    _dst.move(src);
                    return;
                }

                flipBoundingBox(target, src.size());
                flip(src, src, flipCode);
                _dst.move(src);
            }

            void RandomFlip::flipBoundingBox(std::vector<cv::Rect>& target, const Size& size) const{
                /*
                 * flipCode = 0 (flip vertically): (x', y') = (x, img.height - y - bbox.height)
                 * flipCode > 0 (flip horizontally): (x', y') = (img.width - x - bbox.width, y)
                 * flipCode < 0 (flip diagonally): (x', y') = (img.width - x - bbox.width, img.height - y - bbox.height)
                 */
                for(unsigned i = 0; i < target.size(); i++){
                    if(flipCode == 0){
                        target[i].y = size.height - target[i].y - target[i].height;
                    }else if(flipCode > 0){
                        target[i].x = size.width - target[i].x - target[i].width;
                    }else{
                        target[i].x = size.width - target[i].x - target[i].width;
                        target[i].y = size.height - target[i].y - target[i].height;
                    }
                }
            }

            Resize::Resize(const Size& _size, int _interpolation):
                    size(_size), interpolation(_interpolation){};

            void Resize::call(InputArray _src, OutputArray dst, std::vector<cv::Rect>& target, std::vector<int>& labels) const{
                CV_Assert(target.size() == labels.size());
                Mat src = _src.getMat();
                resize(src, dst, size, 0, 0, interpolation);
                resizeBoundingBox(target, src.size());
            }

            void Resize::resizeBoundingBox(std::vector<cv::Rect>& target, const Size& imgSize) const{
                for(unsigned i=0; i<target.size(); i++){
                    target[i].x = static_cast<double>(size.width) / imgSize.width * target[i].x;
                    target[i].y = static_cast<double>(size.height) / imgSize.height * target[i].y;
                    target[i].width = static_cast<double>(size.width) / imgSize.width * target[i].width;
                    target[i].height = static_cast<double>(size.height) / imgSize.height * target[i].height;
                }
            }

            Convert::Convert(int _code):
                    code(_code){};

            void Convert::call(InputArray src, OutputArray dst, std::vector<cv::Rect>& target, std::vector<int>& labels) const{
                CV_Assert(target.size() == labels.size());
                cvtColor(src, dst, code);
            }

            RandomTranslation::RandomTranslation(const cv::Vec2i& _translations, float _threshold):
                translations(_translations),
                threshold(_threshold){};


            void RandomTranslation::call(cv::InputArray _src, cv::OutputArray _dst, std::vector<cv::Rect> &bboxes, std::vector<int>& labels) const {
                CV_Assert(bboxes.size() == labels.size());
                int tx = rng.uniform(-translations[0], translations[0]);
                int ty = rng.uniform(-translations[1], translations[1]);

                Mat translation_matrix = Mat::eye(2, 3, CV_32F);
                float* data = translation_matrix.ptr<float>();
                data[0] = 1;
                data[1] = 0;
                data[2] = tx;
                data[3] = 0;
                data[4] = 1;
                data[5] = ty;

                cv::warpAffine(_src, _dst, translation_matrix, _src.size());
                translateBoundingBox(bboxes, labels, _src.size(), tx, ty);
            }


            void RandomTranslation::translateBoundingBox(std::vector<cv::Rect> &bboxes, std::vector<int> &labels, const cv::Size &imgSize, int tx, int ty) const {
                for(unsigned i=0; i < bboxes.size(); i++){
                    int x1 = clamp(bboxes[i].x + tx, 0, imgSize.width);
                    int y1 = clamp(bboxes[i].y + ty, 0, imgSize.height);
                    int x2 = clamp(bboxes[i].x + bboxes[i].width + tx, 0, imgSize.width);
                    int y2 = clamp(bboxes[i].y + bboxes[i].height + ty, 0, imgSize.height);
                    int w = x2 - x1;
                    int h = y2 - y1;
                    if((float)(w * h) / (bboxes[i].width * bboxes[i].height) < threshold){
                        bboxes.erase(bboxes.begin() + i);
                        labels.erase(labels.begin() + i);
                    }else{
                        bboxes[i].x = x1;
                        bboxes[i].y = y1;
                        bboxes[i].width = x2 - x1;
                        bboxes[i].height = y2 - y1;
                    }
                }
            }

            RandomRotation::RandomRotation(const cv::Vec2d &_angles, double _threshold):
                angles(_angles),
                threshold(_threshold){};

            void RandomRotation::call(cv::InputArray _src, cv::OutputArray _dst, std::vector<cv::Rect> &bboxes,
                                      std::vector<int> &labels) const {
                CV_Assert(bboxes.size() == labels.size());
                Mat src = _src.getMat();
                double angle = rng.uniform(angles[0], angles[1]);
                Mat rotation_matrix = getRotationMatrix2D(cv::Point2f(src.cols/2., src.rows/2.), angle, 1);
                warpAffine(src, _dst, rotation_matrix, src.size());

                Mat dst = _dst.getMat();
                rotateBoundingBoxes(bboxes, labels, angle, src.cols / 2, src.rows / 2, dst.size());
            }

            void RandomRotation::rotateBoundingBoxes(std::vector<cv::Rect> &bboxes, std::vector<int> &labels,
                                                     double angle, int cx, int cy, const Size& imgSize) const {
                angle = -angle * CV_PI / 180;

                for(unsigned i=0; i < bboxes.size(); i++){
                    int x1 = bboxes[i].x;
                    int y1 = bboxes[i].y;
                    int x2 = bboxes[i].x + bboxes[i].width;
                    int y2 = bboxes[i].y;
                    int x3 = bboxes[i].x;
                    int y3 = bboxes[i].y + bboxes[i].height;
                    int x4 = bboxes[i].x + bboxes[i].width;
                    int y4 = bboxes[i].y + bboxes[i].height;

                    // convert unit from degree to radius
                    // rotate the corners
                    rotate(&x1, &y1, cx, cy, angle);
                    rotate(&x2, &y2, cx, cy, angle);
                    rotate(&x3, &y3, cx, cy, angle);
                    rotate(&x4, &y4, cx, cy, angle);

                    // shrink the rotated corners to get an enclosing box
                    int x_min = min({x1, x2, x3, x4});
                    int y_min = min({y1, y2, y3, y4});
                    int x_max = max({x1, x2, x3, x4});
                    int y_max = max({y1, y2, y3, y4});

                    x_min = clamp(x_min, 0, imgSize.width);
                    y_min = clamp(y_min, 0, imgSize.height);
                    x_max = clamp(x_max, 0, imgSize.width);
                    y_max = clamp(y_max, 0, imgSize.height);

                    int w = x_max - x_min;
                    int h = y_max - y_min;

                    if((float)(w * h) / (bboxes[i].width * bboxes[i].height) < threshold){
                        bboxes.erase(bboxes.begin() + i);
                        labels.erase(labels.begin() + i);
                    }else{
                        bboxes[i].x = x_min;
                        bboxes[i].y = y_min;
                        bboxes[i].width = w;
                        bboxes[i].height = h;
                    }

                }
            }

            inline int clamp(int v, int lo, int hi){
                if(v < lo){
                    return lo;
                }
                if(v > hi){
                    return hi;
                }
                return v;
            }

            inline void rotate(int* x, int* y, int cx, int cy, double angle){
                // NOTE: when the unit of angle is degree instead of radius, the result may be incorrect.
                (*x) = (int)round(((*x) - cx) * cos(angle) - ((*y) - cy) * sin(angle) + cx);
                (*y) = (int)round(((*x) - cx) * sin(angle) + ((*y) - cy) * cos(angle) + cy);
            }
        }
    }
}