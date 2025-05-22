// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

namespace cv{
    namespace imgaug{
        extern RNG rng;

        static void getRandomCropParams(int h, int w, int th, int tw, int* x, int* y);
        static void getRandomResizedCropParams(int height, int width, const Vec2d& scale, const Vec2d& ratio, Rect& rect);
        static void getRandomErasingCropParams(int height, int width, const Vec2d& scale, const Vec2d& ratio, Rect& rect);
        static void getRandomAffineParams(const Size& size, const Vec2f& degrees, const Vec2f& translations, const Vec2f& scales, const Vec4f& shears, float* angle, float* translation_x, float* translation_y, float* scale, float* shear_x, float* shear_y);
        static void getAffineMatrix(Mat mat, float angle, float tx, float ty, float scale, float shear_x, float shear_y, int cx, int cy);

        void randomCrop(InputArray _src, OutputArray _dst, const Size& sz, const Vec4i& padding, bool pad_if_need, int fill, int padding_mode){
            Mat src = _src.getMat();

            if(padding != Vec4i()){
                copyMakeBorder(src, src, padding[0], padding[1], padding[2], padding[3], padding_mode, fill);
            }

            // pad the height if needed
            if(pad_if_need && src.rows < sz.height){
                Vec4i _padding = {sz.height - src.rows, sz.height - src.rows, 0, 0};
                copyMakeBorder(src, src, _padding[0], _padding[1], _padding[2], _padding[3], padding_mode, fill);
            }
            // pad the width if needed
            if(pad_if_need && src.cols < sz.width){
                Vec4i _padding = {0, 0, sz.width - src.cols, sz.width - src.cols};
                copyMakeBorder(src, src, _padding[0], _padding[1], _padding[2], _padding[3], padding_mode, fill);
            }

            int x, y;
            getRandomCropParams(src.rows, src.cols, sz.height, sz.width, &x, &y);

            Mat RoI(src, Rect(x, y, sz.width, sz.height));
            RoI.copyTo(_dst);

            // NOTE: inplace operation not works in converting from python to numpy
            // _dst.move(RoI);
        }


        static void getRandomCropParams(int h, int w, int th, int tw, int* x, int* y){
            if(h+1 < th || w+1 < tw){
                CV_Error( Error::StsBadSize, "The cropped size is larger than the image size" );
            }
            if(h == th && w == tw){
                (*x) = 0;
                (*y) = 0;
                return;
            }

            (*x) = rng.uniform(0, w-tw+1);
            (*y) = rng.uniform(0, h-th+1);

        }

        RandomCrop::RandomCrop(const Size& _sz, const Vec4i& _padding, bool _pad_if_need, int _fill, int _padding_mode):
                sz (_sz),
                padding (_padding),
                pad_if_need (_pad_if_need),
                fill (_fill),
                padding_mode (_padding_mode){};

        void RandomCrop::call(InputArray src, OutputArray dst) const{
            randomCrop(src, dst, sz, padding, pad_if_need, fill, padding_mode);
        }

        void randomFlip(InputArray _src, OutputArray _dst, int flipCode, double p){

            bool flag = rng.uniform(0., 1.) < p;

            Mat src = _src.getMat();

            if(!flag){
                _dst.move(src);
                return;
            }
            flip(src, src, flipCode);
            _dst.move(src);
        }

        RandomFlip::RandomFlip(int _flipCode, double _p):
                flipCode(_flipCode),
                p(_p){};

        void RandomFlip::call(InputArray src, OutputArray dst) const{
            randomFlip(src, dst);
        }

        Compose::Compose(std::vector<Ptr<Transform> >& _transforms):
                transforms(_transforms){};

        void Compose::call(InputArray _src, OutputArray _dst) const{
            Mat src = _src.getMat();

            for(auto it = transforms.begin(); it != transforms.end(); ++it){
                (*it)->call(src, src);
            }
            src.copyTo(_dst);
        }

        Resize::Resize(const Size& _sz, int _interpolation):
                sz(_sz),
                interpolation(_interpolation){};

        void Resize::call(InputArray src, OutputArray dst) const{
            resize(src, dst, sz, 0, 0, interpolation);
        }

        void centerCrop(InputArray _src, OutputArray _dst, const Size& size) {
            Mat src = _src.getMat();
            Mat padded(src);
            // pad the input image if needed
            if (size.width > src.cols || size.height > src.rows) {
                int top = size.height - src.rows > 0 ? static_cast<int>((size.height - src.rows) / 2) : 0;
                int bottom = size.height - src.rows > 0 ? static_cast<int>((size.height - src.rows) / 2) : 0;
                int left = size.width - src.cols > 0 ? static_cast<int>((size.width - src.cols) / 2) : 0;
                int right = size.width - src.cols > 0 ? static_cast<int>((size.width - src.cols) / 2) : 0;

                // fill with value 0
                copyMakeBorder(src, padded, top, bottom, left, right, BORDER_CONSTANT, 0);
            }

            int x = static_cast<int>((padded.cols - size.width) / 2);
            int y = static_cast<int>((padded.rows - size.height) / 2);

            Mat cropped(padded, Rect(x, y, size.width, size.height));
            _dst.move(cropped);
        }

        CenterCrop::CenterCrop(const Size& _size) :
                size(_size) {};

        void CenterCrop::call(InputArray src, OutputArray dst) const {
            centerCrop(src, dst, size);
        }

        Pad::Pad(const Vec4i& _padding, const Scalar& _fill, int _padding_mode) :
                padding(_padding),
                fill(_fill),
                padding_mode(_padding_mode) {};

        void Pad::call(InputArray src, OutputArray dst) const {
            copyMakeBorder(src, dst, padding[0], padding[1], padding[2], padding[3], padding_mode, fill);
        }

        void randomResizedCrop(InputArray _src, OutputArray _dst, const Size& size, const Vec2d& scale, const Vec2d& ratio, int interpolation) {
            // Ensure scale range and ratio range are valid
            CV_Assert(scale[0] <= scale[1] && ratio[0] <= ratio[1]);

            Mat src = _src.getMat();

            Rect crop_rect;
            getRandomResizedCropParams(src.rows, src.cols, scale, ratio, crop_rect);
            Mat cropped(src, Rect(crop_rect));
            resize(cropped, _dst, size, 0.0, 0.0, interpolation);
        }

        static void getRandomResizedCropParams(int height, int width, const Vec2d& scale, const Vec2d& ratio, Rect& rect) {
            // This implementation is inspired from the implementation in torchvision
            // https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py

            int area = height * width;

            for (int i = 0; i < 10; i++) {
                double target_area = rng.uniform(scale[0], scale[1]) * area;
                double aspect_ratio = rng.uniform(ratio[0], ratio[1]);

                int w = static_cast<int>(round(sqrt(target_area * aspect_ratio)));
                int h = static_cast<int>(round(sqrt(target_area / aspect_ratio)));

                if (w > 0 && w <= width && h > 0 && h <= height) {
                    rect.x = rng.uniform(0, width - w + 1);
                    rect.y = rng.uniform(0, height - h + 1);
                    rect.width = w;
                    rect.height = h;
                    return;
                }
            }

            // Center Crop
            double in_ratio = static_cast<double>(width) / height;
            if (in_ratio < ratio[0]) {
                rect.width = width;
                rect.height = static_cast<int> (round(width / ratio[0]));
            }
            else if (in_ratio > ratio[1]) {
                rect.height = height;
                rect.width = static_cast<int> (round(height * ratio[1]));
            }
            else {
                rect.width = width;
                rect.height = height;
            }
            rect.x = (width - rect.width) / 2;
            rect.y = (height - rect.height) / 2;

        }

        RandomResizedCrop::RandomResizedCrop(const Size& _size, const Vec2d& _scale, const Vec2d& _ratio, int _interpolation) :
                size(_size),
                scale(_scale),
                ratio(_ratio),
                interpolation(_interpolation) {};

        void RandomResizedCrop::call(InputArray src, OutputArray dst) const{
            randomResizedCrop(src, dst, size, scale, ratio, interpolation);
        }

        void colorJitter(InputArray _src, OutputArray _dst, const Vec2d& brightness, const Vec2d& contrast, const Vec2d& saturation, const Vec2d& hue){
            // TODO: check input values
            Mat src = _src.getMat();

            double brightness_factor = 1, contrast_factor = 1, saturation_factor = 1, hue_factor = 0;

            if(brightness != Vec2d())
                brightness_factor = rng.uniform(brightness[0], brightness[1]);
            if(contrast != Vec2d())
                contrast_factor = rng.uniform(contrast[0], contrast[1]);
            if(saturation != Vec2d())
                saturation_factor = rng.uniform(saturation[0], saturation[1]);
            if(hue != Vec2d())
                hue_factor = rng.uniform(hue[0], hue[1]);

            int order[4] = {1,2,3,4};
            std::random_shuffle(order, order+4);

            for(int i : order){
                if(i == 1 && brightness_factor != 1)
                    cv::adjustBrightness(src, brightness_factor);
                if(i == 2 && contrast_factor != 1)
                    cv::adjustContrast(src, contrast_factor);
                if(i == 3 && saturation_factor != 1)
                    cv::adjustSaturation(src, saturation_factor);
                if(i == 4 && hue_factor != 0)
                    cv::adjustHue(src, hue_factor);
            }

            _dst.move(src);
        }

        ColorJitter::ColorJitter(const Vec2d& _brightness, const Vec2d& _contrast, const Vec2d& _saturation,
                                 const Vec2d& _hue):
                brightness(_brightness),
                contrast(_contrast),
                saturation(_saturation),
                hue(_hue){};

        void ColorJitter::call(InputArray src, OutputArray dst) const{
            colorJitter(src, dst, brightness, contrast, saturation, hue);
        }

        void randomRotation(InputArray _src, OutputArray _dst, const Vec2d& degrees, int interpolation, const Point2f& center, const Scalar& fill){
            Mat src = _src.getMat();
            // TODO: check the validation of degrees
            double angle = rng.uniform(degrees[0], degrees[1]);

            Point2f pt(src.cols/2., src.rows/2.);
            if(center != Point2f()) pt = center;

            Mat r = getRotationMatrix2D(pt, angle, 1.0);

            // TODO: auto expand dst size to fit the rotated image
            warpAffine(src, _dst, r, src.size(), interpolation, BORDER_CONSTANT, fill);
        }

        RandomRotation::RandomRotation(const Vec2d& _degrees, int _interpolation, const Point2f& _center, const Scalar& _fill):
                degrees(_degrees),
                interpolation(_interpolation),
                center(_center),
                fill(_fill){};

        void RandomRotation::call(InputArray src, OutputArray dst) const{
            randomRotation(src, dst, degrees, interpolation, center, fill);
        }

        void grayScale(InputArray _src, OutputArray _dst, int num_channels){
            Mat src = _src.getMat();
            cvtColor(src, src, COLOR_BGR2GRAY);

            if(num_channels == 1){
                _dst.move(src);
                return;
            }
            Mat channels[3] = {src, src, src};
            merge(channels, 3, _dst);
        }

        GrayScale::GrayScale(int _num_channels):
                num_channels(_num_channels){};

        void GrayScale::call(InputArray _src, OutputArray _dst) const{
            grayScale(_src, _dst, num_channels);
        }

        void randomGrayScale(InputArray _src, OutputArray _dst, double p){
            if(rng.uniform(0.0, 1.0) < p){
                grayScale(_src, _dst, _src.channels());
                return;
            }
            Mat src = _src.getMat();
            _dst.move(src);
        }

        RandomGrayScale::RandomGrayScale(double _p):
                p(_p){};

        void RandomGrayScale::call(InputArray src, OutputArray dst) const{
            randomGrayScale(src, dst);
        }

        void randomErasing(InputArray _src, OutputArray _dst, double p, const Vec2d& scale, const Vec2d& ratio, const Scalar& value, bool inplace){
            // TODO: check the range of input values
            Mat src = _src.getMat();
            if(rng.uniform(0., 1.) >= p){
                _dst.move(src);
                return;
            }

            Rect roi;
            getRandomErasingCropParams(src.rows, src.cols, scale, ratio, roi);

            Mat erased(src, roi);

            int rows = erased.rows;
            int cols = erased.cols;
            int cn = erased.channels();
            for(int j=0; j<rows; j++){
                uchar* row = erased.ptr<uchar>(j);
                for(int i=0; i<cols; i++){
                    for(int c=0; c<cn; c++){
                        row[i * cn + c] = value[c];
                    }
                }
            }

            if(inplace)
                _dst.move(src);
            else
                src.copyTo(_dst);
        }

        static void getRandomErasingCropParams(int height, int width, const Vec2d& scale, const Vec2d& ratio, Rect& rect) {
            int area = height * width;

            for (int i = 0; i < 10; i++) {
                double target_area = rng.uniform(scale[0], scale[1]) * area;
                double aspect_ratio = rng.uniform(ratio[0], ratio[1]);

                int w = static_cast<int>(round(sqrt(target_area * aspect_ratio)));
                int h = static_cast<int>(round(sqrt(target_area / aspect_ratio)));

                if (w > 0 && w <= width && h > 0 && h <= height) {
                    rect.x = rng.uniform(0, width - w + 1);
                    rect.y = rng.uniform(0, height - h + 1);
                    rect.width = w;
                    rect.height = h;
                    return;
                }
            }

            // Center Crop
            double in_ratio = static_cast<double>(width) / height;
            if (in_ratio < ratio[0]) {
                rect.width = width;
                rect.height = static_cast<int> (round(width / ratio[0]));
            }
            else if (in_ratio > ratio[1]) {
                rect.height = height;
                rect.width = static_cast<int> (round(height * ratio[1]));
            }
            else {
                rect.width = width;
                rect.height = height;
            }
            rect.x = (width - rect.width) / 2;
            rect.y = (height - rect.height) / 2;
        }

        RandomErasing::RandomErasing(double _p, const Vec2d& _scale, const Vec2d& _ratio, const Scalar& _value, bool _inplace):
                p(_p),
                scale(_scale),
                ratio(_ratio),
                value(_value),
                inplace(_inplace){};

        void RandomErasing::call(InputArray src, OutputArray dst) const{
            randomErasing(src, dst, p, scale, ratio, value, inplace);
        }

        // NOTE: because Scalar contains 4 elements at most, normalize can only apply to image with channels no more than 4.
        Normalize::Normalize(const Scalar& _mean, const Scalar& _std):
                mean(_mean),
                std(_std){};

        void Normalize::call(InputArray _src, OutputArray _dst) const{
            Mat src = _src.getMat();

            _dst.create(src.size(), CV_32FC3);
            Mat dst = _dst.getMat();

            int cn = src.channels();
            std::vector<Mat> channels;
            split(src, channels);

            // normalize each channel to 0-1 first
            for(int i=0; i<cn; i++){
                Mat temp;
                channels[i].convertTo(temp, CV_32FC1, 1.f/255);
                temp = (temp - mean[i])/std[i];
                channels[i] = temp;
            }

            merge(channels, dst);
        }

        void gaussianBlur(InputArray src, OutputArray dst, const Size& kernel_size, const Vec2f& sigma){
            float sigmaX = rng.uniform(sigma[0], sigma[1]);
            cv::GaussianBlur(src, dst, kernel_size, sigmaX);
        }

        GaussianBlur::GaussianBlur(const Size& _kernel_size, const Vec2f& _sigma):
                kernel_size(_kernel_size),
                sigma(_sigma){};

        void GaussianBlur::call(InputArray src, OutputArray dst) const{
            gaussianBlur(src, dst, kernel_size, sigma);
        }

        void randomAffine(InputArray _src, OutputArray _dst, const Vec2f& degrees, const Vec2f& translations, const Vec2f& scales, const Vec4f& shears, int interpolation, const Scalar& fill, const Point2i& _center){
            /*
             * M = T * R * SHx * SHy
             * T is the translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
             * R is the rotation matrix: [ s * cos(a), s * sin(a), (1 - s * cos(a)) * cx - s * sin(a) * cy | -s * sin(a), s * cos(a), s * sin(a) * cx + (1 - s * cos(a)) * cy | 0, 0, 1]
             * in which cx and cy are center coordinates to keep the image rotate around the center, s is the scale factor, a is the rotation angle.
             * SHx and SHy are shear matrices: SHx(s) = [1, -tan(s), 0 | 0, 1, 0 | 0, 0, 1], SHy(s) = [1, 0, 0 | -tan(s), 1, 0 | 0, 0, 1]
             */

            // TODO: check the input value ranges
            Mat src = _src.getMat();
            // when center is default (-1,-1), make the rotation center located in the center of the image
            Point2i center;
            if(_center == Point2i(-1, -1)){
                center.x = static_cast<int>(src.cols / 2);
                center.y = static_cast<int>(src.rows / 2);
            }else{
                center = _center;
            }

            float angle, translation_x, translation_y, scale, shear_x, shear_y;
            getRandomAffineParams(src.size(), degrees, translations, scales, shears, &angle, &translation_x, &translation_y, &scale, &shear_x, &shear_y);

            Mat affine_matrix = Mat::eye(2, 3, CV_32F);

            // TODO: check whether equations are right
            getAffineMatrix(affine_matrix, angle, translation_x, translation_y, scale, shear_x, shear_y, center.x, center.y);
            warpAffine(src, src, affine_matrix, src.size(), interpolation, BORDER_CONSTANT, fill);
            _dst.move(src);
        }

        static void getAffineMatrix(Mat mat, float angle, float tx, float ty, float scale, float shear_x, float shear_y, int cx, int cy){
            float* data = mat.ptr<float>(0);

            // convert from degrees to radians
            angle = (float)(CV_PI * angle) / 180;
            shear_x = (float)(CV_PI * shear_x) / 180;
            shear_y = (float)(CV_PI * shear_y) / 180;

            data[0] = scale * cos(angle - shear_y) / cos(shear_y);
            data[1] = scale * (-cos(angle - shear_y) * tan(shear_x) / cos(shear_y) - sin(angle));
            data[3] = scale * sin(angle - shear_y) / cos(shear_y);
            data[4] = scale * (-sin(angle - shear_y) * tan(shear_x) / cos(shear_y) + cos(angle));
            data[2] = cx * (1-data[0]) + data[1] * (-cy) + tx;
            data[5] = cy * (1-data[4]) + data[3] * (-cx) + ty;
        }

        static void getRandomAffineParams(const Size& size, const Vec2f& degrees, const Vec2f& translations, const Vec2f& scales, const Vec4f& shears, float* angle, float* translation_x, float* translation_y, float* scale, float* shear_x, float* shear_y){

            if(degrees == Vec2f(0, 0)) {
                *angle = 0;
            }
            else{
                *angle = rng.uniform(degrees[0], degrees[1]);
            }

            if(translations == Vec2f(0, 0)) {
                *translation_x = 0;
                *translation_y = 0;
            }
            else{
                *translation_x = rng.uniform(-translations[0], translations[0]) * size.width;
                *translation_y = rng.uniform(-translations[1], translations[1]) * size.height;
            }

            if(scales == Vec2f(1, 1)) {
                *scale = 1;
            }
            else{
                *scale = rng.uniform(scales[0], scales[1]);
            }

            if(shears == Vec4f(0, 0, 0, 0)) {
                *shear_x = 0;
                *shear_y = 0;
            }
            else{
                *shear_x = rng.uniform(shears[0], shears[1]);
                *shear_y = rng.uniform(shears[2], shears[3]);
            }

        }

        RandomAffine::RandomAffine(const Vec2f& _degrees, const Vec2f& _translations, const Vec2f& _scales, const Vec4f& _shears, int _interpolation, const Scalar& _fill, const Point2i& _center):
                degrees(_degrees),
                translations(_translations),
                scales(_scales),
                shears(_shears),
                interpolation(_interpolation),
                fill(_fill),
                center(_center){};

        void RandomAffine::call(InputArray src, OutputArray dst) const{
            randomAffine(src, dst, degrees, translations, scales, shears, interpolation, fill, center);
        }
    }
}
