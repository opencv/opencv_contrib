#ifndef OPENCV_AUG_FUNCTIONAL_HPP
#define OPENCV_AUG_FUNCTIONAL_HPP
#include <opencv2/core.hpp>
#include <vector>

namespace cv {

//    void blend(Mat& img1, Mat& img2, float ratio){
//
//    }

    static void adjust_brightness(Mat& img, double brightness_factor){
        CV_Assert(brightness_factor >= 0);

        int channels = img.channels();
        if(channels != 1 and channels != 3){
            CV_Error(Error::BadNumChannels, "Only support images with 1 or 3 channels");
        }
        img = img * brightness_factor;
        // NOTE: Can substitute for-loop with matrix multiplication for better efficiency?
//        int nc = channels * img.cols;
//        for(int j=0; j<img.rows; j++){
//            uchar* data = img.ptr<uchar>(j);
//            for(int i=0; i<nc; i++){
//                data[i] = static_cast<uchar>( data[i] * brightness_factor);
//            }
//        }
    }

    static void adjust_contrast(Mat& img, double contrast_factor){

        CV_Assert(contrast_factor >= 0);

        int num_channels = img.channels();
        if(num_channels != 1 && num_channels != 3){
            CV_Error(Error::BadNumChannels, "Only support images with 1 or 3 channels");
        }
        Mat channels[num_channels];
        split(img, channels);
        std::vector<Mat> new_channels;
        for(int i=0; i < num_channels; i++){
            Mat& channel = channels[i];
            Scalar avg = mean(channel);
            Mat avg_mat(channel.size(), channel.type(), avg);
            Mat new_channel = contrast_factor * channel + (1-contrast_factor) * avg_mat;
            new_channels.push_back(new_channel);
        }
        merge(new_channels, img);
    }

    static void adjust_saturation(Mat& img, double saturation_factor){
        CV_Assert(saturation_factor >= 0);

        int num_channels = img.channels();
        if(num_channels != 1 && num_channels != 3){
            CV_Error(Error::BadNumChannels, "Only support images with 1 or 3 channels");
        }
        if(img.channels() == 1) return;
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        std::vector<Mat> gray_arrays = {gray, gray, gray};
        merge(gray_arrays, gray);
        img = saturation_factor * img + (1-saturation_factor) * gray;
    }

    static void adjust_hue(Mat& img, double hue_factor) {
        // FIXME: the range of hue_factor needs to be modified
        CV_Assert(hue_factor >= 0);

        int num_channels = img.channels();
        if (num_channels != 1 && num_channels != 3) {
            CV_Error(Error::BadNumChannels, "Only support images with 1 or 3 channels");
        }

        if (num_channels == 1) return;
        int hue_shift = saturate_cast<int> (hue_factor * 180);
        Mat hsv;
        cvtColor(img, hsv, COLOR_BGR2HSV);
        for (int j=0; j<img.rows; j++){
            for (int i=0; i<img.cols; i++){
                int h = hsv.at<Vec3b>(j, i)[0];
                if(h + hue_shift > 180)
                    h =  h + hue_shift - 180;
                else
                    h = h + hue_shift;
                hsv.at<Vec3b>(j, i)[0] = h;
            }
        }
        cvtColor(hsv, img, COLOR_HSV2BGR);
    }
};

#endif