#include "precomp.hpp"

namespace cv{

    void adjustBrightness(Mat& img, double brightness_factor){
        CV_Assert(brightness_factor >= 0);

        int channels = img.channels();
        if(channels != 1 && channels != 3){
            CV_Error(Error::BadNumChannels, "Only support images with 1 or 3 channels");
        }
        img = img * brightness_factor;
    }

    void adjustContrast(Mat& img, double contrast_factor){
        CV_Assert(contrast_factor >= 0);

        int num_channels = img.channels();
        if(num_channels != 1 && num_channels != 3){
            CV_Error(Error::BadNumChannels, "Only support images with 1 or 3 channels");
        }
       Mat* channels = new Mat[num_channels];
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
        delete[] channels;
    }

    void adjustSaturation(Mat& img, double saturation_factor){
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

    void adjustHue(Mat& img, double hue_factor) {
        // FIXME: the range of hue_factor needs to be modified
        CV_Assert(hue_factor >= -1 && hue_factor <= 1);

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
}
