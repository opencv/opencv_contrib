#include "../precomp.hpp"
#include "slic.hpp"
using namespace std;

#define vec2db vector<vector<bool> >

namespace cv{ namespace hfs{ namespace slic{

void cSLIC::init_data(Mat image_) {
    image = image_;
    lab = cvt_img_space();
    map_size[0] = (int)ceil((float)lab.cols / (float)spixel_size);
    map_size[1] = (int)ceil((float)lab.rows / (float)spixel_size);

    // initialize distance normalizer
    // normalizing factors
    max_color_dist = 15.0f / (1.7321f * 128);
    max_color_dist *= max_color_dist;
    max_xy_dist = 1.0f / (2.0f * spixel_size * spixel_size);
    // initialize index map
    idx_img = vector<int>(lab.rows * lab.cols);
    for (int i = 0; i < lab.rows * lab.cols; ++i) {
        idx_img[i] = -1;
    }
    // initialize super pixel list
    spixel_list = vector<cSpixelInfo>(map_size[0] * map_size[1]);

    // initialize cluster
    for (int x = 0; x < map_size[0]; ++x) {
        for (int y = 0; y < map_size[1]; ++y) {
            int cluster_idx = y * map_size[0] + x;

            int img_x = x * spixel_size + spixel_size / 2;
            int img_y = y * spixel_size + spixel_size / 2;

            img_x = img_x >= lab.cols ? (x * spixel_size + lab.cols) / 2 : img_x;
            img_y = img_y >= lab.rows ? (y * spixel_size + lab.rows) / 2 : img_y;

            spixel_list[cluster_idx].id = cluster_idx;
            spixel_list[cluster_idx].center = Vec2f((float)img_x, (float) img_y);

            spixel_list[cluster_idx].color_info = lab.at<Vec3f>(img_y, img_x);
            spixel_list[cluster_idx].num_pixels = 0;
        }
    }
}

Mat cSLIC::cvt_img_space() {
    float epsilon = 0.008856f;	//actual CIE standard
    float kappa = 903.3f;		//actual CIE standard

    float Xr = 0.950456f;	//reference white
	float Yr = 1.0f;		//reference white
	float Zr = 1.088754f;	//reference white

    Mat lab_ = Mat(image.size(), CV_32FC3);

    for(int i = 0; i < image.rows; ++i) {
        for(int j = 0; j < image.cols; ++j){
            Vec3b pix_in = image.at<Vec3b>(i,j);
            float _r = (float)pix_in[0] / 255;
	        float _g = (float)pix_in[1] / 255;
	        float _b = (float)pix_in[2] / 255;

	        if (_b <= 0.04045f)    _b = _b / 12.92f;
	        else                   _b = pow((_b + 0.055f) / 1.055f, 2.4f);
	        if (_g <= 0.04045f)    _g = _g / 12.92f;
	        else                   _g = pow((_g + 0.055f) / 1.055f, 2.4f);
	        if (_r <= 0.04045f)    _r = _r / 12.92f;
	        else                   _r = pow((_r + 0.055f) / 1.055f, 2.4f);

	        float x = _r*0.4124564f + _g*0.3575761f + _b*0.1804375f;
	        float y = _r*0.2126729f + _g*0.7151522f + _b*0.0721750f;
	        float z = _r*0.0193339f + _g*0.1191920f + _b*0.9503041f;


	        float xr = x / Xr;
	        float yr = y / Yr;
	        float zr = z / Zr;

	        float fx, fy, fz;
	        if (xr > epsilon)	fx = pow(xr, 1.0f / 3.0f);
	        else				fx = (kappa*xr + 16.0f) / 116.0f;
	        if (yr > epsilon)	fy = pow(yr, 1.0f / 3.0f);
	        else				fy = (kappa*yr + 16.0f) / 116.0f;
	        if (zr > epsilon)	fz = pow(zr, 1.0f / 3.0f);
	        else				fz = (kappa*zr + 16.0f) / 116.0f;

	        lab_.at<Vec3f>(i, j)[0] = 116.0f*fy - 16.0f;
	        lab_.at<Vec3f>(i, j)[1]  = 500.0f*(fx - fy);
	        lab_.at<Vec3f>(i, j)[2]  = 200.0f*(fy - fz);
        }
    }

    return lab_;
}

float cSLIC::compute_dist(Point pix, cSpixelInfo center_info) {
    Vec3f color = lab.at<Vec3f>(pix.y, pix.x);
    float dcolor =
        (color[0] - center_info.color_info[0])*(color[0] - center_info.color_info[0])
        + (color[1] - center_info.color_info[1])*(color[1] - center_info.color_info[1])
        + (color[2] - center_info.color_info[2])*(color[2] - center_info.color_info[2]);

    float dxy =
        (float)((pix.x - center_info.center[0]) * (pix.x - center_info.center[0])
        + (pix.y - center_info.center[1]) * (pix.y - center_info.center[1]));

    float retval =
        dcolor * max_color_dist + spatial_weight * dxy * max_xy_dist;
    return sqrtf(retval);
}

vector<int> cSLIC::generate_superpixels(Mat image_, int spixel_size_, float spatial_weight_) {
    spixel_size = spixel_size_;
    spatial_weight = spatial_weight_;

    init_data(image_);
    find_association();
    for (int iter = 0; iter < 5; ++iter) {
        update_cluster_center();
        find_association();
    }
    enforce_connect(2, 16);
    enforce_connect(2, 16);
    enforce_connect(1, 5);
    enforce_connect(1, 5);
    return idx_img;
}

void cSLIC::find_association() {
    for (int y = 0; y < lab.rows; ++y) {
        for (int x = 0; x < lab.cols; ++x) {

            int ctr_x = x / spixel_size;
            int ctr_y = y / spixel_size;

            int idx = y * lab.cols + x;

            int minidx = -1;
            float dist = FLT_MAX;

            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int ctr_x_check = ctr_x + j;
                    int ctr_y_check = ctr_y + i;
                    if (ctr_x_check >= 0 && ctr_y_check >= 0 &&
                        ctr_x_check < map_size[0] && ctr_y_check < map_size[1]) {
                        int ctr_idx = ctr_y_check*map_size[0] + ctr_x_check;
                        float cdist = compute_dist(Point(x, y), spixel_list[ctr_idx]);
                        if (cdist < dist) {
                            dist = cdist;
                            minidx = spixel_list[ctr_idx].id;
                        }
                    }
                }
            }
            if (minidx >= 0) {
                idx_img[idx] = minidx;
            }
        }
    }
}

void cSLIC::update_cluster_center() {

    for (int i = 0; i < map_size[0] * map_size[1]; ++i) {
        spixel_list[i].center = Vec2f(0.0f, 0.0f);
        spixel_list[i].color_info = Vec3f(0.0f, 0.0f, 0.0f);
        spixel_list[i].num_pixels = 0;
    }

    for (int i = 0; i < lab.rows; ++i) {
        for (int j = 0; j < lab.cols; ++j) {
            int idx = i * lab.cols + j;
            spixel_list[idx_img[idx]].center += Vec2f((float)j, (float)i);
            spixel_list[idx_img[idx]].color_info += lab.at<Vec3f>(i, j);
            spixel_list[idx_img[idx]].num_pixels += 1;
        }
    }

    for (int i = 0; i < map_size[0] * map_size[1]; ++i) {
        if (spixel_list[i].num_pixels != 0) {
            spixel_list[i].center /= spixel_list[i].num_pixels;
            spixel_list[i].color_info /= spixel_list[i].num_pixels;
        }
        else {
            spixel_list[i].center = Vec2f(-100.0f, -100.0f);
            spixel_list[i].color_info = Vec3f(-100.0f, -100.0f, -100.0f);
        }
    }
}

void cSLIC::enforce_connect(int padding, int diff_threshold) {
    vector<int> idx_img_cpy = idx_img;
    for (int r = 0; r < lab.rows; ++r) {
        for (int c = 0; c < lab.cols; ++c) {
            if (r < padding || r >= lab.rows - padding || c < padding || c >= lab.cols - padding) {
                continue;
            }
            int idx = r*lab.cols + c;
            int num_diff = 0;
            int diff_label = -1;
            for (int i = -padding; i <= padding; ++i) {
                for (int j = -padding; j <= padding; ++j) {
                    int idx_t = (r + i)*lab.cols + (c + j);
                    if (idx_img_cpy[idx] != idx_img_cpy[idx_t]) {
                        ++num_diff;
                        diff_label = idx_img_cpy[idx_t];
                    }
                }
            }
            if (num_diff > diff_threshold) {
                idx_img[idx] = diff_label;
            }
        }
    }
}

}}}
