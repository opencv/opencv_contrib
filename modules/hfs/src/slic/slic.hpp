// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_SLIC_HPP_
#define _OPENCV_SLIC_HPP_

#include "../or_utils/or_types.hpp"
#include "opencv2/core.hpp"
//------------------------------------------------------
//
// Compile time GPU Settings
//
//------------------------------------------------------

#ifndef HFS_BLOCK_DIM
#define HFS_BLOCK_DIM 16
#endif

namespace cv { namespace hfs { namespace slic {

struct slicSettings
{
    Vector2i img_size;
    int spixel_size;
    int num_iters;
    float coh_weight;
};

struct cSpixelInfo
{
    Vec2f center;
    Vec3f color_info;
    int id;
    int num_pixels;
};

class cSLIC {
private:
    cv::Mat image;
    cv::Mat lab;
    std::vector<int> idx_img;

    cv::Vec2i map_size;
    std::vector<cSpixelInfo> spixel_list;
    int spixel_size;
    float spatial_weight;
    float max_xy_dist, max_color_dist;

    float compute_dist(cv::Point pix, cSpixelInfo center_info);

    void init_data(cv::Mat image_);
    cv::Mat cvt_img_space();
    void find_association();
    void update_cluster_center();
    void enforce_connect(int padding, int diff_threshold);
public:
    cSLIC() {}
    ~cSLIC() {}

    std::vector<int> generate_superpixels(cv::Mat image, int spixel_size_, float spatial_weight_);
};

#ifdef _HFS_CUDA_ON_

// utils used only by GPU version of slic
struct gSpixelInfo
{
    Vector2f center;
    Vector4f color_info;
    int id;
    int num_pixels;
};

typedef orutils::Image<gSpixelInfo> gSpixelMap;

namespace engines
{
    class SegEngine
    {
    protected:

        float max_color_dist;
        float max_xy_dist;

        cv::Ptr<UChar4Image> source_img;
        cv::Ptr<Float4Image> cvt_img;
        cv::Ptr<IntImage> idx_img;

        cv::Ptr<gSpixelMap> spixel_map;
        int spixel_size;

        Vector2i img_size;
        Vector2i map_size;

        slicSettings slic_settings;
        virtual void cvtImgSpace(cv::Ptr<UChar4Image> inimg,
                                 cv::Ptr<Float4Image> outimg) = 0;
        virtual void initClusterCenters() = 0;
        virtual void findCenterAssociation() = 0;
        virtual void updateClusterCenter() = 0;
        virtual void enforceConnectivity() = 0;

    public:

        SegEngine(const slicSettings& in_settings);
        virtual ~SegEngine();

        const cv::Ptr<IntImage> getSegMask() const
        {
            idx_img->updateHostFromDevice();
            return idx_img;
        };

        void setImageSize( int x, int y )
        {
            img_size.x = x;
            img_size.y = y;
            map_size.x = (int)ceil((float)x / (float)spixel_size);
            map_size.y = (int)ceil((float)y / (float)spixel_size);
        };

        Vector2i getImageSize()
        {
            return img_size;
        };

        void performSegmentation(cv::Ptr<UChar4Image> in_img);
    };

    class SegEngineGPU : public SegEngine
    {
    private:
        int no_grid_per_center;
        cv::Ptr<gSpixelMap> accum_map;
        cv::Ptr<IntImage> tmp_idx_img;
    protected:
        void cvtImgSpace(cv::Ptr<UChar4Image> inimg,
                         cv::Ptr<Float4Image> outimg);
        void initClusterCenters();
        void findCenterAssociation();
        void updateClusterCenter();
        void enforceConnectivity();
    public:
        SegEngineGPU(const slicSettings& in_settings);
        ~SegEngineGPU();
    };

    class CoreEngine
    {
    private:
        cv::Ptr<SegEngine> slic_seg_engine;

    public:

        CoreEngine(const slicSettings& in_settings);
        ~CoreEngine();

        void setImageSize(int x, int y);

        void processFrame(cv::Ptr<UChar4Image> in_img);

        const cv::Ptr<IntImage> getSegRes();
    };
} // end namespace engine

__host__ __device__ __forceinline__ void rgb2CIELab( const Vector4u& pix_in,
                                           Vector4f& pix_out )
{
    float _b = (float)pix_in.x / 255;
    float _g = (float)pix_in.y / 255;
    float _r = (float)pix_in.z / 255;

    if (_b <= 0.04045f)    _b = _b / 12.92f;
    else                   _b = pow( (_b + 0.055f) / 1.055f, 2.4f );
    if (_g <= 0.04045f)    _g = _g / 12.92f;
    else                   _g = pow( (_g + 0.055f) / 1.055f, 2.4f );
    if (_r <= 0.04045f)    _r = _r / 12.92f;
    else                   _r = pow( (_r + 0.055f) / 1.055f, 2.4f );

    float x = _r*0.4124564f + _g*0.3575761f + _b*0.1804375f;
    float y = _r*0.2126729f + _g*0.7151522f + _b*0.0721750f;
    float z = _r*0.0193339f + _g*0.1191920f + _b*0.9503041f;

    float epsilon = 0.008856f;
    float kappa = 903.3f;

    float Xr = 0.950456f;
    float Yr = 1.0f;
    float Zr = 1.088754f;

    float xr = x / Xr;
    float yr = y / Yr;
    float zr = z / Zr;

    float fx, fy, fz;
    if ( xr > epsilon )    fx = pow( xr, 1.0f / 3.0f );
    else                fx = ( kappa*xr + 16.0f ) / 116.0f;
    if ( yr > epsilon )    fy = pow( yr, 1.0f / 3.0f );
    else                fy = ( kappa*yr + 16.0f ) / 116.0f;
    if ( zr > epsilon )    fz = pow( zr, 1.0f / 3.0f );
    else                fz = ( kappa*zr + 16.0f ) / 116.0f;

    pix_out.x = 116.0f*fy - 16.0f;
    pix_out.y = 500.0f*(fx - fy);
    pix_out.z = 200.0f*(fy - fz);
}

__host__ __device__ __forceinline__ void initClusterCentersShared(
    const Vector4f* inimg, Vector2i map_size, Vector2i img_size,
    int spixel_size, int x, int y, cv::hfs::slic::gSpixelInfo* out_spixel)
{
    int cluster_idx = y * map_size.x + x;

    int img_x = x * spixel_size + spixel_size / 2;
    int img_y = y * spixel_size + spixel_size / 2;

    img_x = img_x >= img_size.x ? (x * spixel_size + img_size.x) / 2 : img_x;
    img_y = img_y >= img_size.y ? (y * spixel_size + img_size.y) / 2 : img_y;

    out_spixel[cluster_idx].id = cluster_idx;
    out_spixel[cluster_idx].center = Vector2f((float)img_x, (float)img_y);
    out_spixel[cluster_idx].color_info = inimg[img_y*img_size.x + img_x];

    out_spixel[cluster_idx].num_pixels = 0;
}

__host__ __device__ __forceinline__ float computeSlicDistance(
    const Vector4f& pix, int x, int y,
    const cv::hfs::slic::gSpixelInfo& center_info,
    float weight, float normalizer_xy, float normalizer_color)
{
    float dcolor =
        (pix.x - center_info.color_info.x)*(pix.x - center_info.color_info.x)
        + (pix.y - center_info.color_info.y)*(pix.y - center_info.color_info.y)
        + (pix.z - center_info.color_info.z)*(pix.z - center_info.color_info.z);

    float dxy =
        (x - center_info.center.x) * (x - center_info.center.x)
        + (y - center_info.center.y) * (y - center_info.center.y);


    float retval =
        dcolor * normalizer_color + weight * dxy * normalizer_xy;
    return sqrtf(retval);
}

__host__ __device__ __forceinline__ void findCenterAssociationShared(
    const Vector4f* inimg,
    const cv::hfs::slic::gSpixelInfo* in_spixel_map,
    Vector2i map_size, Vector2i img_size,
    int spixel_size, float weight, int x, int y,
    float max_xy_dist, float max_color_dist, int* out_idx_img)
{
    int idx_img = y * img_size.x + x;

    int ctr_x = x / spixel_size;
    int ctr_y = y / spixel_size;

    int minidx = -1;
    float dist = 999999.9999f;

    for ( int i = -1; i <= 1; i++ )
    for ( int j = -1; j <= 1; j++ )
    {
        int ctr_x_check = ctr_x + j;
        int ctr_y_check = ctr_y + i;
        if (ctr_x_check >= 0 && ctr_y_check >= 0 &&
            ctr_x_check < map_size.x && ctr_y_check < map_size.y)
        {
            int ctr_idx = ctr_y_check*map_size.x + ctr_x_check;
            float cdist =
                computeSlicDistance(inimg[idx_img], x, y,
                    in_spixel_map[ctr_idx], weight,
                    max_xy_dist, max_color_dist);
            if (cdist < dist)
            {
                dist = cdist;
                minidx = in_spixel_map[ctr_idx].id;
            }
        }
    }

    if (minidx >= 0)
        out_idx_img[idx_img] = minidx;
}

__host__ __device__ __forceinline__ void finalizeReductionResultShared(
    const cv::hfs::slic::gSpixelInfo* accum_map,
    Vector2i map_size, int num_blocks_per_spixel, int x, int y,
    cv::hfs::slic::gSpixelInfo* spixel_list)
{
    int spixel_idx = y * map_size.x + x;

    spixel_list[spixel_idx].center = Vector2f(0, 0);
    spixel_list[spixel_idx].color_info = Vector4f(0, 0, 0, 0);
    spixel_list[spixel_idx].num_pixels = 0;

    for (int i = 0; i < num_blocks_per_spixel; i++)
    {
        int accum_list_idx = spixel_idx * num_blocks_per_spixel + i;

        spixel_list[spixel_idx].center +=
            accum_map[accum_list_idx].center;
        spixel_list[spixel_idx].color_info +=
            accum_map[accum_list_idx].color_info;
        spixel_list[spixel_idx].num_pixels +=
            accum_map[accum_list_idx].num_pixels;
    }

    if (spixel_list[spixel_idx].num_pixels != 0)
    {
        spixel_list[spixel_idx].center /=
            (float)spixel_list[spixel_idx].num_pixels;
        spixel_list[spixel_idx].color_info /=
            (float)spixel_list[spixel_idx].num_pixels;
    }
    else
    {
        spixel_list[spixel_idx].center =
            Vector2f(-100, -100);
        spixel_list[spixel_idx].color_info =
            Vector4f(-100, -100, -100, -100);
    }
}

__host__ __device__ __forceinline__ void supressLocalLable(
    const int* in_idx_img, Vector2i img_size,
    int x, int y, int* out_idx_img)
{
    int clable = in_idx_img[y*img_size.x + x];

    if (x < 2 || y < 2 || x >= img_size.x - 2 || y >= img_size.y - 2)
    {
        out_idx_img[y*img_size.x + x] = clable;
        return;
    }

    int diff_count = 0;
    int diff_lable = -1;

    for ( int j = -2; j <= 2; j++ )
    for ( int i = -2; i <= 2; i++ )
    {
        int nlable = in_idx_img[(y + j)*img_size.x + (x + i)];
        if (nlable != clable)
        {
            diff_lable = nlable;
            diff_count++;
        }
    }

    if (diff_count > 16)
        out_idx_img[y*img_size.x + x] = diff_lable;
    else
        out_idx_img[y*img_size.x + x] = clable;
}

__host__ __device__ __forceinline__ void supressLocalLable2(const int* in_idx_img,
    Vector2i img_size, int x, int y, int* out_idx_img)
{
    int pixel_idx = y*img_size.x + x;
    int clable = in_idx_img[pixel_idx];
    if (x < 1 || y < 1 || x >= img_size.x - 1 || y >= img_size.y - 1)
    {
        out_idx_img[y*img_size.x + x] = clable;
        return;
    }

    int diff_count = 0;
    int diff_lable = -1;

    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++)
    {
        int nlable = in_idx_img[(y + j)*img_size.x + (x + i)];
        if (nlable != clable)
        {
            diff_lable = nlable;
            diff_count++;
        }
    }

    if (diff_count >= 6)
        out_idx_img[pixel_idx] = diff_lable;
    else
        out_idx_img[pixel_idx] = clable;
}

__host__ __device__ __forceinline__ dim3 getGridSize( Vector2i dataSz, dim3 blockSz )
{
    return dim3((dataSz.x + blockSz.x - 1) / blockSz.x,
        (dataSz.y + blockSz.y - 1) / blockSz.y);
}

struct Float4_
{
    __host__ __device__ Float4_() {}
    __host__ __device__ Float4_( float x_, float y_, float z_, float w_ ) {
        x = x_, y = y_, z = z_, w = w_;
    }
    volatile float x, y, z, w;
};

struct Float2_
{
    __host__ __device__ Float2_() {}
    __host__ __device__ Float2_( float x_, float y_ ) {
        x = x_, y = y_;
    }
    volatile float x, y;
};

__host__ __device__ __forceinline__ Float4_ operator+= ( Float4_ &a, Float4_ b )
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__host__ __device__ __forceinline__ Float2_ operator+= ( Float2_ &a, Float2_ b )
{
    a.x += b.x;
    a.y += b.y;
    return a;
}


#endif


}}}

#endif
