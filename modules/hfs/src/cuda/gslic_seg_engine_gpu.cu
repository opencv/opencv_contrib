// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// #ifdef _HFS_CUDA_ON_

#include "../precomp.hpp"
#include "../slic/slic.hpp"

namespace cv { namespace hfs { namespace slic { namespace engines {


__global__ void cvtImgSpaceDevice(const Vector4u* inimg,
    Vector2i img_size, Vector4f* outimg);

__global__ void initClusterCentersDevice(const Vector4f* inimg,
    Vector2i map_size, Vector2i img_size, int spixel_size,
    gSpixelInfo* out_spixel);

__global__ void findCenterAssociationDevice(const Vector4f* inimg,
    const gSpixelInfo* in_spixel_map, Vector2i map_size,
    Vector2i img_size, int spixel_size, float weight,
    float max_xy_dist, float max_color_dist, int* out_idx_img);

__global__ void updateClusterCenterDevice(const Vector4f* inimg,
    const int* in_idx_img, Vector2i map_size, Vector2i img_size,
    int spixel_size, int no_blocks_per_line, gSpixelInfo* accum_map);

__global__ void finalizeReductionResultDevice(const gSpixelInfo* accum_map,
    Vector2i map_size, int no_blocks_per_spixel,
    gSpixelInfo* spixel_list);

__global__ void enforceConnectivityDevice(const int* in_idx_img,
    Vector2i img_size, int* out_idx_img);

__global__ void enforceConnectivityDevice1_2(const int* in_idx_img,
    Vector2i img_size, int* out_idx_img);



SegEngineGPU::SegEngineGPU(const slicSettings& in_settings) : SegEngine(in_settings)
{
    source_img = Ptr<UChar4Image>(new UChar4Image(in_settings.img_size));
    cvt_img = Ptr<Float4Image>(new Float4Image(in_settings.img_size));
    idx_img = Ptr<IntImage>(new IntImage(in_settings.img_size));
    tmp_idx_img = Ptr<IntImage>(new IntImage(in_settings.img_size));

    spixel_size = in_settings.spixel_size;

    int spixel_per_col = (int)ceil((float)in_settings.img_size.x / (float)spixel_size);
    int spixel_per_row = (int)ceil((float)in_settings.img_size.y / (float)spixel_size);

    map_size = Vector2i(spixel_per_col, spixel_per_row);
    spixel_map = Ptr<gSpixelMap>(new gSpixelMap(map_size));

    no_grid_per_center =
        (int)ceil(spixel_size*3.0f / HFS_BLOCK_DIM)*((int)ceil(spixel_size*3.0f / HFS_BLOCK_DIM));

    Vector2i accum_size(map_size.x*no_grid_per_center, map_size.y);
    accum_map = Ptr<gSpixelMap>(new gSpixelMap(accum_size));

    // normalizing factors
    max_color_dist = 15.0f / (1.7321f * 128);
    max_color_dist *= max_color_dist;
    max_xy_dist = 1.0f / (2 * spixel_size * spixel_size);
}

SegEngineGPU::~SegEngineGPU() {}


void SegEngineGPU::cvtImgSpace(Ptr<UChar4Image> inimg, Ptr<Float4Image> outimg)
{
    Vector4u* inimg_ptr = inimg->getGpuData();
    Vector4f* outimg_ptr = outimg->getGpuData();

    dim3 blockSize(HFS_BLOCK_DIM, HFS_BLOCK_DIM);
    dim3 gridSize = getGridSize(img_size, blockSize);
    cvtImgSpaceDevice << <gridSize, blockSize >> >(inimg_ptr, img_size, outimg_ptr);
}

void SegEngineGPU::initClusterCenters()
{
    gSpixelInfo* spixel_list = spixel_map->getGpuData();
    Vector4f* img_ptr = cvt_img->getGpuData();

    dim3 blockSize(HFS_BLOCK_DIM, HFS_BLOCK_DIM);
    dim3 gridSize = getGridSize(map_size, blockSize);
    initClusterCentersDevice << <gridSize, blockSize >> >
        (img_ptr, map_size, img_size, spixel_size, spixel_list);
}

void SegEngineGPU::findCenterAssociation()
{
    gSpixelInfo* spixel_list = spixel_map->getGpuData();
    Vector4f* img_ptr = cvt_img->getGpuData();
    int* idx_ptr = idx_img->getGpuData();

    dim3 blockSize(HFS_BLOCK_DIM, HFS_BLOCK_DIM);
    dim3 gridSize = getGridSize(img_size, blockSize);

    findCenterAssociationDevice << <gridSize, blockSize >> >
        (img_ptr, spixel_list, map_size, img_size,
            spixel_size, slic_settings.coh_weight,
            max_xy_dist, max_color_dist, idx_ptr);
}

void SegEngineGPU::updateClusterCenter()
{
    gSpixelInfo* accum_map_ptr = accum_map->getGpuData();
    gSpixelInfo* spixel_list_ptr = spixel_map->getGpuData();
    Vector4f* img_ptr = cvt_img->getGpuData();
    int* idx_ptr = idx_img->getGpuData();

    int no_blocks_per_line = (int)ceil(spixel_size * 3.0f / HFS_BLOCK_DIM);

    dim3 blockSize(HFS_BLOCK_DIM, HFS_BLOCK_DIM);
    dim3 gridSize(map_size.x, map_size.y, no_grid_per_center);

    updateClusterCenterDevice << <gridSize, blockSize >> >
        (img_ptr, idx_ptr, map_size, img_size,
            spixel_size, no_blocks_per_line, accum_map_ptr);

    dim3 gridSize2(map_size.x, map_size.y);

    finalizeReductionResultDevice << <gridSize2, blockSize >> >
        (accum_map_ptr, map_size, no_grid_per_center, spixel_list_ptr);
}

void SegEngineGPU::enforceConnectivity()
{
    int* idx_ptr = idx_img->getGpuData();
    int* tmp_idx_ptr = tmp_idx_img->getGpuData();

    dim3 blockSize(HFS_BLOCK_DIM, HFS_BLOCK_DIM);
    dim3 gridSize = getGridSize(img_size, blockSize);

    enforceConnectivityDevice << <gridSize, blockSize >> >
        (idx_ptr, img_size, tmp_idx_ptr);
    enforceConnectivityDevice << <gridSize, blockSize >> >
        (tmp_idx_ptr, img_size, idx_ptr);
    enforceConnectivityDevice1_2 << <gridSize, blockSize >> >
        (idx_ptr, img_size, tmp_idx_ptr);
    enforceConnectivityDevice1_2 << <gridSize, blockSize >> >
        (tmp_idx_ptr, img_size, idx_ptr);
}


__global__ void cvtImgSpaceDevice(const Vector4u* inimg, Vector2i img_size,
    Vector4f* outimg)
{
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx_x >= img_size.x || idx_y >= img_size.y)
        return;

    int idx = idx_y*img_size.x + idx_x;
    rgb2CIELab(inimg[idx], outimg[idx]);
}

__global__ void initClusterCentersDevice(const Vector4f* inimg,
    Vector2i map_size, Vector2i img_size, int spixel_size,
    gSpixelInfo* out_spixel)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= map_size.x || y >= map_size.y) return;

    initClusterCentersShared(inimg, map_size,
        img_size, spixel_size, x, y, out_spixel);
}

__global__ void findCenterAssociationDevice(const Vector4f* inimg,
    const gSpixelInfo* in_spixel_map, Vector2i map_size,
    Vector2i img_size, int spixel_size, float weight,
    float max_xy_dist, float max_color_dist, int* out_idx_img)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= img_size.x || y >= img_size.y) return;
    findCenterAssociationShared(inimg, in_spixel_map, map_size, img_size,
        spixel_size, weight, x, y, max_xy_dist, max_color_dist, out_idx_img);
}

__global__ void updateClusterCenterDevice(const Vector4f* inimg,
    const int* in_idx_img, Vector2i map_size, Vector2i img_size,
    int spixel_size, int no_blocks_per_line, gSpixelInfo* accum_map)
{
    int local_id = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ Float4_ color_shared[HFS_BLOCK_DIM*HFS_BLOCK_DIM];
    __shared__ Float2_ xy_shared[HFS_BLOCK_DIM*HFS_BLOCK_DIM];
    __shared__ volatile int count_shared[HFS_BLOCK_DIM*HFS_BLOCK_DIM];
    __shared__ bool should_add;

    color_shared[local_id] = Float4_(0, 0, 0, 0);
    xy_shared[local_id] = Float2_(0, 0);
    count_shared[local_id] = 0;
    should_add = false;
    __syncthreads();

    int no_blocks_per_spixel = gridDim.z;

    int spixel_id = blockIdx.y * map_size.x + blockIdx.x;

    // compute the relative position in the search window
    int block_x = blockIdx.z % no_blocks_per_line;
    int block_y = blockIdx.z / no_blocks_per_line;

    int x_offset = block_x * HFS_BLOCK_DIM + threadIdx.x;
    int y_offset = block_y * HFS_BLOCK_DIM + threadIdx.y;

    if (x_offset < spixel_size * 3 && y_offset < spixel_size * 3)
    {
        // compute the start of the search window
        int x_start = blockIdx.x * spixel_size - spixel_size;
        int y_start = blockIdx.y * spixel_size - spixel_size;

        int x_img = x_start + x_offset;
        int y_img = y_start + y_offset;

        if (x_img >= 0 && x_img < img_size.x && y_img >= 0 && y_img < img_size.y)
        {
            int img_idx = y_img * img_size.x + x_img;
            if (in_idx_img[img_idx] == spixel_id)
            {
                color_shared[local_id] =
                    Float4_(inimg[img_idx].x, inimg[img_idx].y,
                        inimg[img_idx].z, inimg[img_idx].w);
                xy_shared[local_id] = Float2_(x_img, y_img);
                count_shared[local_id] = 1;
                should_add = true;
            }
        }
    }
    __syncthreads();

    if (should_add)
    {
        if (local_id < 128)
        {
            color_shared[local_id] += color_shared[local_id + 128];
            xy_shared[local_id] += xy_shared[local_id + 128];
            count_shared[local_id] += count_shared[local_id + 128];
        }
        __syncthreads();

        if (local_id < 64)
        {
            color_shared[local_id] += color_shared[local_id + 64];
            xy_shared[local_id] += xy_shared[local_id + 64];
            count_shared[local_id] += count_shared[local_id + 64];
        }
        __syncthreads();

        if (local_id < 32)
        {
            color_shared[local_id] += color_shared[local_id + 32];
            color_shared[local_id] += color_shared[local_id + 16];
            color_shared[local_id] += color_shared[local_id + 8];
            color_shared[local_id] += color_shared[local_id + 4];
            color_shared[local_id] += color_shared[local_id + 2];
            color_shared[local_id] += color_shared[local_id + 1];

            xy_shared[local_id] += xy_shared[local_id + 32];
            xy_shared[local_id] += xy_shared[local_id + 16];
            xy_shared[local_id] += xy_shared[local_id + 8];
            xy_shared[local_id] += xy_shared[local_id + 4];
            xy_shared[local_id] += xy_shared[local_id + 2];
            xy_shared[local_id] += xy_shared[local_id + 1];

            count_shared[local_id] += count_shared[local_id + 32];
            count_shared[local_id] += count_shared[local_id + 16];
            count_shared[local_id] += count_shared[local_id + 8];
            count_shared[local_id] += count_shared[local_id + 4];
            count_shared[local_id] += count_shared[local_id + 2];
            count_shared[local_id] += count_shared[local_id + 1];
        }
    }
    __syncthreads();

    if (local_id == 0)
    {
        int accum_map_idx = spixel_id * no_blocks_per_spixel + blockIdx.z;
        accum_map[accum_map_idx].center = Vector2f(xy_shared[0].x, xy_shared[0].y);
        accum_map[accum_map_idx].color_info =
            Vector4f(color_shared[0].x, color_shared[0].y,
                color_shared[0].z, color_shared[0].w);
        accum_map[accum_map_idx].num_pixels = count_shared[0];
    }
}

__global__ void finalizeReductionResultDevice(const gSpixelInfo* accum_map,
    Vector2i map_size, int no_blocks_per_spixel, gSpixelInfo* spixel_list)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= map_size.x || y >= map_size.y) return;

    finalizeReductionResultShared(accum_map,
        map_size, no_blocks_per_spixel, x, y, spixel_list);
}

__global__ void enforceConnectivityDevice(const int* in_idx_img,
    Vector2i img_size, int* out_idx_img)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= img_size.x || y >= img_size.y) return;

    supressLocalLable(in_idx_img, img_size, x, y, out_idx_img);
}

__global__ void enforceConnectivityDevice1_2(const int* in_idx_img,
    Vector2i img_size, int* out_idx_img)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= img_size.x || y >= img_size.y) return;

    supressLocalLable2(in_idx_img, img_size, x, y, out_idx_img);
}

}}}}

// #endif
