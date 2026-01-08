// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <acl/acl.h>
#include <acl/dvpp/hi_dvpp.h>
#include "opencv2/dvpp_call.hpp"
#include <iostream>
#include <memory>
#include <cstdarg>
#include <string>

#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)

namespace cv
{
namespace cann
{

/******************************AscendPicDesc****************************/
AscendPicDesc& AscendPicDesc::setMemAlign()
{
    if (Pic.picture_format == HI_PIXEL_FORMAT_BGR_888 ||
        Pic.picture_format == HI_PIXEL_FORMAT_RGB_888 ||
        Pic.picture_format == HI_PIXEL_FORMAT_YUV_PACKED_444)
    {
        widthAlignment = 16;
        heightAlignment = 1;
        sizeAlignment = 3;
        sizeNum = 3;
    }
    else if (Pic.picture_format == HI_PIXEL_FORMAT_YUV_400)
    {
        widthAlignment = 16;
        heightAlignment = 1;
        sizeAlignment = 1;
        sizeNum = 1;
    }
    else if (Pic.picture_format == HI_PIXEL_FORMAT_ARGB_8888 ||
             Pic.picture_format == HI_PIXEL_FORMAT_ABGR_8888 ||
             Pic.picture_format == HI_PIXEL_FORMAT_RGBA_8888 ||
             Pic.picture_format == HI_PIXEL_FORMAT_BGRA_8888)
    {
        widthAlignment = 16;
        heightAlignment = 1;
        sizeAlignment = 4;
        sizeNum = 4;
    }
    return *this;
}

AscendPicDesc& AscendPicDesc::setPic(hi_pixel_format _picture_format)
{
    // set input
    Pic.picture_format = _picture_format;
    setMemAlign();
    Pic.picture_width_stride = ALIGN_UP(Pic.picture_width, widthAlignment) * sizeAlignment;
    Pic.picture_height_stride = ALIGN_UP(Pic.picture_height, heightAlignment);
    Pic.picture_buffer_size =
        Pic.picture_width_stride * Pic.picture_height_stride * sizeAlignment / sizeNum;
    return *this;
}

std::shared_ptr<hi_void> AscendPicDesc::allocate()
{
    Pic.picture_address = nullptr;
    uint32_t ret = hi_mpi_dvpp_malloc(0, &Pic.picture_address, Pic.picture_buffer_size);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to malloc mem on dvpp");

    return std::shared_ptr<hi_void>(Pic.picture_address, [](void* ptr) { hi_mpi_dvpp_free(ptr); });
}

AscendPicDesc::AscendPicDesc(const AscendMat& ascendMat, hi_pixel_format _picture_format)
{
    Pic.picture_width = ascendMat.cols;
    Pic.picture_height = ascendMat.rows;
    setPic(_picture_format);
    data = allocate();
}

AscendPicDesc::AscendPicDesc(const Mat& mat, hi_pixel_format _picture_format)
{
    Pic.picture_width = mat.cols;
    Pic.picture_height = mat.rows;
    setPic(_picture_format);
    data = allocate();
}

/******************************hi_mpi_vpc warppers****************************/
void vpcCropResizeWarpper(hi_vpc_chn chnId, hi_vpc_pic_info& inPic, hi_vpc_pic_info& outPic,
                          int cnt, uint32_t* taskID, const Rect& rect, Size dsize,
                          int interpolation)
{
    hi_vpc_crop_region cropRegion = {.top_offset = static_cast<hi_u32>(rect.y),
                                     .left_offset = static_cast<hi_u32>(rect.x),
                                     .crop_width = static_cast<hi_u32>(rect.width),
                                     .crop_height = static_cast<hi_u32>(rect.height)};

    hi_vpc_resize_info resize_info = {.resize_width = static_cast<hi_u32>(dsize.width),
                                      .resize_height = static_cast<hi_u32>(dsize.height),
                                      .interpolation = static_cast<hi_u32>(interpolation)};
    hi_vpc_crop_resize_region crop_resize_info[1];
    crop_resize_info[0].dest_pic_info = outPic;
    crop_resize_info[0].crop_region = cropRegion;
    crop_resize_info[0].resize_info = resize_info;
    uint32_t ret = hi_mpi_vpc_crop_resize(chnId, (const hi_vpc_pic_info*)&inPic, crop_resize_info,
                                          cnt, taskID, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to crop and resize image");
}

void vpcCopyMakeBorderWarpper(hi_vpc_chn chnId, hi_vpc_pic_info& inPic, hi_vpc_pic_info& outPic,
                              uint32_t* taskID, int* offsets, int bordertype, Scalar value)
{
    hi_vpc_make_border_info make_border_info;
    make_border_info = {.top = static_cast<hi_u32>(offsets[0]),
                        .bottom = static_cast<hi_u32>(offsets[1]),
                        .left = static_cast<hi_u32>(offsets[2]),
                        .right = static_cast<hi_u32>(offsets[3]),
                        .border_type = saturate_cast<hi_vpc_bord_type>(bordertype)};
    if (outPic.picture_format == HI_PIXEL_FORMAT_BGR_888)
    {
        make_border_info.scalar_value.val[0] = value[2];
        make_border_info.scalar_value.val[1] = value[1];
        make_border_info.scalar_value.val[2] = value[0];
    }
    else if (outPic.picture_format == HI_PIXEL_FORMAT_YUV_400)
    {
        make_border_info.scalar_value.val[0] = value[0];
        make_border_info.scalar_value.val[1] = value[1];
        make_border_info.scalar_value.val[2] = value[2];
    }
    make_border_info.scalar_value.val[3] = value[3];
    uint32_t ret = hi_mpi_vpc_copy_make_border(chnId, (const hi_vpc_pic_info*)&inPic, &outPic,
                                               make_border_info, taskID, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to crop and resize image");
}

void setBatchCropResizeMakeBorder(std::vector<AscendPicDesc>& outPicDesc,
                                  hi_vpc_crop_resize_border_region crop_resize_make_border_info[],
                                  const Rect& rect, Size dsize, int interpolation,
                                  const int borderType, Scalar scalarV, int top, int left,
                                  int batchSize)
{
    hi_vpc_crop_region cropRegion = {.top_offset = static_cast<hi_u32>(rect.y),
                                     .left_offset = static_cast<hi_u32>(rect.x),
                                     .crop_width = static_cast<hi_u32>(rect.width),
                                     .crop_height = static_cast<hi_u32>(rect.height)};

    hi_vpc_resize_info resize_info = {.resize_width = static_cast<hi_u32>(dsize.width),
                                      .resize_height = static_cast<hi_u32>(dsize.height),
                                      .interpolation = static_cast<hi_u32>(interpolation)};
    for (int i = 0; i < batchSize; i++)
    {
        crop_resize_make_border_info[i].dest_pic_info = outPicDesc[i].Pic;
        crop_resize_make_border_info[i].crop_region = cropRegion;
        crop_resize_make_border_info[i].resize_info = resize_info;
        crop_resize_make_border_info[i].dest_top_offset = top;
        crop_resize_make_border_info[i].dest_left_offset = left;
        crop_resize_make_border_info[i].border_type = static_cast<hi_vpc_bord_type>(borderType);
        if (crop_resize_make_border_info[i].dest_pic_info.picture_format == HI_PIXEL_FORMAT_BGR_888)
        {
            crop_resize_make_border_info[i].scalar_value.val[0] = scalarV[2];
            crop_resize_make_border_info[i].scalar_value.val[1] = scalarV[1];
            crop_resize_make_border_info[i].scalar_value.val[2] = scalarV[0];
        }
        else if (crop_resize_make_border_info[i].dest_pic_info.picture_format ==
                 HI_PIXEL_FORMAT_YUV_400)
        {
            crop_resize_make_border_info[i].scalar_value.val[0] = scalarV[0];
            crop_resize_make_border_info[i].scalar_value.val[1] = scalarV[1];
            crop_resize_make_border_info[i].scalar_value.val[2] = scalarV[2];
        }
        crop_resize_make_border_info[i].scalar_value.val[3] = scalarV[3];
    }
}

void vpcCropResizeMakeBorderWarpper(hi_vpc_chn chnId, std::vector<AscendPicDesc>& inPicDesc,
                                    std::vector<AscendPicDesc>& outPicDesc, int cnt,
                                    uint32_t* taskID, const Rect& rect, Size dsize,
                                    int interpolation, const int borderType, Scalar scalarV,
                                    int top, int left)
{
    hi_vpc_crop_resize_border_region crop_resize_make_border_info[1];

    setBatchCropResizeMakeBorder(outPicDesc, crop_resize_make_border_info, rect, dsize,
                                 interpolation, borderType, scalarV, top, left, 1);
    uint32_t ret =
        hi_mpi_vpc_crop_resize_make_border(chnId, (const hi_vpc_pic_info*)&inPicDesc[0].Pic,
                                           crop_resize_make_border_info, cnt, taskID, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to crop, resize and make border of image");
}

/******************************DvppOperatorDesc****************************/
DvppOperatorDesc& DvppOperatorDesc::reset()
{
    uint32_t ret = hi_mpi_vpc_destroy_chn(chnId);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to destory DVPP vpc channel");
    inputDesc_.clear();
    outputDesc_.clear();
    holder.clear();
    return *this;
}
void initDvpp() { hi_mpi_sys_init(); }

void finalizeDvpp() { hi_mpi_sys_exit(); }

DvppOperatorDesc& DvppOperatorDesc::createChannel()
{
    uint32_t ret = hi_mpi_vpc_sys_create_chn(&chnId, &stChnAttr);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to create DVPP vpc channel");
    return *this;
}

// copy input array to dvpp memory
DvppOperatorDesc& DvppOperatorDesc::addInput(AscendPicDesc& picDesc)
{
    inputDesc_.push_back(picDesc);
    holder.insert(picDesc.data);
    return *this;
}

template <typename inMat>
hi_pixel_format setPixelFormat(const inMat& mat)
{
    CV_Assert(mat.channels() == 3 || mat.channels() == 1);
    hi_pixel_format _picture_format;
    if (mat.channels() == 3)
    {
        _picture_format = HI_PIXEL_FORMAT_BGR_888;
    }
    else if (mat.channels() == 1)
    {
        _picture_format = HI_PIXEL_FORMAT_YUV_400;
    }
    return _picture_format;
}

DvppOperatorDesc& DvppOperatorDesc::addInput(const AscendMat& mat)
{
    Mat matHost;
    mat.download(matHost);
    return addInput(matHost);
}

DvppOperatorDesc& DvppOperatorDesc::addInput(const Mat& mat)
{
    hi_pixel_format _picture_format = setPixelFormat(mat);

    AscendPicDesc picDesc(mat, _picture_format);
    aclrtMemcpy2d(picDesc.Pic.picture_address, picDesc.Pic.picture_width_stride, mat.data,
                  mat.step[0], mat.step[0], picDesc.Pic.picture_height, ACL_MEMCPY_HOST_TO_DEVICE);

    return addInput(picDesc);
}

// malloc memory for output
DvppOperatorDesc& DvppOperatorDesc::addOutput(AscendPicDesc& picDesc)
{
    outputDesc_.push_back(picDesc);
    holder.insert(picDesc.data);
    return *this;
}

DvppOperatorDesc& DvppOperatorDesc::addOutput(AscendMat& mat)
{
    hi_pixel_format _picture_format = setPixelFormat(mat);
    AscendPicDesc picDesc(mat, _picture_format);
    return addOutput(picDesc);
}

DvppOperatorDesc& DvppOperatorDesc::addOutput(Mat& mat)
{
    hi_pixel_format _picture_format = setPixelFormat(mat);
    AscendPicDesc picDesc(mat, _picture_format);
    return addOutput(picDesc);
}

// get process result and copy it to host/device
DvppOperatorDesc& DvppOperatorDesc::getResult(Mat& dst, uint32_t& taskIDResult)
{
    uint32_t ret = hi_mpi_vpc_get_process_result(chnId, taskIDResult, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to get process result.");
    const uint32_t esz = CV_ELEM_SIZE(dst.type());
    size_t step = esz * dst.cols;

    aclrtMemcpy2d(dst.data, dst.step[0], outputDesc_[0].Pic.picture_address,
                  outputDesc_[0].Pic.picture_width_stride, dst.step[0],
                  outputDesc_[0].Pic.picture_height, ACL_MEMCPY_DEVICE_TO_HOST);
    return *this;
}

DvppOperatorDesc& DvppOperatorDesc::getResult(AscendMat& dst, uint32_t& taskIDResult)
{
    Mat matHost;
    matHost.create(dst.rows, dst.cols, dst.type());
    getResult(matHost, taskIDResult);
    dst.upload(matHost);
    return *this;
}

} // namespace cann
} // namespace cv
