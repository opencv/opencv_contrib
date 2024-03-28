// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef ENABLE_DVPP_INTERFACE
    #define ENABLE_DVPP_INTERFACE
#endif // ENABLE_DVPP_INTERFACE

#include <vector>
#include <string>
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <acl/dvpp/hi_dvpp.h>
#include "acl/acl_op.h"
#include "cann_call.hpp"

namespace cv
{
namespace cann
{
struct AscendPicDesc
{
    const char* name;
    std::shared_ptr<hi_void> data;
    std::vector<int64_t> batchNum;

    size_t widthAlignment = 16;
    size_t heightAlignment = 1;
    size_t sizeAlignment = 3;
    size_t sizeNum = 3;

    hi_vpc_pic_info Pic;
    AscendPicDesc& setMemAlign();
    AscendPicDesc& setPic(hi_pixel_format _picture_format);
    std::shared_ptr<hi_void> allocate();
    AscendPicDesc(){};
    AscendPicDesc(const AscendMat& ascendMat, hi_pixel_format _picture_format);
    AscendPicDesc(const Mat& mat, hi_pixel_format _picture_format);
};

/*
 ***************************** hi_mpi_vpc warppers ***************************
 The DVPP VPC interfaces here are all version v2. Only the following devices are supported: Atlas
 Inference Series products, Atlas 200/500 A2 Inference products and Atlas A2 Training Series
 products/Atlas 300I A2 Inference products.
*/
inline void vpcResizeWarpper(hi_vpc_chn chnId, hi_vpc_pic_info& inPic, hi_vpc_pic_info& outPic,
                             int interpolation, uint32_t* taskID)
{
    uint32_t ret = hi_mpi_vpc_resize(chnId, &inPic, &outPic, 0, 0, interpolation, taskID, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to resize image");
}
void vpcCropResizeWarpper(hi_vpc_chn chnId, hi_vpc_pic_info& inPic, hi_vpc_pic_info& outPic,
                          int cnt, uint32_t* taskID, const Rect& rect, Size dsize,
                          int interpolation);

void vpcCropResizeMakeBorderWarpper(hi_vpc_chn chnId, std::vector<AscendPicDesc>& inPicDesc,
                                    std::vector<AscendPicDesc>& outPicDesc, int cnt,
                                    uint32_t* taskID, const Rect& rect, Size dsize,
                                    int interpolation, const int borderType, Scalar scalarV,
                                    int top, int left);
void vpcCopyMakeBorderWarpper(hi_vpc_chn chnId, hi_vpc_pic_info& inPic, hi_vpc_pic_info& outPic,
                              uint32_t* taskID, int* offsets, int bordertype, Scalar value);
/*****************************************************************************/

/**
 * @brief Interface for calling DVPP operator descriptors.
 * The DVPP VPC interfaces here are all version v2. Supported devices: Atlas Inference Series
 * products, Atlas 200/500 A2 Inference products and Atlas A2 Training Series products/Atlas 300I A2
 * Inference products.
 */
class DvppOperatorDesc
{
private:
    DvppOperatorDesc& addInput(AscendPicDesc& picDesc);
    DvppOperatorDesc& addOutput(AscendPicDesc& picDesc);
    std::set<std::shared_ptr<hi_void>> holder;

public:
    DvppOperatorDesc()
    {
        chnId = 0;
        stChnAttr = {};
        createChannel();
    }
    virtual ~DvppOperatorDesc() { reset(); }
    DvppOperatorDesc& addInput(const AscendMat& mat);
    DvppOperatorDesc& addOutput(AscendMat& mat);
    DvppOperatorDesc& addInput(const Mat& mat);
    DvppOperatorDesc& addOutput(Mat& mat);

    DvppOperatorDesc& getResult(Mat& dst, uint32_t& taskIDResult);
    DvppOperatorDesc& getResult(AscendMat& dst, uint32_t& taskIDResult);

    DvppOperatorDesc& reset();
    DvppOperatorDesc& createChannel();

    std::vector<AscendPicDesc> inputDesc_;
    std::vector<AscendPicDesc> outputDesc_;

    hi_vpc_chn chnId;
    hi_vpc_chn_attr stChnAttr;
};

} // namespace cann
} // namespace cv