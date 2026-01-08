// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNOPS_CANN_CALL_HPP
#define OPENCV_CANNOPS_CANN_CALL_HPP

#include <vector>
#include <set>
#include <string>
#include <acl/acl_base.h>
#include "cann.hpp"
#include "stream_accessor.hpp"
#include "ascendc_kernels.hpp"

class aclopAttr;

namespace cv
{
namespace cann
{
CV_EXPORTS void checkAclError(aclError err, const char* file, const int line, const char* func);
void checkAclPtr(void* ptr, const char* file, const int line, const char* func);
#define CV_ACL_SAFE_CALL(expr) checkAclError((expr), __FILE__, __LINE__, CV_Func)
#define CV_ACL_SAFE_CALL_PTR(expr)                     \
    ({                                                 \
        auto ptr = (expr);                             \
        checkAclPtr(ptr, __FILE__, __LINE__, CV_Func); \
        ptr;                                           \
    })
// Warpper for functions in CANN, callers should not call CANN's api directly, but should call the
// function provided in cann_call.
void aclrtMallocWarpper(void** data, size_t size);
void aclrtFreeWarpper(void* data);
void aclrtMemcpyWarpper(std::shared_ptr<uchar>& dst, size_t offset, const void* src, size_t size,
                        AscendStream& stream);
void aclrtMemcpyWarpper(void* dst, const std::shared_ptr<uchar>& src, size_t offset, size_t size,
                        AscendStream& stream);
void aclrtMemcpyWarpper(std::shared_ptr<uchar>& dst, size_t dstOffset,
                        const std::shared_ptr<uchar>& src, size_t srcOffset, size_t size,
                        AscendStream& stream);
void aclrtMemcpy2dWarpper(std::shared_ptr<uchar>& dst, size_t offset, size_t dpitch,
                          const void* src, size_t spitch, size_t width, size_t length,
                          AscendStream& stream);
void aclrtMemcpy2dWarpper(void* dst, size_t dpitch, const std::shared_ptr<uchar>& src,
                          size_t offset, size_t spitch, size_t width, size_t length,
                          AscendStream& stream);
void aclrtMemsetWarpper(std::shared_ptr<uchar>& ptr, int32_t value, size_t count,
                        AscendStream& stream);
//! Type mapping between opencv and cann.
aclDataType getACLType(int opencvdepth);
//! Malloc and upload raw data to devices.
CV_EXPORTS std::shared_ptr<uchar> mallocAndUpload(const void* data, size_t size, AscendStream& stream,
                                       AscendMat::Allocator* allocator);
/**
 * @brief Warpper of CANN streams.
 */
class AscendStream::Impl
{
public:
    aclrtStream stream;
    bool ownStream;
    /**
     * @brief Ascend and CANN use stream to implement asynchronous calls. Which means when function
     * returns, operator may not finish, even not start. If caller free any tensors that participate
     * in this operatation, it have a chance to access invalid memory.
     * All tensors should add to holder, holder will be cleaned by waitForCompletion function, or when
     * the stream is destructing.
     */
    std::set<std::shared_ptr<uchar>> tensorHolders;
    Impl();
    explicit Impl(aclrtStream stream);
    void AddTensorHolder(const std::shared_ptr<uchar>& tensorData);
};

/**
 * @brief Warpper of CANN event.
 */
class AscendEvent::Impl
{
public:
    aclrtEvent event;
    bool ownEvent;

    Impl();
    explicit Impl(aclrtEvent event);
    ~Impl();
};

/**
 * @brief Parameter type for call_call interfaces.
 */
struct AscendTensor
{
    const char* name;
    std::shared_ptr<uchar> data;
    size_t dataSize;
    std::vector<int64_t> dims;
    aclDataType dtype;
    aclFormat format;
    AscendTensor(){};
    AscendTensor(std::shared_ptr<uchar> _data, size_t _dataSize, int64_t* _dims, size_t _dimSize,
                 aclDataType _dtype, const char* _name = "", aclFormat _format = ACL_FORMAT_ND);
    AscendTensor(std::shared_ptr<uchar> _data, size_t _dataSize, std::vector<int64_t>& _dims,
                 aclDataType _dtype, const char* _name = "", aclFormat _format = ACL_FORMAT_ND)
        : name(_name), data(_data), dataSize(_dataSize), dims(_dims), dtype(_dtype),
          format(_format){};
    AscendTensor(const AscendMat& ascendMat, const char* _name = "",
                 aclFormat format = ACL_FORMAT_ND);
};

/**
 * @brief Interface to call operators in CANN package.
 */
class OperatorRunner
{
private:
    std::vector<aclDataBuffer*> inputBuffers_;
    std::vector<aclDataBuffer*> outputBuffers_;
    std::vector<aclTensorDesc*> inputDesc_;
    std::vector<aclTensorDesc*> outputDesc_;
    aclopAttr* opAttr_;
    bool opAttrInit;
    std::string op;

    std::set<std::shared_ptr<uchar>> holder;

    OperatorRunner& addInput(AscendTensor& mat);
    OperatorRunner& addOutput(AscendTensor& mat);

public:
    OperatorRunner() : opAttrInit(false) {}
    virtual ~OperatorRunner() { reset(); }
    OperatorRunner& setOp(const char* op);
    OperatorRunner& addInput(const AscendMat& mat);
    OperatorRunner& addOutput(AscendMat& mat);
    OperatorRunner& addAttr(float value, const char* name);
    OperatorRunner& addAttr(const char* value, const char* name);
    OperatorRunner& addAttr(int value, const char* name);
    OperatorRunner& addAttr(bool value, const char* name);
    OperatorRunner& addAttr(const int64_t* value, int size, const char* name);
    OperatorRunner& addInput(const AscendMat& mat, const char* name);
    OperatorRunner& addInput(const Scalar& sc, int type, const char* name);

    template <typename T>
    OperatorRunner& addInput(const T* value, int64_t* dims, size_t dimSize, aclDataType type,
                             const char* name)
    {
        int64_t size = dims[0];
        for (size_t i = 1; i < dimSize; i++)
            size *= dims[i];

        size_t dataSize = size * sizeof(T);
        std::shared_ptr<uchar> ptr =
            mallocAndUpload(value, dataSize, AscendStream::Null(), AscendMat::defaultAllocator());

        AscendTensor tensor(ptr, dataSize, dims, dimSize, type, name);
        return addInput(tensor);
    }
    OperatorRunner& addOutput(AscendMat& mat, const char* name);
    OperatorRunner& reset();
    OperatorRunner& run(AscendStream& stream);
};

template <typename KERNEL_TYPE, typename TILING_TYPE, typename... ARGS>
void kernel_launch(KERNEL_TYPE kernel, AscendStream& stream, TILING_TYPE& tiling, ARGS... args)
{
    std::shared_ptr<uchar> tilingDevice =
        mallocAndUpload(&tiling, sizeof(TILING_TYPE), stream, AscendMat::defaultAllocator());
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    CV_ACL_SAFE_CALL(kernel(1, rawStream, tilingDevice.get(), args...));
    if (rawStream == nullptr)
    {
        stream.waitForCompletion();
    }
}

} // namespace cann
} // namespace cv

#endif // OPENCV_CANNOPS_CANN_CALL_HPP
