// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include "precomp.hpp"
#include "opencv2/core/private.hpp"
namespace cv
{
namespace cann
{
/*******************************Acl Error Checker*****************************/
void checkAclError(aclError err, const char* file, const int line, const char* func)
{
    if (ACL_SUCCESS != err)
    {
        const char* errMsg = aclGetRecentErrMsg();
        cv::error(cv::Error::StsError, errMsg == nullptr ? "" : errMsg, func, file, line);
    }
}

void checkAclPtr(void* ptr, const char* file, const int line, const char* func)
{
    if (nullptr == ptr)
    {
        const char* errMsg = aclGetRecentErrMsg();
        cv::error(cv::Error::StsError, errMsg == nullptr ? "" : errMsg, func, file, line);
    }
}

/******************************Acl Runtime Warpper****************************/
void aclrtMallocWarpper(void** data, size_t size)
{
    CV_ACL_SAFE_CALL(aclrtMalloc(data, size, ACL_MEM_MALLOC_HUGE_FIRST));
}

void aclrtFreeWarpper(void* data) { CV_ACL_SAFE_CALL(aclrtFree(data)); }

void aclrtMemcpyWarpper(std::shared_ptr<uchar>& dst, size_t offset, const void* src, size_t size,
                        AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(
            aclrtMemcpy(dst.get() + offset, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemcpyAsync(dst.get() + offset, size, src, size,
                                          ACL_MEMCPY_HOST_TO_DEVICE, rawStream));
        if (offset == 0)
            stream.addTensorHolder(dst);
    }
}

void aclrtMemcpyWarpper(void* dst, const std::shared_ptr<uchar>& src, size_t offset, size_t size,
                        AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(
            aclrtMemcpy(dst, size, src.get() + offset, size, ACL_MEMCPY_DEVICE_TO_HOST));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemcpyAsync(dst, size, src.get() + offset, size,
                                          ACL_MEMCPY_DEVICE_TO_HOST, rawStream));
        if (offset == 0)
            stream.addTensorHolder(src);
    }
}

void aclrtMemcpyWarpper(std::shared_ptr<uchar>& dst, size_t dstOffset,
                        const std::shared_ptr<uchar>& src, size_t srcOffset, size_t size,
                        AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtMemcpy(dst.get() + dstOffset, size, src.get() + srcOffset, size,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemcpyAsync(dst.get() + dstOffset, size, src.get() + srcOffset, size,
                                          ACL_MEMCPY_DEVICE_TO_DEVICE, rawStream));
        if (srcOffset == 0)
            stream.addTensorHolder(src);
        if (dstOffset == 0)
            stream.addTensorHolder(dst);
    }
}

void aclrtMemcpy2dWarpper(std::shared_ptr<uchar>& dst, size_t offset, size_t dpitch,
                          const void* src, size_t spitch, size_t width, size_t length,
                          AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtMemcpy2d(dst.get() + offset, dpitch, src, spitch, width, length,
                                       ACL_MEMCPY_HOST_TO_DEVICE));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemcpy2dAsync(dst.get() + offset, dpitch, src, spitch, width, length,
                                            ACL_MEMCPY_HOST_TO_DEVICE, rawStream));
        stream.addTensorHolder(dst);
    }
}

void aclrtMemcpy2dWarpper(void* dst, size_t dpitch, const std::shared_ptr<uchar>& src,
                          size_t offset, size_t spitch, size_t width, size_t length,
                          AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtMemcpy2d(dst, dpitch, src.get() + offset, spitch, width, length,
                                       ACL_MEMCPY_DEVICE_TO_HOST));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemcpy2dAsync(dst, dpitch, src.get() + offset, spitch, width, length,
                                            ACL_MEMCPY_DEVICE_TO_HOST, rawStream));
        stream.addTensorHolder(src);
    }
}

void aclrtMemsetWarpper(std::shared_ptr<uchar>& ptr, int32_t value, size_t count,
                        AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtMemset(ptr.get(), count, value, count));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemsetAsync(ptr.get(), count, value, count, rawStream));
        stream.addTensorHolder(ptr);
    }
}

aclDataType getACLType(int opencvdepth)
{
    switch (opencvdepth)
    {
        case CV_8S:
            return ACL_INT8;
        case CV_16S:
            return ACL_INT16;
        case CV_8U:
            return ACL_UINT8;
        case CV_16U:
            return ACL_UINT16;
        case CV_32S:
            return ACL_INT32;
        case CV_32F:
            return ACL_FLOAT;
        case CV_64F:
            return ACL_DOUBLE;
        case CV_16F:
            return ACL_FLOAT16;
        default:
            return ACL_DT_UNDEFINED;
    }
}

std::shared_ptr<uchar> mallocAndUpload(const void* data, size_t size, AscendStream& stream,
                                       AscendMat::Allocator* allocator)
{
    std::shared_ptr<uchar> ptr = allocator->allocate(size);
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);

    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtMemcpy(ptr.get(), size, data, size, ACL_MEMCPY_HOST_TO_DEVICE));
    else
        CV_ACL_SAFE_CALL(
            aclrtMemcpyAsync(ptr.get(), size, data, size, ACL_MEMCPY_HOST_TO_DEVICE, rawStream));
    return ptr;
}

/**************************Acl attribute preparation**************************/

OperatorRunner& OperatorRunner::reset()
{
    holder.clear();
    op.clear();
    for (auto desc : inputDesc_)
    {
        aclDestroyTensorDesc(desc);
    }
    for (auto desc : outputDesc_)
    {
        aclDestroyTensorDesc(desc);
    }
    for (auto buf : inputBuffers_)
    {
        CV_ACL_SAFE_CALL(aclDestroyDataBuffer(buf));
    }
    for (auto buf : outputBuffers_)
    {
        CV_ACL_SAFE_CALL(aclDestroyDataBuffer(buf));
    }
    if (opAttrInit)
        aclopDestroyAttr(opAttr_);
    inputDesc_.clear();
    outputDesc_.clear();
    inputBuffers_.clear();
    outputBuffers_.clear();
    opAttrInit = false;
    return *this;
}

OperatorRunner& OperatorRunner::setOp(const char* opName)
{
    reset();
    opAttr_ = CV_ACL_SAFE_CALL_PTR(aclopCreateAttr());
    opAttrInit = true;
    op = std::string(opName);
    return *this;
}

OperatorRunner& OperatorRunner::addAttr(float value, const char* name)
{
    CV_ACL_SAFE_CALL(aclopSetAttrFloat(opAttr_, name, value));
    return *this;
}

OperatorRunner& OperatorRunner::addAttr(const char* value, const char* name)
{
    CV_ACL_SAFE_CALL(aclopSetAttrString(opAttr_, name, value));
    return *this;
}

OperatorRunner& OperatorRunner::addAttr(int value, const char* name)
{
    CV_ACL_SAFE_CALL(aclopSetAttrInt(opAttr_, name, value));
    return *this;
}

OperatorRunner& OperatorRunner::addAttr(bool value, const char* name)
{
    CV_ACL_SAFE_CALL(aclopSetAttrBool(opAttr_, name, value));
    return *this;
}

OperatorRunner& OperatorRunner::addAttr(const int64_t* value, int size, const char* name)
{
    CV_ACL_SAFE_CALL(aclopSetAttrListInt(opAttr_, name, size, value));
    return *this;
}

OperatorRunner& OperatorRunner::addInput(AscendTensor& tensor)
{
    auto descPtr = CV_ACL_SAFE_CALL_PTR(
        aclCreateTensorDesc(tensor.dtype, tensor.dims.size(), &tensor.dims[0], tensor.format));
    if (descPtr != nullptr)
    {
        if (tensor.name != nullptr && strlen(tensor.name) != 0)
            aclSetTensorDescName(descPtr, tensor.name);
        inputDesc_.push_back(descPtr);
    }
    auto bufPtr = CV_ACL_SAFE_CALL_PTR(aclCreateDataBuffer(tensor.data.get(), tensor.dataSize));
    if (bufPtr != nullptr)
        inputBuffers_.push_back(bufPtr);
    holder.insert(tensor.data);
    return *this;
}

OperatorRunner& OperatorRunner::addOutput(AscendTensor& tensor)
{
    auto descPtr = CV_ACL_SAFE_CALL_PTR(
        aclCreateTensorDesc(tensor.dtype, tensor.dims.size(), &tensor.dims[0], tensor.format));
    if (descPtr != nullptr)
    {
        if (tensor.name != nullptr && strlen(tensor.name) != 0)
            aclSetTensorDescName(descPtr, tensor.name);
        outputDesc_.push_back(descPtr);
    }
    auto bufPtr = CV_ACL_SAFE_CALL_PTR(aclCreateDataBuffer(tensor.data.get(), tensor.dataSize));
    if (bufPtr != nullptr)
        outputBuffers_.push_back(bufPtr);
    holder.insert(tensor.data);
    return *this;
}

OperatorRunner& OperatorRunner::addInput(const AscendMat& mat, const char* name)
{
    AscendTensor tensor(mat, name);
    return addInput(tensor);
}

OperatorRunner& OperatorRunner::addOutput(AscendMat& mat, const char* name)
{
    AscendTensor tensor(mat, name);
    return addOutput(tensor);
}

OperatorRunner& OperatorRunner::addInput(const Scalar& sc, int type, const char* name)
{
    uchar rawData[32];
    cv::scalarToRawData(sc, rawData, type, 0);
    std::shared_ptr<uchar> scPtr = mallocAndUpload(
        rawData, (CV_ELEM_SIZE(type)), AscendStream::Null(), AscendMat::defaultAllocator());

    int64_t dims[] = {1, 1, 1, (CV_MAT_CN(type))};
    AscendTensor tensor(scPtr, (CV_ELEM_SIZE(type)), dims, sizeof(dims) / sizeof(dims[0]),
                        getACLType(CV_MAT_DEPTH(type)), name);
    return addInput(tensor);
}

OperatorRunner& OperatorRunner::run(AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    CV_ACL_SAFE_CALL(aclopCompileAndExecute(op.c_str(), inputDesc_.size(), inputDesc_.data(),
                                            inputBuffers_.data(), outputDesc_.size(),
                                            outputDesc_.data(), outputBuffers_.data(), opAttr_,
                                            ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, rawStream));
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtSynchronizeStream(rawStream));
    else
    {
        for (const auto& ptr : holder)
            stream.addTensorHolder(ptr);
    }
    return *this;
}

/********************************Ascend Tensor********************************/

AscendTensor::AscendTensor(std::shared_ptr<uchar> _data, size_t _dataSize, int64_t* _dims,
                           size_t _dimSize, aclDataType _dtype, const char* _name,
                           aclFormat _format)
    : name(_name), data(_data), dataSize(_dataSize), dtype(_dtype), format(_format)
{
    dims.assign(_dims, _dims + _dimSize);
}

AscendTensor::AscendTensor(const AscendMat& ascendMat, const char* _name, aclFormat _format)
    : name(_name), format(_format)
{
    data = ascendMat.data;
    // Ascend can't process with gaps in matrix.
    CV_Assert(ascendMat.isContinuous());
    dataSize = ascendMat.rows * ascendMat.cols * ascendMat.elemSize();

    switch (_format)
    {
        case ACL_FORMAT_NHWC:
        case ACL_FORMAT_ND:
            dims.resize(4);
            // Batch, default = 1.
            dims[0] = 1;
            // Default OpenCV image format = NHWC.
            dims[1] = ascendMat.rows;
            dims[2] = ascendMat.cols;
            dims[3] = ascendMat.channels();
            break;
        case ACL_FORMAT_NCHW:
            dims.resize(4);
            dims[0] = 1;
            dims[1] = ascendMat.channels();
            dims[2] = ascendMat.rows;
            dims[3] = ascendMat.cols;
            break;
        default:
            CV_Error(Error::StsBadArg, "Unknown/unsupported matrix format");
    }

    dtype = getACLType(ascendMat.depth());
}

/**********************************Device*************************************/
void setDevice(int device_id)
{
    aclrtContext context;
    CV_ACL_SAFE_CALL(aclrtSetDevice(device_id));
    CV_ACL_SAFE_CALL(aclrtCreateContext(&context, device_id));
}

void resetDevice() { CV_ACL_SAFE_CALL(aclrtResetDevice(getDevice())); }

int32_t getDevice()
{
    int32_t deviceId;
    CV_ACL_SAFE_CALL(aclrtGetDevice(&deviceId));
    return deviceId;
}

void initAcl() { CV_ACL_SAFE_CALL(aclInit(nullptr)); }

void finalizeAcl() { CV_ACL_SAFE_CALL(aclFinalize()); }

class DefaultDeviceInitializer
{
public:
    DefaultDeviceInitializer();
    ~DefaultDeviceInitializer();

    AscendStream& getNullAscendStream(int deviceId);

private:
    std::vector<Ptr<AscendStream>> streams_;
    Mutex streams_mtx_;
};

DefaultDeviceInitializer::DefaultDeviceInitializer() {}

DefaultDeviceInitializer::~DefaultDeviceInitializer() { streams_.clear(); }

AscendStream& DefaultDeviceInitializer::getNullAscendStream(int deviceId)
{
    AutoLock lock(streams_mtx_);

    if (streams_.empty())
    {
        uint32_t deviceCount;
        CV_ACL_SAFE_CALL(aclrtGetDeviceCount(&deviceCount));

        if (deviceCount > 0)
            streams_.resize(deviceCount);
    }

    CV_DbgAssert(deviceId >= 0 && deviceId < static_cast<int>(streams_.size()));

    if (streams_[deviceId].empty())
    {
        aclrtStream stream = nullptr;
        Ptr<AscendStream::Impl> impl = makePtr<AscendStream::Impl>(stream);
        streams_[deviceId] = Ptr<AscendStream>(new AscendStream(impl));
    }

    return *streams_[deviceId];
}

DefaultDeviceInitializer initializer;

/***********************************Event*************************************/
AscendEvent::Impl::Impl() : event(nullptr), ownEvent(true)
{
    CV_ACL_SAFE_CALL(aclrtCreateEvent(&event));
}

AscendEvent::Impl::Impl(aclrtEvent e) : event(e), ownEvent(false) {}

AscendEvent::Impl::~Impl()
{
    if (event && ownEvent)
    {
        CV_ACL_SAFE_CALL(aclrtDestroyEvent(event));
    }
}

aclrtEvent AscendEventAccessor::getEvent(const AscendEvent& event) { return event.impl_->event; }

AscendEvent AscendEventAccessor::wrapEvent(aclrtEvent event)
{
    return AscendEvent(makePtr<AscendEvent::Impl>(event));
}

AscendEvent::AscendEvent() { impl_ = makePtr<Impl>(); }

void AscendEvent::record(AscendStream& stream)
{
    CV_ACL_SAFE_CALL(aclrtRecordEvent(impl_->event, AscendStreamAccessor::getStream(stream)));
}

void AscendEvent::waitForComplete() const { CV_ACL_SAFE_CALL(aclrtSynchronizeEvent(impl_->event)); }

/************************************Stream***********************************/
void AscendStream::Impl::AddTensorHolder(const std::shared_ptr<uchar>& tensorData)
{
    tensorHolders.insert(tensorData);
}

AscendStream::Impl::Impl() : stream(nullptr), ownStream(true)
{
    CV_ACL_SAFE_CALL(aclrtCreateStream(&stream));
}

AscendStream::Impl::Impl(aclrtStream s) : stream(s), ownStream(false) {}

aclrtStream AscendStreamAccessor::getStream(const AscendStream& stream)
{
    return stream.impl_->stream;
}

AscendStream AscendStreamAccessor::wrapStream(aclrtStream stream)
{
    return AscendStream(makePtr<AscendStream::Impl>(stream));
}

AscendStream wrapStream(size_t AscendStreamAddress)
{
    return AscendStreamAccessor::wrapStream(reinterpret_cast<aclrtStream>(AscendStreamAddress));
}

AscendStream::AscendStream() { impl_ = makePtr<Impl>(); }

void AscendStream::waitForCompletion()
{
    CV_ACL_SAFE_CALL(aclrtSynchronizeStream(impl_->stream));
    impl_->tensorHolders.clear();
}

void AscendStream::waitAscendEvent(const AscendEvent& event)
{
    CV_ACL_SAFE_CALL(aclrtStreamWaitEvent(impl_->stream, AscendEventAccessor::getEvent(event)));
}

AscendStream& AscendStream::Null()
{
    const uint32_t deviceId = getDevice();
    return initializer.getNullAscendStream(deviceId);
}

void AscendStream::addTensorHolder(const std::shared_ptr<uchar>& holder)
{
    impl_->AddTensorHolder(holder);
}

} // namespace cann
} // namespace cv
