// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNOPS_CANN_HPP
#define OPENCV_CANNOPS_CANN_HPP

#include "opencv2/core.hpp"

/**
  @defgroup cannops Ascend-accelerated Computer Vision
  @{
    @defgroup canncore Core part
    @{
      @defgroup cann_struct Data Structures
      @defgroup cann_init Initialization and Information
    @}
  @}
 */

namespace cv
{
namespace cann
{
class AscendStream;

//! @addtogroup cann_struct
//! @{

//===================================================================================
// AscendMat
//===================================================================================

/** @brief Base storage class for NPU memory with reference counting.
 * AscendMat class has a similar interface with Mat and AscendMat, and work on [Ascend
 * NPU](https://www.hiascend.com/) backend.
 * @sa Mat cuda::GpuMat
 */
class AscendStream;
class CV_EXPORTS_W AscendMat
{
public:
    class CV_EXPORTS_W Allocator
    {
    public:
        virtual ~Allocator() {}
        // basic allocator
        virtual std::shared_ptr<uchar> allocate(size_t size) = 0;
        // allocator must fill data, step and refcount fields
        virtual bool allocate(AscendMat* mat, int rows, int cols, size_t elemSize) = 0;
    };

    /**
     * @brief Create default allocator for AscendMat. This allocator alloc memory from device for
     * specific size.
     */
    CV_WRAP static AscendMat::Allocator* defaultAllocator();

    /**
     * @brief Set allocator for AscendMat.
     * @param allocator
     */
    CV_WRAP static void setDefaultAllocator(AscendMat::Allocator* allocator);

    //! default constructor
    CV_WRAP explicit AscendMat(AscendMat::Allocator* allocator_ = AscendMat::defaultAllocator());

    //! constructs AscendMat of the specified size and type
    CV_WRAP AscendMat(int rows, int cols, int type,
                      AscendMat::Allocator* allocator = AscendMat::defaultAllocator());
    //! constructs AscendMat of the specified size and type
    CV_WRAP AscendMat(Size size, int type,
                      AscendMat::Allocator* allocator = AscendMat::defaultAllocator());

    //! constructs AscendMat and fills it with the specified value s
    CV_WRAP AscendMat(int rows, int cols, int type, Scalar& s,
                      AscendMat::Allocator* allocator = AscendMat::defaultAllocator());
    //! constructs AscendMat and fills it with the specified value s
    CV_WRAP AscendMat(Size size, int type, Scalar& s,
                      AscendMat::Allocator* allocator = AscendMat::defaultAllocator());

    //! copy constructor
    CV_WRAP AscendMat(const AscendMat& m);

    //! constructs AscendMat by crop a certain area from another
    CV_WRAP AscendMat(InputArray _m, const Rect& roi);
    CV_WRAP AscendMat(InputArray _m, const Rect& roi, AscendStream& stream);

    //! builds AscendMat from host memory (Blocking call)
    CV_WRAP explicit AscendMat(InputArray arr, AscendStream& stream,
                               AscendMat::Allocator* allocator = AscendMat::defaultAllocator());

    //! assignment operators
    AscendMat& operator=(const AscendMat& m);

    //! sets some of the AscendMat elements to s (Blocking call)
    CV_WRAP AscendMat& setTo(const Scalar& s);
    //! sets some of the AscendMat elements to s (Non-Blocking call)
    CV_WRAP AscendMat& setTo(const Scalar& s, AscendStream& stream);

    //! sets all of the AscendMat elements to float (Blocking call)
    CV_WRAP AscendMat& setTo(float sc);

    //! sets all of the AscendMat elements to float (Non-Blocking call)
    CV_WRAP AscendMat& setTo(float sc, AscendStream& stream);

    //! swaps with other smart pointer
    CV_WRAP void swap(AscendMat& mat);

    //! allocates new AscendMat data unless the AscendMat already has specified size and type
    CV_WRAP void create(int rows, int cols, int type);

    //! upload host memory data to AscendMat (Blocking call)
    CV_WRAP void upload(InputArray arr);
    //! upload host memory data to AscendMat (Non-Blocking call)
    CV_WRAP void upload(InputArray arr, AscendStream& stream);

    //! download data from AscendMat to host (Blocking call)
    CV_WRAP void download(OutputArray dst) const;
    //! download data from AscendMat to host (Non-Blocking call)
    CV_WRAP void download(OutputArray dst, AscendStream& stream) const;

    //! converts AscendMat to another datatype (Blocking call)
    CV_WRAP void convertTo(CV_OUT AscendMat& dst, int rtype) const;

    //! converts AscendMat to another datatype (Non-Blocking call)
    CV_WRAP void convertTo(CV_OUT AscendMat& dst, int rtype, AscendStream& stream) const;

    //! converts AscendMat to another datatype, dst mat is allocated. (Non-Blocking call)
    CV_WRAP void convertTo(CV_OUT AscendMat& dst, AscendStream& stream) const;

    //! returns true iff the AscendMat data is continuous
    //! (i.e. when there are no gaps between successive rows)
    CV_WRAP bool isContinuous() const;

    //! returns element size in bytes
    CV_WRAP size_t elemSize() const;

    //! returns the size of element channel in bytes
    CV_WRAP size_t elemSize1() const;

    //! returns element type
    CV_WRAP int type() const;

    //! returns element type
    CV_WRAP int depth() const;

    //! returns number of channels
    CV_WRAP int channels() const;

    //! returns step/elemSize1()
    CV_WRAP size_t step1() const;

    //! returns AscendMat size : width == number of columns, height == number of rows
    CV_WRAP Size size() const;

    //! returns true if AscendMat data is NULL
    CV_WRAP bool empty() const;

    //! internal use method: updates the continuity flag
    CV_WRAP void updateContinuityFlag();

    /*! includes several bit-fields:
     - the magic signature
     - continuity flag
     - depth
     - number of channels
     */
    int flags;

    //! the number of rows and columns
    int rows, cols;

    //! a distance between successive rows in bytes; includes the gap if any
    CV_PROP size_t step;

    //! pointer to the data
    std::shared_ptr<uchar> data;

    //! helper fields used in locateROI and adjustROI
    uchar* datastart;
    const uchar* dataend;

    //! allocator
    Allocator* allocator;
};

class AscendStream;
class AscendStreamAccessor;
class AscendEvent;
class AscendEventAccessor;
class DefaultDeviceInitializer;

//===================================================================================
// AscendStream
//===================================================================================

/** @brief In AscendCL Stream(AscendStream) is a task queue. Stream is used to manage the
 * parallelism of tasks. The tasks inside a Stream are executed sequentially, that is, the Stream
 * executes sequentially according to the sent tasks; the tasks in different Streams are executed in
 * parallel.
 *
 * All Non-blocking functions should pass parameter stream, These function returns immediately after
 * the task is submitted. Caller should wait stream until completion.
 *
 * Blocking functions implicityly use the default stream, and synchronize stream before function
 * return.
 * @sa cuda::Stream
 */

// TODO: Stream is defined in namespace cuda, and pybind code does not use a namespace of stream,
// change stream name to AscendStream to avoid confilct.
class CV_EXPORTS_W AscendStream
{
public:
    CV_WRAP AscendStream();

    //! blocks the current CPU thread until all operations in the stream are complete.
    CV_WRAP void waitForCompletion();

    //! blocks the current CPU thread until event trigger.
    CV_WRAP void waitAscendEvent(const cv::cann::AscendEvent& event);

    /**
     * @brief return default AscendStream object for default Acl stream.
     */
    CV_WRAP static AscendStream& Null();

    // acl symbols CANNOT used in any hpp files. Use a inner class to avoid acl symbols defined in
    // hpp.
    class Impl;

    void addTensorHolder(const std::shared_ptr<uchar>& holder);

private:
    Ptr<Impl> impl_;
    AscendStream(const Ptr<Impl>& impl);

    friend class AscendStreamAccessor;
    friend class DefaultDeviceInitializer;
};

/**
 * @brief AscendEvent to synchronize between different streams.
 */
class CV_EXPORTS_W AscendEvent
{
public:
    CV_WRAP AscendEvent();

    //! records an event
    CV_WRAP void record(AscendStream& stream);

    //! waits for an event to complete
    CV_WRAP void waitForComplete() const;

    class Impl;

private:
    Ptr<Impl> impl_;
    AscendEvent(const Ptr<Impl>& impl);

    friend class AscendEventAccessor;
};

/** @brief Bindings overload to create a Stream object from the address stored in an existing CANN
 * Runtime API stream pointer (aclrtStream).
 * @param AscendStreamAddress Memory address stored in a CANN Runtime API stream pointer
 * (aclrtStream). The created Stream object does not perform any allocation or deallocation and
 * simply wraps existing raw CANN Runtime API stream pointer.
 * @note Overload for generation of bindings only, not exported or intended for use internally fro
 * C++.
 */
CV_EXPORTS_W AscendStream wrapStream(size_t AscendStreamAddress);

//! @} cann_struct

//===================================================================================
// Initialization & Info
//===================================================================================

//! @addtogroup cann_init
//! @{

//! Get Ascend matrix object from Input array, upload matrix memory if need. (Non-Blocking call)
AscendMat getInputMat(InputArray src, AscendStream& stream);

//! Get Ascend matrix object from Output array, upload matrix memory if need.
AscendMat getOutputMat(OutputArray dst, int rows, int cols, int type, AscendStream& stream);

//! Sync output matrix to Output array, download matrix memory if need.
void syncOutput(const AscendMat& dst, OutputArray _dst, AscendStream& stream);

/**
 * @brief Choose Ascend npu device.
 */
CV_EXPORTS_W void setDevice(int device);

/**
 * @brief Clear all context created in current Ascend device.
 */
CV_EXPORTS_W void resetDevice();

/**
 * @brief Get current Ascend device.
 */
CV_EXPORTS_W int32_t getDevice();

/**
 * @brief init AscendCL.
 */
CV_EXPORTS_W void initAcl();

/**
 * @brief finalize AscendCL.
 * @note finalizeAcl only can be called once for a process. Call this function after all AscendCL
 * options finished.
 */
CV_EXPORTS_W void finalizeAcl();

/**
 * @brief init DVPP system.
 * @note The DVPP interfaces used are all version V2.
 * Supported devices: Atlas Inference Series products, Atlas 200/500 A2 Inference products and
 * Atlas A2 Training Series products/Atlas 300I A2 Inference products
 */
CV_EXPORTS_W void initDvpp();

/**
 * @brief finalize DVPP system.
 * @note Supported devices: Atlas Inference Series products, Atlas 200/500 A2 Inference products and
 * Atlas A2 Training Series products/Atlas 300I A2 Inference products
 */
CV_EXPORTS_W void finalizeDvpp();

//! @} cann_init

} // namespace cann
} // namespace cv

#include "opencv2/cann.inl.hpp"

#endif // OPENCV_CANNOPS_CANN_HPP
