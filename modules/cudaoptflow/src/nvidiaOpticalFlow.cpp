//
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//M*/
#include "precomp.hpp"

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

cv::Ptr<cv::cuda::NvidiaOpticalFlow_1_0> cv::cuda::NvidiaOpticalFlow_1_0::create
    (cv::Size, NVIDIA_OF_PERF_LEVEL, bool, bool, bool, int, Stream&, Stream&) {
    throw_no_cuda(); return cv::Ptr<cv::cuda::NvidiaOpticalFlow_1_0>(); }

cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> cv::cuda::NvidiaOpticalFlow_2_0::create(
    cv::Size, NVIDIA_OF_PERF_LEVEL, NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE, NVIDIA_OF_HINT_VECTOR_GRID_SIZE,
    bool, bool, bool, int, Stream&, Stream&) {
    throw_no_cuda(); return cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0>();
}

cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> cv::cuda::NvidiaOpticalFlow_2_0::create(
    cv::Size, std::vector<Rect>, NVIDIA_OF_PERF_LEVEL, NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE, NVIDIA_OF_HINT_VECTOR_GRID_SIZE,
    bool, bool, bool, int, Stream&, Stream&) {
    throw_no_cuda(); return cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0>();
}

#elif !defined HAVE_NVIDIA_OPTFLOW

cv::Ptr<cv::cuda::NvidiaOpticalFlow_1_0> cv::cuda::NvidiaOpticalFlow_1_0::create(
    cv::Size, NVIDIA_OF_PERF_LEVEL, bool, bool, bool, int, Stream&, Stream&)
{
    CV_Error(cv::Error::HeaderIsNull, "OpenCV was build without NVIDIA OpticalFlow support");
}

cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> cv::cuda::NvidiaOpticalFlow_2_0::create(
    cv::Size, NVIDIA_OF_PERF_LEVEL, NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE, NVIDIA_OF_HINT_VECTOR_GRID_SIZE,
    bool, bool, bool, int, Stream&, Stream&)
{
    CV_Error(cv::Error::HeaderIsNull, "OpenCV was build without NVIDIA OpticalFlow support");
}

cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> cv::cuda::NvidiaOpticalFlow_2_0::create(
    cv::Size, std::vector<Rect>, NVIDIA_OF_PERF_LEVEL, NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE, NVIDIA_OF_HINT_VECTOR_GRID_SIZE,
    bool, bool, bool, int, Stream&, Stream&)
{
    CV_Error(cv::Error::HeaderIsNull, "OpenCV was build without NVIDIA OpticalFlow support");
}

#else

#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

namespace cv { namespace cuda { namespace device { namespace optflow_nvidia
{
void FlowUpsample(void* srcDevPtr, uint32_t nSrcWidth, uint32_t nSrcPitch, uint32_t nSrcHeight,
    void* dstDevPtr, uint32_t nDstWidth, uint32_t nDstPitch, uint32_t nDstHeight,
    uint32_t nScaleFactor);
}}}}

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#define HMODULE void *
#define _stricmp strcasecmp
#include <dlfcn.h>
#endif

//macro for dll loading
#if defined(_WIN64)
#define OF_MODULENAME TEXT("nvofapi64.dll")
#define CUDA_MODULENAME TEXT("nvcuda.dll")
#elif defined(_WIN32)
#define OF_MODULENAME TEXT("nvofapi.dll")
#define CUDA_MODULENAME TEXT("nvcuda.dll")
#else
#define OF_MODULENAME "libnvidia-opticalflow.so.1"
#define CUDA_MODULENAME "libcuda.so"
#endif

#define NVOF_API_CALL(nvOFAPI)                                                                      \
    do                                                                                              \
    {                                                                                               \
        NV_OF_STATUS errorCode = nvOFAPI;                                                           \
        std::ostringstream errorLog;                                                                \
        if(errorCode != NV_OF_SUCCESS)                                                              \
        {                                                                                           \
            switch (errorCode)                                                                      \
            {                                                                                       \
            case 1:                                                                                 \
                errorLog << #nvOFAPI << " returned error " << (unsigned int)errorCode;              \
                errorLog << ":NV_OF_ERR_OF_NOT_AVAILABLE";                                          \
                CV_Error(Error::StsBadFunc, errorLog.str());                                        \
                break;                                                                              \
            case 2:                                                                                 \
                errorLog << #nvOFAPI << " returned error " << (unsigned int)errorCode;              \
                errorLog << ":NV_OF_ERR_UNSUPPORTED_DEVICE";                                        \
                CV_Error(Error::StsBadArg, errorLog.str());                                         \
                break;                                                                              \
            case 3:                                                                                 \
                errorLog << #nvOFAPI << " returned error " << (unsigned int)errorCode;              \
                errorLog << ":NV_OF_ERR_DEVICE_DOES_NOT_EXIST";                                     \
                CV_Error(Error::StsBadArg, errorLog.str());                                         \
                break;                                                                              \
            case 4:                                                                                 \
                errorLog << #nvOFAPI << " returned error " << (unsigned int)errorCode;              \
                errorLog << ":NV_OF_ERR_INVALID_PTR";                                               \
                CV_Error(Error::StsNullPtr, errorLog.str());                                        \
                break;                                                                              \
            case 5:                                                                                 \
                errorLog << #nvOFAPI << " returned error " << (unsigned int)errorCode;              \
                errorLog << ":NV_OF_ERR_INVALID_PARAM";                                             \
                CV_Error(Error::StsBadArg, errorLog.str());                                         \
                break;                                                                              \
            case 6:                                                                                 \
                errorLog << #nvOFAPI << " returned error " << (unsigned int)errorCode;              \
                errorLog << ":NV_OF_ERR_INVALID_CALL";                                              \
                CV_Error(Error::BadCallBack, errorLog.str());                                       \
                break;                                                                              \
            case 7:                                                                                 \
                errorLog << #nvOFAPI << " returned error " << (unsigned int)errorCode;              \
                errorLog << ":NV_OF_ERR_INVALID_VERSION";                                           \
                CV_Error(Error::StsError, errorLog.str());                                          \
                break;                                                                              \
            case 8:                                                                                 \
                errorLog << #nvOFAPI << " returned error " << (unsigned int)errorCode;              \
                errorLog << ":NV_OF_ERR_OUT_OF_MEMORY";                                             \
                CV_Error(Error::StsNoMem, errorLog.str());                                          \
                break;                                                                              \
            case 9:                                                                                 \
                errorLog << #nvOFAPI << " returned error " << (unsigned int)errorCode;              \
                errorLog << ":NV_OF_ERR_NOT_INITIALIZED";                                           \
                CV_Error(Error::StsBadArg, errorLog.str());                                         \
                break;                                                                              \
            case 10:                                                                                \
                errorLog << #nvOFAPI << " returned error " << (unsigned int)errorCode;              \
                errorLog << ":NV_OF_ERR_UNSUPPORTED_FEATURE";                                       \
                CV_Error(Error::StsBadArg, errorLog.str());                                         \
                break;                                                                              \
            case 11:                                                                                \
                errorLog << #nvOFAPI << " returned error " << (unsigned int)errorCode;              \
                errorLog << ":NV_OF_ERR_GENERIC";                                                   \
                CV_Error(Error::StsInternal, errorLog.str());                                       \
                break;                                                                              \
            default:                                                                                \
                break;                                                                              \
            }                                                                                       \
        }                                                                                           \
    } while (0)                                                                                     \

using namespace std;
using namespace cv;
using namespace cv::cuda;

namespace
{
class LoadNvidiaModules
{
private:
    typedef int(*PFNCudaCuCtxGetCurrent)(CUcontext*);
    typedef NV_OF_STATUS(NVOFAPI *PFNNvOFAPICreateInstanceCuda)
        (int apiVer, NV_OF_CUDA_API_FUNCTION_LIST* cudaOf);

    PFNCudaCuCtxGetCurrent m_cudaDriverAPIGetCurrentCtx;
    PFNNvOFAPICreateInstanceCuda m_NvOFAPICreateInstanceCuda;
    HMODULE m_hOFModule;
    HMODULE m_hCudaModule;
    bool m_isFailed;

    LoadNvidiaModules() :
        m_cudaDriverAPIGetCurrentCtx(NULL),
        m_NvOFAPICreateInstanceCuda(NULL),
        m_isFailed(false)
    {
//Loading Cuda Library
#if defined(_WIN32) || defined(_WIN64)
        HMODULE hCudaModule = LoadLibrary(CUDA_MODULENAME);
#else
        void *hCudaModule = dlopen(CUDA_MODULENAME, RTLD_LAZY);
#endif

        if (hCudaModule == NULL)
        {
            m_isFailed = true;
            CV_Error(Error::StsBadFunc, "Cannot find Cuda library.");
        }
        m_hCudaModule = hCudaModule;

#if defined(_WIN32)
        m_cudaDriverAPIGetCurrentCtx = (PFNCudaCuCtxGetCurrent)GetProcAddress(m_hCudaModule, "cuCtxGetCurrent");
#else
        m_cudaDriverAPIGetCurrentCtx = (PFNCudaCuCtxGetCurrent)dlsym(m_hCudaModule, "cuCtxGetCurrent");
#endif
        if (!m_cudaDriverAPIGetCurrentCtx)
        {
            m_isFailed = true;
            CV_Error(Error::StsBadFunc,
                "Cannot find Cuda Driver API : cuCtxGetCurrent() entry in Cuda library");
        }

//Loading Optical Flow Library
#if defined(_WIN32) || defined(_WIN64)
        HMODULE hOFModule = LoadLibrary(OF_MODULENAME);
#else
        void *hOFModule = dlopen(OF_MODULENAME, RTLD_LAZY);
#endif

        if (hOFModule == NULL)
        {
            m_isFailed = true;
            CV_Error(Error::StsBadFunc, "Cannot find NvOF library.");
        }
        m_hOFModule = hOFModule;

#if defined(_WIN32)
        m_NvOFAPICreateInstanceCuda = (PFNNvOFAPICreateInstanceCuda)GetProcAddress(m_hOFModule, "NvOFAPICreateInstanceCuda");
#else
        m_NvOFAPICreateInstanceCuda = (PFNNvOFAPICreateInstanceCuda)dlsym(m_hOFModule, "NvOFAPICreateInstanceCuda");
#endif
        if (!m_NvOFAPICreateInstanceCuda)
        {
            m_isFailed = true;
            CV_Error(Error::StsBadFunc,
                "Cannot find NvOFAPICreateInstanceCuda() entry in NVOF library");
        }
    };

    ~LoadNvidiaModules()
    {
        if (NULL != m_hCudaModule)
        {
#if defined(_WIN32) || defined(_WIN64)
            FreeLibrary(m_hCudaModule);
#else
            dlclose(m_hCudaModule);
#endif
        }
        if (NULL != m_hOFModule)
        {
#if defined(_WIN32) || defined(_WIN64)
            FreeLibrary(m_hOFModule);
#else
            dlclose(m_hOFModule);
#endif
        }
        m_hCudaModule = NULL;
        m_hOFModule = NULL;
        m_cudaDriverAPIGetCurrentCtx = NULL;
        m_NvOFAPICreateInstanceCuda = NULL;
    }

public:
    static LoadNvidiaModules& Init()
    {
        static LoadNvidiaModules LoadLibraryObj;
        if (LoadLibraryObj.m_isFailed)
            CV_Error(Error::StsError, "Can't initialize LoadNvidiaModules Class Object");
        return LoadLibraryObj;
    }

    PFNCudaCuCtxGetCurrent GetCudaLibraryFunctionPtr() { return m_cudaDriverAPIGetCurrentCtx; }
    PFNNvOFAPICreateInstanceCuda GetOFLibraryFunctionPtr() { return m_NvOFAPICreateInstanceCuda; }
};

class NvidiaOpticalFlowImpl : public cv::cuda::NvidiaOpticalFlow_1_0
{
private:
    int m_width;
    int m_height;
    NV_OF_PERF_LEVEL m_preset;
    bool m_enableTemporalHints;
    bool m_enableExternalHints;
    bool m_enableCostBuffer;
    int m_gpuId;
    Stream m_inputStream;
    Stream m_outputStream;

    CUcontext m_cuContext;
    NV_OF_BUFFER_FORMAT m_format;

    NV_OF_OUTPUT_VECTOR_GRID_SIZE m_gridSize;

    NV_OF_BUFFER_DESCRIPTOR m_inputBufferDesc;
    NV_OF_BUFFER_DESCRIPTOR m_outputBufferDesc;
    NV_OF_BUFFER_DESCRIPTOR m_hintBufferDesc;
    NV_OF_BUFFER_DESCRIPTOR m_costBufferDesc;

    int m_outputElementSize;
    int m_costBufElementSize;
    int m_hintBufElementSize;

    NV_OF_INIT_PARAMS m_initParams;

    std::unique_ptr<NV_OF_CUDA_API_FUNCTION_LIST> m_ofAPI;
    NvOFHandle m_hOF; //nvof handle

    NvOFGPUBufferHandle m_hInputBuffer;
    NvOFGPUBufferHandle m_hReferenceBuffer;
    NvOFGPUBufferHandle m_hOutputBuffer;
    NvOFGPUBufferHandle m_hHintBuffer;
    NvOFGPUBufferHandle m_hCostBuffer;

    CUdeviceptr m_frame0cuDevPtr;
    CUdeviceptr m_frame1cuDevPtr;
    CUdeviceptr m_flowXYcuDevPtr;
    CUdeviceptr m_hintcuDevPtr;
    CUdeviceptr m_costcuDevPtr;

    NV_OF_CUDA_BUFFER_STRIDE_INFO m_inputBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_referenceBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_outputBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_hintBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_costBufferStrideInfo;

    NV_OF_CUDA_API_FUNCTION_LIST* GetAPI()
    {
        return  m_ofAPI.get();
    }

    NvOFHandle GetHandle() { return m_hOF; }

public:
    NvidiaOpticalFlowImpl(cv::Size imageSize, NV_OF_PERF_LEVEL perfPreset, bool bEnableTemporalHints,
        bool bEnableExternalHints, bool bEnableCostBuffer, int gpuId, Stream inputStream, Stream outputStream);

    virtual void calc(InputArray inputImage, InputArray referenceImage,
        InputOutputArray flow, Stream& stream = Stream::Null(),
        InputArray hint = cv::noArray(), OutputArray cost = cv::noArray());


    virtual void collectGarbage();

    virtual void upSampler(InputArray flow, cv::Size imageSize,
        int gridSize, InputOutputArray upsampledFlow);

    virtual int getGridSize() const { return m_gridSize; }

    ~NvidiaOpticalFlowImpl();
};

NvidiaOpticalFlowImpl::NvidiaOpticalFlowImpl(
    cv::Size imageSize, NV_OF_PERF_LEVEL perfPreset, bool bEnableTemporalHints,
    bool bEnableExternalHints, bool bEnableCostBuffer, int gpuId,
    Stream inputStream, Stream outputStream) :
    m_width(imageSize.width), m_height(imageSize.height), m_preset(perfPreset),
    m_enableTemporalHints((NV_OF_BOOL)bEnableTemporalHints),
    m_enableExternalHints((NV_OF_BOOL)bEnableExternalHints),
    m_enableCostBuffer((NV_OF_BOOL)bEnableCostBuffer), m_gpuId(gpuId),
    m_inputStream(inputStream), m_outputStream(outputStream),
    m_cuContext(nullptr), m_format(NV_OF_BUFFER_FORMAT_GRAYSCALE8),
    m_gridSize(NV_OF_OUTPUT_VECTOR_GRID_SIZE_4)
{
    LoadNvidiaModules& LoadNvidiaModulesObj = LoadNvidiaModules::Init();

    int nGpu = 0;

    cuSafeCall(cudaGetDeviceCount(&nGpu));
    if (m_gpuId < 0 || m_gpuId >= nGpu)
    {
        CV_Error(Error::StsBadArg, "Invalid GPU Ordinal");
    }

    cuSafeCall(cudaSetDevice(m_gpuId));
    cuSafeCall(cudaFree(m_cuContext));

    cuSafeCall(LoadNvidiaModulesObj.GetCudaLibraryFunctionPtr()(&m_cuContext));

    if (m_gridSize != NV_OF_OUTPUT_VECTOR_GRID_SIZE_4)
    {
        CV_Error(Error::StsBadArg, "Unsupported grid size");
    }

    auto nOutWidth = (m_width + m_gridSize - 1) / m_gridSize;
    auto nOutHeight = (m_height + m_gridSize - 1) / m_gridSize;

    auto outBufFmt = NV_OF_BUFFER_FORMAT_SHORT2;

    memset(&m_inputBufferDesc, 0, sizeof(m_inputBufferDesc));
    m_inputBufferDesc.width = m_width;
    m_inputBufferDesc.height = m_height;
    m_inputBufferDesc.bufferFormat = m_format;
    m_inputBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;

    memset(&m_outputBufferDesc, 0, sizeof(m_outputBufferDesc));
    m_outputBufferDesc.width = nOutWidth;
    m_outputBufferDesc.height = nOutHeight;
    m_outputBufferDesc.bufferFormat = outBufFmt;
    m_outputBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
    m_outputElementSize = sizeof(NV_OF_FLOW_VECTOR);

    if (m_enableExternalHints)
    {
        memset(&m_hintBufferDesc, 0, sizeof(m_hintBufferDesc));
        m_hintBufferDesc.width = nOutWidth;
        m_hintBufferDesc.height = nOutHeight;
        m_hintBufferDesc.bufferFormat = outBufFmt;
        m_hintBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_HINT;
        m_hintBufElementSize = m_outputElementSize;
    }

    if (m_enableCostBuffer)
    {
        memset(&m_costBufferDesc, 0, sizeof(m_costBufferDesc));
        m_costBufferDesc.width = nOutWidth;
        m_costBufferDesc.height = nOutHeight;
        m_costBufferDesc.bufferFormat = NV_OF_BUFFER_FORMAT_UINT;
        m_costBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_COST;
        m_costBufElementSize = sizeof(int);
    }

    m_ofAPI.reset(new NV_OF_CUDA_API_FUNCTION_LIST());

    NVOF_API_CALL(LoadNvidiaModulesObj.GetOFLibraryFunctionPtr()(NV_OF_API_VERSION, m_ofAPI.get()));
    NVOF_API_CALL(GetAPI()->nvCreateOpticalFlowCuda(m_cuContext, &m_hOF));

    memset(&m_initParams, 0, sizeof(m_initParams));
    m_initParams.width = m_inputBufferDesc.width;
    m_initParams.height = m_inputBufferDesc.height;
    m_initParams.enableExternalHints = (NV_OF_BOOL)m_enableExternalHints;
    m_initParams.enableOutputCost = (NV_OF_BOOL)m_enableCostBuffer;
    m_initParams.hintGridSize = (NV_OF_BOOL)m_enableExternalHints == NV_OF_TRUE ?
        NV_OF_HINT_VECTOR_GRID_SIZE_4 : NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    m_initParams.outGridSize = m_gridSize;
    m_initParams.mode = NV_OF_MODE_OPTICALFLOW;
    m_initParams.perfLevel = m_preset;

    NVOF_API_CALL(GetAPI()->nvOFInit(GetHandle(), &m_initParams));

    if (m_inputStream || m_outputStream)
    {
        NVOF_API_CALL(GetAPI()->nvOFSetIOCudaStreams(GetHandle(),
            StreamAccessor::getStream(m_inputStream), StreamAccessor::getStream(m_outputStream)));
    }

    //Input Buffer 1
    NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(GetHandle(),
        &m_inputBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hInputBuffer));
    m_frame0cuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hInputBuffer);
    NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
        m_hInputBuffer, &m_inputBufferStrideInfo));

    //Input Buffer 2
    NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(GetHandle(),
        &m_inputBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hReferenceBuffer));
    m_frame1cuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hReferenceBuffer);
    NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
        m_hReferenceBuffer, &m_referenceBufferStrideInfo));

    //Output Buffer
    NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(GetHandle(),
        &m_outputBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hOutputBuffer));
    m_flowXYcuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hOutputBuffer);
    NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
        m_hOutputBuffer, &m_outputBufferStrideInfo));

    //Hint Buffer
    if (m_enableExternalHints)
    {
        NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(GetHandle(),
            &m_hintBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hHintBuffer));
        m_hintcuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hHintBuffer);
        NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
            m_hHintBuffer, &m_hintBufferStrideInfo));
    }

    //Cost Buffer
    if (m_enableCostBuffer)
    {
        NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(GetHandle(),
            &m_costBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hCostBuffer));
        m_costcuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hCostBuffer);
        NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
            m_hCostBuffer, &m_costBufferStrideInfo));
    }
}

void NvidiaOpticalFlowImpl::calc(InputArray _frame0, InputArray _frame1, InputOutputArray _flow,
    Stream& stream, InputArray hint, OutputArray cost)
{
    if (stream && !m_inputStream)
    {
        m_inputStream = stream;
        NVOF_API_CALL(GetAPI()->nvOFSetIOCudaStreams(GetHandle(),
            StreamAccessor::getStream(m_inputStream), StreamAccessor::getStream(m_outputStream)));
    }

    GpuMat frame0GpuMat(_frame0.size(), _frame0.type(), (void*)m_frame0cuDevPtr,
        m_inputBufferStrideInfo.strideInfo[0].strideXInBytes);
    GpuMat frame1GpuMat(_frame1.size(), _frame1.type(), (void*)m_frame1cuDevPtr,
        m_referenceBufferStrideInfo.strideInfo[0].strideXInBytes);
    GpuMat flowXYGpuMat(Size((m_width + m_gridSize - 1) / m_gridSize,
        (m_height + m_gridSize - 1) / m_gridSize), CV_16SC2,
        (void*)m_flowXYcuDevPtr, m_outputBufferStrideInfo.strideInfo[0].strideXInBytes);

    //check whether frame0 is Mat or GpuMat
    if (_frame0.isMat())
    {
        //Get Mats from InputArrays
        Mat __frame0 = _frame0.getMat();
        frame0GpuMat.upload(__frame0, m_inputStream);
    }
    else if (_frame0.isGpuMat())
    {
        //Get GpuMats from InputArrays
        GpuMat __frame0 = _frame0.getGpuMat();
        __frame0.copyTo(frame0GpuMat, m_inputStream);
    }
    else
    {
        CV_Error(Error::StsBadArg,
            "Incorrect input. Pass input image (frame0) as Mat or GpuMat");
    }

    //check whether frame1 is Mat or GpuMat
    if (_frame1.isMat())
    {
        //Get Mats from InputArrays
        Mat __frame1 = _frame1.getMat();
        frame1GpuMat.upload(__frame1, m_inputStream);
    }
    else if (_frame1.isGpuMat())
    {
        //Get GpuMats from InputArrays
        GpuMat __frame1 = _frame1.getGpuMat();
        __frame1.copyTo(frame1GpuMat, m_inputStream);
    }
    else
    {
        CV_Error(Error::StsBadArg,
            "Incorrect input. Pass reference image (frame1) as Mat or GpuMat");
    }

    if (m_enableExternalHints)
    {
        GpuMat hintGpuMat(hint.size(), hint.type(), (void*)m_hintcuDevPtr,
            m_hintBufferStrideInfo.strideInfo[0].strideXInBytes);

        if (hint.isMat())
        {
            //Get Mat from InputArray hint
            Mat _hint = hint.getMat();
            hintGpuMat.upload(_hint, m_inputStream);
        }
        else if(hint.isGpuMat())
        {
            //Get GpuMat from InputArray hint
            GpuMat _hint = hint.getGpuMat();
            _hint.copyTo(hintGpuMat, m_inputStream);
        }
        else
        {
            CV_Error(Error::StsBadArg,"Incorrect hint buffer passed. Pass Mat or GpuMat");
        }
    }

    //Execute Call
    NV_OF_EXECUTE_INPUT_PARAMS exeInParams;
    NV_OF_EXECUTE_OUTPUT_PARAMS exeOutParams;
    memset(&exeInParams, 0, sizeof(exeInParams));
    exeInParams.inputFrame = m_hInputBuffer;
    exeInParams.referenceFrame = m_hReferenceBuffer;
    exeInParams.disableTemporalHints = (NV_OF_BOOL)m_enableTemporalHints == NV_OF_TRUE ?
        NV_OF_FALSE : NV_OF_TRUE;
    exeInParams.externalHints = m_initParams.enableExternalHints == NV_OF_TRUE ?
        m_hHintBuffer : nullptr;
    memset(&exeOutParams, 0, sizeof(exeOutParams));
    exeOutParams.outputBuffer = m_hOutputBuffer;
    exeOutParams.outputCostBuffer = m_initParams.enableOutputCost == NV_OF_TRUE ?
        m_hCostBuffer : nullptr;
    NVOF_API_CALL(GetAPI()->nvOFExecute(GetHandle(), &exeInParams, &exeOutParams));

    if (_flow.isMat())
        flowXYGpuMat.download(_flow, m_outputStream);
    else if(_flow.isGpuMat())
        flowXYGpuMat.copyTo(_flow, m_outputStream);
    else
        CV_Error(Error::StsBadArg, "Incorrect flow buffer passed. Pass Mat or GpuMat");

    if (m_enableCostBuffer)
    {
        GpuMat costGpuMat(Size((m_width + m_gridSize - 1) / m_gridSize,
            (m_height + m_gridSize - 1) / m_gridSize), CV_32SC1, (void*)m_costcuDevPtr,
            m_costBufferStrideInfo.strideInfo[0].strideXInBytes);

        if (cost.isMat())
            costGpuMat.download(cost, m_outputStream);
        else if(cost.isGpuMat())
            costGpuMat.copyTo(cost, m_outputStream);
        else
            CV_Error(Error::StsBadArg, "Incorrect cost buffer passed. Pass Mat or GpuMat");
    }
    m_outputStream.waitForCompletion();
}

void NvidiaOpticalFlowImpl::collectGarbage()
{
    if (m_hInputBuffer)
    {
        NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hInputBuffer));
        m_hInputBuffer = nullptr;
    }
    if (m_hReferenceBuffer)
    {
        NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hReferenceBuffer));
        m_hReferenceBuffer = nullptr;
    }
    if (m_hOutputBuffer)
    {
        NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hOutputBuffer));
        m_hOutputBuffer = nullptr;
    }
    if (m_enableExternalHints)
    {
        if (m_hHintBuffer)
        {
            NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hHintBuffer));
            m_hHintBuffer = nullptr;
        }
    }
    if (m_enableCostBuffer)
    {
        if (m_hCostBuffer)
        {
            NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hCostBuffer));
            m_hCostBuffer = nullptr;
        }
    }
    if (m_inputStream)
    {
        m_inputStream.waitForCompletion();
    }
    if (m_outputStream)
    {
        m_outputStream.waitForCompletion();
    }
    if (m_hOF)
    {
        NVOF_API_CALL(GetAPI()->nvOFDestroy(m_hOF));
        m_hOF = nullptr;
    }
}

NvidiaOpticalFlowImpl::~NvidiaOpticalFlowImpl()
{
    collectGarbage();
}

void NvidiaOpticalFlowImpl::upSampler(InputArray _flow, cv::Size imageSize,
    int gridSize, InputOutputArray upsampledFlow)
{
    Mat flow;
    if (_flow.isMat())
    {
        Mat __flow = _flow.getMat();
        __flow.copyTo(flow);
    }
    else if (_flow.isGpuMat())
    {
        GpuMat __flow = _flow.getGpuMat();
        __flow.download(flow);
    }
    else
    {
        CV_Error(Error::StsBadArg,
            "Incorrect flow buffer passed. Pass either Mat or GpuMat");
    }

    std::unique_ptr<float[]> flowVectors = nullptr;
    const NV_OF_FLOW_VECTOR* _flowVectors = static_cast<const NV_OF_FLOW_VECTOR*>((const void*)flow.data);
    flowVectors.reset(new float[2 * imageSize.width * imageSize.height]);
    for (int y = 0; y < imageSize.height; ++y)
    {
        for (int x = 0; x < imageSize.width; ++x)
        {
            int blockIdX = x / gridSize;
            int blockIdY = y / gridSize;
            int widthInBlocks = ((imageSize.width + gridSize - 1) / gridSize);
            int heightInBlocks = ((imageSize.height + gridSize - 1) / gridSize);;
            if ((blockIdX < widthInBlocks) && (blockIdY < heightInBlocks))
            {
                flowVectors[(y * 2 * imageSize.width) + 2 * x] = (float)
                    (_flowVectors[blockIdX + (blockIdY * widthInBlocks)].flowx / (float)(1 << 5));
                flowVectors[(y * 2 * imageSize.width) + 2 * x + 1] = (float)
                    (_flowVectors[blockIdX + (blockIdY * widthInBlocks)].flowy / (float)(1 << 5));
            }
        }
    }

    Mat output(Size(imageSize.width, imageSize.height), CV_32FC2, flowVectors.get());
    if (upsampledFlow.isMat())
    {
        output.copyTo(upsampledFlow);
    }
    else if (upsampledFlow.isGpuMat())
    {
        GpuMat _output(output);
        _output.copyTo(upsampledFlow);
    }
    else
    {
        CV_Error(Error::StsBadArg,
            "Incorrect flow buffer passed for upsampled flow. Pass either Mat or GpuMat");
    }
}

class NvidiaOpticalFlowImpl_2 : public cv::cuda::NvidiaOpticalFlow_2_0
{
private:
    int m_width;
    int m_height;
    NV_OF_PERF_LEVEL m_preset;
    NV_OF_OUTPUT_VECTOR_GRID_SIZE m_gridSize;
    NV_OF_HINT_VECTOR_GRID_SIZE m_hintGridSize;
    bool m_enableROI;
    std::vector<Rect> m_roiDataRect;
    NV_OF_ROI_RECT* m_roiData;
    bool m_enableTemporalHints;
    bool m_enableExternalHints;
    bool m_enableCostBuffer;
    int m_gpuId;
    Stream m_inputStream;
    Stream m_outputStream;

    CUcontext m_cuContext;
    int m_scaleFactor;
    NV_OF_BUFFER_FORMAT m_format;
    NV_OF_OUTPUT_VECTOR_GRID_SIZE m_hwGridSize;

    NV_OF_BUFFER_DESCRIPTOR m_inputBufferDesc;
    NV_OF_BUFFER_DESCRIPTOR m_outputBufferDesc;
    NV_OF_BUFFER_DESCRIPTOR m_hintBufferDesc;
    NV_OF_BUFFER_DESCRIPTOR m_costBufferDesc;

    int m_outputElementSize;
    int m_costBufElementSize;
    int m_hintBufElementSize;

    NV_OF_INIT_PARAMS m_initParams;

    std::unique_ptr<NV_OF_CUDA_API_FUNCTION_LIST> m_ofAPI;
    NvOFHandle m_hOF; //nvof handle

    NvOFGPUBufferHandle m_hInputBuffer;
    NvOFGPUBufferHandle m_hReferenceBuffer;
    NvOFGPUBufferHandle m_hOutputBuffer;
    NvOFGPUBufferHandle m_hOutputUpScaledBuffer;
    NvOFGPUBufferHandle m_hHintBuffer;
    NvOFGPUBufferHandle m_hCostBuffer;

    CUdeviceptr m_frame0cuDevPtr;
    CUdeviceptr m_frame1cuDevPtr;
    CUdeviceptr m_flowXYcuDevPtr;
    CUdeviceptr m_flowXYUpScaledcuDevPtr;
    CUdeviceptr m_hintcuDevPtr;
    CUdeviceptr m_costcuDevPtr;

    NV_OF_CUDA_BUFFER_STRIDE_INFO m_inputBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_referenceBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_outputBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_outputUpScaledBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_hintBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_costBufferStrideInfo;

    NV_OF_CUDA_API_FUNCTION_LIST* GetAPI()
    {
        return  m_ofAPI.get();
    }

    NvOFHandle GetHandle() { return m_hOF; }

public:
    NvidiaOpticalFlowImpl_2(cv::Size imageSize, NV_OF_PERF_LEVEL perfPreset,
        NV_OF_OUTPUT_VECTOR_GRID_SIZE outputGridSize, NV_OF_HINT_VECTOR_GRID_SIZE hintGridSize,
        bool bEnableROI, std::vector<Rect> roiData, bool bEnableTemporalHints,
        bool bEnableExternalHints, bool bEnableCostBuffer, int gpuId, Stream inputStream, Stream outputStream);

    virtual void calc(InputArray inputImage, InputArray referenceImage,
        InputOutputArray flow, Stream& stream = Stream::Null(),
        InputArray hint = cv::noArray(), OutputArray cost = cv::noArray());

    virtual void collectGarbage();

    virtual void convertToFloat(InputArray flow, InputOutputArray floatFlow);

    virtual int getGridSize() const { return m_gridSize; }

    ~NvidiaOpticalFlowImpl_2();
};

NvidiaOpticalFlowImpl_2::NvidiaOpticalFlowImpl_2(
    cv::Size imageSize, NV_OF_PERF_LEVEL perfPreset,
    NV_OF_OUTPUT_VECTOR_GRID_SIZE outputGridSize, NV_OF_HINT_VECTOR_GRID_SIZE hintGridSize,
    bool bEnableROI, std::vector<Rect> roiData, bool bEnableTemporalHints,
    bool bEnableExternalHints, bool bEnableCostBuffer, int gpuId, Stream inputStream, Stream outputStream) :
    m_width(imageSize.width), m_height(imageSize.height), m_preset(perfPreset),
    m_gridSize(outputGridSize), m_hintGridSize(hintGridSize),
    m_enableROI(bEnableROI), m_roiDataRect(roiData),
    m_enableTemporalHints((NV_OF_BOOL)bEnableTemporalHints),
    m_enableExternalHints((NV_OF_BOOL)bEnableExternalHints),
    m_enableCostBuffer((NV_OF_BOOL)bEnableCostBuffer), m_gpuId(gpuId),
    m_inputStream(inputStream), m_outputStream(outputStream),
    m_cuContext(nullptr), m_scaleFactor(1), m_format(NV_OF_BUFFER_FORMAT_GRAYSCALE8),
    m_hwGridSize((NV_OF_OUTPUT_VECTOR_GRID_SIZE)0)
{
    LoadNvidiaModules& LoadNvidiaModulesObj = LoadNvidiaModules::Init();

    int nGpu = 0;

    cuSafeCall(cudaGetDeviceCount(&nGpu));
    if (m_gpuId < 0 || m_gpuId >= nGpu)
    {
        CV_Error(Error::StsBadArg, "Invalid GPU Ordinal");
    }

    cuSafeCall(cudaSetDevice(m_gpuId));
    cuSafeCall(cudaFree(m_cuContext));

    cuSafeCall(LoadNvidiaModulesObj.GetCudaLibraryFunctionPtr()(&m_cuContext));

    if (m_gridSize != (NV_OF_OUTPUT_VECTOR_GRID_SIZE)NV_OF_OUTPUT_VECTOR_GRID_SIZE_1 &&
        m_gridSize != (NV_OF_OUTPUT_VECTOR_GRID_SIZE)NV_OF_OUTPUT_VECTOR_GRID_SIZE_2 &&
        m_gridSize != (NV_OF_OUTPUT_VECTOR_GRID_SIZE)NV_OF_OUTPUT_VECTOR_GRID_SIZE_4)
    {
        CV_Error(Error::StsBadArg, "Unsupported output grid size");
    }

    if (m_enableExternalHints)
    {
        if (m_hintGridSize != (NV_OF_HINT_VECTOR_GRID_SIZE)NV_OF_HINT_VECTOR_GRID_SIZE_1 &&
            m_hintGridSize != (NV_OF_HINT_VECTOR_GRID_SIZE)NV_OF_HINT_VECTOR_GRID_SIZE_2 &&
            m_hintGridSize != (NV_OF_HINT_VECTOR_GRID_SIZE)NV_OF_HINT_VECTOR_GRID_SIZE_4 &&
            m_hintGridSize != (NV_OF_HINT_VECTOR_GRID_SIZE)NV_OF_HINT_VECTOR_GRID_SIZE_8)
        {
            CV_Error(Error::StsBadArg, "Unsupported hint grid size");
        }
    }

    m_ofAPI.reset(new NV_OF_CUDA_API_FUNCTION_LIST());

    NVOF_API_CALL(LoadNvidiaModulesObj.GetOFLibraryFunctionPtr()(NV_OF_API_VERSION, m_ofAPI.get()));
    NVOF_API_CALL(GetAPI()->nvCreateOpticalFlowCuda(m_cuContext, &m_hOF));

    m_roiData = (NV_OF_ROI_RECT*)m_roiDataRect.data();

    uint32_t size = 0;
    if (m_enableROI)
    {
        NVOF_API_CALL(GetAPI()->nvOFGetCaps(GetHandle(), NV_OF_CAPS_SUPPORT_ROI, nullptr, &size));
        std::unique_ptr<uint32_t[]> val1(new uint32_t[size]);
        NVOF_API_CALL(GetAPI()->nvOFGetCaps(GetHandle(), NV_OF_CAPS_SUPPORT_ROI, val1.get(), &size));
        if (val1[0] != NV_OF_TRUE)
        {
            m_enableROI = false;
            m_roiData = nullptr;
            CV_Error(Error::StsBadFunc, "ROI not supported on this GPU");
        }
    }
    size = 0;
    NVOF_API_CALL(GetAPI()->nvOFGetCaps(GetHandle(), NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, nullptr, &size));
    std::unique_ptr<uint32_t[]> val2(new uint32_t[size]);
    NVOF_API_CALL(GetAPI()->nvOFGetCaps(GetHandle(), NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, val2.get(), &size));
    for (uint32_t i = 0; i < size; i++)
    {
        if (m_gridSize != val2[i])
        {
            size = 0;
            NVOF_API_CALL(GetAPI()->nvOFGetCaps(GetHandle(), NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, nullptr, &size));
            std::unique_ptr<uint32_t[]> val3(new uint32_t[size]);
            NVOF_API_CALL(GetAPI()->nvOFGetCaps(GetHandle(), NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, val3.get(), &size));

            m_hwGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)NV_OF_OUTPUT_VECTOR_GRID_SIZE_MAX;
            for (uint32_t i = 0; i < size; i++)
            {
                if (m_gridSize == val3[i])
                {
                    m_hwGridSize = m_gridSize;
                    break;
                }
                if (m_gridSize < val3[i] && val3[i] < m_hwGridSize)
                {
                    m_hwGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)val3[i];
                }
            }
            if (m_hwGridSize >= (NV_OF_OUTPUT_VECTOR_GRID_SIZE)NV_OF_OUTPUT_VECTOR_GRID_SIZE_MAX)
            {
                CV_Error(Error::StsBadArg, "Invalid Grid Size");
            }
            else
            {
                m_scaleFactor = m_hwGridSize / m_gridSize;
            }
        }
        else
        {
            m_hwGridSize = m_gridSize;
        }
    }

    auto nOutWidth = (m_width + m_hwGridSize - 1) / m_hwGridSize;
    auto nOutHeight = (m_height + m_hwGridSize - 1) / m_hwGridSize;

    auto outBufFmt = NV_OF_BUFFER_FORMAT_SHORT2;

    memset(&m_inputBufferDesc, 0, sizeof(m_inputBufferDesc));
    m_inputBufferDesc.width = m_width;
    m_inputBufferDesc.height = m_height;
    m_inputBufferDesc.bufferFormat = m_format;
    m_inputBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;

    memset(&m_outputBufferDesc, 0, sizeof(m_outputBufferDesc));
    m_outputBufferDesc.width = nOutWidth;
    m_outputBufferDesc.height = nOutHeight;
    m_outputBufferDesc.bufferFormat = outBufFmt;
    m_outputBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
    m_outputElementSize = sizeof(NV_OF_FLOW_VECTOR);

    if (m_enableExternalHints)
    {
        memset(&m_hintBufferDesc, 0, sizeof(m_hintBufferDesc));
        m_hintBufferDesc.width = nOutWidth;
        m_hintBufferDesc.height = nOutHeight;
        m_hintBufferDesc.bufferFormat = outBufFmt;
        m_hintBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_HINT;
        m_hintBufElementSize = m_outputElementSize;
    }

    if (m_enableCostBuffer)
    {
        memset(&m_costBufferDesc, 0, sizeof(m_costBufferDesc));
        m_costBufferDesc.width = nOutWidth;
        m_costBufferDesc.height = nOutHeight;
        m_costBufferDesc.bufferFormat = NV_OF_BUFFER_FORMAT_UINT8;
        m_costBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_COST;
        m_costBufElementSize = sizeof(int);
    }

    memset(&m_initParams, 0, sizeof(m_initParams));
    m_initParams.width = m_inputBufferDesc.width;
    m_initParams.height = m_inputBufferDesc.height;
    m_initParams.enableExternalHints = (NV_OF_BOOL)m_enableExternalHints;
    m_initParams.enableOutputCost = (NV_OF_BOOL)m_enableCostBuffer;
    m_initParams.hintGridSize = (NV_OF_BOOL)m_enableExternalHints == NV_OF_TRUE ?
        m_hintGridSize : (NV_OF_HINT_VECTOR_GRID_SIZE)NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    m_initParams.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)m_hwGridSize;
    m_initParams.mode = NV_OF_MODE_OPTICALFLOW;
    m_initParams.perfLevel = m_preset;
    m_initParams.enableRoi = (NV_OF_BOOL)m_enableROI;

    NVOF_API_CALL(GetAPI()->nvOFInit(GetHandle(), &m_initParams));

    if (m_inputStream || m_outputStream)
    {
        NVOF_API_CALL(GetAPI()->nvOFSetIOCudaStreams(GetHandle(),
            StreamAccessor::getStream(m_inputStream), StreamAccessor::getStream(m_outputStream)));
    }

    //Input Buffer 1
    NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(GetHandle(),
        &m_inputBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hInputBuffer));
    m_frame0cuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hInputBuffer);
    NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
        m_hInputBuffer, &m_inputBufferStrideInfo));

    //Input Buffer 2
    NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(GetHandle(),
        &m_inputBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hReferenceBuffer));
    m_frame1cuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hReferenceBuffer);
    NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
        m_hReferenceBuffer, &m_referenceBufferStrideInfo));

    //Output Buffer
    NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(GetHandle(),
        &m_outputBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hOutputBuffer));
    m_flowXYcuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hOutputBuffer);
    NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
        m_hOutputBuffer, &m_outputBufferStrideInfo));

    if (m_scaleFactor > 1)
    {
        m_outputBufferDesc.width = (m_width + m_gridSize - 1) / m_gridSize;;
        m_outputBufferDesc.height = (m_height + m_gridSize - 1) / m_gridSize;;

        //Output UpScaled Buffer
        NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(GetHandle(),
            &m_outputBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hOutputUpScaledBuffer));
        m_flowXYUpScaledcuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hOutputUpScaledBuffer);
        NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
            m_hOutputUpScaledBuffer, &m_outputUpScaledBufferStrideInfo));
    }

    //Hint Buffer
    if (m_enableExternalHints)
    {
        NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(GetHandle(),
            &m_hintBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hHintBuffer));
        m_hintcuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hHintBuffer);
        NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
            m_hHintBuffer, &m_hintBufferStrideInfo));
    }

    //Cost Buffer
    if (m_enableCostBuffer)
    {
        NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(GetHandle(),
            &m_costBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hCostBuffer));
        m_costcuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hCostBuffer);
        NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
            m_hCostBuffer, &m_costBufferStrideInfo));
    }
}

void NvidiaOpticalFlowImpl_2::calc(InputArray _frame0, InputArray _frame1, InputOutputArray _flow,
    Stream& stream, InputArray hint, OutputArray cost)
{
    CV_UNUSED(stream);
    GpuMat frame0GpuMat(_frame0.size(), _frame0.type(), (void*)m_frame0cuDevPtr,
        m_inputBufferStrideInfo.strideInfo[0].strideXInBytes);
    GpuMat frame1GpuMat(_frame1.size(), _frame1.type(), (void*)m_frame1cuDevPtr,
        m_referenceBufferStrideInfo.strideInfo[0].strideXInBytes);
    GpuMat flowXYGpuMat(Size((m_width + m_hwGridSize - 1) / m_hwGridSize,
        (m_height + m_hwGridSize - 1) / m_hwGridSize), CV_16SC2,
        (void*)m_flowXYcuDevPtr, m_outputBufferStrideInfo.strideInfo[0].strideXInBytes);
    GpuMat flowXYGpuMatUpScaled(Size((m_width + m_gridSize - 1) / m_gridSize,
        (m_height + m_gridSize - 1) / m_gridSize), CV_16SC2,
        (void*)m_flowXYUpScaledcuDevPtr, m_outputUpScaledBufferStrideInfo.strideInfo[0].strideXInBytes);

    //check whether frame0 is Mat or GpuMat
    if (_frame0.isMat())
    {
        //Get Mats from InputArrays
        Mat __frame0 = _frame0.getMat();
        frame0GpuMat.upload(__frame0, m_inputStream);
    }
    else if (_frame0.isGpuMat())
    {
        //Get GpuMats from InputArrays
        GpuMat __frame0 = _frame0.getGpuMat();
        __frame0.copyTo(frame0GpuMat, m_inputStream);
    }
    else
    {
        CV_Error(Error::StsBadArg,
            "Incorrect input. Pass input image (frame0) as Mat or GpuMat");
    }

    //check whether frame1 is Mat or GpuMat
    if (_frame1.isMat())
    {
        //Get Mats from InputArrays
        Mat __frame1 = _frame1.getMat();
        frame1GpuMat.upload(__frame1, m_inputStream);
    }
    else if (_frame1.isGpuMat())
    {
        //Get GpuMats from InputArrays
        GpuMat __frame1 = _frame1.getGpuMat();
        __frame1.copyTo(frame1GpuMat, m_inputStream);
    }
    else
    {
        CV_Error(Error::StsBadArg,
            "Incorrect input. Pass reference image (frame1) as Mat or GpuMat");
    }

    if (m_enableExternalHints)
    {
        GpuMat hintGpuMat(hint.size(), hint.type(), (void*)m_hintcuDevPtr,
            m_hintBufferStrideInfo.strideInfo[0].strideXInBytes);

        if (hint.isMat())
        {
            //Get Mat from InputArray hint
            Mat _hint = hint.getMat();
            hintGpuMat.upload(_hint, m_inputStream);
        }
        else if (hint.isGpuMat())
        {
            //Get GpuMat from InputArray hint
            GpuMat _hint = hint.getGpuMat();
            _hint.copyTo(hintGpuMat, m_inputStream);
        }
        else
        {
            CV_Error(Error::StsBadArg, "Incorrect hint buffer passed. Pass Mat or GpuMat");
        }
    }

    //Execute Call
    NV_OF_EXECUTE_INPUT_PARAMS exeInParams;
    NV_OF_EXECUTE_OUTPUT_PARAMS exeOutParams;
    memset(&exeInParams, 0, sizeof(exeInParams));
    exeInParams.inputFrame = m_hInputBuffer;
    exeInParams.referenceFrame = m_hReferenceBuffer;
    exeInParams.disableTemporalHints = (NV_OF_BOOL)m_enableTemporalHints == NV_OF_TRUE ?
        NV_OF_FALSE : NV_OF_TRUE;
    exeInParams.externalHints = m_initParams.enableExternalHints == NV_OF_TRUE ?
        m_hHintBuffer : nullptr;
    exeInParams.numRois = m_initParams.enableRoi == NV_OF_TRUE ? m_roiDataRect.size() : 0;
    exeInParams.roiData = m_initParams.enableRoi == NV_OF_TRUE ? m_roiData : nullptr;
    memset(&exeOutParams, 0, sizeof(exeOutParams));
    exeOutParams.outputBuffer = m_hOutputBuffer;
    exeOutParams.outputCostBuffer = m_initParams.enableOutputCost == NV_OF_TRUE ?
        m_hCostBuffer : nullptr;
    NVOF_API_CALL(GetAPI()->nvOFExecute(GetHandle(), &exeInParams, &exeOutParams));

    if (m_scaleFactor > 1)
    {
        uint32_t nSrcWidth = flowXYGpuMat.size().width;
        uint32_t nSrcHeight = flowXYGpuMat.size().height;
        uint32_t nSrcPitch = m_outputBufferStrideInfo.strideInfo[0].strideXInBytes;
        uint32_t nDstWidth = flowXYGpuMatUpScaled.size().width;
        uint32_t nDstHeight = flowXYGpuMatUpScaled.size().height;
        uint32_t nDstPitch = m_outputUpScaledBufferStrideInfo.strideInfo[0].strideXInBytes;
        cv::cuda::device::optflow_nvidia::FlowUpsample((void*)m_flowXYcuDevPtr, nSrcWidth, nSrcPitch,
            nSrcHeight, (void*)m_flowXYUpScaledcuDevPtr, nDstWidth, nDstPitch, nDstHeight, m_scaleFactor);

        if (_flow.isMat())
            flowXYGpuMatUpScaled.download(_flow, m_outputStream);
        else if (_flow.isGpuMat())
            flowXYGpuMatUpScaled.copyTo(_flow, m_outputStream);
        else
            CV_Error(Error::StsBadArg, "Incorrect flow buffer passed. Pass Mat or GpuMat");
    }
    else
    {
        if (_flow.isMat())
            flowXYGpuMat.download(_flow, m_outputStream);
        else if (_flow.isGpuMat())
            flowXYGpuMat.copyTo(_flow, m_outputStream);
        else
            CV_Error(Error::StsBadArg, "Incorrect flow buffer passed. Pass Mat or GpuMat");
    }

    if (m_enableCostBuffer)
    {
        GpuMat costGpuMat(Size((m_width + m_hwGridSize - 1) / m_hwGridSize,
            (m_height + m_hwGridSize - 1) / m_hwGridSize), CV_8SC1, (void*)m_costcuDevPtr,
            m_costBufferStrideInfo.strideInfo[0].strideXInBytes);

        if (cost.isMat())
            costGpuMat.download(cost, m_outputStream);
        else if (cost.isGpuMat())
            costGpuMat.copyTo(cost, m_outputStream);
        else
            CV_Error(Error::StsBadArg, "Incorrect cost buffer passed. Pass Mat or GpuMat");
    }
    m_outputStream.waitForCompletion();
}

void NvidiaOpticalFlowImpl_2::collectGarbage()
{
    if (m_enableROI)
    {
        m_roiData = nullptr;
    }
    if (m_hInputBuffer)
    {
        NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hInputBuffer));
        m_hInputBuffer = nullptr;
    }
    if (m_hReferenceBuffer)
    {
        NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hReferenceBuffer));
        m_hReferenceBuffer = nullptr;
    }
    if (m_hOutputBuffer)
    {
        NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hOutputBuffer));
        m_hOutputBuffer = nullptr;
    }
    if (m_scaleFactor > 1 && m_hOutputUpScaledBuffer)
    {
        NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hOutputUpScaledBuffer));
        m_hOutputUpScaledBuffer = nullptr;
    }
    if (m_enableExternalHints)
    {
        if (m_hHintBuffer)
        {
            NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hHintBuffer));
            m_hHintBuffer = nullptr;
        }
    }
    if (m_enableCostBuffer)
    {
        if (m_hCostBuffer)
        {
            NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hCostBuffer));
            m_hCostBuffer = nullptr;
        }
    }
    if (m_inputStream)
    {
        m_inputStream.waitForCompletion();
    }
    if (m_outputStream)
    {
        m_outputStream.waitForCompletion();
    }
    if (m_hOF)
    {
        NVOF_API_CALL(GetAPI()->nvOFDestroy(m_hOF));
        m_hOF = nullptr;
    }
}

NvidiaOpticalFlowImpl_2::~NvidiaOpticalFlowImpl_2()
{
    collectGarbage();
}

void NvidiaOpticalFlowImpl_2::convertToFloat(InputArray _flow, InputOutputArray floatFlow)
{
    Mat flow;
    if (_flow.isMat())
    {
        Mat __flow = _flow.getMat();
        __flow.copyTo(flow);
    }
    else if (_flow.isGpuMat())
    {
        GpuMat __flow = _flow.getGpuMat();
        __flow.download(flow);
    }
    else
    {
        CV_Error(Error::StsBadArg,
            "Incorrect flow buffer passed. Pass either Mat or GpuMat");
    }

    int width = flow.size().width;
    int height = flow.size().height;

    Mat output(Size(width, height), CV_32FC2);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < (int)(width * sizeof(int16_t)); ++x)
        {
            output.at<float>(y, x) = (float)(flow.at<int16_t>(y, x) / (float)(1 << 5));
        }
    }

    if (floatFlow.isMat())
    {
        output.copyTo(floatFlow);
    }
    else if (floatFlow.isGpuMat())
    {
        GpuMat _output(output);
        _output.copyTo(floatFlow);
    }
    else
    {
        CV_Error(Error::StsBadArg,
            "Incorrect flow buffer passed for upsampled flow. Pass either Mat or GpuMat");
    }
}}

Ptr<cv::cuda::NvidiaOpticalFlow_1_0> cv::cuda::NvidiaOpticalFlow_1_0::create(
    cv::Size imageSize, NVIDIA_OF_PERF_LEVEL perfPreset,
    bool bEnableTemporalHints, bool bEnableExternalHints,
    bool bEnableCostBuffer, int gpuId,
    Stream& inputStream, Stream& outputStream)
{
    return makePtr<NvidiaOpticalFlowImpl>(
        imageSize,
        (NV_OF_PERF_LEVEL)perfPreset,
        bEnableTemporalHints,
        bEnableExternalHints,
        bEnableCostBuffer,
        gpuId,
        inputStream,
        outputStream);
}

Ptr<cv::cuda::NvidiaOpticalFlow_2_0> cv::cuda::NvidiaOpticalFlow_2_0::create(
    cv::Size imageSize, NVIDIA_OF_PERF_LEVEL perfPreset,
    NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE outputGridSize, NVIDIA_OF_HINT_VECTOR_GRID_SIZE hintGridSize,
    bool bEnableTemporalHints, bool bEnableExternalHints, bool bEnableCostBuffer,
    int gpuId, Stream& inputStream, Stream& outputStream)
{
    std::vector<Rect> roi(0);
    return makePtr<NvidiaOpticalFlowImpl_2>(
        imageSize,
        (NV_OF_PERF_LEVEL)perfPreset,
        (NV_OF_OUTPUT_VECTOR_GRID_SIZE)outputGridSize,
        (NV_OF_HINT_VECTOR_GRID_SIZE)hintGridSize,
        false,
        roi,
        bEnableTemporalHints,
        bEnableExternalHints,
        bEnableCostBuffer,
        gpuId,
        inputStream,
        outputStream);
}

Ptr<cv::cuda::NvidiaOpticalFlow_2_0> cv::cuda::NvidiaOpticalFlow_2_0::create(
  cv::Size imageSize, std::vector<Rect> roiData, NVIDIA_OF_PERF_LEVEL perfPreset,
  NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE outputGridSize, NVIDIA_OF_HINT_VECTOR_GRID_SIZE hintGridSize,
  bool bEnableTemporalHints, bool bEnableExternalHints, bool bEnableCostBuffer,
  int gpuId, Stream& inputStream, Stream& outputStream)
{
    return makePtr<NvidiaOpticalFlowImpl_2>(
        imageSize,
        (NV_OF_PERF_LEVEL)perfPreset,
        (NV_OF_OUTPUT_VECTOR_GRID_SIZE)outputGridSize,
        (NV_OF_HINT_VECTOR_GRID_SIZE)hintGridSize,
        true,
        roiData,
        bEnableTemporalHints,
        bEnableExternalHints,
        bEnableCostBuffer,
        gpuId,
        inputStream,
        outputStream);
}
#endif
