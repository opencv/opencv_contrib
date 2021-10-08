/*M///////////////////////////////////////////////////////////////////////////////////////
 * //
 * //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 * //
 * //  By downloading, copying, installing or using the software you agree to this license.
 * //  If you do not agree to this license, do not download, install,
 * //  copy or use the software.
 * //
 * //
 * //                           License Agreement
 * //                For Open Source Computer Vision Library
 * //
 * // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 * // Third party copyrights are property of their respective owners.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * //   * Redistribution's of source code must retain the above copyright notice,
 * //     this list of conditions and the following disclaimer.
 * //
 * //   * Redistribution's in binary form must reproduce the above copyright notice,
 * //     this list of conditions and the following disclaimer in the documentation
 * //     and/or other materials provided with the distribution.
 * //
 * //   * The name of the copyright holders may not be used to endorse or promote products
 * //     derived from this software without specific prior written permission.
 * //
 * // This software is provided by the copyright holders and contributors "as is" and
 * // any express or implied warranties, including, but not limited to, the implied
 * // warranties of merchantability and fitness for a particular purpose are disclaimed.
 * // In no event shall the Intel Corporation or contributors be liable for any direct,
 * // indirect, incidental, special, exemplary, or consequential damages
 * // (including, but not limited to, procurement of substitute goods or services;
 * // loss of use, data, or profits; or business interruption) however caused
 * // and on any theory of liability, whether in contract, strict liability,
 * // or tort (including negligence or otherwise) arising in any way out of
 * // the use of this software, even if advised of the possibility of such damage.
 * //
 * //M*/

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || !defined(HAVE_NVIDIA_OPTFLOW) || defined (CUDA_DISABLER)

Ptr<cuda::NvidiaHWStereoBM> cv::cuda::createNvidiaHWStereoBM(int, int) { throw_no_cuda(); return Ptr<cuda::createNvidiaHWStereoBM>(); }

#else /* !defined (HAVE_CUDA) */

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

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
#elif defined(_WIN32)
#define OF_MODULENAME TEXT("nvofapi.dll")
#else
#define OF_MODULENAME "libnvidia-opticalflow.so.1"
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


namespace
{

class LoadNvidiaModules
{
private:
    typedef int(*PFNCudaCuCtxGetCurrent)(CUcontext*);
    typedef NV_OF_STATUS(NVOFAPI *PFNNvOFAPICreateInstanceCuda)
        (int apiVer, NV_OF_CUDA_API_FUNCTION_LIST* cudaOf);

    PFNNvOFAPICreateInstanceCuda m_NvOFAPICreateInstanceCuda;
    HMODULE m_hOFModule;
    HMODULE m_hCudaModule;
    bool m_isFailed;

    LoadNvidiaModules() :
        m_NvOFAPICreateInstanceCuda(NULL),
        m_isFailed(false)
    {

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

    PFNNvOFAPICreateInstanceCuda GetOFLibraryFunctionPtr() { return m_NvOFAPICreateInstanceCuda; }
};

class NvidiaHWStereoBMImlp : public cuda::NvidiaHWStereoBM
{
public:
    NvidiaHWStereoBMImlp(NV_OF_PERF_LEVEL perfPreset, NV_OF_OUTPUT_VECTOR_GRID_SIZE grid_size):
        m_ready(false),
        m_width(0),
        m_height(0),
        m_channels(0),
        m_preset(perfPreset),
        m_gridSize(grid_size)
    {
        LoadNvidiaModules& LoadNvidiaModulesObj = LoadNvidiaModules::Init();

        // to ensure that CUDA Runtime API is properly initialized
        cudaFree(0);
        cuSafeCall(cuCtxGetCurrent(&m_cuContext));

        if (m_gridSize != (NV_OF_OUTPUT_VECTOR_GRID_SIZE)NV_OF_OUTPUT_VECTOR_GRID_SIZE_1 &&
            m_gridSize != (NV_OF_OUTPUT_VECTOR_GRID_SIZE)NV_OF_OUTPUT_VECTOR_GRID_SIZE_2 &&
            m_gridSize != (NV_OF_OUTPUT_VECTOR_GRID_SIZE)NV_OF_OUTPUT_VECTOR_GRID_SIZE_4)
        {
            CV_Error(Error::StsBadArg, "Unsupported output grid size");
        }

        m_ofAPI.reset(new NV_OF_CUDA_API_FUNCTION_LIST());

        NVOF_API_CALL(LoadNvidiaModulesObj.GetOFLibraryFunctionPtr()(NV_OF_API_VERSION, m_ofAPI.get()));
        NVOF_API_CALL(GetAPI()->nvCreateOpticalFlowCuda(m_cuContext, &m_hOF));

        uint32_t size = 0;
        NVOF_API_CALL(GetAPI()->nvOFGetCaps(m_hOF, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, nullptr, &size));
        std::unique_ptr<uint32_t[]> val2(new uint32_t[size]);
        NVOF_API_CALL(GetAPI()->nvOFGetCaps(m_hOF, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, val2.get(), &size));
        for (uint32_t i = 0; i < size; i++)
        {
            if (m_gridSize != val2[i])
            {
                size = 0;
                NVOF_API_CALL(GetAPI()->nvOFGetCaps(m_hOF, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, nullptr, &size));
                std::unique_ptr<uint32_t[]> val3(new uint32_t[size]);
                NVOF_API_CALL(GetAPI()->nvOFGetCaps(m_hOF, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, val3.get(), &size));

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
    }

    bool init(int width, int height, int channels)
    {
        m_width = width;
        m_height = height;
        m_channels = channels;

        auto nOutWidth = (m_width + m_hwGridSize - 1) / m_hwGridSize;
        auto nOutHeight = (m_height + m_hwGridSize - 1) / m_hwGridSize;

        memset(&m_inputBufferDesc, 0, sizeof(m_inputBufferDesc));
        m_inputBufferDesc.width = m_width;
        m_inputBufferDesc.height = m_height;
        m_inputBufferDesc.bufferFormat = (m_channels == 1) ? NV_OF_BUFFER_FORMAT_GRAYSCALE8 : NV_OF_BUFFER_FORMAT_ABGR8;
        m_inputBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;

        memset(&m_outputBufferDesc, 0, sizeof(m_outputBufferDesc));
        m_outputBufferDesc.width = nOutWidth;
        m_outputBufferDesc.height = nOutHeight;
        m_outputBufferDesc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;
        m_outputBufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;

        memset(&m_initParams, 0, sizeof(m_initParams));
        m_initParams.width = m_inputBufferDesc.width;
        m_initParams.height = m_inputBufferDesc.height;
        m_initParams.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)m_hwGridSize;
        // Use mode Optical Flow as Mode Stereo matching is deprecated
        m_initParams.mode = NV_OF_MODE_OPTICALFLOW;
        m_initParams.perfLevel = m_preset;

        NVOF_API_CALL(GetAPI()->nvOFInit(m_hOF, &m_initParams));

        //Input Buffer 1
        NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(m_hOF,
                                                        &m_inputBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hRightInput));
        m_frame0cuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hRightInput);
        NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
            m_hRightInput, &m_RightBufferStrideInfo));

        //Input Buffer 2
        NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(m_hOF,
                                                        &m_inputBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hLeftInput));
        m_frame1cuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hLeftInput);
        NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
            m_hLeftInput, &m_LeftBufferStrideInfo));

        //Output Buffer
        NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(m_hOF,
                                                        &m_outputBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hOutputBuffer));
        m_flowXYcuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hOutputBuffer);
        NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
            m_hOutputBuffer, &m_outputBufferStrideInfo));

        if (m_scaleFactor > 1)
        {
            m_outputBufferDesc.width = (m_width + m_gridSize - 1) / m_gridSize;
            m_outputBufferDesc.height = (m_height + m_gridSize - 1) / m_gridSize;

            //Output UpScaled Buffer
            NVOF_API_CALL(GetAPI()->nvOFCreateGPUBufferCuda(m_hOF,
                                                            &m_outputBufferDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_hOutputUpScaledBuffer));
            m_flowXYUpScaledcuDevPtr = GetAPI()->nvOFGPUBufferGetCUdeviceptr(m_hOutputUpScaledBuffer);
            NVOF_API_CALL(GetAPI()->nvOFGPUBufferGetStrideInfo(
                m_hOutputUpScaledBuffer, &m_outputUpScaledBufferStrideInfo));
        }

        m_convertBuffersL.resize(4);
        m_convertBuffersL[0] = cv::cuda::GpuMat(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
        m_convertBuffersL[1] = cv::cuda::GpuMat(height, width, CV_8UC1);
        m_convertBuffersL[2] = cv::cuda::GpuMat(height, width, CV_8UC1);
        m_convertBuffersL[3] = cv::cuda::GpuMat(height, width, CV_8UC1);

        m_convertBuffersR.resize(4);
        m_convertBuffersR[0] = cv::cuda::GpuMat(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
        m_convertBuffersR[1] = cv::cuda::GpuMat(height, width, CV_8UC1);
        m_convertBuffersR[2] = cv::cuda::GpuMat(height, width, CV_8UC1);
        m_convertBuffersR[3] = cv::cuda::GpuMat(height, width, CV_8UC1);

        return true;
    }

    void compute(InputArray left, InputArray right, OutputArray disparity, Stream inputStream, Stream outputStream)
    {
        CV_Assert(left.channels() == 1 || left.channels() == 3 || left.channels() == 4);
        CV_Assert((left.size() == right.size()) && (left.channels() == right.channels()));

        if(!m_ready)
        {
            m_ready = init(left.cols(), left.rows(), left.channels());
        }
        else
        {
            if((left.cols() != m_width) || (left.rows() != m_height) || (left.channels() != m_channels))
            {
                fini();
                m_ready = init(left.cols(), left.rows(), left.channels());
            }
        }

        if (inputStream || outputStream)
        {
            NVOF_API_CALL(GetAPI()->nvOFSetIOCudaStreams(m_hOF,
                                                         (CUstream)StreamAccessor::getStream(inputStream), (CUstream)StreamAccessor::getStream(outputStream)));
        }

        int input_type = (left.channels() == 1) ? CV_8UC1 : CV_8UC4;
        GpuMat frame0GpuMat(left.size(), input_type, (void*)m_frame0cuDevPtr,
                            m_RightBufferStrideInfo.strideInfo[0].strideXInBytes);
        GpuMat frame1GpuMat(right.size(), input_type, (void*)m_frame1cuDevPtr,
                            m_LeftBufferStrideInfo.strideInfo[0].strideXInBytes);
        GpuMat flowXYGpuMat(Size((m_width + m_hwGridSize - 1) / m_hwGridSize,
                                 (m_height + m_hwGridSize - 1) / m_hwGridSize), CV_16SC2,
                            (void*)m_flowXYcuDevPtr, m_outputBufferStrideInfo.strideInfo[0].strideXInBytes);
        GpuMat flowXYGpuMatUpScaled(Size((m_width + m_gridSize - 1) / m_gridSize,
                                         (m_height + m_gridSize - 1) / m_gridSize), CV_16SC2,
                                    (void*)m_flowXYUpScaledcuDevPtr, m_outputUpScaledBufferStrideInfo.strideInfo[0].strideXInBytes);

        if (left.isMat())
        {
            Mat __frame0 = left.getMat();
            frame0GpuMat.upload(__frame0, inputStream);
        }
        else if (left.isGpuMat())
        {
            GpuMat __frame0 = left.getGpuMat();
            if(__frame0.channels() == 3)
            {
                cv::cuda::split(__frame0, &m_convertBuffersL[1], inputStream);
                cv::cuda::merge(m_convertBuffersL, frame0GpuMat, inputStream);
            }
            else
            {
                __frame0.copyTo(frame0GpuMat, inputStream);
            }
        }
        else
        {
            CV_Error(Error::StsBadArg,
                     "Incorrect input. Pass input image (frame0) as Mat or GpuMat");
        }

        if (right.isMat())
        {
            Mat __frame1 = right.getMat();
            frame1GpuMat.upload(__frame1, inputStream);
        }
        else if (right.isGpuMat())
        {
            GpuMat __frame1 = right.getGpuMat();
            if(__frame1.channels() == 3)
            {
                cv::cuda::split(__frame1, &m_convertBuffersR[1], inputStream);
                cv::cuda::merge(m_convertBuffersR, frame1GpuMat, inputStream);
            }
            else
            {
                __frame1.copyTo(frame1GpuMat, inputStream);
            }
        }
        else
        {
            CV_Error(Error::StsBadArg,
                     "Incorrect input. Pass reference image (frame1) as Mat or GpuMat");
        }

        //Execute Call
        NV_OF_EXECUTE_INPUT_PARAMS exeInParams;
        NV_OF_EXECUTE_OUTPUT_PARAMS exeOutParams;
        memset(&exeInParams, 0, sizeof(exeInParams));
        exeInParams.inputFrame = m_hLeftInput;
        exeInParams.referenceFrame = m_hRightInput;
        exeInParams.disableTemporalHints = NV_OF_TRUE;
        exeInParams.externalHints = nullptr;
        exeInParams.numRois =  0; // m_initParams.enableRoi == NV_OF_TRUE ? m_roiDataRect.size() : 0;
        exeInParams.roiData =  nullptr; // m_initParams.enableRoi == NV_OF_TRUE ? m_roiData : nullptr;
        memset(&exeOutParams, 0, sizeof(exeOutParams));
        exeOutParams.outputBuffer = m_hOutputBuffer;
        exeOutParams.outputCostBuffer = nullptr; // m_initParams.enableOutputCost == NV_OF_TRUE ? m_hCostBuffer : nullptr;
        NVOF_API_CALL(GetAPI()->nvOFExecute(m_hOF, &exeInParams, &exeOutParams));

        GpuMat split_xy[2];
        if (m_scaleFactor > 1)
        {
            uint32_t nSrcWidth = flowXYGpuMat.size().width;
            uint32_t nSrcHeight = flowXYGpuMat.size().height;
            uint32_t nSrcPitch = m_outputBufferStrideInfo.strideInfo[0].strideXInBytes;
            uint32_t nDstWidth = flowXYGpuMatUpScaled.size().width;
            uint32_t nDstHeight = flowXYGpuMatUpScaled.size().height;
            uint32_t nDstPitch = m_outputUpScaledBufferStrideInfo.strideInfo[0].strideXInBytes;
            cv::cuda::device::optflow_nvidia::FlowUpsample((void*)m_flowXYcuDevPtr, nSrcWidth, nSrcPitch,
                                                           nSrcHeight, (void*)m_flowXYUpScaledcuDevPtr,
                                                           nDstWidth, nDstPitch, nDstHeight,
                                                           m_scaleFactor);

            cv::cuda::split(flowXYGpuMatUpScaled, split_xy, outputStream);
        }
        else
        {
            cv::cuda::split(flowXYGpuMat, split_xy, outputStream);
        }

        if (disparity.isMat())
        {
            split_xy[0].download(disparity.getMat(), outputStream);
        }
        else if (disparity.isGpuMat())
        {
            split_xy[0].copyTo(disparity, outputStream);
        }
        else
        {
            CV_Error(Error::StsBadArg, "Incorrect flow buffer passed. Pass Mat or GpuMat");
        }
    }

    void compute(InputArray left, InputArray right, OutputArray disparity) CV_OVERRIDE
    {
        compute(left, right, disparity, Stream::Null(), Stream::Null());
    }

    void fini()
    {
        m_ready = false;
        if (m_hLeftInput)
        {
            NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hLeftInput));
        }
        if (m_hRightInput)
        {
            NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hRightInput));
        }
        if (m_hOutputBuffer)
        {
            NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hOutputBuffer));
        }
        if (m_scaleFactor > 1 && m_hOutputUpScaledBuffer)
        {
            NVOF_API_CALL(GetAPI()->nvOFDestroyGPUBufferCuda(m_hOutputUpScaledBuffer));
        }
        if (m_hOF)
        {
            NVOF_API_CALL(GetAPI()->nvOFDestroy(m_hOF));
        }
    }

    ~NvidiaHWStereoBMImlp()
    {
        fini();
    }

    int getMinDisparity() const { return -1; };
    void setMinDisparity(int minDisparity) { CV_UNUSED(minDisparity); };

    int getNumDisparities() const { return -1; };
    void setNumDisparities(int numDisparities) { CV_UNUSED(numDisparities); };

    int getBlockSize() const { return -1; };
    void setBlockSize(int blockSize) { CV_UNUSED(blockSize); };

    int getSpeckleWindowSize() const { return -1; };
    void setSpeckleWindowSize(int speckleWindowSize) { CV_UNUSED(speckleWindowSize); };

    int getSpeckleRange() const { return -1; };
    void setSpeckleRange(int speckleRange) { CV_UNUSED(speckleRange); };

    int getDisp12MaxDiff() const { return -1; };
    void setDisp12MaxDiff(int disp12MaxDiff) { CV_UNUSED(disp12MaxDiff); };

protected:

    NV_OF_CUDA_API_FUNCTION_LIST* GetAPI()
    {
        return  m_ofAPI.get();
    }

    bool m_ready;
    int m_width;
    int m_height;
    int m_channels;

    NV_OF_PERF_LEVEL m_preset;
    NV_OF_OUTPUT_VECTOR_GRID_SIZE m_gridSize;

    CUcontext m_cuContext;
    NV_OF_INIT_PARAMS m_initParams;
    std::unique_ptr<NV_OF_CUDA_API_FUNCTION_LIST> m_ofAPI;
    NvOFHandle m_hOF; // Optical Flow Engine handle
    NV_OF_BUFFER_DESCRIPTOR m_inputBufferDesc;
    NV_OF_BUFFER_DESCRIPTOR m_outputBufferDesc;
    int m_scaleFactor;
    NV_OF_BUFFER_FORMAT m_format;
    NV_OF_OUTPUT_VECTOR_GRID_SIZE m_hwGridSize;
    NvOFGPUBufferHandle m_hRightInput;
    NvOFGPUBufferHandle m_hLeftInput;
    NvOFGPUBufferHandle m_hOutputBuffer;
    NvOFGPUBufferHandle m_hOutputUpScaledBuffer;

    CUdeviceptr m_frame0cuDevPtr;
    CUdeviceptr m_frame1cuDevPtr;
    CUdeviceptr m_flowXYcuDevPtr;
    CUdeviceptr m_flowXYUpScaledcuDevPtr;

    NV_OF_CUDA_BUFFER_STRIDE_INFO m_RightBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_LeftBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_outputBufferStrideInfo;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_outputUpScaledBufferStrideInfo;

    std::vector<cv::cuda::GpuMat> m_convertBuffersL;
    std::vector<cv::cuda::GpuMat> m_convertBuffersR;
};

}

namespace cv { namespace cuda {
CV_EXPORTS_W Ptr<cuda::NvidiaHWStereoBM> createNvidiaHWStereoBM(StereoQualityPreset preset, int gridSize)
{
    return makePtr<NvidiaHWStereoBMImlp>((NV_OF_PERF_LEVEL)preset, (NV_OF_OUTPUT_VECTOR_GRID_SIZE)gridSize);
}

}} // namespaces cv::cuda

#endif // HAVE_CUDA
