// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_NVENCODER_HPP
#define OPENCV_NVENCODER_HPP
#include <vector>
#include "nvEncodeAPI.h"
#include <stdint.h>
#include <mutex>
#include <string>
#include <iostream>
#include <sstream>
#include <string.h>

namespace cv { namespace cudacodec {

#ifndef _WIN32
#include <cstring>
    static inline bool operator==(const GUID& guid1, const GUID& guid2) {
        return !memcmp(&guid1, &guid2, sizeof(GUID));
    }

    static inline bool operator!=(const GUID& guid1, const GUID& guid2) {
        return !(guid1 == guid2);
    }
#endif

#define NVENC_THROW_ERROR( errorStr, errorCode ) \
do \
{ \
cv::String msg = cv::format("%s [Code = %d]", errorStr, errorCode); \
cv::error(cv::Error::GpuApiCallError, msg, __FUNCTION__, __FILE__, __LINE__); \
} while (0)


#define NVENC_API_CALL( nvencAPI ) \
do \
{ \
NVENCSTATUS errorCode = nvencAPI; \
if( errorCode != NV_ENC_SUCCESS) \
{ \
cv::String msg = cv::format("NVENC returned error [Code = %d]", errorCode); \
cv::error(cv::Error::GpuApiCallError, msg, __FUNCTION__, __FILE__, __LINE__); \
} \
} while (0)

struct NvEncInputFrame
{
    void* inputPtr = nullptr;
    uint32_t chromaOffsets[2];
    uint32_t numChromaPlanes;
    uint32_t pitch;
    uint32_t chromaPitch;
    NV_ENC_BUFFER_FORMAT bufferFormat;
    NV_ENC_INPUT_RESOURCE_TYPE resourceType;
};

/**
* @brief Shared base class for different encoder interfaces.
*/
class NvEncoder
{
public:
    /**
    * @brief This function is used to initialize the encoder session.
    * Application must call this function to initialize the encoder, before
    * starting to encode any frames.
    */
    virtual void CreateEncoder(const NV_ENC_INITIALIZE_PARAMS* pEncodeParams);

    /**
    * @brief This function is used to destroy the encoder session.
    * Application must call this function to destroy the encoder session and
    * clean up any allocated resources. The application must call EndEncode()
    * function to get any queued encoded frames before calling DestroyEncoder().
    */
    virtual void DestroyEncoder();

    /**
    * @brief This function is used to reconfigure an existing encoder session.
    * Application can use this function to dynamically change the bitrate,
    * resolution and other QOS parameters. If the application changes the
    * resolution, it must set NV_ENC_RECONFIGURE_PARAMS::forceIDR.
    */
    bool Reconfigure(const NV_ENC_RECONFIGURE_PARAMS* pReconfigureParams);

    /**
    * @brief This function is used to get the next available input buffer.
    * Applications must call this function to obtain a pointer to the next
    * input buffer. The application must copy the uncompressed data to the
    * input buffer and then call EncodeFrame() function to encode it.
    */
    const NvEncInputFrame* GetNextInputFrame();


    /**
    * @brief This function is used to encode a frame.
    * Applications must call EncodeFrame() function to encode the uncompressed
    * data, which has been copied to an input buffer obtained from the
    * GetNextInputFrame() function.
    */
    virtual void EncodeFrame(std::vector<std::vector<uint8_t>>& vPacket, std::vector<uint64_t>& outputTimeStamps, NV_ENC_PIC_PARAMS* pPicParams = nullptr);

    /**
    * @brief This function to flush the encoder queue.
    * The encoder might be queuing frames for B picture encoding or lookahead;
    * the application must call EndEncode() to get all the queued encoded frames
    * from the encoder. The application must call this function before destroying
    * an encoder session.
    */
    virtual void EndEncode(std::vector<std::vector<uint8_t>>& vPacket, std::vector<uint64_t>& outputTimeStamps);

    /**
    * @brief This function is used to query hardware encoder capabilities.
    * Applications can call this function to query capabilities like maximum encode
    * dimensions, support for lookahead or the ME-only mode etc.
    */
    int GetCapabilityValue(GUID guidCodec, NV_ENC_CAPS capsToQuery);

    /**
    * @brief This function is used to get the current device on which encoder is running.
    */
    void* GetDevice() const { return m_pDevice; }

    /**
    * @brief This function is used to get the current device type which encoder is running.
    */
    NV_ENC_DEVICE_TYPE GetDeviceType() const { return m_eDeviceType; }

    /**
    * @brief This function is used to get the current encode width.
    * The encode width can be modified by Reconfigure() function.
    */
    int GetEncodeWidth() const { return m_nWidth; }

    /**
    * @brief This function is used to get the current encode height.
    * The encode height can be modified by Reconfigure() function.
    */
    int GetEncodeHeight() const { return m_nHeight; }

    /**
    * @brief This function is used to get the current frame size based on pixel format.
    */
    int GetFrameSize() const;

    /**
    * @brief This function is used to initialize config parameters based on
    * given codec and preset guids.
    * The application can call this function to get the default configuration
    * for a certain preset. The application can either use these parameters
    * directly or override them with application-specific settings before
    * using them in CreateEncoder() function.
    */
    void CreateDefaultEncoderParams(NV_ENC_INITIALIZE_PARAMS* pIntializeParams, GUID codecGuid, GUID presetGuid, NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_UNDEFINED);

    /**
    * @brief This function is used to get the current initialization parameters,
    * which had been used to configure the encoder session.
    * The initialization parameters are modified if the application calls
    * Reconfigure() function.
    */
    void GetInitializeParams(NV_ENC_INITIALIZE_PARAMS* pInitializeParams);

    /**
    * @brief This function is used to get sequence and picture parameter headers.
    * Application can call this function after encoder is initialized to get SPS and PPS
    * nalus for the current encoder instance. The sequence header data might change when
    * application calls Reconfigure() function.
    */
    void GetSequenceParams(std::vector<uint8_t>& seqParams);

    /**
    * @brief NvEncoder class virtual destructor.
    */
    virtual ~NvEncoder();

public:
    /**
    * @brief This a static function to get chroma offsets for YUV planar formats.
    */
    static void GetChromaSubPlaneOffsets(const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t pitch,
        const uint32_t height, std::vector<uint32_t>& chromaOffsets);
    /**
    * @brief This a static function to get the chroma plane pitch for YUV planar formats.
    */
    static uint32_t GetChromaPitch(const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t lumaPitch);

    /**
    * @brief This a static function to get the number of chroma planes for YUV planar formats.
    */
    static uint32_t GetNumChromaPlanes(const NV_ENC_BUFFER_FORMAT bufferFormat);

    /**
    * @brief This a static function to get the chroma plane width in bytes for YUV planar formats.
    */
    static uint32_t GetChromaWidthInBytes(const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t lumaWidth);

    /**
    * @brief This a static function to get the chroma planes height in bytes for YUV planar formats.
    */
    static uint32_t GetChromaHeight(const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t lumaHeight);


    /**
    * @brief This a static function to get the width in bytes for the frame.
    * For YUV planar format this is the width in bytes of the luma plane.
    */
    static uint32_t GetWidthInBytes(const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t width);

    /**
    * @brief This function returns the number of allocated buffers.
    */
    uint32_t GetEncoderBufferCount() const { return m_nEncoderBuffer; }
protected:

    /**
    * @brief NvEncoder class constructor.
    * NvEncoder class constructor cannot be called directly by the application.
    */
    NvEncoder(NV_ENC_DEVICE_TYPE eDeviceType, void* pDevice, uint32_t nWidth, uint32_t nHeight,
        NV_ENC_BUFFER_FORMAT eBufferFormat, uint32_t nOutputDelay);

    /**
    * @brief This function is used to check if hardware encoder is properly initialized.
    */
    bool IsHWEncoderInitialized() const { return m_hEncoder != NULL && m_bEncoderInitialized; }

    /**
    * @brief This function is used to register CUDA, D3D or OpenGL input buffers with NvEncodeAPI.
    * This is non public function and is called by derived class for allocating
    * and registering input buffers.
    */
    void RegisterInputResources(std::vector<void*> inputframes, NV_ENC_INPUT_RESOURCE_TYPE eResourceType,
        int width, int height, int pitch, NV_ENC_BUFFER_FORMAT bufferFormat, bool bReferenceFrame = false);

    /**
    * @brief This function is used to unregister resources which had been previously registered for encoding
    * using RegisterInputResources() function.
    */
    void UnregisterInputResources();

    /**
    * @brief This function is used to register CUDA, D3D or OpenGL input or output buffers with NvEncodeAPI.
    */
    NV_ENC_REGISTERED_PTR RegisterResource(void* pBuffer, NV_ENC_INPUT_RESOURCE_TYPE eResourceType,
        int width, int height, int pitch, NV_ENC_BUFFER_FORMAT bufferFormat, NV_ENC_BUFFER_USAGE bufferUsage = NV_ENC_INPUT_IMAGE,
        NV_ENC_FENCE_POINT_D3D12* pInputFencePoint = NULL);

    /**
    * @brief This function returns maximum width used to open the encoder session.
    * All encode input buffers are allocated using maximum dimensions.
    */
    uint32_t GetMaxEncodeWidth() const { return m_nMaxEncodeWidth; }

    /**
    * @brief This function returns maximum height used to open the encoder session.
    * All encode input buffers are allocated using maximum dimensions.
    */
    uint32_t GetMaxEncodeHeight() const { return m_nMaxEncodeHeight; }

    /**
    * @brief This function returns the completion event.
    */
    void* GetCompletionEvent(uint32_t eventIdx) { return (m_vpCompletionEvent.size() == m_nEncoderBuffer) ? m_vpCompletionEvent[eventIdx] : nullptr; }

    /**
    * @brief This function returns the current pixel format.
    */
    NV_ENC_BUFFER_FORMAT GetPixelFormat() const { return m_eBufferFormat; }

    /**
    * @brief This function is used to submit the encode commands to the
    * NVENC hardware.
    */
    NVENCSTATUS DoEncode(NV_ENC_INPUT_PTR inputBuffer, NV_ENC_OUTPUT_PTR outputBuffer, NV_ENC_PIC_PARAMS* pPicParams);

    /**
    * @brief This function is used to submit the encode commands to the
    * NVENC hardware for ME only mode.
    */
    //NVENCSTATUS DoMotionEstimation(NV_ENC_INPUT_PTR inputBuffer, NV_ENC_INPUT_PTR inputBufferForReference, NV_ENC_OUTPUT_PTR outputBuffer);

    /**
    * @brief This function is used to map the input buffers to NvEncodeAPI.
    */
    void MapResources(uint32_t bfrIdx);

    /**
    * @brief This function is used to wait for completion of encode command.
    */
    void WaitForCompletionEvent(int iEvent);

    /**
    * @brief This function is used to send EOS to HW encoder.
    */
    void SendEOS();

private:
    /**
    * @brief This is a private function which is used to check if there is any
    buffering done by encoder.
    * The encoder generally buffers data to encode B frames or for lookahead
    * or pipelining.
    */
    bool IsZeroDelay() { return m_nOutputDelay == 0; }

    /**
    * @brief This is a private function which is used to load the encode api shared library.
    */
    void LoadNvEncApi();

    /**
    * @brief This is a private function which is used to get the output packets
    * from the encoder HW.
    * This is called by DoEncode() function. If there is buffering enabled,
    * this may return without any output data.
    */
    void GetEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR>& vOutputBuffer, std::vector<std::vector<uint8_t>>& vPacket, std::vector<uint64_t>& outputTimeStamps, bool bOutputDelay);

    /**
    * @brief This is a private function which is used to initialize the bitstream buffers.
    * This is only used in the encoding mode.
    */
    void InitializeBitstreamBuffer();

    /**
    * @brief This is a private function which is used to destroy the bitstream buffers.
    * This is only used in the encoding mode.
    */
    void DestroyBitstreamBuffer();

    /**
    * @brief This is a private function which is used to destroy HW encoder.
    */
    void DestroyHWEncoder();

    /**
    * @brief This function is used to flush the encoder queue.
    */
    void FlushEncoder();

private:
    /**
    * @brief This is a pure virtual function which is used to allocate input buffers.
    * The derived classes must implement this function.
    */
    virtual void AllocateInputBuffers(int32_t numInputBuffers) = 0;

    /**
    * @brief This is a pure virtual function which is used to destroy input buffers.
    * The derived classes must implement this function.
    */
    virtual void ReleaseInputBuffers() = 0;

protected:
    void* m_hEncoder = nullptr;
    NV_ENCODE_API_FUNCTION_LIST m_nvenc;
    std::vector<NvEncInputFrame> m_vInputFrames;
    std::vector<NV_ENC_REGISTERED_PTR> m_vRegisteredResources;
    std::vector<NvEncInputFrame> m_vReferenceFrames;
    std::vector<NV_ENC_REGISTERED_PTR> m_vRegisteredResourcesForReference;
    std::vector<NV_ENC_INPUT_PTR> m_vMappedInputBuffers;
    std::vector<NV_ENC_INPUT_PTR> m_vMappedRefBuffers;
    std::vector<void*> m_vpCompletionEvent;

    int32_t m_iToSend = 0;
    int32_t m_iGot = 0;
    int32_t m_nEncoderBuffer = 0;
    int32_t m_nOutputDelay = 0;
    int32_t m_iInputFrame = 0;

private:
    void* m_pDevice;
    NV_ENC_DEVICE_TYPE m_eDeviceType;
    uint32_t m_nWidth;
    uint32_t m_nHeight;
    uint32_t m_nMaxEncodeWidth = 0;
    uint32_t m_nMaxEncodeHeight = 0;
    NV_ENC_BUFFER_FORMAT m_eBufferFormat;
    NV_ENC_INITIALIZE_PARAMS m_initializeParams = {};
    NV_ENC_CONFIG m_encodeConfig = {};
    bool m_bEncoderInitialized = false;
    uint32_t m_nExtraOutputDelay = 3; // To ensure encode and graphics can work in parallel, m_nExtraOutputDelay should be set to at least 1
    std::vector<NV_ENC_OUTPUT_PTR> m_vBitstreamOutputBuffer;
};
}}
#endif