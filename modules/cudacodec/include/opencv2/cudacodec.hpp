/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CUDACODEC_HPP
#define OPENCV_CUDACODEC_HPP

#ifndef __cplusplus
#  error cudacodec.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"

/**
  @addtogroup cuda
  @{
    @defgroup cudacodec Video Encoding/Decoding
  @}
 */

namespace cv { namespace cudacodec {

using namespace cuda;  // Stream

//! @addtogroup cudacodec
//! @{

////////////////////////////////// Video Encoding //////////////////////////////////

// Works only under Windows.
// Supports only H264 video codec and AVI files.

enum SurfaceFormat
{
    SF_UYVY = 0,
    SF_YUY2,
    SF_YV12,
    SF_NV12,
    SF_IYUV,
    SF_BGR,
    SF_GRAY = SF_BGR
};

/** @brief Different parameters for CUDA video encoder.
 */
struct CV_EXPORTS_W EncoderParams
{
    int P_Interval;      //!< NVVE_P_INTERVAL,
    int IDR_Period;      //!< NVVE_IDR_PERIOD,
    int DynamicGOP;      //!< NVVE_DYNAMIC_GOP,
    int RCType;          //!< NVVE_RC_TYPE,
    int AvgBitrate;      //!< NVVE_AVG_BITRATE,
    int PeakBitrate;     //!< NVVE_PEAK_BITRATE,
    int QP_Level_Intra;  //!< NVVE_QP_LEVEL_INTRA,
    int QP_Level_InterP; //!< NVVE_QP_LEVEL_INTER_P,
    int QP_Level_InterB; //!< NVVE_QP_LEVEL_INTER_B,
    int DeblockMode;     //!< NVVE_DEBLOCK_MODE,
    int ProfileLevel;    //!< NVVE_PROFILE_LEVEL,
    int ForceIntra;      //!< NVVE_FORCE_INTRA,
    int ForceIDR;        //!< NVVE_FORCE_IDR,
    int ClearStat;       //!< NVVE_CLEAR_STAT,
    int DIMode;          //!< NVVE_SET_DEINTERLACE,
    int Presets;         //!< NVVE_PRESETS,
    int DisableCabac;    //!< NVVE_DISABLE_CABAC,
    int NaluFramingType; //!< NVVE_CONFIGURE_NALU_FRAMING_TYPE
    int DisableSPSPPS;   //!< NVVE_DISABLE_SPS_PPS

    EncoderParams();
    /** @brief Constructors.

    @param configFile Config file name.

    Creates default parameters or reads parameters from config file.
     */
    explicit EncoderParams(const String& configFile);

    /** @brief Reads parameters from config file.

    @param configFile Config file name.
     */
    void load(const String& configFile);
    /** @brief Saves parameters to config file.

    @param configFile Config file name.
     */
    void save(const String& configFile) const;
};

/** @brief Callbacks for CUDA video encoder.
 */
class CV_EXPORTS_W EncoderCallBack
{
public:
    enum PicType
    {
        IFRAME = 1,
        PFRAME = 2,
        BFRAME = 3
    };

    virtual ~EncoderCallBack() {}

    /** @brief Callback function to signal the start of bitstream that is to be encoded.

    Callback must allocate buffer for CUDA encoder and return pointer to it and it's size.
     */
    virtual uchar* acquireBitStream(int* bufferSize) = 0;

    /** @brief Callback function to signal that the encoded bitstream is ready to be written to file.
    */
    virtual void releaseBitStream(unsigned char* data, int size) = 0;

    /** @brief Callback function to signal that the encoding operation on the frame has started.

    @param frameNumber
    @param picType Specify frame type (I-Frame, P-Frame or B-Frame).
     */
    CV_WRAP virtual void onBeginFrame(int frameNumber, EncoderCallBack::PicType picType) = 0;

    /** @brief Callback function signals that the encoding operation on the frame has finished.

    @param frameNumber
    @param picType Specify frame type (I-Frame, P-Frame or B-Frame).
     */
    CV_WRAP virtual void onEndFrame(int frameNumber, EncoderCallBack::PicType picType) = 0;
};

/** @brief Video writer interface.

The implementation uses H264 video codec.

@note Currently only Windows platform is supported.

@note
   -   An example on how to use the videoWriter class can be found at
        opencv_source_code/samples/gpu/video_writer.cpp
 */
class CV_EXPORTS_W VideoWriter
{
public:
    virtual ~VideoWriter() {}

    /** @brief Writes the next video frame.

    @param frame The written frame.
    @param lastFrame Indicates that it is end of stream. The parameter can be ignored.

    The method write the specified image to video file. The image must have the same size and the same
    surface format as has been specified when opening the video writer.
     */
    CV_WRAP virtual void write(InputArray frame, bool lastFrame = false) = 0;

    CV_WRAP virtual EncoderParams getEncoderParams() const = 0;
};

/** @brief Creates video writer.

@param fileName Name of the output video file. Only AVI file format is supported.
@param frameSize Size of the input video frames.
@param fps Framerate of the created video stream.
@param format Surface format of input frames ( SF_UYVY , SF_YUY2 , SF_YV12 , SF_NV12 ,
SF_IYUV , SF_BGR or SF_GRAY). BGR or gray frames will be converted to YV12 format before
encoding, frames with other formats will be used as is.

The constructors initialize video writer. FFMPEG is used to write videos. User can implement own
multiplexing with cudacodec::EncoderCallBack .
 */
CV_EXPORTS_W Ptr<cudacodec::VideoWriter> createVideoWriter(const String& fileName, Size frameSize, double fps, SurfaceFormat format = SF_BGR);
/** @overload
@param fileName Name of the output video file. Only AVI file format is supported.
@param frameSize Size of the input video frames.
@param fps Framerate of the created video stream.
@param params Encoder parameters. See cudacodec::EncoderParams .
@param format Surface format of input frames ( SF_UYVY , SF_YUY2 , SF_YV12 , SF_NV12 ,
SF_IYUV , SF_BGR or SF_GRAY). BGR or gray frames will be converted to YV12 format before
encoding, frames with other formats will be used as is.
*/
CV_EXPORTS_W Ptr<cudacodec::VideoWriter> createVideoWriter(const String& fileName, Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR);

/** @overload
@param encoderCallback Callbacks for video encoder. See cudacodec::EncoderCallBack . Use it if you
want to work with raw video stream.
@param frameSize Size of the input video frames.
@param fps Framerate of the created video stream.
@param format Surface format of input frames ( SF_UYVY , SF_YUY2 , SF_YV12 , SF_NV12 ,
SF_IYUV , SF_BGR or SF_GRAY). BGR or gray frames will be converted to YV12 format before
encoding, frames with other formats will be used as is.
*/
CV_EXPORTS_W Ptr<cudacodec::VideoWriter> createVideoWriter(const Ptr<EncoderCallBack>& encoderCallback, Size frameSize, double fps, SurfaceFormat format = SF_BGR);
/** @overload
@param encoderCallback Callbacks for video encoder. See cudacodec::EncoderCallBack . Use it if you
want to work with raw video stream.
@param frameSize Size of the input video frames.
@param fps Framerate of the created video stream.
@param params Encoder parameters. See cudacodec::EncoderParams.
@param format Surface format of input frames ( SF_UYVY , SF_YUY2 , SF_YV12 , SF_NV12 ,
SF_IYUV , SF_BGR or SF_GRAY). BGR or gray frames will be converted to YV12 format before
encoding, frames with other formats will be used as is.
*/
CV_EXPORTS_W Ptr<cudacodec::VideoWriter> createVideoWriter(const Ptr<EncoderCallBack>& encoderCallback, Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR);

////////////////////////////////// Video Decoding //////////////////////////////////////////

/** @brief Video codecs supported by cudacodec::VideoReader .
 */
enum Codec
{
    MPEG1 = 0,
    MPEG2,
    MPEG4,
    VC1,
    H264,
    JPEG,
    H264_SVC,
    H264_MVC,
    HEVC,
    VP8,
    VP9,
    AV1,
    NumCodecs,

    Uncompressed_YUV420 = (('I'<<24)|('Y'<<16)|('U'<<8)|('V')),   //!< Y,U,V (4:2:0)
    Uncompressed_YV12   = (('Y'<<24)|('V'<<16)|('1'<<8)|('2')),   //!< Y,V,U (4:2:0)
    Uncompressed_NV12   = (('N'<<24)|('V'<<16)|('1'<<8)|('2')),   //!< Y,UV  (4:2:0)
    Uncompressed_YUYV   = (('Y'<<24)|('U'<<16)|('Y'<<8)|('V')),   //!< YUYV/YUY2 (4:2:2)
    Uncompressed_UYVY   = (('U'<<24)|('Y'<<16)|('V'<<8)|('Y'))    //!< UYVY (4:2:2)
};

/** @brief Chroma formats supported by cudacodec::VideoReader.
 */
enum ChromaFormat
{
    Monochrome = 0,
    YUV420,
    YUV422,
    YUV444,
    NumFormats
};

/** @brief Deinterlacing mode used by decoder.
* @param Weave Weave both fields (no deinterlacing). For progressive content and for content that doesn't need deinterlacing.
* Bob Drop one field.
* @param Adaptive Adaptive deinterlacing needs more video memory than other deinterlacing modes.
* */
enum DeinterlaceMode
{
    Weave = 0,
    Bob = 1,
    Adaptive = 2
};

/** @brief Struct providing information about video file format. :
 */
struct CV_EXPORTS_W_SIMPLE FormatInfo
{
    CV_WRAP FormatInfo() : nBitDepthMinus8(-1), ulWidth(0), ulHeight(0), width(0), height(0), ulMaxWidth(0), ulMaxHeight(0), valid(false),
        fps(0), ulNumDecodeSurfaces(0) {};

    CV_PROP_RW Codec codec;
    CV_PROP_RW ChromaFormat chromaFormat;
    CV_PROP_RW int nBitDepthMinus8;
    CV_PROP_RW int ulWidth;//!< Coded sequence width in pixels.
    CV_PROP_RW int ulHeight;//!< Coded sequence height in pixels.
    CV_PROP_RW int width;//!< Width of the decoded frame returned by nextFrame(frame).
    CV_PROP_RW int height;//!< Height of the decoded frame returned by nextFrame(frame).
    int ulMaxWidth;
    int ulMaxHeight;
    CV_PROP_RW Rect displayArea;//!< ROI inside the decoded frame returned by nextFrame(frame), containing the useable video frame.
    CV_PROP_RW bool valid;
    CV_PROP_RW double fps;
    CV_PROP_RW int ulNumDecodeSurfaces;//!< Maximum number of internal decode surfaces.
    CV_PROP_RW DeinterlaceMode deinterlaceMode;
};

/** @brief cv::cudacodec::VideoReader generic properties identifier.
*/
enum class VideoReaderProps {
    PROP_DECODED_FRAME_IDX = 0, //!< Index for retrieving the decoded frame using retrieve().
    PROP_EXTRA_DATA_INDEX = 1, //!< Index for retrieving the extra data associated with a video source using retrieve().
    PROP_RAW_PACKAGES_BASE_INDEX = 2, //!< Base index for retrieving raw encoded data using retrieve().
    PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB = 3, //!< Number of raw packages recieved since the last call to grab().
    PROP_RAW_MODE = 4, //!< Status of raw mode.
    PROP_LRF_HAS_KEY_FRAME = 5, //!< FFmpeg source only - Indicates whether the Last Raw Frame (LRF), output from VideoReader::retrieve() when VideoReader is initialized in raw mode, contains encoded data for a key frame.
    PROP_COLOR_FORMAT = 6, //!< Set the ColorFormat of the decoded frame.  This can be changed before every call to nextFrame() and retrieve().
    PROP_UDP_SOURCE = 7, //!< Status of VideoReaderInitParams::udpSource initialization.
    PROP_ALLOW_FRAME_DROP = 8, //!< Status of VideoReaderInitParams::allowFrameDrop initialization.
#ifndef CV_DOXYGEN
    PROP_NOT_SUPPORTED
#endif
};

/** @brief ColorFormat for the frame returned by nextFrame()/retrieve().
*/
enum class ColorFormat {
    BGRA = 1,
    BGR = 2,
    GRAY = 3,
    YUV = 4,
#ifndef CV_DOXYGEN
    PROP_NOT_SUPPORTED
#endif
};

/** @brief Video reader interface.

@note
   -   An example on how to use the videoReader class can be found at
        opencv_source_code/samples/gpu/video_reader.cpp
 */
class CV_EXPORTS_W VideoReader
{
public:
    virtual ~VideoReader() {}

    /** @brief Grabs, decodes and returns the next video frame.

    @param [out] frame The video frame.
    @param stream Stream for the asynchronous version.
    @return `false` if no frames have been grabbed.

    If no frames have been grabbed (there are no more frames in video file), the methods return false.
    The method throws an Exception if error occurs.
     */
    CV_WRAP virtual bool nextFrame(CV_OUT GpuMat& frame, Stream &stream = Stream::Null()) = 0;

    /** @brief Returns information about video file format.
    */
    CV_WRAP virtual FormatInfo format() const = 0;

    /** @brief Grabs the next frame from the video source.

    @param stream Stream for the asynchronous version.
    @return `true` (non-zero) in the case of success.

    The method/function grabs the next frame from video file or camera and returns true (non-zero) in
    the case of success.

    The primary use of the function is for reading both the encoded and decoded video data when rawMode is enabled.  With rawMode enabled
    retrieve() can be called following grab() to retrieve all the data associated with the current video source since the last call to grab() or the creation of the VideoReader.
     */
    CV_WRAP virtual bool grab(Stream& stream = Stream::Null()) = 0;

    /** @brief Returns previously grabbed video data.

    @param [out] frame The returned data which depends on the provided idx.
    @param idx Determines the returned data inside image. The returned data can be the:
     - Decoded frame, idx = get(PROP_DECODED_FRAME_IDX).
     - Extra data if available, idx = get(PROP_EXTRA_DATA_INDEX).
     - Raw encoded data package.  To retrieve package i,  idx = get(PROP_RAW_PACKAGES_BASE_INDEX) + i with i < get(PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB)
    @return `false` if no frames have been grabbed

    The method returns data associated with the current video source since the last call to grab() or the creation of the VideoReader. If no data is present
    the method returns false and the function returns an empty image.
     */
    virtual bool retrieve(OutputArray frame, const size_t idx = static_cast<size_t>(VideoReaderProps::PROP_DECODED_FRAME_IDX)) const = 0;

    /** @brief Returns previously grabbed encoded video data.

    @param [out] frame The encoded video data.
    @param idx Determines the returned data inside image. The returned data can be the:
     - Extra data if available, idx = get(PROP_EXTRA_DATA_INDEX).
     - Raw encoded data package.  To retrieve package i,  idx = get(PROP_RAW_PACKAGES_BASE_INDEX) + i with i < get(PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB)
    @return `false` if no frames have been grabbed

    The method returns data associated with the current video source since the last call to grab() or the creation of the VideoReader. If no data is present
    the method returns false and the function returns an empty image.
     */
    CV_WRAP inline bool retrieve(CV_OUT Mat& frame, const size_t idx) const {
        return retrieve(OutputArray(frame), idx);
    }

    /** @brief Returns the next video frame.

    @param [out] frame The video frame.  If grab() has not been called then this will be empty().
    @return `false` if no frames have been grabbed

    The method returns data associated with the current video source since the last call to grab(). If no data is present
    the method returns false and the function returns an empty image.
     */
    CV_WRAP inline bool retrieve(CV_OUT GpuMat& frame) const {
        return retrieve(OutputArray(frame));
    }

    /** @brief Sets a property in the VideoReader.

    @param propertyId Property identifier from cv::cudacodec::VideoReaderProps (eg. cv::cudacodec::VideoReaderProps::PROP_DECODED_FRAME_IDX,
    cv::cudacodec::VideoReaderProps::PROP_EXTRA_DATA_INDEX, ...).
    @param propertyVal Value of the property.
    @return `true` if the property has been set.
     */
    virtual bool set(const VideoReaderProps propertyId, const double propertyVal) = 0;
    CV_WRAP inline bool setVideoReaderProps(const VideoReaderProps propertyId, double propertyVal) {
        return set(propertyId, propertyVal);
    }

    /** @brief Set the desired ColorFormat for the frame returned by nextFrame()/retrieve().

    @param colorFormat Value of the ColorFormat.
     */
    CV_WRAP virtual void set(const ColorFormat colorFormat) = 0;

    /** @brief Returns the specified VideoReader property

    @param propertyId Property identifier from cv::cudacodec::VideoReaderProps (eg. cv::cudacodec::VideoReaderProps::PROP_DECODED_FRAME_IDX,
    cv::cudacodec::VideoReaderProps::PROP_EXTRA_DATA_INDEX, ...).
    @param propertyVal
     - In: Optional value required for querying specific propertyId's, e.g. the index of the raw package to be checked for a key frame (cv::cudacodec::VideoReaderProps::PROP_LRF_HAS_KEY_FRAME).
     - Out: Value of the property.
    @return `true` unless the property is not supported.
    */
    virtual bool get(const VideoReaderProps propertyId, double& propertyVal) const = 0;
    CV_WRAP virtual bool getVideoReaderProps(const VideoReaderProps propertyId,  CV_OUT double& propertyValOut, double propertyValIn = 0) const = 0;

    /** @brief Retrieves the specified property used by the VideoSource.

    @param propertyId Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...)
    or one from @ref videoio_flags_others.
    @param propertyVal Value for the specified property.

    @return `true` unless the property is unset set or not supported.
     */
    CV_WRAP virtual bool get(const int propertyId, CV_OUT double& propertyVal) const = 0;
};

/** @brief Interface for video demultiplexing. :

User can implement own demultiplexing by implementing this interface.
 */
class CV_EXPORTS_W RawVideoSource
{
public:
    virtual ~RawVideoSource() {}

    /** @brief Returns next packet with RAW video frame.

    @param data Pointer to frame data.
    @param size Size in bytes of current frame.
     */
    virtual bool getNextPacket(unsigned char** data, size_t* size) = 0;

    /** @brief Returns true if the last packet contained a key frame.
     */
    virtual bool lastPacketContainsKeyFrame() const { return false; }

    /** @brief Returns information about video file format.
    */
    virtual FormatInfo format() const = 0;

    /** @brief Updates the coded width and height inside format.
    */
    virtual void updateFormat(const FormatInfo& videoFormat) = 0;

    /** @brief Returns any extra data associated with the video source.

    @param extraData 1D cv::Mat containing the extra data if it exists.
     */
    virtual void getExtraData(cv::Mat& extraData) const = 0;

    /** @brief Retrieves the specified property used by the VideoSource.

    @param propertyId Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...)
    or one from @ref videoio_flags_others.
    @param propertyVal Value for the specified property.

    @return `true` unless the property is unset set or not supported.
     */
    virtual bool get(const int propertyId, double& propertyVal) const = 0;
};

/** @brief VideoReader initialization parameters
@param udpSource Remove validation which can cause VideoReader() to throw exceptions when reading from a UDP source.
@param allowFrameDrop Allow frames to be dropped when ingesting from a live capture source to prevent delay and eventual disconnection
when calls to nextFrame()/grab() cannot keep up with the source's fps.  Only use if delay and disconnection are a problem, i.e. not when decoding from
video files where setting this flag will cause frames to be unnecessarily discarded.
@param minNumDecodeSurfaces Minimum number of internal decode surfaces used by the hardware decoder.  NVDEC will automatically determine the minimum number of
surfaces it requires for correct functionality and optimal video memory usage but not necessarily for best performance, which depends on the design of the
overall application. The optimal number of decode surfaces (in terms of performance and memory utilization) should be decided by experimentation for each application,
but it cannot go below the number determined by NVDEC.
@param rawMode Allow the raw encoded data which has been read up until the last call to grab() to be retrieved by calling retrieve(rawData,RAW_DATA_IDX).
*/
struct CV_EXPORTS_W_SIMPLE VideoReaderInitParams {
    CV_WRAP VideoReaderInitParams() : udpSource(false), allowFrameDrop(false), minNumDecodeSurfaces(0), rawMode(0) {};
    CV_PROP_RW bool udpSource;
    CV_PROP_RW bool allowFrameDrop;
    CV_PROP_RW int minNumDecodeSurfaces;
    CV_PROP_RW bool rawMode;
};

/** @brief Creates video reader.

@param filename Name of the input video file.
@param sourceParams Pass through parameters for VideoCapure.  VideoCapture with the FFMpeg back end (CAP_FFMPEG) is used to parse the video input.
The `sourceParams` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`.
    See cv::VideoCaptureProperties
e.g. when streaming from an RTSP source CAP_PROP_OPEN_TIMEOUT_MSEC may need to be set.
@param params Initializaton parameters. See cv::cudacodec::VideoReaderInitParams.

FFMPEG is used to read videos. User can implement own demultiplexing with cudacodec::RawVideoSource
 */
CV_EXPORTS_W Ptr<VideoReader> createVideoReader(const String& filename, const std::vector<int>& sourceParams = {}, const VideoReaderInitParams params = VideoReaderInitParams());

/** @overload
@param source RAW video source implemented by user.
@param params Initializaton parameters. See cv::cudacodec::VideoReaderInitParams.
*/
CV_EXPORTS_W Ptr<VideoReader> createVideoReader(const Ptr<RawVideoSource>& source, const VideoReaderInitParams params = VideoReaderInitParams());

//! @}

}} // namespace cv { namespace cudacodec {

#endif /* OPENCV_CUDACODEC_HPP */
