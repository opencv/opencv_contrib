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

/** @brief Video codecs supported by cudacodec::VideoReader and cudacodec::VideoWriter.
@note
   -   Support will depend on your hardware, refer to the Nvidia Video Codec SDK Video Encode and Decode GPU Support Matrix for details.
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

    Uncompressed_YUV420 = (('I' << 24) | ('Y' << 16) | ('U' << 8) | ('V')),   //!< Y,U,V (4:2:0)
    Uncompressed_YV12 = (('Y' << 24) | ('V' << 16) | ('1' << 8) | ('2')),   //!< Y,V,U (4:2:0)
    Uncompressed_NV12 = (('N' << 24) | ('V' << 16) | ('1' << 8) | ('2')),   //!< Y,UV  (4:2:0)
    Uncompressed_YUYV = (('Y' << 24) | ('U' << 16) | ('Y' << 8) | ('V')),   //!< YUYV/YUY2 (4:2:2)
    Uncompressed_UYVY = (('U' << 24) | ('Y' << 16) | ('V' << 8) | ('Y'))    //!< UYVY (4:2:2)
};

/** @brief ColorFormat for the frame returned by VideoReader::nextFrame() and VideoReader::retrieve() or used to initialize a VideoWriter.
*/
enum class ColorFormat {
    UNDEFINED = 0,
    BGRA = 1, //!< OpenCV color format, can be used with both VideoReader and VideoWriter.
    BGR = 2, //!< OpenCV color format, can be used with both VideoReader and VideoWriter.
    GRAY = 3, //!< OpenCV color format, can be used with both VideoReader and VideoWriter.
    NV_NV12 = 4, //!< Nvidia color format - equivalent to YUV - Semi-Planar YUV [Y plane followed by interleaved UV plane], can be used with both VideoReader and VideoWriter.

    RGB = 5, //!< OpenCV color format, can only be used with VideoWriter.
    RGBA = 6, //!< OpenCV color format, can only be used with VideoWriter.
    NV_YV12 = 8, //!< Nvidia Buffer Format - Planar YUV [Y plane followed by V and U planes], use with VideoReader, can only be used with VideoWriter.
    NV_IYUV = 9, //!< Nvidia Buffer Format - Planar YUV [Y plane followed by U and V planes], use with VideoReader, can only be used with VideoWriter.
    NV_YUV444 = 10, //!< Nvidia Buffer Format - Planar YUV [Y plane followed by U and V planes], use with VideoReader, can only be used with VideoWriter.
    NV_AYUV = 11, //!< Nvidia Buffer Format - 8 bit Packed A8Y8U8V8. This is a word-ordered format where a pixel is represented by a 32-bit word with V in the lowest 8 bits, U in the next 8 bits, Y in the 8 bits after that and A in the highest 8 bits, can only be used with VideoWriter.
#ifndef CV_DOXYGEN
    PROP_NOT_SUPPORTED
#endif
};

/** @brief Rate Control Modes.
*/
enum EncodeParamsRcMode {
    ENC_PARAMS_RC_CONSTQP = 0x0, //!< Constant QP mode.
    ENC_PARAMS_RC_VBR = 0x1, //!< Variable bitrate mode.
    ENC_PARAMS_RC_CBR = 0x2 //!< Constant bitrate mode.
};

/** @brief Multi Pass Encoding.
*/
enum EncodeMultiPass
{
    ENC_MULTI_PASS_DISABLED = 0x0, //!< Single Pass.
    ENC_TWO_PASS_QUARTER_RESOLUTION = 0x1, //!< Two Pass encoding is enabled where first Pass is quarter resolution.
    ENC_TWO_PASS_FULL_RESOLUTION = 0x2, //!< Two Pass encoding is enabled where first Pass is full resolution.
};


/** @brief Supported Encoder Profiles.
*/
enum EncodeProfile {
    ENC_CODEC_PROFILE_AUTOSELECT = 0,
    ENC_H264_PROFILE_BASELINE = 1,
    ENC_H264_PROFILE_MAIN = 2,
    ENC_H264_PROFILE_HIGH = 3,
    ENC_H264_PROFILE_HIGH_444 = 4,
    ENC_H264_PROFILE_STEREO = 5,
    ENC_H264_PROFILE_PROGRESSIVE_HIGH = 6,
    ENC_H264_PROFILE_CONSTRAINED_HIGH = 7,
    ENC_HEVC_PROFILE_MAIN = 8,
    ENC_HEVC_PROFILE_MAIN10 = 9,
    ENC_HEVC_PROFILE_FREXT = 10
};

/** @brief Nvidia Encoding Presets. Performance degrades and quality improves as we move from P1 to P7.
*/
enum EncodePreset {
    ENC_PRESET_P1 = 1,
    ENC_PRESET_P2 = 2,
    ENC_PRESET_P3 = 3,
    ENC_PRESET_P4 = 4,
    ENC_PRESET_P5 = 5,
    ENC_PRESET_P6 = 6,
    ENC_PRESET_P7 = 7
};

/** @brief Tuning information.
*/
enum EncodeTuningInfo {
    ENC_TUNING_INFO_UNDEFINED = 0, //!< Undefined tuningInfo. Invalid value for encoding.
    ENC_TUNING_INFO_HIGH_QUALITY = 1, //!< Tune presets for latency tolerant encoding.
    ENC_TUNING_INFO_LOW_LATENCY = 2, //!< Tune presets for low latency streaming.
    ENC_TUNING_INFO_ULTRA_LOW_LATENCY = 3, //!< Tune presets for ultra low latency streaming.
    ENC_TUNING_INFO_LOSSLESS = 4, //!< Tune presets for lossless encoding.
    ENC_TUNING_INFO_COUNT
};

/** Quantization Parameter for each type of frame when using ENC_PARAMS_RC_MODE::ENC_PARAMS_RC_CONSTQP.
*/
struct CV_EXPORTS_W_SIMPLE EncodeQp
{
    CV_PROP_RW uint32_t qpInterP; //!< Specifies QP value for P-frame.
    CV_PROP_RW uint32_t qpInterB; //!< Specifies QP value for B-frame.
    CV_PROP_RW uint32_t qpIntra; //!< Specifies QP value for Intra Frame.
};

/** @brief Different parameters for CUDA video encoder.
*/
struct CV_EXPORTS_W_SIMPLE EncoderParams
{
public:
    CV_WRAP EncoderParams() : nvPreset(ENC_PRESET_P3), tuningInfo(ENC_TUNING_INFO_HIGH_QUALITY), encodingProfile(ENC_CODEC_PROFILE_AUTOSELECT),
        rateControlMode(ENC_PARAMS_RC_VBR), multiPassEncoding(ENC_MULTI_PASS_DISABLED), constQp({ 0,0,0 }), averageBitRate(0), maxBitRate(0),
        targetQuality(30), gopLength(0) {};

    CV_PROP_RW EncodePreset nvPreset;
    CV_PROP_RW EncodeTuningInfo tuningInfo;
    CV_PROP_RW EncodeProfile encodingProfile;
    CV_PROP_RW EncodeParamsRcMode rateControlMode;
    CV_PROP_RW EncodeMultiPass multiPassEncoding;
    CV_PROP_RW EncodeQp constQp; //!< QP's for ENC_PARAMS_RC_CONSTQP.
    CV_PROP_RW int averageBitRate; //!< target bitrate for ENC_PARAMS_RC_VBR and ENC_PARAMS_RC_CBR.
    CV_PROP_RW int maxBitRate; //!< upper bound on bitrate for ENC_PARAMS_RC_VBR and ENC_PARAMS_RC_CONSTQP.
    CV_PROP_RW uint8_t targetQuality; //!< value 0 - 51 where video quality decreases as targetQuality increases, used with ENC_PARAMS_RC_VBR.
    CV_PROP_RW int gopLength;
};
CV_EXPORTS bool operator==(const EncoderParams& lhs, const EncoderParams& rhs);

/** @brief Interface for encoder callbacks.

User can implement own multiplexing by implementing this interface.
*/
class CV_EXPORTS_W EncoderCallback {
public:
    /** @brief Callback function to signal that the encoded bitstream for one or more frames is ready.

    @param vPacket The raw bitstream for one or more frames.
    */
    virtual void onEncoded(std::vector<std::vector<uint8_t>> vPacket) = 0;

    /** @brief Callback function to that the encoding has finished.
    * */
    virtual void onEncodingFinished() = 0;

    virtual ~EncoderCallback() {}
};

/** @brief Video writer interface.

Available when built with WITH_NVCUVENC=ON while Nvidia's Video Codec SDK is installed.

Encoding support is dependent on the GPU, refer to the Nvidia Video Codec SDK Video Encode and Decode GPU Support Matrix for details.

@note
   -   An example on how to use the videoWriter class can be found at
        opencv_source_code/samples/gpu/video_writer.cpp
*/
class CV_EXPORTS_W VideoWriter
{
public:
    virtual ~VideoWriter() {}

    /** @brief Writes the next video frame.

    @param frame The framet to be written.

    The method encodes the specified image to a video stream. The image must have the same size and the same
    surface format as has been specified when opening the video writer.
    */
    CV_WRAP virtual void write(InputArray frame) = 0;

    /** @brief Retrieve the encoding parameters.
    */
    CV_WRAP virtual EncoderParams getEncoderParams() const = 0;

    /** @brief Waits until the encoding process has finished before calling EncoderCallback::onEncodingFinished().
    */
    CV_WRAP virtual void release() = 0;
};

/** @brief Creates video writer.

@param fileName Name of the output video file. Only raw h264 or hevc files are supported.
@param frameSize Size of the input video frames.
@param codec Codec.
@param fps Framerate of the created video stream.
@param colorFormat OpenCv color format of the frames to be encoded.
@param encoderCallback Callbacks for video encoder. See cudacodec::EncoderCallback. Required for working with the encoded video stream.
@param stream Stream for frame pre-processing.
*/
CV_EXPORTS_W Ptr<cudacodec::VideoWriter> createVideoWriter(const String& fileName, const Size frameSize, const Codec codec = Codec::H264, const double fps = 25.0,
    const ColorFormat colorFormat = ColorFormat::BGR, Ptr<EncoderCallback> encoderCallback = 0, const Stream& stream = Stream::Null());

/** @brief Creates video writer.

@param fileName Name of the output video file. Only raw h264 or hevc files are supported.
@param frameSize Size of the input video frames.
@param codec Codec.
@param fps Framerate of the created video stream.
@param colorFormat OpenCv color format of the frames to be encoded.
@param params Additional encoding parameters.
@param encoderCallback Callbacks for video encoder. See cudacodec::EncoderCallback. Required for working with the encoded video stream.
@param stream Stream for frame pre-processing.
*/
CV_EXPORTS_W Ptr<cudacodec::VideoWriter> createVideoWriter(const String& fileName, const Size frameSize, const Codec codec, const double fps,  const ColorFormat colorFormat,
    const EncoderParams& params, Ptr<EncoderCallback> encoderCallback = 0, const Stream& stream = Stream::Null());

////////////////////////////////// Video Decoding //////////////////////////////////////////

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
    CV_WRAP FormatInfo() : nBitDepthMinus8(-1), nBitDepthChromaMinus8(-1), ulWidth(0), ulHeight(0), width(0), height(0), ulMaxWidth(0), ulMaxHeight(0), valid(false),
        fps(0), ulNumDecodeSurfaces(0), videoFullRangeFlag(false) {};

    CV_PROP_RW Codec codec;
    CV_PROP_RW ChromaFormat chromaFormat;
    CV_PROP_RW int nBitDepthMinus8;
    CV_PROP_RW int nBitDepthChromaMinus8;
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
    CV_PROP_RW cv::Size targetSz;//!< Post-processed size of the output frame.
    CV_PROP_RW cv::Rect srcRoi;//!< Region of interest decoded from video source.
    CV_PROP_RW cv::Rect targetRoi;//!< Region of interest in the output frame containing the decoded frame.
    CV_PROP_RW bool videoFullRangeFlag;//!< Output value indicating if the black level, luma and chroma of the source are represented using the full or limited range (AKA TV or "analogue" range) of values as defined in Annex E of the ITU-T Specification.  Internally the conversion from NV12 to BGR obeys ITU 709.
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

/** @brief Video reader interface.

Available when built with WITH_NVCUVID=ON while Nvidia's Video Codec SDK is installed.

Decoding support is dependent on the GPU, refer to the Nvidia Video Codec SDK Video Encode and Decode GPU Support Matrix for details.

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
    @return `true` unless the colorFormat is not supported.
     */
    CV_WRAP virtual bool set(const ColorFormat colorFormat) = 0;

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
@param targetSz Post-processed size (width/height should be multiples of 2) of the output frame, defaults to the size of the encoded video source.
@param srcRoi Region of interest (x/width should be multiples of 4 and y/height multiples of 2) decoded from video source, defaults to the full frame.
@param targetRoi Region of interest (x/width should be multiples of 4 and y/height multiples of 2) within the output frame to copy and resize the decoded frame to,
defaults to the full frame.
*/
struct CV_EXPORTS_W_SIMPLE VideoReaderInitParams {
    CV_WRAP VideoReaderInitParams() : udpSource(false), allowFrameDrop(false), minNumDecodeSurfaces(0), rawMode(0) {};
    CV_PROP_RW bool udpSource;
    CV_PROP_RW bool allowFrameDrop;
    CV_PROP_RW int minNumDecodeSurfaces;
    CV_PROP_RW bool rawMode;
    CV_PROP_RW cv::Size targetSz;
    CV_PROP_RW cv::Rect srcRoi;
    CV_PROP_RW cv::Rect targetRoi;
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
