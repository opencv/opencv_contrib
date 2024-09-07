#include "opencv2/cudev/util/vec_traits.hpp"
#include "opencv2/core/cuda_types.hpp"

namespace
{
template<typename T, int cn>
constexpr T __CV_CUDA_HOST_DEVICE__ get_channel(const int index, const typename cv::cudev::MakeVec<T, cn>::type& value) noexcept
{
  return reinterpret_cast<const T*>(&value)[index];
}

template<typename T, int cn>
__global__ void extract_channel_kernel(cv::cuda::PtrStepSz<typename cv::cudev::MakeVec<T, cn>::type> many_channel,
                                       cv::cuda::PtrStepSz<T> single_channel, const int channel_index)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= many_channel.cols || y >= many_channel.rows) {
    return;
  }

  single_channel(y, x) = ::get_channel<T, cn>(channel_index, many_channel(y, x));
}

template<typename T, int cn>
void extract_channel_impl(cv::cuda::PtrStepSz<typename cv::cudev::MakeVec<T, cn>::type> many_channel,
                          cv::cuda::PtrStepSz<T> single_channel, const int channel_index, cv::cuda::Stream& stream)
{

  static constexpr dim3 block(32, 8);
  const dim3 grid(cv::cudev::divUp(many_channel.cols, block.x), cv::cudev::divUp(many_channel.rows, block.y));
  ::extract_channel_kernel<T, cn><<<grid, block, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(many_channel, single_channel, channel_index);
}

template<typename T, int depth>
void extract_channel_t(const cv::cuda::GpuMat input, cv::cuda::GpuMat& output, const int channel_index, cv::cuda::Stream& stream)
{
  static constexpr auto output_type = CV_MAKE_TYPE(depth, 1);
  if (output.size() != input.size() || output.type() != output_type) {
    output = cv::cuda::GpuMat(input.size(), output_type);
  }

  switch (input.channels()) {
  case 1:
    ::extract_channel_impl<T, 1>(input, output, channel_index, stream);
    break;
  case 2:
    ::extract_channel_impl<T, 2>(input, output, channel_index, stream);
    break;
  case 3:
    ::extract_channel_impl<T, 3>(input, output, channel_index, stream);
    break;
  case 4:
    ::extract_channel_impl<T, 4>(input, output, channel_index, stream);
    break;
    default:
    CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported channel count");
  }
}
}  // namespace

namespace cv {
namespace cuda {
void extractChannel(const GpuMat input, GpuMat& output, const int channel_index, Stream& stream)
{
  switch (CV_MAT_DEPTH(input.type())) {
  case CV_8U:
    ::extract_channel_t<std::uint8_t, CV_8U>(input, output, channel_index, stream);
    break;
  case CV_8S:
    ::extract_channel_t<std::int8_t, CV_8S>(input, output, channel_index, stream);
    break;
  case CV_16S:
    ::extract_channel_t<std::int16_t, CV_16S>(input, output, channel_index, stream);
    break;
  case CV_16U:
    ::extract_channel_t<std::uint16_t, CV_16U>(input, output, channel_index, stream);
    break;
  case CV_32F:
    ::extract_channel_t<std::float_t, CV_32F>(input, output, channel_index, stream);
    break;
  case CV_32S:
    ::extract_channel_t<std::int32_t, CV_32S>(input, output, channel_index, stream);
    break;
  case CV_64F:
    ::extract_channel_t<std::double_t, CV_64F>(input, output, channel_index, stream);
    break;
  case CV_16F:
    [[fallthrough]];
  default:
    CV_Error(Error::StsUnsupportedFormat, "Unsupported data type");
  }
}
}  // namespace cuda
}  // namespace cv
