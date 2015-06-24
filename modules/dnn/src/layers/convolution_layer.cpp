#include "../precomp.hpp"
#include "layers_common.hpp"

namespace cv
{
namespace dnn
{
    //TODO: simultaneously convolution and bias addition for cache optimization
    class ConvolutionLayer : public Layer
    {
        bool bias;
        int numOutput, group;
        int padH, padW;
        int strideH, strideW;
        int kernelH, kernelW;

        int inH, inW, inCn, kerSize;
        int outH, outW;
        int groupCn, groupCnOut;

        Mat srcColsMat, biasOnesMat;

        void computeOutputShape(int inH, int inW);

    public:
        ConvolutionLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };


    REGISTER_LAYER_CLASS(Convolution, ConvolutionLayer)


    ConvolutionLayer::ConvolutionLayer(LayerParams &params)
    {
        getKernelParams(params, kernelH, kernelW, padH, padW, strideH, strideW);

        numOutput = params.get<int>("num_output");
        bias = params.get<bool>("bias_term", true);
        group = params.get<int>("group", 1);
        CV_Assert(numOutput % group == 0);

        CV_Assert(params.learnedBlobs.size() >= 1 && (!bias || params.learnedBlobs.size() >= 2));
        learnedParams.assign(params.learnedBlobs.begin(), params.learnedBlobs.begin() + (bias ? 2 : 1));

        Blob &weightBlob = learnedParams[0];
        CV_Assert(weightBlob.cols() == kernelW && weightBlob.rows() == kernelH && weightBlob.num() == numOutput);

        if (bias)
        {
            Blob &biasBlob = learnedParams[1];
            CV_Assert(biasBlob.total() == numOutput);
        }
    }

    void ConvolutionLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() > 0);

        Blob &weightBlob = learnedParams[0];

        inCn = inputs[0]->channels();
        CV_Assert(inCn % group == 0 && numOutput % group == 0 && weightBlob.channels() == inCn/group);
        groupCnOut = numOutput / group;
        groupCn = inCn / group;

        inH = inputs[0]->rows();
        inW = inputs[0]->cols();
        computeOutputShape(inH, inW);

        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->rows() == inH && inputs[i]->cols() == inW && inputs[i]->channels() == inCn);
            int num = inputs[i]->num();

            outputs[i].create(num, numOutput, outH, outW);
        }

        kerSize = kernelH * kernelW * groupCn;
        srcColsMat.create(kerSize, outH * outW, CV_32F);

        if (bias)
        {
            biasOnesMat = Mat::ones(1, outH * outW, CV_32F);
        }
    }
    
    template <typename Dtype>
    void im2col_cpu(const Dtype* data_im, const int channels,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        Dtype* data_col)
    {
        int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        int channels_col = channels * kernel_h * kernel_w;
        for (int c = 0; c < channels_col; ++c) {
            int w_offset = c % kernel_w;
            int h_offset = (c / kernel_w) % kernel_h;
            int c_im = c / kernel_h / kernel_w;
            for (int h = 0; h < height_col; ++h) {
                for (int w = 0; w < width_col; ++w) {
                    int h_pad = h * stride_h - pad_h + h_offset;
                    int w_pad = w * stride_w - pad_w + w_offset;
                    if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                        data_col[(c * height_col + h) * width_col + w] =
                        data_im[(c_im * height + h_pad) * width + w_pad];
                    else
                        data_col[(c * height_col + h) * width_col + w] = 0;
                }
            }
        }
    }

    void ConvolutionLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() == outputs.size());

        float *srcColPtr = srcColsMat.ptr<float>();

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            Blob &input = *inputs[ii];
            Blob &output = outputs[ii];
            int num = input.num();

            for (int n = 0; n < num; n++)
            {
                for (int g = 0; g < group; g++)
                {
                    float *srcPtr = input.ptr<float>(n, g*groupCn);
                    im2col_cpu(srcPtr, groupCn, inH, inW, kernelH, kernelW, padH, padW, strideH, strideW, srcColPtr);

                    float *kerPtr = learnedParams[0].ptr<float>(g*groupCnOut);
                    float *dstPtr = output.ptr<float>(n, g*groupCnOut);

                    Mat kerMat(groupCnOut, kerSize, CV_32F, kerPtr);
                    Mat dstMat(groupCnOut, outH*outW, CV_32F, dstPtr);

                    cv::gemm(kerMat, srcColsMat, 1, noArray(), 0, dstMat);

                    if (bias)
                    {
                        float *biasPtr = learnedParams[1].ptr<float>() + g*groupCnOut;
                        Mat biasMat(groupCnOut, 1, CV_32F, biasPtr);
                        cv::gemm(biasMat, biasOnesMat, 1, dstMat, 1, dstMat);
                    }
                }
            }
        }
    }

    void ConvolutionLayer::computeOutputShape(int inH, int inW)
    {
        outH = (inH + 2 * padH - kernelH) / strideH + 1;
        outW = (inW + 2 * padW - kernelW) / strideW + 1;
    }
}
}