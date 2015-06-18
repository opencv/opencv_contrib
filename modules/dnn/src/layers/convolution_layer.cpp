#include "../precomp.hpp"
#include "layers_common.hpp"

namespace cv
{
namespace dnn
{
    //TODO: implement group parameter
    //TODO: simultaneously convolution and bias addition for cache optimization
    class ConvolutionLayer : public Layer
    {
        bool bias;
        int numOutput, group;
        int padH, padW;
        int strideH, strideW;
        int kernelH, kernelW;

        int inH, inW, inCn, colCn;
        int outH, outW;

        Mat imColsMat, biasOnesMat;

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
        CV_Assert(inCn % group == 0 && weightBlob.channels() == inCn);

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

        colCn = kernelH * kernelW * inCn;
        imColsMat.create(colCn, outH * outW, CV_32F);

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

        float *colPtr = imColsMat.ptr<float>();
        float *weigtsPtr = learnedParams[0].ptr<float>();
        float *biasPtr = (bias) ? learnedParams[1].ptr<float>() : NULL;

        CV_Assert(group == 1);

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            int num = inputs[ii]->num();

            for (int n = 0; n < num; n++)
            {
                float *srcImPtr = inputs[ii]->ptr<float>(n);
                float *dstImPtr = outputs[ii].ptr<float>(n);

                im2col_cpu(srcImPtr, inCn, inH, inW, kernelH, kernelW, padH, padW, strideH, strideW, colPtr);

                Mat weightsMat(numOutput, colCn, CV_32F, weigtsPtr);
                Mat dstIm(numOutput, outH*outW, CV_32F, dstImPtr);

                cv::gemm(weightsMat, imColsMat, 1, noArray(), 0, dstIm);

                if (bias)
                {
                    Mat biasMat(numOutput, 1, CV_32F, biasPtr);
                    cv::gemm(biasMat, biasOnesMat, 1, dstIm, 1, dstIm);
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