/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
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

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "op_im2col.hpp"
#include "op_blas.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <iostream>

namespace cv
{
namespace dnn
{

class BaseConvolutionLayerImpl : public ConvolutionLayer
{
public:
    BaseConvolutionLayerImpl()
    {
#ifdef HAVE_LAPACK
        int nthreads = cv::getThreadNum();
        if (getBlasThreads() != nthreads)
        {
            setBlasThreads(nthreads);
        }
#endif
    }
    void finalize(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        CV_Assert(inputs.size() > 0);

        CV_Assert(blobs.size() >= 1 && blobs.size() <= 2);
        CV_Assert(blobs[0].dims == 4 && blobs[0].size[3] == kernel.width && blobs[0].size[2] == kernel.height);

        const Mat &input = *inputs[0];
        CV_Assert(input.dims == 4 && (input.type() == CV_32F || input.type() == CV_64F));
        for (size_t i = 0; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->type() == input.type());
            CV_Assert(inputs[i]->dims == 4 && inputs[i]->size[1] == input.size[1]);
            CV_Assert(inputs[i]->size[2] == input.size[2] && inputs[i]->size[3] == input.size[3]);
        }

        Size outSize = Size(outputs[0].size[3], outputs[0].size[2]);
        getConvPoolPaddings(Size(input.size[3], input.size[2]), outSize,
                kernel, stride, padMode, pad);
    }

    bool hasBias() const
    {
        return blobs.size() >= 2;
    }

    virtual MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const = 0;
    bool is1x1() const
    {
        return (kernel.height == 1 && kernel.width == 1) &&
        (stride.height == 1 && stride.width == 1) &&
        (dilation.height == 1 && dilation.width == 1);
    }
    bool setActivation(const Ptr<ActivationLayer>& ) { return false; }
};

//TODO: simultaneously convolution and bias addition for cache optimization
class ConvolutionLayerImpl : public BaseConvolutionLayerImpl
{
public:
    enum { VEC_ALIGN = 8 };
    Mat weightsMat;
    Ptr<ActivationLayer> activ;

    MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const
    {
        Size out(outShape[3], outShape[2]);
        int inpGroupCn = blobs[0].size[1];
        int ksize = inpGroupCn * kernel.height * kernel.width;
        return shape(out.area(), ksize);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(blobs.size() != 0);
        CV_Assert(!hasBias() || blobs[1].total() == (size_t)blobs[0].size[0]);
        CV_Assert(inputs.size() == (size_t)1);

        internals.clear();

        int inpCn = inputs[0][1];
        int inpH = inputs[0][2];
        int inpW = inputs[0][3];

        int outCn = blobs[0].size[0];
        Size out;

        if (padMode.empty())
        {
            out.height = (inpH + 2 * pad.height - (dilation.height * (kernel.height - 1) + 1)) / stride.height + 1;
            out.width = (inpW + 2 * pad.width - (dilation.width * (kernel.width - 1) + 1)) / stride.width + 1;
        }
        else
        {
            getConvPoolOutParams(Size(inpH, inpW), kernel, stride, padMode, out);
        }

        int ngroups = inpCn / blobs[0].size[1];
        CV_Assert(inpCn % ngroups == 0 && outCn % ngroups == 0);

        int dims[] = {inputs[0][0], outCn, out.height, out.width};
        outputs.resize(inputs.size(), shape(dims));

        return false;
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) { activ = layer; return true; }

    class ParallelConv : public cv::ParallelLoopBody
    {
    public:
        enum { BLK_SIZE = 32, BLK_SIZE_CN = 64 };

        const Mat* input_;
        const Mat* weights_;
        Mat* output_;
        int outShape[4];
        Size kernel_, pad_, stride_, dilation_;
        int ngroups_, nstripes_;
        std::vector<int> ofstab_;
        std::vector<float> biasvec_;
        const ActivationLayer* activ_;
        bool is1x1_;
        bool useAVX2;

        ParallelConv() {}

        static void run( const Mat& input, Mat& output,
                         const Mat& weights, const Mat& bias,
                         Size kernel, Size pad, Size stride, Size dilation,
                         int ngroups, int nstripes, const ActivationLayer* activ )
        {
            CV_Assert( input.dims == 4 && output.dims == 4 &&
                       input.size[0] == output.size[0] &&
                       weights.rows == output.size[1] &&
                       weights.cols == (input.size[1]/ngroups)*kernel.width*kernel.height &&
                       input.type() == output.type() &&
                       input.type() == weights.type() &&
                       input.type() == CV_32F &&
                       input.isContinuous() &&
                       output.isContinuous() &&
                       (bias.empty() || (bias.isContinuous() && bias.type() == CV_32F &&
                                         bias.total() == (size_t)output.size[1])));
            ParallelConv p;

            p.input_ = &input;
            p.weights_ = &weights;
            p.output_ = &output;
            for( int i = 0; i < 4; i++ ) p.outShape[i] = output.size[i];
            p.outShape[1] /= ngroups;
            p.kernel_ = kernel; p.pad_ = pad; p.stride_ = stride; p.dilation_ = dilation;
            p.ngroups_ = ngroups;
            p.nstripes_ = nstripes;
            p.activ_ = activ;
            int inpCnAll = input.size[1], width = input.size[3], height = input.size[2];
            int inpCn = inpCnAll / ngroups;
            int k, outCn = output.size[1];
            p.is1x1_ = kernel == Size(0,0) && pad == Size(0, 0);
            p.useAVX2 = checkHardwareSupport(CPU_AVX2);

            int ncn = std::min(inpCn, (int)BLK_SIZE_CN);
            p.ofstab_.resize(kernel.width*kernel.height*ncn);
            int* ofstab = &p.ofstab_[0];

            for( k = 0; k < ncn; k++ )
                for( int k_r = 0; k_r < kernel.height; k_r++ )
                    for( int k_c = 0; k_c < kernel.width; k_c++ )
                        ofstab[(k*kernel.height + k_r)*kernel.width + k_c] =
                        (k*height + k_r*dilation.height)*width + k_c*dilation.width;

            p.biasvec_.resize(outCn+2);
            float* biasvec = &p.biasvec_[0];
            if( bias.empty() )
            {
                for( k = 0; k < outCn; k++ )
                    biasvec[k] = 0.f;
            }
            else
            {
                for( k = 0; k < outCn; k++ )
                    biasvec[k] = bias.at<float>(k);
            }
            biasvec[outCn] = biasvec[outCn+1] = biasvec[outCn-1];
            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        virtual void operator ()(const Range &r0) const
        {
            const int valign = ConvolutionLayerImpl::VEC_ALIGN;
            int ngroups = ngroups_, batchSize = input_->size[0]*ngroups;
            int outW = output_->size[3], outH = output_->size[2], outCn = output_->size[1]/ngroups;
            int width = input_->size[3], height = input_->size[2], inpCn = input_->size[1]/ngroups;
            int nstripes = nstripes_;
            int kernel_w = kernel_.width, kernel_h = kernel_.height;
            int pad_w = pad_.width, pad_h = pad_.height;
            int stride_w = stride_.width, stride_h = stride_.height;
            int dilation_w = dilation_.width, dilation_h = dilation_.height;
            int karea = kernel_w*kernel_h;
            int i, j, k;
            size_t inpPlaneSize = width*height;
            size_t outPlaneSize = outW*outH;
            bool is1x1 = is1x1_;

            int stripesPerSample;
            size_t stripeSize;
            Range r = r0;

            if( nstripes >= batchSize*2 )
            {
                stripesPerSample = nstripes/batchSize;
                stripeSize = alignSize((outPlaneSize + stripesPerSample - 1)/stripesPerSample, valign);
                stripeSize = std::min(stripeSize, outPlaneSize);
            }
            else
            {
                stripesPerSample = 1;
                int samplesPerStripe = std::max((batchSize + nstripes - 1)/nstripes, 1);
                r.start *= samplesPerStripe;
                r.end *= samplesPerStripe;
                nstripes *= samplesPerStripe;
                stripeSize = outPlaneSize;
            }

            const float* data_inp0_ = input_->ptr<float>();
            const int* ofstab = &ofstab_[0];
            const float* wptr_orig_ = weights_->ptr<float>();
            size_t wstep = weights_->step1();
            const float* biasvec = &biasvec_[0];
            float* data_out0_ = output_->ptr<float>();
            size_t rowbufsz = (size_t)karea*BLK_SIZE_CN*BLK_SIZE;
            const int valignBytes = (int)(valign*sizeof(float));
            AutoBuffer<float> rowbuf0_(rowbufsz + valignBytes);
            float* rowbuf0 = alignPtr((float*)rowbuf0_, valignBytes);

            // we clear the buffer once; ultimately, it lets us to avoid
            // tail processing after running the unrolled/vectorized loop.
            // the main idea is to make sure that the tail (a.k.a. padding) of each row
            // (i.e. the elements with indices between vsz=karea*ncn and vsz_a)
            // does not contain NaNs or Infs. Because the padding in the weights
            // matrix is explicitly initialized with 0's, we handle all other
            // cases nicely, i.e. we can skip expliciting re-initialization
            // of the padding - we just retain elements from the previous iteration
            // of the loop over channels (cn0).
            memset(rowbuf0, 0, rowbufsz*sizeof(rowbuf0[0]) );

            for( int stripe = r.start; stripe < r.end; stripe++ )
            {
                int subsampleIdx = stripe/stripesPerSample;
                if( subsampleIdx >= batchSize )
                    break;
                int stripeStart = (int)((stripe - subsampleIdx*stripesPerSample)*stripeSize);
                int stripeEnd = (int)std::min(stripeStart + stripeSize, outPlaneSize);
                const float* data_inp0 = data_inp0_ + subsampleIdx*inpPlaneSize*inpCn;
                float* data_out0 = data_out0_ + subsampleIdx*outPlaneSize*outCn;
                int startOutCn = (subsampleIdx % ngroups)*outCn;
                const float* wptr_orig = wptr_orig_ + wstep*startOutCn;
                const float* biasptr = biasvec + startOutCn;

                for( int cn0 = 0; cn0 < inpCn; cn0 += BLK_SIZE_CN )
                {
                    int cn1 = std::min(cn0 + BLK_SIZE_CN, inpCn);
                    int ncn = cn1 - cn0, vsz = karea*ncn;
                    int vsz_a = (int)alignSize(vsz, valign);
                    const float* wptr = wptr_orig + cn0*karea;

                    for( int ofs0 = stripeStart; ofs0 < stripeEnd; ofs0 += BLK_SIZE )
                    {
                        int ofs, ofs1 = std::min(ofs0 + BLK_SIZE, stripeEnd);

                        // do im2row for a part of input tensor
                        if( is1x1 )
                        {
                            for( ofs = ofs0; ofs < ofs1; ofs++ )
                            {
                                int out_i = ofs / outW;
                                int out_j = ofs - out_i * outW;
                                float* rowbuf = rowbuf0 + (ofs - ofs0)*vsz_a;

                                int in_i = out_i * stride_h - pad_h;
                                int in_j = out_j * stride_w - pad_w;
                                const float* imgptr = data_inp0 + (cn0*height + in_i)*width + in_j;

                                for( k = 0; k < vsz; k++ )
                                    rowbuf[k] = imgptr[k*inpPlaneSize];
                            }
                        }
                        else
                        {
                            for( ofs = ofs0; ofs < ofs1; ofs++ )
                            {
                                int out_i = ofs / outW;
                                int out_j = ofs - out_i * outW;
                                float* rowbuf = rowbuf0 + (ofs - ofs0)*vsz_a;

                                int in_i = out_i * stride_h - pad_h;
                                int in_j = out_j * stride_w - pad_w;
                                const float* imgptr = data_inp0 + (cn0*height + in_i)*width + in_j;

                                // this condition should be true for most of the tensor elements, i.e.
                                // most of the time the kernel aperture is inside the tensor X-Y plane.
                                if( 0 <= in_i && in_i < height - (kernel_h-1)*dilation_h &&
                                    0 <= in_j && in_j < width - (kernel_w-1)*dilation_w )
                                {
                                    for( k = 0; k < vsz; k++ )
                                        rowbuf[k] = imgptr[ofstab[k]];
                                }
                                else
                                {
                                    int i0 = std::max(0, (-in_i + dilation_h-1)/dilation_h);
                                    int i1 = std::min(kernel_h, (height - in_i + dilation_h-1)/dilation_h);
                                    int j0 = std::max(0, (-in_j + dilation_w-1)/dilation_w);
                                    int j1 = std::min(kernel_w, (width - in_j + dilation_w-1)/dilation_w);

                                    // here some non-continous sub-row of the row will not be
                                    // filled from the tensor; we need to make sure that the uncovered
                                    // elements are explicitly set to 0's. the easiest way is to
                                    // set all the elements to 0's before the loop.
                                    memset(rowbuf, 0, vsz*sizeof(rowbuf[0]));
                                    for( k = 0; k < ncn; k++, imgptr += width*height )
                                    {
                                        for( i = i0; i < i1; i++ )
                                        {
                                            for( j = j0; j < j1; j++ )
                                            {
                                                int imgofs = i*(dilation_h*width) + j*dilation_w;
                                                rowbuf[(k*kernel_h + i)*kernel_w + j] = imgptr[imgofs];
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // now compute dot product of the weights
                        // and im2row-transformed part of the tensor
                        int bsz = ofs1 - ofs0;
                    #if CV_DNN_TRY_AVX2
                        if(useAVX2)
                            fastConv_avx2(wptr, wstep, biasptr, rowbuf0, data_out0 + ofs0, outShape, bsz, vsz, vsz_a, cn0 == 0);
                        else
                    #endif
                        for( int i = 0; i < outCn; i += 2 )
                        {
                            const float* wptr0 = wptr + i*wstep;
                            const float* wptr1 = wptr0 + wstep;
                            float* outptr0 = data_out0 + ofs0 + i*outPlaneSize;
                            float* outptr1 = outptr0 + outPlaneSize;
                            float bias0 = biasptr[i], bias1 = biasptr[i+1];

                            if( i+1 >= outCn )
                            {
                                wptr1 = wptr0;
                                outptr1 = outptr0;
                                bias1 = bias0;
                            }

                            int j = 0;
                        #if CV_SIMD128
                            for( ; j <= bsz - 4; j += 4 )
                            {
                                const float* rptr = rowbuf0 + j*vsz_a;
                                v_float32x4 s0, s1;

                                if( cn0 == 0 )
                                {
                                    s0 = v_setall_f32(bias0);
                                    s1 = v_setall_f32(bias1);
                                }
                                else
                                {
                                    s0 = v_load(outptr0 + j);
                                    s1 = v_load(outptr1 + j);
                                }

                                v_float32x4 vs00 = v_setzero_f32(), vs01 = v_setzero_f32(),
                                            vs02 = v_setzero_f32(), vs03 = v_setzero_f32(),
                                            vs10 = v_setzero_f32(), vs11 = v_setzero_f32(),
                                            vs12 = v_setzero_f32(), vs13 = v_setzero_f32();
                                for( k = 0; k < vsz; k += 4, rptr += 4 )
                                {
                                    v_float32x4 w0 = v_load_aligned(wptr0 + k), w1 = v_load_aligned(wptr1 + k);
                                    v_float32x4 r0 = v_load_aligned(rptr), r1 = v_load_aligned(rptr + vsz_a),
                                                r2 = v_load_aligned(rptr + vsz_a*2), r3 = v_load_aligned(rptr + vsz_a*3);

                                    vs00 += w0*r0;
                                    vs01 += w0*r1;
                                    vs02 += w0*r2;
                                    vs03 += w0*r3;

                                    vs10 += w1*r0;
                                    vs11 += w1*r1;
                                    vs12 += w1*r2;
                                    vs13 += w1*r3;
                                }
                                s0 += v_reduce_sum4(vs00, vs01, vs02, vs03);
                                s1 += v_reduce_sum4(vs10, vs11, vs12, vs13);

                                v_store(outptr0 + j, s0);
                                v_store(outptr1 + j, s1);
                            }
                        #endif
                            for( ; j < bsz; j++ )
                            {
                                const float* rptr = rowbuf0 + j*vsz_a;
                                float s00, s10;

                                if( cn0 == 0 )
                                {
                                    s00 = bias0;
                                    s10 = bias1;
                                }
                                else
                                {
                                    s00 = outptr0[j];
                                    s10 = outptr1[j];
                                }

                                for( k = 0; k < vsz; k++ )
                                {
                                    float r0 = rptr[k];
                                    s00 += wptr0[k]*r0;
                                    s10 += wptr1[k]*r0;
                                }

                                outptr0[j] = s00;
                                outptr1[j] = s10;
                            }
                        }
                    }
                }

                if( activ_ )
                    activ_->forwardSlice(data_out0 + stripeStart, data_out0 + stripeStart,
                                         (int)(stripeEnd - stripeStart),
                                         outPlaneSize, startOutCn, startOutCn + outCn);
            }
        }
    };

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_Assert(inputs.size() == (size_t)1 && inputs[0]->size[1] % blobs[0].size[1] == 0);
        int ngroups = inputs[0]->size[1]/blobs[0].size[1];
        CV_Assert(outputs[0].size[1] % ngroups == 0);

        int outCn = blobs[0].size[0];

        if( weightsMat.empty() )
        {
            Mat wm = blobs[0].reshape(1, outCn);
            if( wm.step1() % VEC_ALIGN != 0 )
            {
                int newcols = (int)alignSize(wm.step1(), VEC_ALIGN);
                Mat wm_buffer = Mat(outCn, newcols, wm.type());
                Mat wm_padding = wm_buffer.colRange(wm.cols, newcols);
                wm_padding.setTo(Scalar::all(0.));
                Mat wm_aligned = wm_buffer.colRange(0, wm.cols);
                wm.copyTo(wm_aligned);
                wm = wm_aligned;
            }
            weightsMat = wm;
        }
        Mat biasesMat = hasBias() ? blobs[1].reshape(1, outCn) : Mat();

        int nstripes = std::max(getNumThreads(), 1);
        ParallelConv::run(*inputs[0], outputs[0], weightsMat, biasesMat,
                          kernel, pad, stride, dilation, ngroups, nstripes, activ.get());
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        CV_Assert(inputs.size() == outputs.size());

        int64 flops = 0;
        for (int i = 0; i < inputs.size(); i++)
        {
            flops += total(outputs[i])*(2*kernel.area()*inputs[i][1] + 1);
        }

        return flops;
    }
};

class DeConvolutionLayerImpl : public BaseConvolutionLayerImpl
{
public:
    MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const
    {
        int inpCn = inpShape[1];
        int inpH = inpShape[2];
        int inpW = inpShape[3];
        int outCn = outShape[1];
        int ngroups = inpCn / blobs[0].size[1];
        int outGroupCn = outCn / ngroups;
        int ksize = outGroupCn * kernel.height * kernel.width;
        return shape(ksize, inpH * inpW);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(!hasBias() || blobs[1].total() == (size_t)blobs[0].size[0]);
        CV_Assert(inputs.size() != 0);

        int inpCn = inputs[0][1];
        int inpH = inputs[0][2];
        int inpW = inputs[0][3];

        int outH = stride.height * (inpH - 1) + kernel.height - 2 * pad.height + adjustPad.height;
        int outW = stride.width * (inpW - 1) + kernel.width - 2 * pad.width + adjustPad.width;
        int outCn = blobs[0].size[0];

        int ngroups = inpCn / blobs[0].size[1];

        CV_Assert(inpCn % ngroups == 0 && outCn % ngroups == 0);
        CV_Assert(blobs[0].size[0] == outCn && blobs[0].size[1] == inpCn / ngroups);

        int dims[] = {inputs[0][0], outCn, outH, outW};
        outputs.resize(inputs.size(), shape(dims));

        internals.push_back(MatShape());
        if (!is1x1())
            internals[0] = computeColRowShape(inputs[0], outputs[0]);

        if (hasBias())
            internals.push_back(shape(1, outH*outW));

        return false;
    }


    void forward(std::vector<Mat *> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        internals[0].setTo(0);
        if (hasBias())
            internals[1].setTo(1);

        int outCn = blobs[0].size[0];
        int inpCn = inputs[0]->size[1];
        Mat weightsMat = blobs[0].reshape(1, inpCn);
        Mat biasesMat  = hasBias() ? blobs[1].reshape(1, outCn) : Mat();

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            int ngroups = inpCn / blobs[0].size[1];
            int inpGroupCn = blobs[0].size[1];
            int outGroupCn = outCn / ngroups;
            int numImg = inputs[ii]->size[0];

            Mat convBlob = inputs[ii]->reshape(1, numImg*inpCn);
            Mat decnBlob = outputs[ii].reshape(1, numImg*outCn);

            for (int n = 0; n < numImg; n++)
            {
                for (int g = 0; g < ngroups; g++)
                {
                    Mat dstMat = decnBlob.rowRange(_Range((g + n * ngroups) * outGroupCn, outGroupCn));
                    Mat &colMat = (is1x1()) ? dstMat : internals[0];

                    Mat convMat = convBlob.rowRange(_Range((g + n * ngroups) * inpGroupCn, inpGroupCn));
                    Mat wghtMat = weightsMat.rowRange(_Range(g * inpGroupCn, inpGroupCn));

                    dnn::gemm(wghtMat, convMat, 1, colMat, 0, GEMM_1_T);

                    if (!is1x1())
                        col2im(colMat, dstMat, shape(*inputs[ii]), shape(outputs[ii]));

                    if (hasBias())
                    {
                        Mat curBiasMat = biasesMat.rowRange(_Range(g * outGroupCn, outGroupCn));
                        dnn::gemm(curBiasMat, internals[1], 1, dstMat, 1);
                    }
                }
            }
        }
    }

    void col2im(const Mat &colMat, Mat &dstImg, const MatShape& inShape, const MatShape& outShape)
    {
        int outCn = outShape[1], outH = outShape[2], outW = outShape[3];
        int inpCn = inShape[1];
        int ngroups = inpCn / blobs[0].size[1];
        int outGroupCn = outCn / ngroups;

        if (is1x1())
        {
            dstImg = colMat;
            return;
        }
        cv::dnn::col2im(colMat.ptr<float>(), outGroupCn, outH, outW, kernel.height, kernel.width,
                        pad.height, pad.width, stride.height, stride.width,
                        dilation.height, dilation.width, dstImg.ptr<float>(), &ofsbuf[0]);
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        CV_Assert(inputs.size() == outputs.size());

        float flops = 0;
        int outChannels = blobs[0].size[0];

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += 2*outChannels*kernel.area()*total(inputs[i]);
        }

        return flops;
    }

    std::vector<int> ofsbuf;
};

//Convolution and Deconvolution
static void initConvDeconvLayerFromCaffe(Ptr<BaseConvolutionLayer> l, const LayerParams &params)
{
    l->setParamsFrom(params);
    getConvolutionKernelParams(params, l->kernel.height, l->kernel.width, l->pad.height,
                               l->pad.width, l->stride.height, l->stride.width, l->dilation.height,
                               l->dilation.width, l->padMode);

    bool bias = params.get<bool>("bias_term", true);
    int numOutput = params.get<int>("num_output");
    int ngroups = params.get<int>("group", 1);

    l->adjustPad.height = params.get<int>("adj_h", 0);
    l->adjustPad.width = params.get<int>("adj_w", 0);

    CV_Assert(numOutput % ngroups == 0);
    CV_Assert((bias && l->blobs.size() == 2) || (!bias && l->blobs.size() == 1));
}

Ptr<BaseConvolutionLayer> ConvolutionLayer::create(const LayerParams &params)
{
    Ptr<BaseConvolutionLayer> l(new ConvolutionLayerImpl);
    initConvDeconvLayerFromCaffe(l, params);
    return l;
}

Ptr<BaseConvolutionLayer> DeconvolutionLayer::create(const LayerParams &params)
{
    Ptr<BaseConvolutionLayer> l(new DeConvolutionLayerImpl);
    initConvDeconvLayerFromCaffe(l, params);

    return l;
}

}
}
