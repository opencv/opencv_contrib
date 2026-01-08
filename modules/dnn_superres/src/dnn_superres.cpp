// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/dnn_superres.hpp"

namespace cv
{
namespace dnn_superres
{

/** @brief Class for importing DepthToSpace layer from the ESPCN model
*/
class DepthToSpace CV_FINAL : public cv::dnn::Layer
{
public:
    DepthToSpace(const cv::dnn::LayerParams &params);

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params);

    virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
                                 const int,
                                 std::vector<std::vector<int> > &outputs,
                                 std::vector<std::vector<int> > &) const CV_OVERRIDE;

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
                         cv::OutputArrayOfArrays outputs_arr,
                         cv::OutputArrayOfArrays) CV_OVERRIDE;

    /// Register this layer
    static void registerLayer()
    {
        static bool initialized = false;
        if (!initialized)
        {
            //Register custom layer that implements pixel shuffling
            std::string name = "DepthToSpace";
            dnn::LayerParams layerParams = dnn::LayerParams();
            cv::dnn::LayerFactory::registerLayer("DepthToSpace", DepthToSpace::create);
            initialized = true;
        }
    }
};

Ptr<DnnSuperResImpl> DnnSuperResImpl::create(){
    return Ptr<DnnSuperResImpl>(new DnnSuperResImpl());
}

DnnSuperResImpl::DnnSuperResImpl()
{
    DepthToSpace::registerLayer();
}

DnnSuperResImpl::DnnSuperResImpl(const String& algo, int scale)
    : alg(algo), sc(scale)
{
    DepthToSpace::registerLayer();
}

void DnnSuperResImpl::readModel(const String& path)
{
    if ( path.size() )
    {
        this->net = dnn::readNetFromTensorflow(path);
        CV_LOG_INFO(NULL, "Successfully loaded model: " << path);
    }
    else
    {
        CV_Error(Error::StsBadArg, String("Could not load model: ") + path);
    }
}

void DnnSuperResImpl::readModel(const String& weights, const String& definition)
{
    if ( weights.size() && definition.size() )
    {
        this->net = dnn::readNetFromTensorflow(weights, definition);
        CV_LOG_INFO(NULL, "Successfully loaded model: " << weights << " " << definition);
    }
    else
    {
        CV_Error(Error::StsBadArg, String("Could not load model: ") + weights + " " + definition);
    }
}

void DnnSuperResImpl::setModel(const String& algo, int scale)
{
    this->sc = scale;
    this->alg = algo;
}

void DnnSuperResImpl::setPreferableBackend(int backendId)
{
    if (net.empty())
        CV_Error(Error::StsError, "Model is emtpy. Please read a model before setting the backend.");

    net.setPreferableBackend(backendId);
    CV_LOG_INFO(NULL, "Successfully set computation backend.");
}

void DnnSuperResImpl::setPreferableTarget(int targetId)
{
    if (net.empty())
        CV_Error(Error::StsError, "Model is empty. Please read a model before setting the target.");

    net.setPreferableTarget(targetId);
    CV_LOG_INFO(NULL, "Successfully set target device.");
}

void DnnSuperResImpl::upsample(InputArray img, OutputArray result)
{
    if (net.empty())
        CV_Error(Error::StsError, "Model not specified. Please set model via setModel().");

    if (this->alg == "espcn" || this->alg == "lapsrn" || this->alg == "fsrcnn")
    {
        //Preprocess the image: convert to YCrCb float image and normalize
        Mat preproc_img;
        preprocess_YCrCb(img, preproc_img);

        //Split the image: only the Y channel is used for inference
        Mat ycbcr_channels[3];
        split(preproc_img, ycbcr_channels);

        Mat Y = ycbcr_channels[0];

        //Create blob from image so it has size 1,1,Width,Height
        cv::Mat blob;
        dnn::blobFromImage(Y, blob, 1.0);

        //Get the HR output
        this->net.setInput(blob);

        Mat blob_output = this->net.forward();

        //Convert from blob
        std::vector <Mat> model_outs;
        dnn::imagesFromBlob(blob_output, model_outs);
        Mat out_img = model_outs[0];

        //Reconstruct: upscale the Cr and Cb space and merge the three layer
        reconstruct_YCrCb(out_img, preproc_img, result, this->sc);
    }
    else if (this->alg == "edsr")
    {
        //BGR mean of the Div2K dataset
        Scalar mean = Scalar(103.1545782, 111.561547, 114.35629928);

        //Convert to float
        Mat float_img;
        img.getMat().convertTo(float_img, CV_32F, 1.0);

        //Create blob from image so it has size [1,3,Width,Height] and subtract dataset mean
        cv::Mat blob;
        dnn::blobFromImage(float_img, blob, 1.0, Size(), mean);

        //Get the HR output
        this->net.setInput(blob);
        Mat blob_output = this->net.forward();

        //Convert from blob
        std::vector <Mat> model_outs;
        dnn::imagesFromBlob(blob_output, model_outs);

        //Post-process: add mean.
        Mat(model_outs[0] + mean).convertTo(result, CV_8U);
    }
    else
    {
        CV_Error(cv::Error::StsNotImplemented, String("Unknown/unsupported superres algorithm: ") + this->alg);
    }
}

void DnnSuperResImpl::upsampleMultioutput(InputArray img, std::vector<Mat> &imgs_new, const std::vector<int>& scale_factors, const std::vector<String>& node_names)
{
    CV_Assert(!img.empty());
    CV_Assert(scale_factors.size() == node_names.size());
    CV_Assert(!scale_factors.empty());
    CV_Assert(!node_names.empty());

    if ( this->alg != "lapsrn" )
    {
        CV_Error(cv::Error::StsBadArg, "Only LapSRN support multiscale upsampling for now.");
        return;
    }

    if (net.empty())
        CV_Error(Error::StsError, "Model not specified. Please set model via setModel().");

    if (this->alg == "lapsrn")
    {
        Mat orig = img.getMat();

        //Preprocess the image: convert to YCrCb float image and normalize
        Mat preproc_img;
        preprocess_YCrCb(orig, preproc_img);

        //Split the image: only the Y channel is used for inference
        Mat ycbcr_channels[3];
        split(preproc_img, ycbcr_channels);

        Mat Y = ycbcr_channels[0];

        //Create blob from image so it has size 1,1,Width,Height
        cv::Mat blob;
        dnn::blobFromImage(Y, blob, 1.0);

        //Get the HR outputs
        std::vector <Mat> outputs_blobs;
        this->net.setInput(blob);
        this->net.forward(outputs_blobs, node_names);

        for(unsigned int i = 0; i < scale_factors.size(); i++)
        {
            std::vector <Mat> model_outs;
            dnn::imagesFromBlob(outputs_blobs[i], model_outs);
            Mat out_img = model_outs[0];
            Mat reconstructed;

            reconstruct_YCrCb(out_img, preproc_img, reconstructed, scale_factors[i]);

            imgs_new.push_back(reconstructed);
        }
    }
}

int DnnSuperResImpl::getScale()
{
    return this->sc;
}

String DnnSuperResImpl::getAlgorithm()
{
    return this->alg;
}

void DnnSuperResImpl::preprocess_YCrCb(InputArray inpImg, OutputArray outImg)
{
    if ( inpImg.type() == CV_8UC1 )
    {
        inpImg.getMat().convertTo(outImg, CV_32F, 1.0 / 255.0);
    }
    else if ( inpImg.type() == CV_32FC1 )
    {
        inpImg.getMat().convertTo(outImg, CV_32F, 1.0 / 255.0);
    }
    else if ( inpImg.type() == CV_32FC3 )
    {
        Mat img_float;
        inpImg.getMat().convertTo(img_float, CV_32F, 1.0 / 255.0);
        cvtColor(img_float, outImg, COLOR_BGR2YCrCb);
    }
    else if ( inpImg.type() == CV_8UC3 )
    {
        Mat ycrcb;
        cvtColor(inpImg, ycrcb, COLOR_BGR2YCrCb);
        ycrcb.convertTo(outImg, CV_32F, 1.0 / 255.0);
    }
    else
    {
        CV_Error(Error::StsBadArg, String("Not supported image type: ") + typeToString(inpImg.type()));
    }
}

void DnnSuperResImpl::reconstruct_YCrCb(InputArray inpImg, InputArray origImg, OutputArray outImg, int scale)
{
    if ( origImg.type() == CV_32FC3 )
    {
        Mat orig_channels[3];
        split(origImg.getMat(), orig_channels);

        Mat Cr, Cb;
        cv::resize(orig_channels[1], Cr, cv::Size(), scale, scale);
        cv::resize(orig_channels[2], Cb, cv::Size(), scale, scale);

        std::vector <Mat> channels;
        channels.push_back(inpImg.getMat());
        channels.push_back(Cr);
        channels.push_back(Cb);

        Mat merged_img;
        merge(channels, merged_img);

        Mat merged_8u_img;
        merged_img.convertTo(merged_8u_img, CV_8U, 255.0);

        cvtColor(merged_8u_img, outImg, COLOR_YCrCb2BGR);
    }
    else if ( origImg.type() == CV_32FC1 )
    {
        inpImg.getMat().convertTo(outImg, CV_8U, 255.0);
    }
    else
    {
        CV_Error(Error::StsBadArg, String("Not supported image type: ") + typeToString(origImg.type()));
    }
}


DepthToSpace::DepthToSpace(const cv::dnn::LayerParams &params) : Layer(params)
{
}

cv::Ptr<cv::dnn::Layer> DepthToSpace::create(cv::dnn::LayerParams &params)
{
    return cv::Ptr<cv::dnn::Layer>(new DepthToSpace(params));
}

bool DepthToSpace::getMemoryShapes(const std::vector <std::vector<int>> &inputs,
        const int, std::vector <std::vector<int>> &outputs, std::vector <std::vector<int>> &) const
{
    std::vector<int> outShape(4);

    int scale;
    if( inputs[0][1] == 4 || inputs[0][1] == 9 || inputs[0][1] == 16 ) //Only one image channel
    {
        scale = static_cast<int>(sqrt(inputs[0][1]));
    }
    else // Three image channels
    {
        scale = static_cast<int>(sqrt(inputs[0][1]/3));
    }

    outShape[0] = inputs[0][0];
    outShape[1] = static_cast<int>(inputs[0][1] / pow(scale,2));
    outShape[2] = static_cast<int>(scale * inputs[0][2]);
    outShape[3] = static_cast<int>(scale * inputs[0][3]);

    outputs.assign(4, outShape);

    return false;
}

void DepthToSpace::forward(cv::InputArrayOfArrays inputs_arr, cv::OutputArrayOfArrays outputs_arr,
    cv::OutputArrayOfArrays)
{
    std::vector <cv::Mat> inputs, outputs;
    inputs_arr.getMatVector(inputs);
    outputs_arr.getMatVector(outputs);
    cv::Mat &inp = inputs[0];
    cv::Mat &out = outputs[0];
    const float *inpData = (float *) inp.data;
    float *outData = (float *) out.data;

    const int inpHeight = inp.size[2];
    const int inpWidth = inp.size[3];

    const int numChannels = out.size[1];
    const int outHeight = out.size[2];
    const int outWidth = out.size[3];

    int scale = int(outHeight / inpHeight);
    int count = 0;

    for (int ch = 0; ch < numChannels; ch++)
    {
        for (int y = 0; y < outHeight; y++)
        {
            for (int x = 0; x < outWidth; x++)
            {
                int x_coord = static_cast<int>(floor((y / scale)));
                int y_coord = static_cast<int>(floor((x / scale)));
                int c_coord = numChannels * scale * (y % scale) + numChannels * (x % scale) + ch;

                int index = (((c_coord * inpHeight) + x_coord) * inpWidth) + y_coord;

                outData[count++] = inpData[index];
            }
        }
    }
}

}} // cv::dnn_superres::
