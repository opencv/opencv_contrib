#include "precomp.hpp"
using namespace caffe;

namespace cv
{
namespace cnn_3dobj
{
    descriptorExtractor::descriptorExtractor(const String& device_type, int device_id)
    {
        net_ready = 0;
        if (strcmp(device_type.c_str(), "CPU") == 0 || strcmp(device_type.c_str(), "GPU") == 0)
        {
            if (strcmp(device_type.c_str(), "CPU") == 0)
            {
                caffe::Caffe::set_mode(caffe::Caffe::CPU);
                deviceType = "CPU";
                std::cout << "Using CPU" << std::endl;
            }
            else
            {
                caffe::Caffe::set_mode(caffe::Caffe::GPU);
                caffe::Caffe::SetDevice(device_id);
                deviceType = "GPU";
                std::cout << "Using GPU" << std::endl;
                std::cout << "Using Device_id=" << device_id << std::endl;
            }
            net_set = true;
        }
        else
        {
            std::cout << "Error: Device name must be 'GPU' together with an device number or 'CPU'." << std::endl;
            net_set = false;
        }
    };

    String descriptorExtractor::getDeviceType()
    {
        String device_info_out;
        device_info_out = deviceType;
        return device_info_out;
    };

    int descriptorExtractor::getDeviceId()
    {
        int device_info_out;
        device_info_out = deviceId;
        return device_info_out;
    };

    void descriptorExtractor::setDeviceType(const String& device_type)
    {
        if (strcmp(device_type.c_str(), "CPU") == 0 || strcmp(device_type.c_str(), "GPU") == 0)
        {
            if (strcmp(device_type.c_str(), "CPU") == 0)
            {
                caffe::Caffe::set_mode(caffe::Caffe::CPU);
                deviceType = "CPU";
                std::cout << "Using CPU" << std::endl;
            }
            else
            {
                caffe::Caffe::set_mode(caffe::Caffe::GPU);
                deviceType = "GPU";
                std::cout << "Using GPU" << std::endl;
            }
        }
        else
        {
            std::cout << "Error: Device name must be 'GPU' or 'CPU'." << std::endl;
        }
    };

    void descriptorExtractor::setDeviceId(const int& device_id)
    {
        if (strcmp(deviceType.c_str(), "GPU") == 0)
        {
            caffe::Caffe::SetDevice(device_id);
            deviceId = device_id;
            std::cout << "Using GPU with Device ID = " << device_id << std::endl;
        }
        else
        {
            std::cout << "Error: Device ID only need to be set when GPU is used." << std::endl;
        }
    };

    void descriptorExtractor::loadNet(const String& model_file, const String& trained_file, const String& mean_file)
    {
        if (net_set)
        {
            /* Load the network. */
            convnet = new Net<float>(model_file, TEST);
            convnet->CopyTrainedLayersFrom(trained_file);
            if (convnet->num_inputs() != 1)
                std::cout << "Network should have exactly one input." << std::endl;
            if (convnet->num_outputs() != 1)
                std::cout << "Network should have exactly one output." << std::endl;
            Blob<float>* input_layer = convnet->input_blobs()[0];
            num_channels = input_layer->channels();
            if (num_channels != 3 && num_channels != 1)
                std::cout << "Input layer should have 1 or 3 channels." << std::endl;
            input_geometry = cv::Size(input_layer->width(), input_layer->height());
            /* Load the binaryproto mean file. */
            if (!mean_file.empty())
            {
                setMean(mean_file);
                net_ready = 2;
            }
            else
            {
                net_ready = 1;
            }
        }
        else
        {
            std::cout << "Error: Net is not set properly in advance using construtor." << std::endl;
        }
    };

    /* Load the mean file in binaryproto format. */
    void descriptorExtractor::setMean(const String& mean_file)
    {
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        if (mean_blob.channels() != num_channels)
            std::cout << "Number of channels of mean file doesn't match input layer." << std::endl;
        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels; ++i)
        {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }
        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);
        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        cv::Scalar channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry, mean.type(), channel_mean);
    };

    void descriptorExtractor::extract(InputArrayOfArrays inputimg, OutputArray feature, String feature_blob)
    {
        if (net_ready)
        {
            Blob<float>* input_layer = convnet->input_blobs()[0];
            input_layer->Reshape(1, num_channels,
            input_geometry.height, input_geometry.width);
            /* Forward dimension change to all layers. */
            convnet->Reshape();
            std::vector<cv::Mat> input_channels;
            wrapInput(&input_channels);
            if (inputimg.kind() == 65536)
            {/* this is a Mat */
                Mat img = inputimg.getMat();
                preprocess(img, &input_channels);
                convnet->ForwardPrefilled();
                /* Copy the output layer to a std::vector */
                Blob<float>* output_layer = convnet->blob_by_name(feature_blob).get();
                const float* begin = output_layer->cpu_data();
                const float* end = begin + output_layer->channels();
                std::vector<float> featureVec = std::vector<float>(begin, end);
                cv::Mat feature_mat = cv::Mat(featureVec, true).t();
                feature_mat.copyTo(feature);
            }
            else
            {/* This is a vector<Mat> */
                vector<Mat> img;
                inputimg.getMatVector(img);
                Mat feature_vector;
                for (unsigned int i = 0; i < img.size(); ++i)
                {
                    preprocess(img[i], &input_channels);
                    convnet->ForwardPrefilled();
                    /* Copy the output layer to a std::vector */
                    Blob<float>* output_layer = convnet->blob_by_name(feature_blob).get();
                    const float* begin = output_layer->cpu_data();
                    const float* end = begin + output_layer->channels();
                    std::vector<float> featureVec = std::vector<float>(begin, end);
                    if (i == 0)
                    {
                        feature_vector = cv::Mat(featureVec, true).t();
                        int dim_feature = feature_vector.cols;
                        feature_vector.resize(img.size(), dim_feature);
                    }
                    feature_vector.row(i) = cv::Mat(featureVec, true).t();
                }
                feature_vector.copyTo(feature);
            }
        }
        else
          std::cout << "Device must be set properly using constructor and the net must be set in advance using loadNet.";
    };

    /* Wrap the input layer of the network in separate cv::Mat objects
     * (one per channel). This way we save one memcpy operation and we
     * don't need to rely on cudaMemcpy2D. The last preprocessing
     * operation will write the separate channels directly to the input
     * layer. */
    void descriptorExtractor::wrapInput(std::vector<cv::Mat>* input_channels)
    {
        Blob<float>* input_layer = convnet->input_blobs()[0];
        int width = input_layer->width();
        int height = input_layer->height();
        float* input_data = input_layer->mutable_cpu_data();
        for (int i = 0; i < input_layer->channels(); ++i)
        {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += width * height;
        }
    };

    void descriptorExtractor::preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
    {
        /* Convert the input image to the input image format of the network. */
        cv::Mat sample;
        if (img.channels() == 3 && num_channels == 1)
            cv::cvtColor(img, sample, CV_BGR2GRAY);
        else if (img.channels() == 4 && num_channels == 1)
            cv::cvtColor(img, sample, CV_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels == 3)
            cv::cvtColor(img, sample, CV_BGRA2BGR);
        else if (img.channels() == 1 && num_channels == 3)
            cv::cvtColor(img, sample, CV_GRAY2BGR);
        else
            sample = img;
        cv::Mat sample_resized;
        if (sample.size() != input_geometry)
            cv::resize(sample, sample_resized, input_geometry);
        else
        sample_resized = sample;
        cv::Mat sample_float;
        if (num_channels == 3)
            sample_resized.convertTo(sample_float, CV_32FC3);
        else
            sample_resized.convertTo(sample_float, CV_32FC1);
        cv::Mat sample_normalized;
        if (net_ready == 2)
            cv::subtract(sample_float, mean_, sample_normalized);
        else
            sample_normalized = sample_float;
        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        cv::split(sample_normalized, *input_channels);
        if (reinterpret_cast<float*>(input_channels->at(0).data)
      != convnet->input_blobs()[0]->cpu_data())
            std::cout << "Input channels are not wrapping the input layer of the network." << std::endl;
    };
} /* namespace cnn_3dobj */
} /* namespace cv */
