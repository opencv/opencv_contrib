#include "precomp.hpp"
using namespace caffe;
using std::string;

namespace cv
{
namespace cnn_3dobj
{
  Classification::Classification(){};
  void Classification::list_dir(const char *path,vector<string>& files,bool r)
  {
    DIR *pDir;
    struct dirent *ent;
    char childpath[512];
    pDir = opendir(path);
    memset(childpath, 0, sizeof(childpath));
    while ((ent = readdir(pDir)) != NULL)
    {
      if (ent->d_type & DT_DIR)
      {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
        {
          continue;
        }
        if(r)
        {
          sprintf(childpath, "%s/%s", path, ent->d_name);
          Classification::list_dir(childpath,files,false);
        }
      }
      else
      {
        files.push_back(ent->d_name);
      }
    }
    sort(files.begin(),files.end());
  };

  void Classification::NetSetter(const string& model_file, const string& trained_file, const string& mean_file, const string& cpu_only, int device_id)
  {
    if (strcmp(cpu_only.c_str(), "CPU") == 0)
    {
      caffe::Caffe::set_mode(caffe::Caffe::CPU);
    }
    else
    {
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
      caffe::Caffe::SetDevice(device_id);
    }
    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    /* Load the binaryproto mean file. */
    SetMean(mean_file);
  };

  void Classification::GetLabellist(const std::vector<string>& name_gallery)
  {
    for (unsigned int i = 0; i < name_gallery.size(); ++i)
    labels_.push_back(name_gallery[i]);
  };

  /* Return the indices of the top N values of vector v. */
  std::vector<int> Classification::Argmax(const std::vector<float>& v, int N)
  {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
      pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end());
    std::vector<int> result;
    for (int i = 0; i < N; ++i)
      result.push_back(pairs[i].second);
    return result;
  };

  //Return the top N predictions.
  std::vector<std::pair<string, float> > Classification::Classify(const cv::Mat& reference, const cv::Mat& img, int N, bool mean_substract)
  {
    cv::Mat feature;
    Classification::FeatureExtract(img, feature, mean_substract);
    std::vector<float> output;
    for (int i = 0; i < reference.rows; i++)
    {
      cv::Mat f1 = reference.row(i);
      cv::Mat f2 = feature;
      cv::Mat output_temp = f1-f2;
      output.push_back(cv::norm(output_temp));
    }
    std::vector<int> maxN = Argmax(output, N);
    std::vector<std::pair<string, float> > predictions;
    for (int i = 0; i < N; ++i)
    {
      int idx = maxN[i];
      predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }
    return predictions;
  };

  /* Load the mean file in binaryproto format. */
  void Classification::SetMean(const string& mean_file)
  {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";
    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i)
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
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  };

  void Classification::FeatureExtract(InputArray inputimg, OutputArray feature, bool mean_subtract)
  {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
    input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    if (inputimg.kind() == 65536)
    {/* this is a Mat */
      Mat img = inputimg.getMat();
      Preprocess(img, &input_channels, mean_subtract);
      net_->ForwardPrefilled();
      /* Copy the output layer to a std::vector */
      Blob<float>* output_layer = net_->output_blobs()[0];
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
        Preprocess(img[i], &input_channels, mean_subtract);
        net_->ForwardPrefilled();
        /* Copy the output layer to a std::vector */
        Blob<float>* output_layer = net_->output_blobs()[0];
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
  };

  /* Wrap the input layer of the network in separate cv::Mat objects
  * (one per channel). This way we save one memcpy operation and we
  * don't need to rely on cudaMemcpy2D. The last preprocessing
  * operation will write the separate channels directly to the input
  * layer. */
  void Classification::WrapInputLayer(std::vector<cv::Mat>* input_channels)
  {
    Blob<float>* input_layer = net_->input_blobs()[0];
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

  void Classification::Preprocess(const cv::Mat& img,
std::vector<cv::Mat>* input_channels, bool mean_subtract)
  {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
      sample = img;
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
      cv::resize(sample, sample_resized, input_geometry_);
    else
      sample_resized = sample;
    cv::Mat sample_float;
    if (num_channels_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);
    cv::Mat sample_normalized;
    if (mean_subtract)
      cv::subtract(sample_float, mean_, sample_normalized);
    else
      sample_normalized = sample_float;
    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
      == net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
  };
}
}
