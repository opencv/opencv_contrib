#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <cstdlib>

int main(int argc, char **argv)
{

    if (argc < 4)
    {
      std::cerr << "Usage " << argv[0] << ": "
                << "<model-definition-file> " << " "
                << "<model-weights-file> " << " "
                << "<test-image>\n";
      return -1;

    }
    cv::String model_prototxt = argv[1];
    cv::String model_binary = argv[2];
    cv::String test_image = argv[3];
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(model_prototxt, model_binary);

    if (net.empty())
    {
        std::cerr << "Couldn't load the model !\n";
        return -2;
    }
    cv::Mat img = cv::imread(test_image);
    if (img.empty())
    {
        std::cerr << "Couldn't load image: " << test_image << "\n";
        return -3;
    }

    cv::Mat input_blob = cv::dnn::blobFromImage(
      img, 1.0, cv::Size(416, 416), cv::Scalar(104, 117, 123), false);

    cv::Mat prob;
    cv::TickMeter t;

    net.setInput(input_blob);
    t.start();
    prob = net.forward("predictions");
    t.stop();

    int prob_size[3] = {1000, 1, 1};
    cv::Mat prob_data(3, prob_size, CV_32F, prob.ptr<float>(0));

    double max_prob = -1.0;
    int class_idx = -1;
    for (int idx = 0; idx < prob.size[1]; ++idx)
    {
        double current_prob = prob_data.at<float>(idx, 0, 0);
        if (current_prob > max_prob)
        {
          max_prob = current_prob;
          class_idx = idx;
        }
    }
    std::cout << "Best class Index: " << class_idx << "\n";
    std::cout << "Time taken: " << t.getTimeSec() << "\n";
    std::cout << "Probability: " << max_prob * 100.0<< "\n";

    return 0;
}
