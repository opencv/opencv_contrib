#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <iostream>
#include <cstdlib>

#include <opencv2/core_detect.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::dnn_objdetect;

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage " << argv[0] << ": "
                  << "<model-definition-file> "
                  << "<model-weights-file> "
                  << "<test-image> "
                  << "<threshold>(optional)\n";
        return -1;
    }

    std::string model_prototxt = argv[1];
    std::string model_binary = argv[2];
    std::string test_input_image = argv[3];
    double threshold = 0.7;

    if (argc == 5)
    {
      threshold = atof(argv[4]);
      if (threshold > 1.0 || threshold < 0.0)
      {
        std::cerr << "Threshold should belong to [0, 1]\n";
        return -1;
      }
    }

    // Load the network
    std::cout << "Loading the network...\n";
    Net net = dnn::readNetFromCaffe(model_prototxt, model_binary);
    if (net.empty())
    {
       std::cerr << "Couldn't load the model !\n";
       return -2;
    }
    else
    {
      std::cout << "Done loading the network !\n\n";
    }

    // Load the test image
    Mat img = cv::imread(test_input_image);
    Mat original_img(img);
    if (img.empty())
    {
        std::cerr << "Couldn't load image: " << test_input_image << "\n";
        return -3;
    }

    cv::namedWindow("Initial Image", WINDOW_AUTOSIZE);
    cv::imshow("Initial Image", img);

    cv::resize(img, img, cv::Size(416, 416));
    Mat img_copy(img);
    img.convertTo(img, CV_32FC3);
    Mat input_blob = blobFromImage(img, 1.0, Size(), cv::Scalar(104, 117, 123), false);

    // Set the input blob

    // Set the output layers
    std::cout << "Getting the output of all the three blobs...\n";
    std::vector<Mat> outblobs(3);
    std::vector<cv::String> out_layers;
    out_layers.push_back("slice");
    out_layers.push_back("softmax");
    out_layers.push_back("sigmoid");

    // Bbox delta blob
    std::vector<Mat> temp_blob;
    net.setInput(input_blob);
    cv::TickMeter t;

    t.start();
    net.forward(temp_blob, out_layers[0]);
    t.stop();
    outblobs[0] = temp_blob[2];

    // class_scores blob
    net.setInput(input_blob);
    t.start();
    outblobs[1] = net.forward(out_layers[1]);
    t.stop();

    // conf_scores blob
    net.setInput(input_blob);
    t.start();
    outblobs[2] = net.forward(out_layers[2]);
    t.stop();

    // Check that the blobs are valid
    for (size_t i = 0; i < outblobs.size(); ++i)
    {
        if (outblobs[i].empty())
        {
          std::cerr << "Blob: " << i << " is empty !\n";
        }
    }

    int delta_bbox_size[3] = {23, 23, 36};
    Mat delta_bbox(3, delta_bbox_size, CV_32F, outblobs[0].ptr<float>());

    int class_scores_size[2] = {4761, 20};
    Mat class_scores(2, class_scores_size, CV_32F, outblobs[1].ptr<float>());

    int conf_scores_size[3] = {23, 23, 9};
    Mat conf_scores(3, conf_scores_size, CV_32F, outblobs[2].ptr<float>());

    InferBbox inf(delta_bbox, class_scores, conf_scores);
    inf.filter(threshold);


    double average_time = t.getTimeSec() / t.getCounter();
    std::cout << "\nTotal objects detected: " << inf.detections.size()
              << " in " << average_time << " seconds\n";
    std::cout << "------\n";
    float x_ratio = (float)original_img.cols / img_copy.cols;
    float y_ratio = (float)original_img.rows / img_copy.rows;
    for (size_t i = 0; i < inf.detections.size(); ++i)
    {

      int xmin = inf.detections[i].xmin;
      int ymin = inf.detections[i].ymin;
      int xmax = inf.detections[i].xmax;
      int ymax = inf.detections[i].ymax;
      cv::String class_name = inf.detections[i].label_name;
      std::cout << "Class: " << class_name << "\n"
                << "Probability: " << inf.detections[i].class_prob << "\n"
                << "Co-ordinates: " << inf.detections[i].xmin << " "
                << inf.detections[i].ymin << " "
                << inf.detections[i].xmax << " "
                << inf.detections[i].ymax << "\n";
      std::cout << "------\n";
      // Draw the corresponding bounding box(s)
      cv::rectangle(original_img, cv::Point((int)(xmin * x_ratio), (int)(ymin * y_ratio)),
          cv::Point((int)(xmax * x_ratio), (int)(ymax * y_ratio)), cv::Scalar(255, 0, 0), 2);
      cv::putText(original_img, class_name, cv::Point((int)(xmin * x_ratio), (int)(ymin * y_ratio)),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 1);
    }

    try
    {
      cv::namedWindow("Final Detections", WINDOW_AUTOSIZE);
      cv::imshow("Final Detections", original_img);
      cv::imwrite("image.png", original_img);
      cv::waitKey(0);
    }
    catch (const char* msg)
    {
      std::cerr << msg << "\n";
      return -4;
    }

    return 0;
}
