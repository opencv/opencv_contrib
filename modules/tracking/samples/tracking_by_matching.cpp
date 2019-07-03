
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking/tracking_by_matching.hpp>
#include <iostream>
#include <cstring>
#include "samples_utility.hpp"

//TODO: ifdef HAVE_DNN!!!!!!!!!!!!!!!!!!!!!!!!!

using namespace std;
using namespace cv;
using namespace cv::tbm;

static const char* keys =
{   "{video_name       | | video name                       }"
    "{start_frame      |0| Start frame                      }"
    "{frame_step       |5| Frame step                       }"
    "{detector_model   | | Path to detector's Caffe model   }"
    "{detector_weights | | Path to detector's Caffe weights }"
    "{desired_class_id |-1| The desired class that should be tracked }"
};

static void help()
{
  cout << "\nThis example shows the functionality of \"Tracking-by-Matching\" approach:"
      " detector is used to detect objects on frames, \n"
      "matching is used to find correspondences between new detections and tracked objects.\n"
      "Detection is made by DNN detection network every `--frame_step` frame.\n"
      "Point .prototxt file of the network as the parameter `--detector_model`, and .caffemodel file"
      " as the parameter `--detector_weights`.\n"
      "(As an example of such detection network is a popular MobileNet_SSD network trained on VOC dataset.)\n"
      "If `--desired_class_id` parameter is set, the detection result is filtered by class id,"
      " returned by the detection network.\n"
      "(That is, if a detection net was trained on VOC dataset, then to track pedestrians point --desired_class_id=15)\n"
       "Example of <video_name> is in opencv_extra/testdata/cv/tracking/\n"
       "Call:\n"
       "./example_tracking_tracking_by_matching --video_name=<video_name> --detector_model=<detector_model_path> --detector_weights=<detector_weights_path> \\\n"
       "                                       [--start_frame=<start_frame>] \\\n"
       "                                       [--frame_step=<frame_step>] \\\n"
       "                                       [--desired_class_id=<desired_class_id>]\n"
       << endl;

  cout << "\n\nHot keys: \n"
       "\tq - quit the program\n"
       "\tp - pause/resume video\n";
}

#ifdef HAVE_OPENCV_DNN
#include <opencv2/dnn.hpp>
class DnnObjectDetector
{
public:
    DnnObjectDetector(const String& net_caffe_model_path, const String& net_caffe_weights_path,
                      int desired_class_id=-1,
                      float confidence_threshold = 0.2,
                      //the following parameters are default for popular MobileNet_SSD caffe model
                      const String& net_input_name="data",
                      const String& net_output_name="detection_out",
                      double net_scalefactor=0.007843,
                      const Size& net_size = Size(300,300),
                      const Scalar& net_mean = Scalar(127.5, 127.5, 127.5),
                      bool net_swapRB=false)
        :desired_class_id(desired_class_id),
        confidence_threshold(confidence_threshold),
        net_input_name(net_input_name),
        net_output_name(net_output_name),
        net_scalefactor(net_scalefactor),
        net_size(net_size),
        net_mean(net_mean),
        net_swapRB(net_swapRB)
    {
        net = dnn::readNetFromCaffe(net_caffe_model_path, net_caffe_weights_path);
        if (net.empty())
            CV_Error(Error::StsError, "Cannot read Caffe net");
    }
    TrackedObjects detect(const cv::Mat& frame, int frame_idx)
    {
        Mat resized_frame;
        resize(frame, resized_frame, net_size);
        Mat inputBlob = cv::dnn::blobFromImage(resized_frame, net_scalefactor, net_size, net_mean, net_swapRB);

        net.setInput(inputBlob, net_input_name);
        Mat detection = net.forward(net_output_name);
        Mat detection_as_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        TrackedObjects res;
        for (int i = 0; i < detection_as_mat.rows; i++)
        {
            float cur_confidence = detection_as_mat.at<float>(i, 2);
            int cur_class_id = static_cast<int>(detection_as_mat.at<float>(i, 1));
            int x_left = static_cast<int>(detection_as_mat.at<float>(i, 3) * frame.cols);
            int y_bottom = static_cast<int>(detection_as_mat.at<float>(i, 4) * frame.rows);
            int x_right = static_cast<int>(detection_as_mat.at<float>(i, 5) * frame.cols);
            int y_top = static_cast<int>(detection_as_mat.at<float>(i, 6) * frame.rows);

            Rect cur_rect(x_left, y_bottom, (x_right - x_left), (y_top - y_bottom));

            //TODO cout << "detectObjectsByNet: " << cur_rect << " conf=" << cur_confidence << " class=" << cur_class_id << endl;
            if (cur_confidence < confidence_threshold)
                continue;
            if ((desired_class_id >= 0) && (cur_class_id != desired_class_id))
                continue;

            //clipping by frame size
            cur_rect = cur_rect & Rect(Point(), frame.size());
            if (cur_rect.empty())
                continue;

            //TODO: remove it
            //cout << "detectObjectsByNet:+" << cur_rect << " conf=" << cur_confidence << " class=" << cur_class_id << endl;

            TrackedObject cur_obj(cur_rect, cur_confidence, frame_idx, -1);
            res.push_back(cur_obj);
        }
        return res;
    }
private:
    cv::dnn::Net net;
    int desired_class_id;
    float confidence_threshold;
    String net_input_name;
    String net_output_name;
    double net_scalefactor;
    Size net_size;
    Scalar net_mean;
    bool net_swapRB;
};
#else
/// Stub for the case when opencv_dnn module is not built
class DnnObjectDetector
{
public:
    DnnObjectDetector(const String& , const String& , ...) {}
    TrackedObjects detect(const cv::Mat& frame, int frame_idx)
    {
        return TrackedObjects();
    }
};
#endif

cv::Ptr<ITrackerByMatching>
CreateTrackerByMatchingWithFastDescriptor() {
    cv::tbm::TrackerParams params;

    cv::Ptr<ITrackerByMatching> tracker = CreateTrackerByMatching(params);

    std::shared_ptr<IImageDescriptor> descriptor_fast =
        std::make_shared<ResizedImageDescriptor>(
            cv::Size(16, 32), cv::InterpolationFlags::INTER_LINEAR);
    std::shared_ptr<IDescriptorDistance> distance_fast =
        std::make_shared<MatchTemplateDistance>();

    tracker->set_descriptor_fast(descriptor_fast);
    tracker->set_distance_fast(distance_fast);

    return tracker;
}
int main( int argc, char** argv ){
#ifndef HAVE_OPENCV_DNN
    CV_Error(Error::StsNotImplemented, "At the moment the sample 'tracking_by_matching' can work only when opencv_dnn module is built.");
#endif

    CommandLineParser parser( argc, argv, keys );
    cv::Ptr<ITrackerByMatching> tracker = CreateTrackerByMatchingWithFastDescriptor();

    String video_name = parser.get<String>("video_name");
    int start_frame = parser.get<int>("start_frame");
    int frame_step = parser.get<int>("frame_step");
    String detector_model = parser.get<String>("detector_model");
    String detector_weights = parser.get<String>("detector_weights");
    int desired_class_id = parser.get<int>("desired_class_id");

    if( video_name.empty() || detector_model.empty() || detector_weights.empty() )
    {
        help();
        return -1;
    }


    //open the capture
    VideoCapture cap;
    cap.open( video_name );
    cap.set( CAP_PROP_POS_FRAMES, start_frame );

    if( !cap.isOpened() )
    {
        help();
        cout << "***Could not initialize capturing...***\n";
        cout << "Current parameter's value: \n";
        parser.printMessage();
        return -1;
    }

    // If you use the popular MobileNet_SSD detector, the default parameters may be used.
    // Otherwise, set your own parameters (net_mean, net_scalefactor, etc).
    DnnObjectDetector detector(detector_model, detector_weights, desired_class_id);

    Mat frame;
    namedWindow( "Tracking by Matching", 1 );

    int frame_counter = 0;
    int64 time_total = 0;
    bool paused = false;
    for ( ;; )
    {
        if( paused )
        {
            char c = (char) waitKey(30);
            if (c == 'p')
                paused = !paused;
            if (c == 'q')
                break;
            continue;
        }

        cap >> frame;
        if(frame.empty()){
            break;
        }
        if (frame_counter < start_frame) {
            frame_counter++;
            continue;
        }


        int64 frame_time = getTickCount();

        TrackedObjects detections = detector.detect(frame, frame_counter);

        // timestamp in milliseconds
        uint64_t cur_timestamp = 1000.0 / 30 * frame_counter;
        tracker->Process(frame, detections, cur_timestamp);

        frame_time = getTickCount() - frame_time;
        time_total += frame_time;

        // Drawing colored "worms" (tracks).
        frame = tracker->DrawActiveTracks(frame);


        // Drawing all detected objects on a frame by BLUE COLOR
        for (const auto &detection : detections) {
            cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 3);
        }

        // Drawing tracked detections only by RED color and print ID and detection
        // confidence level.
        for (const auto &detection : tracker->TrackedDetections()) {
            cv::rectangle(frame, detection.rect, cv::Scalar(0, 0, 255), 3);
            std::string text = std::to_string(detection.object_id) +
                " conf: " + std::to_string(detection.confidence);
            cv::putText(frame, text, detection.rect.tl(), cv::FONT_HERSHEY_COMPLEX,
                        1.0, cv::Scalar(0, 0, 255), 3);
        }

        imshow( "Tracking by Matching", frame );
        frame_counter++;

        char c = (char) waitKey( 2 );
        if (c == 'q')
            break;
        if (c == 'p')
            paused = !paused;
    }

    double s = frame_counter / (time_total / getTickFrequency());
    printf("FPS: %f\n", s);

    return 0;
}
